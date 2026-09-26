//! Node CRUD operations on the knowledge graph.
//!
//! All methods use parameterized Cypher to prevent injection.  Labels and
//! property names are validated before interpolation; values are always
//! bound as `$pN` parameters.

use std::collections::HashMap;

use uni_db::{Transaction, Value};

use super::{build_inline_props, build_set_clause, validate_label, validate_property_name};
use crate::error::{Result, UnikoError};
use crate::storage::KnowledgeBase;
use crate::storage::filter::Filter;
use crate::types::NodeId;

impl KnowledgeBase {
    /// Create a node with the given label and properties.
    ///
    /// Returns the internal [`NodeId`] of the new node.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Schema`] if `label` is unknown or a property
    /// name is invalid, or [`UnikoError::Storage`] on database failure.
    pub async fn create_node(
        &self,
        label: &str,
        properties: &HashMap<String, Value>,
    ) -> Result<NodeId> {
        self.transact_with_retry(uni_db::RetryOptions::default(), move |tx| async move {
            let r = self.create_node_in_tx(&tx, label, properties).await;
            (tx, r)
        })
        .await
    }

    /// Create a node within an existing transaction without committing.
    ///
    /// Use to group several writes (Message + edges + chunks, entity
    /// upsert sub-steps, etc.) under one commit. Mirrors
    /// [`KnowledgeBase::create_node`]'s shape but defers the
    /// transaction lifecycle to the caller.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Schema`] if `label` is unknown or a
    /// property name is invalid, or [`UnikoError::Storage`] on
    /// transaction failure.
    pub async fn create_node_in_tx(
        &self,
        tx: &Transaction,
        label: &str,
        properties: &HashMap<String, Value>,
    ) -> Result<NodeId> {
        let t_total = std::time::Instant::now();
        validate_label(label)?;

        // Properties must be inline in CREATE to satisfy NOT NULL constraints.
        let (inline_props, params) = build_inline_props(properties, 0)?;
        let cypher = format!("CREATE (n:{label} {{{inline_props}}}) RETURN id(n) AS vid");

        let mut qb = tx.query_with(&cypher);
        for (k, v) in &params {
            qb = qb.param(k.as_str(), v.clone());
        }
        let result = qb.fetch_all().await?;
        let row = result
            .rows()
            .first()
            .ok_or_else(|| UnikoError::Storage("CREATE returned no rows".into()))?;
        let vid: i64 = row.get("vid")?;

        // uni-db's own `QueryMetrics` already splits parse/plan/exec —
        // re-emitting per-Rust-stage timers (validate / build / bind /
        // fetch / extract) duplicates info and inflates per-write
        // overhead. Keep only the wall-clock total + label.
        tracing::info!(
            target: "query_metrics",
            site = "create_node_in_tx",
            label,
            total_us = t_total.elapsed().as_micros() as u64,
            cache_hit = result.metrics().plan_cache_hit,
            "query metrics",
        );
        Ok(vid)
    }

    /// Retrieve a node by its internal [`NodeId`].
    ///
    /// Returns `None` if the node does not exist.  The tuple contains the
    /// primary label and all stored properties.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn get_node(
        &self,
        node_id: NodeId,
    ) -> Result<Option<(String, HashMap<String, Value>)>> {
        let session = self.db.session();
        let result = session
            .query_with("MATCH (n) WHERE id(n) = $vid RETURN n")
            .param("vid", node_id)
            .fetch_all()
            .await?;

        match result.rows().first() {
            None => Ok(None),
            Some(row) => {
                let node: uni_db::Node = row.get("n")?;
                let label = node.labels.into_iter().next().unwrap_or_default();
                Ok(Some((label, node.properties)))
            }
        }
    }

    /// Cypher for an external-id lookup, shared by the committing and
    /// in-transaction variants so the two cannot drift apart.
    fn ext_id_cypher(label: &str, id_field: &str) -> Result<String> {
        validate_label(label)?;
        validate_property_name(id_field)?;
        Ok(format!(
            "MATCH (n:{label} {{{id_field}: $eid}}) RETURN n, id(n) AS vid"
        ))
    }

    /// Look up a node by its external ID field (e.g. `message_id`).
    ///
    /// Uses the Hash index on the ID field for O(1) lookup.
    ///
    /// Runs on a fresh session, so it sees only *committed* state. Inside a
    /// transaction you own, use
    /// [`get_node_by_ext_id_in_tx`](Self::get_node_by_ext_id_in_tx).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Schema`] if `label` or `id_field` is invalid,
    /// or [`UnikoError::Storage`] on database failure.
    pub async fn get_node_by_ext_id(
        &self,
        label: &str,
        id_field: &str,
        ext_id: &str,
    ) -> Result<Option<(NodeId, HashMap<String, Value>)>> {
        let cypher = Self::ext_id_cypher(label, id_field)?;
        let session = self.db.session();
        let result = session
            .query_with(&cypher)
            .param("eid", ext_id)
            .fetch_all()
            .await?;
        Self::first_ext_id_row(&result)
    }

    /// Look up a node by its external ID field **inside** `tx`, so the read
    /// joins that transaction's snapshot and its SSI read set.
    ///
    /// [`get_node_by_ext_id`](Self::get_node_by_ext_id) runs on a fresh
    /// session and therefore *cannot* observe writes the caller has made
    /// but not yet committed. Any check-then-create that lives inside a
    /// caller-owned transaction must use this variant — otherwise a second
    /// write of the same logical row in the same transaction reads "absent"
    /// and creates a duplicate.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Schema`] if `label` or `id_field` is invalid,
    /// or [`UnikoError::Storage`] on database failure.
    pub async fn get_node_by_ext_id_in_tx(
        &self,
        tx: &uni_db::Transaction,
        label: &str,
        id_field: &str,
        ext_id: &str,
    ) -> Result<Option<(NodeId, HashMap<String, Value>)>> {
        let cypher = Self::ext_id_cypher(label, id_field)?;
        let result = tx
            .query_with(&cypher)
            .param("eid", ext_id)
            .fetch_all()
            .await?;
        Self::first_ext_id_row(&result)
    }

    /// Decode the first row of an [`ext_id_cypher`](Self::ext_id_cypher)
    /// result, shared by both lookup variants.
    fn first_ext_id_row(
        result: &uni_db::QueryResult,
    ) -> Result<Option<(NodeId, HashMap<String, Value>)>> {
        match result.rows().first() {
            None => Ok(None),
            Some(row) => {
                let vid: i64 = row.get("vid")?;
                let node: uni_db::Node = row.get("n")?;
                Ok(Some((vid, node.properties)))
            }
        }
    }

    /// Update specific properties on an existing node.
    ///
    /// Only the supplied properties are changed; others are preserved.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure or if the node
    /// does not exist.
    pub async fn update_node(
        &self,
        node_id: NodeId,
        properties: &HashMap<String, Value>,
    ) -> Result<()> {
        if properties.is_empty() {
            return Ok(());
        }
        let (set_clause, params) = build_set_clause("n", properties, 0)?;
        let cypher = format!("MATCH (n) WHERE id(n) = $vid {set_clause}");

        self.transact_with_retry(uni_db::RetryOptions::default(), move |tx| {
            // Re-clone per attempt: the closure is re-run on a retriable conflict.
            let cypher = cypher.clone();
            let params = params.clone();
            async move {
                let mut qb = tx.execute_with(&cypher);
                qb = qb.param("vid", node_id);
                for (k, v) in &params {
                    qb = qb.param(k.as_str(), v.clone());
                }
                let r = qb.run().await.map(|_| ()).map_err(UnikoError::from);
                (tx, r)
            }
        })
        .await
    }

    /// Delete a node and all its edges.
    ///
    /// Returns `true` if a node was deleted, `false` if it did not exist.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn delete_node(&self, node_id: NodeId) -> Result<bool> {
        self.transact_with_retry(uni_db::RetryOptions::default(), move |tx| async move {
            let r = async {
                let result = tx
                    .execute_with("MATCH (n) WHERE id(n) = $vid DETACH DELETE n")
                    .param("vid", node_id)
                    .run()
                    .await?;
                Ok(result.nodes_deleted() > 0)
            }
            .await;
            (tx, r)
        })
        .await
    }

    /// Upsert a node using its external ID field.
    ///
    /// If no node with the given `ext_id` exists, one is created with
    /// `properties`.  If it already exists, the existing node is updated.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Schema`] if `label` or `ext_id_field` is
    /// invalid, or [`UnikoError::Storage`] on database failure.
    pub async fn merge_node(
        &self,
        label: &str,
        ext_id_field: &str,
        ext_id: &str,
        properties: &HashMap<String, Value>,
    ) -> Result<NodeId> {
        validate_label(label)?;
        validate_property_name(ext_id_field)?;

        // Serialize the check-then-create window for this `ext_id`.
        // Without the lock two callers can both observe "not present"
        // and each create their own node, producing duplicate rows.
        // The lock also covers the subsequent update path so a later
        // RMW on the same logical row is consistent.
        let mut lock_key =
            Vec::with_capacity(5 + label.len() + 1 + ext_id_field.len() + 1 + ext_id.len());
        lock_key.extend_from_slice(b"node:");
        lock_key.extend_from_slice(label.as_bytes());
        lock_key.push(0);
        lock_key.extend_from_slice(ext_id_field.as_bytes());
        lock_key.push(0);
        lock_key.extend_from_slice(ext_id.as_bytes());
        let _rmw_guard = self.rmw_locks.lock(&lock_key).await;

        // Explicit get-then-(update|create) rather than a single MERGE.
        // uni-db has no atomic CAS and SSI does not catch insert-phantoms
        // (an empty match registers no read-set conflict), so a bare
        // MERGE on a non-UNIQUE key would still let two concurrent first
        // upserts both create — the `rmw_locks` guard above serializes the
        // window. (The older reason — MERGE's CREATE checking NOT NULL
        // before ON CREATE SET — is fixed in uni 2.1.0; the RMW lock is
        // what keeps this split, not that.)
        if let Some((existing_id, _)) = self.get_node_by_ext_id(label, ext_id_field, ext_id).await?
        {
            if !properties.is_empty() {
                self.update_node(existing_id, properties).await?;
            }
            return Ok(existing_id);
        }

        // Node doesn't exist — create with all properties plus the ext_id.
        let mut all_props = properties.clone();
        all_props
            .entry(ext_id_field.to_string())
            .or_insert_with(|| Value::String(ext_id.to_string()));
        self.create_node(label, &all_props).await
    }

    /// Query nodes of a given label, optionally filtered and limited.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Schema`] if `label` is unknown, or
    /// [`UnikoError::Storage`] on database failure.
    pub async fn query_nodes(
        &self,
        label: &str,
        filter: Option<&Filter>,
        limit: Option<usize>,
    ) -> Result<Vec<(NodeId, HashMap<String, Value>)>> {
        validate_label(label)?;

        let mut cypher = format!("MATCH (n:{label})");
        let mut all_params: Vec<(String, Value)> = Vec::new();

        if let Some(f) = filter {
            let cf = f.to_cypher("n", 0)?;
            cypher.push_str(&format!(" WHERE {}", cf.fragment));
            all_params.extend(cf.params);
        }

        cypher.push_str(" RETURN n, id(n) AS vid");

        if let Some(lim) = limit {
            cypher.push_str(&format!(" LIMIT {lim}"));
        }

        let session = self.db.session();
        let mut qb = session.query_with(&cypher);
        for (k, v) in &all_params {
            qb = qb.param(k.as_str(), v.clone());
        }
        let result = qb.fetch_all().await?;

        let mut nodes = Vec::with_capacity(result.len());
        for row in result.rows() {
            let vid: i64 = row.get("vid")?;
            let node: uni_db::Node = row.get("n")?;
            nodes.push((vid, node.properties));
        }
        Ok(nodes)
    }
}
