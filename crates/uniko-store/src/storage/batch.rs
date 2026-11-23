//! Batch creation via UNWIND for bulk ingestion.
//!
//! Uses single `UNWIND`-based Cypher statements instead of per-row
//! loops.  This leverages the HashJoin rewrite (uni-db #53/#54) that
//! converts `UNWIND … MATCH WHERE id() = e.col` into an efficient
//! hash join rather than a cross-join + filter.

use std::collections::{BTreeSet, HashMap};

use uni_db::{Transaction, Value, Vid};

use super::{KnowledgeBase, validate_edge_type, validate_label, validate_property_name};
use crate::error::Result;
use crate::types::{EdgeId, NodeId};

impl KnowledgeBase {
    /// Create multiple nodes of the same label in a single UNWIND query.
    ///
    /// Significantly faster than individual [`KnowledgeBase::create_node`]
    /// calls for bulk ingestion (e.g. chunking an artifact into many
    /// chunks).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Schema`] if `label` is unknown or a
    /// property name is invalid, or [`UnikoError::Storage`] on database
    /// failure.  The transaction is rolled back on any error.
    pub async fn batch_create_nodes(
        &self,
        label: &str,
        items: &[HashMap<String, Value>],
    ) -> Result<Vec<NodeId>> {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        let session = self.db.session();
        let tx = session.tx().await?;
        let node_ids = self.batch_create_nodes_in_tx(&tx, label, items).await?;
        tx.commit().await?;
        Ok(node_ids)
    }

    /// Like [`KnowledgeBase::batch_create_nodes`] but executed against
    /// an existing transaction without committing.
    ///
    /// Use when grouping multiple batched writes for a single message
    /// under one commit (e.g. the per-message ingest path which folds
    /// Message + edges + chunks into one tx).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Schema`] if `label` is unknown or a
    /// property name is invalid, or [`UnikoError::Storage`] on
    /// transaction failure. The caller's transaction is left in its
    /// current state on error — invoke `tx.commit()` or drop to
    /// rollback per the standard uni-db tx semantics.
    pub async fn batch_create_nodes_in_tx(
        &self,
        tx: &Transaction,
        label: &str,
        items: &[HashMap<String, Value>],
    ) -> Result<Vec<NodeId>> {
        validate_label(label)?;
        if items.is_empty() {
            return Ok(Vec::new());
        }
        // Validate property names upfront — uni-db's bulk API does
        // not re-check them.
        for props in items {
            for k in props.keys() {
                validate_property_name(k)?;
            }
        }

        // Capture the batch for the bulk-vs-UNWIND benchmark (no-op
        // unless the `batch-record` feature is on and recording is
        // enabled). The thunk clones the rows only when recording.
        super::batch_record::record_node_batch(label, || items.to_vec());

        // Route through `tx.bulk_insert_vertices` (uni-db public API).
        // Same rationale as `batch_create_edges_inner_in_tx`'s bulk
        // arm: skip Cypher parse + plan + per-row executor wrapping,
        // hand the properties straight to `writer.insert_vertices_batch`
        // which still runs `process_embeddings_for_batch` so auto-embed
        // semantics are preserved.
        let t_start = std::time::Instant::now();
        let properties_list: Vec<HashMap<String, Value>> = items.to_vec();
        let vids = tx.bulk_insert_vertices(label, properties_list).await?;
        let t_query = t_start.elapsed().as_micros() as u64;
        let node_ids: Vec<NodeId> = vids.iter().map(|v| v.as_u64() as i64).collect();

        tracing::debug!(
            label,
            count = items.len(),
            query_us = t_query,
            "batch_create_nodes_in_tx micro (bulk)"
        );
        tracing::info!(
            target: "query_metrics",
            site = "batch_create_nodes_in_tx",
            label,
            node_count = items.len() as u64,
            parse_us = 0u64,
            plan_us = 0u64,
            exec_us = t_query,
            total_us = t_query,
            cache_hit = false,
            bulk = true,
            "query metrics",
        );
        Ok(node_ids)
    }

    /// Create multiple edges of the same type in a single UNWIND query.
    ///
    /// Each tuple is `(from_node_id, to_node_id, properties)`.  Returns
    /// the created edge IDs.  When the IDs are not needed, prefer
    /// [`KnowledgeBase::batch_create_edges_fast`] which skips the
    /// `RETURN` clause and is materially faster on auto-embed graphs.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Schema`] if `edge_type` is unknown or a
    /// property name is invalid, or [`UnikoError::Storage`] on database
    /// failure.
    pub async fn batch_create_edges(
        &self,
        edge_type: &str,
        edges: &[(NodeId, NodeId, HashMap<String, Value>)],
    ) -> Result<Vec<EdgeId>> {
        self.batch_create_edges_inner(edge_type, None, None, edges, true)
            .await
    }

    /// Same as [`KnowledgeBase::batch_create_edges`] without returning
    /// edge IDs.  Optional `src_label` and `dst_label` hints constrain
    /// the `MATCH` to a labelled node set, letting the planner narrow
    /// the scan.
    ///
    /// Pass `Some("Message|Chunk")` for label disjunction.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Schema`] if `edge_type` is unknown or a
    /// property name is invalid, or [`UnikoError::Storage`] on database
    /// failure.
    pub async fn batch_create_edges_fast(
        &self,
        edge_type: &str,
        src_label: Option<&str>,
        dst_label: Option<&str>,
        edges: &[(NodeId, NodeId, HashMap<String, Value>)],
    ) -> Result<()> {
        self.batch_create_edges_inner(edge_type, src_label, dst_label, edges, false)
            .await?;
        Ok(())
    }

    /// Like [`KnowledgeBase::batch_create_edges_fast`] but executed
    /// against an existing transaction without committing.
    ///
    /// Use when grouping multiple batched writes under one commit
    /// (per-message ingest, observation step folding three edge
    /// batches into one tx, etc.).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Schema`] if `edge_type` is unknown or a
    /// property name is invalid, or [`UnikoError::Storage`] on
    /// transaction failure.
    pub async fn batch_create_edges_fast_in_tx(
        &self,
        tx: &Transaction,
        edge_type: &str,
        src_label: Option<&str>,
        dst_label: Option<&str>,
        edges: &[(NodeId, NodeId, HashMap<String, Value>)],
    ) -> Result<()> {
        self.batch_create_edges_inner_in_tx(tx, edge_type, src_label, dst_label, edges, false)
            .await?;
        Ok(())
    }

    async fn batch_create_edges_inner(
        &self,
        edge_type: &str,
        src_label: Option<&str>,
        dst_label: Option<&str>,
        edges: &[(NodeId, NodeId, HashMap<String, Value>)],
        return_ids: bool,
    ) -> Result<Vec<EdgeId>> {
        if edges.is_empty() {
            return Ok(Vec::new());
        }
        let session = self.db.session();
        let tx = session.tx().await?;
        let edge_ids = self
            .batch_create_edges_inner_in_tx(&tx, edge_type, src_label, dst_label, edges, return_ids)
            .await?;
        tx.commit().await?;
        Ok(edge_ids)
    }

    async fn batch_create_edges_inner_in_tx(
        &self,
        tx: &Transaction,
        edge_type: &str,
        src_label: Option<&str>,
        dst_label: Option<&str>,
        edges: &[(NodeId, NodeId, HashMap<String, Value>)],
        return_ids: bool,
    ) -> Result<Vec<EdgeId>> {
        validate_edge_type(edge_type)?;
        if edges.is_empty() {
            return Ok(Vec::new());
        }

        // Capture the batch for the bulk-vs-UNWIND benchmark (no-op
        // unless the `batch-record` feature is on and recording is
        // enabled). The thunk clones per-edge props only when recording.
        super::batch_record::record_edge_batch(edge_type, || {
            edges.iter().map(|(_, _, p)| p.clone()).collect()
        });

        // ── Fast path: `tx.bulk_insert_edges` (uni-db public API).
        //
        // When the caller doesn't need EIDs back, skip the entire
        // Cypher pipeline (parse + plan + per-row MATCH id-lookup +
        // executor wrapping) and write straight to the tx's private
        // L0 via the bulk API. uni-db's `edge_commit_contention.rs`
        // microbenchmark shows this path holds at ~150 µs/edge at
        // sess=24 (10 edges/tx, COLD); the Cypher path through
        // `tx.execute_with` measures ~147 ms/edge in this workload
        // — a ~980× gap that's purely Cypher executor overhead per
        // row, since the VIDs are already known at the call site.
        if !return_ids {
            let t_start = std::time::Instant::now();
            // uni-db's `bulk_insert_edges` takes property names as-is,
            // so validate every key up front and propagate failures
            // before the bulk write commits.
            for (_, _, props) in edges {
                for k in props.keys() {
                    validate_property_name(k)?;
                }
            }
            let bulk_edges: Vec<(Vid, Vid, HashMap<String, Value>)> = edges
                .iter()
                .map(|(s, d, props)| (Vid::new(*s as u64), Vid::new(*d as u64), props.clone()))
                .collect();
            tx.bulk_insert_edges(edge_type, bulk_edges).await?;
            let t_query = t_start.elapsed().as_micros();
            tracing::debug!(
                edge_type,
                count = edges.len(),
                return_ids,
                query_us = t_query as u64,
                "batch_create_edges_in_tx micro (bulk)"
            );
            // Emit synthetic query_metrics so post-hoc aggregation
            // keeps working. The bulk path has no Cypher parse/plan;
            // we report 0 for those and the elapsed wall as exec_us.
            tracing::info!(
                target: "query_metrics",
                site = "batch_create_edges_fast_in_tx",
                edge_type,
                edge_count = edges.len() as u64,
                parse_us = 0u64,
                plan_us = 0u64,
                exec_us = t_query as u64,
                total_us = t_query as u64,
                cache_hit = false,
                bulk = true,
                "query metrics",
            );
            // Edge IDs are intentionally not returned on this path —
            // callers that need them should use the `return_ids` arm
            // below (Cypher with RETURN id(r) AS eid).
            return Ok(Vec::new());
        }

        // ── Cypher path: still needed when caller wants EIDs back
        // (uni-db's `bulk_insert_edges` doesn't surface allocated
        // EIDs at the public API today).

        // Collect the union of all property keys (sorted for determinism).
        let prop_keys = collect_property_keys_from_edges(edges)?;

        // Build UNWIND list: one Map per edge with src, dst, and properties.
        let list: Vec<Value> = edges
            .iter()
            .map(|(from, to, props)| {
                let mut m = HashMap::with_capacity(2 + prop_keys.len());
                m.insert("src".to_string(), Value::Int(*from));
                m.insert("dst".to_string(), Value::Int(*to));
                for k in &prop_keys {
                    m.insert(k.clone(), props.get(k).cloned().unwrap_or(Value::Null));
                }
                Value::Map(m)
            })
            .collect();

        // Build SET clause for properties: `SET r.p1 = e.p1, r.p2 = e.p2`.
        let set_clause = if prop_keys.is_empty() {
            String::new()
        } else {
            let assigns: Vec<String> = prop_keys.iter().map(|k| format!("r.{k} = e.{k}")).collect();
            format!(" SET {}", assigns.join(", "))
        };

        // Multi-MATCH (avoids implicit cartesian product of MATCH (a),(b)).
        // Optional label hints narrow the scan to known label sets.
        let src_lbl = src_label.map(|l| format!(":{l}")).unwrap_or_default();
        let dst_lbl = dst_label.map(|l| format!(":{l}")).unwrap_or_default();

        let cypher = format!(
            "UNWIND $edges AS e \
             MATCH (a{src_lbl}) WHERE id(a) = e.src \
             MATCH (b{dst_lbl}) WHERE id(b) = e.dst \
             CREATE (a)-[r:{edge_type}]->(b){set_clause} \
             RETURN id(r) AS eid"
        );

        let t_start = std::time::Instant::now();
        let result = tx
            .query_with(&cypher)
            .param("edges", Value::List(list))
            .fetch_all()
            .await?;
        let mut edge_ids = Vec::with_capacity(edges.len());
        for row in result.rows() {
            let eid: i64 = row.get("eid")?;
            edge_ids.push(eid);
        }
        let metrics = result.metrics().clone();
        let t_query = t_start.elapsed().as_micros();

        tracing::debug!(
            edge_type,
            count = edges.len(),
            return_ids,
            query_us = t_query as u64,
            "batch_create_edges_in_tx micro"
        );
        tracing::info!(
            target: "query_metrics",
            site = "batch_create_edges_fast_in_tx",
            edge_type,
            edge_count = edges.len() as u64,
            parse_us = metrics.parse_time.as_micros() as u64,
            plan_us = metrics.plan_time.as_micros() as u64,
            exec_us = metrics.exec_time.as_micros() as u64,
            total_us = metrics.total_time.as_micros() as u64,
            cache_hit = metrics.plan_cache_hit,
            bulk = false,
            "query metrics",
        );
        Ok(edge_ids)
    }
}

// ── Helpers ────────────────────────────────────────────────────────

/// Collect and validate the union of property keys from edge tuples.
fn collect_property_keys_from_edges(
    edges: &[(NodeId, NodeId, HashMap<String, Value>)],
) -> Result<Vec<String>> {
    let mut keys = BTreeSet::new();
    for (_, _, props) in edges {
        for k in props.keys() {
            validate_property_name(k)?;
            keys.insert(k.clone());
        }
    }
    Ok(keys.into_iter().collect())
}
