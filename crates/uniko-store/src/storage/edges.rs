//! Edge CRUD operations on the knowledge graph.

use std::collections::HashMap;

use uni_db::{Transaction, Value};

use super::{KnowledgeBase, build_set_clause, validate_edge_type};
use crate::error::{Result, UnikoError};
use crate::types::{EdgeId, NodeId};

/// Direction for edge traversal queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Edges leaving the node.
    Outgoing,
    /// Edges arriving at the node.
    Incoming,
    /// Edges in either direction.
    Both,
}

/// A retrieved edge with endpoints and properties.
#[derive(Debug, Clone)]
pub struct EdgeRecord {
    /// Internal edge identifier.
    pub id: EdgeId,
    /// Relationship type name.
    pub edge_type: String,
    /// Source node identifier.
    pub from: NodeId,
    /// Target node identifier.
    pub to: NodeId,
    /// Edge properties.
    pub properties: HashMap<String, Value>,
}

impl KnowledgeBase {
    /// Create a directed edge between two nodes.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Schema`] if `edge_type` is unknown, or
    /// [`UnikoError::Storage`] on database failure.
    pub async fn create_edge(
        &self,
        edge_type: &str,
        from: NodeId,
        to: NodeId,
        properties: &HashMap<String, Value>,
    ) -> Result<EdgeId> {
        validate_edge_type(edge_type)?;
        let (set_clause, params) = build_set_clause("r", properties, 0)?;

        let cypher = if set_clause.is_empty() {
            format!(
                "MATCH (a), (b) WHERE id(a) = $src AND id(b) = $dst \
                 CREATE (a)-[r:{edge_type}]->(b) RETURN id(r) AS eid"
            )
        } else {
            format!(
                "MATCH (a), (b) WHERE id(a) = $src AND id(b) = $dst \
                 CREATE (a)-[r:{edge_type}]->(b) {set_clause} RETURN id(r) AS eid"
            )
        };

        let session = self.db.session();
        let tx = session.tx().await?;

        let mut qb = tx.query_with(&cypher);
        qb = qb.param("src", from).param("dst", to);
        for (k, v) in &params {
            qb = qb.param(k.as_str(), v.clone());
        }
        let result = qb.fetch_all().await?;

        tx.commit().await?;

        let row = result
            .rows()
            .first()
            .ok_or_else(|| UnikoError::Storage("CREATE edge returned no rows".into()))?;
        let eid: i64 = row.get("eid")?;
        Ok(eid)
    }

    /// Create a property-less `edge_type` edge from `from` to `to` unless
    /// one already exists, returning the existing or new edge id.
    ///
    /// Use this for structural links that must stay unique per node pair
    /// when the caller re-runs an upsert (for example re-creating a Goal
    /// under the same external id): [`create_edge`](Self::create_edge)
    /// always adds a new parallel edge.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Schema`] if `edge_type` is unknown, or
    /// [`UnikoError::Storage`] on database failure.
    pub async fn ensure_edge(&self, edge_type: &str, from: NodeId, to: NodeId) -> Result<EdgeId> {
        let existing = self.get_edges(from, edge_type, Direction::Outgoing).await?;
        if let Some(edge) = existing.iter().find(|e| e.to == to) {
            return Ok(edge.id);
        }
        self.create_edge(edge_type, from, to, &HashMap::new()).await
    }

    /// Create multiple edges of potentially different types in a single
    /// transaction.
    ///
    /// Each entry is `(edge_type, from, to, properties)`. All edges are
    /// committed atomically — one WAL fsync instead of one per edge.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Schema`] if any edge type is unknown, or
    /// [`UnikoError::Storage`] on database failure. The transaction is
    /// rolled back on any error.
    pub async fn create_edges_tx(
        &self,
        edges: &[(&str, NodeId, NodeId, HashMap<String, Value>)],
    ) -> Result<Vec<EdgeId>> {
        if edges.is_empty() {
            return Ok(Vec::new());
        }
        let session = self.db.session();
        let tx = session.tx().await?;
        let edge_ids = self.create_edges_in_tx(&tx, edges).await?;
        tx.commit().await?;
        Ok(edge_ids)
    }

    /// Create mixed-type edges within an existing transaction.
    ///
    /// Like [`KnowledgeBase::create_edges_tx`] but defers the commit
    /// to the caller, so several batches can share one tx (per-message
    /// ingest path folds Message + edges + chunks under a single
    /// commit using this).
    ///
    /// Internally groups input edges by `edge_type` and issues one bulk
    /// write per type via [`KnowledgeBase::batch_create_edges_fast_in_tx`].
    /// That path routes to uni-db's `bulk_insert_edges` — a direct columnar
    /// append with no Cypher parse/plan and no per-row `MATCH` — so the perf
    /// lever is bypassing the Cypher executor entirely (endpoint VIDs are
    /// already known). The (src_label, dst_label) hints threaded through from
    /// the schema table are only consulted on the `return_ids` Cypher arm,
    /// which this path never takes; on the bulk path they are inert.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Schema`] if any edge type is unknown or
    /// a property name is invalid, or [`UnikoError::Storage`] on
    /// transaction failure.
    pub async fn create_edges_in_tx(
        &self,
        tx: &Transaction,
        edges: &[(&str, NodeId, NodeId, HashMap<String, Value>)],
    ) -> Result<Vec<EdgeId>> {
        if edges.is_empty() {
            return Ok(Vec::new());
        }
        for &(edge_type, _, _, _) in edges {
            validate_edge_type(edge_type)?;
        }

        // Group by edge_type, preserving input order within each group.
        // BTreeMap so the per-type call order is deterministic — eases
        // log/diff inspection and is irrelevant to correctness.
        type EdgeTriple = (NodeId, NodeId, HashMap<String, Value>);
        let mut by_type: std::collections::BTreeMap<&str, Vec<EdgeTriple>> =
            std::collections::BTreeMap::new();
        for (edge_type, from, to, props) in edges {
            by_type
                .entry(*edge_type)
                .or_default()
                .push((*from, *to, props.clone()));
        }

        for (edge_type, group) in by_type {
            let (src_label, dst_label) = edge_type_label_hints(edge_type);
            self.batch_create_edges_fast_in_tx(tx, edge_type, src_label, dst_label, &group)
                .await?;
        }

        // Edge IDs are intentionally discarded by this helper — callers
        // that need them should use `batch_create_edges` directly.
        Ok(Vec::new())
    }

    /// Write all per-message edges via `tx.bulk_insert_edges`.
    ///
    /// Specialised for the message-ingest hot path: SENT_BY (always),
    /// IN_SESSION (always), 0..N ADDRESSED_TO, and 0..1 NEXT. Each
    /// edge type is one direct call into uni-db's bulk API — no
    /// Cypher parse/plan, no per-row MATCH lookup. The VIDs are
    /// already known at the call site (Message + Participant +
    /// Session were created earlier in the same tx).
    ///
    /// Prior implementation packed all edge types into one
    /// multi-clause Cypher; per-call exec at sess=24 averaged ~370 ms
    /// despite the writes being trivially fast — the cost was Cypher
    /// executor overhead, not the writes themselves. uni-db's own
    /// `edge_commit_contention.rs` microbench shows the bulk path
    /// achieves ~150 µs/edge at sess=24 (COLD, 10 edges/tx). Switching
    /// pulls us off the Cypher critical path entirely.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on transaction failure.
    #[expect(
        clippy::too_many_arguments,
        reason = "specialised hot-path helper — bundling args into a struct \
                  would just move the boilerplate to the call site"
    )]
    pub async fn create_message_edges_in_tx(
        &self,
        tx: &Transaction,
        msg_nid: NodeId,
        sender_nid: NodeId,
        sender_role: &str,
        session_nid: NodeId,
        recipient_nids: &[NodeId],
        prev_msg_nid: Option<NodeId>,
        gap_ms: Option<i64>,
    ) -> Result<()> {
        use uni_db::Vid;
        let t_start = std::time::Instant::now();

        // SENT_BY — always one edge, with `role` property.
        let mut sent_props: HashMap<String, Value> = HashMap::with_capacity(1);
        sent_props.insert("role".into(), Value::String(sender_role.to_string()));
        // Capture for the bulk-vs-UNWIND benchmark (no-op without the
        // `batch-record` feature). Message edges bypass `batch.rs`, so
        // they need their own capture hook at each sub-call.
        super::batch_record::record_edge_batch("SENT_BY", || vec![sent_props.clone()]);
        tx.bulk_insert_edges(
            "SENT_BY",
            vec![(
                Vid::new(msg_nid as u64),
                Vid::new(sender_nid as u64),
                sent_props,
            )],
        )
        .await?;

        // IN_SESSION — always one edge, no properties.
        super::batch_record::record_edge_batch("IN_SESSION", || vec![HashMap::new()]);
        tx.bulk_insert_edges(
            "IN_SESSION",
            vec![(
                Vid::new(msg_nid as u64),
                Vid::new(session_nid as u64),
                HashMap::new(),
            )],
        )
        .await?;

        // ADDRESSED_TO — 0..N edges, batched in one call.
        if !recipient_nids.is_empty() {
            super::batch_record::record_edge_batch("ADDRESSED_TO", || {
                vec![HashMap::new(); recipient_nids.len()]
            });
            let edges: Vec<(Vid, Vid, HashMap<String, Value>)> = recipient_nids
                .iter()
                .map(|&r| (Vid::new(msg_nid as u64), Vid::new(r as u64), HashMap::new()))
                .collect();
            tx.bulk_insert_edges("ADDRESSED_TO", edges).await?;
        }

        // NEXT — 0 or 1 edge. `gap_ms` is the inter-message delay computed
        // by the caller from the previous message's timestamp; defaults to
        // 0 when unknown (e.g. first message or a caller that omits it).
        if let Some(prev_nid) = prev_msg_nid {
            let mut next_props: HashMap<String, Value> = HashMap::with_capacity(1);
            next_props.insert("gap_ms".into(), Value::Int(gap_ms.unwrap_or(0)));
            super::batch_record::record_edge_batch("NEXT", || vec![next_props.clone()]);
            tx.bulk_insert_edges(
                "NEXT",
                vec![(
                    Vid::new(prev_nid as u64),
                    Vid::new(msg_nid as u64),
                    next_props,
                )],
            )
            .await?;
        }

        // Single emission per call. Per-sub-bulk timers and the
        // synthetic `parse_us/plan_us = 0` fields added no information
        // — these are pure bulk writes with no Cypher pipeline.
        tracing::info!(
            target: "query_metrics",
            site = "create_message_edges_in_tx",
            total_us = t_start.elapsed().as_micros() as u64,
            recipients = recipient_nids.len() as u64,
            has_prev = prev_msg_nid.is_some(),
            bulk = true,
            "query metrics",
        );
        Ok(())
    }

    /// Retrieve edges of a given type incident to `node_id`.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Schema`] if `edge_type` is unknown, or
    /// [`UnikoError::Storage`] on database failure.
    pub async fn get_edges(
        &self,
        node_id: NodeId,
        edge_type: &str,
        direction: Direction,
    ) -> Result<Vec<EdgeRecord>> {
        validate_edge_type(edge_type)?;
        self.get_edges_filtered(node_id, Some(edge_type), direction)
            .await
    }

    /// Retrieve all edges incident to `node_id` regardless of type.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn get_all_edges(
        &self,
        node_id: NodeId,
        direction: Direction,
    ) -> Result<Vec<EdgeRecord>> {
        self.get_edges_filtered(node_id, None, direction).await
    }

    async fn get_edges_filtered(
        &self,
        node_id: NodeId,
        edge_type: Option<&str>,
        direction: Direction,
    ) -> Result<Vec<EdgeRecord>> {
        let type_filter = edge_type.map(|t| format!(":{t}")).unwrap_or_default();
        let cypher = match direction {
            Direction::Outgoing => format!(
                "MATCH (a)-[r{type_filter}]->(b) WHERE id(a) = $nid \
                 RETURN r, id(r) AS eid, id(a) AS src, id(b) AS dst, type(r) AS rtype"
            ),
            Direction::Incoming => format!(
                "MATCH (a)-[r{type_filter}]->(b) WHERE id(b) = $nid \
                 RETURN r, id(r) AS eid, id(a) AS src, id(b) AS dst, type(r) AS rtype"
            ),
            Direction::Both => format!(
                "MATCH (a)-[r{type_filter}]-(b) WHERE id(a) = $nid \
                 RETURN r, id(r) AS eid, id(startNode(r)) AS src, id(endNode(r)) AS dst, type(r) AS rtype"
            ),
        };

        let session = self.db.session();
        let result = session
            .query_with(&cypher)
            .param("nid", node_id)
            .fetch_all()
            .await?;

        rows_to_edge_records(&result)
    }

    /// Delete a single edge by its internal [`EdgeId`].
    ///
    /// Returns `true` if an edge was deleted.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn delete_edge(&self, edge_id: EdgeId) -> Result<bool> {
        let session = self.db.session();
        let tx = session.tx().await?;
        let result = tx
            .execute_with("MATCH ()-[r]->() WHERE id(r) = $eid DELETE r")
            .param("eid", edge_id)
            .run()
            .await?;
        tx.commit().await?;
        Ok(result.relationships_deleted() > 0)
    }

    /// Delete all edges of `edge_type` between two specific nodes.
    ///
    /// Returns the number of edges deleted.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Schema`] if `edge_type` is unknown, or
    /// [`UnikoError::Storage`] on database failure.
    pub async fn delete_edges_between(
        &self,
        edge_type: &str,
        from: NodeId,
        to: NodeId,
    ) -> Result<u64> {
        validate_edge_type(edge_type)?;
        let cypher =
            format!("MATCH (a)-[r:{edge_type}]->(b) WHERE id(a) = $src AND id(b) = $dst DELETE r");
        let session = self.db.session();
        let tx = session.tx().await?;
        let result = tx
            .execute_with(&cypher)
            .param("src", from)
            .param("dst", to)
            .run()
            .await?;
        tx.commit().await?;
        Ok(result.relationships_deleted() as u64)
    }

    /// Update properties on an existing edge.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn update_edge(
        &self,
        edge_id: EdgeId,
        properties: &HashMap<String, Value>,
    ) -> Result<()> {
        if properties.is_empty() {
            return Ok(());
        }
        let (set_clause, params) = build_set_clause("r", properties, 0)?;
        let cypher = format!("MATCH ()-[r]->() WHERE id(r) = $eid {set_clause}");

        let session = self.db.session();
        let tx = session.tx().await?;

        let mut qb = tx.execute_with(&cypher);
        qb = qb.param("eid", edge_id);
        for (k, v) in &params {
            qb = qb.param(k.as_str(), v.clone());
        }
        qb.run().await?;
        tx.commit().await?;
        Ok(())
    }
}

/// Extract [`EdgeRecord`]s from a query result.
fn rows_to_edge_records(result: &uni_db::QueryResult) -> Result<Vec<EdgeRecord>> {
    let mut records = Vec::with_capacity(result.len());
    for row in result.rows() {
        let eid: i64 = row.get("eid")?;
        let src: i64 = row.get("src")?;
        let dst: i64 = row.get("dst")?;
        let rtype: String = row.get("rtype")?;
        let edge: uni_db::Edge = row.get("r")?;
        records.push(EdgeRecord {
            id: eid,
            edge_type: rtype,
            from: src,
            to: dst,
            properties: edge.properties,
        });
    }
    Ok(records)
}

/// Static (src_label, dst_label) hints for edge types created by
/// the per-message ingest path.
///
/// Used by [`KnowledgeBase::create_edges_in_tx`] to narrow the
/// `MATCH ... WHERE id() = $src` scan to a single label table. Edge
/// types not in this table fall back to (None, None) — the planner
/// will then scan all labels for the id match, which is the legacy
/// behavior.
///
/// The map is intentionally local to this module: it's a small,
/// closed set of ingest-path edge types, and centralising it next
/// to the helper that consumes it keeps the schema knowledge in one
/// place. If observation- or fact-side helpers ever route mixed-type
/// edges through `create_edges_in_tx`, extend this table.
fn edge_type_label_hints(edge_type: &str) -> (Option<&'static str>, Option<&'static str>) {
    use crate::schema::constants::labels;
    match edge_type {
        "SENT_BY" => (Some(labels::MESSAGE), Some(labels::PARTICIPANT)),
        "IN_SESSION" => (Some(labels::MESSAGE), Some(labels::SESSION)),
        "ADDRESSED_TO" => (Some(labels::MESSAGE), Some(labels::PARTICIPANT)),
        "NEXT" => (Some(labels::MESSAGE), Some(labels::MESSAGE)),
        _ => (None, None),
    }
}
