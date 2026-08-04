//! Cascade deletion and soft-forget for the memory graph.
//!
//! Two families of operation, mirroring the soft/hard precedent already in
//! the store (BTIC-invalidate a Fact vs `DETACH DELETE` an Episode):
//!
//! - **forget** — soft redaction. Derived Facts/Observations get a
//!   `redacted:`-scheme `visibility` (auto-excluded by the fail-closed
//!   recall policy); raw Messages/Chunks get a `redacted = true` tombstone
//!   that the recall post-filter drops. Nodes and lineage survive.
//! - **delete** — hard erasure. The owned subtree is `DETACH DELETE`d, the
//!   `NEXT` message chain is spliced back together, and any derived Fact
//!   that loses its *last* supporting Observation is soft-invalidated
//!   (BTIC closure) rather than deleted, preserving the bitemporal audit.
//!
//! Shared nodes (Entity, Participant, Session, Topic, deduped
//! ArtifactContent) are never cascaded. Each verb runs as one
//! [`transact_with_retry`](KnowledgeBase::transact_with_retry) unit so an
//! SSI conflict re-runs the whole consistent traversal+delete.

use std::collections::HashSet;

use chrono::{DateTime, Utc};
use serde::Serialize;
use uni_db::common::uni_btic::btic::POS_INF;
use uni_db::{RetryOptions, Transaction, Value};

use crate::operations::facts::{btic_to_value, extract_btic};
use crate::schema::btic::btic_invalidate;
use crate::temporal::Btic;
use crate::{KnowledgeBase, NodeId, Result};

/// Audit of what a delete/forget verb changed.
///
/// `root_existed = false` (with all counts zero) signals an idempotent
/// no-op: the target id was already gone.
#[derive(Debug, Clone, Default, Serialize)]
pub struct DeletionReport {
    /// Hard-deleted (`DETACH DELETE`d) node count.
    pub nodes_deleted: u64,
    /// Relationships removed alongside the deleted nodes.
    pub edges_deleted: u64,
    /// Facts soft-invalidated (BTIC closure) after losing their evidence.
    pub facts_invalidated: u64,
    /// `NEXT` / reading-order chain splices performed.
    pub chains_repaired: u64,
    /// Nodes soft-forgotten (visibility-redacted or `redacted` flag set).
    pub nodes_redacted: u64,
    /// Whether the target root existed (`false` ⇒ idempotent no-op).
    pub root_existed: bool,
}

impl DeletionReport {
    /// Fold another report's counts into this one.
    fn merge(&mut self, other: &DeletionReport) {
        self.nodes_deleted += other.nodes_deleted;
        self.edges_deleted += other.edges_deleted;
        self.facts_invalidated += other.facts_invalidated;
        self.chains_repaired += other.chains_repaired;
        self.nodes_redacted += other.nodes_redacted;
        self.root_existed |= other.root_existed;
    }
}

impl KnowledgeBase {
    /// Hard-delete one conversation turn and its owned derivations.
    ///
    /// Cascades the Message's Chunks and Observations, re-evaluates Facts
    /// that lose their last support, and splices the `NEXT` chain closed.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`](crate::UnikoError) on a traversal or write
    /// failure; the whole cascade is one transaction (all-or-nothing).
    pub async fn delete_message(&self, message_id: &str) -> Result<DeletionReport> {
        let now = Utc::now();
        let message_id = message_id.to_string();
        self.transact_with_retry(RetryOptions::default(), move |tx| {
            let message_id = message_id.clone();
            async move {
                let r = async {
                    match first_nid(&tx, "Message", "message_id", &message_id).await? {
                        None => Ok(DeletionReport::default()),
                        Some(mnid) => delete_message_nid(&tx, mnid, now).await,
                    }
                }
                .await;
                (tx, r)
            }
        })
        .await
    }

    /// Soft-forget one turn: hide it from recall, keep node + lineage.
    ///
    /// Sets a `redacted` tombstone on the Message and its Chunks, and a
    /// `redacted:forgotten` visibility on its Observations and any Facts
    /// supported *only* by this turn.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`](crate::UnikoError) on a write failure.
    pub async fn forget_message(&self, message_id: &str) -> Result<DeletionReport> {
        let message_id = message_id.to_string();
        self.transact_with_retry(RetryOptions::default(), move |tx| {
            let message_id = message_id.clone();
            async move {
                let r = async {
                    let Some(mnid) = first_nid(&tx, "Message", "message_id", &message_id).await?
                    else {
                        return Ok(DeletionReport::default());
                    };
                    let mut report = DeletionReport {
                        root_existed: true,
                        ..Default::default()
                    };
                    report.nodes_redacted += exec_count(
                        &tx,
                        "MATCH (m:Message) WHERE id(m) = $m SET m.redacted = true \
                         RETURN count(m) AS c",
                        mnid,
                    )
                    .await?;
                    report.nodes_redacted += exec_count(
                        &tx,
                        "MATCH (m:Message)-[:HAS_CHUNK]->(c:Chunk) WHERE id(m) = $m \
                         SET c.redacted = true RETURN count(c) AS c",
                        mnid,
                    )
                    .await?;
                    report.nodes_redacted += exec_count(
                        &tx,
                        "MATCH (o:Observation)-[:OBSERVED_IN]->(m:Message) WHERE id(m) = $m \
                         SET o.visibility = 'redacted:forgotten' RETURN count(o) AS c",
                        mnid,
                    )
                    .await?;
                    // Facts supported *only* by this turn's observations.
                    report.nodes_redacted += exec_count(
                        &tx,
                        "MATCH (f:Fact)-[:SUPPORTED_BY]->(:Observation)-[:OBSERVED_IN]->\
                            (m:Message) WHERE id(m) = $m \
                         WITH DISTINCT f \
                         MATCH (f)-[:SUPPORTED_BY]->(:Observation)-[:OBSERVED_IN]->(mm:Message) \
                         WITH f, sum(CASE WHEN id(mm) = $m THEN 0 ELSE 1 END) AS others \
                         WHERE others = 0 \
                         SET f.visibility = 'redacted:forgotten' RETURN count(f) AS c",
                        mnid,
                    )
                    .await?;
                    Ok(report)
                }
                .await;
                (tx, r)
            }
        })
        .await
    }

    /// Hard-delete an entire conversation and everything owned by it.
    ///
    /// Removes the Session, all its Messages and their Chunks/Observations,
    /// and any Artifact attached *solely* to this session (with its
    /// document subtree). Facts losing their last support are
    /// soft-invalidated. `NEXT` chains live entirely within the session, so
    /// no splicing is needed.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`](crate::UnikoError) on a traversal or write
    /// failure.
    pub async fn delete_session_cascade(&self, session_id: &str) -> Result<DeletionReport> {
        let now = Utc::now();
        let session_id = session_id.to_string();
        self.transact_with_retry(RetryOptions::default(), move |tx| {
            let session_id = session_id.clone();
            async move {
                let r = delete_session_in_tx(&tx, &session_id, now).await;
                (tx, r)
            }
        })
        .await
    }

    /// Hard-delete a document Artifact and its structure subtree.
    ///
    /// Removes the Artifact, its Pages, Blocks, and Chunks (direct and
    /// per-block). The deduped, possibly-shared `:ArtifactContent` blob is
    /// left in place (content-addressed storage, garbage-collected
    /// separately).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`](crate::UnikoError) on a traversal or write
    /// failure.
    pub async fn delete_artifact(&self, artifact_id: &str) -> Result<DeletionReport> {
        let artifact_id = artifact_id.to_string();
        self.transact_with_retry(RetryOptions::default(), move |tx| {
            let artifact_id = artifact_id.clone();
            async move {
                let r = async {
                    let Some(anid) =
                        first_nid(&tx, "Artifact", "artifact_id", &artifact_id).await?
                    else {
                        return Ok(DeletionReport::default());
                    };
                    let mut report = DeletionReport {
                        root_existed: true,
                        ..Default::default()
                    };
                    let mut owned = artifact_subtree_ids(&tx, anid).await?;
                    // Drop the content-addressed :ArtifactContent only when
                    // no *other* Artifact shares it (content is deduped). The
                    // physical blob in blob storage is GC'd separately.
                    owned.extend(unshared_content_ids(&tx, anid).await?);
                    let (n, e) = detach_delete_ids(&tx, &owned).await?;
                    report.nodes_deleted = n;
                    report.edges_deleted = e;
                    Ok(report)
                }
                .await;
                (tx, r)
            }
        })
        .await
    }

    /// GDPR erasure: remove a Participant and the data they authored.
    ///
    /// Deletes every Message sent by the participant (each with its turn
    /// cascade and `NEXT` splice), every Observation about them, and the
    /// Participant node itself. Facts supported by evidence about the
    /// participant are force-invalidated (`subject_erased`) even when other
    /// support remains — erasure overrides reference-counting.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`](crate::UnikoError) on a traversal or write
    /// failure.
    pub async fn forget_participant(&self, participant_id: &str) -> Result<DeletionReport> {
        let now = Utc::now();
        let participant_id = participant_id.to_string();
        self.transact_with_retry(RetryOptions::default(), move |tx| {
            let participant_id = participant_id.clone();
            async move {
                let r = forget_participant_in_tx(&tx, &participant_id, now).await;
                (tx, r)
            }
        })
        .await
    }

    /// Erase the entire graph. Intended for dev/test reset.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`](crate::UnikoError) on a write failure.
    pub async fn purge_all(&self) -> Result<DeletionReport> {
        self.transact_with_retry(RetryOptions::default(), move |tx| async move {
            let r = async {
                let res = tx.execute_with("MATCH (n) DETACH DELETE n").run().await?;
                Ok(DeletionReport {
                    nodes_deleted: res.nodes_deleted() as u64,
                    edges_deleted: res.relationships_deleted() as u64,
                    root_existed: res.nodes_deleted() > 0,
                    ..Default::default()
                })
            }
            .await;
            (tx, r)
        })
        .await
    }

    /// Subset of `node_ids` whose `redacted` flag is `true`.
    ///
    /// Used by the recall post-filter to drop soft-forgotten Messages and
    /// Chunks (which carry no visibility scheme).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`](crate::UnikoError) on a read failure.
    pub async fn fetch_redacted(&self, node_ids: &[NodeId]) -> Result<HashSet<NodeId>> {
        if node_ids.is_empty() {
            return Ok(HashSet::new());
        }
        let list: Vec<Value> = node_ids.iter().map(|n| Value::Int(*n)).collect();
        let session = self.db.session();
        let result = session
            .query_with("MATCH (n) WHERE id(n) IN $ids AND n.redacted = true RETURN id(n) AS nid")
            .param("ids", Value::List(list))
            .fetch_all()
            .await?;
        let mut out = HashSet::with_capacity(result.rows().len());
        for row in result.rows() {
            out.insert(row.get::<i64>("nid")?);
        }
        Ok(out)
    }

    /// `DETACH DELETE` `ids` inside a caller-owned transaction.
    ///
    /// Returns `(nodes_deleted, relationships_deleted)`. Exposed so callers
    /// that rebuild a derived surface can delete the old nodes and write the
    /// replacements in one transaction, rather than leaving a window where
    /// recall observes neither.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`](crate::UnikoError) on a write failure.
    pub async fn detach_delete_nodes_in_tx(
        &self,
        tx: &Transaction,
        ids: &[NodeId],
    ) -> Result<(u64, u64)> {
        detach_delete_ids(tx, ids).await
    }
}

// ── In-transaction cascade steps ────────────────────────────────────────

/// Hard-delete a single Message (by node id) and its owned derivations.
async fn delete_message_nid(
    tx: &Transaction,
    mnid: NodeId,
    now: DateTime<Utc>,
) -> Result<DeletionReport> {
    let mut report = DeletionReport {
        root_existed: true,
        ..Default::default()
    };

    // Observations are OBSERVED_IN exactly one Message — owned.
    let observations = ids_by_anchor(
        tx,
        "MATCH (o:Observation)-[:OBSERVED_IN]->(m:Message) WHERE id(m) = $a RETURN id(o) AS nid",
        mnid,
    )
    .await?;
    // Re-evaluate Facts BEFORE the observations vanish.
    report.facts_invalidated = reevaluate_facts(tx, &observations, now).await?;

    let chunks = ids_by_anchor(
        tx,
        "MATCH (m:Message)-[:HAS_CHUNK]->(c:Chunk) WHERE id(m) = $a RETURN id(c) AS nid",
        mnid,
    )
    .await?;

    // Splice the NEXT chain closed while the victim still exists.
    report.chains_repaired = splice_next(tx, mnid).await?;

    let mut doomed = Vec::with_capacity(1 + chunks.len() + observations.len());
    doomed.push(mnid);
    doomed.extend(chunks);
    doomed.extend(observations);
    let (n, e) = detach_delete_ids(tx, &doomed).await?;
    report.nodes_deleted = n;
    report.edges_deleted = e;
    Ok(report)
}

/// Hard-delete a whole session subtree, set-based.
async fn delete_session_in_tx(
    tx: &Transaction,
    session_id: &str,
    now: DateTime<Utc>,
) -> Result<DeletionReport> {
    let Some(snid) = first_nid(tx, "Session", "session_id", session_id).await? else {
        return Ok(DeletionReport::default());
    };
    let mut report = DeletionReport {
        root_existed: true,
        ..Default::default()
    };

    let observations = ids_by_anchor(
        tx,
        "MATCH (o:Observation)-[:OBSERVED_IN]->(:Message)-[:IN_SESSION]->(s:Session) \
         WHERE id(s) = $a RETURN id(o) AS nid",
        snid,
    )
    .await?;
    report.facts_invalidated = reevaluate_facts(tx, &observations, now).await?;

    let messages = ids_by_anchor(
        tx,
        "MATCH (m:Message)-[:IN_SESSION]->(s:Session) WHERE id(s) = $a RETURN id(m) AS nid",
        snid,
    )
    .await?;
    let chunks = ids_by_anchor(
        tx,
        "MATCH (:Message)-[:IN_SESSION]->(s:Session) WHERE id(s) = $a \
         WITH s MATCH (m:Message)-[:IN_SESSION]->(s)-[]-() \
         WITH DISTINCT m MATCH (m)-[:HAS_CHUNK]->(c:Chunk) RETURN DISTINCT id(c) AS nid",
        snid,
    )
    .await?;
    // Session-anchored chunks (transcript + observation surfaces built by
    // `Session::finalize`) hang off the Session itself, not off a Message,
    // so the Message-anchored walk above misses them. Left behind they stay
    // live in the vector and full-text indexes and remain recallable after
    // the session is deleted.
    let session_chunks = ids_by_anchor(
        tx,
        "MATCH (s:Session)-[:HAS_CHUNK]->(c:Chunk) WHERE id(s) = $a RETURN DISTINCT id(c) AS nid",
        snid,
    )
    .await?;

    // Artifacts attached *solely* to this session → delete their subtrees.
    let solo_artifacts = ids_by_anchor(
        tx,
        "MATCH (a:Artifact)-[:ATTACHED_TO]->(s:Session) WHERE id(s) = $a \
         OPTIONAL MATCH (a)-[:ATTACHED_TO]->(other:Session) WHERE id(other) <> $a \
         WITH a, count(other) AS others WHERE others = 0 RETURN id(a) AS nid",
        snid,
    )
    .await?;
    let mut artifact_owned: Vec<NodeId> = Vec::new();
    for anid in &solo_artifacts {
        artifact_owned.extend(artifact_subtree_ids(tx, *anid).await?);
    }

    let mut doomed = Vec::new();
    doomed.push(snid);
    doomed.extend(messages);
    doomed.extend(chunks);
    doomed.extend(session_chunks);
    doomed.extend(observations);
    doomed.extend(artifact_owned);
    let (n, e) = detach_delete_ids(tx, &doomed).await?;
    report.nodes_deleted = n;
    report.edges_deleted = e;
    Ok(report)
}

/// GDPR cascade for a participant.
async fn forget_participant_in_tx(
    tx: &Transaction,
    participant_id: &str,
    now: DateTime<Utc>,
) -> Result<DeletionReport> {
    let Some(pnid) = first_nid(tx, "Participant", "participant_id", participant_id).await? else {
        return Ok(DeletionReport::default());
    };
    let mut report = DeletionReport {
        root_existed: true,
        ..Default::default()
    };

    // Force-invalidate Facts whose evidence is about this participant —
    // erasure overrides reference-counting.
    report.facts_invalidated += force_invalidate_facts_about(tx, pnid, now).await?;

    // Delete every Message the participant sent, each with its turn
    // cascade + NEXT splice (handles scattered / contiguous runs).
    let sent = ids_by_anchor(
        tx,
        "MATCH (m:Message)-[:SENT_BY]->(p:Participant) WHERE id(p) = $a RETURN id(m) AS nid",
        pnid,
    )
    .await?;
    for mnid in sent {
        let sub = delete_message_nid(tx, mnid, now).await?;
        report.merge(&sub);
    }

    // Remaining Observations about the participant (not tied to a sent
    // message) — re-eval their Facts, then delete.
    let about_obs = ids_by_anchor(
        tx,
        "MATCH (o:Observation)-[:ABOUT]->(p:Participant) WHERE id(p) = $a RETURN id(o) AS nid",
        pnid,
    )
    .await?;
    report.facts_invalidated += reevaluate_facts(tx, &about_obs, now).await?;

    let mut doomed = vec![pnid];
    doomed.extend(about_obs);
    let (n, e) = detach_delete_ids(tx, &doomed).await?;
    report.nodes_deleted += n;
    report.edges_deleted += e;
    Ok(report)
}

/// Collect every node owned by a PDF/document Artifact: its Pages, Blocks,
/// and Chunks (direct + per-block), plus the Artifact itself.
async fn artifact_subtree_ids(tx: &Transaction, anid: NodeId) -> Result<Vec<NodeId>> {
    ids_by_anchor(
        tx,
        // Group by `aid` (a scalar) so the `collect`s don't mix with a
        // non-grouped `id(a)` in one projection (which the SQL planner
        // rejects as an ambiguous aggregation).
        "MATCH (a:Artifact) WHERE id(a) = $a \
         OPTIONAL MATCH (a)-[:HAS_PAGE]->(pg:Page) \
         OPTIONAL MATCH (pg)-[:CONTAINS]->(bl:Block) \
         OPTIONAL MATCH (a)-[:HAS_CHUNK]->(dc:Chunk) \
         OPTIONAL MATCH (bl)-[:HAS_CHUNK]->(bc:Chunk) \
         WITH id(a) AS aid, collect(DISTINCT id(pg)) AS pgs, collect(DISTINCT id(bl)) AS bls, \
              collect(DISTINCT id(dc)) AS dcs, collect(DISTINCT id(bc)) AS bcs \
         UNWIND ([aid] + pgs + bls + dcs + bcs) AS nid \
         WITH DISTINCT nid WHERE nid IS NOT NULL RETURN nid",
        anid,
    )
    .await
}

/// The Artifact's `:ArtifactContent` node id, but only when no other
/// Artifact references it (deduped content may be shared). Counts all
/// referrers; `refs == 1` means only this Artifact points at it. The
/// non-optional second MATCH keeps the planner happy and `ac` survives
/// because the doomed Artifact itself is still a referrer at this point.
async fn unshared_content_ids(tx: &Transaction, anid: NodeId) -> Result<Vec<NodeId>> {
    ids_by_anchor(
        tx,
        "MATCH (a:Artifact)-[:HAS_CONTENT]->(ac:ArtifactContent) WHERE id(a) = $a \
         WITH DISTINCT ac \
         MATCH (ref:Artifact)-[:HAS_CONTENT]->(ac) \
         WITH ac, count(ref) AS refs \
         WHERE refs = 1 RETURN id(ac) AS nid",
        anid,
    )
    .await
}

/// Soft-invalidate Facts that lose their last supporting Observation.
///
/// Returns the count invalidated.
async fn reevaluate_facts(
    tx: &Transaction,
    doomed_observations: &[NodeId],
    now: DateTime<Utc>,
) -> Result<u64> {
    if doomed_observations.is_empty() {
        return Ok(0);
    }
    let list: Vec<Value> = doomed_observations.iter().map(|n| Value::Int(*n)).collect();
    // Count *surviving* supports with a CASE-sum rather than
    // `OPTIONAL MATCH ... WHERE`: the doomed SUPPORTED_BY edges still exist
    // at this point (deletion happens after re-eval), so the inner MATCH
    // always matches, and we avoid the optional-row-drop ambiguity some
    // Cypher engines have when the WHERE eliminates the only match.
    let result = tx
        .query_with(
            "MATCH (f:Fact)-[:SUPPORTED_BY]->(o:Observation) WHERE id(o) IN $doomed \
             WITH DISTINCT f \
             MATCH (f)-[:SUPPORTED_BY]->(allo:Observation) \
             WITH f, sum(CASE WHEN id(allo) IN $doomed THEN 0 ELSE 1 END) AS remaining \
             WHERE remaining = 0 \
             RETURN id(f) AS fid, f.valid_at AS valid_at",
        )
        .param("doomed", Value::List(list))
        .fetch_all()
        .await?;

    let targets = collect_open_facts(result.rows())?;
    close_facts_in_tx(tx, targets, "evidence_removed", now).await
}

/// Force-invalidate every Fact supported by an Observation about `pnid`,
/// regardless of remaining support (GDPR `subject_erased`).
async fn force_invalidate_facts_about(
    tx: &Transaction,
    pnid: NodeId,
    now: DateTime<Utc>,
) -> Result<u64> {
    let result = tx
        .query_with(
            "MATCH (f:Fact)-[:SUPPORTED_BY]->(:Observation)-[:ABOUT]->(p:Participant) \
             WHERE id(p) = $a \
             RETURN DISTINCT id(f) AS fid, f.valid_at AS valid_at",
        )
        .param("a", pnid)
        .fetch_all()
        .await?;
    let targets = collect_open_facts(result.rows())?;
    close_facts_in_tx(tx, targets, "subject_erased", now).await
}

/// Decode `(fid, valid_at)` rows, keeping only Facts whose interval is still
/// **open** (`hi == POS_INF`). Skipping already-closed Facts makes
/// invalidation idempotent and avoids double-counting when a Fact is reached
/// by more than one pass (e.g. GDPR force-invalidate then per-message
/// re-eval).
fn collect_open_facts(rows: &[uni_db::Row]) -> Result<Vec<(NodeId, Btic)>> {
    let mut targets = Vec::with_capacity(rows.len());
    for row in rows {
        let fid: i64 = row.get("fid")?;
        if let Some(btic) = extract_btic(row.value("valid_at"))
            && btic.hi() == POS_INF
        {
            targets.push((fid, btic));
        }
    }
    Ok(targets)
}

/// Close each Fact's `valid_at` interval and stamp the node-level audit
/// (`invalidation_reason`, `invalidated_at`). Returns the count closed.
async fn close_facts_in_tx(
    tx: &Transaction,
    targets: Vec<(NodeId, Btic)>,
    reason: &str,
    now: DateTime<Utc>,
) -> Result<u64> {
    let count = targets.len() as u64;
    let now_val = crate::types::datetime_value(now);
    for (fid, btic) in targets {
        tx.execute_with(
            "MATCH (f:Fact) WHERE id(f) = $fid \
             SET f.valid_at = $va, f.invalidation_reason = $reason, f.invalidated_at = $now",
        )
        .param("fid", fid)
        .param("va", btic_to_value(&btic_invalidate(&btic, now)))
        .param("reason", reason)
        .param("now", now_val.clone())
        .run()
        .await?;
    }
    Ok(count)
}

/// Splice `(prev)-[:NEXT]->(victim)-[:NEXT]->(next)` into `(prev)-[:NEXT]->
/// (next)`, summing the two `gap_ms` values. Returns 1 if spliced, else 0.
async fn splice_next(tx: &Transaction, mnid: NodeId) -> Result<u64> {
    let result = tx
        .query_with(
            "MATCH (prev:Message)-[r1:NEXT]->(m:Message)-[r2:NEXT]->(next:Message) \
             WHERE id(m) = $m \
             RETURN id(prev) AS prev, id(next) AS next, \
                    coalesce(r1.gap_ms, 0) + coalesce(r2.gap_ms, 0) AS gap",
        )
        .param("m", mnid)
        .fetch_all()
        .await?;
    let Some(row) = result.rows().first() else {
        return Ok(0);
    };
    let prev: i64 = row.get("prev")?;
    let next: i64 = row.get("next")?;
    let gap: i64 = row.get("gap")?;
    tx.execute_with(
        "MATCH (p:Message), (n:Message) WHERE id(p) = $p AND id(n) = $n \
         MERGE (p)-[r:NEXT]->(n) SET r.gap_ms = $gap",
    )
    .param("p", prev)
    .param("n", next)
    .param("gap", gap)
    .run()
    .await?;
    Ok(1)
}

/// `DETACH DELETE` a set of node ids, returning (nodes, relationships).
async fn detach_delete_ids(tx: &Transaction, ids: &[NodeId]) -> Result<(u64, u64)> {
    if ids.is_empty() {
        return Ok((0, 0));
    }
    let list: Vec<Value> = ids.iter().map(|n| Value::Int(*n)).collect();
    let res = tx
        .execute_with("MATCH (n) WHERE id(n) IN $ids DETACH DELETE n")
        .param("ids", Value::List(list))
        .run()
        .await?;
    Ok((
        res.nodes_deleted() as u64,
        res.relationships_deleted() as u64,
    ))
}

/// Resolve a node id from a label + external id field.
async fn first_nid(
    tx: &Transaction,
    label: &str,
    id_field: &str,
    ext_id: &str,
) -> Result<Option<NodeId>> {
    // `label` / `id_field` are static schema constants, never user input.
    let cypher = format!("MATCH (n:{label} {{{id_field}: $id}}) RETURN id(n) AS nid");
    let result = tx
        .query_with(&cypher)
        .param("id", ext_id)
        .fetch_all()
        .await?;
    match result.rows().first() {
        Some(row) => Ok(Some(row.get::<i64>("nid")?)),
        None => Ok(None),
    }
}

/// Run a `RETURN ... AS nid` query anchored on `$a`, collecting node ids.
async fn ids_by_anchor(tx: &Transaction, cypher: &str, anchor: NodeId) -> Result<Vec<NodeId>> {
    let result = tx.query_with(cypher).param("a", anchor).fetch_all().await?;
    let mut out = Vec::with_capacity(result.rows().len());
    for row in result.rows() {
        out.push(row.get::<i64>("nid")?);
    }
    Ok(out)
}

/// Run a write anchored on `$m` that ends in `RETURN count(...) AS c`.
async fn exec_count(tx: &Transaction, cypher: &str, anchor: NodeId) -> Result<u64> {
    let result = tx.query_with(cypher).param("m", anchor).fetch_all().await?;
    match result.rows().first() {
        Some(row) => Ok(row.get::<i64>("c")? as u64),
        None => Ok(0),
    }
}
