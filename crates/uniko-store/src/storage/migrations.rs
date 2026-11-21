//! One-shot, idempotent schema migrations for the multimodal store
//! work landed in Phase 1.
//!
//! All migrations are host-side (Rust) rather than pure Cypher so they
//! can branch on [`BlobStorage`](crate::blob_store::BlobStorage). Each
//! migration is safe to re-run — MERGE-on-`content_id` is naturally
//! idempotent and backend PUTs are idempotent on the hash key (CAS
//! dedup).
//
use std::collections::HashMap;

use chrono::Utc;
use uni_db::Value;

use crate::error::Result;
use crate::schema::constants::edges as edge_consts;
use crate::storage::KnowledgeBase;
use crate::storage::blob::MergeContent;
use crate::types::{NodeId, datetime_value};

/// Summary of a migration pass.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct MigrationReport {
    /// Rows scanned by the migration.
    pub scanned: usize,
    /// Rows migrated (newly written content / edge).
    pub migrated: usize,
    /// Rows skipped because they were already migrated.
    pub already_migrated: usize,
}

impl KnowledgeBase {
    /// Migrate `Artifact.content` → `:ArtifactContent` per §8.1.
    ///
    /// Reads every `:Artifact` row with `content IS NOT NULL`, writes
    /// the bytes to the configured blob backend, MERGEs the
    /// `:ArtifactContent` node, MERGEs a `HAS_CONTENT` edge, and
    /// removes `a.content`. Idempotent — re-running on an already-
    /// migrated KB returns `MigrationReport { migrated: 0, ... }`.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database / backend failure.
    pub async fn migrate_artifact_content(&self) -> Result<MigrationReport> {
        let session = self.db.session();
        let result = session
            .query_with(
                "MATCH (a:Artifact) WHERE a.content IS NOT NULL \
                 RETURN id(a) AS vid, a.content AS content, a.hash AS hash, \
                        a.kind AS kind",
            )
            .fetch_all()
            .await?;

        let mut report = MigrationReport::default();
        for row in result.rows() {
            report.scanned += 1;

            let vid: NodeId = row.get("vid")?;
            let content = match row.value("content") {
                Some(Value::String(s)) => s.clone(),
                _ => continue, // already-null content from a partial migration
            };
            let cached_hash = match row.value("hash") {
                Some(Value::String(s)) => Some(s.clone()),
                _ => None,
            };
            let kind = match row.value("kind") {
                Some(Value::String(s)) => s.clone(),
                _ => "text".into(),
            };

            let bytes = content.as_bytes();
            let hash = cached_hash.unwrap_or_else(|| Self::sha256_hex(bytes));

            // PUT bytes via the configured backend (no-op return for
            // Lance; precedes the MERGE for Fs/S3).
            let put = self.put_blob(&hash, bytes).await?;
            let content_nid = self
                .merge_artifact_content(MergeContent {
                    content_id: hash.clone(),
                    bytes: put.bytes_inline,
                    uri: put.uri,
                    mime: best_mime_guess(&kind),
                    size: bytes.len() as i64,
                    perceptual_hash: None,
                    audio_fingerprint: None,
                })
                .await?;

            self.ensure_writable()?;

            let tx = session.tx().await?;
            // Detect a prior migration (for the report counter only).
            let existing_edge = tx
                .query_with(
                    "MATCH (a:Artifact)-[r:HAS_CONTENT]->(c:ArtifactContent) \
                     WHERE id(a) = $a AND id(c) = $c RETURN id(r) AS eid LIMIT 1",
                )
                .param("a", vid)
                .param("c", content_nid)
                .fetch_all()
                .await?;

            // uni-db 2.0 supports MERGE on a relationship between two
            // id()-addressed endpoints, so the write is idempotent in one
            // statement (was a manual conditional CREATE before edge-MERGE
            // was expressible).
            tx.query_with(
                "MATCH (a:Artifact), (c:ArtifactContent) \
                 WHERE id(a) = $a AND id(c) = $c \
                 MERGE (a)-[:HAS_CONTENT {role: 'primary'}]->(c)",
            )
            .param("a", vid)
            .param("c", content_nid)
            .fetch_all()
            .await?;

            // Strip the legacy content field.
            tx.query_with(
                "MATCH (a:Artifact) WHERE id(a) = $a \
                 SET a.content = NULL, a.hash = $hash, a.updated_at = $now",
            )
            .param("a", vid)
            .param("hash", Value::String(hash))
            .param("now", datetime_value(Utc::now()))
            .fetch_all()
            .await?;

            tx.commit().await?;

            if existing_edge.rows().is_empty() {
                report.migrated += 1;
            } else {
                report.already_migrated += 1;
            }
        }

        Ok(report)
    }

    /// Mean-pool a single Artifact's child Chunk embeddings into
    /// `Artifact.text_embedding`. Skips Chunks whose `embedding` is
    /// NULL. No-op when the artifact has zero pool-able chunks.
    ///
    /// Returns `true` when a pooled vector was written, `false` if
    /// nothing was poolable (no chunks, or all embeddings NULL).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn mean_pool_artifact_text_embedding(&self, artifact_nid: NodeId) -> Result<bool> {
        let session = self.db.session();
        let result = session
            .query_with(
                "MATCH (a:Artifact)-[:HAS_CHUNK]->(c:Chunk) \
                 WHERE id(a) = $a AND c.embedding IS NOT NULL \
                 RETURN c.embedding AS emb",
            )
            .param("a", artifact_nid)
            .fetch_all()
            .await?;

        let mut vectors: Vec<Vec<f32>> = Vec::new();
        for row in result.rows() {
            if let Some(Value::Vector(v)) = row.value("emb") {
                vectors.push(v.clone());
            }
        }
        let Some(pooled) = mean_pool(&vectors) else {
            return Ok(false);
        };

        let mut props = HashMap::new();
        props.insert("text_embedding".into(), Value::Vector(pooled));
        self.update_node(artifact_nid, &props).await?;
        Ok(true)
    }

    /// Backfill `Artifact.text_embedding` for all Artifacts that have
    /// chunks but no pooled vector yet. Idempotent — skips rows where
    /// `text_embedding IS NOT NULL`.
    ///
    /// Returns the count of artifacts that received a pooled vector.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn backfill_artifact_text_embedding(&self) -> Result<usize> {
        let session = self.db.session();
        let result = session
            .query_with(
                "MATCH (a:Artifact) \
                 WHERE a.text_embedding IS NULL AND (a)-[:HAS_CHUNK]->(:Chunk) \
                 RETURN id(a) AS vid",
            )
            .fetch_all()
            .await?;

        let mut count = 0;
        for row in result.rows() {
            let vid: NodeId = row.get("vid")?;
            if self.mean_pool_artifact_text_embedding(vid).await? {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Migrate every `:Entity` to the canonical `entity_id` scheme (issue #1).
    ///
    /// Historically three writers used three divergent id schemes for the same
    /// logical entity (`type:name`, `hash(name,UPPER_TYPE)`, and bare `name`).
    /// This recomputes `entity_id = id::entity_id(name, canonical_type)` and
    /// normalizes the stored `entity_type` to the shared lowercase vocabulary
    /// for every Entity, then **merges** rows that collapse to the same
    /// canonical id: a single survivor keeps the row, its inbound `MENTIONS`
    /// and `ABOUT` edges and outbound `BELONGS_TO` edges are repointed from the
    /// duplicates, `frequency`/`invalidation_count` are summed, and the
    /// duplicates are deleted.
    ///
    /// Because `Topic.topic_id` is derived from member `entity_id`s, all
    /// `:Topic` nodes are dropped so they re-derive on the next topic-detection
    /// pass.
    ///
    /// Operator-invoked (like [`KnowledgeBase::migrate_artifact_content`]); not
    /// auto-run at open. Idempotent — a second run finds every id already
    /// canonical and merges nothing.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn migrate_entity_ids_to_canonical(&self) -> Result<MigrationReport> {
        let session = self.db.session();
        let rows = session
            .query_with(
                "MATCH (e:Entity) RETURN id(e) AS vid, e.name AS name, \
                 e.entity_type AS etype, e.entity_id AS eid, \
                 coalesce(e.frequency, 0) AS freq, \
                 coalesce(e.invalidation_count, 0) AS invc",
            )
            .fetch_all()
            .await?;

        // Group current rows by their target canonical id.
        struct Ent {
            vid: NodeId,
            freq: i64,
            invc: i64,
            current_eid: String,
        }
        let mut groups: HashMap<String, (String /*ctype*/, Vec<Ent>)> = HashMap::new();
        let mut report = MigrationReport::default();
        for row in rows.rows() {
            report.scanned += 1;
            let vid: NodeId = row.get("vid")?;
            let name: String = match row.value("name") {
                Some(Value::String(s)) => s.clone(),
                _ => continue, // nameless entity — cannot canonicalize
            };
            let etype = match row.value("etype") {
                Some(Value::String(s)) => s.clone(),
                _ => String::new(),
            };
            let ctype = canonical_entity_type(&etype);
            let canonical_eid = crate::id::entity_id(&name, ctype);
            let current_eid = match row.value("eid") {
                Some(Value::String(s)) => s.clone(),
                _ => String::new(),
            };
            let freq: i64 = row.get("freq").unwrap_or(0);
            let invc: i64 = row.get("invc").unwrap_or(0);
            groups
                .entry(canonical_eid)
                .or_insert_with(|| (ctype.to_string(), Vec::new()))
                .1
                .push(Ent {
                    vid,
                    freq,
                    invc,
                    current_eid,
                });
        }

        for (canonical_eid, (ctype, mut members)) in groups {
            // Deterministic survivor = lowest vid.
            members.sort_by_key(|m| m.vid);
            let survivor = members[0].vid;
            let total_freq: i64 = members.iter().map(|m| m.freq).sum();
            let total_invc: i64 = members.iter().map(|m| m.invc).sum();
            let already_canonical = members.len() == 1 && members[0].current_eid == canonical_eid;

            self.ensure_writable()?;

            let tx = session.tx().await?;
            // Repoint the known Entity edge types from each duplicate onto the
            // survivor, then delete the duplicate.
            for dup in members.iter().skip(1) {
                for (etype, inbound) in [
                    (edge_consts::MENTIONS, true),
                    (edge_consts::ABOUT, true),
                    (edge_consts::BELONGS_TO, false),
                ] {
                    let cypher = if inbound {
                        format!(
                            "MATCH (x)-[r:{etype}]->(d) WHERE id(d) = $d \
                             MATCH (s) WHERE id(s) = $s \
                             MERGE (x)-[:{etype}]->(s) DELETE r"
                        )
                    } else {
                        format!(
                            "MATCH (d)-[r:{etype}]->(x) WHERE id(d) = $d \
                             MATCH (s) WHERE id(s) = $s \
                             MERGE (s)-[:{etype}]->(x) DELETE r"
                        )
                    };
                    tx.query_with(&cypher)
                        .param("d", dup.vid)
                        .param("s", survivor)
                        .fetch_all()
                        .await?;
                }
                tx.query_with("MATCH (d) WHERE id(d) = $d DETACH DELETE d")
                    .param("d", dup.vid)
                    .fetch_all()
                    .await?;
            }

            // Canonicalize the survivor.
            tx.query_with(
                "MATCH (s) WHERE id(s) = $s \
                 SET s.entity_id = $eid, s.entity_type = $etype, \
                     s.frequency = $freq, s.invalidation_count = $invc",
            )
            .param("s", survivor)
            .param("eid", Value::String(canonical_eid))
            .param("etype", Value::String(ctype))
            .param("freq", Value::Int(total_freq))
            .param("invc", Value::Int(total_invc))
            .fetch_all()
            .await?;
            tx.commit().await?;

            if already_canonical {
                report.already_migrated += 1;
            } else {
                report.migrated += 1;
            }
        }

        // Topic ids are derived from member entity_ids — drop Topics so they
        // re-derive on the next detection pass.
        self.ensure_writable()?;
        let tx = session.tx().await?;
        tx.query_with("MATCH (t:Topic) DETACH DELETE t")
            .fetch_all()
            .await?;
        tx.commit().await?;

        Ok(report)
    }

    /// Backfill `Chunk.modality = 'text'` for rows where it is NULL.
    /// Per §8.2. Idempotent.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn backfill_chunk_modality_text(&self) -> Result<usize> {
        let session = self.db.session();
        self.ensure_writable()?;
        let tx = session.tx().await?;
        let result = tx
            .query_with(
                "MATCH (c:Chunk) WHERE c.modality IS NULL \
                 SET c.modality = 'text' RETURN count(c) AS updated",
            )
            .fetch_all()
            .await?;
        let updated: i64 = result
            .rows()
            .first()
            .and_then(|r| r.get("updated").ok())
            .unwrap_or(0);
        tx.commit().await?;
        Ok(updated as usize)
    }

    /// Up to `limit` chunks that have no `search_text` yet, as
    /// `(node id, text, context)`. Paired with
    /// [`set_chunk_search_text`](Self::set_chunk_search_text) by the
    /// `search_text` backfill, which owns the normalisation.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn chunks_missing_search_text(
        &self,
        limit: usize,
    ) -> Result<Vec<(NodeId, String, Option<String>)>> {
        let rows = self
            .db
            .session()
            .query_with(
                "MATCH (c:Chunk) WHERE c.search_text IS NULL \
                 RETURN id(c) AS nid, c.text AS text, c.context AS ctx LIMIT $lim",
            )
            .param("lim", Value::Int(i64::try_from(limit).unwrap_or(i64::MAX)))
            .fetch_all()
            .await?;
        Ok(rows
            .rows()
            .iter()
            .filter_map(|r| {
                let nid: i64 = r.get("nid").ok()?;
                let text: String = r.get("text").unwrap_or_default();
                let ctx: Option<String> = r.get("ctx").ok();
                Some((nid, text, ctx))
            })
            .collect())
    }

    /// Write `search_text` for each `(node id, text)` in one transaction.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn set_chunk_search_text(&self, updates: &[(NodeId, String)]) -> Result<usize> {
        if updates.is_empty() {
            return Ok(0);
        }
        let session = self.db.session();
        self.ensure_writable()?;
        let tx = session.tx().await?;
        for (nid, text) in updates {
            tx.query_with("MATCH (c:Chunk) WHERE id(c) = $nid SET c.search_text = $t")
                .param("nid", Value::Int(*nid))
                .param("t", Value::String(text.clone()))
                .fetch_all()
                .await?;
        }
        tx.commit().await?;
        Ok(updates.len())
    }
}

/// Average a slice of equal-length f32 vectors. Returns `None` for
/// empty input or when widths disagree (an internal-invariant failure
/// we surface rather than silently mis-pool).
fn mean_pool(vectors: &[Vec<f32>]) -> Option<Vec<f32>> {
    let first = vectors.first()?;
    let dim = first.len();
    if dim == 0 || vectors.iter().any(|v| v.len() != dim) {
        return None;
    }
    let n = vectors.len() as f32;
    let mut acc = vec![0.0f32; dim];
    for v in vectors {
        for (a, x) in acc.iter_mut().zip(v.iter()) {
            *a += *x;
        }
    }
    for a in &mut acc {
        *a /= n;
    }
    Some(acc)
}

/// Normalize any historical or current `entity_type` string to the shared
/// canonical lowercase vocabulary (issue #1). Maps the legacy uppercase
/// abbreviations the action path used (`PERSON`, `ORG`, …), the lowercase
/// forms the dedup path used (`person`, `organization`, …), and `other`/empty
/// to a single value each. Unknown values pass through lowercased.
fn canonical_entity_type(raw: &str) -> &'static str {
    match raw.trim().to_ascii_lowercase().as_str() {
        "person" => "person",
        "organization" | "org" => "organization",
        "location" | "loc" => "location",
        "date" => "date",
        "numeric" | "num" => "numeric",
        "event" => "event",
        "product" => "product",
        "work_of_art" => "work_of_art",
        "group" => "group",
        "url" => "url",
        "email" => "email",
        "measurement" => "measurement",
        "preference" => "preference",
        "quoted_string" => "quoted_string",
        "code_symbol" => "code_symbol",
        "code_import" => "code_import",
        // `other`, `misc`, empty, and anything unrecognized.
        _ => "misc",
    }
}

/// Coarse MIME guess from the legacy `Artifact.kind` value. Text-only
/// path was the only producer of `Artifact.content`, so the guess is
/// always a `text/*` variant.
fn best_mime_guess(kind: &str) -> String {
    match kind {
        "html" => "text/html".into(),
        "markdown" | "md" => "text/markdown".into(),
        "code" => "text/plain".into(),
        "csv" => "text/csv".into(),
        "json" => "application/json".into(),
        _ => "text/plain".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_pool_basic() {
        let vecs = vec![vec![1.0_f32, 2.0, 3.0], vec![3.0, 2.0, 1.0]];
        let p = mean_pool(&vecs).expect("pool");
        assert_eq!(p, vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_mean_pool_empty_input() {
        assert!(mean_pool(&[] as &[Vec<f32>]).is_none());
    }

    #[test]
    fn test_mean_pool_dim_mismatch() {
        let vecs = vec![vec![1.0_f32, 2.0], vec![1.0, 2.0, 3.0]];
        assert!(mean_pool(&vecs).is_none());
    }
}
