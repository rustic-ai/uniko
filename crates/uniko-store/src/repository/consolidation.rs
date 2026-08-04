//! Consolidation reads: pull unprocessed Observations carrying a
//! structured triple. Backs `uniko_memory::consolidation`.

use std::collections::HashMap;

use chrono::{DateTime, Utc};

use crate::error::Result;
use crate::storage::KnowledgeBase;
use crate::types::{NodeId, optional_datetime_from_row};

/// An Observation awaiting consolidation into a Fact.
///
/// Carries the structured triple (`subject`/`predicate`/`object`) plus
/// the raw `content` and the temporal columns the consolidator groups by.
#[derive(Debug, Clone)]
pub struct UnprocessedObs {
    /// uni-db node id of the Observation.
    pub node_id: NodeId,
    /// Triple subject (required — rows without it are skipped).
    pub subject: String,
    /// Triple predicate (required — rows without it are skipped).
    pub predicate: String,
    /// Triple object, when present.
    pub object: Option<String>,
    /// Raw observation text.
    pub content: String,
    /// When the observation was recorded (defaults to now if unreadable).
    pub observed_at: DateTime<Utc>,
    /// Resolved absolute date from `ARGM-TMP` (Phase A). Absent when the
    /// source observation had no temporal modifier or used the rule-based
    /// fallback path.
    pub temporal_anchor: Option<DateTime<Utc>>,
}

impl KnowledgeBase {
    /// Pull up to `limit` unprocessed Observations carrying a structured
    /// triple, oldest first.
    ///
    /// Filters out rule-fallback Observations (`predicate IS NULL`) and
    /// any already wired to a prior `ConsolidationCycle` via `PROCESSED`.
    /// Rows missing a required column (`nid`/`subject`/`predicate`) are
    /// silently skipped.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) when
    /// the query fails.
    pub async fn fetch_unprocessed_observations(&self, limit: i64) -> Result<Vec<UnprocessedObs>> {
        let session = self.db.session();
        let cypher = "MATCH (o:Observation) \
                      WHERE o.subject IS NOT NULL AND o.predicate IS NOT NULL \
                      AND NOT EXISTS { MATCH (:ConsolidationCycle)-[:PROCESSED]->(o) } \
                      RETURN id(o) AS nid, o.subject AS subject, o.predicate AS predicate, \
                             o.object AS object, o.content AS content, \
                             o.observed_at AS observed_at, \
                             o.temporal_anchor AS temporal_anchor \
                      ORDER BY o.observed_at ASC \
                      LIMIT $lim";
        let result = session
            .query_with(cypher)
            .param("lim", limit)
            .fetch_all()
            .await?;

        Ok(result
            .rows()
            .iter()
            .filter_map(decode_unprocessed_obs)
            .collect())
    }

    /// Batched read of `Observation.embedding` for the given node ids.
    ///
    /// Observations are auto-embedded on `content` at ingest. An id whose
    /// `embedding` column is null or does not decode as a float vector is
    /// omitted from the map rather than reported — callers treat absence as
    /// "no embedding" and fall back to a default weight. Read-only: runs on
    /// a plain session, no transaction needed.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) when the
    /// query itself fails.
    pub async fn fetch_observation_embeddings(
        &self,
        ids: &[NodeId],
    ) -> Result<HashMap<NodeId, Vec<f32>>> {
        let mut out: HashMap<NodeId, Vec<f32>> = HashMap::new();
        if ids.is_empty() {
            return Ok(out);
        }
        let id_list = ids
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        // UNWIND over an inline id-literal list mirrors the proven
        // `fetch_entities_for_upsert_in_tx` pattern (ids are i64, injection-safe).
        let cypher = format!(
            "UNWIND [{id_list}] AS oid \
             MATCH (o:Observation) WHERE id(o) = oid \
             RETURN id(o) AS id, o.embedding AS emb"
        );
        let result = self.db.session().query_with(&cypher).fetch_all().await?;
        for row in result.rows() {
            if let (Ok(id), Ok(emb)) = (row.get::<i64>("id"), row.get::<Vec<f32>>("emb")) {
                out.insert(id, emb);
            }
        }
        Ok(out)
    }
}

/// Decode one Observation row into [`UnprocessedObs`]. Returns `None`
/// when any required column (`nid`/`subject`/`predicate`) is missing or
/// malformed — those rows are silently skipped, matching the legacy
/// per-field guards.
fn decode_unprocessed_obs(row: &uni_db::Row) -> Option<UnprocessedObs> {
    let nid = row.get::<i64>("nid").ok()?;
    let subject = row.get::<String>("subject").ok()?;
    let predicate = row.get::<String>("predicate").ok()?;
    // uni-db 2.1.0 returns DateTime props as `Value::Temporal`, not as
    // RFC-3339 strings (uni `4583ee870`). `optional_datetime_from_row`
    // reads the `Value::Temporal` form and still accepts a legacy string.
    let observed_at = optional_datetime_from_row(row, "observed_at").unwrap_or_else(Utc::now);
    let temporal_anchor = optional_datetime_from_row(row, "temporal_anchor");
    Some(UnprocessedObs {
        node_id: nid,
        subject,
        predicate,
        object: row.get::<String>("object").ok(),
        content: row.get::<String>("content").unwrap_or_default(),
        observed_at,
        temporal_anchor,
    })
}
