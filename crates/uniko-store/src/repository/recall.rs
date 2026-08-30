//! Recall read path: the vector / fulltext / temporal / graph / chunk
//! search shapes the cascade issues. Backs `uniko_memory::recall`.
//!
//! These methods return decoded rows; the cascade's scoring (tier
//! weights, RRF fusion, min-max normalization, coverage) stays in
//! `uniko-memory`. Label/field *identifiers* are interpolated (Cypher
//! cannot parameterize them) — callers pass fixed identifiers from their
//! own target tables, never user input. Every *value* is bound as a
//! parameter.

use chrono::{DateTime, Utc};
use uni_db::Value;

use crate::error::Result;
use crate::storage::KnowledgeBase;
use crate::types::{NodeId, datetime_value};

/// A scored search hit: node id, primary label, a content string, and the
/// `similar_to` score. Shared by the vector, fulltext, and chunk/entity-
/// scoped shapes (all project the same `nid`/`lbl`/`content`/`score`).
#[derive(Debug, Clone)]
pub struct ScoredRow {
    pub node_id: NodeId,
    pub label: String,
    pub content: String,
    pub score: f64,
}

/// A temporal-channel hit, with a `source` discriminator (`fact` /
/// `observation` / `episode`) so the caller can score per arm.
#[derive(Debug, Clone)]
pub struct TemporalRow {
    pub node_id: NodeId,
    pub label: String,
    pub source: String,
    pub content: String,
}

/// A node's label + best-effort content, used to hydrate PPR-activated
/// node ids.
#[derive(Debug, Clone)]
pub struct NodeContentRow {
    pub node_id: NodeId,
    pub label: String,
    pub content: String,
}

/// Dimensional recall scope: restrict candidates to nodes anchored to
/// these sessions / participants / time window.
///
/// Resolved once per recall into an allow-set of node ids (see
/// [`KnowledgeBase::resolve_scope_allow_set`]); each candidate query then
/// adds `AND id(n) IN $allow`. Session-/time-less tiers (Facts, Topics) are
/// absent from the allow-set, so a dimensional scope excludes them.
#[derive(Debug, Clone, Default)]
pub struct ScopeFilter {
    /// Allowed `Session.session_id`s.
    pub sessions: Option<Vec<String>>,
    /// Allowed `Participant` ids/names (sender or subject).
    pub participants: Option<Vec<String>>,
    /// Inclusive lower time bound.
    pub since: Option<DateTime<Utc>>,
    /// Exclusive upper time bound.
    pub until: Option<DateTime<Utc>>,
}

impl ScopeFilter {
    /// Whether any dimension constrains the candidate set.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.sessions.is_some()
            || self.participants.is_some()
            || self.since.is_some()
            || self.until.is_some()
    }
}

/// Build the `AND id(<alias>) IN $allow` fragment, or empty when no
/// allow-set is threaded.
fn allow_clause(alias: &str, allow: Option<&[NodeId]>) -> String {
    if allow.is_some() {
        format!(" AND id({alias}) IN $allow")
    } else {
        String::new()
    }
}

/// The `$allow` parameter value for an allow-set, or `None` to skip binding.
fn allow_param(allow: Option<&[NodeId]>) -> Option<Value> {
    allow.map(|ids| Value::List(ids.iter().map(|&n| Value::Int(n)).collect()))
}

/// Build the SQL filter string `_vid IN (id, …)` that restricts a vector /
/// sparse procedure to a candidate node-id set.
///
/// `_vid` is the Lance vector-store id, which equals the Cypher `id(n)` /
/// [`NodeId`] (a verified 1:1 mapping), so candidate ids drop straight in.
fn vid_in_string(ids: &[NodeId]) -> String {
    let list = ids
        .iter()
        .map(|n| n.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!("_vid IN ({list})")
}

impl KnowledgeBase {
    /// Vector search over `label.embed_field`, projecting `content_field`.
    ///
    /// `label`/`embed_field`/`content_field` are trusted identifiers from
    /// the caller's fixed target table; `qvec` and `limit` are bound.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure (the caller decides whether to skip the tier).
    /// Resolve a [`ScopeFilter`] into the allow-set of in-scope node ids.
    ///
    /// Unions the node types that can anchor *every* active dimension:
    /// Messages, Observations, and Episodes for session/time; Messages and
    /// Observations for participant; Chunks for a session-only scope (via
    /// their owning Session/Message). Types with no anchor for an active
    /// dimension (Facts, Topics) are absent, so they fall out of recall.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on query
    /// failure.
    /// Run read-only Cypher through the graph query engine, decoding each
    /// row into a column→[`Value`] map.
    ///
    /// Unlike [`execute_rule`](KnowledgeBase::execute_rule) (the Locy logic
    /// runtime, which serves derived facts, not raw node `MATCH`es), this is
    /// the same Cypher engine the recall cascade uses — so `id(n) IN $allow`
    /// and other graph predicates work. Callers must ensure the query is
    /// read-only.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on query
    /// failure.
    pub async fn query_cypher(
        &self,
        cypher: &str,
        params: &std::collections::HashMap<String, Value>,
    ) -> Result<Vec<std::collections::HashMap<String, Value>>> {
        let session = self.db.session();
        let mut builder = session.query_with(cypher);
        for (k, v) in params {
            builder = builder.param(k.as_str(), v.clone());
        }
        let result = builder.fetch_all().await?;
        let mut out = Vec::with_capacity(result.rows().len());
        for row in result.rows() {
            let mut rec = std::collections::HashMap::new();
            for col in row.columns() {
                if let Some(v) = row.value(col) {
                    rec.insert(col.clone(), v.clone());
                }
            }
            out.push(rec);
        }
        Ok(out)
    }

    /// Resolve a [`ScopeFilter`] into the allow-set of in-scope node ids.
    pub async fn resolve_scope_allow_set(&self, f: &ScopeFilter) -> Result<Vec<NodeId>> {
        // An explicit empty list matches nothing — short-circuit before
        // building an `IN ()` predicate (which the SQL backend rejects).
        if f.sessions.as_ref().is_some_and(|s| s.is_empty())
            || f.participants.as_ref().is_some_and(|p| p.is_empty())
        {
            return Ok(Vec::new());
        }
        let want_session = f.sessions.is_some();
        let want_part = f.participants.is_some();
        let want_time = f.since.is_some() || f.until.is_some();

        let sess_pred = if want_session {
            " AND sc.session_id IN $sessions"
        } else {
            ""
        };
        let part_pred = if want_part {
            " AND (pc.participant_id IN $participants OR pc.name IN $participants)"
        } else {
            ""
        };
        let time_pred = |field: &str| {
            let mut s = String::new();
            if f.since.is_some() {
                s.push_str(&format!(" AND {field} >= $since"));
            }
            if f.until.is_some() {
                s.push_str(&format!(" AND {field} < $until"));
            }
            s
        };

        let mut arms: Vec<String> = Vec::new();

        // Message — anchors session, participant (sender/recipient), time.
        {
            let mut a = String::from("MATCH (m:Message)");
            if want_session {
                a.push_str("-[:IN_SESSION]->(sc:Session)");
            }
            if want_part {
                a.push_str(" MATCH (m)-[:SENT_BY|ADDRESSED_TO]->(pc:Participant)");
            }
            a.push_str(" WHERE 1=1");
            a.push_str(sess_pred);
            a.push_str(part_pred);
            a.push_str(&time_pred("m.timestamp"));
            a.push_str(" RETURN id(m) AS nid");
            arms.push(a);
        }
        // Observation — session via its message, participant via ABOUT.
        {
            let mut a = String::from("MATCH (m:Observation)");
            if want_session {
                a.push_str(" MATCH (m)-[:OBSERVED_IN]->(:Message)-[:IN_SESSION]->(sc:Session)");
            }
            if want_part {
                a.push_str(" MATCH (m)-[:ABOUT]->(pc:Participant)");
            }
            a.push_str(" WHERE 1=1");
            a.push_str(sess_pred);
            a.push_str(part_pred);
            a.push_str(&time_pred("m.temporal_anchor"));
            a.push_str(" RETURN id(m) AS nid");
            arms.push(a);
        }
        // Episode — session + time, but no participant anchor.
        if !want_part {
            let mut a = String::from("MATCH (m:Episode)");
            if want_session {
                a.push_str("-[:IN_SESSION]->(sc:Session)");
            }
            a.push_str(" WHERE 1=1");
            a.push_str(sess_pred);
            a.push_str(&time_pred("m.timestamp"));
            a.push_str(" RETURN id(m) AS nid");
            arms.push(a);
        }
        // Chunk — only under a session-only scope (no time/participant
        // anchor on a chunk); owned by an in-scope Session or Message, or
        // by an Artifact attached directly to the Session / to one of its
        // Messages.
        if want_session && !want_part && !want_time {
            arms.push(String::from(
                "MATCH (sc:Session)-[:HAS_CHUNK]->(m:Chunk) \
                 WHERE sc.session_id IN $sessions RETURN id(m) AS nid",
            ));
            arms.push(String::from(
                "MATCH (msg:Message)-[:HAS_CHUNK]->(m:Chunk) \
                 MATCH (msg)-[:IN_SESSION]->(sc:Session) \
                 WHERE sc.session_id IN $sessions RETURN id(m) AS nid",
            ));
            arms.push(String::from(
                "MATCH (a:Artifact)-[:HAS_CHUNK]->(m:Chunk) \
                 MATCH (a)-[:ATTACHED_TO]->(sc:Session) \
                 WHERE sc.session_id IN $sessions RETURN id(m) AS nid",
            ));
            arms.push(String::from(
                "MATCH (a:Artifact)-[:HAS_CHUNK]->(m:Chunk) \
                 MATCH (a)-[:ATTACHED_TO]->(:Message)-[:IN_SESSION]->(sc:Session) \
                 WHERE sc.session_id IN $sessions RETURN id(m) AS nid",
            ));
        }

        let query = arms.join(" UNION ");
        let session = self.db.session();
        let mut builder = session.query_with(&query);
        if let Some(s) = &f.sessions {
            let list: Vec<Value> = s.iter().map(|x| Value::String(x.clone())).collect();
            builder = builder.param("sessions", Value::List(list));
        }
        if let Some(p) = &f.participants {
            let list: Vec<Value> = p.iter().map(|x| Value::String(x.clone())).collect();
            builder = builder.param("participants", Value::List(list));
        }
        if let Some(since) = f.since {
            builder = builder.param("since", datetime_value(since));
        }
        if let Some(until) = f.until {
            builder = builder.param("until", datetime_value(until));
        }
        let result = builder.fetch_all().await?;
        Ok(result
            .rows()
            .iter()
            .filter_map(|row| row.get::<i64>("nid").ok())
            .collect())
    }

    pub async fn recall_vector_search(
        &self,
        label: &str,
        embed_field: &str,
        content_field: &str,
        qvec: &[f32],
        limit: i64,
        allow: Option<&[NodeId]>,
    ) -> Result<Vec<ScoredRow>> {
        let session = self.db.session();
        let cypher = format!(
            "MATCH (m:{label}) \
             WHERE m.{embed_field} IS NOT NULL{allow_and} \
             RETURN id(m) AS nid, labels(m)[0] AS lbl, \
                    coalesce(m.{content_field}, '') AS content, \
                    similar_to(m.{embed_field}, $qvec) AS score \
             ORDER BY score DESC LIMIT $lim",
            allow_and = allow_clause("m", allow),
        );
        let mut builder = session
            .query_with(&cypher)
            .param("qvec", Value::Vector(qvec.to_vec()))
            .param("lim", limit);
        if let Some(v) = allow_param(allow) {
            builder = builder.param("allow", v);
        }
        let result = builder.fetch_all().await?;
        Ok(decode_scored_rows(result.rows()))
    }

    /// BM25 fulltext search over `label.content_field`.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure.
    pub async fn recall_fulltext_search(
        &self,
        label: &str,
        content_field: &str,
        qtxt: &str,
        limit: i64,
        allow: Option<&[NodeId]>,
    ) -> Result<Vec<ScoredRow>> {
        let session = self.db.session();
        let where_clause = if allow.is_some() {
            "WHERE id(m) IN $allow "
        } else {
            ""
        };
        let cypher = format!(
            "MATCH (m:{label}) \
             {where_clause}RETURN id(m) AS nid, labels(m)[0] AS lbl, \
                    coalesce(m.{content_field}, '') AS content, \
                    similar_to(m.{content_field}, $qtxt) AS score \
             ORDER BY score DESC LIMIT $lim"
        );
        let mut builder = session
            .query_with(&cypher)
            .param("qtxt", qtxt)
            .param("lim", limit);
        if let Some(v) = allow_param(allow) {
            builder = builder.param("allow", v);
        }
        let result = builder.fetch_all().await?;
        Ok(decode_scored_rows(result.rows()))
    }

    /// Learned-sparse search over `label.sparse_field` via `uni.sparse.query`.
    ///
    /// `qtxt` is auto-embedded into a sparse query vector by the hybrid embed
    /// alias, so no precomputed sparse vector is needed. Scores are sparse
    /// dot products (higher = more similar). When `allow` is set, the
    /// candidate set is restricted at the index level via the procedure's
    /// `_vid IN (…)` SQL filter.
    ///
    /// `label`/`sparse_field`/`content_field` are trusted identifiers from
    /// the caller's fixed target table; `qtxt` and `limit` are bound.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on query
    /// failure.
    pub async fn recall_sparse_search(
        &self,
        label: &str,
        sparse_field: &str,
        content_field: &str,
        qtxt: &str,
        limit: i64,
        allow: Option<&[NodeId]>,
    ) -> Result<Vec<ScoredRow>> {
        let session = self.db.session();
        // The proc's 5th arg is a SQL filter; pass `_vid IN (…)` for an
        // allow-set (applied during retrieval), else the literal `null`.
        let filter_allow = allow.filter(|ids| !ids.is_empty());
        let filter_arg = if filter_allow.is_some() {
            "$filter"
        } else {
            "null"
        };
        let cypher = format!(
            "CALL uni.sparse.query('{label}', '{sparse_field}', $qtxt, $lim, {filter_arg}, null, {{}}) \
             YIELD node, score \
             RETURN id(node) AS nid, labels(node)[0] AS lbl, \
                    coalesce(node.{content_field}, '') AS content, score"
        );
        let mut builder = session
            .query_with(&cypher)
            .param("qtxt", qtxt)
            .param("lim", limit);
        if let Some(ids) = filter_allow {
            builder = builder.param("filter", vid_in_string(ids));
        }
        let result = builder.fetch_all().await?;
        Ok(decode_scored_rows(result.rows()))
    }

    /// ColBERT MaxSim re-scoring of a candidate node-id set.
    ///
    /// Runs `uni.vector.query` over the multi-vector `colbert_field`,
    /// restricting the exact-MaxSim scan to `node_ids` via the `_vid IN (…)`
    /// filter — so it re-scores exactly the supplied candidates rather than
    /// retrieving globally. `query_multivector` is the query's per-token
    /// embedding; scores are MaxSim (`Σ_q max_d cos`). Returns an empty vec
    /// when either input is empty.
    ///
    /// `label`/`colbert_field`/`content_field` are trusted identifiers.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on query
    /// failure.
    pub async fn recall_colbert_maxsim(
        &self,
        label: &str,
        colbert_field: &str,
        content_field: &str,
        query_multivector: &[Vec<f32>],
        node_ids: &[NodeId],
    ) -> Result<Vec<ScoredRow>> {
        if node_ids.is_empty() || query_multivector.is_empty() {
            return Ok(Vec::new());
        }
        let session = self.db.session();
        let cypher = format!(
            "CALL uni.vector.query('{label}', '{colbert_field}', $q, $k, $filter, null, {{}}) \
             YIELD node, score \
             RETURN id(node) AS nid, labels(node)[0] AS lbl, \
                    coalesce(node.{content_field}, '') AS content, score"
        );
        let qval = Value::List(
            query_multivector
                .iter()
                .map(|t| Value::Vector(t.clone()))
                .collect(),
        );
        let result = session
            .query_with(&cypher)
            .param("q", qval)
            .param("k", node_ids.len() as i64)
            .param("filter", vid_in_string(node_ids))
            .fetch_all()
            .await?;
        Ok(decode_scored_rows(result.rows()))
    }

    /// Resolve entity/participant `names` to graph seed node ids for
    /// spreading activation. Deduped; infallible (logs and returns empty
    /// on query error, matching the cascade's best-effort contract).
    #[must_use]
    pub async fn resolve_entity_seeds(&self, names: &[String]) -> Vec<NodeId> {
        if names.is_empty() {
            return Vec::new();
        }
        let session = self.db.session();
        let cypher = "UNWIND $names AS name \
                      MATCH (n) \
                      WHERE (n:Entity AND n.name = name) \
                         OR (n:Participant AND n.name = name) \
                      RETURN DISTINCT id(n) AS nid";
        let names_param: Vec<Value> = names.iter().map(|n| Value::String(n.clone())).collect();
        match session
            .query_with(cypher)
            .param("names", Value::List(names_param))
            .fetch_all()
            .await
        {
            Ok(r) => r
                .rows()
                .iter()
                .filter_map(|row| row.get::<i64>("nid").ok())
                .collect(),
            Err(e) => {
                tracing::debug!(error = %e, "resolve_entity_seeds query failed");
                Vec::new()
            }
        }
    }

    /// Whether any of `names` resolves to an `:Entity` flagged
    /// `unstable` (F39 drift). Backs the F58 recall drift override.
    ///
    /// Best-effort: logs and returns `false` on query error or when
    /// `names` is empty, matching the cascade's best-effort contract.
    #[must_use]
    pub async fn any_unstable_entities(&self, names: &[String]) -> bool {
        if names.is_empty() {
            return false;
        }
        let session = self.db.session();
        let cypher = "UNWIND $names AS name \
                      MATCH (e:Entity) \
                      WHERE e.name = name AND e.unstable = true \
                      RETURN count(e) AS k";
        let names_param: Vec<Value> = names.iter().map(|n| Value::String(n.clone())).collect();
        match session
            .query_with(cypher)
            .param("names", Value::List(names_param))
            .fetch_all()
            .await
        {
            Ok(r) => {
                r.rows()
                    .first()
                    .and_then(|row| row.get::<i64>("k").ok())
                    .unwrap_or(0)
                    > 0
            }
            Err(e) => {
                tracing::debug!(error = %e, "any_unstable_entities query failed");
                false
            }
        }
    }

    /// Temporal-channel fan-out over `[lo, hi)`: Fact via BTIC overlap,
    /// Observation + Episode via BTree range, with per-arm limits.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure.
    pub async fn temporal_window_hits(
        &self,
        lo: DateTime<Utc>,
        hi: DateTime<Utc>,
        lim_fact: i64,
        lim_obs: i64,
        lim_eps: i64,
        allow: Option<&[NodeId]>,
    ) -> Result<Vec<TemporalRow>> {
        let qbtic = temporal_window_to_btic_value(lo, hi);
        let session = self.db.session();
        let cypher = format!(
            "MATCH (f:Fact) \
                      WHERE f.valid_at IS NOT NULL \
                        AND btic_overlaps(f.valid_at, $qbtic){af} \
                      WITH f LIMIT $lim_fact \
                      RETURN id(f) AS nid, labels(f)[0] AS lbl, 'fact' AS source, \
                             (coalesce(f.subject, '') + ' ' \
                              + coalesce(f.predicate, '') + ' ' \
                              + coalesce(f.object, '')) AS content \
                      UNION ALL \
                      MATCH (o:Observation) \
                      WHERE o.temporal_anchor >= $lo \
                        AND o.temporal_anchor < $hi{ao} \
                      WITH o LIMIT $lim_obs \
                      RETURN id(o) AS nid, labels(o)[0] AS lbl, 'observation' AS source, \
                             coalesce(o.content, '') AS content \
                      UNION ALL \
                      MATCH (e:Episode) \
                      WHERE e.timestamp >= $lo \
                        AND e.timestamp < $hi{ae} \
                      WITH e LIMIT $lim_eps \
                      RETURN id(e) AS nid, labels(e)[0] AS lbl, 'episode' AS source, \
                             coalesce(e.action_type, '') AS content",
            af = allow_clause("f", allow),
            ao = allow_clause("o", allow),
            ae = allow_clause("e", allow),
        );
        let mut builder = session
            .query_with(&cypher)
            .param("qbtic", qbtic)
            .param("lo", datetime_value(lo))
            .param("hi", datetime_value(hi))
            .param("lim_fact", lim_fact)
            .param("lim_obs", lim_obs)
            .param("lim_eps", lim_eps);
        if let Some(v) = allow_param(allow) {
            builder = builder.param("allow", v);
        }
        let result = builder.fetch_all().await?;
        let mut out = Vec::with_capacity(result.rows().len());
        for row in result.rows() {
            let Ok(nid) = row.get::<i64>("nid") else {
                continue;
            };
            let source: String = row.get("source").unwrap_or_default();
            let default_label = match source.as_str() {
                "fact" => "Fact",
                "observation" => "Observation",
                "episode" => "Episode",
                _ => "Unknown",
            };
            let label: String = row.get("lbl").unwrap_or_else(|_| default_label.into());
            let content: String = row.get("content").unwrap_or_default();
            out.push(TemporalRow {
                node_id: nid,
                label,
                source,
                content,
            });
        }
        Ok(out)
    }

    /// Hydrate PPR-activated node ids with their label + best-effort
    /// content (one round-trip). `content` falls through the common text-
    /// bearing fields across label types.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure.
    pub async fn fetch_node_contents(&self, nids: &[NodeId]) -> Result<Vec<NodeContentRow>> {
        let session = self.db.session();
        let cypher = "UNWIND $nids AS nid \
                      MATCH (n) WHERE id(n) = nid \
                      RETURN nid, labels(n)[0] AS lbl, \
                             coalesce(n.content, \
                                      n.action_type, \
                                      (coalesce(n.subject, '') + ' ' \
                                       + coalesce(n.predicate, '') + ' ' \
                                       + coalesce(n.object, '')), \
                                      n.name, \
                                      '') AS content";
        let nids_param: Vec<Value> = nids.iter().map(|&v| Value::Int(v)).collect();
        let result = session
            .query_with(cypher)
            .param("nids", Value::List(nids_param))
            .fetch_all()
            .await?;
        let mut out = Vec::with_capacity(result.rows().len());
        for row in result.rows() {
            let Ok(nid) = row.get::<i64>("nid") else {
                continue;
            };
            out.push(NodeContentRow {
                node_id: nid,
                label: row.get("lbl").unwrap_or_default(),
                content: row.get("content").unwrap_or_default(),
            });
        }
        Ok(out)
    }

    /// Distinct session-Chunk node ids reachable from Fact `fact_node_id`
    /// via `SUPPORTED_BY → OBSERVED_IN → IN_SESSION → HAS_CHUNK`.
    ///
    /// The node id is **bound** as a parameter (issue #2: this walk
    /// previously interpolated the id directly into the Cypher string).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure.
    pub async fn fact_session_chunk_ids(&self, fact_node_id: NodeId) -> Result<Vec<NodeId>> {
        let session = self.db.session();
        // `SUPPORTED_BY` is registered Fact → Observation
        // (`schema/facts.rs`), so it is traversed outbound from the Fact.
        // This previously read `(f)<-[:SUPPORTED_BY]-(:Observation)`, which
        // can never match and silently returned no chunks — leaving the
        // Phase 1 session boost with an empty boost map on every call.
        let cypher = "MATCH (f) WHERE id(f) = $nid \
                      MATCH (f)-[:SUPPORTED_BY]->(:Observation)-[:OBSERVED_IN]->(:Message) \
                           -[:IN_SESSION]->(:Session)-[:HAS_CHUNK]->(c:Chunk) \
                      RETURN DISTINCT id(c) AS chunk_id";
        let result = session
            .query_with(cypher)
            .param("nid", fact_node_id)
            .fetch_all()
            .await?;
        Ok(result
            .rows()
            .iter()
            .filter_map(|row| row.get::<i64>("chunk_id").ok())
            .collect())
    }

    /// Does any Entity reachable from `node_id` (via `MENTIONS`/`ABOUT`)
    /// carry `entity_type = target_type` — or is the node itself such an
    /// Entity? Both values are bound as parameters.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure (the caller treats errors as "no match").
    pub async fn entity_type_matches(&self, node_id: NodeId, target_type: &str) -> Result<bool> {
        let session = self.db.session();
        let q = "MATCH (n) WHERE id(n) = $nid \
                 OPTIONAL MATCH (n)-[:MENTIONS|ABOUT]->(e1:Entity {entity_type: $t}) \
                 OPTIONAL MATCH (n)<-[:ABOUT]-(:Observation)-[:ABOUT]->(e2:Entity {entity_type: $t}) \
                 RETURN (n.entity_type = $t OR e1 IS NOT NULL OR e2 IS NOT NULL) AS hit \
                 LIMIT 1";
        let result = session
            .query_with(q)
            .param("nid", node_id)
            .param("t", target_type)
            .fetch_all()
            .await?;
        Ok(result
            .rows()
            .first()
            .and_then(|row| row.get::<bool>("hit").ok())
            .unwrap_or(false))
    }

    /// Per-variant chunk hybrid (vector+BM25 over `session`/`observation`
    /// chunks and Artifact-owned chunks) plus the entity-scoped fan-out,
    /// max-merged per node id.
    ///
    /// `session` and `observation` each receive `per_variant_limit` candidates;
    /// the Artifact-owned arm receives twice that limit, preserving the former
    /// combined `block` + `page` budget without enumerating Artifact types.
    ///
    /// Infallible by contract: individual sub-queries that fail are logged
    /// at debug and skipped (the cascade is best-effort). The structural
    /// `similar_to` source/query/fusion lists are interpolated; all values
    /// (`chunk_type`, `ename`, vectors, text, limit) are bound.
    ///
    /// Returns the merged hits including empty-content rows — the caller
    /// applies its own empty-content filter and ranking.
    #[must_use]
    #[expect(clippy::too_many_arguments, reason = "recall channel knobs + scope")]
    pub async fn recall_chunk_and_entity_scoped(
        &self,
        qvec: &[f32],
        qtxt: &str,
        entity_refs: &[String],
        per_variant_limit: i64,
        vector_weight: f64,
        bm25_weight: f64,
        allow: Option<&[NodeId]>,
    ) -> Vec<ScoredRow> {
        use std::collections::HashMap;

        let allow_and = allow_clause("m", allow);

        let session = self.db.session();
        let mut scored: HashMap<NodeId, (String, String, f64)> = HashMap::new();
        let has_vec = !qvec.is_empty();
        let t_start = std::time::Instant::now();

        // Hybrid (vector + BM25) over conversational chunks plus every chunk
        // owned by an Artifact. Artifact chunk types are intentionally open:
        // text, heading, PDF, code, structured, and future chunkers all flow
        // through the HAS_CHUNK ownership boundary.
        let (sources, queries, fusion, needs_qvec, needs_qtxt) = if has_vec {
            (
                "[m.embedding, m.text]",
                "[$qvec, $qtxt]",
                format!(", {{method: 'weighted', weights: [{vector_weight}, {bm25_weight}]}}"),
                true,
                true,
            )
        } else {
            ("m.text", "$qtxt", String::new(), false, true)
        };

        // Preserve the historical per-type conversational budgets: a dense
        // Artifact must not consume the slots reserved for session or
        // observation chunks before RRF fusion.
        for chunk_type in ["session", "observation"] {
            let cypher = format!(
                "MATCH (m:Chunk) WHERE m.chunk_type = $ctype{allow_and} \
                 RETURN id(m) AS nid, labels(m)[0] AS lbl, \
                        m.text AS content, \
                        similar_to({sources}, {queries}{fusion}) AS score \
                 ORDER BY score DESC LIMIT $lim"
            );
            let mut builder = session
                .query_with(&cypher)
                .param("ctype", chunk_type)
                .param("lim", per_variant_limit);
            if let Some(v) = allow_param(allow) {
                builder = builder.param("allow", v);
            }
            if needs_qvec {
                builder = builder.param("qvec", Value::Vector(qvec.to_vec()));
            }
            if needs_qtxt {
                builder = builder.param("qtxt", qtxt);
            }
            match builder.fetch_all().await {
                Ok(result) => merge_scored_rows(result.rows(), &mut scored),
                Err(e) => tracing::debug!(chunk_type, error = %e, "hybrid similar_to failed"),
            }
        }

        // The former block/page arms each had their own limit. Artifact
        // ownership replaces both closed type checks, so retain their combined
        // 2 × limit as one open-ended Artifact candidate budget.
        let artifact_limit = per_variant_limit.saturating_mul(2);
        let cypher = format!(
            "MATCH (m:Chunk) \
             WHERE (m)<-[:HAS_CHUNK]-(:Artifact){allow_and} \
             RETURN id(m) AS nid, labels(m)[0] AS lbl, \
                    m.text AS content, \
                    similar_to({sources}, {queries}{fusion}) AS score \
             ORDER BY score DESC LIMIT $lim"
        );
        let mut builder = session.query_with(&cypher).param("lim", artifact_limit);
        if let Some(v) = allow_param(allow) {
            builder = builder.param("allow", v);
        }
        if needs_qvec {
            builder = builder.param("qvec", Value::Vector(qvec.to_vec()));
        }
        if needs_qtxt {
            builder = builder.param("qtxt", qtxt);
        }
        match builder.fetch_all().await {
            Ok(result) => merge_scored_rows(result.rows(), &mut scored),
            Err(e) => tracing::debug!(error = %e, "Artifact hybrid similar_to failed"),
        }
        let chunk_ms = t_start.elapsed().as_millis() as u64;

        // Entity-scoped fan-out. The cypher per target depends only on
        // `has_vec` + the per-target tuple, not on `entity_name`, so build
        // the 6 templates once and bind $ename per entity.
        let entity_scoped_targets: &[(&str, &str, &str, &str)] = &[
            (
                "Chunk",
                "text",
                "embedding",
                "(m)<-[:HAS_CHUNK]-(:Session)<-[:PARTICIPATED_IN]-(:Participant {name: $ename})",
            ),
            (
                "Chunk",
                "text",
                "embedding",
                "(m)-[:ABOUT]->(:Participant {name: $ename})",
            ),
            (
                "Chunk",
                "text",
                "embedding",
                "(m)-[:ABOUT]->(:Entity {name: $ename})",
            ),
            (
                "Observation",
                "content",
                "embedding",
                "(m)-[:ABOUT]->(:Participant {name: $ename})",
            ),
            (
                "Observation",
                "content",
                "embedding",
                "(m)-[:ABOUT]->(:Entity {name: $ename})",
            ),
            ("Observation", "content", "embedding", "m.subject = $ename"),
        ];
        let entity_cyphers: Vec<(String, &str)> = entity_scoped_targets
            .iter()
            .map(|&(label, content_field, embed_field, pattern)| {
                let cypher = if has_vec {
                    format!(
                        "MATCH (m:{label}) WHERE {pattern}{allow_and} \
                         RETURN id(m) AS nid, labels(m)[0] AS lbl, \
                                m.{content_field} AS content, \
                                similar_to([m.{embed_field}, m.{content_field}], [$qvec, $qtxt]) AS score \
                         ORDER BY score DESC LIMIT $lim"
                    )
                } else {
                    format!(
                        "MATCH (m:{label}) WHERE {pattern}{allow_and} \
                         RETURN id(m) AS nid, labels(m)[0] AS lbl, \
                                m.{content_field} AS content, \
                                similar_to(m.{content_field}, $qtxt) AS score \
                         ORDER BY score DESC LIMIT $lim"
                    )
                };
                (cypher, label)
            })
            .collect();

        for entity_name in entity_refs {
            for (cypher, label) in &entity_cyphers {
                let mut builder = session.query_with(cypher);
                builder = builder
                    .param("ename", entity_name.as_str())
                    .param("qtxt", qtxt)
                    .param("lim", per_variant_limit);
                if let Some(v) = allow_param(allow) {
                    builder = builder.param("allow", v);
                }
                if has_vec {
                    builder = builder.param("qvec", Value::Vector(qvec.to_vec()));
                }
                match builder.fetch_all().await {
                    Ok(result) => merge_scored_rows(result.rows(), &mut scored),
                    Err(e) => tracing::debug!(
                        entity = entity_name,
                        label,
                        error = %e,
                        "entity-scoped search failed"
                    ),
                }
            }
        }
        eprintln!(
            "RECALL_PROF chunk_and_entity_scoped entities={} chunk_search_ms={} entity_scoped_ms={} total_ms={}",
            entity_refs.len(),
            chunk_ms,
            (t_start.elapsed().as_millis() as u64).saturating_sub(chunk_ms),
            t_start.elapsed().as_millis() as u64
        );

        scored
            .into_iter()
            .map(|(node_id, (label, content, score))| ScoredRow {
                node_id,
                label,
                content,
                score,
            })
            .collect()
    }
}

/// Decode a batch of `nid`/`lbl`/`content`/`score` rows into
/// [`ScoredRow`]s. Rows with an unreadable `nid` are skipped.
fn decode_scored_rows(rows: &[uni_db::Row]) -> Vec<ScoredRow> {
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let Ok(nid) = row.get::<i64>("nid") else {
            continue;
        };
        out.push(ScoredRow {
            node_id: nid,
            label: row.get("lbl").unwrap_or_default(),
            content: row.get("content").unwrap_or_default(),
            score: row.get("score").unwrap_or(0.0),
        });
    }
    out
}

/// Merge a row batch into the per-variant `scored` accumulator, keeping
/// the max score per node id (first label/content wins).
fn merge_scored_rows(
    rows: &[uni_db::Row],
    scored: &mut std::collections::HashMap<NodeId, (String, String, f64)>,
) {
    for row in rows {
        let nid: i64 = row.get("nid").unwrap_or(0);
        let lbl: String = row.get("lbl").unwrap_or_default();
        let content: String = row.get("content").unwrap_or_default();
        let score: f64 = row.get("score").unwrap_or(0.0);
        scored
            .entry(nid)
            .and_modify(|(_, _, s)| *s = s.max(score))
            .or_insert((lbl, content, score));
    }
}

/// Build a `Value::Temporal(Btic { ... })` covering `[lo, hi)` for the
/// `btic_overlaps()` UDF.
fn temporal_window_to_btic_value(lo: DateTime<Utc>, hi: DateTime<Utc>) -> Value {
    let btic = crate::schema::btic::btic_query_window(lo, hi);
    Value::Temporal(uni_db::common::TemporalValue::Btic {
        lo: btic.lo(),
        hi: btic.hi(),
        meta: btic.meta(),
    })
}
