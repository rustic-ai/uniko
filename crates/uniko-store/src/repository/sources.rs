//! Reads behind the deterministic read tools (`Sources` in uniko-memory):
//! lexical chunk search with each hit's owner, chunk ranges by parent, and
//! tool-call (Action) lookups.
//!
//! Nothing here applies visibility; the facade admits every returned node
//! against its viewer before handing anything out.

use std::collections::HashMap;

use uni_db::Value;

use crate::error::Result;
use crate::storage::KnowledgeBase;
use crate::types::NodeId;

/// One chunk with where it came from.
#[derive(Debug, Clone, Default)]
pub struct ChunkRow {
    /// Chunk node id.
    pub node_id: NodeId,
    /// `Chunk.chunk_id`.
    pub chunk_id: String,
    /// Position within its parent.
    pub index: i64,
    /// Verbatim text.
    pub text: String,
    /// Byte span in the parent's source.
    pub start: i64,
    /// Byte span end.
    pub end: i64,
    /// Context header, for a structure-aware chunk.
    pub context: Option<String>,
    /// Address in the source, for a structure-aware chunk.
    pub address: Option<String>,
    /// Lexical score, for a search hit; 0 otherwise.
    pub score: f64,
    /// Owning node's label: `Artifact`, `Message`, `Session` or `Block`.
    pub owner_label: String,
    /// Owner's external id (for a Block: its Artifact's id).
    pub owner_id: String,
    /// The session the chunk belongs to, when known.
    pub session_id: Option<String>,
    /// The tool call that produced the owning artifact, when one did.
    pub call_id: Option<String>,
    /// That call's tool.
    pub tool_name: Option<String>,
}

/// One recorded Action, as the read tools show it.
#[derive(Debug, Clone, Default)]
pub struct ActionRow {
    /// Action node id.
    pub node_id: NodeId,
    /// `Action.action_id`.
    pub action_id: String,
    /// `Action.action_type`.
    pub action_type: String,
    /// `Action.tool_name`.
    pub tool_name: Option<String>,
    /// `Action.input`.
    pub input: Option<Value>,
    /// `Action.status`.
    pub status: Option<String>,
    /// `Action.error`.
    pub error: Option<String>,
    /// `Action.started_at`.
    pub started_at: Option<Value>,
    /// `Action.ended_at`.
    pub ended_at: Option<Value>,
    /// Node id and external id of the triggering Message.
    pub triggered_by: Option<(NodeId, String)>,
    /// External id of its Session.
    pub session_id: Option<String>,
    /// Node id and external id of the result Artifact.
    pub artifact: Option<(NodeId, String)>,
}

fn s(row: &uni_db::Row, k: &str) -> Option<String> {
    row.get::<String>(k).ok()
}

fn i(row: &uni_db::Row, k: &str) -> Option<i64> {
    row.get::<i64>(k).ok()
}

fn ids_value(ids: &[NodeId]) -> Value {
    Value::List(ids.iter().map(|n| Value::Int(*n)).collect())
}

impl KnowledgeBase {
    /// Lexical search over `Chunk.search_text` with each hit's owner,
    /// session and producing tool call. `lex` is normalised search text.
    /// Ordered by score, then `chunk_id`, so equal scores come back in a
    /// stable order.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on query
    /// failure.
    pub async fn search_chunks_lexical(&self, lex: &str, limit: i64) -> Result<Vec<ChunkRow>> {
        let session = self.db.session();
        let rows = session
            .query_with(
                "MATCH (c:Chunk) WHERE c.search_text IS NOT NULL \
                 RETURN id(c) AS nid, c.chunk_id AS cid, c.index AS idx, c.text AS text, \
                        c.start AS s, c.end AS e, c.context AS ctx, c.address AS addr, \
                        similar_to(c.search_text, $lex) AS score \
                 ORDER BY score DESC LIMIT $lim",
            )
            .param("lex", lex)
            .param("lim", limit)
            .fetch_all()
            .await?;
        let mut out: Vec<ChunkRow> = rows
            .rows()
            .iter()
            .filter_map(|r| {
                Some(ChunkRow {
                    node_id: i(r, "nid")?,
                    chunk_id: s(r, "cid").unwrap_or_default(),
                    index: i(r, "idx").unwrap_or(0),
                    text: s(r, "text").unwrap_or_default(),
                    start: i(r, "s").unwrap_or(0),
                    end: i(r, "e").unwrap_or(0),
                    context: s(r, "ctx"),
                    address: s(r, "addr"),
                    score: r.get::<f64>("score").unwrap_or(0.0),
                    ..ChunkRow::default()
                })
            })
            .filter(|c| c.score > 0.0)
            .collect();
        self.resolve_chunk_owners(&mut out).await?;
        out.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.chunk_id.cmp(&b.chunk_id))
        });
        Ok(out)
    }

    /// Chunks of one parent with `index` in `[from, to]`, in order. The
    /// parent is named by label and external id (`Artifact`/`artifact_id`,
    /// `Message`/`message_id`, `Session`/`session_id`).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on query
    /// failure, or [`UnikoError::Config`](crate::UnikoError::Config) for an
    /// unknown parent label.
    pub async fn chunk_range(
        &self,
        parent_label: &str,
        parent_id: &str,
        from: i64,
        to: i64,
    ) -> Result<Vec<ChunkRow>> {
        let id_field = match parent_label {
            "Artifact" => "artifact_id",
            "Message" => "message_id",
            "Session" => "session_id",
            other => {
                return Err(crate::UnikoError::Config(format!(
                    "chunk_range: no chunk parent label '{other}'"
                )));
            }
        };
        let cypher = format!(
            "MATCH (p:{parent_label} {{{id_field}: $pid}})-[:HAS_CHUNK]->(c:Chunk) \
             WHERE c.index >= $from AND c.index <= $to \
             RETURN id(c) AS nid, c.chunk_id AS cid, c.index AS idx, c.text AS text, \
                    c.start AS s, c.end AS e, c.context AS ctx, c.address AS addr \
             ORDER BY c.index"
        );
        let rows = self
            .db
            .session()
            .query_with(&cypher)
            .param("pid", parent_id)
            .param("from", from)
            .param("to", to)
            .fetch_all()
            .await?;
        let mut out: Vec<ChunkRow> = rows
            .rows()
            .iter()
            .filter_map(|r| {
                Some(ChunkRow {
                    node_id: i(r, "nid")?,
                    chunk_id: s(r, "cid").unwrap_or_default(),
                    index: i(r, "idx").unwrap_or(0),
                    text: s(r, "text").unwrap_or_default(),
                    start: i(r, "s").unwrap_or(0),
                    end: i(r, "e").unwrap_or(0),
                    context: s(r, "ctx"),
                    address: s(r, "addr"),
                    ..ChunkRow::default()
                })
            })
            .collect();
        self.resolve_chunk_owners(&mut out).await?;
        Ok(out)
    }

    /// Fill each row's owner, session and producing call.
    async fn resolve_chunk_owners(&self, rows: &mut [ChunkRow]) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let ids: Vec<NodeId> = rows.iter().map(|r| r.node_id).collect();
        let session = self.db.session();
        let mut owner: HashMap<NodeId, (String, String, Option<String>)> = HashMap::new();
        for (label, cypher) in [
            (
                "Message",
                "MATCH (m:Message)-[:HAS_CHUNK]->(c:Chunk) WHERE id(c) IN $ids \
                 OPTIONAL MATCH (m)-[:IN_SESSION]->(s:Session) \
                 RETURN id(c) AS nid, m.message_id AS oid, s.session_id AS sid",
            ),
            (
                "Session",
                "MATCH (s:Session)-[:HAS_CHUNK]->(c:Chunk) WHERE id(c) IN $ids \
                 RETURN id(c) AS nid, s.session_id AS oid, s.session_id AS sid",
            ),
            (
                "Artifact",
                "MATCH (a:Artifact)-[:HAS_CHUNK]->(c:Chunk) WHERE id(c) IN $ids \
                 OPTIONAL MATCH (a)-[:ATTACHED_TO]->(s:Session) \
                 RETURN id(c) AS nid, a.artifact_id AS oid, s.session_id AS sid",
            ),
        ] {
            let res = session
                .query_with(cypher)
                .param("ids", ids_value(&ids))
                .fetch_all()
                .await?;
            for r in res.rows() {
                if let (Some(nid), Some(oid)) = (i(r, "nid"), s(r, "oid")) {
                    // An Artifact owner wins over a Block's: tiered PDF
                    // chunks hang off both.
                    owner.insert(nid, (label.to_string(), oid, s(r, "sid")));
                }
            }
        }
        let artifact_ids: Vec<Value> = owner
            .values()
            .filter(|(l, _, _)| l == "Artifact")
            .map(|(_, id, _)| Value::String(id.clone()))
            .collect();
        let mut produced: HashMap<String, (String, Option<String>, Option<String>)> =
            HashMap::new();
        if !artifact_ids.is_empty() {
            let res = session
                .query_with(
                    "MATCH (a:Action)-[:PRODUCED]->(art:Artifact) WHERE art.artifact_id IN $aids \
                     OPTIONAL MATCH (a)-[:IN_SESSION]->(s:Session) \
                     RETURN art.artifact_id AS aid, a.action_id AS cid, a.tool_name AS tool, \
                            s.session_id AS sid",
                )
                .param("aids", Value::List(artifact_ids))
                .fetch_all()
                .await?;
            for r in res.rows() {
                if let (Some(aid), Some(cid)) = (s(r, "aid"), s(r, "cid")) {
                    produced.insert(aid, (cid, s(r, "tool"), s(r, "sid")));
                }
            }
        }
        for row in rows.iter_mut() {
            if let Some((label, oid, sid)) = owner.remove(&row.node_id) {
                if let Some((cid, tool, asid)) = produced.get(&oid) {
                    row.call_id = Some(cid.clone());
                    row.tool_name = tool.clone();
                    row.session_id = asid.clone().or(sid);
                } else {
                    row.session_id = sid;
                }
                row.owner_label = label;
                row.owner_id = oid;
            }
        }
        Ok(())
    }

    /// Recorded Actions, newest first, optionally limited to one session,
    /// one tool, or one `action_id`.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on query
    /// failure.
    pub async fn actions(
        &self,
        action_id: Option<&str>,
        session_id: Option<&str>,
        tool_name: Option<&str>,
        limit: i64,
    ) -> Result<Vec<ActionRow>> {
        let mut preds = Vec::new();
        if action_id.is_some() {
            preds.push("a.action_id = $aid");
        }
        if tool_name.is_some() {
            preds.push("a.tool_name = $tool");
        }
        let session_match = if session_id.is_some() {
            "MATCH (a)-[:IN_SESSION]->(:Session {session_id: $sid}) "
        } else {
            ""
        };
        let where_clause = if preds.is_empty() {
            String::new()
        } else {
            format!("WHERE {} ", preds.join(" AND "))
        };
        let cypher = format!(
            "MATCH (a:Action) {where_clause}{session_match}\
             OPTIONAL MATCH (a)-[:TRIGGERED_BY]->(m:Message) \
             OPTIONAL MATCH (a)-[:IN_SESSION]->(s:Session) \
             OPTIONAL MATCH (a)-[:PRODUCED]->(art:Artifact) \
             RETURN id(a) AS nid, a.action_id AS aid, a.action_type AS atype, \
                    a.tool_name AS tool, a.input AS input, a.status AS status, \
                    a.error AS err, a.started_at AS started, a.ended_at AS ended, \
                    id(m) AS mnid, m.message_id AS mid, s.session_id AS sid, \
                    id(art) AS artnid, art.artifact_id AS artid \
             ORDER BY a.started_at DESC, a.action_id LIMIT $lim"
        );
        let session = self.db.session();
        let mut builder = session.query_with(&cypher).param("lim", limit);
        if let Some(a) = action_id {
            builder = builder.param("aid", a);
        }
        if let Some(t) = tool_name {
            builder = builder.param("tool", t);
        }
        if let Some(sid) = session_id {
            builder = builder.param("sid", sid);
        }
        let rows = builder.fetch_all().await?;
        Ok(rows
            .rows()
            .iter()
            .filter_map(|r| {
                Some(ActionRow {
                    node_id: i(r, "nid")?,
                    action_id: s(r, "aid")?,
                    action_type: s(r, "atype").unwrap_or_default(),
                    tool_name: s(r, "tool"),
                    input: r
                        .value("input")
                        .filter(|v| !matches!(v, Value::Null))
                        .cloned(),
                    status: s(r, "status"),
                    error: s(r, "err"),
                    started_at: r
                        .value("started")
                        .filter(|v| !matches!(v, Value::Null))
                        .cloned(),
                    ended_at: r
                        .value("ended")
                        .filter(|v| !matches!(v, Value::Null))
                        .cloned(),
                    triggered_by: i(r, "mnid").zip(s(r, "mid")),
                    session_id: s(r, "sid"),
                    artifact: i(r, "artnid").zip(s(r, "artid")),
                })
            })
            .collect())
    }

    /// The content id of an artifact's primary content, for a blob read.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on query
    /// failure.
    pub async fn artifact_content_id(&self, artifact_id: &str) -> Result<Option<(NodeId, String)>> {
        let rows = self
            .db
            .session()
            .query_with(
                "MATCH (a:Artifact {artifact_id: $id})-[:HAS_CONTENT]->(ac:ArtifactContent) \
                 RETURN id(a) AS nid, ac.content_id AS cid",
            )
            .param("id", artifact_id)
            .fetch_all()
            .await?;
        Ok(rows
            .rows()
            .first()
            .and_then(|r| i(r, "nid").zip(s(r, "cid"))))
    }

    /// Node ids of every chunk a parent owns.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on query
    /// failure.
    pub async fn owned_chunk_ids(&self, parent_nid: NodeId) -> Result<Vec<NodeId>> {
        let rows = self
            .db
            .session()
            .query_with("MATCH (p)-[:HAS_CHUNK]->(c:Chunk) WHERE id(p) = $pid RETURN id(c) AS nid")
            .param("pid", Value::Int(parent_nid))
            .fetch_all()
            .await?;
        Ok(rows.rows().iter().filter_map(|r| i(r, "nid")).collect())
    }
}
