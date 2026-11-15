//! Addressed retrieval — dereference recall sources by id.
//!
//! [`crate::recall::RecallItem::sources`] hands back *references*
//! (`message_id` / `artifact_id`); this handle turns them into content.
//! Reached via [`Agent::data`](crate::Agent::data) so the high-frequency
//! recall verbs stay flat while by-id lookup lives in one coherent place.
//!
//! Every read applies the same policy as recall: forgotten content is never
//! returned, and an agent scoped to a viewer (see
//! [`UnikoBuilder::scope_to_agent`](crate::UnikoBuilder::scope_to_agent))
//! reads only what that viewer is admitted to. A refused read is
//! indistinguishable from an unknown id: both are `Ok(None)`.

use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Utc};
use uniko_store::{KnowledgeBase, NodeId, UnikoError, Value};

use crate::Agent;
use crate::policy::{Viewer, admitted_nodes};
use crate::recall::{RecallKind, ViewerScope};

/// A retrieved conversation message.
#[derive(Debug, Clone)]
pub struct MessageView {
    /// External message id.
    pub message_id: String,
    /// External id of the sending participant.
    pub sender_id: String,
    /// The message text.
    pub content: String,
    /// The turn's content type (`"text"` unless the turn set one).
    pub content_type: String,
    /// When the message was recorded.
    pub timestamp: DateTime<Utc>,
    /// External id of the session the message belongs to.
    pub session_id: String,
    /// External ids of the addressed recipients.
    pub addressed_to: Vec<String>,
    /// External ids of artifacts attached to this turn.
    pub attachments: Vec<String>,
    /// The caller's record category, when the turn set one.
    pub category: Option<String>,
    /// Who may see the turn (`private:{id}`, `team:{id}`, `org:{id}`);
    /// `None` is public.
    pub visibility: Option<String>,
}

/// A retrieved artifact (attachment or standalone document).
#[derive(Debug, Clone)]
pub struct ArtifactView {
    /// External artifact id.
    pub artifact_id: String,
    /// Artifact kind (`text` / `markdown` / `pdf` / …).
    pub kind: String,
    /// MIME type, when known.
    pub mime: Option<String>,
    /// Source path, when ingested from a file.
    pub path: Option<String>,
    /// Text reassembled from the artifact's chunks the reader may see, in
    /// order, with the overlap between neighbouring chunks removed.
    pub text: String,
    /// The message this artifact was attached to, when it was shared in a turn.
    pub attached_to_message: Option<String>,
}

/// Addressed (by-id) retrieval over an agent's store.
///
/// Borrows the [`Agent`]; obtain one with [`Agent::data`](crate::Agent::data).
/// All lookups return `Ok(None)` when the id is unknown, and equally when
/// the content is forgotten or the agent's viewer may not read it.
#[derive(Debug)]
pub struct Data<'a> {
    agent: &'a Agent,
}

/// What an artifact read may return, after policy.
pub(crate) struct ArtifactAccess {
    /// The artifact itself is readable.
    pub(crate) readable: bool,
    /// Its chunks the reader may see, by node id.
    pub(crate) chunks: HashSet<NodeId>,
    /// Every chunk is readable, so the original bytes may be returned.
    pub(crate) whole: bool,
}

impl<'a> Data<'a> {
    /// Construct a handle over `agent`. Crate-internal; callers use
    /// [`Agent::data`](crate::Agent::data).
    pub(crate) fn new(agent: &'a Agent) -> Self {
        Self { agent }
    }

    fn kb(&self) -> &KnowledgeBase {
        self.agent.kb()
    }

    /// The viewer reads are checked against; `None` for an unrestricted
    /// agent, which still never sees forgotten content.
    pub(crate) async fn viewer(&self) -> Result<Option<Viewer>, UnikoError> {
        Ok(match self.agent.resolve_viewer_scope().await? {
            ViewerScope::Unrestricted => None,
            ViewerScope::As(v) => Some(v),
            ViewerScope::AsParticipant(pid) => Some(Viewer::new(self.kb(), &pid).await?),
        })
    }

    /// Fetch a message by its external id.
    ///
    /// Returns `Ok(None)` when no message has that id, when it was
    /// forgotten, or when the agent's viewer may not see it.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when a graph query fails.
    pub async fn message(&self, message_id: &str) -> Result<Option<MessageView>, UnikoError> {
        let params = str_param("id", message_id);
        let rows = self
            .kb()
            .query_cypher(
                "MATCH (m:Message {message_id: $id}) \
                 OPTIONAL MATCH (m)-[:SENT_BY]->(s:Participant) \
                 OPTIONAL MATCH (m)-[:IN_SESSION]->(sess:Session) \
                 RETURN id(m) AS nid, m.content AS content, m.timestamp AS ts, \
                        m.content_type AS ctype, m.category AS category, \
                        m.visibility AS visibility, \
                        s.participant_id AS sender, sess.session_id AS session",
                &params,
            )
            .await?;
        let Some(row) = rows.first() else {
            return Ok(None);
        };
        let Some(nid) = cell_int(row, "nid") else {
            return Ok(None);
        };
        let viewer = self.viewer().await?;
        if admitted_nodes(self.kb(), viewer.as_ref(), &[(nid, RecallKind::Message)])
            .await?
            .is_empty()
        {
            return Ok(None);
        }
        let addressed_to = self
            .ext_id_list(
                "MATCH (m:Message {message_id: $id})-[:ADDRESSED_TO]->(p:Participant) \
                 RETURN p.participant_id AS v",
                message_id,
            )
            .await?;
        let attachments = self
            .ext_id_list(
                "MATCH (a:Artifact)-[:ATTACHED_TO]->(m:Message {message_id: $id}) \
                 RETURN a.artifact_id AS v",
                message_id,
            )
            .await?;
        Ok(Some(MessageView {
            message_id: message_id.to_string(),
            sender_id: cell_str(row, "sender").unwrap_or_default(),
            content: cell_str(row, "content").unwrap_or_default(),
            content_type: cell_str(row, "ctype").unwrap_or_else(|| "text".into()),
            timestamp: cell_datetime(row, "ts"),
            session_id: cell_str(row, "session").unwrap_or_default(),
            addressed_to,
            attachments,
            category: cell_str(row, "category"),
            visibility: cell_str(row, "visibility"),
        }))
    }

    /// Decide what of the artifact `artifact_nid` the reader may see.
    ///
    /// An artifact shared in turns — attached to a message, or produced by
    /// a tool call a message triggered — is readable when at least one of
    /// those messages is; a standalone document is readable by anyone the
    /// store admits. Its chunks are then checked one by one, since a
    /// forgotten or scoped chunk must not leak through its artifact, and an
    /// artifact none of whose chunks the reader may see is not readable at
    /// all.
    pub(crate) async fn artifact_access(
        &self,
        artifact_nid: NodeId,
        chunk_nids: &[NodeId],
    ) -> Result<ArtifactAccess, UnikoError> {
        let kb = self.kb();
        let viewer = self.viewer().await?;
        let aid = HashMap::from([("aid".to_string(), Value::Int(artifact_nid))]);
        let mut msg_rows = kb
            .query_cypher(
                "MATCH (a:Artifact)-[:ATTACHED_TO]->(m:Message) WHERE id(a) = $aid \
                 RETURN id(m) AS nid",
                &aid,
            )
            .await?;
        msg_rows.extend(
            kb.query_cypher(
                "MATCH (act:Action)-[:PRODUCED]->(a:Artifact) WHERE id(a) = $aid \
                 MATCH (act)-[:TRIGGERED_BY]->(m:Message) RETURN id(m) AS nid",
                &aid,
            )
            .await?,
        );
        let msgs: Vec<(NodeId, RecallKind)> = msg_rows
            .iter()
            .filter_map(|r| cell_int(r, "nid"))
            .map(|n| (n, RecallKind::Message))
            .collect();
        let readable =
            msgs.is_empty() || !admitted_nodes(kb, viewer.as_ref(), &msgs).await?.is_empty();
        if !readable {
            return Ok(ArtifactAccess {
                readable,
                chunks: HashSet::new(),
                whole: false,
            });
        }
        let pairs: Vec<(NodeId, RecallKind)> =
            chunk_nids.iter().map(|n| (*n, RecallKind::Chunk)).collect();
        let chunks = admitted_nodes(kb, viewer.as_ref(), &pairs).await?;
        let whole = chunks.len() == chunk_nids.len();
        Ok(ArtifactAccess {
            readable: chunk_nids.is_empty() || !chunks.is_empty(),
            chunks,
            whole,
        })
    }

    /// Fetch an artifact's metadata + reassembled text by its external id.
    ///
    /// Returns `Ok(None)` when no artifact has that id, or when the agent's
    /// viewer may not read it. The text is reassembled from the artifact's
    /// direct `HAS_CHUNK` chunks the reader may see, in index order; for
    /// tiered-PDF artifacts (text under page→block chains) the text may be
    /// empty — fetch [`artifact_bytes`](Self::artifact_bytes) for the
    /// original.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when a graph query fails.
    pub async fn artifact(&self, artifact_id: &str) -> Result<Option<ArtifactView>, UnikoError> {
        let params = str_param("id", artifact_id);
        let rows = self
            .kb()
            .query_cypher(
                "MATCH (a:Artifact {artifact_id: $id}) \
                 OPTIONAL MATCH (a)-[:ATTACHED_TO]->(m:Message) \
                 RETURN id(a) AS nid, a.kind AS kind, a.mime_type AS mime, a.path AS path, \
                        m.message_id AS attached",
                &params,
            )
            .await?;
        let Some(row) = rows.first() else {
            return Ok(None);
        };
        let Some(artifact_nid) = cell_int(row, "nid") else {
            return Ok(None);
        };
        let chunk_rows = self
            .kb()
            .query_cypher(
                "MATCH (a:Artifact {artifact_id: $id})-[:HAS_CHUNK]->(c:Chunk) \
                 RETURN id(c) AS nid, c.text AS text, c.start AS s, c.end AS e \
                 ORDER BY c.index",
                &params,
            )
            .await?;
        let chunk_nids: Vec<NodeId> = chunk_rows
            .iter()
            .filter_map(|r| cell_int(r, "nid"))
            .collect();
        let access = self.artifact_access(artifact_nid, &chunk_nids).await?;
        if !access.readable {
            return Ok(None);
        }
        let pieces: Vec<ChunkPiece> = chunk_rows
            .iter()
            .filter(|r| cell_int(r, "nid").is_some_and(|n| access.chunks.contains(&n)))
            .filter_map(|r| {
                Some(ChunkPiece {
                    text: cell_str(r, "text")?,
                    start: cell_int(r, "s").and_then(|v| usize::try_from(v).ok()),
                    end: cell_int(r, "e").and_then(|v| usize::try_from(v).ok()),
                })
            })
            .collect();
        Ok(Some(ArtifactView {
            artifact_id: artifact_id.to_string(),
            kind: cell_str(row, "kind").unwrap_or_default(),
            mime: cell_str(row, "mime"),
            path: cell_str(row, "path"),
            text: reassemble(&pieces),
            attached_to_message: cell_str(row, "attached"),
        }))
    }

    /// Fetch an artifact's original bytes by its external id.
    ///
    /// Resolves the artifact's content blob and reads it (inline or from the
    /// backing blob store). Returns `Ok(None)` when the artifact or its
    /// content is unknown, when the agent's viewer may not read it, or when
    /// any of its chunks is forgotten or hidden from that viewer — the
    /// original bytes would reveal what those chunks withhold.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when the lookup or blob read fails.
    pub async fn artifact_bytes(&self, artifact_id: &str) -> Result<Option<Vec<u8>>, UnikoError> {
        let params = str_param("id", artifact_id);
        let rows = self
            .kb()
            .query_cypher(
                "MATCH (a:Artifact {artifact_id: $id})-[:HAS_CONTENT]->(ac:ArtifactContent) \
                 RETURN id(a) AS nid, ac.content_id AS cid",
                &params,
            )
            .await?;
        let Some(row) = rows.first() else {
            return Ok(None);
        };
        let (Some(artifact_nid), Some(content_id)) = (cell_int(row, "nid"), cell_str(row, "cid"))
        else {
            return Ok(None);
        };
        let chunk_rows = self
            .kb()
            .query_cypher(
                "MATCH (a:Artifact {artifact_id: $id})-[:HAS_CHUNK]->(c:Chunk) RETURN id(c) AS nid",
                &params,
            )
            .await?;
        let chunk_nids: Vec<NodeId> = chunk_rows
            .iter()
            .filter_map(|r| cell_int(r, "nid"))
            .collect();
        let access = self.artifact_access(artifact_nid, &chunk_nids).await?;
        if !access.readable || !access.whole {
            return Ok(None);
        }
        Ok(Some(self.kb().fetch_blob(&content_id).await?))
    }

    /// Collect a single `v`-aliased external-id column into a `Vec`.
    async fn ext_id_list(&self, cypher: &str, id: &str) -> Result<Vec<String>, UnikoError> {
        let rows = self.kb().query_cypher(cypher, &str_param("id", id)).await?;
        Ok(rows.iter().filter_map(|r| cell_str(r, "v")).collect())
    }
}

/// One chunk's text and its byte span in the artifact's source.
struct ChunkPiece {
    text: String,
    start: Option<usize>,
    end: Option<usize>,
}

/// Join chunks in order, dropping the part of each chunk that overlaps the
/// previous one.
///
/// A chunk whose text is exactly its `[start, end)` span lets the overlap
/// be cut by offset. Chunks written before spans were exact, or with no
/// offsets, are joined with a newline as they always were.
fn reassemble(pieces: &[ChunkPiece]) -> String {
    let mut out = String::new();
    let mut prev_end: Option<usize> = None;
    for p in pieces {
        let exact =
            matches!((p.start, p.end), (Some(s), Some(e)) if e >= s && e - s == p.text.len());
        let text = match (exact, p.start, prev_end) {
            (true, Some(s), Some(pe)) if s < pe => {
                let cut = pe - s;
                if cut >= p.text.len() || !p.text.is_char_boundary(cut) {
                    ""
                } else {
                    &p.text[cut..]
                }
            }
            _ => p.text.as_str(),
        };
        let contiguous = exact && matches!((p.start, prev_end), (Some(s), Some(pe)) if s <= pe);
        if !out.is_empty() && !contiguous {
            out.push('\n');
        }
        out.push_str(text);
        prev_end = if exact { p.end } else { None };
    }
    out
}

/// Build a single `$key = string` Cypher parameter map.
fn str_param(key: &str, value: &str) -> HashMap<String, Value> {
    HashMap::from([(key.to_string(), Value::String(value.to_string()))])
}

/// Read a string cell from a decoded row.
fn cell_str(row: &HashMap<String, Value>, key: &str) -> Option<String> {
    match row.get(key) {
        Some(Value::String(s)) => Some(s.clone()),
        _ => None,
    }
}

/// Read an integer cell.
fn cell_int(row: &HashMap<String, Value>, key: &str) -> Option<i64> {
    match row.get(key) {
        Some(Value::Int(n)) => Some(*n),
        _ => None,
    }
}

/// Read a datetime cell, falling back to the Unix epoch when absent/malformed.
fn cell_datetime(row: &HashMap<String, Value>, key: &str) -> DateTime<Utc> {
    row.get(key)
        .and_then(|v| uniko_store::types::datetime_from_value(v, "MessageView.timestamp").ok())
        .unwrap_or_else(|| DateTime::<Utc>::from_timestamp(0, 0).expect("epoch is valid"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn piece(text: &str, start: usize) -> ChunkPiece {
        ChunkPiece {
            text: text.into(),
            start: Some(start),
            end: Some(start + text.len()),
        }
    }

    #[test]
    fn reassemble_drops_overlap_by_offset() {
        let src = "alpha beta gamma delta";
        // chunk 0 = [0,10) "alpha beta", chunk 1 starts at 6 with overlap.
        let pieces = [piece(&src[0..10], 0), piece(&src[6..22], 6)];
        assert_eq!(reassemble(&pieces), src);
    }

    #[test]
    fn reassemble_separates_gaps_and_legacy_chunks() {
        let pieces = [
            piece("one", 0),
            piece("two", 10),
            ChunkPiece {
                text: "legacy".into(),
                start: Some(0),
                end: Some(99),
            },
        ];
        assert_eq!(reassemble(&pieces), "one\ntwo\nlegacy");
    }
}
