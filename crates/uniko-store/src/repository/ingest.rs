//! Session-chunking reads: transcript/observation aggregation and the
//! entity/participant fan-out for observation chunks. Backs
//! `uniko_extract::ingest::session_chunk`.

use crate::error::Result;
use crate::storage::KnowledgeBase;
use crate::types::NodeId;

/// One message row for transcript chunking: text + resolved speaker name
/// (`"unknown"` when the `SENT_BY` participant is absent).
#[derive(Debug, Clone)]
pub struct TranscriptRow {
    pub content: String,
    pub speaker: String,
}

/// One observation row for observation chunking: text + its subject.
#[derive(Debug, Clone)]
pub struct ObservationRow {
    pub content: String,
    pub subject: String,
}

/// One existing session-anchored Chunk: its node id and stored text.
#[derive(Debug, Clone)]
pub struct SessionChunkRow {
    pub node_id: NodeId,
    pub text: String,
}

impl KnowledgeBase {
    /// All messages in `session_id`, oldest first, with the sender's name.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure.
    pub async fn session_transcript_rows(&self, session_id: &str) -> Result<Vec<TranscriptRow>> {
        let session = self.db.session();
        let cypher = "\
            MATCH (m:Message)-[:IN_SESSION]->(s:Session {session_id: $sid}) \
            OPTIONAL MATCH (m)-[:SENT_BY]->(p:Participant) \
            RETURN m.content AS content, p.name AS speaker, m.timestamp AS ts \
            ORDER BY m.timestamp";
        let result = session
            .query_with(cypher)
            .param("sid", session_id)
            .fetch_all()
            .await?;
        Ok(result
            .rows()
            .iter()
            .map(|row| TranscriptRow {
                content: row.get("content").unwrap_or_default(),
                speaker: row.get("speaker").unwrap_or_else(|_| "unknown".to_string()),
            })
            .collect())
    }

    /// Existing observation-chunk node ids for `session_id` (idempotency
    /// check for `chunk_type = 'observation'`).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure.
    pub async fn session_observation_chunk_ids(&self, session_id: &str) -> Result<Vec<NodeId>> {
        let session = self.db.session();
        let cypher = "\
            MATCH (s:Session {session_id: $sid})-[:HAS_CHUNK]->(c:Chunk {chunk_type: 'observation'}) \
            RETURN id(c) AS cid";
        let result = session
            .query_with(cypher)
            .param("sid", session_id)
            .fetch_all()
            .await?;
        Ok(result
            .rows()
            .iter()
            .filter_map(|r| r.get::<NodeId>("cid").ok())
            .collect())
    }

    /// Session-anchored Chunks of one `chunk_type`, ordered by chunk index.
    ///
    /// Returns both the node id and the stored text so a caller rebuilding
    /// a session-level surface can decide whether anything actually changed
    /// (compare the texts) and, if so, which nodes to delete (the ids).
    /// Comparing stored text is deliberate: it is exact, and it avoids a
    /// round-trip through the `CypherValue` `metadata` column, which nothing
    /// else in the engine reads back.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure.
    pub async fn session_chunk_rows(
        &self,
        session_id: &str,
        chunk_type: &str,
    ) -> Result<Vec<SessionChunkRow>> {
        let session = self.db.session();
        let cypher = "\
            MATCH (s:Session {session_id: $sid})-[:HAS_CHUNK]->(c:Chunk) \
            WHERE c.chunk_type = $ct \
            RETURN id(c) AS cid, c.text AS text, c.index AS idx \
            ORDER BY c.index";
        let result = session
            .query_with(cypher)
            .param("sid", session_id)
            .param("ct", chunk_type)
            .fetch_all()
            .await?;
        Ok(result
            .rows()
            .iter()
            .filter_map(|r| {
                Some(SessionChunkRow {
                    node_id: r.get::<NodeId>("cid").ok()?,
                    text: r.get("text").unwrap_or_default(),
                })
            })
            .collect())
    }

    /// External ids of Sessions that own no session-level Chunk yet.
    ///
    /// Drives a one-off backfill of knowledge bases ingested before session
    /// chunking was wired into the facade. A Session with messages but no
    /// `HAS_CHUNK` contributes nothing to session-scoped recall or to the
    /// Phase 1 session boost.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure.
    pub async fn unfinalized_session_ids(&self) -> Result<Vec<String>> {
        let session = self.db.session();
        let cypher = "\
            MATCH (s:Session) \
            WHERE NOT (s)-[:HAS_CHUNK]->(:Chunk) \
            RETURN s.session_id AS sid";
        let result = session.query_with(cypher).fetch_all().await?;
        Ok(result
            .rows()
            .iter()
            .filter_map(|r| r.get::<String>("sid").ok())
            .collect())
    }

    /// Stamp `ended_at` on a Session from its most recent message.
    ///
    /// Mirrors `close_inactive_sessions`: `ended_at` records when activity
    /// actually stopped, not when the call ran. Returns the stamped instant,
    /// or `None` when the Session or its messages do not exist.
    ///
    /// Note a Session is considered *open* while `ended_at IS NULL`, so
    /// stamping it excludes the Session from the inactivity auto-close sweep.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure.
    pub async fn stamp_session_ended_at(
        &self,
        session_id: &str,
    ) -> Result<Option<chrono::DateTime<chrono::Utc>>> {
        let session = self.db.session();
        let rows = session
            .query_with(
                "MATCH (m:Message)-[:IN_SESSION]->(s:Session {session_id: $sid}) \
                 RETURN id(s) AS nid, m.timestamp AS ts",
            )
            .param("sid", session_id)
            .fetch_all()
            .await?;
        let mut nid: Option<NodeId> = None;
        let mut last: Option<chrono::DateTime<chrono::Utc>> = None;
        for row in rows.rows() {
            let Ok(n) = row.get::<NodeId>("nid") else {
                continue;
            };
            nid = Some(n);
            if let Some(ts) = row
                .value("ts")
                .and_then(|v| crate::types::datetime_from_value(v, "Message.timestamp").ok())
                && last.is_none_or(|cur| ts > cur)
            {
                last = Some(ts);
            }
        }
        let Some(nid) = nid else {
            return Ok(None);
        };
        let last = last.unwrap_or_else(chrono::Utc::now);
        let mut props: std::collections::HashMap<String, crate::Value> =
            std::collections::HashMap::new();
        props.insert("ended_at".into(), crate::datetime_value(last));
        self.update_node(nid, &props).await?;
        Ok(Some(last))
    }

    /// All Observations linked to messages in `session_id`, oldest first.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure.
    pub async fn session_observation_rows(&self, session_id: &str) -> Result<Vec<ObservationRow>> {
        let session = self.db.session();
        let cypher = "\
            MATCH (o:Observation)-[:OBSERVED_IN]->(m:Message)-[:IN_SESSION]->(s:Session {session_id: $sid}) \
            RETURN o.content AS content, o.subject AS subject \
            ORDER BY m.timestamp";
        let result = session
            .query_with(cypher)
            .param("sid", session_id)
            .fetch_all()
            .await?;
        Ok(result
            .rows()
            .iter()
            .map(|row| ObservationRow {
                content: row.get("content").unwrap_or_default(),
                subject: row.get("subject").unwrap_or_default(),
            })
            .collect())
    }

    /// Distinct Entity node ids referenced (via `ABOUT`) by observations
    /// in `session_id`.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure.
    pub async fn session_observation_entity_ids(&self, session_id: &str) -> Result<Vec<NodeId>> {
        let session = self.db.session();
        let cypher = "\
            MATCH (o:Observation)-[:OBSERVED_IN]->(m:Message)-[:IN_SESSION]->(s:Session {session_id: $sid}) \
            MATCH (o)-[:ABOUT]->(e:Entity) \
            RETURN DISTINCT id(e) AS eid";
        let result = session
            .query_with(cypher)
            .param("sid", session_id)
            .fetch_all()
            .await?;
        Ok(result
            .rows()
            .iter()
            .filter_map(|row| row.get::<NodeId>("eid").ok())
            .collect())
    }

    /// Distinct Participant node ids referenced (via `ABOUT`) by
    /// observations in `session_id`.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure.
    pub async fn session_observation_participant_ids(
        &self,
        session_id: &str,
    ) -> Result<Vec<NodeId>> {
        let session = self.db.session();
        let cypher = "\
            MATCH (o:Observation)-[:OBSERVED_IN]->(m:Message)-[:IN_SESSION]->(s:Session {session_id: $sid}) \
            MATCH (o)-[:ABOUT]->(p:Participant) \
            RETURN DISTINCT id(p) AS pid";
        let result = session
            .query_with(cypher)
            .param("sid", session_id)
            .fetch_all()
            .await?;
        Ok(result
            .rows()
            .iter()
            .filter_map(|row| row.get::<NodeId>("pid").ok())
            .collect())
    }
}
