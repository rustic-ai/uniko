//! Media sources: the original image, audio or video that a text artifact
//! was derived from.
//!
//! A media source is an `:Artifact` with `kind = "media"`, keyed by the
//! SHA-256 of its bytes. It has no chunks. The text derived from it is a
//! separate artifact linked by
//! `(derived:Artifact)-[:DERIVED_FROM {derivation_kind, derived_at}]->(media)`.
//! When the bytes are retained, the source also has a `HAS_CONTENT` edge to
//! the `:ArtifactContent` row that holds them; when only the identity is
//! retained it has none.
//!
//! Nothing here applies visibility: the facade admits the derived
//! artifacts' chunks against its viewer before handing anything out.

use std::collections::HashMap;

use chrono::Utc;
use uni_db::{Transaction, Value};

use crate::error::{Result, UnikoError};
use crate::schema::constants::edges;
use crate::storage::KnowledgeBase;
use crate::storage::blob::MergeContent;
use crate::types::{NodeId, datetime_value};

/// The `Artifact.kind` of a media source.
pub const MEDIA_KIND: &str = "media";

/// What a media payload's header says about it. Every field is optional.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct MediaHeader {
    /// Image or video width, in pixels.
    pub width: Option<i32>,
    /// Image or video height, in pixels.
    pub height: Option<i32>,
    /// Audio or video duration, in milliseconds.
    pub duration_ms: Option<i64>,
    /// Audio sample rate, in Hz.
    pub sample_rate: Option<i32>,
    /// Audio channel count.
    pub channels: Option<i32>,
}

/// A media source to record inside a transaction.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MediaSourceSpec {
    /// SHA-256 hex of the media bytes: the source's identity.
    pub hash: String,
    /// `image`, `audio` or `video`.
    pub modality: String,
    /// Resolved MIME type.
    pub mime: String,
    /// Byte length.
    pub size: i64,
    /// Source path, when known.
    pub path: Option<String>,
    /// What the header says.
    pub header: MediaHeader,
    /// The bytes to retain, already PUT to the blob backend. `None` records
    /// the identity only.
    pub content: Option<MergeContent>,
}

impl MediaSourceSpec {
    /// A spec for media of `modality` with `hash`, `mime` and `size`, that
    /// records the identity only.
    #[must_use]
    pub fn new(
        hash: impl Into<String>,
        modality: impl Into<String>,
        mime: impl Into<String>,
        size: i64,
    ) -> Self {
        Self {
            hash: hash.into(),
            modality: modality.into(),
            mime: mime.into(),
            size,
            path: None,
            header: MediaHeader::default(),
            content: None,
        }
    }

    /// Set the path.
    #[must_use]
    pub fn with_path(mut self, path: Option<String>) -> Self {
        self.path = path;
        self
    }

    /// Set the header fields.
    #[must_use]
    pub fn with_header(mut self, header: MediaHeader) -> Self {
        self.header = header;
        self
    }

    /// Retain `content` (the merge spec for the bytes).
    #[must_use]
    pub fn with_content(mut self, content: MergeContent) -> Self {
        self.content = Some(content);
        self
    }
}

/// A recorded media source, as read back.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct MediaSource {
    /// The source artifact's node id.
    pub node_id: NodeId,
    /// The source artifact's external id. With retained bytes, pass it to
    /// the facade's `artifact_bytes` to read them.
    pub artifact_id: String,
    /// `image`, `audio` or `video`.
    pub modality: Option<String>,
    /// MIME type.
    pub mime: Option<String>,
    /// SHA-256 hex of the bytes.
    pub hash: String,
    /// Byte length.
    pub size: Option<i64>,
    /// Source path, when one was known.
    pub path: Option<String>,
    /// What the header said.
    pub header: MediaHeader,
    /// How the derived artifact was made from it (`ocr`, `transcript`, ...).
    pub derivation_kind: Option<String>,
    /// Whether the bytes are retained.
    pub bytes_retained: bool,
}

fn s(row: &uni_db::Row, k: &str) -> Option<String> {
    row.get::<String>(k).ok()
}

fn i(row: &uni_db::Row, k: &str) -> Option<i64> {
    row.get::<i64>(k).ok()
}

fn i32_of(row: &uni_db::Row, k: &str) -> Option<i32> {
    i(row, k).and_then(|v| i32::try_from(v).ok())
}

impl KnowledgeBase {
    /// Find or create the media source for `spec.hash` **inside** `tx`, and
    /// attach its bytes when `spec.content` is set and it has none yet.
    /// Returns its node id and external id.
    ///
    /// The caller must hold the `content:<hash>` lock, acquired before `tx`
    /// was opened and held across the commit (as for
    /// [`merge_artifact_content_in_tx`](Self::merge_artifact_content_in_tx)):
    /// without it two writers can both read "absent" and both create one.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on a graph failure.
    pub async fn merge_media_source_in_tx(
        &self,
        tx: &Transaction,
        spec: &MediaSourceSpec,
    ) -> Result<(NodeId, String)> {
        let existing = tx
            .query_with(
                "MATCH (a:Artifact {hash: $hash, kind: $kind}) \
                 OPTIONAL MATCH (a)-[:HAS_CONTENT]->(c:ArtifactContent) \
                 RETURN id(a) AS nid, a.artifact_id AS aid, count(c) AS contents \
                 LIMIT 1",
            )
            .param("hash", Value::String(spec.hash.clone()))
            .param("kind", Value::String(MEDIA_KIND.to_string()))
            .fetch_all()
            .await?;
        let (nid, artifact_id, has_content) = match existing.rows().first() {
            Some(row) => (
                row.get::<i64>("nid")?,
                s(row, "aid").unwrap_or_default(),
                i(row, "contents").unwrap_or(0) > 0,
            ),
            None => {
                let artifact_id = crate::id::new_id();
                let mut props: HashMap<String, Value> = HashMap::from([
                    ("artifact_id".into(), Value::String(artifact_id.clone())),
                    ("kind".into(), Value::String(MEDIA_KIND.to_string())),
                    ("hash".into(), Value::String(spec.hash.clone())),
                    ("size".into(), Value::Int(spec.size)),
                    ("mime_type".into(), Value::String(spec.mime.clone())),
                    ("created_at".into(), datetime_value(Utc::now())),
                    (
                        "modality_meta".into(),
                        Value::Map(HashMap::from([(
                            "modality".to_string(),
                            Value::String(spec.modality.clone()),
                        )])),
                    ),
                ]);
                if let Some(path) = &spec.path {
                    props.insert("path".into(), Value::String(path.clone()));
                }
                let h = &spec.header;
                for (k, v) in [
                    ("width", h.width.map(i64::from)),
                    ("height", h.height.map(i64::from)),
                    ("duration_ms", h.duration_ms),
                    ("sample_rate", h.sample_rate.map(i64::from)),
                    ("channels", h.channels.map(i64::from)),
                ] {
                    if let Some(v) = v {
                        props.insert(k.into(), Value::Int(v));
                    }
                }
                let nid = self.create_node_in_tx(tx, "Artifact", &props).await?;
                (nid, artifact_id, false)
            }
        };
        if let Some(content) = &spec.content
            && !has_content
        {
            let content_nid = self
                .merge_artifact_content_in_tx(tx, content.clone())
                .await?;
            let props = HashMap::from([("role".to_string(), Value::String("primary".into()))]);
            self.create_edges_in_tx(tx, &[(edges::HAS_CONTENT, nid, content_nid, props)])
                .await?;
        }
        Ok((nid, artifact_id))
    }

    /// Record that `derived` was made from the media source `source`, as
    /// `derivation_kind`, **inside** `tx`. Does nothing when the edge
    /// already exists, so a replayed ingest does not duplicate it. Returns
    /// whether an edge was written.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on a graph failure.
    pub async fn link_derived_from_in_tx(
        &self,
        tx: &Transaction,
        derived: NodeId,
        source: NodeId,
        derivation_kind: &str,
    ) -> Result<bool> {
        let existing = tx
            .query_with(
                "MATCH (d:Artifact)-[r:DERIVED_FROM]->(m:Artifact) \
                 WHERE id(d) = $d AND id(m) = $m RETURN id(r) AS eid LIMIT 1",
            )
            .param("d", Value::Int(derived))
            .param("m", Value::Int(source))
            .fetch_all()
            .await?;
        if !existing.rows().is_empty() {
            return Ok(false);
        }
        let props = HashMap::from([
            (
                "derivation_kind".to_string(),
                Value::String(derivation_kind.to_string()),
            ),
            ("derived_at".to_string(), datetime_value(Utc::now())),
        ]);
        self.create_edges_in_tx(tx, &[(edges::DERIVED_FROM, derived, source, props)])
            .await?;
        Ok(true)
    }

    /// The hashes of the media sources the artifact `derived` was derived
    /// from, read **inside** `tx` (so it sees this transaction's writes).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on a graph failure.
    pub async fn media_hashes_in_tx(
        &self,
        tx: &Transaction,
        derived: NodeId,
    ) -> Result<Vec<String>> {
        let rows = tx
            .query_with(
                "MATCH (d:Artifact)-[:DERIVED_FROM]->(m:Artifact) \
                 WHERE id(d) = $d AND m.kind = $kind RETURN m.hash AS hash",
            )
            .param("d", Value::Int(derived))
            .param("kind", Value::String(MEDIA_KIND.to_string()))
            .fetch_all()
            .await?;
        Ok(rows.rows().iter().filter_map(|r| s(r, "hash")).collect())
    }

    /// The media sources the artifact `artifact_id` was derived from, or
    /// the source itself when `artifact_id` names one.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on a graph failure.
    pub async fn media_sources(&self, artifact_id: &str) -> Result<Vec<MediaSource>> {
        let session = self.db.session();
        let params = || Value::String(artifact_id.to_string());
        let derived = session
            .query_with(
                "MATCH (d:Artifact {artifact_id: $id})-[r:DERIVED_FROM]->(m:Artifact) \
                 WHERE m.kind = $kind \
                 OPTIONAL MATCH (m)-[:HAS_CONTENT]->(c:ArtifactContent) \
                 RETURN id(m) AS nid, m.artifact_id AS aid, m.modality_meta AS meta, \
                        m.mime_type AS mime, m.hash AS hash, m.size AS size, m.path AS path, \
                        m.width AS width, m.height AS height, m.duration_ms AS duration_ms, \
                        m.sample_rate AS sample_rate, m.channels AS channels, \
                        r.derivation_kind AS dk, count(c) AS contents",
            )
            .param("id", params())
            .param("kind", Value::String(MEDIA_KIND.to_string()))
            .fetch_all()
            .await?;
        let rows = if derived.rows().is_empty() {
            session
                .query_with(
                    "MATCH (m:Artifact {artifact_id: $id}) WHERE m.kind = $kind \
                     OPTIONAL MATCH (m)-[:HAS_CONTENT]->(c:ArtifactContent) \
                     RETURN id(m) AS nid, m.artifact_id AS aid, m.modality_meta AS meta, \
                            m.mime_type AS mime, m.hash AS hash, m.size AS size, m.path AS path, \
                            m.width AS width, m.height AS height, m.duration_ms AS duration_ms, \
                            m.sample_rate AS sample_rate, m.channels AS channels, \
                            null AS dk, count(c) AS contents",
                )
                .param("id", params())
                .param("kind", Value::String(MEDIA_KIND.to_string()))
                .fetch_all()
                .await?
        } else {
            derived
        };
        let mut out = Vec::new();
        for row in rows.rows() {
            let modality = match row.value("meta") {
                Some(Value::Map(m)) => match m.get("modality") {
                    Some(Value::String(v)) => Some(v.clone()),
                    _ => None,
                },
                _ => None,
            };
            out.push(MediaSource {
                node_id: row.get::<i64>("nid")?,
                artifact_id: s(row, "aid")
                    .ok_or_else(|| UnikoError::Storage("media source has no artifact_id".into()))?,
                modality,
                mime: s(row, "mime"),
                hash: s(row, "hash").unwrap_or_default(),
                size: i(row, "size"),
                path: s(row, "path"),
                header: MediaHeader {
                    width: i32_of(row, "width"),
                    height: i32_of(row, "height"),
                    duration_ms: i(row, "duration_ms"),
                    sample_rate: i32_of(row, "sample_rate"),
                    channels: i32_of(row, "channels"),
                },
                derivation_kind: s(row, "dk"),
                bytes_retained: i(row, "contents").unwrap_or(0) > 0,
            });
        }
        out.sort_by(|a, b| a.artifact_id.cmp(&b.artifact_id));
        Ok(out)
    }

    /// The node ids of every artifact derived from the media source
    /// `source`, and each one's chunk node ids. Empty when `source` is not
    /// a media source.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on a graph failure.
    pub async fn artifacts_derived_from(
        &self,
        source: NodeId,
    ) -> Result<Vec<(NodeId, Vec<NodeId>)>> {
        let rows = self
            .db
            .session()
            .query_with(
                "MATCH (d:Artifact)-[:DERIVED_FROM]->(m:Artifact) \
                 WHERE id(m) = $m AND m.kind = $kind \
                 OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk) \
                 WITH id(d) AS did, collect(id(c)) AS chunks RETURN did, chunks",
            )
            .param("m", Value::Int(source))
            .param("kind", Value::String(MEDIA_KIND.to_string()))
            .fetch_all()
            .await?;
        let mut out = Vec::new();
        for row in rows.rows() {
            let did = row.get::<i64>("did")?;
            let chunks = match row.value("chunks") {
                Some(Value::List(items)) => items
                    .iter()
                    .filter_map(|v| match v {
                        Value::Int(n) => Some(*n),
                        _ => None,
                    })
                    .collect(),
                _ => Vec::new(),
            };
            out.push((did, chunks));
        }
        Ok(out)
    }
}
