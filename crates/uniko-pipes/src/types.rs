//! Data types for pipeline task routing.
//!
//! These types flow through channels between the pipeline system and its
//! workers.  They carry no logic — only data.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use uniko_store::NodeId;

// ── Ingest tasks ────────────────────────────────────────────────────

/// A message to ingest into the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestMessage {
    /// Caller-provided or auto-generated UUID v7.
    pub message_id: String,
    /// Raw message text.
    pub content: String,
    /// MIME-like content type: `"text"`, `"code"`, `"tool_result"`, etc.
    pub content_type: String,
    /// Participant who sent this message.
    pub sender_id: String,
    /// Session this message belongs to.
    pub session_id: String,
    /// Recipient participant IDs.  If `None`, inferred from session
    /// participants (all except the sender).
    pub addressed_to: Option<Vec<String>>,
    /// When the message was sent.
    pub timestamp: DateTime<Utc>,
    /// Arbitrary caller metadata forwarded to pipeline steps.
    pub metadata: HashMap<String, serde_json::Value>,
    /// Caller's own record class — a user assertion, an executed result, a
    /// model interpretation (issue #39). Typed provenance: it is filterable
    /// by recall and deliberately NOT part of the searchable text.
    #[serde(default)]
    pub category: Option<String>,
    /// Stable logical source this record came from. Materialised as a
    /// `:Source` node with a `FROM_SOURCE` edge, and denormalised onto the
    /// record so the recall filter needs no traversal.
    #[serde(default)]
    pub source_id: Option<String>,
    /// Immutable revision identity for this ingest (issue #41). Together
    /// with the content fingerprint it decides idempotency: the same
    /// revision with the same bytes is a no-op, the same revision with
    /// changed bytes is rejected.
    #[serde(default)]
    pub revision_id: Option<String>,
}

/// An artifact (file, document, URL) to ingest.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IngestArtifact {
    /// Caller-provided or auto-generated UUID v7.
    pub artifact_id: String,
    /// Whether `artifact_id` is the caller's own stable id rather than an
    /// auto-generated UUID.
    ///
    /// This is what decides identity on ingest. A caller-chosen id *is* the
    /// artifact's identity: two ids over identical bytes are two artifacts
    /// (sharing one stored copy of the content), and reusing one id for
    /// different bytes is an error. An auto-generated id means the caller
    /// expressed no identity, so identical bytes dedup onto the existing
    /// artifact — which is what keeps re-running a corpus load from
    /// duplicating every document.
    #[serde(default)]
    pub caller_supplied_id: bool,
    /// Text content (empty for binary artifacts).
    pub content: String,
    /// Artifact kind: `"file"`, `"document"`, `"url"`, `"snippet"`, etc.
    pub kind: String,
    /// Filesystem path, URL, or identifier.
    pub path: Option<String>,
    /// Arbitrary caller metadata.
    pub metadata: HashMap<String, serde_json::Value>,
    /// Optional `session_id` this artifact was shared in.  When set and
    /// resolvable, an `ATTACHED_TO` edge links the Artifact to the
    /// Session (F22).
    #[serde(default)]
    pub session_id: Option<String>,
    /// Optional `message_id` that introduced this artifact.  When set and
    /// resolvable, an `ATTACHED_TO` edge links the Artifact to the
    /// Message (F30).
    #[serde(default)]
    pub triggered_by_message_id: Option<String>,
    /// Optional `action_id` that produced this artifact.  When set and
    /// resolvable, a `PRODUCED` edge links the Action to the Artifact
    /// (F18).
    #[serde(default)]
    pub produced_by_action_id: Option<String>,
    /// Caller's own record class — a user assertion, an executed result, a
    /// model interpretation (issue #39). Typed provenance: it is filterable
    /// by recall and deliberately NOT part of the searchable text.
    #[serde(default)]
    pub category: Option<String>,
    /// Stable logical source this record came from. Materialised as a
    /// `:Source` node with a `FROM_SOURCE` edge, and denormalised onto the
    /// record so the recall filter needs no traversal.
    #[serde(default)]
    pub source_id: Option<String>,
    /// Immutable revision identity for this ingest (issue #41). Together
    /// with the content fingerprint it decides idempotency: the same
    /// revision with the same bytes is a no-op, the same revision with
    /// changed bytes is rejected.
    #[serde(default)]
    pub revision_id: Option<String>,
}

/// Source of PDF bytes — mirrors `uniko_extract::ingest::pdf::PdfInput`.
///
/// Mirrored here (rather than re-exported) because `uniko-pipes` is
/// upstream of `uniko-extract` in the workspace dep graph. The wire
/// types intentionally stay serializable; the runtime
/// `PdfIngestOptions::extractor` knob lives only on the extract side.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PdfInput {
    /// In-memory bytes (base64-decoded by the caller before this point).
    Bytes(Vec<u8>),
    /// Filesystem path; bytes are read at ingest time.
    Path(std::path::PathBuf),
}

/// A PDF document to ingest into the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestPdf {
    /// Caller-provided artifact identifier.
    pub artifact_id: String,
    /// PDF byte source.
    pub input: PdfInput,
    /// Optional source path / URL recorded on `:Artifact.path`.
    pub source_path: Option<String>,
    /// Arbitrary caller metadata.
    pub metadata: HashMap<String, serde_json::Value>,
}

/// The payload of an [`IngestSource`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IngestData {
    /// Raw bytes (MIME sniffed from magic bytes when not given).
    Bytes(Vec<u8>),
    /// UTF-8 text.
    Text(String),
    /// A filesystem path; bytes/text are read at ingest time.
    Path(std::path::PathBuf),
}

/// A blob to ingest through the unified, MIME-routed dispatch.
///
/// Construct with [`IngestSource::bytes`] / [`text`](IngestSource::text) /
/// [`path`](IngestSource::path) and refine with the chainable setters. The
/// MIME is resolved as: explicit [`mime`](IngestSource::mime) → magic bytes →
/// file extension → text/plain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestSource {
    /// The content payload.
    pub data: IngestData,
    /// Explicit MIME override; skips sniffing when set.
    pub mime: Option<crate::content::Mime>,
    /// Explicit artifact id (otherwise a UUID v7 is generated).
    pub id: Option<String>,
    /// Source path / URL recorded on the artifact.
    pub path: Option<String>,
    /// Arbitrary metadata forwarded to ingest.
    pub metadata: HashMap<String, serde_json::Value>,
    /// Caller's own record class — a user assertion, an executed result, a
    /// model interpretation (issue #39). Typed provenance: it is filterable
    /// by recall and deliberately NOT part of the searchable text.
    #[serde(default)]
    pub category: Option<String>,
    /// Stable logical source this record came from. Materialised as a
    /// `:Source` node with a `FROM_SOURCE` edge, and denormalised onto the
    /// record so the recall filter needs no traversal.
    #[serde(default)]
    pub source_id: Option<String>,
    /// Immutable revision identity for this ingest (issue #41). Together
    /// with the content fingerprint it decides idempotency: the same
    /// revision with the same bytes is a no-op, the same revision with
    /// changed bytes is rejected.
    #[serde(default)]
    pub revision_id: Option<String>,
}

impl IngestSource {
    /// A byte payload.
    #[must_use]
    pub fn bytes(bytes: Vec<u8>) -> Self {
        Self::with_data(IngestData::Bytes(bytes))
    }

    /// A text payload.
    #[must_use]
    pub fn text(content: impl Into<String>) -> Self {
        Self::with_data(IngestData::Text(content.into()))
    }

    /// A filesystem path payload.
    #[must_use]
    pub fn path(path: impl Into<std::path::PathBuf>) -> Self {
        let path = path.into();
        let recorded = path.display().to_string();
        Self {
            data: IngestData::Path(path),
            mime: None,
            id: None,
            path: Some(recorded),
            metadata: HashMap::new(),
            category: None,
            source_id: None,
            revision_id: None,
        }
    }

    fn with_data(data: IngestData) -> Self {
        Self {
            data,
            mime: None,
            id: None,
            path: None,
            metadata: HashMap::new(),
            category: None,
            source_id: None,
            revision_id: None,
        }
    }

    /// Set an explicit MIME, skipping sniffing.
    #[must_use]
    pub fn with_mime(mut self, mime: crate::content::Mime) -> Self {
        self.mime = Some(mime);
        self
    }

    /// Set an explicit artifact id.
    ///
    /// The id — not the content hash — is this artifact's identity, and is
    /// what `agent.data().artifact(..)` fetches it back by. Re-ingesting an
    /// id with identical content is idempotent; reusing it for different
    /// content is rejected as an id conflict. Two ids over identical bytes
    /// give two artifacts sharing one stored copy of the content, so a
    /// second session that ingests the same document under its own id keeps
    /// its own handle on it.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Tag this source with the caller's own record category (issue #39).
    ///
    /// Typed provenance: recall can filter on it, and it never enters the
    /// searchable text — which is the point, since the alternative is
    /// encoding a class into the prose and parsing it back out of results.
    #[must_use]
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// Attribute this source to a stable logical source id (issue #39).
    ///
    /// Materialised as a `:Source` node with a `FROM_SOURCE` edge, so the
    /// same logical origin can be recognised across separate ingests.
    #[must_use]
    pub fn with_source(mut self, source_id: impl Into<String>) -> Self {
        self.source_id = Some(source_id.into());
        self
    }

    /// Declare which revision of that source these bytes are (issue #41).
    ///
    /// Re-ingesting the same revision with identical bytes is idempotent;
    /// the same revision with changed bytes is rejected, because a revision
    /// id is a promise about the content. A NEW revision of the same source
    /// supersedes the previous one, which then stops grounding current
    /// answers while staying attributable to history.
    #[must_use]
    pub fn with_revision(mut self, revision_id: impl Into<String>) -> Self {
        self.revision_id = Some(revision_id.into());
        self
    }

    /// Record a source path / URL on the artifact.
    #[must_use]
    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }
}

/// A task submitted to the ingest pipeline.
#[derive(Debug, Clone)]
pub enum IngestTask {
    /// Ingest a conversation message.
    Message(IngestMessage),
    /// Ingest an artifact (file, document, etc.).
    Artifact(IngestArtifact),
    /// Ingest a PDF document.
    Pdf(IngestPdf),
    /// Ingest a MIME-routed blob (document / PDF / future image-audio).
    Source(IngestSource),
}

// ── Consolidation tasks ─────────────────────────────────────────────

/// Notification that new observations are ready for consolidation.
#[derive(Debug, Clone)]
pub struct ObservationsReady {
    /// Agent whose observations should be consolidated.
    pub agent_id: String,
    /// Number of new observations since last consolidation.
    pub observation_count: u32,
    /// Node IDs of the source observations.
    pub source_node_ids: Vec<NodeId>,
}

/// A task submitted to the consolidation pipeline.
#[derive(Debug, Clone)]
pub enum ConsolidationTask {
    /// New observations are available — may trigger consolidation.
    ObservationsReady(ObservationsReady),
    /// Force an immediate consolidation cycle for the agent.
    ForceConsolidate {
        /// Target agent.
        agent_id: String,
    },
    /// Run a consolidation cycle (internal timer trigger).
    RunCycle {
        /// Target agent.
        agent_id: String,
    },
}

// ── Step outcomes ───────────────────────────────────────────────────

/// What to do when a pipeline step fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepErrorPolicy {
    /// Log warning, continue to next step.
    Skip,
    /// Store in dead-letter queue for later retry, continue.
    DeadLetter,
    /// Stop remaining steps for this item.
    Abort,
}

/// Outcome of a single pipeline step execution.
#[derive(Debug, Clone)]
pub enum StepOutcome {
    /// Step succeeded normally.
    Completed,
    /// Step was intentionally skipped.
    Skipped {
        /// Why the step was skipped.
        reason: String,
    },
    /// Step failed with a specific error policy.
    Failed {
        /// Error description.
        error: String,
        /// How to handle this failure.
        policy: StepErrorPolicy,
    },
}

/// Aggregated result of running a full step chain on one item.
#[derive(Debug, Clone)]
pub struct ItemResult {
    /// Node ID of the processed item.
    pub node_id: NodeId,
    /// Steps that completed successfully.
    pub steps_succeeded: Vec<String>,
    /// Steps that failed: `(step_name, error_message)`.
    pub steps_failed: Vec<(String, String)>,
    /// Steps that were skipped.
    pub steps_skipped: Vec<String>,
}

impl ItemResult {
    /// Create an empty result for a given node.
    pub fn new(node_id: NodeId) -> Self {
        Self {
            node_id,
            steps_succeeded: Vec::new(),
            steps_failed: Vec::new(),
            steps_skipped: Vec::new(),
        }
    }
}
