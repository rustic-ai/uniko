//! The `Step` trait — cornerstone of all pipeline processing.
//!
//! Every pipeline step (NER, observation extraction, embedding, etc.)
//! implements [`Step`].  Steps are composed into chains and executed
//! per-item with error isolation.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use uniko_store::{KnowledgeBase, NodeId, UnikoError};

use crate::circuit_breaker::CircuitBreaker;
use crate::types::StepOutcome;

/// A single processing step in a pipeline.
///
/// Steps are stateless and reusable across items.  Each item gets its
/// own [`PipelineContext`] for per-item isolation.
///
/// # Implementors
///
/// Return [`StepOutcome::Completed`] on success, or
/// [`StepOutcome::Failed`] with the appropriate [`StepErrorPolicy`].
// `async_trait` puts a bare `#[must_use]` on each bodiless method, whose boxed
// future is already `#[must_use]`; see `uniko_store::blob_store::BlobStore`.
#[allow(clippy::double_must_use)]
#[async_trait]
pub trait Step: Send + Sync {
    /// Human-readable name for logging and metrics.
    fn name(&self) -> &str;

    /// Whether this step should execute for the given context.
    ///
    /// Returns `false` to skip (e.g. code NER skips non-code content).
    /// Defaults to always-run; override to gate on content type or
    /// other context state.
    fn should_run(&self, _ctx: &PipelineContext) -> bool {
        true
    }

    /// Execute the step, mutating context with results.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on unrecoverable failure.  For expected
    /// failures, prefer returning [`StepOutcome::Failed`] with the
    /// appropriate policy instead.
    async fn execute(&self, ctx: &mut PipelineContext) -> Result<StepOutcome, UnikoError>;
}

/// Mutable context passed through the step chain for a single item.
///
/// Each item gets its own context — no shared mutable state between
/// items.
#[derive(Debug)]
pub struct PipelineContext {
    /// Internal node ID of the item being processed.
    pub node_id: NodeId,
    /// Raw content of the item.
    pub content: String,
    /// Content type (e.g. `"text"`, `"code"`, `"image"`).
    pub content_type: String,
    /// Per-item cancellation token (child of worker token).
    pub cancel: CancellationToken,
    /// Graph storage access.
    pub kb: Arc<KnowledgeBase>,
    /// LLM circuit breaker — check before making LLM calls.
    pub llm_breaker: Arc<CircuitBreaker>,
    /// `(node_id, name)` pairs for entities extracted so far
    /// (populated by NER step). The name is the canonical-cased label
    /// matching `Entity.name`, so downstream steps can do substring
    /// matching against speaker / observation subjects without a
    /// per-message round-trip to fetch the property.
    pub extracted_entities: Vec<(NodeId, String)>,
    /// `(participant_node_id, participant_name)` for the message's
    /// sender, populated by the ingest step that creates the
    /// `SENT_BY` edge. The observation step uses this to wire ABOUT
    /// edges from observations back to the speaker without
    /// re-querying the SENT_BY edge and refetching the Participant
    /// node. `None` when the ingest path didn't populate it (e.g.
    /// unit tests that build a context manually); downstream
    /// consumers fall back to a DB lookup in that case.
    pub sender: Option<(NodeId, String)>,
    /// Observation node IDs extracted so far.
    pub extracted_observations: Vec<NodeId>,
    /// Arbitrary step-to-step metadata.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl PipelineContext {
    /// Create a new context for processing one item.
    pub fn new(
        node_id: NodeId,
        content: String,
        content_type: String,
        cancel: CancellationToken,
        kb: Arc<KnowledgeBase>,
        llm_breaker: Arc<CircuitBreaker>,
    ) -> Self {
        Self {
            node_id,
            content,
            content_type,
            cancel,
            kb,
            llm_breaker,
            extracted_entities: Vec::new(),
            extracted_observations: Vec::new(),
            sender: None,
            metadata: HashMap::new(),
        }
    }
}
