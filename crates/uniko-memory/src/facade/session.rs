//! The [`Session`] conversation handle and its [`Turn`] input.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use chrono::{DateTime, Utc};

use uniko_extract::ingest::atomic::ingest_message_atomic;
use uniko_extract::ingest::context::SessionContext;
use uniko_extract::ingest::session_chunk::{
    ChunkMode, chunk_session_observations_with, chunk_session_with,
};
use uniko_extract::ingest::{
    AtomicIngestResult, IngestContext, IngestOutcome, IngestSource, ModalityRegistry, ingest_source,
};
use uniko_pipes::IngestMessage;
use uniko_pipes::types::{ConsolidationTask, IngestTask, ObservationsReady};
use uniko_store::{DeletionReport, KnowledgeBase, NodeId, UnikoError};

use crate::pipeline::PipelineSystem;
use crate::summary::generate_session_summary;

/// A conversation scope that feeds turns into memory.
///
/// Obtain one with [`Agent::session`](crate::Agent::session). A `Session`
/// owns the per-session [`SessionContext`] (turn chain, speaker window,
/// participant cache), so feeding turns through it preserves cross-turn
/// conversational state. A `Session` is single-threaded: feed its turns
/// in order.
///
/// Use [`observe`](Session::observe) for durable, immediately-recallable
/// ingest; [`submit`](Session::submit) + [`flush`](Session::flush) for
/// streaming throughput when the instance was built with
/// [`streaming(true)`](crate::UnikoBuilder::streaming). Use one path or
/// the other per session, not both.
pub struct Session {
    kb: KnowledgeBase,
    ctx: SessionContext,
    streaming: Option<Arc<PipelineSystem>>,
    llm_alias: Option<String>,
    extractors: Arc<ModalityRegistry>,
    /// Agent this session belongs to — the consolidation unit that new
    /// Observations are attributed to.
    agent_id: String,
}

impl fmt::Debug for Session {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // KnowledgeBase is not Debug; surface the session identity only.
        f.debug_struct("Session")
            .field("session_id", &self.ctx.session_id)
            .field("streaming", &self.streaming.is_some())
            .finish_non_exhaustive()
    }
}

impl Session {
    /// Create a session bound to `session_id`.
    ///
    /// The Session node is created lazily on the first
    /// [`observe`](Session::observe), so this performs no I/O.
    pub(crate) fn new(
        kb: KnowledgeBase,
        session_id: impl Into<String>,
        streaming: Option<Arc<PipelineSystem>>,
        llm_alias: Option<String>,
        extractors: Arc<ModalityRegistry>,
        agent_id: impl Into<String>,
    ) -> Self {
        // `session_nid = 0` is the sentinel the ingest path resolves /
        // creates on first sight (see `ensure_session_and_sender`).
        let ctx = SessionContext::new(session_id.into(), 0);
        Self {
            kb,
            ctx,
            streaming,
            llm_alias,
            extractors,
            agent_id: agent_id.into(),
        }
    }

    /// Notify the consolidation worker that `count` Observations landed.
    ///
    /// No-op without streaming (there is no worker to notify — use
    /// [`Agent::consolidate`](crate::Agent::consolidate) instead), and
    /// best-effort when there is: a full channel must never fail an ingest
    /// that already committed.
    fn notify_observations(&self, observations: &[NodeId]) {
        if observations.is_empty() {
            return;
        }
        let Some(pipeline) = self.streaming.as_ref() else {
            return;
        };
        let notice = ConsolidationTask::ObservationsReady(ObservationsReady {
            agent_id: self.agent_id.clone(),
            observation_count: observations.len() as u32,
            source_node_ids: observations.to_vec(),
        });
        if let Err(e) = pipeline.submit_consolidation(notice) {
            tracing::debug!(error = %e, "consolidation notify skipped");
        }
    }

    /// The session's external identifier.
    pub fn session_id(&self) -> &str {
        &self.ctx.session_id
    }

    /// Ingest one turn durably, committing before returning.
    ///
    /// The write is immediately visible to
    /// [`Agent::recall`](crate::Agent::recall) (read-after-write). Runs the
    /// full per-message pipeline — chunking, entity extraction, observation
    /// extraction — in a single transaction.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on any extraction or write failure; on error
    /// no partial state persists for the turn.
    pub async fn observe(&mut self, turn: Turn) -> Result<ObserveResult, UnikoError> {
        // Update speaker / pronoun window before ingest so recipient
        // inference and pronoun resolution see the right context.
        self.ctx.set_current_speaker(&turn.sender_id);
        let session_id = self.ctx.session_id.clone();

        // Resolve the message id up front so attachments can link to it.
        let message_id = turn
            .message_id
            .clone()
            .unwrap_or_else(uniko_store::id::new_id);
        let mut turn = turn;
        turn.message_id = Some(message_id.clone());
        let attachments = std::mem::take(&mut turn.attachments);

        let msg = turn.into_ingest_message(session_id.clone());
        let message = ingest_message_atomic(&self.kb, &msg, &mut self.ctx).await?;

        // Ingest each attachment linked to this message (and session).
        let mut attachment_outcomes = Vec::with_capacity(attachments.len());
        for source in attachments {
            let context = IngestContext {
                session_id: Some(session_id.clone()),
                triggered_by_message_id: Some(message_id.clone()),
            };
            attachment_outcomes
                .push(ingest_source(&self.kb, &self.extractors, source, context).await?);
        }

        // New Observations advance the consolidation counter. Only reaches a
        // worker when streaming is on; otherwise call `Agent::consolidate`.
        self.notify_observations(&message.extracted_observations);

        Ok(ObserveResult {
            message,
            attachments: attachment_outcomes,
        })
    }

    /// Enqueue one turn for asynchronous streaming ingest.
    ///
    /// Returns once the task is accepted by the pipeline (fire-and-forget).
    /// Streamed turns are processed independently and do **not** advance
    /// this session's cross-turn context — use [`observe`](Session::observe)
    /// for conversational fidelity. Await [`flush`](Session::flush) before a
    /// recall that must see streamed turns.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] when the instance was not built with
    /// [`streaming(true)`](crate::UnikoBuilder::streaming), or
    /// [`UnikoError::Pipeline`] when the ingest queue is full.
    pub async fn submit(&self, turn: Turn) -> Result<(), UnikoError> {
        let pipeline = self.require_streaming("submit")?;
        let session_id = self.ctx.session_id.clone();
        let mut msg = turn.into_ingest_message(session_id);
        // Reserved key: the ingest worker reads this to attribute the
        // resulting Observations to an agent when it notifies consolidation.
        msg.metadata.insert(
            "agent_id".to_string(),
            serde_json::Value::String(self.agent_id.clone()),
        );
        pipeline.submit_ingest(IngestTask::Message(msg))
    }

    /// Enqueue a MIME-routed blob ([`IngestSource`]) for streaming ingest.
    ///
    /// The async analogue of [`ingest`](Session::ingest): documents and PDFs
    /// flow through the pipeline; image/audio/video require a registered
    /// extractor (none on the streaming path → [`UnikoError::Unsupported`]
    /// at processing time). Streamed sources are **not** session-linked, the
    /// same caveat as [`submit`](Session::submit). Await
    /// [`flush`](Session::flush) before a recall that must see them.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] when streaming was not enabled, or
    /// [`UnikoError::Pipeline`] when the ingest queue is full.
    pub async fn submit_source(&self, source: IngestSource) -> Result<(), UnikoError> {
        let pipeline = self.require_streaming("submit_source")?;
        pipeline.submit_ingest(IngestTask::Source(source))
    }

    /// Await full processing of everything [`submit`](Session::submit)ted.
    ///
    /// A true barrier: returns only once the ingest queue is drained and no
    /// in-flight task remains, so a following recall sees all streamed
    /// turns.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] when streaming is not enabled.
    pub async fn flush(&self) -> Result<(), UnikoError> {
        let pipeline = self.require_streaming("flush")?;
        pipeline.quiesce().await;
        Ok(())
    }

    /// Ingest a standalone blob through the unified, MIME-routed dispatch.
    ///
    /// Resolves the source's MIME (explicit → magic bytes → file extension →
    /// text) and routes it: text/code/markup/structured/document become an
    /// artifact attached to this session; PDF takes the tiered PDF path;
    /// image/audio/video need a registered modality extractor (registered via
    /// [`UnikoBuilder::extractor`](crate::UnikoBuilder::extractor)) and
    /// otherwise return [`UnikoError::Unsupported`].
    ///
    /// This is the **standalone** blob path (corpus / knowledge-base load).
    /// To attach a document to a conversation turn, use
    /// [`Turn::attach`](Turn::attach) + [`observe`](Session::observe).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on an ingest failure, or
    /// [`UnikoError::Unsupported`] for a modality with no extractor.
    pub async fn ingest(&self, source: IngestSource) -> Result<IngestOutcome, UnikoError> {
        let context = IngestContext {
            session_id: Some(self.ctx.session_id.clone()),
            triggered_by_message_id: None,
        };
        ingest_source(&self.kb, &self.extractors, source, context).await
    }

    /// Build (or refresh) this session's session-level retrieval surfaces.
    ///
    /// Concatenates every turn ingested into this session into a transcript
    /// and chunks it, then aggregates the session's observations into dense
    /// chunks wired `ABOUT` the entities and participants they mention.
    /// These are what session-scoped recall and the Phase 1 session boost
    /// (`phase1_strategy = "boost"`, the default) retrieve — without them a
    /// session contributes only its per-turn chunks.
    ///
    /// Cheap and idempotent when the session has not grown since the last
    /// call: nothing is rewritten and nothing is re-embedded. Awaits
    /// [`flush`](Session::flush) first when streaming is enabled, so
    /// in-flight turns are included. Call it at the end of a conversation,
    /// or periodically during a long-running one.
    ///
    /// [`summarize`](Session::summarize) calls this for you on a
    /// best-effort basis.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on a read or write failure.
    pub async fn finalize(&self) -> Result<FinalizeReport, UnikoError> {
        // Streamed turns land asynchronously, and the transcript read below
        // goes to the graph — without the barrier an in-flight turn is
        // simply missing from the chunks.
        if self.streaming.is_some() {
            self.flush().await?;
        }
        finalize_session(&self.kb, &self.ctx.session_id).await
    }

    /// Generate (or refresh) a synopsis of this session.
    ///
    /// Extractive (deterministic) when no LLM was configured on the
    /// instance, abstractive (LLM-rewritten) when one was. Idempotent on
    /// the session's summary id. Returns the summary node id, or `None`
    /// when the session has no content to summarize.
    ///
    /// Also refreshes the session-level retrieval surfaces
    /// ([`finalize`](Session::finalize)) on a best-effort basis, since a
    /// summary and those chunks derive from the same transcript and this is
    /// the natural end-of-session verb.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on a read, write, or generation failure.
    /// A failure to refresh the chunks is logged, not returned.
    pub async fn summarize(&self) -> Result<Option<NodeId>, UnikoError> {
        // Best-effort: this is post-processing the caller did not ask for,
        // and failing summary generation because a chunk rebuild hit a
        // transient conflict would be a regression for existing callers.
        if let Err(e) = self.finalize().await {
            tracing::warn!(
                session_id = %self.ctx.session_id,
                error = %e,
                "summarize: session chunk refresh failed; continuing with stale chunks",
            );
        }
        generate_session_summary(
            &self.kb,
            &self.ctx.session_id,
            Utc::now(),
            self.llm_alias.as_deref(),
        )
        .await
    }

    /// Soft-forget one turn: hide it from recall, keep the node + lineage.
    ///
    /// Derived Facts/Observations are visibility-redacted; the Message and
    /// its Chunks get a redaction tombstone. Idempotent: an unknown
    /// `message_id` returns a report with `root_existed = false`.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on a write failure.
    pub async fn forget_turn(&self, message_id: &str) -> Result<DeletionReport, UnikoError> {
        self.kb.forget_message(message_id).await
    }

    /// Hard-delete one turn and its owned derivations.
    ///
    /// Cascades the Message's Chunks and Observations, re-evaluates Facts
    /// that lose their last support (soft-invalidating orphans), and
    /// splices the `NEXT` chain closed. Idempotent on an unknown id.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on a traversal or write failure.
    pub async fn delete_turn(&self, message_id: &str) -> Result<DeletionReport, UnikoError> {
        self.kb.delete_message(message_id).await
    }

    /// Hard-delete a document Artifact and its structure subtree.
    ///
    /// Removes the Artifact, its Pages, Blocks, and Chunks. The deduped,
    /// content-addressed `:ArtifactContent` blob is left in place.
    /// Idempotent on an unknown id.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on a traversal or write failure.
    pub async fn delete_document(&self, artifact_id: &str) -> Result<DeletionReport, UnikoError> {
        self.kb.delete_artifact(artifact_id).await
    }

    /// Borrow the streaming pipeline or explain that it is disabled.
    fn require_streaming(&self, method: &str) -> Result<&Arc<PipelineSystem>, UnikoError> {
        self.streaming.as_ref().ok_or_else(|| {
            UnikoError::Config(format!(
                "{method}() requires streaming; build with Uniko::builder().streaming(true)"
            ))
        })
    }
}

/// What [`Session::finalize`] built or refreshed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FinalizeReport {
    /// Transcript chunk node ids (`chunk_type = "session"`).
    pub transcript_chunks: Vec<NodeId>,
    /// Observation chunk node ids (`chunk_type = "observation"`).
    pub observation_chunks: Vec<NodeId>,
    /// `true` when chunks were written, `false` when already current.
    pub rebuilt: bool,
    /// The `ended_at` stamped on the Session — the timestamp of its most
    /// recent message. `None` when the session has no messages.
    ///
    /// A Session counts as *open* while `ended_at` is null, so a finalized
    /// session is skipped by the inactivity auto-close sweep. Finalizing
    /// again after more turns re-stamps it.
    pub ended_at: Option<DateTime<Utc>>,
}

/// Build or refresh the session-level retrieval surfaces for `session_id`.
///
/// Shared by [`Session::finalize`] and
/// [`Agent::finalize_session`](crate::Agent::finalize_session); neither
/// flushes here, so a streaming caller must quiesce first.
///
/// # Errors
///
/// Returns [`UnikoError`] on a read or write failure.
pub(crate) async fn finalize_session(
    kb: &KnowledgeBase,
    session_id: &str,
) -> Result<FinalizeReport, UnikoError> {
    let transcript = chunk_session_with(kb, session_id, ChunkMode::Refresh).await?;
    let observations = chunk_session_observations_with(kb, session_id, ChunkMode::Refresh).await?;
    let ended_at = kb.stamp_session_ended_at(session_id).await?;

    Ok(FinalizeReport {
        transcript_chunks: transcript.ids,
        observation_chunks: observations.ids,
        rebuilt: transcript.rebuilt || observations.rebuilt,
        ended_at,
    })
}

/// What [`Session::observe`] ingested: the message plus any attachments.
#[derive(Debug)]
pub struct ObserveResult {
    /// The ingested conversation message.
    pub message: AtomicIngestResult,
    /// One outcome per [`Turn`] attachment, in attachment order. Each
    /// attachment is linked `Artifact -ATTACHED_TO-> Message`.
    pub attachments: Vec<IngestOutcome>,
}

/// One conversation turn to feed into a [`Session`].
///
/// Construct with [`Turn::new`] (sender + content) and refine with the
/// chainable setters. Maps to a single message ingest; a UUID v7 message
/// id is generated automatically. Attach documents/files shared in the turn
/// with [`attach`](Turn::attach) — they ingest linked to this message.
///
/// # Examples
///
/// ```
/// use uniko_memory::{IngestSource, Turn};
///
/// let turn = Turn::new("alice", "here's the spec we discussed")
///     .addressed_to(vec!["bob".to_string()])
///     .attach(IngestSource::text("# Spec\n\n- requirement one"));
/// # let _ = turn;
/// ```
#[derive(Debug, Clone)]
pub struct Turn {
    message_id: Option<String>,
    sender_id: String,
    content: String,
    content_type: String,
    addressed_to: Option<Vec<String>>,
    timestamp: DateTime<Utc>,
    metadata: HashMap<String, serde_json::Value>,
    attachments: Vec<IngestSource>,
}

impl Turn {
    /// A text turn from `sender_id` carrying `content`, stamped now.
    pub fn new(sender_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            message_id: None,
            sender_id: sender_id.into(),
            content: content.into(),
            content_type: "text".to_string(),
            addressed_to: None,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
            attachments: Vec::new(),
        }
    }

    /// Set an explicit message id for idempotent ingest.
    ///
    /// Ingest is idempotent on `message_id`: re-feeding a turn with the
    /// same id is a no-op rather than a duplicate. When unset, a fresh
    /// UUID v7 is generated per turn.
    #[must_use]
    pub fn id(mut self, message_id: impl Into<String>) -> Self {
        self.message_id = Some(message_id.into());
        self
    }

    /// Override the content type (defaults to `"text"`).
    #[must_use]
    pub fn content_type(mut self, content_type: impl Into<String>) -> Self {
        self.content_type = content_type.into();
        self
    }

    /// Set the send time (defaults to now).
    #[must_use]
    pub fn at(mut self, timestamp: DateTime<Utc>) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Set explicit recipient participant ids.
    ///
    /// When unset, recipients are inferred from session participants.
    #[must_use]
    pub fn addressed_to(mut self, recipients: Vec<String>) -> Self {
        self.addressed_to = Some(recipients);
        self
    }

    /// Attach one arbitrary metadata entry forwarded to ingest.
    #[must_use]
    pub fn metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Attach a document/file shared in this turn.
    ///
    /// On [`observe`](Session::observe) each attachment is ingested and
    /// linked `Artifact -ATTACHED_TO-> Message` (and to the session).
    /// Chainable; attachments ingest in the order added.
    #[must_use]
    pub fn attach(mut self, source: IngestSource) -> Self {
        self.attachments.push(source);
        self
    }

    /// Attach several documents/files at once (see [`attach`](Turn::attach)).
    #[must_use]
    pub fn attachments(mut self, sources: impl IntoIterator<Item = IngestSource>) -> Self {
        self.attachments.extend(sources);
        self
    }

    /// Lower into the wire ingest message for `session_id`.
    fn into_ingest_message(self, session_id: String) -> IngestMessage {
        IngestMessage {
            message_id: self.message_id.unwrap_or_else(uniko_store::id::new_id),
            content: self.content,
            content_type: self.content_type,
            sender_id: self.sender_id,
            session_id,
            addressed_to: self.addressed_to,
            timestamp: self.timestamp,
            metadata: self.metadata,
        }
    }
}
