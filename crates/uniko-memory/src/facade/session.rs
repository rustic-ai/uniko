//! The [`Session`] conversation handle and its [`Turn`] input.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use chrono::{DateTime, Utc};

use uniko_extract::ingest::context::SessionContext;
use uniko_extract::ingest::session_chunk::{
    ChunkMode, chunk_session_observations_with, chunk_session_with,
};
use uniko_extract::ingest::{
    AtomicIngestResult, IngestContext, IngestOutcome, IngestSource, ModalityRegistry, UnitTurn,
    ingest_source, ingest_turns_atomic, prepare_source,
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
        // One turn IS a one-turn unit. Sharing the path means the rollback,
        // speaker-ordering and attachment-atomicity semantics cannot drift
        // between the two entry points — a second implementation is where
        // the next bug would live.
        //
        // Two behaviour changes fall out of this, both of them what #40
        // asks for: attachments now commit WITH the message rather than in
        // separate transactions afterwards, and an idempotent replay now
        // advances the chain head, so the following turn still gets its
        // NEXT edge.
        let mut result = self.commit_unit(vec![turn]).await?;
        Ok(result
            .turns
            .pop()
            .expect("commit_unit returns one result per input turn"))
    }

    /// Begin a multi-turn unit: several related turns recorded as ONE
    /// durable, idempotent write.
    ///
    /// Every turn added — and every attachment on them — lands in a single
    /// transaction. Either the whole unit is visible to later recall, or
    /// none of it is. That is the guarantee two separate [`observe`] calls
    /// cannot give: an interruption between them leaves a question with no
    /// answer, indistinguishable from a real one.
    ///
    /// [`observe`]: Self::observe
    ///
    /// ```no_run
    /// # async fn demo(session: &mut uniko_memory::Session) -> Result<(), uniko_store::UnikoError> {
    /// use uniko_memory::Turn;
    /// session
    ///     .unit()
    ///     .turn(Turn::new("alice", "what's the plan?").id("m-1"))
    ///     .turn(Turn::new("bob", "ship it friday").id("m-2"))
    ///     .commit()
    ///     .await?;
    /// # Ok(()) }
    /// ```
    pub fn unit(&mut self) -> TurnUnit<'_> {
        TurnUnit {
            session: self,
            turns: Vec::new(),
        }
    }

    /// Commit `turns` as one atomic unit.
    ///
    /// The owned form [`TurnUnit::commit`] delegates to. Use it directly
    /// when the turns are built elsewhere — notably the Python bridge,
    /// which holds the `Session` behind an `Arc<Mutex<_>>` and so can call
    /// a method but cannot lend out a borrow.
    ///
    /// Re-committing a unit whose `message_id`s are all present with
    /// identical content is a no-op ([`UnitResult::was_replay`]). Reusing
    /// an id with different content, or submitting a unit that is only
    /// *partly* recorded, is rejected with
    /// [`UnikoError::IdConflict`](uniko_store::UnikoError::IdConflict).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on any extraction or write failure. On error
    /// **nothing** from the unit persists.
    pub async fn commit_unit(&mut self, turns: Vec<Turn>) -> Result<UnitResult, UnikoError> {
        let session_id = self.ctx.session_id.clone();

        // Resolve every message id and prepare every attachment BEFORE the
        // transaction opens: blob PUTs, chunking and model inference must
        // never be re-paid on a retry, and the unit needs each attachment's
        // content ids to take its striped locks before opening the tx.
        let mut unit_turns = Vec::with_capacity(turns.len());
        for mut turn in turns {
            let message_id = turn
                .message_id
                .clone()
                .unwrap_or_else(uniko_store::id::new_id);
            turn.message_id = Some(message_id.clone());
            let attachments = std::mem::take(&mut turn.attachments);

            let context = IngestContext {
                session_id: Some(session_id.clone()),
                triggered_by_message_id: Some(message_id),
            };
            let mut prepared = Vec::with_capacity(attachments.len());
            for source in attachments {
                prepared.push(prepare_source(&self.kb, &self.extractors, source, &context).await?);
            }

            unit_turns.push(UnitTurn {
                message: turn.into_ingest_message(session_id.clone()),
                attachments: prepared,
            });
        }

        let result = ingest_turns_atomic(&self.kb, &unit_turns, &mut self.ctx).await?;

        // One atomic batch is one consolidation notice.
        let all_observations: Vec<uniko_store::NodeId> = result
            .turns
            .iter()
            .flat_map(|t| t.extracted_observations.iter().copied())
            .collect();
        self.notify_observations(&all_observations);

        let mut attachments = result.attachments;
        Ok(UnitResult {
            turns: result
                .turns
                .into_iter()
                .enumerate()
                .map(|(i, message)| ObserveResult {
                    message,
                    attachments: std::mem::take(attachments.get_mut(i).unwrap_or(&mut Vec::new())),
                })
                .collect(),
            was_replay: result.was_replay,
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
    /// A failure to refresh the chunks does not fail the call; it is
    /// reported in [`SummarizeReport::finalize_error`].
    pub async fn summarize(&self) -> Result<SummarizeReport, UnikoError> {
        // Still best-effort: this is post-processing the caller did not ask
        // for, and failing summary generation because a chunk rebuild hit a
        // transient conflict would be a regression for existing callers. But
        // the outcome is now REPORTED rather than only logged — issue #40
        // requires finalization success or failure to be observable, and a
        // warn! in someone else's log is not.
        let (finalize, finalize_error) = match self.finalize().await {
            Ok(report) => (Some(report), None),
            Err(e) => {
                tracing::warn!(
                    session_id = %self.ctx.session_id,
                    error = %e,
                    "summarize: session chunk refresh failed; continuing with stale chunks",
                );
                (None, Some(e.to_string()))
            }
        };
        let summary = generate_session_summary(
            &self.kb,
            &self.ctx.session_id,
            Utc::now(),
            self.llm_alias.as_deref(),
        )
        .await?;

        Ok(SummarizeReport {
            summary,
            finalize,
            finalize_error,
        })
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
    category: Option<String>,
    source_id: Option<String>,
    revision_id: Option<String>,
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
            category: None,
            source_id: None,
            revision_id: None,
        }
    }

    /// Set an explicit message id for idempotent ingest.
    ///
    /// Ingest is idempotent on `message_id`: re-feeding a turn with the
    /// same id **and the same content** is a no-op rather than a
    /// duplicate. Reusing an id for *different* content is rejected with
    /// [`UnikoError::IdConflict`](uniko_store::UnikoError::IdConflict) —
    /// the id already names a different turn, and accepting it silently
    /// would drop the new one. When unset, a fresh UUID v7 is generated
    /// per turn.
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

    /// Tag this turn with the caller's own record category (issue #39).
    ///
    /// Typed provenance: a recall scope can filter on it, and it never
    /// enters the searchable text. That is the point — the alternative is
    /// encoding a class into the prose and parsing it back out of results,
    /// which makes a coverage score describe candidates the consumer never
    /// received.
    #[must_use]
    pub fn category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// Attribute this turn to a stable logical source id (issue #39).
    ///
    /// Materialised as a `:Source` node with a `FROM_SOURCE` edge, and
    /// denormalised onto the message and its chunks so the recall filter is
    /// a property predicate rather than a traversal.
    #[must_use]
    pub fn source(mut self, source_id: impl Into<String>) -> Self {
        self.source_id = Some(source_id.into());
        self
    }

    /// Declare which revision of that source this turn reflects (issue #41).
    ///
    /// A new revision of the same source supersedes the previous one, which
    /// then stops grounding current answers while staying attributable to
    /// the results it did ground.
    #[must_use]
    pub fn revision(mut self, revision_id: impl Into<String>) -> Self {
        self.revision_id = Some(revision_id.into());
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
            category: self.category,
            source_id: self.source_id,
            revision_id: self.revision_id,
        }
    }
}

/// Accumulates turns for one atomic commit.
///
/// Borrows the [`Session`] mutably for the builder's lifetime, so the
/// compiler prevents using the session mid-build, and `commit` consumes the
/// builder to regain unique access. That borrow is also why this type is not
/// `'static`: bindings that hold a `Session` behind a lock call
/// [`Session::commit_unit`] instead.
#[derive(Debug)]
pub struct TurnUnit<'s> {
    session: &'s mut Session,
    turns: Vec<Turn>,
}

impl TurnUnit<'_> {
    /// Add a turn to the unit.
    #[must_use]
    pub fn turn(mut self, turn: Turn) -> Self {
        self.turns.push(turn);
        self
    }

    /// Add several turns, in order.
    #[must_use]
    pub fn turns(mut self, turns: impl IntoIterator<Item = Turn>) -> Self {
        self.turns.extend(turns);
        self
    }

    /// Write every accumulated turn, and their attachments, in ONE
    /// transaction.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on any extraction or write failure; on error
    /// nothing from the unit persists. See [`Session::commit_unit`] for the
    /// idempotency and conflict rules.
    pub async fn commit(self) -> Result<UnitResult, UnikoError> {
        let Self { session, turns } = self;
        session.commit_unit(turns).await
    }
}

/// What one [`TurnUnit::commit`] recorded.
#[derive(Debug)]
pub struct UnitResult {
    /// One per turn, in unit order — the same shape [`Session::observe`]
    /// returns.
    pub turns: Vec<ObserveResult>,
    /// True when the unit was already recorded verbatim, so nothing was
    /// written and no transaction was opened.
    pub was_replay: bool,
}

impl UnitResult {
    /// The message node ids this unit recorded, in unit order.
    #[must_use]
    pub fn message_node_ids(&self) -> Vec<NodeId> {
        self.turns
            .iter()
            .map(|t| t.message.message_node_id)
            .collect()
    }
}

/// What one [`Session::summarize`] did.
///
/// Carries the chunk-refresh outcome alongside the summary so a caller can
/// see that finalization failed, rather than that fact existing only as a
/// `warn!` in a log the caller may not read (issue #40).
#[derive(Debug)]
pub struct SummarizeReport {
    /// The generated `:Summary` node, or `None` when there was nothing to
    /// summarize.
    pub summary: Option<NodeId>,
    /// The chunk refresh, when it succeeded.
    pub finalize: Option<FinalizeReport>,
    /// Why the chunk refresh failed, when it did. The summary was still
    /// generated, but from stale chunks.
    pub finalize_error: Option<String>,
}

impl SummarizeReport {
    /// True when the chunk refresh succeeded, so the summary was built from
    /// current chunks.
    #[must_use]
    pub fn finalized(&self) -> bool {
        self.finalize_error.is_none()
    }
}
