// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Industries Inc.

//! Typed `#[pyclass]` wrappers for values that cross the boundary outward.
//!
//! Each wrapper is hand-written with `#[pyo3(get)]` fields, a `__repr__`, and a
//! `from_rust` constructor that walks the Rust value field by field. This is
//! the "premium" path — a first-class SDK shape rather than a dict-based RPC
//! shim. Wrappers are `frozen`: outputs are immutable snapshots.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use uniko_api::tools::{
    AbductionResult, Answer, ArtifactView, AtomicIngestResult, ContextBundle, CycleStats,
    DeletionReport, FinalizeReport, GoalContext, GoalPhase, GoalView, IngestOutcome, MessageView,
    ObserveResult, RecallItem, RecallKind, RecallSource, TaskPhase, TaskView,
};

use crate::convert::{json_to_py, utc_datetime_to_py};

/// Map a [`GoalPhase`] to its lowercase wire string.
fn goal_phase_str(phase: GoalPhase) -> &'static str {
    match phase {
        GoalPhase::Planned => "planned",
        GoalPhase::Active => "active",
        GoalPhase::Completed => "completed",
        GoalPhase::Abandoned => "abandoned",
    }
}

/// Map a [`TaskPhase`] to its lowercase wire string.
fn task_phase_str(phase: TaskPhase) -> &'static str {
    match phase {
        TaskPhase::Planned => "planned",
        TaskPhase::Active => "active",
        TaskPhase::Completed => "completed",
        TaskPhase::Blocked => "blocked",
    }
}

/// Build an optional tz-aware UTC `datetime`, or `None`.
fn opt_datetime(
    py: Python<'_>,
    dt: Option<chrono::DateTime<chrono::Utc>>,
) -> PyResult<Option<Py<PyAny>>> {
    match dt {
        Some(dt) => Ok(Some(utc_datetime_to_py(py, dt)?.unbind())),
        None => Ok(None),
    }
}

/// Map a [`RecallKind`] to its lowercase wire string.
fn recall_kind_str(kind: RecallKind) -> &'static str {
    match kind {
        RecallKind::Chunk => "chunk",
        RecallKind::Observation => "observation",
        RecallKind::Fact => "fact",
        RecallKind::Procedure => "procedure",
        RecallKind::Topic => "topic",
        RecallKind::Episode => "episode",
        RecallKind::Message => "message",
        RecallKind::Other => "other",
    }
}

/// Lineage of a recalled item: where its content originated.
#[pyclass(name = "RecallSource", module = "uniko", frozen)]
pub struct PyRecallSource {
    /// `"message"`, `"attachment"`, or `"document"`.
    #[pyo3(get)]
    kind: String,
    /// The message this traces to, when applicable.
    #[pyo3(get)]
    message_id: Option<String>,
    /// The document/artifact this traces to, when applicable.
    #[pyo3(get)]
    artifact_id: Option<String>,
    /// The specific evidence chunk, when content-bearing.
    #[pyo3(get)]
    chunk_id: Option<String>,
}

impl PyRecallSource {
    /// Build the Python wrapper from a Rust [`RecallSource`].
    pub fn from_rust(py: Python<'_>, src: &RecallSource) -> PyResult<Py<Self>> {
        let wrapper = match src {
            RecallSource::Message {
                message_id,
                chunk_id,
            } => Self {
                kind: "message".to_string(),
                message_id: Some(message_id.clone()),
                artifact_id: None,
                chunk_id: chunk_id.clone(),
            },
            RecallSource::Attachment {
                message_id,
                artifact_id,
                chunk_id,
            } => Self {
                kind: "attachment".to_string(),
                message_id: Some(message_id.clone()),
                artifact_id: Some(artifact_id.clone()),
                chunk_id: chunk_id.clone(),
            },
            RecallSource::Document {
                artifact_id,
                chunk_id,
            } => Self {
                kind: "document".to_string(),
                message_id: None,
                artifact_id: Some(artifact_id.clone()),
                chunk_id: chunk_id.clone(),
            },
        };
        Py::new(py, wrapper)
    }
}

#[pymethods]
impl PyRecallSource {
    fn __repr__(&self) -> String {
        format!(
            "RecallSource(kind='{}', message_id={:?}, artifact_id={:?}, chunk_id={:?})",
            self.kind, self.message_id, self.artifact_id, self.chunk_id
        )
    }
}

/// A single recalled item: content plus its kind, score, and lineage.
#[pyclass(name = "RecallItem", module = "uniko", frozen)]
pub struct PyRecallItem {
    /// Graph node id.
    #[pyo3(get)]
    node_id: i64,
    /// What this item is: `chunk`/`observation`/`fact`/`procedure`/`topic`/
    /// `episode`/`message`/`other`.
    #[pyo3(get)]
    kind: String,
    /// Fused score after RRF and tier weighting.
    #[pyo3(get)]
    score: f64,
    /// Display text.
    #[pyo3(get)]
    content: String,
    /// Lineage — the messages/attachments this content came from.
    #[pyo3(get)]
    sources: Vec<Py<PyRecallSource>>,
}

impl PyRecallItem {
    /// Build the Python wrapper from a Rust [`RecallItem`].
    pub fn from_rust(py: Python<'_>, item: &RecallItem) -> PyResult<Py<Self>> {
        let sources = item
            .sources
            .iter()
            .map(|s| PyRecallSource::from_rust(py, s))
            .collect::<PyResult<Vec<_>>>()?;
        Py::new(
            py,
            Self {
                node_id: item.node_id,
                kind: recall_kind_str(item.kind).to_string(),
                score: item.score,
                content: item.content.clone(),
                sources,
            },
        )
    }
}

#[pymethods]
impl PyRecallItem {
    fn __repr__(&self) -> String {
        let preview: String = self.content.chars().take(48).collect();
        format!(
            "RecallItem(node_id={}, kind='{}', score={:.4}, content={preview:?})",
            self.node_id, self.kind, self.score
        )
    }
}

/// A ranked bundle of recalled context.
#[pyclass(name = "ContextBundle", module = "uniko", frozen)]
pub struct PyContextBundle {
    /// Ranked items.
    #[pyo3(get)]
    items: Vec<Py<PyRecallItem>>,
    /// Estimated total tokens across all items.
    #[pyo3(get)]
    total_tokens: usize,
    /// Whether Compact (Phase 1 of the cascade) alone satisfied coverage.
    #[pyo3(get)]
    phase1_only: bool,
    /// Whether the cascade exited after Expand (Phase 2).
    #[pyo3(get)]
    phase2_only: bool,
    /// Coverage score in `[0.0, 1.0]`.
    #[pyo3(get)]
    coverage: f64,
}

impl PyContextBundle {
    /// Build the Python wrapper from a Rust [`ContextBundle`].
    pub fn from_rust(py: Python<'_>, bundle: &ContextBundle) -> PyResult<Py<Self>> {
        let items = bundle
            .items
            .iter()
            .map(|i| PyRecallItem::from_rust(py, i))
            .collect::<PyResult<Vec<_>>>()?;
        Py::new(
            py,
            Self {
                items,
                total_tokens: bundle.total_tokens,
                phase1_only: bundle.phase1_only,
                phase2_only: bundle.phase2_only,
                coverage: bundle.coverage,
            },
        )
    }
}

#[pymethods]
impl PyContextBundle {
    fn __len__(&self) -> usize {
        self.items.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "ContextBundle(items={}, total_tokens={}, coverage={:.3})",
            self.items.len(),
            self.total_tokens,
            self.coverage
        )
    }
}

/// A generated answer plus the context it was grounded in.
#[pyclass(name = "Answer", module = "uniko", frozen)]
pub struct PyAnswer {
    /// Final answer text.
    #[pyo3(get)]
    text: String,
    /// Model id the generator used, when reported.
    #[pyo3(get)]
    model: Option<String>,
    /// Provider-reported prompt tokens, when available.
    #[pyo3(get)]
    input_tokens: Option<u64>,
    /// Provider-reported completion tokens, when available.
    #[pyo3(get)]
    output_tokens: Option<u64>,
    /// `Some(episode_id)` when query recording was opted into and succeeded.
    #[pyo3(get)]
    recorded_episode: Option<i64>,
    /// The recall bundle the answer was grounded in.
    #[pyo3(get)]
    context: Py<PyContextBundle>,
    /// Deduped grounding sources, precomputed from the bundle.
    citations: Vec<Py<PyRecallSource>>,
}

impl PyAnswer {
    /// Build the Python wrapper from a Rust [`Answer`].
    pub fn from_rust(py: Python<'_>, answer: &Answer) -> PyResult<Py<Self>> {
        let context = PyContextBundle::from_rust(py, &answer.context)?;
        let citations = answer
            .citations()
            .into_iter()
            .map(|s| PyRecallSource::from_rust(py, s))
            .collect::<PyResult<Vec<_>>>()?;
        Py::new(
            py,
            Self {
                text: answer.text.clone(),
                model: answer.model.clone(),
                input_tokens: answer.input_tokens,
                output_tokens: answer.output_tokens,
                recorded_episode: answer.recorded_episode,
                context,
                citations,
            },
        )
    }
}

#[pymethods]
impl PyAnswer {
    /// The deduped sources this answer was grounded in, for citations.
    fn citations(&self, py: Python<'_>) -> Vec<Py<PyRecallSource>> {
        self.citations.iter().map(|c| c.clone_ref(py)).collect()
    }

    fn __repr__(&self) -> String {
        let preview: String = self.text.chars().take(64).collect();
        format!("Answer(text={preview:?}, model={:?})", self.model)
    }
}

/// What [`Session::observe`](uniko_api::tools::Session) ingested.
///
/// Flattens the inner `AtomicIngestResult` (message + extraction node ids) and
/// reports the attachment count. The full per-attachment `IngestOutcome`
/// surface arrives in a later phase.
#[pyclass(name = "ObserveResult", module = "uniko", frozen)]
pub struct PyObserveResult {
    /// Node id of the ingested message.
    #[pyo3(get)]
    message_node_id: i64,
    /// Node ids of the message's content chunks.
    #[pyo3(get)]
    chunk_node_ids: Vec<i64>,
    /// Node id of the (created or resolved) session.
    #[pyo3(get)]
    session_node_id: i64,
    /// Node id of the resolved sender participant, when known.
    #[pyo3(get)]
    sender_node_id: Option<i64>,
    /// External id of the resolved sender participant, when known.
    #[pyo3(get)]
    sender_id: Option<String>,
    /// `(node_id, name)` of each entity extracted from the message.
    #[pyo3(get)]
    extracted_entities: Vec<(i64, String)>,
    /// Node ids of each observation extracted from the message.
    #[pyo3(get)]
    extracted_observations: Vec<i64>,
    /// Number of attachments ingested alongside the message.
    #[pyo3(get)]
    attachment_count: usize,
}

impl PyObserveResult {
    /// Build the Python wrapper from a Rust [`ObserveResult`].
    pub fn from_rust(py: Python<'_>, result: &ObserveResult) -> PyResult<Py<Self>> {
        let m: &AtomicIngestResult = &result.message;
        let (sender_node_id, sender_id) = match &m.sender {
            Some((nid, name)) => (Some(*nid), Some(name.clone())),
            None => (None, None),
        };
        Py::new(
            py,
            Self {
                message_node_id: m.message_node_id,
                chunk_node_ids: m.chunk_node_ids.clone(),
                session_node_id: m.session_node_id,
                sender_node_id,
                sender_id,
                extracted_entities: m.extracted_entities.clone(),
                extracted_observations: m.extracted_observations.clone(),
                attachment_count: result.attachments.len(),
            },
        )
    }
}

#[pymethods]
impl PyObserveResult {
    fn __repr__(&self) -> String {
        format!(
            "ObserveResult(message_node_id={}, chunks={}, entities={}, observations={}, attachments={})",
            self.message_node_id,
            self.chunk_node_ids.len(),
            self.extracted_entities.len(),
            self.extracted_observations.len(),
            self.attachment_count
        )
    }
}

/// Tally of what a deletion / redaction removed from the graph.
#[pyclass(name = "DeletionReport", module = "uniko", frozen)]
pub struct PyDeletionReport {
    /// Nodes hard-deleted.
    #[pyo3(get)]
    nodes_deleted: u64,
    /// Edges hard-deleted.
    #[pyo3(get)]
    edges_deleted: u64,
    /// Facts soft-invalidated after losing their last support.
    #[pyo3(get)]
    facts_invalidated: u64,
    /// Turn chains repaired around deleted messages.
    #[pyo3(get)]
    chains_repaired: u64,
    /// Nodes redacted (content cleared) rather than removed.
    #[pyo3(get)]
    nodes_redacted: u64,
    /// Whether the deletion root existed at all.
    #[pyo3(get)]
    root_existed: bool,
}

impl PyDeletionReport {
    /// Build the Python wrapper from a Rust [`DeletionReport`].
    pub fn from_rust(py: Python<'_>, report: &DeletionReport) -> PyResult<Py<Self>> {
        Py::new(
            py,
            Self {
                nodes_deleted: report.nodes_deleted,
                edges_deleted: report.edges_deleted,
                facts_invalidated: report.facts_invalidated,
                chains_repaired: report.chains_repaired,
                nodes_redacted: report.nodes_redacted,
                root_existed: report.root_existed,
            },
        )
    }
}

#[pymethods]
impl PyDeletionReport {
    fn __repr__(&self) -> String {
        format!(
            "DeletionReport(nodes_deleted={}, edges_deleted={}, facts_invalidated={}, root_existed={})",
            self.nodes_deleted, self.edges_deleted, self.facts_invalidated, self.root_existed
        )
    }
}

/// What `Session.finalize()` built or refreshed.
#[pyclass(name = "FinalizeReport", module = "uniko", frozen)]
pub struct PyFinalizeReport {
    /// Transcript chunk node ids (`chunk_type = "session"`).
    #[pyo3(get)]
    transcript_chunks: Vec<i64>,
    /// Observation chunk node ids (`chunk_type = "observation"`).
    #[pyo3(get)]
    observation_chunks: Vec<i64>,
    /// Whether this call wrote to the graph.
    #[pyo3(get)]
    rebuilt: bool,
    /// `ended_at` stamped on the Session (its latest message time), or
    /// `None` when the session has no messages.
    #[pyo3(get)]
    ended_at: Option<Py<PyAny>>,
}

impl PyFinalizeReport {
    /// Build the Python wrapper from a Rust [`FinalizeReport`].
    pub fn from_rust(py: Python<'_>, report: &FinalizeReport) -> PyResult<Py<Self>> {
        Py::new(
            py,
            Self {
                transcript_chunks: report.transcript_chunks.clone(),
                observation_chunks: report.observation_chunks.clone(),
                rebuilt: report.rebuilt,
                ended_at: opt_datetime(py, report.ended_at)?,
            },
        )
    }
}

#[pymethods]
impl PyFinalizeReport {
    fn __repr__(&self) -> String {
        format!(
            "FinalizeReport(transcript_chunks={}, observation_chunks={}, rebuilt={})",
            self.transcript_chunks.len(),
            self.observation_chunks.len(),
            self.rebuilt
        )
    }
}

/// What one consolidation cycle derived.
#[pyclass(name = "CycleStats", module = "uniko", frozen)]
pub struct PyCycleStats {
    /// Observations processed (PROCESSED edges emitted).
    #[pyo3(get)]
    observations_processed: usize,
    /// Facts newly created.
    #[pyo3(get)]
    facts_created: usize,
    /// Facts reinforced rather than created.
    #[pyo3(get)]
    facts_reinforced: usize,
    /// Facts whose validity interval was closed by contradiction detection.
    #[pyo3(get)]
    facts_invalidated: usize,
    /// Entities that flipped to `unstable` (drift alerts).
    #[pyo3(get)]
    drift_alerts: usize,
}

impl PyCycleStats {
    /// Build the Python wrapper from a Rust [`CycleStats`].
    pub fn from_rust(py: Python<'_>, stats: &CycleStats) -> PyResult<Py<Self>> {
        Py::new(
            py,
            Self {
                observations_processed: stats.observations_processed,
                facts_created: stats.facts_created,
                facts_reinforced: stats.facts_reinforced,
                facts_invalidated: stats.facts_invalidated,
                drift_alerts: stats.drift_alerts,
            },
        )
    }
}

#[pymethods]
impl PyCycleStats {
    fn __repr__(&self) -> String {
        format!(
            "CycleStats(observations_processed={}, facts_created={}, facts_reinforced={}, facts_invalidated={}, drift_alerts={})",
            self.observations_processed,
            self.facts_created,
            self.facts_reinforced,
            self.facts_invalidated,
            self.drift_alerts
        )
    }
}

/// A retrieved conversation message.
#[pyclass(name = "MessageView", module = "uniko", frozen)]
pub struct PyMessageView {
    /// External message id.
    #[pyo3(get)]
    message_id: String,
    /// External id of the sending participant.
    #[pyo3(get)]
    sender_id: String,
    /// The message text.
    #[pyo3(get)]
    content: String,
    /// When the message was recorded (tz-aware UTC `datetime`).
    #[pyo3(get)]
    timestamp: Py<PyAny>,
    /// External id of the session the message belongs to.
    #[pyo3(get)]
    session_id: String,
    /// External ids of the addressed recipients.
    #[pyo3(get)]
    addressed_to: Vec<String>,
    /// External ids of artifacts attached to this turn.
    #[pyo3(get)]
    attachments: Vec<String>,
}

impl PyMessageView {
    /// Build the Python wrapper from a Rust [`MessageView`].
    pub fn from_rust(py: Python<'_>, view: &MessageView) -> PyResult<Py<Self>> {
        Py::new(
            py,
            Self {
                message_id: view.message_id.clone(),
                sender_id: view.sender_id.clone(),
                content: view.content.clone(),
                timestamp: utc_datetime_to_py(py, view.timestamp)?.unbind(),
                session_id: view.session_id.clone(),
                addressed_to: view.addressed_to.clone(),
                attachments: view.attachments.clone(),
            },
        )
    }
}

#[pymethods]
impl PyMessageView {
    fn __repr__(&self) -> String {
        format!(
            "MessageView(message_id='{}', sender_id='{}')",
            self.message_id, self.sender_id
        )
    }
}

/// A retrieved artifact (attachment or standalone document).
#[pyclass(name = "ArtifactView", module = "uniko", frozen)]
pub struct PyArtifactView {
    /// External artifact id.
    #[pyo3(get)]
    artifact_id: String,
    /// Artifact kind (`text` / `markdown` / `pdf` / ...).
    #[pyo3(get)]
    kind: String,
    /// MIME type, when known.
    #[pyo3(get)]
    mime: Option<String>,
    /// Source path, when ingested from a file.
    #[pyo3(get)]
    path: Option<String>,
    /// Text reassembled from the artifact's chunks (in order).
    #[pyo3(get)]
    text: String,
    /// The message this artifact was attached to, when shared in a turn.
    #[pyo3(get)]
    attached_to_message: Option<String>,
}

impl PyArtifactView {
    /// Build the Python wrapper from a Rust [`ArtifactView`].
    pub fn from_rust(py: Python<'_>, view: &ArtifactView) -> PyResult<Py<Self>> {
        Py::new(
            py,
            Self {
                artifact_id: view.artifact_id.clone(),
                kind: view.kind.clone(),
                mime: view.mime.clone(),
                path: view.path.clone(),
                text: view.text.clone(),
                attached_to_message: view.attached_to_message.clone(),
            },
        )
    }
}

#[pymethods]
impl PyArtifactView {
    fn __repr__(&self) -> String {
        format!(
            "ArtifactView(artifact_id='{}', kind='{}')",
            self.artifact_id, self.kind
        )
    }
}

/// A goal with its lifecycle phase and recorded result.
#[pyclass(name = "GoalView", module = "uniko", frozen)]
pub struct PyGoalView {
    /// External goal id.
    #[pyo3(get)]
    goal_id: String,
    /// Objective statement.
    #[pyo3(get)]
    title: String,
    /// Optional detail.
    #[pyo3(get)]
    description: Option<String>,
    /// Raw status string (free-form; the source of `phase`).
    #[pyo3(get)]
    status: String,
    /// Lifecycle phase: `planned`/`active`/`completed`/`abandoned`.
    #[pyo3(get)]
    phase: String,
    /// When the goal was created (tz-aware UTC `datetime`, or `None`).
    #[pyo3(get)]
    created_at: Option<Py<PyAny>>,
    /// Target completion date, when set.
    #[pyo3(get)]
    deadline: Option<Py<PyAny>>,
    /// When the goal was completed, when set.
    #[pyo3(get)]
    completed_at: Option<Py<PyAny>>,
    /// Success criteria and/or recorded result (free-form JSON).
    #[pyo3(get)]
    metrics: Option<Py<PyAny>>,
}

impl PyGoalView {
    /// Build the Python wrapper from a Rust [`GoalView`].
    pub fn from_rust(py: Python<'_>, view: &GoalView) -> PyResult<Py<Self>> {
        let metrics = match &view.metrics {
            Some(json) => Some(json_to_py(py, json)?),
            None => None,
        };
        Py::new(
            py,
            Self {
                goal_id: view.goal_id.clone(),
                title: view.title.clone(),
                description: view.description.clone(),
                status: view.status.clone(),
                phase: goal_phase_str(view.phase).to_string(),
                created_at: opt_datetime(py, view.created_at)?,
                deadline: opt_datetime(py, view.deadline)?,
                completed_at: opt_datetime(py, view.completed_at)?,
                metrics,
            },
        )
    }
}

#[pymethods]
impl PyGoalView {
    fn __repr__(&self) -> String {
        format!(
            "GoalView(goal_id='{}', title={:?}, phase='{}')",
            self.goal_id, self.title, self.phase
        )
    }
}

/// A task with its lifecycle phase.
#[pyclass(name = "TaskView", module = "uniko", frozen)]
pub struct PyTaskView {
    /// External task id.
    #[pyo3(get)]
    task_id: String,
    /// Work description.
    #[pyo3(get)]
    title: String,
    /// Optional detail.
    #[pyo3(get)]
    description: Option<String>,
    /// Raw status string.
    #[pyo3(get)]
    status: String,
    /// Lifecycle phase: `planned`/`active`/`completed`/`blocked`.
    #[pyo3(get)]
    phase: String,
    /// Numeric priority in `[0.0, 1.0]`, when set.
    #[pyo3(get)]
    priority: Option<f64>,
    /// The goal this task is part of, when linked.
    #[pyo3(get)]
    goal_id: Option<String>,
    /// When the task was created.
    #[pyo3(get)]
    created_at: Option<Py<PyAny>>,
    /// When the task was completed, when set.
    #[pyo3(get)]
    completed_at: Option<Py<PyAny>>,
}

impl PyTaskView {
    /// Build the Python wrapper from a Rust [`TaskView`].
    pub fn from_rust(py: Python<'_>, view: &TaskView) -> PyResult<Py<Self>> {
        Py::new(
            py,
            Self {
                task_id: view.task_id.clone(),
                title: view.title.clone(),
                description: view.description.clone(),
                status: view.status.clone(),
                phase: task_phase_str(view.phase).to_string(),
                priority: view.priority,
                goal_id: view.goal_id.clone(),
                created_at: opt_datetime(py, view.created_at)?,
                completed_at: opt_datetime(py, view.completed_at)?,
            },
        )
    }
}

#[pymethods]
impl PyTaskView {
    fn __repr__(&self) -> String {
        format!(
            "TaskView(task_id='{}', title={:?}, phase='{}')",
            self.task_id, self.title, self.phase
        )
    }
}

/// A goal plus its working context.
#[pyclass(name = "GoalContext", module = "uniko", frozen)]
pub struct PyGoalContext {
    /// The goal itself.
    #[pyo3(get)]
    goal: Py<PyGoalView>,
    /// Tasks under the goal.
    #[pyo3(get)]
    tasks: Vec<Py<PyTaskView>>,
    /// Sessions scoped to the goal (display text).
    #[pyo3(get)]
    sessions: Vec<String>,
    /// Recent messages in those sessions (newest first).
    #[pyo3(get)]
    recent_messages: Vec<String>,
    /// Facts about entities mentioned while pursuing the goal.
    #[pyo3(get)]
    facts: Vec<String>,
    /// Entities mentioned while pursuing the goal.
    #[pyo3(get)]
    entities: Vec<String>,
}

impl PyGoalContext {
    /// Build the Python wrapper from a Rust [`GoalContext`].
    pub fn from_rust(py: Python<'_>, ctx: &GoalContext) -> PyResult<Py<Self>> {
        let tasks = ctx
            .tasks
            .iter()
            .map(|t| PyTaskView::from_rust(py, t))
            .collect::<PyResult<Vec<_>>>()?;
        Py::new(
            py,
            Self {
                goal: PyGoalView::from_rust(py, &ctx.goal)?,
                tasks,
                sessions: ctx.sessions.clone(),
                recent_messages: ctx.recent_messages.clone(),
                facts: ctx.facts.clone(),
                entities: ctx.entities.clone(),
            },
        )
    }
}

#[pymethods]
impl PyGoalContext {
    fn __repr__(&self) -> String {
        format!(
            "GoalContext(tasks={}, facts={})",
            self.tasks.len(),
            self.facts.len()
        )
    }
}

/// What a unified `Session.ingest` produced.
#[pyclass(name = "IngestOutcome", module = "uniko", frozen)]
pub struct PyIngestOutcome {
    /// `"artifact"` or `"pdf"`.
    #[pyo3(get)]
    kind: String,
    /// External artifact id — pass to `data.artifact(id)` to fetch it back.
    #[pyo3(get)]
    artifact_id: String,
    /// Internal node id of the artifact row.
    #[pyo3(get)]
    artifact_node_id: i64,
    /// Node ids of the created chunks.
    #[pyo3(get)]
    chunk_node_ids: Vec<i64>,
    /// Whether the artifact already existed (skipped chunking).
    #[pyo3(get)]
    was_deduplicated: bool,
    /// Page count, for PDFs only.
    #[pyo3(get)]
    page_count: Option<u32>,
    /// Whether PDF extraction failed (artifact still persisted), PDFs only.
    #[pyo3(get)]
    extraction_failed: Option<bool>,
}

impl PyIngestOutcome {
    /// Build the Python wrapper from a Rust [`IngestOutcome`].
    pub fn from_rust(py: Python<'_>, outcome: &IngestOutcome) -> PyResult<Py<Self>> {
        let wrapper = match outcome {
            IngestOutcome::Artifact(a) => Self {
                kind: "artifact".to_string(),
                artifact_id: a.artifact_id.clone(),
                artifact_node_id: a.artifact_node_id,
                chunk_node_ids: a.chunk_node_ids.clone(),
                was_deduplicated: a.was_deduplicated,
                page_count: None,
                extraction_failed: None,
            },
            IngestOutcome::Pdf(p) => Self {
                kind: "pdf".to_string(),
                artifact_id: p.artifact_id.clone(),
                artifact_node_id: p.artifact_node_id,
                chunk_node_ids: p.chunk_node_ids.clone(),
                was_deduplicated: p.was_deduplicated,
                page_count: Some(p.page_count),
                extraction_failed: Some(p.extraction_failure.is_some()),
            },
        };
        Py::new(py, wrapper)
    }
}

#[pymethods]
impl PyIngestOutcome {
    fn __repr__(&self) -> String {
        format!(
            "IngestOutcome(kind='{}', artifact_id='{}', chunks={}, deduplicated={})",
            self.kind,
            self.artifact_id,
            self.chunk_node_ids.len(),
            self.was_deduplicated
        )
    }
}

/// The result of an abductive query: ranked, validated graph modifications.
#[pyclass(name = "AbductionResult", module = "uniko", frozen)]
pub struct PyAbductionResult {
    /// Proposed modifications, each a dict
    /// `{modification: <dict|str>, validated: bool, cost: float}`.
    #[pyo3(get)]
    modifications: Py<PyAny>,
}

impl PyAbductionResult {
    /// Build the Python wrapper from a Rust [`AbductionResult`].
    pub fn from_rust(py: Python<'_>, result: &AbductionResult) -> PyResult<Py<Self>> {
        let mods = PyList::empty(py);
        for m in &result.modifications {
            let dict = PyDict::new(py);
            dict.set_item("modification", json_to_py(py, &m.modification)?)?;
            dict.set_item("validated", m.validated)?;
            dict.set_item("cost", m.cost)?;
            mods.append(dict)?;
        }
        Py::new(
            py,
            Self {
                modifications: mods.into_any().unbind(),
            },
        )
    }
}

#[pymethods]
impl PyAbductionResult {
    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        self.modifications.bind(py).len()
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let count = self.modifications.bind(py).len().unwrap_or(0);
        format!("AbductionResult(modifications={count})")
    }
}

/// Register the output pyclasses on the module.
///
/// # Errors
///
/// Returns a `PyErr` if adding any class fails.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRecallSource>()?;
    m.add_class::<PyRecallItem>()?;
    m.add_class::<PyContextBundle>()?;
    m.add_class::<PyAnswer>()?;
    m.add_class::<PyObserveResult>()?;
    m.add_class::<PyDeletionReport>()?;
    m.add_class::<PyFinalizeReport>()?;
    m.add_class::<PyCycleStats>()?;
    m.add_class::<PyMessageView>()?;
    m.add_class::<PyArtifactView>()?;
    m.add_class::<PyGoalView>()?;
    m.add_class::<PyTaskView>()?;
    m.add_class::<PyGoalContext>()?;
    m.add_class::<PyIngestOutcome>()?;
    m.add_class::<PyAbductionResult>()?;
    Ok(())
}
