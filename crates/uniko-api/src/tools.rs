//! The lean developer API surface.
//!
//! End users touch the [`Uniko`] facade and its intent types only. Engine
//! internals are intentionally **not** re-exported here — these must not
//! compile:
//!
//! ```compile_fail
//! let _: uniko_api::tools::KnowledgeBase;
//! ```
//! ```compile_fail
//! let _: uniko_api::tools::PipelineSystem;
//! ```
//! ```compile_fail
//! let _: uniko_api::tools::IngestMessage;
//! ```
//!
//! Removed/redundant surface stays gone — the older typed ingest types are
//! subsumed by [`IngestSource`], and the cognition cluster lives at the
//! `uniko_memory` crate root, not the facade:
//!
//! ```compile_fail
//! let _: uniko_api::tools::Document;
//! ```
//! ```compile_fail
//! let _: uniko_api::tools::PdfSource;
//! ```
//! ```compile_fail
//! let _ = uniko_api::tools::create_goal;
//! ```
//! ```compile_fail
//! let _ = uniko_api::tools::working_memory;
//! ```
//!
//! The answer type is [`Answer`] (with `citations()`); the older
//! `QueryOutcome` / `GeneratedAnswer` names are not part of the surface:
//!
//! ```compile_fail
//! let _: uniko_api::tools::QueryOutcome;
//! ```
//! ```compile_fail
//! let _: uniko_api::tools::GeneratedAnswer;
//! ```

// Engine internals (`KnowledgeBase`, `PipelineSystem`, `IngestMessage`,
// uni-db types) are deliberately NOT re-exported — see the `compile_fail`
// guards in this module's docs. Subjective-state operations are surfaced
// as `Agent` / `Session` methods rather than as free functions taking a
// `&KnowledgeBase`, so the public surface never exposes the store handle.
pub use uniko_memory::{
    AbducedModification, AbductionResult, AddObservationParams, Agent, Answer,
    ArtifactIngestResult, ArtifactView, AssertFactParams, AssumeBuilder, AtomicIngestResult,
    CircuitState, ConsolidationReport, ContentType, ContextBundle, CreateGoalParams,
    CreateTaskParams, CycleStats, DeletionReport, Dimensions, EmbeddingConfig, FactUpsert,
    FinalizeReport, GoalContext, GoalPhase, GoalView, IngestContext, IngestData, IngestOutcome,
    IngestSource, InvalidateFactParams, LlmSpec, MessageView, Mime, Modality, ModalityExtractor,
    ModalityPrepared, ModalityRegistry, NodeId, ObserveResult, PdfIngestResult, PipelineHealth,
    PromotionReport, RecallItem, RecallKind, RecallOverrides, RecallScope, RecallSource,
    RecallTier, Record, RecordActionParams, RecordActionResult, RecordEpisodeParams,
    RerankerConfig, RuleView, Scope, Session, SummarizeReport, TaskPhase, TaskView, TopicReport,
    Turn, TurnUnit, Uniko, UnikoBuilder, UnikoConfig, UnikoError, UnitResult, Value, Viewer,
    ViewerScope, Visibility, WorkerHealth, WorkerStatus, fact_id_for, ingest_source,
    nl_to_cypher::is_safe_read_only, policy::visibility_admits, resolve_mime,
    temporal_epoch_millis,
};
