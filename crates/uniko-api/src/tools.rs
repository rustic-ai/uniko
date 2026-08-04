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
    AbducedModification, AbductionResult, Agent, Answer, ArtifactIngestResult, ArtifactView,
    AssumeBuilder, AtomicIngestResult, ContentType, ContextBundle, CreateGoalParams,
    CreateTaskParams, DeletionReport, DerivationNode, DerivationTree, Dimensions, EmbeddingConfig,
    GoalContext, GoalPhase, GoalView, IngestContext, IngestData, IngestOutcome, IngestSource,
    LlmSpec, MessageView, Mime, Modality, ModalityExtractor, ModalityRegistry, NodeId,
    ObserveResult, PdfIngestResult, RecallItem, RecallKind, RecallScope, RecallSource, RecallTier,
    Record, Scope, Session, TaskPhase, TaskView, Turn, Uniko, UnikoBuilder, UnikoConfig,
    UnikoError, Value, ViewerScope, ingest_source,
    nl_to_cypher::is_safe_read_only,
    policy::{Viewer, visibility_admits},
    resolve_mime, temporal_epoch_millis,
};
