//! # uniko-memory — Layer 4: Memory Management
//!
//! PipelineSystem orchestration, recall cascade (3-phase with coverage gating),
//! consolidation (fact derivation, contradiction, drift), and stdlib rules.
//!
//! Depends on `uniko-cortex`, `uniko-extract`, `uniko-pipes`, and
//! `uniko-store`. The cortex dependency is deliberate even though cortex
//! is "Layer 5": layer numbers describe cognitive altitude, not build
//! order. Consolidation (P4) is the heartbeat — the consolidation worker
//! triggers cortex sweeps (P5 procedure promotion + P6 topic detection)
//! after successful cycles, because the trigger policy (per-agent cycle
//! counters, min-interval throttling) is keyed off consolidation success
//! and lives in the worker loop. See `pipeline::consolidation_worker`.
//!
//! Revisit this direction only if cortex ever needs uniko-memory APIs at
//! runtime (e.g. MCTS planning calling recall) — that would create a true
//! cycle, and the fix is to invert: define a sweep trait here and inject
//! cortex's implementation at the composition root.

// Bumped from the default 128 to clear E0275 overflow when checking
// `Send` on the `consolidation_worker.run()` future in
// `pipeline::mod::PipelineSystem::start`. The future captures
// uni-db's `Session`/`Transaction`, which transitively contain
// datafusion + sqlparser AST types whose `Vec<FunctionArg>` /
// `Vec<ObjectNamePart>` / etc. trait-checking chain hits the default
// limit. 256 is enough; uni-db's own crates set the same.
#![recursion_limit = "256"]

pub mod action;
pub mod agent;
pub mod consolidation;
pub mod episode;
pub mod facade;
pub mod fact;
pub mod goal;
pub mod llm_triples;
pub mod nl_to_cypher;
pub mod observation;
pub mod pipeline;
pub mod policy;
pub mod query;
pub mod recall;
pub mod rules;
pub mod summary;
pub mod task;
pub(crate) mod value_convert;

#[doc(inline)]
pub use action::{RecordActionParams, RecordActionResult, record_action};
#[doc(inline)]
pub use agent::Agent;
#[doc(inline)]
pub use episode::{RecordEpisodeParams, record_episode};
#[doc(inline)]
pub use facade::{
    ArtifactView, Data, GoalContext, GoalPhase, GoalView, Goals, LlmSpec, MessageView,
    ObserveResult, RecallScope, Session, TaskPhase, TaskView, Turn, Uniko, UnikoBuilder,
};
#[doc(inline)]
pub use fact::{AssertFactParams, InvalidateFactParams, assert_fact, invalidate_fact};
#[doc(inline)]
pub use goal::{CreateGoalParams, create_goal};
#[doc(inline)]
pub use policy::Viewer;
#[doc(inline)]
pub use recall::{
    ContextBundle, Dimensions, RecallConfig, RecallItem, RecallKind, RecallSource, RecallTier,
    Scope, ViewerScope,
};
// Method return / field types surfaced so the facade exposes no
// un-nameable types: `Agent::assert_fact` -> FactUpsert, the `Session`
// ingest verbs -> *IngestResult.
#[doc(no_inline)]
pub use uniko_extract::ingest::{
    ArtifactIngestResult, AtomicIngestResult, IngestContext, IngestData, IngestOutcome,
    IngestSource, ModalityExtractor, ModalityRegistry, PdfIngestResult, ingest_source,
    resolve_mime,
};
// Content-type taxonomy shared by ingest routing and recall channels.
#[doc(no_inline)]
pub use uniko_pipes::content::{ContentType, Mime, Modality};
#[doc(no_inline)]
pub use uniko_store::operations::facts::FactUpsert;
// The canonical error type and node-id alias that pervade every facade
// method signature, surfaced so callers handle errors / name ids without
// reaching into `uniko_store`.
#[doc(no_inline)]
pub use uniko_store::{DeletionReport, NodeId, UnikoError};
// Logic / query surface: `Agent::query` / `run_rule` / `abduce` return
// these; `Value` is the graph value type for query params and rows.
#[doc(no_inline)]
pub use uniko_store::Value;
#[doc(no_inline)]
pub use uniko_store::locy::{
    AbducedModification, AbductionResult, AssumeBuilder, DerivationNode, DerivationTree, Record,
};
// Decoding companion for the `Value::Temporal` arm: uni-db 3.2.0 dropped the
// `TemporalValue::epoch_millis` accessor, so the conversion now lives in
// `uniko_store` and is surfaced here for the bindings, which render temporal
// rows as Python datetimes and must not re-derive it.
#[doc(no_inline)]
pub use uniko_store::temporal_epoch_millis;
// Config presets surfaced so end users touch only the facade crate.
#[doc(inline)]
pub use observation::{AddObservationParams, add_observation};
#[doc(inline)]
pub use pipeline::PipelineSystem;
#[doc(inline)]
pub use query::{
    Answer, GeneratedAnswer, QueryRecordOptions, RecordQueryEpisodeParams, answer_query,
    record_query_episode,
};
#[doc(inline)]
pub use summary::generate_session_summary;
#[doc(inline)]
pub use task::{CreateTaskParams, create_task};
#[doc(no_inline)]
pub use uniko_store::config::{EmbeddingConfig, UnikoConfig};

/// Sort `items` in descending order by the `f64` key returned by `score`.
///
/// `NaN` scores are treated as `Equal` so the sort is total. Used across
/// the recall cascade; centralised here so the
/// `partial_cmp(...).unwrap_or(Equal)` boilerplate lives in exactly one place.
pub(crate) fn sort_by_score_desc<T>(items: &mut [T], score: impl Fn(&T) -> f64) {
    items.sort_by(|a, b| {
        score(b)
            .partial_cmp(&score(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}
