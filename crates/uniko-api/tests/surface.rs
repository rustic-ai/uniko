//! Surface guard for the public facade.
//!
//! The point of `uniko-api` is that an end user touches intent types only.
//! This test pins the *positive* surface (the facade types must stay
//! reachable); the `compile_fail` doctests on `uniko_api::tools` pin the
//! *negative* surface (internals must stay hidden).

/// Compile-time proof the facade entry points are exported through the
/// `tools` namespace.
#[test]
fn facade_entry_points_are_exported() {
    fn assert_exists<T>() {}

    assert_exists::<uniko_api::tools::Uniko>();
    assert_exists::<uniko_api::tools::UnikoBuilder>();
    assert_exists::<uniko_api::tools::Session>();
    assert_exists::<uniko_api::tools::Turn>();
    assert_exists::<uniko_api::tools::ObserveResult>();
    assert_exists::<uniko_api::tools::Agent>();
    assert_exists::<uniko_api::tools::LlmSpec>();
    assert_exists::<uniko_api::tools::RecallScope>();
    assert_exists::<uniko_api::tools::ContextBundle>();
    assert_exists::<uniko_api::tools::RecallOverrides>();
    assert_exists::<uniko_api::tools::RerankerConfig>();
    assert_exists::<uniko_api::tools::PipelineHealth>();
    assert_exists::<uniko_api::tools::WorkerHealth>();
    assert_exists::<uniko_api::tools::WorkerStatus>();
    assert_exists::<uniko_api::tools::CircuitState>();
    assert_exists::<uniko_api::tools::TurnUnit>();
    assert_exists::<uniko_api::tools::UnitResult>();
    assert_exists::<uniko_api::tools::Answer>();
    // Recall item provenance: typed kind + source lineage.
    assert_exists::<uniko_api::tools::RecallItem>();
    assert_exists::<uniko_api::tools::RecallKind>();
    assert_exists::<uniko_api::tools::RecallSource>();
    assert_exists::<uniko_api::tools::RecallTier>();
    // Addressed-retrieval views (agent.data()).
    assert_exists::<uniko_api::tools::MessageView>();
    assert_exists::<uniko_api::tools::ArtifactView>();
    // Goal/task lifecycle surface (agent.goals()).
    assert_exists::<uniko_api::tools::GoalView>();
    assert_exists::<uniko_api::tools::TaskView>();
    assert_exists::<uniko_api::tools::GoalPhase>();
    assert_exists::<uniko_api::tools::TaskPhase>();
    assert_exists::<uniko_api::tools::GoalContext>();
    assert_exists::<uniko_api::tools::CreateGoalParams>();
    assert_exists::<uniko_api::tools::CreateTaskParams>();
    assert_exists::<uniko_api::tools::AtomicIngestResult>();
    assert_exists::<uniko_api::tools::ArtifactIngestResult>();
    assert_exists::<uniko_api::tools::PdfIngestResult>();
    // Logic / query surface types.
    assert_exists::<uniko_api::tools::Record>();
    assert_exists::<uniko_api::tools::Value>();
    assert_exists::<uniko_api::tools::AbductionResult>();
    assert_exists::<uniko_api::tools::AssumeBuilder<'static>>();
    // Error / id types must be nameable for handling and storing results.
    assert_exists::<uniko_api::tools::UnikoError>();
    assert_exists::<uniko_api::tools::NodeId>();
    // Per-call scoping surface.
    assert_exists::<uniko_api::tools::Scope>();
    assert_exists::<uniko_api::tools::Dimensions>();
    // Delete / forget surface.
    assert_exists::<uniko_api::tools::DeletionReport>();
    // Content-type taxonomy + unified ingest surface.
    assert_exists::<uniko_api::tools::Mime>();
    assert_exists::<uniko_api::tools::Modality>();
    assert_exists::<uniko_api::tools::ContentType>();
    assert_exists::<uniko_api::tools::IngestSource>();
    assert_exists::<uniko_api::tools::IngestData>();
    assert_exists::<uniko_api::tools::IngestOutcome>();
    assert_exists::<uniko_api::tools::IngestContext>();
    assert_exists::<uniko_api::tools::ModalityRegistry>();
}
