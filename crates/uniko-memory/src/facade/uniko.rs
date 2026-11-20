//! The owning [`Uniko`] handle and its [`UnikoBuilder`].

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use uniko_extract::ingest::{ModalityExtractor, ModalityRegistry, ModelExtractor};
use uniko_pipes::config::PipelineConfig;
use uniko_pipes::health::PipelineHealth;
use uniko_store::config::{EmbeddingConfig, NlpConfig, RerankerConfig, UnikoConfig};
use uniko_store::repository::policy::Membership;
use uniko_store::xervo::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uniko_store::{
    DeletionReport, KnowledgeBase, ModelEntry, PrefetchPolicy, SharedRuntime, UnikoError,
};

use super::RecallScope;
use crate::Agent;
use crate::pipeline::PipelineSystem;

/// A cognitive-memory instance: the single end-user entry point.
///
/// Owns the underlying store (and, when streaming is enabled, the ingest
/// pipeline) and hands out agent-scoped [`Agent`] views. Opening with
/// [`Uniko::open`] / [`Uniko::in_memory`] uses the validated best-known
/// configuration; use [`Uniko::builder`] to override models, enable an
/// LLM, turn on streaming ingest, or scope visibility.
///
/// Cloning is cheap (the store is `Arc`-backed) and yields another handle
/// to the same instance.
///
/// # Examples
///
/// ```no_run
/// # async fn demo() -> Result<(), uniko_store::UnikoError> {
/// use uniko_memory::Uniko;
///
/// let memory = Uniko::open("./agent-memory").await?;
/// let agent = memory.agent("assistant-1");
/// # let _ = agent;
/// memory.shutdown().await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Uniko {
    pub(super) kb: KnowledgeBase,
    pub(super) llm_alias: Option<String>,
    streaming: Option<Arc<PipelineSystem>>,
    scope: RecallScope,
    pub(super) extractors: Arc<ModalityRegistry>,
    /// Whether readiness checks also verify remote aliases
    /// ([`UnikoBuilder::verify_remote`]).
    pub(super) verify_remote: bool,
}

/// What [`Uniko::inspect_store`] found.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct StoreReport {
    /// What the store was built with.
    pub info: uniko_store::StoreInfo,
    /// How it compares with the config it was checked against.
    pub compatibility: uniko_store::Compatibility,
}

impl fmt::Debug for Uniko {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // KnowledgeBase wraps the live store handle and is not Debug;
        // surface only the facade-level configuration.
        f.debug_struct("Uniko")
            .field("llm_alias", &self.llm_alias)
            .field("streaming", &self.streaming.is_some())
            .field("scope", &self.scope)
            .finish_non_exhaustive()
    }
}

impl Uniko {
    /// Open or create a persistent instance at `path` with best defaults.
    ///
    /// # Errors
    ///
    /// Propagates [`KnowledgeBase::open`] failures (config validation,
    /// store open).
    pub async fn open(path: impl AsRef<Path>) -> Result<Self, UnikoError> {
        let kb = KnowledgeBase::open(path, UnikoConfig::default()).await?;
        crate::rules::register_stdlib_rules(&kb).await?;
        Ok(Self::from_parts(
            kb,
            None,
            None,
            RecallScope::Unrestricted,
            Arc::new(ModalityRegistry::default()),
        ))
    }

    /// Open an ephemeral in-memory instance with best defaults.
    ///
    /// # Errors
    ///
    /// Propagates [`KnowledgeBase::in_memory`] failures.
    pub async fn in_memory() -> Result<Self, UnikoError> {
        let kb = KnowledgeBase::in_memory(UnikoConfig::default()).await?;
        crate::rules::register_stdlib_rules(&kb).await?;
        Ok(Self::from_parts(
            kb,
            None,
            None,
            RecallScope::Unrestricted,
            Arc::new(ModalityRegistry::default()),
        ))
    }

    /// Start a [`UnikoBuilder`] for non-default construction.
    #[must_use]
    pub fn builder() -> UnikoBuilder {
        UnikoBuilder::default()
    }

    /// A view scoped to `agent_id` exposing the agent-facing tools.
    pub fn agent(&self, agent_id: impl Into<String>) -> Agent {
        Agent::with_context(
            self.kb.clone(),
            agent_id,
            self.llm_alias.clone(),
            self.streaming.clone(),
            self.scope.clone(),
            self.extractors.clone(),
        )
    }

    /// The validated configuration this instance runs with.
    pub fn config(&self) -> &UnikoConfig {
        self.kb.config()
    }

    /// Record that `participant_id` is a member of team `team_id`, so it
    /// can see items with [`Visibility::Team`](crate::Visibility::Team)
    /// for that team. Creates the participant and team when absent.
    /// Returns `false` when the membership was already recorded.
    ///
    /// An `Agent` handle resolves its own memberships once, on its first
    /// scoped read, so take a fresh handle from [`agent`](Uniko::agent) to
    /// see a change.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] for an empty id, or [`UnikoError`] on
    /// a write failure.
    pub async fn add_team_member(
        &self,
        participant_id: &str,
        team_id: &str,
    ) -> Result<bool, UnikoError> {
        self.kb
            .add_membership(Membership::ParticipantInTeam, participant_id, team_id)
            .await
    }

    /// Undo [`add_team_member`](Uniko::add_team_member). Returns `false`
    /// when the membership was not recorded.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on a write failure.
    pub async fn remove_team_member(
        &self,
        participant_id: &str,
        team_id: &str,
    ) -> Result<bool, UnikoError> {
        self.kb
            .remove_membership(Membership::ParticipantInTeam, participant_id, team_id)
            .await
    }

    /// Record that `participant_id` is a member of org `org_id`, so it can
    /// see items with [`Visibility::Org`](crate::Visibility::Org) for that
    /// org. Returns `false` when already recorded.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] for an empty id, or [`UnikoError`] on
    /// a write failure.
    pub async fn add_org_member(
        &self,
        participant_id: &str,
        org_id: &str,
    ) -> Result<bool, UnikoError> {
        self.kb
            .add_membership(Membership::ParticipantInOrg, participant_id, org_id)
            .await
    }

    /// Undo [`add_org_member`](Uniko::add_org_member). Returns `false`
    /// when the membership was not recorded.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on a write failure.
    pub async fn remove_org_member(
        &self,
        participant_id: &str,
        org_id: &str,
    ) -> Result<bool, UnikoError> {
        self.kb
            .remove_membership(Membership::ParticipantInOrg, participant_id, org_id)
            .await
    }

    /// Record that team `team_id` belongs to org `org_id`: every member of
    /// the team then sees that org's items too. Returns `false` when
    /// already recorded.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] for an empty id, or [`UnikoError`] on
    /// a write failure.
    pub async fn add_team_to_org(&self, team_id: &str, org_id: &str) -> Result<bool, UnikoError> {
        self.kb
            .add_membership(Membership::TeamInOrg, team_id, org_id)
            .await
    }

    /// Undo [`add_team_to_org`](Uniko::add_team_to_org). Returns `false`
    /// when it was not recorded.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on a write failure.
    pub async fn remove_team_from_org(
        &self,
        team_id: &str,
        org_id: &str,
    ) -> Result<bool, UnikoError> {
        self.kb
            .remove_membership(Membership::TeamInOrg, team_id, org_id)
            .await
    }

    /// A health snapshot of the streaming pipeline: queue depth and
    /// capacity, throughput and latency for the ingest and consolidation
    /// workers, and the LLM circuit-breaker state.
    ///
    /// `None` when the instance was built without
    /// [`streaming`](UnikoBuilder::streaming), since there is no
    /// pipeline to report on.
    pub fn health(&self) -> Option<PipelineHealth> {
        self.streaming.as_ref().map(|pipeline| pipeline.health())
    }

    /// What the store at `path` was built with, and whether `config` can
    /// open it as is.
    ///
    /// Reads the stored schema and model record only: it applies no schema,
    /// changes no index or data and loads no model, so it is safe to call
    /// before deciding how to open a store (uni-db's own open and close may
    /// still write bookkeeping files). Opening runs the same check and
    /// refuses an incompatible config with [`UnikoError::IncompatibleStore`]
    /// before the store changes.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] when the store is open, inspected or
    /// migrated in this process (use [`store_info`](Self::store_info) on an
    /// open instance), [`UnikoError::Unsupported`] for a remote URI, and
    /// [`UnikoError`] when the store can't be read.
    pub async fn inspect_store(
        path: impl AsRef<Path>,
        config: &UnikoConfig,
    ) -> Result<StoreReport, UnikoError> {
        let (info, compatibility) = KnowledgeBase::inspect(path, config).await?;
        Ok(StoreReport {
            info,
            compatibility,
        })
    }

    /// What this instance's store was built with.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when the store can't be read.
    pub async fn store_info(&self) -> Result<uniko_store::StoreInfo, UnikoError> {
        self.kb.store_info().await
    }

    /// Bring the store at `path` in line with `config` in place. See
    /// [`KnowledgeBase::migrate`] for what each [`Migration`](uniko_store::Migration)
    /// does; anything else, such as a new model or dimension size, needs a
    /// fresh store. `progress` is called after each phase and backfill batch.
    /// The store must not be open elsewhere.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::IncompatibleStore`] when the migration can't
    /// resolve the store's differences from `config`, or the error that
    /// stopped it; the work already done is kept, and a rerun resumes.
    pub async fn migrate_store(
        path: impl AsRef<Path>,
        config: UnikoConfig,
        migration: uniko_store::Migration,
        mut progress: impl FnMut(&uniko_store::MigrationProgress) + Send,
    ) -> Result<uniko_store::MigrationReport, UnikoError> {
        KnowledgeBase::migrate(path, config, migration, &mut progress).await
    }

    /// The streaming pipeline's dead letters: list, count, retry and
    /// resolve ingest items that failed. See [`DeadLetters`](crate::DeadLetters).
    #[must_use]
    pub fn dead_letters(&self) -> crate::DeadLetters {
        crate::DeadLetters::new(&self.kb, self.streaming.clone())
    }

    /// Fill `Chunk.search_text` for chunks written before it existed, so
    /// the lexical recall channel and [`Sources::search`](crate::Sources::search)
    /// see them. Run it once after opening a store from an earlier release;
    /// idempotent, and returns how many chunks it filled.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on a read or write failure.
    pub async fn backfill_search_text(&self) -> Result<usize, UnikoError> {
        uniko_extract::ingest::structure::backfill_chunk_search_text(&self.kb, 0).await
    }

    /// Erase the entire graph. Intended for dev/test reset.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on a write failure.
    pub async fn purge(&self) -> Result<DeletionReport, UnikoError> {
        self.kb.purge_all().await
    }

    /// Shut down the pipeline (if any) and then the store.
    ///
    /// Call after all [`Agent`] / [`Session`](super::Session) handles are
    /// dropped: a pipeline still shared by a live clone cannot be drained
    /// here and is skipped with a warning.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Pipeline`] if the pipeline drain times out, or
    /// propagates [`KnowledgeBase::shutdown`] failures.
    pub async fn shutdown(self) -> Result<(), UnikoError> {
        if let Some(pipeline) = self.streaming {
            match Arc::try_unwrap(pipeline) {
                Ok(pipeline) => pipeline.shutdown().await.map_err(UnikoError::Pipeline)?,
                Err(_) => tracing::warn!(
                    "Uniko::shutdown: pipeline still shared by Agent, Session or DeadLetters handles; \
                     skipping pipeline drain (drop those handles first)"
                ),
            }
        }
        self.kb.shutdown().await.map_err(|e| match e {
            // Translate the store's low-level ownership error into
            // actionable guidance — this is a usage issue, not a bug.
            UnikoError::Internal(msg) if msg.contains("outstanding references") => {
                UnikoError::Config(
                    "Uniko::shutdown requires all Agent, Session and DeadLetters handles to be dropped first"
                        .to_string(),
                )
            }
            other => other,
        })
    }

    /// Assemble a [`Uniko`] from already-built parts.
    fn from_parts(
        kb: KnowledgeBase,
        llm_alias: Option<String>,
        streaming: Option<Arc<PipelineSystem>>,
        scope: RecallScope,
        extractors: Arc<ModalityRegistry>,
    ) -> Self {
        Self {
            kb,
            llm_alias,
            streaming,
            scope,
            extractors,
            verify_remote: false,
        }
    }
}

/// Where the store lives — chosen on the [`UnikoBuilder`].
#[derive(Debug, Clone, Default)]
enum Location {
    /// Ephemeral, process-lifetime store.
    #[default]
    InMemory,
    /// Persistent store rooted at this path.
    Path(PathBuf),
}

/// Builder for a non-default [`Uniko`] instance.
///
/// Obtain one with [`Uniko::builder`]. All setters are chainable and the
/// terminal method is [`build`](UnikoBuilder::build).
///
/// # Examples
///
/// ```no_run
/// # async fn demo() -> Result<(), uniko_store::UnikoError> {
/// use uniko_memory::{LlmSpec, Uniko};
/// use uniko_store::config::EmbeddingConfig;
///
/// let memory = Uniko::builder()
///     .path("./agent-memory")
///     .embedding(EmbeddingConfig::nomic_v15())
///     .llm(LlmSpec::openai("generate/answer", "gpt-4o-mini", None))
///     .scope_to_agent()
///     .build()
///     .await?;
/// # let _ = memory;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Default)]
pub struct UnikoBuilder {
    location: Location,
    config: Option<UnikoConfig>,
    embedding: Option<EmbeddingConfig>,
    reranker: Option<RerankerConfig>,
    nlp: Option<NlpConfig>,
    llms: Vec<LlmSpec>,
    answer_alias: Option<String>,
    streaming: bool,
    scope: RecallScope,
    extractors: ModalityRegistry,
    runtime: Option<SharedRuntime>,
    prefetch: PrefetchPolicy,
    requirements: Vec<super::Requirement>,
    readiness_timeout: Option<std::time::Duration>,
    verify_remote: bool,
}

impl UnikoBuilder {
    /// Use a persistent store rooted at `path`.
    #[must_use]
    pub fn path(mut self, path: impl AsRef<Path>) -> Self {
        self.location = Location::Path(path.as_ref().to_path_buf());
        self
    }

    /// Use an ephemeral in-memory store (the default).
    #[must_use]
    pub fn in_memory(mut self) -> Self {
        self.location = Location::InMemory;
        self
    }

    /// Override the embedding model (defaults to BGE-small-en-v1.5).
    #[must_use]
    pub fn embedding(mut self, embedding: EmbeddingConfig) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Override the cross-encoder reranker (defaults to MiniLM, enabled).
    ///
    /// Pass a config with `enabled: false` to turn reranking off. Like
    /// [`embedding`](UnikoBuilder::embedding), this wins over the
    /// `reranker` section of [`raw_config`](UnikoBuilder::raw_config)
    /// whichever is called first.
    #[must_use]
    pub fn reranker(mut self, reranker: RerankerConfig) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Override the NLP model behind NER and observation extraction
    /// (defaults to the kniv v1 cascade). [`NlpConfig::preset`] names the
    /// published ones, every kniv v7 size included. Like
    /// [`embedding`](UnikoBuilder::embedding), this wins over the `nlp`
    /// section of [`raw_config`](UnikoBuilder::raw_config) whichever is
    /// called first.
    #[must_use]
    pub fn nlp(mut self, nlp: NlpConfig) -> Self {
        self.nlp = Some(nlp);
        self
    }

    /// Register an LLM, enabling [`Agent::answer`](crate::Agent::answer).
    ///
    /// Chainable: each call registers one more generation model. The first
    /// one answers unless [`answer_alias`](UnikoBuilder::answer_alias)
    /// names another. Registering the same alias twice fails
    /// [`build`](UnikoBuilder::build).
    #[must_use]
    pub fn llm(mut self, spec: LlmSpec) -> Self {
        self.llms.push(spec);
        self
    }

    /// Choose which generation alias [`Agent::answer`](crate::Agent::answer)
    /// uses.
    ///
    /// The alias may come from [`llm`](UnikoBuilder::llm) or from the
    /// catalog file in [`UnikoConfig::catalog_path`]. Defaults to the first
    /// [`llm`](UnikoBuilder::llm). [`build`](UnikoBuilder::build) fails with
    /// [`UnikoError::Config`] when the alias is not in the catalog or is
    /// not a generation model.
    #[must_use]
    pub fn answer_alias(mut self, alias: impl Into<String>) -> Self {
        self.answer_alias = Some(alias.into());
        self
    }

    /// Replace the entire configuration (escape hatch).
    ///
    /// This is the base config; [`embedding`](UnikoBuilder::embedding) and
    /// [`reranker`](UnikoBuilder::reranker) still override its sections
    /// when set, whatever the call order. Its queue, consolidation and
    /// circuit-breaker fields configure the streaming pipeline.
    #[must_use]
    pub fn raw_config(mut self, config: UnikoConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Enable the asynchronous streaming ingest pipeline.
    ///
    /// When on, [`Session::submit`](super::Session::submit) /
    /// [`Session::flush`](super::Session::flush) become available for
    /// fire-and-forget throughput ingest. The synchronous
    /// [`Session::observe`](super::Session::observe) path works
    /// regardless.
    #[must_use]
    pub fn streaming(mut self, on: bool) -> Self {
        self.streaming = on;
        self
    }

    /// Scope every agent's reads to its own visibility (shorthand for
    /// [`scope`](UnikoBuilder::scope) with [`RecallScope::AsAgent`]).
    #[must_use]
    pub fn scope_to_agent(mut self) -> Self {
        self.scope = RecallScope::AsAgent;
        self
    }

    /// Set the read visibility scope (defaults to
    /// [`RecallScope::Unrestricted`]).
    #[must_use]
    pub fn scope(mut self, scope: RecallScope) -> Self {
        self.scope = scope;
        self
    }

    /// Register a [`ModalityExtractor`] for image/audio/video ingest.
    ///
    /// Without one for a given modality, [`Session::ingest`](super::Session::ingest)
    /// of that modality returns [`UnikoError::Unsupported`]. Chainable;
    /// later registrations replace earlier ones for the same modality.
    #[must_use]
    pub fn extractor(mut self, extractor: Arc<dyn ModalityExtractor>) -> Self {
        self.extractors.register(extractor);
        self
    }

    /// Register a model-backed extractor: OCR, captions, document
    /// extraction or transcription through catalog aliases, for one
    /// modality. Replaces any earlier extractor for that modality.
    ///
    /// [`build`](Self::build) checks it against the catalog before the
    /// store opens: a mode the modality cannot use, a missing alias or one
    /// serving the wrong task, a caption alias whose provider drops images,
    /// and a video extractor without a composer all fail there.
    #[must_use]
    pub fn model_extractor(self, extractor: ModelExtractor) -> Self {
        self.extractor(Arc::new(extractor.into_extractor()))
    }

    /// Open the store over a [`SharedRuntime`] instead of loading its own
    /// models, so several instances load each model once.
    ///
    /// Build the runtime with [`build_runtime`](UnikoBuilder::build_runtime)
    /// from a builder with the same models (embedding, reranker, NLP and
    /// LLMs). [`build`](UnikoBuilder::build) fails with
    /// [`UnikoError::Config`] when the runtime does not serve this
    /// builder's models. [`prefetch`](UnikoBuilder::prefetch) does not
    /// apply: the runtime's own policy already decided that.
    ///
    /// Shutting this instance down leaves the runtime loaded for the
    /// others; it is freed when the last handle and store holding it go.
    #[must_use]
    pub fn runtime(mut self, runtime: SharedRuntime) -> Self {
        self.runtime = Some(runtime);
        self
    }

    /// When to load models: [`PrefetchPolicy::Eager`] (the default) loads
    /// them all before [`build`](UnikoBuilder::build) returns, `Lazy` on
    /// first use, and `Background` on a background task.
    ///
    /// With any policy, a model that fails to load is recorded (see
    /// [`Uniko::capabilities`]) and the others still load.
    #[must_use]
    pub fn prefetch(mut self, policy: PrefetchPolicy) -> Self {
        self.prefetch = policy;
        self
    }

    /// Require models or stages before [`build`](UnikoBuilder::build)
    /// returns. Adds to earlier calls.
    ///
    /// `build` loads each required alias whatever the
    /// [`prefetch`](UnikoBuilder::prefetch) policy, and fails with
    /// [`UnikoError::NotReady`] listing every unmet requirement. A
    /// requirement the catalog, the compiled features or the registered
    /// extractors cannot meet fails before the store is created. Models
    /// that are not required are left to the prefetch policy.
    #[must_use]
    pub fn require(mut self, requirements: impl IntoIterator<Item = super::Requirement>) -> Self {
        self.requirements.extend(requirements);
        self
    }

    /// How long [`build`](UnikoBuilder::build) waits for the required
    /// models to load. A model still loading when it runs out is reported
    /// as `timed_out`. Without a timeout (or with [`Duration::MAX`](std::time::Duration::MAX)),
    /// `build` waits as long as loading takes.
    ///
    /// The timeout covers the required models only. They load first; the
    /// [`prefetch`](UnikoBuilder::prefetch) policy then runs for the rest,
    /// so under the default `Eager` policy `build` still waits for those
    /// afterwards.
    #[must_use]
    pub fn readiness_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.readiness_timeout = Some(timeout);
        self
    }

    /// Also check, when readiness runs, that each required remote alias
    /// can reach its endpoint with a valid credential. Off by default.
    ///
    /// Each required alias that loaded gets one cheap authenticated request
    /// (a model lookup for OpenAI and most providers, `/health` for
    /// llama.cpp), at most eight at a time, within the
    /// [`readiness_timeout`](UnikoBuilder::readiness_timeout). A failure is
    /// reported as `verification_failed`, with the reason (`unauthorized`,
    /// `unreachable`, `not found`, ...) in its `error`. Local models pass
    /// without any request.
    ///
    /// Applies to [`build`](UnikoBuilder::build) and to every later
    /// [`check_readiness`](Uniko::check_readiness) on the instance, so a
    /// health endpoint probing often makes these requests each time.
    /// Without it, nothing contacts a remote provider before its first
    /// request, and a remote alias reads as ready once its client is
    /// configured.
    #[must_use]
    pub fn verify_remote(mut self, verify: bool) -> Self {
        self.verify_remote = verify;
        self
    }

    /// Build a [`SharedRuntime`] for this builder's models (embedding,
    /// reranker, NLP, catalog file and LLMs), loading them as
    /// [`prefetch`](UnikoBuilder::prefetch) says. Pass it to
    /// [`runtime`](UnikoBuilder::runtime) on each builder that should share
    /// it. The store settings (path, streaming, scope, extractors) are not
    /// used here.
    ///
    /// # Errors
    ///
    /// [`UnikoError::Config`] for an invalid config or [`LlmSpec`], as for
    /// [`build`](UnikoBuilder::build).
    pub async fn build_runtime(self) -> Result<SharedRuntime, UnikoError> {
        let prefetch = self.prefetch;
        let answer_alias = self.default_answer_alias();
        let (config, extra_catalog) = self.models().await?;
        let runtime = SharedRuntime::build(&config, extra_catalog, prefetch).await?;
        match answer_alias {
            Some(alias) => runtime.with_answer_alias(alias),
            None => Ok(runtime),
        }
    }

    /// `answer_alias`, else the first `.llm()`.
    fn default_answer_alias(&self) -> Option<String> {
        self.answer_alias
            .clone()
            .or_else(|| self.llms.first().map(|l| l.alias.clone()))
    }

    /// The merged config and the validated LLM catalog entries.
    async fn models(&self) -> Result<(UnikoConfig, Vec<ModelAliasSpec>), UnikoError> {
        let mut config = self.config.clone().unwrap_or_default();
        if let Some(embedding) = &self.embedding {
            config.embedding = embedding.clone();
        }
        if let Some(reranker) = &self.reranker {
            config.reranker = reranker.clone();
        }
        if let Some(nlp) = &self.nlp {
            config.nlp = nlp.clone();
        }
        let mut seen = std::collections::HashSet::new();
        for spec in &self.llms {
            if !seen.insert(spec.alias.as_str()) {
                return Err(UnikoError::Config(format!(
                    "LLM alias '{}' is registered more than once",
                    spec.alias
                )));
            }
            spec.validate().await?;
        }
        let extra_catalog = self
            .llms
            .iter()
            .cloned()
            .map(LlmSpec::into_alias_spec)
            .collect();
        Ok((config, extra_catalog))
    }

    /// Build the [`Uniko`] instance.
    ///
    /// # Errors
    ///
    /// - [`UnikoError::Config`] when an [`LlmSpec`] is invalid (unknown or
    ///   non-generation provider, bad or secret-bearing options, an alias
    ///   registered twice) or the answer alias is missing from the catalog
    ///   or is not a generation model.
    /// - [`UnikoError::NotReady`] when a [`require`](UnikoBuilder::require)d
    ///   model or stage is unavailable.
    /// - Store-open failures (config validation, model warmup).
    pub async fn build(self) -> Result<Uniko, UnikoError> {
        let (config, extra_catalog) = self.models().await?;
        let pipeline_config = PipelineConfig::from_uniko_config(&config);
        // Check the answer alias against the catalog the store will have,
        // before opening it: a typo should not first create the store and
        // load every model.
        let entries = match &self.runtime {
            Some(runtime) => {
                check_llms_served(runtime, &extra_catalog)?;
                runtime.model_entries()
            }
            None => KnowledgeBase::catalog_entries(&config, &extra_catalog)?,
        };
        // Over a shared runtime the LLMs belong to the runtime, so the
        // answer alias recorded when it was built applies by default.
        let llm_alias = self.default_answer_alias().or_else(|| {
            self.runtime
                .as_ref()
                .and_then(|r| r.answer_alias().map(str::to_string))
        });
        if let Some(alias) = &llm_alias {
            check_answer_alias(&entries, alias)?;
        }

        // A model-backed extractor must run on this catalog: refuse a mode
        // it cannot serve before anything is created.
        for extractor in self.extractors.extractors() {
            extractor.validate(&entries)?;
        }
        // Readiness, static step: catalog, features and extractors only,
        // so an unmet requirement fails before the store is created.
        super::readiness::check_static(
            &self.requirements,
            &super::readiness::Context {
                config: &config,
                entries: &entries,
                extractors: &self.extractors,
                answer_alias: llm_alias.as_deref(),
                verify_remote: self.verify_remote,
            },
        )?;
        let requirement_config = (!self.requirements.is_empty()).then(|| config.clone());
        // With requirements, open without prefetching so the readiness
        // timeout covers the required models first; the policy then runs
        // for the rest once they are ready.
        let open_policy = if requirement_config.is_some() {
            PrefetchPolicy::Lazy
        } else {
            self.prefetch
        };

        let mut kb = match (&self.runtime, self.location) {
            (Some(runtime), location) => match location {
                Location::InMemory => KnowledgeBase::in_memory_shared(config, runtime).await?,
                Location::Path(path) => KnowledgeBase::open_shared(path, config, runtime).await?,
            },
            (None, Location::InMemory) => {
                KnowledgeBase::in_memory_with_prefetch(config, extra_catalog, open_policy).await?
            }
            (None, Location::Path(path)) => {
                KnowledgeBase::open_with_prefetch(path, config, extra_catalog, open_policy).await?
            }
        };
        // Readiness, dynamic step: load the required aliases whatever the
        // prefetch policy, and check how they loaded.
        if let Some(config) = &requirement_config {
            let cx = super::readiness::Context {
                config,
                entries: &entries,
                extractors: &self.extractors,
                answer_alias: llm_alias.as_deref(),
                verify_remote: self.verify_remote,
            };
            let failures =
                super::readiness::check(&kb, &self.requirements, &cx, self.readiness_timeout).await;
            if !failures.is_empty() {
                // Release the store (and a persistent path's lock) before
                // reporting. The unmet requirements are what the caller
                // needs, so a shutdown error is only logged.
                if let Err(e) = kb.shutdown().await {
                    tracing::warn!(error = %e, "shutdown after a failed readiness check");
                }
                return Err(UnikoError::NotReady(failures));
            }
            if self.runtime.is_none() {
                kb.start_prefetch(self.prefetch).await;
            }
        }
        crate::rules::register_stdlib_rules(&kb).await?;

        // One registry for both paths: a source queued through the streaming
        // pipeline resolves the same modality extractors as one ingested
        // directly.
        let extractors = Arc::new(self.extractors);

        let streaming = if self.streaming {
            let steps: Vec<Box<dyn uniko_pipes::Step>> = vec![Box::new(
                uniko_extract::ingest::IngestStep::new(extractors.clone()),
            )];
            let pipeline = PipelineSystem::new(pipeline_config, Arc::new(kb.clone()), steps);
            Some(Arc::new(pipeline))
        } else {
            None
        };

        let mut memory = Uniko::from_parts(kb, llm_alias, streaming, self.scope, extractors);
        memory.verify_remote = self.verify_remote;
        Ok(memory)
    }
}

/// Check that a shared runtime serves every LLM the builder registered.
///
/// LLMs belong to the runtime (they are part of its catalog), so a builder
/// using one may repeat them, as long as they match what the runtime holds.
fn check_llms_served(runtime: &SharedRuntime, llms: &[ModelAliasSpec]) -> Result<(), UnikoError> {
    for spec in llms {
        runtime.check_spec(spec).map_err(|e| {
            let detail = match e {
                UnikoError::Config(m) => m,
                other => other.to_string(),
            };
            UnikoError::Config(format!(
                "{detail}; register LLM alias '{}' the same way on the builder that \
                 built the runtime",
                spec.alias
            ))
        })?;
    }
    Ok(())
}

/// Check that the answer alias names a generation model in `entries`.
fn check_answer_alias(entries: &[ModelEntry], alias: &str) -> Result<(), UnikoError> {
    match entries.iter().find(|e| e.alias == alias) {
        None => Err(UnikoError::Config(format!(
            "answer alias '{alias}' is not in the model catalog"
        ))),
        Some(e) if e.task != ModelTask::Generate => Err(UnikoError::Config(format!(
            "answer alias '{alias}' is a {:?} model, not a generation model",
            e.task
        ))),
        Some(_) => Ok(()),
    }
}

/// Option keys that would carry a credential inline. Credentials are named
/// by environment variable instead ([`LlmSpec::api_key_env`]), so the
/// secret never sits in the catalog, logs or `Debug` output.
const SECRET_OPTION_KEYS: &[&str] = &[
    "api_key",
    "apikey",
    "key",
    "token",
    "access_token",
    "bearer_token",
    "password",
    "secret",
    "client_secret",
    "authorization",
];

/// An LLM registration for [`Agent::answer`](crate::Agent::answer).
///
/// Construct via a provider helper ([`LlmSpec::openai`],
/// [`LlmSpec::llamacpp`]) or, for any other compiled generation provider,
/// [`LlmSpec::new`]. The `alias` is the handle the answer path generates
/// under.
///
/// `Debug` shows the alias, provider and model, and only the option *keys*.
#[derive(Clone)]
pub struct LlmSpec {
    alias: String,
    provider_id: String,
    model_id: String,
    options: serde_json::Value,
}

impl fmt::Debug for LlmSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let keys: Vec<&str> = self
            .options
            .as_object()
            .map(|m| m.keys().map(String::as_str).collect())
            .unwrap_or_default();
        f.debug_struct("LlmSpec")
            .field("alias", &self.alias)
            .field("provider_id", &self.provider_id)
            .field("model_id", &self.model_id)
            .field("option_keys", &keys)
            .finish()
    }
}

impl LlmSpec {
    /// A generation model served by any compiled provider, such as
    /// `remote/anthropic`, `remote/mistral` or `remote/gemini`
    /// ([`uniko_store::GENERATION_PROVIDERS`] lists them).
    ///
    /// Add provider options with [`option`](LlmSpec::option) and the
    /// credential's environment variable with
    /// [`api_key_env`](LlmSpec::api_key_env). Nothing is checked here;
    /// [`UnikoBuilder::build`] validates the provider and options before
    /// opening the store.
    pub fn new(
        alias: impl Into<String>,
        provider_id: impl Into<String>,
        model_id: impl Into<String>,
    ) -> Self {
        Self {
            alias: alias.into(),
            provider_id: provider_id.into(),
            model_id: model_id.into(),
            options: serde_json::Value::Object(serde_json::Map::new()),
        }
    }

    /// Set one provider option (e.g. `base_url`, `anthropic_version`).
    ///
    /// Do not pass a credential here: keys such as `api_key` or `token`
    /// are rejected at build. Use [`api_key_env`](LlmSpec::api_key_env).
    #[must_use]
    pub fn option(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        if !self.options.is_object() {
            self.options = serde_json::Value::Object(serde_json::Map::new());
        }
        self.options[key.into()] = value.into();
        self
    }

    /// Read the provider credential from environment variable `var` at
    /// call time.
    #[must_use]
    ///
    /// Sets the provider's credential option: `api_token_env` for
    /// `remote/vertexai`, `api_key_env` for every other provider.
    pub fn api_key_env(self, var: impl Into<String>) -> Self {
        let var: String = var.into();
        let key = if self.provider_id == "remote/vertexai" {
            "api_token_env"
        } else {
            "api_key_env"
        };
        self.option(key, var)
    }

    /// The alias this model registers under.
    #[must_use]
    pub fn alias(&self) -> &str {
        &self.alias
    }

    /// Validate provider, credentials and options without loading.
    async fn validate(&self) -> Result<(), UnikoError> {
        if !uniko_store::GENERATION_PROVIDERS.contains(&self.provider_id.as_str()) {
            return Err(UnikoError::Config(format!(
                "LLM alias '{}': provider '{}' cannot serve generation in this build \
                 (available: {})",
                self.alias,
                self.provider_id,
                uniko_store::GENERATION_PROVIDERS.join(", ")
            )));
        }
        if let Some(map) = self.options.as_object()
            && let Some(key) = map
                .keys()
                .find(|k| SECRET_OPTION_KEYS.contains(&k.to_ascii_lowercase().as_str()))
        {
            return Err(UnikoError::Config(format!(
                "LLM alias '{}': option '{key}' would hold a credential inline; \
                 name its environment variable with api_key_env instead",
                self.alias
            )));
        }
        KnowledgeBase::validate_model_spec(&self.clone().into_alias_spec()).await
    }
    /// An OpenAI-compatible chat model (optionally at a custom `base_url`).
    ///
    /// The API key is read at call time from the `OPENAI_API_KEY`
    /// environment variable; use [`LlmSpec::openai_with_key_env`] for a
    /// different variable.
    pub fn openai(
        alias: impl Into<String>,
        model_id: impl Into<String>,
        base_url: Option<&str>,
    ) -> Self {
        Self::openai_with_key_env(alias, model_id, base_url, "OPENAI_API_KEY")
    }

    /// An OpenAI-compatible chat model reading its key from `key_env`.
    pub fn openai_with_key_env(
        alias: impl Into<String>,
        model_id: impl Into<String>,
        base_url: Option<&str>,
        key_env: impl Into<String>,
    ) -> Self {
        let mut options = serde_json::json!({ "api_key_env": key_env.into() });
        if let Some(url) = base_url {
            options["base_url"] = serde_json::Value::String(url.to_string());
        }
        Self {
            alias: alias.into(),
            provider_id: "remote/openai".to_string(),
            model_id: model_id.into(),
            options,
        }
    }

    /// A chat model served by a llama.cpp
    /// [`llama-server`](https://github.com/ggml-org/llama.cpp).
    ///
    /// `base_url` is the OpenAI-compatible root **including** `/v1`, e.g.
    /// `http://127.0.0.1:8080/v1`. A `llama-server` process holds one
    /// loaded model, so `model_id` is only echoed back in the request
    /// body — pass whatever name the server was started with.
    ///
    /// Inference happens in that server, not in this process: no
    /// inference kernels are compiled in and no CUDA or Metal toolchain
    /// is needed at build time. That makes this the route to
    /// open-weights generation on a build that leaves `candle` off.
    ///
    /// Assumes a server started without `--api-key`; use
    /// [`LlmSpec::llamacpp_with_key_env`] when it requires a bearer
    /// token.
    pub fn llamacpp(
        alias: impl Into<String>,
        model_id: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            alias: alias.into(),
            provider_id: "remote/llamacpp".to_string(),
            model_id: model_id.into(),
            options: serde_json::json!({ "base_url": base_url.into() }),
        }
    }

    /// A `llama-server` chat model whose bearer token is read from
    /// `key_env` at call time.
    ///
    /// Matches a server started with `--api-key`; see
    /// [`LlmSpec::llamacpp`] for the rest.
    pub fn llamacpp_with_key_env(
        alias: impl Into<String>,
        model_id: impl Into<String>,
        base_url: impl Into<String>,
        key_env: impl Into<String>,
    ) -> Self {
        let mut spec = Self::llamacpp(alias, model_id, base_url);
        spec.options["api_key_env"] = serde_json::Value::String(key_env.into());
        spec
    }

    /// Lower into the uni-db catalog entry (kept private to avoid leaking
    /// `ModelAliasSpec` into the public surface).
    fn into_alias_spec(self) -> ModelAliasSpec {
        ModelAliasSpec {
            alias: self.alias,
            task: ModelTask::Generate,
            provider_id: self.provider_id,
            model_id: self.model_id,
            revision: None,
            warmup: WarmupPolicy::Lazy,
            required: false,
            timeout: None,
            load_timeout: None,
            retry: None,
            options: self.options,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn api_key_env_uses_vertexais_token_option() {
        let spec = LlmSpec::new("generate/v", "remote/vertexai", "gemini-x")
            .api_key_env("VERTEX_TOKEN")
            .option("project_id", "p")
            .option("location", "us-central1");
        spec.validate()
            .await
            .expect("vertexai accepts its token option");
        let options = spec.into_alias_spec().options;
        assert_eq!(options["api_token_env"], "VERTEX_TOKEN");
        assert!(options.get("api_key_env").is_none());
    }

    /// `remote/llamacpp` rejects unknown option keys and requires
    /// `base_url`, so the lowered alias must carry exactly that key and
    /// nothing else when no token env is named.
    #[test]
    fn llamacpp_lowers_to_base_url_only() {
        let spec = LlmSpec::llamacpp("gen", "qwen", "http://127.0.0.1:8080/v1").into_alias_spec();

        assert_eq!(spec.provider_id, "remote/llamacpp");
        assert_eq!(spec.model_id, "qwen");
        assert_eq!(spec.task, ModelTask::Generate);
        assert_eq!(
            spec.options,
            serde_json::json!({ "base_url": "http://127.0.0.1:8080/v1" }),
            "an `api_key_env` the caller did not ask for would point the \
             provider at an unset variable"
        );
    }

    #[test]
    fn llamacpp_with_key_env_adds_the_token_var() {
        let spec =
            LlmSpec::llamacpp_with_key_env("gen", "qwen", "http://host:8080/v1", "LLAMA_TOKEN")
                .into_alias_spec();

        assert_eq!(
            spec.options,
            serde_json::json!({
                "base_url": "http://host:8080/v1",
                "api_key_env": "LLAMA_TOKEN",
            })
        );
    }

    #[test]
    fn new_lowers_provider_options_and_key_env() {
        let spec = LlmSpec::new("claude", "remote/anthropic", "claude-x")
            .option("anthropic_version", "2023-06-01")
            .api_key_env("ANTHROPIC_API_KEY")
            .into_alias_spec();

        assert_eq!(spec.provider_id, "remote/anthropic");
        assert_eq!(spec.task, ModelTask::Generate);
        assert_eq!(
            spec.options,
            serde_json::json!({
                "anthropic_version": "2023-06-01",
                "api_key_env": "ANTHROPIC_API_KEY",
            })
        );
    }

    #[test]
    fn debug_shows_option_keys_not_values() {
        let spec = LlmSpec::openai("gen", "gpt", Some("https://user:pw@example.test/v1"));
        let shown = format!("{spec:?}");
        assert!(shown.contains("base_url"), "{shown}");
        assert!(!shown.contains("user:pw"), "{shown}");
    }

    #[tokio::test]
    async fn validate_rejects_inline_credentials() {
        let err = LlmSpec::new("gen/m", "remote/mistral", "mistral-small")
            .option("api_key", "sk-123")
            .validate()
            .await
            .unwrap_err();
        assert!(
            matches!(&err, UnikoError::Config(m) if m.contains("'api_key'") && !m.contains("sk-123")),
            "{err}"
        );
    }

    #[tokio::test]
    async fn validate_rejects_non_generation_providers() {
        for provider in ["remote/voyageai", "local/onnx", "remote/nope"] {
            let err = LlmSpec::new("gen/m", provider, "x")
                .validate()
                .await
                .unwrap_err();
            assert!(
                matches!(&err, UnikoError::Config(m) if m.contains(provider) && m.contains("'gen/m'")),
                "{err}"
            );
        }
    }

    #[tokio::test]
    async fn validate_accepts_a_well_formed_provider_spec() {
        LlmSpec::new("gen/m", "remote/mistral", "mistral-small")
            .api_key_env("MISTRAL_API_KEY")
            .validate()
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn validate_maps_bad_options_to_config() {
        let err = LlmSpec::new("gen/m", "remote/anthropic", "claude-x")
            .option("temperature_typo", 1)
            .validate()
            .await
            .unwrap_err();
        assert!(
            matches!(&err, UnikoError::Config(m) if m.contains("'gen/m'") && m.contains("temperature_typo")),
            "{err}"
        );
    }
}
