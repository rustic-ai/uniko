//! The owning [`Uniko`] handle and its [`UnikoBuilder`].

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use uniko_extract::ingest::{ModalityExtractor, ModalityRegistry};
use uniko_pipes::config::PipelineConfig;
use uniko_pipes::health::PipelineHealth;
use uniko_store::config::{EmbeddingConfig, RerankerConfig, UnikoConfig};
use uniko_store::repository::policy::Membership;
use uniko_store::xervo::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uniko_store::{DeletionReport, KnowledgeBase, UnikoError};

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
    kb: KnowledgeBase,
    llm_alias: Option<String>,
    streaming: Option<Arc<PipelineSystem>>,
    scope: RecallScope,
    extractors: Arc<ModalityRegistry>,
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
                    "Uniko::shutdown: pipeline still shared by Agent/Session handles; \
                     skipping pipeline drain (drop those handles first)"
                ),
            }
        }
        self.kb.shutdown().await.map_err(|e| match e {
            // Translate the store's low-level ownership error into
            // actionable guidance — this is a usage issue, not a bug.
            UnikoError::Internal(msg) if msg.contains("outstanding references") => {
                UnikoError::Config(
                    "Uniko::shutdown requires all Agent and Session handles to be dropped first"
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
///     .llm(LlmSpec::openai("answerer", "gpt-4o-mini", None))
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
    llm: Option<LlmSpec>,
    streaming: bool,
    scope: RecallScope,
    extractors: ModalityRegistry,
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

    /// Register an LLM, enabling [`Agent::answer`](crate::Agent::answer).
    #[must_use]
    pub fn llm(mut self, spec: LlmSpec) -> Self {
        self.llm = Some(spec);
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

    /// Build the [`Uniko`] instance.
    ///
    /// # Errors
    ///
    /// Propagates store-open failures (config validation, model warmup).
    pub async fn build(self) -> Result<Uniko, UnikoError> {
        let mut config = self.config.unwrap_or_default();
        if let Some(embedding) = self.embedding {
            config.embedding = embedding;
        }
        if let Some(reranker) = self.reranker {
            config.reranker = reranker;
        }
        let pipeline_config = PipelineConfig::from_uniko_config(&config);

        let llm_alias = self.llm.as_ref().map(|l| l.alias.clone());
        let extra_catalog: Vec<ModelAliasSpec> = self
            .llm
            .map(|l| vec![l.into_alias_spec()])
            .unwrap_or_default();

        let kb = match self.location {
            Location::InMemory => {
                KnowledgeBase::in_memory_with_xervo(config, extra_catalog).await?
            }
            Location::Path(path) => {
                KnowledgeBase::open_with_xervo(path, config, extra_catalog).await?
            }
        };
        crate::rules::register_stdlib_rules(&kb).await?;

        let streaming = if self.streaming {
            let steps: Vec<Box<dyn uniko_pipes::Step>> =
                vec![Box::new(uniko_extract::ingest::IngestStep)];
            let pipeline = PipelineSystem::new(pipeline_config, Arc::new(kb.clone()), steps);
            Some(Arc::new(pipeline))
        } else {
            None
        };

        Ok(Uniko::from_parts(
            kb,
            llm_alias,
            streaming,
            self.scope,
            Arc::new(self.extractors),
        ))
    }
}

/// An LLM registration for [`Agent::answer`](crate::Agent::answer).
///
/// Construct via a provider helper ([`LlmSpec::openai`],
/// [`LlmSpec::llamacpp`]); the `alias` is the handle the answer path
/// generates under.
#[derive(Debug, Clone)]
pub struct LlmSpec {
    alias: String,
    provider_id: String,
    model_id: String,
    options: serde_json::Value,
}

impl LlmSpec {
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
}
