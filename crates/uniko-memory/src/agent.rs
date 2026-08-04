//! The [`Agent`] facade — one ergonomic handle over the agent-facing
//! tools.
//!
//! Every tool in this crate is a free function taking `(&KnowledgeBase,
//! agent_id, params)`.  That is the right shape for the library core,
//! but a caller driving a single agent repeats `kb` and `agent_id` on
//! every call.  `Agent` binds both once and exposes the same tools as
//! methods; the free functions remain the implementation, so there is no
//! behavioural divergence between the two surfaces.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use tokio::sync::OnceCell;
use uniko_extract::ingest::ModalityRegistry;
use uniko_store::locy::{AbductionResult, AssumeBuilder, Record};
use uniko_store::xervo::{GenerationOptions, Message};
use uniko_store::{DeletionReport, KnowledgeBase, NodeId, UnikoError, Value};

use crate::consolidation::CycleStats;
use crate::facade::{FinalizeReport, RecallScope, Session, finalize_session};
use crate::nl_to_cypher::is_safe_read_only;
use crate::pipeline::PipelineSystem;
use crate::policy::Viewer;
use crate::query::{Answer, GeneratedAnswer, QueryRecordOptions, answer_query};
use crate::recall::{ContextBundle, RecallConfig, Scope, ViewerScope, recall};
use crate::rules::{AddRuleParams, add_rule};

/// System prompt used by [`Agent::answer`] when generating a reply.
const ANSWER_SYSTEM_PROMPT: &str = "You are a helpful assistant. Answer the question using only the \
    provided context. If the context does not contain the answer, say so. Answer concisely.";

/// An agent-scoped handle over the cognitive-memory surface.
///
/// Obtained from [`Uniko::agent`](crate::Uniko::agent). Binds an agent
/// identity to the store so `recall` / `answer` / `query` / `session` /
/// delete verbs run without re-passing it.
///
/// # Examples
///
/// ```no_run
/// # async fn demo() -> Result<(), uniko_store::UnikoError> {
/// use uniko_memory::Uniko;
///
/// let memory = Uniko::in_memory().await?;
/// let agent = memory.agent("assistant-1");
/// let context = agent.recall("hobbies").await?;
/// # let _ = context;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Agent {
    kb: KnowledgeBase,
    agent_id: String,
    /// LLM alias enabling [`Agent::answer`], when one was configured.
    llm_alias: Option<String>,
    /// Streaming ingest pipeline, shared from the owning `Uniko`.
    streaming: Option<Arc<PipelineSystem>>,
    /// Read visibility scope resolved at recall time.
    scope: RecallScope,
    /// Modality extractors for [`Session::ingest`], shared from `Uniko`.
    extractors: Arc<ModalityRegistry>,
    /// Memoized [`Viewer`] for [`RecallScope::AsAgent`], resolved once.
    viewer_cache: Arc<OnceCell<Viewer>>,
}

impl fmt::Debug for Agent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // KnowledgeBase is not Debug (it wraps the live store handle);
        // redact it and surface only the agent identity.
        f.debug_struct("Agent")
            .field("agent_id", &self.agent_id)
            .finish_non_exhaustive()
    }
}

impl Agent {
    /// Create a facade carrying the owning instance's LLM, streaming
    /// pipeline, visibility scope, and modality extractors.
    pub(crate) fn with_context(
        kb: KnowledgeBase,
        agent_id: impl Into<String>,
        llm_alias: Option<String>,
        streaming: Option<Arc<PipelineSystem>>,
        scope: RecallScope,
        extractors: Arc<ModalityRegistry>,
    ) -> Self {
        Self {
            kb,
            agent_id: agent_id.into(),
            llm_alias,
            streaming,
            scope,
            extractors,
            viewer_cache: Arc::new(OnceCell::new()),
        }
    }

    /// The underlying knowledge base. Test-only seam for graph-assertion
    /// fixtures; the facade never exposes `KnowledgeBase` publicly.
    #[cfg(test)]
    pub(crate) fn kb(&self) -> &KnowledgeBase {
        &self.kb
    }

    /// The bound agent's `participant_id`.
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    /// Addressed retrieval — dereference recall sources by id.
    ///
    /// Recall returns [`RecallSource`](crate::recall::RecallSource)
    /// references (`message_id` / `artifact_id`); the returned [`Data`]
    /// handle turns them into content via
    /// [`message`](crate::facade::Data::message) /
    /// [`artifact`](crate::facade::Data::artifact) /
    /// [`artifact_bytes`](crate::facade::Data::artifact_bytes).
    #[must_use]
    pub fn data(&self) -> crate::facade::Data<'_> {
        crate::facade::Data::new(&self.kb)
    }

    /// The agent's goal/task lifecycle surface.
    ///
    /// Create, transition (`start`/`complete`/`abandon`), read sliced by
    /// phase (`active`/`planned`/`completed`), and expand a goal's working
    /// context — see [`Goals`](crate::facade::Goals).
    #[must_use]
    pub fn goals(&self) -> crate::facade::Goals<'_> {
        crate::facade::Goals::new(&self.kb, &self.agent_id)
    }

    /// Run a read-only Cypher query and return the result rows.
    ///
    /// Each row is a [`Record`] (column name → [`Value`]). The query is
    /// rejected unless it is read-only, so this can never mutate the graph.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] when the query is not read-only, or
    /// [`UnikoError::Locy`] on an evaluation error.
    pub async fn query(&self, cypher: &str) -> Result<Vec<Record>, UnikoError> {
        if !is_safe_read_only(cypher) {
            return Err(UnikoError::Storage(
                "query() accepts read-only Cypher only".to_string(),
            ));
        }
        // Use the graph query engine, not the Locy logic runtime: Locy
        // serves derived facts/rules, not raw node `MATCH`es.
        self.kb.query_cypher(cypher, &HashMap::new()).await
    }

    /// Run read-only Cypher with `scope`'s dimensional allow-set bound as
    /// `$allow`.
    ///
    /// Resolves the session/participant/time scope into an allow-set of node
    /// ids and binds it as the `$allow` list parameter, so the caller's
    /// Cypher can restrict candidates with `WHERE id(n) IN $allow`. `$allow`
    /// is bound only when `scope` constrains at least one dimension (an empty
    /// scope behaves like [`query`](Agent::query)).
    ///
    /// Unlike [`recall_in`](Agent::recall_in), this returns **raw**
    /// [`Record`]s and applies **no visibility filtering** — use `recall_in`
    /// when results must be policy-scoped to a viewer.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] when the query is not read-only, or
    /// [`UnikoError::Locy`] on an evaluation error.
    pub async fn query_in(&self, cypher: &str, scope: &Scope) -> Result<Vec<Record>, UnikoError> {
        if !is_safe_read_only(cypher) {
            return Err(UnikoError::Storage(
                "query_in() accepts read-only Cypher only".to_string(),
            ));
        }
        let filter = uniko_store::repository::recall::ScopeFilter {
            sessions: scope.dims.sessions.clone(),
            participants: scope.dims.participants.clone(),
            since: scope.dims.since,
            until: scope.dims.until,
        };
        let mut params: HashMap<String, Value> = HashMap::new();
        if filter.is_active() {
            let allow = self.kb.resolve_scope_allow_set(&filter).await?;
            params.insert(
                "allow".to_string(),
                Value::List(allow.into_iter().map(Value::Int).collect()),
            );
        }
        // Use the graph query engine (not the Locy runtime `query` uses), so
        // `id(n) IN $allow` and other graph predicates resolve.
        self.kb.query_cypher(cypher, &params).await
    }

    /// Define a Locy derivation rule from `source`, tracked with a
    /// confidence lifecycle.
    ///
    /// `source` is Locy code (e.g. `CREATE RULE <name> AS MATCH … YIELD …`).
    /// The rule is registered with the logic runtime and recorded as a
    /// `candidate` `:Rule`. From then on the cortex sweep **executes** the rule
    /// each pass: a bound match rewards confidence (promoting candidate →
    /// active), while passes with no match decay it (demotion / pruning). You
    /// can also run it on demand with [`Agent::run_rule`]. The rule's `YIELD`
    /// rows feed the lifecycle; to act on them with side effects, run it
    /// explicitly and consume the rows, or wire a consumer (see the stdlib
    /// rules).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Locy`] on a syntax error (no node is left
    /// behind), or [`UnikoError::Storage`] on a write failure.
    pub async fn define_rule(
        &self,
        name: impl Into<String>,
        source: impl Into<String>,
    ) -> Result<NodeId, UnikoError> {
        add_rule(
            &self.kb,
            AddRuleParams {
                name: name.into(),
                source: source.into(),
                source_type: "authored".to_string(),
                ..AddRuleParams::default()
            },
        )
        .await
    }

    /// Run a registered Locy rule by `name`, returning its `YIELD` rows.
    ///
    /// `return_cols` names the rule's `YIELD` aliases; `params` injects
    /// rule body parameters (e.g. `("agent_id", Value::from(...))`).
    ///
    /// The stdlib rule parameters are bound first (see
    /// [`Agent::locy_params`]); anything in `params` wins on collision.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Locy`] if the rule is unregistered or
    /// evaluation fails.
    pub async fn run_rule(
        &self,
        name: &str,
        return_cols: &[&str],
        params: HashMap<String, Value>,
    ) -> Result<Vec<Record>, UnikoError> {
        let mut merged = self.locy_params();
        merged.extend(params);
        self.kb.query_rule(name, return_cols, &merged).await
    }

    /// The parameters every Locy program run through this agent must carry.
    ///
    /// uni-db resolves each *registered* rule as a sub-plan of any Locy
    /// program — including one that references no rules at all
    /// (rustic-ai/uni-db#157) — so an unresolved parameter in an unrelated
    /// stdlib rule fails the whole run.
    /// `Uniko` registers four parameterized stdlib rules at construction, so
    /// without these bindings every `assume` / `abduce` / `run_rule` call on a
    /// facade-built instance fails with `Unresolved parameter: $agent_id`.
    /// See [`crate::rules::run_active_rules`], which carries the same union.
    fn locy_params(&self) -> HashMap<String, Value> {
        let cfg = self.kb.config();
        crate::rules::stdlib_rule_params(&self.agent_id, cfg.half_life_days, cfg.prune_below)
    }

    /// Begin a hypothetical (`ASSUME`) query: fork state, apply mutations,
    /// query, then roll back — the real graph is never modified.
    ///
    /// Finish with [`AssumeBuilder::then_query`] and
    /// [`AssumeBuilder::run`].
    ///
    /// The stdlib rule parameters are pre-bound (see [`Agent::locy_params`]);
    /// [`AssumeBuilder::param`] overrides any of them.
    pub fn assume(&self, assume_block: &str) -> AssumeBuilder<'_> {
        let mut builder = self.kb.assume(assume_block);
        for (k, v) in self.locy_params() {
            builder = builder.param(k, v);
        }
        builder
    }

    /// Run an abductive query: given a conclusion, find the minimal set of
    /// facts that support it.
    ///
    /// The stdlib rule parameters are bound first (see
    /// [`Agent::locy_params`]); anything in `params` wins on collision.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Locy`] on an evaluation failure.
    pub async fn abduce(
        &self,
        program: &str,
        params: HashMap<String, Value>,
    ) -> Result<AbductionResult, UnikoError> {
        let mut merged = self.locy_params();
        merged.extend(params);
        self.kb.abduce(program, &merged).await
    }

    /// Hard-delete an entire conversation and everything it owns.
    ///
    /// Removes the Session, its Messages and their Chunks/Observations, and
    /// any Artifact attached solely to it. Facts losing their last support
    /// are soft-invalidated. Idempotent on an unknown id.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on a traversal or write failure.
    pub async fn delete_session(&self, session_id: &str) -> Result<DeletionReport, UnikoError> {
        self.kb.delete_session_cascade(session_id).await
    }

    /// Build (or refresh) the session-level retrieval surfaces for
    /// `session_id`, without holding its [`Session`](crate::Session) handle.
    ///
    /// Same work as [`Session::finalize`](crate::Session::finalize), and the
    /// backfill entry point for knowledge bases ingested before session
    /// chunking was wired into the facade:
    ///
    /// ```no_run
    /// # async fn f(agent: &uniko_memory::Agent) -> Result<(), uniko_store::UnikoError> {
    /// // Sessions that never got their chunks built.
    /// for sid in agent.unfinalized_session_ids().await? {
    ///     agent.finalize_session(&sid).await?;
    /// }
    /// # Ok(()) }
    /// ```
    ///
    /// Unlike [`Session::finalize`](crate::Session::finalize) this does not
    /// quiesce the streaming pipeline first — there is no session-to-pipeline
    /// binding at this level, so it chunks whatever is committed at call time.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on a read or write failure.
    pub async fn finalize_session(&self, session_id: &str) -> Result<FinalizeReport, UnikoError> {
        finalize_session(&self.kb, session_id).await
    }

    /// Run one consolidation cycle for this agent, now.
    ///
    /// Compiles unprocessed `Observation`s into `Fact`s — voting on the
    /// canonical object, stamping each Fact with a bitemporal validity
    /// interval, reinforcing or invalidating prior beliefs, and flagging
    /// entity drift. Returns what the cycle did.
    ///
    /// This is the synchronous, always-available path: it needs no streaming
    /// pipeline and runs on the calling task. A streaming instance also
    /// consolidates on its own (a 20-observation threshold or a 15-minute
    /// timer), so this is for callers that want to decide the moment — at the
    /// end of a conversation, before a report, or in a batch job.
    ///
    /// Cheap when there is nothing to do: a cycle with no unprocessed
    /// Observations returns zeroed [`CycleStats`] without writing.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on a read, write, or embedding failure.
    pub async fn consolidate(&self) -> Result<CycleStats, UnikoError> {
        crate::consolidation::run_cycle(&self.kb, &self.agent_id, None).await
    }

    /// Session ids that have no session-level chunks yet.
    ///
    /// Drives the [`finalize_session`](Agent::finalize_session) backfill loop.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on a read failure.
    pub async fn unfinalized_session_ids(&self) -> Result<Vec<String>, UnikoError> {
        self.kb.unfinalized_session_ids().await
    }

    /// GDPR erasure: remove a participant and the data they authored.
    ///
    /// Deletes Messages they sent (each with its turn cascade) and
    /// Observations about them, and force-invalidates Facts grounded in
    /// that evidence. Idempotent on an unknown id.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on a traversal or write failure.
    pub async fn forget_participant(
        &self,
        participant_id: &str,
    ) -> Result<DeletionReport, UnikoError> {
        self.kb.forget_participant(participant_id).await
    }

    /// Open a [`Session`] for feeding conversation turns.
    pub fn session(&self, session_id: impl Into<String>) -> Session {
        Session::new(
            self.kb.clone(),
            session_id,
            self.streaming.clone(),
            self.llm_alias.clone(),
            self.extractors.clone(),
            self.agent_id.clone(),
        )
    }

    /// Run the recall cascade for `query` with the best-known config.
    ///
    /// Derives the configuration from the instance's [`UnikoConfig`] (the
    /// validated stack) and applies this agent's visibility scope. For a
    /// custom configuration use [`recall_with`](Agent::recall_with).
    ///
    /// # Errors
    ///
    /// Propagates [`recall`]'s errors and any membership-resolution
    /// failure when scoping to the agent.
    pub async fn recall(&self, query: &str) -> Result<ContextBundle, UnikoError> {
        self.recall_in(query, Scope::default()).await
    }

    /// Run the recall cascade scoped to `scope`, for this call only.
    ///
    /// `scope` carries both visibility ([`ViewerScope`]) and dimensional
    /// hard-filters (session / participant / time), mirroring mem0's
    /// per-call `filters`. A [`ViewerScope::Unrestricted`] visibility — the
    /// [`Scope`] default — means "use this agent's instance-default
    /// visibility"; pass [`Scope::as_viewer`](crate::Scope::as_viewer) to
    /// override it explicitly.
    ///
    /// # Errors
    ///
    /// Propagates [`recall`]'s errors and any membership-resolution failure
    /// when falling back to the agent's scope.
    pub async fn recall_in(&self, query: &str, scope: Scope) -> Result<ContextBundle, UnikoError> {
        let config = self.config_for_scope(scope).await?;
        recall(&self.kb, query, &config).await
    }

    /// Recall context for `question` and generate an answer with the
    /// configured LLM, recording a query Episode.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] when no LLM was configured (build
    /// with [`Uniko::builder`](crate::Uniko::builder)`.llm(...)`).
    /// Propagates recall and generation failures.
    pub async fn answer(&self, question: &str) -> Result<Answer, UnikoError> {
        self.answer_in(question, Scope::default()).await
    }

    /// Recall scoped to `scope`, then generate an answer with the
    /// configured LLM, recording a query Episode.
    ///
    /// The scoping semantics match [`recall_in`](Agent::recall_in).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] when no LLM was configured. Propagates
    /// recall and generation failures.
    pub async fn answer_in(&self, question: &str, scope: Scope) -> Result<Answer, UnikoError> {
        let alias = self.llm_alias.as_deref().ok_or_else(|| {
            UnikoError::Config(
                "answer() requires an LLM; build with Uniko::builder().llm(...)".to_string(),
            )
        })?;

        let config = self.config_for_scope(scope).await?;

        let kb = &self.kb;
        // Build the prompt synchronously in the closure body so the
        // returned future captures only owned data (`user`) plus the
        // `self`-lifetime borrows (`kb`, `alias`) — never the closure's
        // `&ContextBundle` / `&str` arguments, which would require an
        // (inexpressible) higher-ranked lifetime on the future.
        let generator = |bundle: &ContextBundle, question: &str| {
            let context = format_context(bundle);
            let user = format!("Context:\n{context}\n\nQuestion: {question}\n\nAnswer:");
            async move {
                let messages = vec![Message::system(ANSWER_SYSTEM_PROMPT), Message::user(&user)];
                let options = GenerationOptions {
                    max_tokens: Some(2048),
                    temperature: Some(0.1),
                    ..Default::default()
                };
                let text = kb.generate(alias, &messages, options).await?;
                Ok(GeneratedAnswer {
                    text,
                    input_tokens: None,
                    output_tokens: None,
                    model: Some(alias.to_string()),
                })
            }
        };

        let record = Some(QueryRecordOptions {
            participant_id: self.agent_id.clone(),
            outcome: Some("success".to_string()),
            ..Default::default()
        });
        answer_query(kb, question, &config, generator, record).await
    }

    /// Build a [`RecallConfig`] from the validated stack with `scope`
    /// applied.
    ///
    /// A [`ViewerScope::Unrestricted`] visibility in `scope` falls back to
    /// this agent's instance-default visibility (so an empty [`Scope`]
    /// reproduces the unscoped [`recall`] path).
    async fn config_for_scope(&self, scope: Scope) -> Result<RecallConfig, UnikoError> {
        let mut config = RecallConfig::from_uniko_config(self.kb.config());
        config.viewer = match scope.viewer {
            ViewerScope::Unrestricted => self.resolve_viewer_scope().await?,
            viewer => viewer,
        };
        config.dimensions = scope.dims;
        Ok(config)
    }

    /// Resolve this agent's [`RecallScope`] into a [`ViewerScope`].
    ///
    /// For [`RecallScope::AsAgent`] the [`Viewer`] is resolved against the
    /// store once and cached for the lifetime of this `Agent` handle.
    async fn resolve_viewer_scope(&self) -> Result<ViewerScope, UnikoError> {
        match &self.scope {
            RecallScope::Unrestricted => Ok(ViewerScope::Unrestricted),
            RecallScope::As(viewer) => Ok(ViewerScope::As(viewer.clone())),
            RecallScope::AsAgent => {
                let viewer = self
                    .viewer_cache
                    .get_or_try_init(|| Viewer::new(&self.kb, &self.agent_id))
                    .await?;
                Ok(ViewerScope::As(viewer.clone()))
            }
        }
    }
}

/// Render a recall bundle into a numbered context block for the LLM.
fn format_context(bundle: &ContextBundle) -> String {
    bundle
        .items
        .iter()
        .enumerate()
        .map(|(i, item)| format!("[{}] {}", i + 1, item.content))
        .collect::<Vec<_>>()
        .join("\n")
}
