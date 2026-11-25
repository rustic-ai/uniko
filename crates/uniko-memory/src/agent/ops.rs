//! `Agent` methods over the memory operations that used to be reachable
//! only as free functions taking a `KnowledgeBase`: facts, actions,
//! episodes and observations, visibility, rule management, and a
//! consolidation pass that includes the cortex sweeps.
//!
//! Each method delegates to the existing free function, binding this
//! agent's store handle and id, as the rest of `Agent` does.

use std::collections::HashMap;

use uniko_cortex::procedures::{
    LifecycleConfig, MatchedProcedure, PromotionReport, match_procedures, promote_procedures_once,
    record_procedure_use,
};
use uniko_cortex::topics::{TopicConfig, TopicReport, detect_topics_once};
use uniko_store::{NodeId, UnikoError, Value};

use super::Agent;
use crate::action::{RecordActionParams, RecordActionResult, record_action};
use crate::consolidation::{CycleStats, TripleSource};
use crate::episode::{RecordEpisodeParams, record_episode};
use crate::fact::{AssertFactParams, InvalidateFactParams, assert_fact, invalidate_fact};
use crate::observation::{AddObservationParams, add_observation};
use crate::policy::{Visibility, set_visibility};
use crate::rules::is_stdlib_rule;

/// What [`Agent::consolidate`] did: the fact cycle, then the procedure and
/// topic sweeps.
#[derive(Debug, Clone, Default)]
pub struct ConsolidationReport {
    /// The observation-to-fact cycle.
    pub cycle: CycleStats,
    /// Procedure promotion (P5) for this agent.
    pub procedures: PromotionReport,
    /// Topic detection (P6), which covers the whole store.
    pub topics: TopicReport,
}

/// How [`Agent::consolidate_with`] runs the observation-to-fact cycle.
///
/// The default reproduces [`Agent::consolidate`]: triples come from the
/// rule-based SRL/DEP extraction already stored on each Observation, with
/// no model call.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ConsolidateOptions {
    /// Where the cycle's `(subject, predicate, object)` triples come from.
    pub triple_source: TripleSource,
}

impl ConsolidateOptions {
    /// Refine each Observation's triple with the generation model at
    /// `alias` before grouping.
    ///
    /// One model call per unprocessed Observation, so this belongs on an
    /// offline or scheduled consolidation pass, not on the ingest path.
    /// Observations the model declines or answers unparseably keep their
    /// SRL/DEP triple; [`CycleStats::triples_refined`] counts the rest, and
    /// [`CycleStats::triples_failed`] the calls that failed outright.
    #[must_use]
    pub fn llm_triples(alias: impl Into<String>) -> Self {
        Self {
            triple_source: TripleSource::Llm {
                alias: alias.into(),
            },
        }
    }
}

/// A rule as [`Agent::list_rules`] and [`Agent::explain_rule`] show it.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RuleView {
    /// The rule's name.
    pub name: String,
    /// `stdlib`, `authored` or `derived`.
    pub source_type: String,
    /// `candidate`, `active`, `demoted` or `pruned`.
    pub status: String,
    /// Confidence in `[0, 1]`, decayed while the rule goes unmatched.
    pub confidence: f64,
    /// The Locy source. Filled by `explain_rule`; `None` from `list_rules`.
    pub source: Option<String>,
    /// A plain-language description, when one was recorded.
    pub natural_language: Option<String>,
}

/// Make sure `agent_id` has a Participant node of kind `agent`, which
/// goals, tasks, actions, episodes and observations hang off.
///
/// Ingest creates the node on the agent's first observed turn; these
/// methods may run before any. An existing node keeps its name, metadata
/// and memberships; one of kind `unknown` (from ingest or an org
/// membership) becomes `agent`.
pub(crate) async fn ensure_agent_participant(
    kb: &uniko_store::KnowledgeBase,
    agent_id: &str,
) -> Result<uniko_store::EnsureOutcome, UnikoError> {
    ensure_agent_participant_named(kb, agent_id, None).await
}

/// [`ensure_agent_participant`], also giving the node `name` when it has
/// none.
pub(crate) async fn ensure_agent_participant_named(
    kb: &uniko_store::KnowledgeBase,
    agent_id: &str,
    name: Option<&str>,
) -> Result<uniko_store::EnsureOutcome, UnikoError> {
    let mut if_absent = HashMap::new();
    if let Some(n) = name {
        if_absent.insert("name".to_string(), Value::String(n.to_string()));
    }
    let (_, outcome) = kb
        .ensure_participant(agent_id, "agent", &if_absent, &HashMap::new())
        .await?;
    Ok(outcome)
}

/// Options for [`Agent::register`].
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct AgentRegistration {
    /// Display name, set only when the participant has none yet, or has
    /// only the placeholder ingest gives it (its own id).
    pub name: Option<String>,
}

impl AgentRegistration {
    /// No options: register the agent under its id alone.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Give the participant a display name if it has none yet.
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

/// What [`Agent::register`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Registration {
    /// The agent's participant id.
    pub participant_id: String,
    /// Whether the participant was created, upgraded from kind `unknown`,
    /// or already known.
    pub outcome: uniko_store::EnsureOutcome,
}

/// Rule status set by [`Agent::delete_rule`], the same one pruning uses.
const STATUS_PRUNED: &str = "pruned";

impl Agent {
    async fn ensure_participant(&self) -> Result<(), UnikoError> {
        ensure_agent_participant(&self.kb, &self.agent_id).await?;
        Ok(())
    }

    /// Register this agent as a participant of kind `agent`, without
    /// recording a session, message or action.
    ///
    /// Idempotent and safe to call concurrently: the first call creates
    /// the participant and later ones return
    /// [`EnsureOutcome::Existing`](uniko_store::EnsureOutcome::Existing).
    /// A participant already created with kind `unknown` (by ingest or
    /// [`Uniko::add_org_member`](crate::Uniko::add_org_member)) becomes
    /// `agent`; its name, metadata and memberships are kept. Registration
    /// is optional: goals, tasks, actions and tool calls ensure the agent
    /// themselves.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on a storage failure.
    pub async fn register(&self, opts: AgentRegistration) -> Result<Registration, UnikoError> {
        let outcome =
            ensure_agent_participant_named(&self.kb, &self.agent_id, opts.name.as_deref()).await?;
        Ok(Registration {
            participant_id: self.agent_id.clone(),
            outcome,
        })
    }

    /// Assert a fact directly, without an observation to derive it from.
    ///
    /// Facts are keyed by their triple, so asserting one that exists
    /// reinforces it. Set its visibility afterwards with
    /// [`set_visibility`](Agent::set_visibility) on the returned node id.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when embedding the triple or a write fails.
    pub async fn assert_fact(
        &self,
        params: AssertFactParams,
    ) -> Result<uniko_store::operations::facts::FactUpsert, UnikoError> {
        assert_fact(&self.kb, params).await
    }

    /// Invalidate a fact, closing its validity interval, optionally
    /// pointing at the fact that replaces it.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when the fact does not exist or a write fails.
    pub async fn invalidate_fact(&self, params: InvalidateFactParams) -> Result<(), UnikoError> {
        invalidate_fact(&self.kb, params).await
    }

    /// Record an action this agent took (a tool call, an API request).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when a referenced message, session or prior
    /// action is missing, or a write fails.
    pub async fn record_action(
        &self,
        params: RecordActionParams,
    ) -> Result<RecordActionResult, UnikoError> {
        self.ensure_participant().await?;
        record_action(&self.kb, &self.agent_id, params).await
    }

    /// Record an episode: an outcome this agent experienced, which
    /// procedure promotion learns from.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when embedding or a write fails.
    pub async fn record_episode(&self, params: RecordEpisodeParams) -> Result<NodeId, UnikoError> {
        self.ensure_participant().await?;
        record_episode(&self.kb, &self.agent_id, params).await
    }

    /// Add an observation by hand, anchored to an existing message.
    ///
    /// Ingest extracts observations from every observed turn; this is for
    /// ones the caller derived itself. Consolidation picks it up like any
    /// other.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when the message does not exist or a write
    /// fails.
    pub async fn add_observation(
        &self,
        params: AddObservationParams,
    ) -> Result<NodeId, UnikoError> {
        self.ensure_participant().await?;
        add_observation(&self.kb, &self.agent_id, params).await
    }

    /// Set who may see a Fact or Observation.
    ///
    /// Recall then drops the item for every viewer the visibility does not
    /// admit: scope reads with
    /// [`UnikoBuilder::scope_to_agent`](crate::UnikoBuilder::scope_to_agent),
    /// [`Scope::as_viewer`](crate::Scope::as_viewer) or
    /// [`Scope::as_participant`](crate::Scope::as_participant). Unscoped
    /// reads still see everything. Returns `false` when no node has
    /// `node_id`.
    ///
    /// A Fact is shared by every assertion of its triple, so this scopes
    /// it for all of them.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] for a node that is not a Fact or
    /// Observation, an empty id, or a forgotten item (it stays hidden).
    pub async fn set_visibility(
        &self,
        node_id: NodeId,
        visibility: Visibility,
    ) -> Result<bool, UnikoError> {
        set_visibility(&self.kb, node_id, &visibility).await
    }

    /// Every rule in the store, stdlib ones included.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on a read failure.
    pub async fn list_rules(&self) -> Result<Vec<RuleView>, UnikoError> {
        let mut rules: Vec<RuleView> = self
            .kb
            .fetch_lifecycle_rules()
            .await?
            .into_iter()
            .map(|r| RuleView {
                name: r.name,
                source_type: r.source_type,
                status: r.status,
                confidence: r.confidence,
                source: None,
                natural_language: None,
            })
            .collect();
        rules.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(rules)
    }

    /// A rule's definition and state, or `None` when no rule has `name`.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on a read failure.
    pub async fn explain_rule(&self, name: &str) -> Result<Option<RuleView>, UnikoError> {
        let Some(snapshot) = self.find_rule(name).await? else {
            return Ok(None);
        };
        let props = self
            .kb
            .get_node(snapshot.node_id)
            .await?
            .map(|(_, props)| props)
            .unwrap_or_default();
        let text = |key: &str| match props.get(key) {
            Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
            _ => None,
        };
        Ok(Some(RuleView {
            source: text("source"),
            natural_language: text("natural_language"),
            name: snapshot.name,
            source_type: snapshot.source_type,
            status: snapshot.status,
            confidence: snapshot.confidence,
        }))
    }

    /// Delete a rule: stop it running and mark it pruned. The rule node
    /// stays for provenance, as with rules the lifecycle prunes. Returns
    /// `false` when no rule has `name`.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] for a stdlib rule, which the decay
    /// sweep and procedure promotion depend on, or [`UnikoError`] on a
    /// write failure.
    pub async fn delete_rule(&self, name: &str) -> Result<bool, UnikoError> {
        let Some(snapshot) = self.find_rule(name).await? else {
            return Ok(false);
        };
        if is_stdlib_rule(&snapshot.source_type) {
            return Err(UnikoError::Config(format!(
                "'{name}' is a stdlib rule; the engine depends on it, so it cannot be deleted"
            )));
        }
        self.kb.delete_rule(name).await?;
        let mut props = HashMap::new();
        props.insert(
            "status".to_string(),
            Value::String(STATUS_PRUNED.to_string()),
        );
        self.kb.update_node(snapshot.node_id, &props).await?;
        Ok(true)
    }

    async fn find_rule(
        &self,
        name: &str,
    ) -> Result<Option<uniko_store::repository::rules::RuleSnapshot>, UnikoError> {
        Ok(self
            .kb
            .fetch_lifecycle_rules()
            .await?
            .into_iter()
            .find(|r| r.name == name))
    }

    /// Active procedures whose precondition holds in `state`: what has
    /// tended to work next, for an agent deciding its next step.
    ///
    /// Procedures are learned by [`consolidate`](Agent::consolidate) from
    /// repeated successful action sequences; each has a precondition such
    /// as `last_action_type=search`, matched against `state`'s keys.
    /// Report how one worked out with
    /// [`record_procedure_use`](Agent::record_procedure_use).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on a read failure.
    pub async fn match_procedures(
        &self,
        state: &HashMap<String, String>,
    ) -> Result<Vec<MatchedProcedure>, UnikoError> {
        match_procedures(&self.kb, state).await
    }

    /// Record whether following a procedure succeeded. Its effectiveness
    /// moves with the result, and a procedure that keeps failing is
    /// demoted out of [`match_procedures`](Agent::match_procedures).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when the procedure does not exist or a write
    /// fails.
    pub async fn record_procedure_use(
        &self,
        procedure_id: &str,
        succeeded: bool,
    ) -> Result<(), UnikoError> {
        record_procedure_use(
            &self.kb,
            procedure_id,
            succeeded,
            LifecycleConfig::default(),
        )
        .await
    }

    /// Run a full consolidation pass for this agent, now: the
    /// observation-to-fact cycle, then procedure promotion and topic
    /// detection.
    ///
    /// The cycle compiles unprocessed `Observation`s into `Fact`s — voting
    /// on the canonical object, stamping each Fact with a bitemporal
    /// validity interval, reinforcing or invalidating prior beliefs, and
    /// flagging entity drift. Procedure promotion then learns repeated
    /// successful action sequences, and topic detection groups related
    /// entities, so recall can return procedure and topic items.
    ///
    /// This is the synchronous, always-available path: it needs no
    /// streaming pipeline. A streaming instance also runs all of it on its
    /// own (the cycle on a 20-observation threshold or a 15-minute timer,
    /// the sweeps every few cycles), so this is for callers that want to
    /// decide the moment. Topic detection covers the whole store, so it is
    /// the slowest part on a large one.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when the cycle fails. A failed sweep is
    /// logged and reported as zero, as the streaming pipeline does, so one
    /// failing sweep does not lose the cycle's work.
    pub async fn consolidate(&self) -> Result<ConsolidationReport, UnikoError> {
        self.consolidate_with(ConsolidateOptions::default()).await
    }

    /// [`consolidate`](Self::consolidate) with explicit options — today,
    /// the source of the cycle's triples.
    ///
    /// With [`ConsolidateOptions::llm_triples`], the alias is checked
    /// before any work: it must be registered (for example through
    /// [`UnikoBuilder::llm`](crate::UnikoBuilder::llm)) and resolve to a
    /// generation model, so a misconfigured alias fails here rather than
    /// silently falling back on every Observation.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] when the refinement alias is missing
    /// or not a generation model, and otherwise as
    /// [`consolidate`](Self::consolidate).
    pub async fn consolidate_with(
        &self,
        options: ConsolidateOptions,
    ) -> Result<ConsolidationReport, UnikoError> {
        if let TripleSource::Llm { alias } = &options.triple_source {
            self.kb.ensure_generator(alias).await?;
        }
        let cycle = crate::consolidation::run_cycle_with(
            &self.kb,
            &self.agent_id,
            None,
            &options.triple_source,
        )
        .await?;
        let procedures = match promote_procedures_once(
            &self.kb,
            &self.agent_id,
            LifecycleConfig::default(),
        )
        .await
        {
            Ok(report) => report,
            Err(e) => {
                tracing::warn!(agent = %self.agent_id, error = %e, "consolidate: procedure promotion failed");
                PromotionReport::default()
            }
        };
        let topics = match detect_topics_once(&self.kb, TopicConfig::default()).await {
            Ok(report) => report,
            Err(e) => {
                tracing::warn!(error = %e, "consolidate: topic detection failed");
                TopicReport::default()
            }
        };
        Ok(ConsolidationReport {
            cycle,
            procedures,
            topics,
        })
    }
}
