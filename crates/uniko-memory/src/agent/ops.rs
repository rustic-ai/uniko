//! `Agent` methods over the memory operations that used to be reachable
//! only as free functions taking a `KnowledgeBase`: facts, actions,
//! episodes and observations, visibility, rule management, and a
//! consolidation pass that includes the cortex sweeps.
//!
//! Each method delegates to the existing free function, binding this
//! agent's store handle and id, as the rest of `Agent` does.

use std::collections::HashMap;

use uniko_cortex::procedures::{LifecycleConfig, PromotionReport, promote_procedures_once};
use uniko_cortex::topics::{TopicConfig, TopicReport, detect_topics_once};
use uniko_store::schema::constants::labels;
use uniko_store::{NodeId, UnikoError, Value};

use super::Agent;
use crate::action::{RecordActionParams, RecordActionResult, record_action};
use crate::consolidation::CycleStats;
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

/// Make sure `agent_id` has a Participant node, which actions, episodes
/// and observations hang off. Ingest creates it on the agent's first
/// observed turn; these methods may run before any. An existing node is
/// left as it is.
pub(crate) async fn ensure_agent_participant(
    kb: &uniko_store::KnowledgeBase,
    agent_id: &str,
) -> Result<(), UnikoError> {
    if kb
        .get_node_by_ext_id(labels::PARTICIPANT, "participant_id", agent_id)
        .await?
        .is_some()
    {
        return Ok(());
    }
    let mut props = HashMap::new();
    props.insert("kind".to_string(), Value::String("agent".to_string()));
    kb.merge_node(labels::PARTICIPANT, "participant_id", agent_id, &props)
        .await?;
    Ok(())
}

/// Rule status set by [`Agent::delete_rule`], the same one pruning uses.
const STATUS_PRUNED: &str = "pruned";

impl Agent {
    async fn ensure_participant(&self) -> Result<(), UnikoError> {
        ensure_agent_participant(&self.kb, &self.agent_id).await
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
        let cycle = crate::consolidation::run_cycle(&self.kb, &self.agent_id, None).await?;
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
