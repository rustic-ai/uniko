//! Goal creation — the `create_goal` agent tool (F8).
//!
//! Goals are top-level objectives the agent is pursuing.  Unlike
//! episodic or semantic memory (which is inferred from messages), a Goal
//! is an explicit subjective commitment: only the agent knows what it is
//! trying to achieve, so the spec (Part VI) classifies `create_goal` as
//! a tool rather than a pipeline.
//!
//! A Goal anchors Tasks ([`crate::task`]), and its `title`/`description`
//! are embedded so working-memory recall can surface the active goal
//! alongside relevant context.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde_json::Value as JsonValue;

use uniko_extract::embedding::embed_document;
use uniko_store::id::new_id;
use uniko_store::schema::{edges, labels};
use uniko_store::types::datetime_value;
use uniko_store::{KnowledgeBase, NodeId, UnikoError, Value};

use crate::value_convert::json_to_value;

/// Inputs for [`create_goal`].
///
/// Only `title` is required.  `metrics` and `guardrails` accept
/// arbitrary JSON (success criteria and constraints) stored verbatim on
/// the node.  Pre-set `goal_id` to integrate with an external ID space
/// (ADR-1); otherwise a UUID v7 is generated.
#[derive(Debug, Clone, Default)]
pub struct CreateGoalParams {
    /// Optional pre-assigned id.  When `None`, a fresh UUID v7 is used.
    pub goal_id: Option<String>,
    /// Required: short statement of the objective.
    pub title: String,
    /// Optional longer description of the objective.
    pub description: Option<String>,
    /// Lifecycle status (e.g. `"active"`, `"done"`).  Defaults to
    /// `"active"`.
    pub status: Option<String>,
    /// Free-form success metrics stored on the node.
    pub metrics: Option<JsonValue>,
    /// Free-form guardrails / constraints stored on the node.
    pub guardrails: Option<JsonValue>,
    /// Optional deadline for the goal.
    pub deadline: Option<DateTime<Utc>>,
    /// Optional parent goal (`goal_id`) for sub-goal hierarchies.  When
    /// set and resolvable, a `PARENT_GOAL` edge is created.
    pub parent_goal_id: Option<String>,
    /// Creation time.  Defaults to `now()`.
    pub created_at: Option<DateTime<Utc>>,
}

/// Create a Goal node, wire ownership, and embed its title/description.
///
/// Wires `OWNED_BY` → the agent's Participant (mandatory) and, when
/// `parent_goal_id` resolves, `PARENT_GOAL` → the parent Goal
/// (best-effort, skipped with a tracing debug when missing).  The
/// embedding is computed from `title` plus `description`.
///
/// The Participant identified by `agent_id` must already exist (same
/// contract as [`crate::record_episode`]).
///
/// Returns the new Goal's NodeId.
///
/// # Errors
///
/// - [`UnikoError::Storage`] when the agent is missing or graph
///   operations fail.
/// - [`UnikoError::Embedding`] when embedding fails — the Goal node is
///   still created so it can be back-filled later.
pub async fn create_goal(
    kb: &KnowledgeBase,
    agent_id: &str,
    params: CreateGoalParams,
) -> Result<NodeId, UnikoError> {
    let participant_node = kb
        .get_node_by_ext_id(labels::PARTICIPANT, "participant_id", agent_id)
        .await?
        .ok_or_else(|| {
            UnikoError::Storage(format!(
                "create_goal: Participant '{agent_id}' not found — create it before recording"
            ))
        })?;
    let participant_id = participant_node.0;

    let goal_id = params.goal_id.unwrap_or_else(new_id);
    let created_at = params.created_at.unwrap_or_else(Utc::now);

    let mut props: HashMap<String, Value> = HashMap::new();
    props.insert("goal_id".into(), Value::String(goal_id.clone()));
    props.insert("title".into(), Value::String(params.title.clone()));
    if let Some(desc) = params.description.as_deref() {
        props.insert("description".into(), Value::String(desc.to_string()));
    }
    props.insert(
        "status".into(),
        Value::String(params.status.clone().unwrap_or_else(|| "active".into())),
    );
    if let Some(ref m) = params.metrics {
        props.insert("metrics".into(), json_to_value(m));
    }
    if let Some(ref g) = params.guardrails {
        props.insert("guardrails".into(), json_to_value(g));
    }
    props.insert("owner_id".into(), Value::String(agent_id.to_string()));
    props.insert("created_at".into(), datetime_value(created_at));
    if let Some(deadline) = params.deadline {
        props.insert("deadline".into(), datetime_value(deadline));
    }

    let goal_node = kb
        .merge_node(labels::GOAL, "goal_id", &goal_id, &props)
        .await?;

    // ── Mandatory OWNED_BY edge (Goal → Participant) ──
    kb.ensure_edge(edges::OWNED_BY, goal_node, participant_id)
        .await?;

    // ── Optional PARENT_GOAL (this → parent) ──
    if let Some(parent_id) = params.parent_goal_id.as_deref() {
        match kb
            .get_node_by_ext_id(labels::GOAL, "goal_id", parent_id)
            .await?
        {
            Some(parent) => {
                kb.ensure_edge(edges::PARENT_GOAL, goal_node, parent.0)
                    .await?;
            }
            None => {
                tracing::debug!(
                    parent_goal_id = parent_id,
                    "create_goal: parent Goal not found — PARENT_GOAL skipped"
                );
            }
        }
    }

    // ── Embedding from title + description ──
    let embed_text = match params.description.as_deref() {
        Some(desc) if !desc.is_empty() => format!("{}\n{desc}", params.title),
        _ => params.title.clone(),
    };
    let vec = embed_document(kb, &embed_text).await?;
    let mut emb_props = HashMap::new();
    emb_props.insert("embedding".into(), Value::Vector(vec));
    kb.update_node(goal_node, &emb_props).await?;

    Ok(goal_node)
}
