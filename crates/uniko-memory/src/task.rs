//! Task creation — the `create_task` agent tool (F9).
//!
//! Tasks are the concrete units of work that advance a [`crate::goal`].
//! Like goals, they are explicit subjective commitments the agent
//! records rather than facts inferred from messages.  A Task links to
//! its parent Goal (`PART_OF`), the Participant it is assigned to
//! (`ASSIGNED_TO`), and optionally to prerequisite or parent Tasks
//! (`DEPENDS_ON` / `SUBTASK_OF`).

use std::collections::HashMap;

use chrono::{DateTime, Utc};

use uniko_extract::embedding::embed_document;
use uniko_store::id::new_id;
use uniko_store::schema::{edges, labels};
use uniko_store::types::datetime_value;
use uniko_store::{KnowledgeBase, NodeId, UnikoError, Value};

/// Inputs for [`create_task`].
///
/// Only `title` is required.  `goal_id`, `depends_on_task_id`, and
/// `subtask_of_task_id` are external keys resolved best-effort; each
/// missing reference is skipped with a tracing debug rather than
/// failing the call.
#[derive(Debug, Clone, Default)]
pub struct CreateTaskParams {
    /// Optional pre-assigned id.  When `None`, a fresh UUID v7 is used.
    pub task_id: Option<String>,
    /// Required: short statement of the work to be done.
    pub title: String,
    /// Optional longer description.
    pub description: Option<String>,
    /// Lifecycle status (e.g. `"todo"`, `"doing"`, `"done"`).  Defaults
    /// to `"todo"`.
    pub status: Option<String>,
    /// Priority in `[0.0, 1.0]`; higher is more urgent.
    pub priority: Option<f64>,
    /// Parent Goal (`goal_id`).  When set and resolvable, a `PART_OF`
    /// edge is created.
    pub goal_id: Option<String>,
    /// Prerequisite Task (`task_id`).  When set and resolvable, a
    /// `DEPENDS_ON` edge is created.
    pub depends_on_task_id: Option<String>,
    /// Parent Task (`task_id`) for sub-task hierarchies.  When set and
    /// resolvable, a `SUBTASK_OF` edge is created.
    pub subtask_of_task_id: Option<String>,
    /// Creation time.  Defaults to `now()`.
    pub created_at: Option<DateTime<Utc>>,
}

/// Create a Task node, wire it to its goal/assignee/dependencies, embed.
///
/// Wires `ASSIGNED_TO` → the agent's Participant (mandatory).
/// Best-effort optional edges: `PART_OF` → Goal, `DEPENDS_ON` → Task,
/// `SUBTASK_OF` → Task.  The embedding is computed from `title` plus
/// `description`.
///
/// The Participant identified by `agent_id` must already exist (same
/// contract as [`crate::record_episode`]).
///
/// Returns the new Task's NodeId.
///
/// # Errors
///
/// - [`UnikoError::Storage`] when the agent is missing or graph
///   operations fail.
/// - [`UnikoError::Embedding`] when embedding fails — the Task node is
///   still created so it can be back-filled later.
pub async fn create_task(
    kb: &KnowledgeBase,
    agent_id: &str,
    params: CreateTaskParams,
) -> Result<NodeId, UnikoError> {
    let participant_node = kb
        .get_node_by_ext_id(labels::PARTICIPANT, "participant_id", agent_id)
        .await?
        .ok_or_else(|| {
            UnikoError::Storage(format!(
                "create_task: Participant '{agent_id}' not found — create it before recording"
            ))
        })?;
    let participant_id = participant_node.0;

    let task_id = params.task_id.unwrap_or_else(new_id);
    let created_at = params.created_at.unwrap_or_else(Utc::now);

    let mut props: HashMap<String, Value> = HashMap::new();
    props.insert("task_id".into(), Value::String(task_id.clone()));
    props.insert("title".into(), Value::String(params.title.clone()));
    if let Some(desc) = params.description.as_deref() {
        props.insert("description".into(), Value::String(desc.to_string()));
    }
    props.insert(
        "status".into(),
        Value::String(params.status.clone().unwrap_or_else(|| "todo".into())),
    );
    if let Some(priority) = params.priority {
        props.insert("priority".into(), Value::Float(priority));
    }
    props.insert("created_at".into(), datetime_value(created_at));

    let task_node = kb
        .merge_node(labels::TASK, "task_id", &task_id, &props)
        .await?;

    // ── Mandatory ASSIGNED_TO edge (Task → Participant) ──
    kb.ensure_edge(edges::ASSIGNED_TO, task_node, participant_id)
        .await?;

    // ── Optional PART_OF (Task → Goal) ──
    if let Some(goal_id) = params.goal_id.as_deref() {
        link_best_effort(
            kb,
            edges::PART_OF,
            task_node,
            labels::GOAL,
            "goal_id",
            goal_id,
        )
        .await?;
    }

    // ── Optional DEPENDS_ON (Task → Task) ──
    if let Some(dep_id) = params.depends_on_task_id.as_deref() {
        link_best_effort(
            kb,
            edges::DEPENDS_ON,
            task_node,
            labels::TASK,
            "task_id",
            dep_id,
        )
        .await?;
    }

    // ── Optional SUBTASK_OF (Task → Task) ──
    if let Some(parent_id) = params.subtask_of_task_id.as_deref() {
        link_best_effort(
            kb,
            edges::SUBTASK_OF,
            task_node,
            labels::TASK,
            "task_id",
            parent_id,
        )
        .await?;
    }

    // ── Embedding from title + description ──
    let embed_text = match params.description.as_deref() {
        Some(desc) if !desc.is_empty() => format!("{}\n{desc}", params.title),
        _ => params.title.clone(),
    };
    let vec = embed_document(kb, &embed_text).await?;
    let mut emb_props = HashMap::new();
    emb_props.insert("embedding".into(), Value::Vector(vec));
    kb.update_node(task_node, &emb_props).await?;

    Ok(task_node)
}

/// Resolve a node by external id and create `edge_type` from `from` to
/// it, skipping with a tracing debug when the target is missing.
async fn link_best_effort(
    kb: &KnowledgeBase,
    edge_type: &str,
    from: NodeId,
    target_label: &str,
    id_field: &str,
    ext_id: &str,
) -> Result<(), UnikoError> {
    match kb
        .get_node_by_ext_id(target_label, id_field, ext_id)
        .await?
    {
        Some(target) => {
            kb.ensure_edge(edge_type, from, target.0).await?;
        }
        None => {
            tracing::debug!(
                edge = edge_type,
                target_label,
                ext_id,
                "create_task: target not found — edge skipped"
            );
        }
    }
    Ok(())
}
