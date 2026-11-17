//! The agent's goal/task lifecycle surface.
//!
//! Reached via [`Agent::goals`](crate::Agent::goals). Covers the full
//! lifecycle — create, transition (`start`/`complete`/`abandon`), read
//! sliced by phase (planned / active / completed / abandoned), and expand a
//! goal's working context. Replaces the removed `working_memory` expander
//! with typed views instead of recall items.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use uniko_store::id::new_id;
use uniko_store::types::{datetime_from_value, datetime_value};
use uniko_store::{KnowledgeBase, NodeId, UnikoError, Value};

use crate::value_convert::{json_to_value, value_to_json};
use crate::{CreateGoalParams, CreateTaskParams, create_goal, create_task};

/// Lifecycle phase of a goal, derived from its free-form `status` +
/// `completed_at`. See [`Goals`] for the status→phase mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GoalPhase {
    /// Exists but not started (future plan).
    Planned,
    /// Currently being pursued.
    Active,
    /// Finished.
    Completed,
    /// Dropped without completion.
    Abandoned,
}

/// Lifecycle phase of a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskPhase {
    /// Not started.
    Planned,
    /// In progress.
    Active,
    /// Finished.
    Completed,
    /// Stalled on a dependency.
    Blocked,
}

/// A goal with its lifecycle phase and recorded result.
#[derive(Debug, Clone, Serialize)]
pub struct GoalView {
    /// External goal id.
    pub goal_id: String,
    /// Objective statement.
    pub title: String,
    /// Optional detail.
    pub description: Option<String>,
    /// Raw status string (free-form; the source of `phase`).
    pub status: String,
    /// Phase derived from `status` + `completed_at`.
    pub phase: GoalPhase,
    /// When the goal was created.
    pub created_at: Option<DateTime<Utc>>,
    /// Target completion date, when set.
    pub deadline: Option<DateTime<Utc>>,
    /// When the goal was completed, when set.
    pub completed_at: Option<DateTime<Utc>>,
    /// Success criteria and/or recorded result (free-form JSON).
    pub metrics: Option<JsonValue>,
}

/// A task with its lifecycle phase.
#[derive(Debug, Clone, Serialize)]
pub struct TaskView {
    /// External task id.
    pub task_id: String,
    /// Work description.
    pub title: String,
    /// Optional detail.
    pub description: Option<String>,
    /// Raw status string.
    pub status: String,
    /// Phase derived from `status` + `completed_at`.
    pub phase: TaskPhase,
    /// Numeric priority in `[0.0, 1.0]`, when set.
    pub priority: Option<f64>,
    /// The goal this task is part of, when linked.
    pub goal_id: Option<String>,
    /// When the task was created.
    pub created_at: Option<DateTime<Utc>>,
    /// When the task was completed, when set.
    pub completed_at: Option<DateTime<Utc>>,
}

/// A goal plus its working context — the typed expansion that replaced
/// `working_memory`.
#[derive(Debug, Clone, Serialize)]
pub struct GoalContext {
    /// The goal itself.
    pub goal: GoalView,
    /// Tasks under the goal.
    pub tasks: Vec<TaskView>,
    /// Sessions scoped to the goal (display text).
    pub sessions: Vec<String>,
    /// Recent messages in those sessions (display text, newest first).
    pub recent_messages: Vec<String>,
    /// Facts about entities mentioned while pursuing the goal.
    pub facts: Vec<String>,
    /// Entities mentioned while pursuing the goal.
    pub entities: Vec<String>,
}

/// The agent's goal/task surface.
///
/// Borrows the agent's store + identity; obtain one with
/// [`Agent::goals`](crate::Agent::goals). Listings are scoped to goals
/// the agent owns (`OWNED_BY`) and tasks assigned to it (`ASSIGNED_TO`);
/// lookups by id ([`get`](Self::get), [`task`](Self::task)) are not.
/// Transitions return `Ok(false)` when the id doesn't resolve.
#[derive(Debug)]
pub struct Goals<'a> {
    kb: &'a KnowledgeBase,
    agent_id: &'a str,
}

impl<'a> Goals<'a> {
    /// Construct a handle. Crate-internal; callers use
    /// [`Agent::goals`](crate::Agent::goals).
    pub(crate) fn new(kb: &'a KnowledgeBase, agent_id: &'a str) -> Self {
        Self { kb, agent_id }
    }

    // ── Reads ────────────────────────────────────────────────────────

    /// All goals the agent owns, newest first.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when the query fails.
    pub async fn all(&self) -> Result<Vec<GoalView>, UnikoError> {
        let rows = self
            .kb
            .query_cypher(
                "MATCH (g:Goal)-[:OWNED_BY]->(:Participant {participant_id: $agent}) \
                 RETURN g.goal_id AS goal_id, g.title AS title, g.description AS description, \
                        g.status AS status, g.created_at AS created_at, g.deadline AS deadline, \
                        g.completed_at AS completed_at, g.metrics AS metrics \
                 ORDER BY g.created_at DESC",
                &str_param("agent", self.agent_id),
            )
            .await?;
        Ok(rows.iter().map(goal_from_row).collect())
    }

    /// Goals in a given lifecycle phase.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when the query fails.
    pub async fn in_phase(&self, phase: GoalPhase) -> Result<Vec<GoalView>, UnikoError> {
        Ok(self
            .all()
            .await?
            .into_iter()
            .filter(|g| g.phase == phase)
            .collect())
    }

    /// Goals currently being pursued.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when the query fails.
    pub async fn active(&self) -> Result<Vec<GoalView>, UnikoError> {
        self.in_phase(GoalPhase::Active).await
    }

    /// Goals planned but not yet started (future work).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when the query fails.
    pub async fn planned(&self) -> Result<Vec<GoalView>, UnikoError> {
        self.in_phase(GoalPhase::Planned).await
    }

    /// Completed goals (history + recorded results).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when the query fails.
    pub async fn completed(&self) -> Result<Vec<GoalView>, UnikoError> {
        self.in_phase(GoalPhase::Completed).await
    }

    /// Fetch one goal by id. `Ok(None)` when unknown.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when the query fails.
    pub async fn get(&self, goal_id: &str) -> Result<Option<GoalView>, UnikoError> {
        let rows = self
            .kb
            .query_cypher(
                "MATCH (g:Goal {goal_id: $id}) \
                 RETURN g.goal_id AS goal_id, g.title AS title, g.description AS description, \
                        g.status AS status, g.created_at AS created_at, g.deadline AS deadline, \
                        g.completed_at AS completed_at, g.metrics AS metrics LIMIT 1",
                &str_param("id", goal_id),
            )
            .await?;
        Ok(rows.first().map(goal_from_row))
    }

    /// Fetch one task by id. `Ok(None)` when unknown.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when the query fails.
    pub async fn task(&self, task_id: &str) -> Result<Option<TaskView>, UnikoError> {
        let rows = self
            .kb
            .query_cypher(
                "MATCH (t:Task {task_id: $id}) \
                 OPTIONAL MATCH (t)-[:PART_OF]->(g:Goal) \
                 RETURN t.task_id AS task_id, t.title AS title, t.description AS description, \
                        t.status AS status, t.priority AS priority, g.goal_id AS goal_id, \
                        t.created_at AS created_at, t.completed_at AS completed_at LIMIT 1",
                &str_param("id", task_id),
            )
            .await?;
        Ok(rows.first().map(task_from_row))
    }

    /// All tasks assigned to the agent.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when the query fails.
    pub async fn tasks(&self) -> Result<Vec<TaskView>, UnikoError> {
        let rows = self
            .kb
            .query_cypher(
                "MATCH (t:Task)-[:ASSIGNED_TO]->(:Participant {participant_id: $agent}) \
                 OPTIONAL MATCH (t)-[:PART_OF]->(g:Goal) \
                 RETURN t.task_id AS task_id, t.title AS title, t.description AS description, \
                        t.status AS status, t.priority AS priority, g.goal_id AS goal_id, \
                        t.created_at AS created_at, t.completed_at AS completed_at \
                 ORDER BY t.priority DESC",
                &str_param("agent", self.agent_id),
            )
            .await?;
        Ok(rows.iter().map(task_from_row).collect())
    }

    /// Tasks in a given lifecycle phase.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when the query fails.
    pub async fn tasks_in(&self, phase: TaskPhase) -> Result<Vec<TaskView>, UnikoError> {
        Ok(self
            .tasks()
            .await?
            .into_iter()
            .filter(|t| t.phase == phase)
            .collect())
    }

    /// Tasks that are part of a given goal.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when the query fails.
    pub async fn tasks_of(&self, goal_id: &str) -> Result<Vec<TaskView>, UnikoError> {
        let rows = self
            .kb
            .query_cypher(
                "MATCH (t:Task)-[:PART_OF]->(g:Goal {goal_id: $id}) \
                 RETURN t.task_id AS task_id, t.title AS title, t.description AS description, \
                        t.status AS status, t.priority AS priority, g.goal_id AS goal_id, \
                        t.created_at AS created_at, t.completed_at AS completed_at \
                 ORDER BY t.priority DESC",
                &str_param("id", goal_id),
            )
            .await?;
        Ok(rows.iter().map(task_from_row).collect())
    }

    /// Expand a goal into its working context — tasks, sessions, recent
    /// messages, facts, and entities. `Ok(None)` when the goal is unknown.
    ///
    /// Wraps the store's goal-scope + working-memory queries
    /// (`goal_scope_ids` + `fetch_wm_*`), including descendant goals.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when a query fails.
    pub async fn context(&self, goal_id: &str) -> Result<Option<GoalContext>, UnikoError> {
        let Some(goal) = self.get(goal_id).await? else {
            return Ok(None);
        };
        let scope = self.kb.goal_scope_ids(goal_id, true).await?;
        let tasks = self.tasks_of(goal_id).await?;
        let sessions = self.kb.fetch_wm_sessions(&scope).await?;
        let messages = self.kb.fetch_wm_messages(&scope).await?;
        let facts = self.kb.fetch_wm_facts(&scope).await?;
        let entities = self.kb.fetch_wm_entities(&scope).await?;
        Ok(Some(GoalContext {
            goal,
            tasks,
            sessions: sessions.into_iter().map(|r| r.content).collect(),
            recent_messages: messages.into_iter().map(|r| r.content).collect(),
            facts: facts.into_iter().map(|r| r.content).collect(),
            entities: entities.into_iter().map(|r| r.content).collect(),
        }))
    }

    // ── Transitions ──────────────────────────────────────────────────

    /// Mark a goal active. `Ok(false)` when unknown.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when the update fails.
    pub async fn start(&self, goal_id: &str) -> Result<bool, UnikoError> {
        self.set_status(goal_id, "active").await
    }

    /// Mark a goal abandoned. `Ok(false)` when unknown.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when the update fails.
    pub async fn abandon(&self, goal_id: &str) -> Result<bool, UnikoError> {
        self.set_status(goal_id, "abandoned").await
    }

    /// Set a goal's raw status string. `Ok(false)` when unknown.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when the update fails.
    pub async fn set_status(&self, goal_id: &str, status: &str) -> Result<bool, UnikoError> {
        let mut props = HashMap::new();
        props.insert("status".to_string(), Value::String(status.to_string()));
        update_by_ext_id(self.kb, "Goal", "goal_id", goal_id, props).await
    }

    /// Complete a goal: status `"done"`, stamp `completed_at`, and merge
    /// `result` into its `metrics` (recording the outcome). `Ok(false)`
    /// when unknown.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when the lookup or update fails.
    pub async fn complete(
        &self,
        goal_id: &str,
        result: Option<JsonValue>,
    ) -> Result<bool, UnikoError> {
        let Some((nid, node_props)) = self
            .kb
            .get_node_by_ext_id("Goal", "goal_id", goal_id)
            .await?
        else {
            return Ok(false);
        };
        let mut props = HashMap::new();
        props.insert("status".to_string(), Value::String("done".to_string()));
        props.insert("completed_at".to_string(), datetime_value(Utc::now()));
        if let Some(result) = result {
            let existing = node_props.get("metrics").map(value_to_json);
            props.insert(
                "metrics".to_string(),
                json_to_value(&merge_json(existing, result)),
            );
        }
        self.kb.update_node(nid, &props).await?;
        Ok(true)
    }

    /// Mark a task in progress. `Ok(false)` when unknown.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when the update fails.
    pub async fn start_task(&self, task_id: &str) -> Result<bool, UnikoError> {
        self.set_task_status(task_id, "doing").await
    }

    /// Mark a task blocked. `Ok(false)` when unknown.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when the update fails.
    pub async fn block_task(&self, task_id: &str) -> Result<bool, UnikoError> {
        self.set_task_status(task_id, "blocked").await
    }

    /// Complete a task: status `"done"` and stamp `completed_at`.
    /// `Ok(false)` when unknown.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when the lookup or update fails.
    pub async fn complete_task(&self, task_id: &str) -> Result<bool, UnikoError> {
        let Some((nid, _)) = self
            .kb
            .get_node_by_ext_id("Task", "task_id", task_id)
            .await?
        else {
            return Ok(false);
        };
        let mut props = HashMap::new();
        props.insert("status".to_string(), Value::String("done".to_string()));
        props.insert("completed_at".to_string(), datetime_value(Utc::now()));
        self.kb.update_node(nid, &props).await?;
        Ok(true)
    }

    /// Set a task's raw status string. `Ok(false)` when unknown.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when the update fails.
    pub async fn set_task_status(&self, task_id: &str, status: &str) -> Result<bool, UnikoError> {
        let mut props = HashMap::new();
        props.insert("status".to_string(), Value::String(status.to_string()));
        update_by_ext_id(self.kb, "Task", "task_id", task_id, props).await
    }

    // ── Create ───────────────────────────────────────────────────────

    /// Create a goal owned by this agent. Returns its node id.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when creation fails.
    pub async fn create(&self, params: CreateGoalParams) -> Result<NodeId, UnikoError> {
        create_goal(self.kb, self.agent_id, params).await
    }

    /// Create a task assigned to this agent. Returns its node id.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when creation fails.
    pub async fn create_task(&self, params: CreateTaskParams) -> Result<NodeId, UnikoError> {
        create_task(self.kb, self.agent_id, params).await
    }

    /// Create a goal owned by this agent and return it as a [`GoalView`].
    ///
    /// Unlike [`create`](Self::create), the result carries the goal's
    /// external id, which every other goal method takes. When
    /// `params.goal_id` is `None` a fresh id is generated on each call; when
    /// it is set, the goal is created under that id.
    ///
    /// Re-creating under an existing id overwrites the goal's properties
    /// and keeps one link per relationship, so a retry with the same
    /// explicit id is safe.
    ///
    /// If only the embedding step fails, the goal is still stored (to be
    /// back-filled later) and this returns its view rather than the error,
    /// so the caller never loses a generated id.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when creation fails, or
    /// [`UnikoError::Internal`] if the goal cannot be read back.
    pub async fn create_view(&self, mut params: CreateGoalParams) -> Result<GoalView, UnikoError> {
        let goal_id = params.goal_id.get_or_insert_with(new_id).clone();
        keep_on_embedding_failure(create_goal(self.kb, self.agent_id, params).await, &goal_id)?;
        self.get(&goal_id).await?.ok_or_else(|| {
            UnikoError::Internal(format!("goal '{goal_id}' not readable after create"))
        })
    }

    /// Create a task assigned to this agent and return it as a
    /// [`TaskView`].
    ///
    /// Unlike [`create_task`](Self::create_task), the result carries the
    /// task's external id, which every other task method takes. When
    /// `params.task_id` is `None` a fresh id is generated on each call; when
    /// it is set, the task is created under that id.
    ///
    /// Re-creating under an existing id overwrites the task's properties
    /// and keeps one link per relationship, so a retry with the same
    /// explicit id is safe.
    ///
    /// If only the embedding step fails, the task is still stored (to be
    /// back-filled later) and this returns its view rather than the error,
    /// so the caller never loses a generated id.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] when creation fails, or
    /// [`UnikoError::Internal`] if the task cannot be read back.
    pub async fn create_task_view(
        &self,
        mut params: CreateTaskParams,
    ) -> Result<TaskView, UnikoError> {
        let task_id = params.task_id.get_or_insert_with(new_id).clone();
        keep_on_embedding_failure(create_task(self.kb, self.agent_id, params).await, &task_id)?;
        self.task(&task_id).await?.ok_or_else(|| {
            UnikoError::Internal(format!("task '{task_id}' not readable after create"))
        })
    }
}

/// Resolve a node by ext-id and update `props`; `Ok(false)` when unknown.
async fn update_by_ext_id(
    kb: &KnowledgeBase,
    label: &str,
    id_field: &str,
    ext_id: &str,
    props: HashMap<String, Value>,
) -> Result<bool, UnikoError> {
    match kb.get_node_by_ext_id(label, id_field, ext_id).await? {
        Some((nid, _)) => {
            kb.update_node(nid, &props).await?;
            Ok(true)
        }
        None => Ok(false),
    }
}

/// Shallow-merge `add` into `base` when both are JSON objects; otherwise
/// `add` wins (used to fold a completion result into existing metrics).
fn merge_json(base: Option<JsonValue>, add: JsonValue) -> JsonValue {
    match (base, add) {
        (Some(JsonValue::Object(mut b)), JsonValue::Object(a)) => {
            for (k, v) in a {
                b.insert(k, v);
            }
            JsonValue::Object(b)
        }
        (_, add) => add,
    }
}

/// Map a goal's status + completion time to a phase. `completed_at` wins.
fn goal_phase(status: &str, completed_at: Option<DateTime<Utc>>) -> GoalPhase {
    if completed_at.is_some() {
        return GoalPhase::Completed;
    }
    match status.to_ascii_lowercase().as_str() {
        "done" | "completed" | "complete" | "achieved" => GoalPhase::Completed,
        "abandoned" | "cancelled" | "canceled" | "dropped" | "failed" => GoalPhase::Abandoned,
        "planned" | "pending" | "proposed" => GoalPhase::Planned,
        _ => GoalPhase::Active,
    }
}

/// Map a task's status + completion time to a phase. `completed_at` wins.
fn task_phase(status: &str, completed_at: Option<DateTime<Utc>>) -> TaskPhase {
    if completed_at.is_some() {
        return TaskPhase::Completed;
    }
    match status.to_ascii_lowercase().as_str() {
        "done" | "completed" => TaskPhase::Completed,
        "blocked" => TaskPhase::Blocked,
        "todo" | "planned" | "pending" => TaskPhase::Planned,
        _ => TaskPhase::Active,
    }
}

fn goal_from_row(row: &HashMap<String, Value>) -> GoalView {
    let status = cell_str(row, "status").unwrap_or_default();
    let completed_at = cell_dt(row, "completed_at");
    GoalView {
        goal_id: cell_str(row, "goal_id").unwrap_or_default(),
        title: cell_str(row, "title").unwrap_or_default(),
        description: cell_str(row, "description"),
        phase: goal_phase(&status, completed_at),
        status,
        created_at: cell_dt(row, "created_at"),
        deadline: cell_dt(row, "deadline"),
        completed_at,
        metrics: cell_json(row, "metrics"),
    }
}

fn task_from_row(row: &HashMap<String, Value>) -> TaskView {
    let status = cell_str(row, "status").unwrap_or_default();
    let completed_at = cell_dt(row, "completed_at");
    TaskView {
        task_id: cell_str(row, "task_id").unwrap_or_default(),
        title: cell_str(row, "title").unwrap_or_default(),
        description: cell_str(row, "description"),
        phase: task_phase(&status, completed_at),
        status,
        priority: cell_f64(row, "priority"),
        goal_id: cell_str(row, "goal_id"),
        created_at: cell_dt(row, "created_at"),
        completed_at,
    }
}

fn str_param(key: &str, value: &str) -> HashMap<String, Value> {
    HashMap::from([(key.to_string(), Value::String(value.to_string()))])
}

fn cell_str(row: &HashMap<String, Value>, key: &str) -> Option<String> {
    match row.get(key) {
        Some(Value::String(s)) => Some(s.clone()),
        _ => None,
    }
}

fn cell_f64(row: &HashMap<String, Value>, key: &str) -> Option<f64> {
    match row.get(key) {
        Some(Value::Float(f)) => Some(*f),
        Some(Value::Int(i)) => Some(*i as f64),
        _ => None,
    }
}

fn cell_dt(row: &HashMap<String, Value>, key: &str) -> Option<DateTime<Utc>> {
    row.get(key)
        .and_then(|v| datetime_from_value(v, "goal/task timestamp").ok())
}

fn cell_json(row: &HashMap<String, Value>, key: &str) -> Option<JsonValue> {
    match row.get(key) {
        None | Some(Value::Null) => None,
        Some(v) => Some(value_to_json(v)),
    }
}

/// Treat an embedding failure from `create_goal` / `create_task` as
/// success: the node is already stored and only lacks its vector, so the
/// caller should still get the view (and with it the id) back.
fn keep_on_embedding_failure(
    result: Result<NodeId, UnikoError>,
    id: &str,
) -> Result<(), UnikoError> {
    match result {
        Ok(_) => Ok(()),
        Err(UnikoError::Embedding(e)) => {
            tracing::warn!(id, error = %e, "created without an embedding; returning the stored view");
            Ok(())
        }
        Err(e) => Err(e),
    }
}
