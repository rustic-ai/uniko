// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Industries Inc.

//! The [`PyGoals`] handle — the agent's goal/task lifecycle surface.
//!
//! Like [`PyData`](crate::data::PyData), the facade's `Goals<'_>` borrows the
//! store, so `PyGoals` holds an Agent clone and rebuilds `agent.goals()` inside
//! each async block. Phases are exchanged as lowercase strings
//! (`planned`/`active`/`completed`/`abandoned`|`blocked`).

use pyo3::prelude::*;
use uniko_api::tools::{Agent, CreateGoalParams, CreateTaskParams, GoalPhase, NodeId, TaskPhase};

use crate::convert::{py_to_datetime, py_to_json};
use crate::errors::to_pyerr;
use crate::macros::pymethods_with_sync;
use crate::outputs::{PyGoalContext, PyGoalView, PyTaskView};

/// Parse a goal phase string into the facade enum.
fn parse_goal_phase(s: &str) -> PyResult<GoalPhase> {
    match s {
        "planned" => Ok(GoalPhase::Planned),
        "active" => Ok(GoalPhase::Active),
        "completed" => Ok(GoalPhase::Completed),
        "abandoned" => Ok(GoalPhase::Abandoned),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "invalid goal phase '{other}' (expected planned/active/completed/abandoned)"
        ))),
    }
}

/// Parse a task phase string into the facade enum.
fn parse_task_phase(s: &str) -> PyResult<TaskPhase> {
    match s {
        "planned" => Ok(TaskPhase::Planned),
        "active" => Ok(TaskPhase::Active),
        "completed" => Ok(TaskPhase::Completed),
        "blocked" => Ok(TaskPhase::Blocked),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "invalid task phase '{other}' (expected planned/active/completed/blocked)"
        ))),
    }
}

/// Convert an optional Python value into optional JSON.
fn opt_json(value: Option<Bound<'_, PyAny>>) -> PyResult<Option<serde_json::Value>> {
    value.map(|b| py_to_json(&b)).transpose()
}

/// The agent's goal/task surface.
///
/// Obtained from [`Agent.goals`](crate::agent::PyAgent). Reads are scoped to
/// goals the agent owns and tasks assigned to it; transitions resolve to
/// `False` when the id doesn't match.
#[pyclass(name = "Goals", module = "uniko")]
pub struct PyGoals {
    inner: Agent,
}

impl PyGoals {
    /// Wrap an Agent clone.
    pub(crate) fn wrap(agent: Agent) -> Self {
        Self { inner: agent }
    }
}

pymethods_with_sync! {
    PyGoals;
    plain {
        fn __repr__(&self) -> String {
            format!("Goals(agent_id='{}')", self.inner.agent_id())
        }

    }
    bridged {

        /// All goals the agent owns, newest first.
        fn all / all_sync(&self, py) -> Vec<Py<PyGoalView>> {
            prepare {}
            run(agent = self.inner.clone()) {
                let goals = agent.goals().all().await.map_err(to_pyerr)?;
                Python::attach(|py| {
                    goals
                        .iter()
                        .map(|g| PyGoalView::from_rust(py, g))
                        .collect::<PyResult<Vec<_>>>()
                })

            }
        }

        /// Goals in the given phase.
        fn in_phase / in_phase_sync(&self, py, phase: String) -> Vec<Py<PyGoalView>> {
            prepare {
                let phase = parse_goal_phase(&phase)?;

            }
            run(agent = self.inner.clone()) {
                let goals = agent.goals().in_phase(phase).await.map_err(to_pyerr)?;
                Python::attach(|py| {
                    goals
                        .iter()
                        .map(|g| PyGoalView::from_rust(py, g))
                        .collect::<PyResult<Vec<_>>>()
                })

            }
        }

        /// Active goals.
        fn active / active_sync(&self, py) -> Vec<Py<PyGoalView>> {
            prepare {}
            run(agent = self.inner.clone()) {
                let goals = agent.goals().active().await.map_err(to_pyerr)?;
                Python::attach(|py| {
                    goals
                        .iter()
                        .map(|g| PyGoalView::from_rust(py, g))
                        .collect::<PyResult<Vec<_>>>()
                })

            }
        }

        /// Planned goals.
        fn planned / planned_sync(&self, py) -> Vec<Py<PyGoalView>> {
            prepare {}
            run(agent = self.inner.clone()) {
                let goals = agent.goals().planned().await.map_err(to_pyerr)?;
                Python::attach(|py| {
                    goals
                        .iter()
                        .map(|g| PyGoalView::from_rust(py, g))
                        .collect::<PyResult<Vec<_>>>()
                })

            }
        }

        /// Completed goals.
        fn completed / completed_sync(&self, py) -> Vec<Py<PyGoalView>> {
            prepare {}
            run(agent = self.inner.clone()) {
                let goals = agent.goals().completed().await.map_err(to_pyerr)?;
                Python::attach(|py| {
                    goals
                        .iter()
                        .map(|g| PyGoalView::from_rust(py, g))
                        .collect::<PyResult<Vec<_>>>()
                })

            }
        }

        /// Fetch one goal by id, or `None`.
        fn get / get_sync(&self, py, goal_id: String) -> Py<PyAny> {
            prepare {}
            run(agent = self.inner.clone()) {
                let goal = agent.goals().get(&goal_id).await.map_err(to_pyerr)?;
                Python::attach(|py| match goal {
                    Some(g) => PyGoalView::from_rust(py, &g).map(|p| p.into_any()),
                    None => Ok(py.None()),
                })

            }
        }

        /// All tasks assigned to the agent.
        fn tasks / tasks_sync(&self, py) -> Vec<Py<PyTaskView>> {
            prepare {}
            run(agent = self.inner.clone()) {
                let tasks = agent.goals().tasks().await.map_err(to_pyerr)?;
                Python::attach(|py| {
                    tasks
                        .iter()
                        .map(|t| PyTaskView::from_rust(py, t))
                        .collect::<PyResult<Vec<_>>>()
                })

            }
        }

        /// Tasks in the given phase.
        fn tasks_in / tasks_in_sync(&self, py, phase: String) -> Vec<Py<PyTaskView>> {
            prepare {
                let phase = parse_task_phase(&phase)?;

            }
            run(agent = self.inner.clone()) {
                let tasks = agent.goals().tasks_in(phase).await.map_err(to_pyerr)?;
                Python::attach(|py| {
                    tasks
                        .iter()
                        .map(|t| PyTaskView::from_rust(py, t))
                        .collect::<PyResult<Vec<_>>>()
                })

            }
        }

        /// Tasks under a specific goal.
        fn tasks_of / tasks_of_sync(&self, py, goal_id: String) -> Vec<Py<PyTaskView>> {
            prepare {}
            run(agent = self.inner.clone()) {
                let tasks = agent.goals().tasks_of(&goal_id).await.map_err(to_pyerr)?;
                Python::attach(|py| {
                    tasks
                        .iter()
                        .map(|t| PyTaskView::from_rust(py, t))
                        .collect::<PyResult<Vec<_>>>()
                })

            }
        }

        /// Expand a goal's working context, or `None` if the goal is unknown.
        fn context / context_sync(&self, py, goal_id: String) -> Py<PyAny> {
            prepare {}
            run(agent = self.inner.clone()) {
                let ctx = agent.goals().context(&goal_id).await.map_err(to_pyerr)?;
                Python::attach(|py| match ctx {
                    Some(c) => PyGoalContext::from_rust(py, &c).map(|p| p.into_any()),
                    None => Ok(py.None()),
                })

            }
        }


        /// Start a goal. Returns `False` if the id doesn't resolve.
        fn start / start_sync(&self, py, goal_id: String) -> bool {
            prepare {}
            run(agent = self.inner.clone()) {
                agent.goals().start(&goal_id).await.map_err(to_pyerr)

            }
        }

        /// Abandon a goal.
        fn abandon / abandon_sync(&self, py, goal_id: String) -> bool {
            prepare {}
            run(agent = self.inner.clone()) {
                agent.goals().abandon(&goal_id).await.map_err(to_pyerr)

            }
        }

        /// Set a goal's free-form status string.
        fn set_status / set_status_sync(&self, py, goal_id: String, status: String) -> bool {
            prepare {}
            run(agent = self.inner.clone()) {
                agent
                    .goals()
                    .set_status(&goal_id, &status)
                    .await
                    .map_err(to_pyerr)

            }
        }

        /// Complete a goal, optionally recording a result.
        #[pyo3(signature = (goal_id, result=None))]
        fn complete / complete_sync(&self, py, goal_id: String, result: Option<Bound<'_, PyAny>>) -> bool {
            prepare {
                let result = opt_json(result)?;

            }
            run(agent = self.inner.clone()) {
                agent
                    .goals()
                    .complete(&goal_id, result)
                    .await
                    .map_err(to_pyerr)

            }
        }

        /// Start a task.
        fn start_task / start_task_sync(&self, py, task_id: String) -> bool {
            prepare {}
            run(agent = self.inner.clone()) {
                agent.goals().start_task(&task_id).await.map_err(to_pyerr)

            }
        }

        /// Mark a task blocked.
        fn block_task / block_task_sync(&self, py, task_id: String) -> bool {
            prepare {}
            run(agent = self.inner.clone()) {
                agent.goals().block_task(&task_id).await.map_err(to_pyerr)

            }
        }

        /// Complete a task.
        fn complete_task / complete_task_sync(&self, py, task_id: String) -> bool {
            prepare {}
            run(agent = self.inner.clone()) {
                agent
                    .goals()
                    .complete_task(&task_id)
                    .await
                    .map_err(to_pyerr)

            }
        }

        /// Set a task's free-form status string.
        fn set_task_status / set_task_status_sync(&self, py, task_id: String, status: String) -> bool {
            prepare {}
            run(agent = self.inner.clone()) {
                agent
                    .goals()
                    .set_task_status(&task_id, &status)
                    .await
                    .map_err(to_pyerr)

            }
        }


        /// Create a goal, returning its node id.
        #[pyo3(signature = (
            title, *, goal_id=None, description=None, status=None,
            metrics=None, guardrails=None, deadline=None, parent_goal_id=None
        ))]
        #[expect(
            clippy::too_many_arguments,
            reason = "mirrors the facade CreateGoalParams field set as keyword args"
        )]
        fn create / create_sync(&self, py, title: String, goal_id: Option<String>, description: Option<String>, status: Option<String>, metrics: Option<Bound<'_, PyAny>>, guardrails: Option<Bound<'_, PyAny>>, deadline: Option<Bound<'_, PyAny>>, parent_goal_id: Option<String>) -> NodeId {
            prepare {
                let deadline = deadline.map(|d| py_to_datetime(&d)).transpose()?;
                let params = CreateGoalParams {
                    goal_id,
                    title,
                    description,
                    status,
                    metrics: opt_json(metrics)?,
                    guardrails: opt_json(guardrails)?,
                    deadline,
                    parent_goal_id,
                    created_at: None,
                };

            }
            run(agent = self.inner.clone()) {
                agent.goals().create(params).await.map_err(to_pyerr)

            }
        }

        /// Create a task, returning its node id.
        #[pyo3(signature = (
            title, *, task_id=None, description=None, status=None, priority=None,
            goal_id=None, depends_on_task_id=None, subtask_of_task_id=None
        ))]
        #[expect(
            clippy::too_many_arguments,
            reason = "mirrors the facade CreateTaskParams field set as keyword args"
        )]
        fn create_task / create_task_sync(&self, py, title: String, task_id: Option<String>, description: Option<String>, status: Option<String>, priority: Option<f64>, goal_id: Option<String>, depends_on_task_id: Option<String>, subtask_of_task_id: Option<String>) -> NodeId {
            prepare {
                let params = CreateTaskParams {
                    task_id,
                    title,
                    description,
                    status,
                    priority,
                    goal_id,
                    depends_on_task_id,
                    subtask_of_task_id,
                    created_at: None,
                };

            }
            run(agent = self.inner.clone()) {
                agent.goals().create_task(params).await.map_err(to_pyerr)

            }
        }

    }
}
