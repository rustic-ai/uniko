// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Industries Inc.

//! The [`PyAgent`] handle — the agent-scoped surface over recall, answer,
//! query, and conversation sessions.
//!
//! `Agent` is `Clone` and `Arc`-backed, so each async method clones it cheaply
//! and runs inside one `await`. The scoped variants (`recall_in`/`query_in`/
//! `answer_in`) arrive with the `Scope` type in the logic-surface phase.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use uniko_api::tools::{Agent, NodeId};

use crate::convert::{self, params_to_rust};
use crate::data::PyData;
use crate::errors::to_pyerr;
use crate::goals::PyGoals;
use crate::logic::PyAssumeBuilder;
use crate::macros::{bridge, bridge_sync};
use crate::outputs::{
    PyAbductionResult, PyAnswer, PyContextBundle, PyCycleStats, PyDeletionReport, PyFinalizeReport,
};
use crate::scope::PyScope;
use crate::session::PySession;

/// An agent-scoped handle over the cognitive-memory surface.
///
/// Obtained from [`Uniko.agent`](crate::facade::PyUniko). Binds an agent
/// identity to the store so recall / answer / query / session run without
/// re-passing it.
#[pyclass(name = "Agent", module = "uniko")]
pub struct PyAgent {
    inner: Agent,
}

impl PyAgent {
    /// Wrap an owned facade [`Agent`].
    pub(crate) fn wrap(agent: Agent) -> Self {
        Self { inner: agent }
    }
}

#[pymethods]
impl PyAgent {
    /// The bound agent's participant id.
    #[getter]
    fn agent_id(&self) -> &str {
        self.inner.agent_id()
    }

    /// Run the recall cascade for `query`, returning a ranked context bundle.
    fn recall<'py>(&self, py: Python<'py>, query: String) -> PyResult<Bound<'py, PyAny>> {
        bridge!(py, agent = self.inner.clone(), {
            let bundle = agent.recall(&query).await.map_err(to_pyerr)?;
            Python::attach(|py| PyContextBundle::from_rust(py, &bundle))
        })
    }

    /// Recall context for `question` and generate an answer with the
    /// configured LLM.
    ///
    /// Raises `ConfigError` if the instance was built without an LLM.
    fn answer<'py>(&self, py: Python<'py>, question: String) -> PyResult<Bound<'py, PyAny>> {
        bridge!(py, agent = self.inner.clone(), {
            let answer = agent.answer(&question).await.map_err(to_pyerr)?;
            Python::attach(|py| PyAnswer::from_rust(py, &answer))
        })
    }

    /// Run a read-only Cypher query and return the rows as `list[dict]`.
    ///
    /// Rejects any non-read-only Cypher, so this can never mutate the graph.
    fn query<'py>(&self, py: Python<'py>, cypher: String) -> PyResult<Bound<'py, PyAny>> {
        bridge!(py, agent = self.inner.clone(), {
            let rows = agent.query(&cypher).await.map_err(to_pyerr)?;
            Python::attach(|py| convert::records_to_py(py, &rows))
        })
    }

    /// Recall scoped to `scope` (session / participant / time filters).
    fn recall_in<'py>(
        &self,
        py: Python<'py>,
        query: String,
        scope: &PyScope,
    ) -> PyResult<Bound<'py, PyAny>> {
        let scope = scope.snapshot();
        bridge!(py, agent = self.inner.clone(), {
            let bundle = agent.recall_in(&query, scope).await.map_err(to_pyerr)?;
            Python::attach(|py| PyContextBundle::from_rust(py, &bundle))
        })
    }

    /// Generate an answer with recall scoped to `scope`.
    fn answer_in<'py>(
        &self,
        py: Python<'py>,
        question: String,
        scope: &PyScope,
    ) -> PyResult<Bound<'py, PyAny>> {
        let scope = scope.snapshot();
        bridge!(py, agent = self.inner.clone(), {
            let answer = agent.answer_in(&question, scope).await.map_err(to_pyerr)?;
            Python::attach(|py| PyAnswer::from_rust(py, &answer))
        })
    }

    /// Run read-only Cypher with `scope`'s allow-set bound as `$allow`.
    ///
    /// Returns raw rows with no visibility filtering — use `recall_in` when
    /// results must be policy-scoped.
    fn query_in<'py>(
        &self,
        py: Python<'py>,
        cypher: String,
        scope: &PyScope,
    ) -> PyResult<Bound<'py, PyAny>> {
        let scope = scope.snapshot();
        bridge!(py, agent = self.inner.clone(), {
            let rows = agent.query_in(&cypher, &scope).await.map_err(to_pyerr)?;
            Python::attach(|py| convert::records_to_py(py, &rows))
        })
    }

    /// Define a Locy derivation rule, returning its node id.
    fn define_rule<'py>(
        &self,
        py: Python<'py>,
        name: String,
        source: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        bridge!(py, agent = self.inner.clone(), {
            agent.define_rule(name, source).await.map_err(to_pyerr)
        })
    }

    /// Run a registered Locy rule, returning its `YIELD` rows as `list[dict]`.
    #[pyo3(signature = (name, return_cols, params=None))]
    fn run_rule<'py>(
        &self,
        py: Python<'py>,
        name: String,
        return_cols: Vec<String>,
        params: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let params: HashMap<_, _> = params_to_rust(params.as_ref())?;
        bridge!(py, agent = self.inner.clone(), {
            let cols: Vec<&str> = return_cols.iter().map(String::as_str).collect();
            let rows = agent
                .run_rule(&name, &cols, params)
                .await
                .map_err(to_pyerr)?;
            Python::attach(|py| convert::records_to_py(py, &rows))
        })
    }

    /// Run an abductive query: find the minimal facts supporting a conclusion.
    #[pyo3(signature = (program, params=None))]
    fn abduce<'py>(
        &self,
        py: Python<'py>,
        program: String,
        params: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let params = params_to_rust(params.as_ref())?;
        bridge!(py, agent = self.inner.clone(), {
            let result = agent.abduce(&program, params).await.map_err(to_pyerr)?;
            Python::attach(|py| PyAbductionResult::from_rust(py, &result))
        })
    }

    /// Begin a hypothetical (`ASSUME`) query. Finish with `then_query` + `run`.
    fn assume(&self, assume_block: String) -> PyAssumeBuilder {
        PyAssumeBuilder::new(self.inner.clone(), assume_block)
    }

    /// Open a [`Session`](crate::session::PySession) for feeding turns.
    fn session(&self, session_id: String) -> PySession {
        PySession::wrap(self.inner.session(session_id))
    }

    /// Addressed (by-id) retrieval of messages and artifacts.
    #[getter]
    fn data(&self) -> PyData {
        PyData::wrap(self.inner.clone())
    }

    /// The agent's goal/task lifecycle surface.
    #[getter]
    fn goals(&self) -> PyGoals {
        PyGoals::wrap(self.inner.clone())
    }

    /// Hard-delete an entire conversation and everything it owns.
    fn delete_session<'py>(
        &self,
        py: Python<'py>,
        session_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        bridge!(py, agent = self.inner.clone(), {
            let report = agent.delete_session(&session_id).await.map_err(to_pyerr)?;
            Python::attach(|py| PyDeletionReport::from_rust(py, &report))
        })
    }

    /// GDPR erasure: remove a participant and the data they authored.
    fn forget_participant<'py>(
        &self,
        py: Python<'py>,
        participant_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        bridge!(py, agent = self.inner.clone(), {
            let report = agent
                .forget_participant(&participant_id)
                .await
                .map_err(to_pyerr)?;
            Python::attach(|py| PyDeletionReport::from_rust(py, &report))
        })
    }

    // ── Blocking (`*_sync`) skins ────────────────────────────────────
    //
    // Each is its async sibling's body run to completion on the shared
    // runtime; see `bridge_sync!`. The return type is the body's resolved
    // value rather than an awaitable.

    /// Blocking variant of [`recall`](Self::recall).
    fn recall_sync(&self, py: Python<'_>, query: String) -> PyResult<Py<PyContextBundle>> {
        bridge_sync!(py, agent = self.inner.clone(), {
            let bundle = agent.recall(&query).await.map_err(to_pyerr)?;
            Python::attach(|py| PyContextBundle::from_rust(py, &bundle))
        })
    }

    /// Blocking variant of [`answer`](Self::answer).
    fn answer_sync(&self, py: Python<'_>, question: String) -> PyResult<Py<PyAnswer>> {
        bridge_sync!(py, agent = self.inner.clone(), {
            let answer = agent.answer(&question).await.map_err(to_pyerr)?;
            Python::attach(|py| PyAnswer::from_rust(py, &answer))
        })
    }

    /// Blocking variant of [`query`](Self::query).
    fn query_sync(&self, py: Python<'_>, cypher: String) -> PyResult<Py<PyAny>> {
        bridge_sync!(py, agent = self.inner.clone(), {
            let rows = agent.query(&cypher).await.map_err(to_pyerr)?;
            Python::attach(|py| convert::records_to_py(py, &rows))
        })
    }

    /// Blocking variant of [`recall_in`](Self::recall_in).
    fn recall_in_sync(
        &self,
        py: Python<'_>,
        query: String,
        scope: &PyScope,
    ) -> PyResult<Py<PyContextBundle>> {
        let scope = scope.snapshot();
        bridge_sync!(py, agent = self.inner.clone(), {
            let bundle = agent.recall_in(&query, scope).await.map_err(to_pyerr)?;
            Python::attach(|py| PyContextBundle::from_rust(py, &bundle))
        })
    }

    /// Blocking variant of [`answer_in`](Self::answer_in).
    fn answer_in_sync(
        &self,
        py: Python<'_>,
        question: String,
        scope: &PyScope,
    ) -> PyResult<Py<PyAnswer>> {
        let scope = scope.snapshot();
        bridge_sync!(py, agent = self.inner.clone(), {
            let answer = agent.answer_in(&question, scope).await.map_err(to_pyerr)?;
            Python::attach(|py| PyAnswer::from_rust(py, &answer))
        })
    }

    /// Blocking variant of [`query_in`](Self::query_in).
    fn query_in_sync(
        &self,
        py: Python<'_>,
        cypher: String,
        scope: &PyScope,
    ) -> PyResult<Py<PyAny>> {
        let scope = scope.snapshot();
        bridge_sync!(py, agent = self.inner.clone(), {
            let rows = agent.query_in(&cypher, &scope).await.map_err(to_pyerr)?;
            Python::attach(|py| convert::records_to_py(py, &rows))
        })
    }

    /// Blocking variant of [`define_rule`](Self::define_rule).
    fn define_rule_sync(&self, py: Python<'_>, name: String, source: String) -> PyResult<NodeId> {
        bridge_sync!(py, agent = self.inner.clone(), {
            agent.define_rule(name, source).await.map_err(to_pyerr)
        })
    }

    /// Blocking variant of [`run_rule`](Self::run_rule).
    #[pyo3(signature = (name, return_cols, params=None))]
    fn run_rule_sync(
        &self,
        py: Python<'_>,
        name: String,
        return_cols: Vec<String>,
        params: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let params: HashMap<_, _> = params_to_rust(params.as_ref())?;
        bridge_sync!(py, agent = self.inner.clone(), {
            let cols: Vec<&str> = return_cols.iter().map(String::as_str).collect();
            let rows = agent
                .run_rule(&name, &cols, params)
                .await
                .map_err(to_pyerr)?;
            Python::attach(|py| convert::records_to_py(py, &rows))
        })
    }

    /// Blocking variant of [`abduce`](Self::abduce).
    #[pyo3(signature = (program, params=None))]
    fn abduce_sync(
        &self,
        py: Python<'_>,
        program: String,
        params: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAbductionResult>> {
        let params = params_to_rust(params.as_ref())?;
        bridge_sync!(py, agent = self.inner.clone(), {
            let result = agent.abduce(&program, params).await.map_err(to_pyerr)?;
            Python::attach(|py| PyAbductionResult::from_rust(py, &result))
        })
    }

    /// Blocking variant of [`delete_session`](Self::delete_session).
    fn delete_session_sync(
        &self,
        py: Python<'_>,
        session_id: String,
    ) -> PyResult<Py<PyDeletionReport>> {
        bridge_sync!(py, agent = self.inner.clone(), {
            let report = agent.delete_session(&session_id).await.map_err(to_pyerr)?;
            Python::attach(|py| PyDeletionReport::from_rust(py, &report))
        })
    }

    /// Build (or refresh) session-level retrieval surfaces for `session_id`
    /// without holding its `Session` handle.
    ///
    /// The backfill entry point for knowledge bases ingested before session
    /// chunking was wired into the facade. Unlike `Session.finalize()` this
    /// does not wait for streamed turns first.
    fn finalize_session<'py>(
        &self,
        py: Python<'py>,
        session_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        bridge!(py, agent = self.inner.clone(), {
            let report = agent
                .finalize_session(&session_id)
                .await
                .map_err(to_pyerr)?;
            Python::attach(|py| PyFinalizeReport::from_rust(py, &report))
        })
    }

    fn finalize_session_sync(
        &self,
        py: Python<'_>,
        session_id: String,
    ) -> PyResult<Py<PyFinalizeReport>> {
        bridge_sync!(py, agent = self.inner.clone(), {
            let report = agent
                .finalize_session(&session_id)
                .await
                .map_err(to_pyerr)?;
            Python::attach(|py| PyFinalizeReport::from_rust(py, &report))
        })
    }

    /// Run one consolidation cycle for this agent, now.
    ///
    /// Compiles unprocessed Observations into Facts, reinforcing or
    /// invalidating prior beliefs and flagging entity drift. Always
    /// available — no streaming pipeline required.
    fn consolidate<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        bridge!(py, agent = self.inner.clone(), {
            let stats = agent.consolidate().await.map_err(to_pyerr)?;
            Python::attach(|py| PyCycleStats::from_rust(py, &stats))
        })
    }

    fn consolidate_sync(&self, py: Python<'_>) -> PyResult<Py<PyCycleStats>> {
        bridge_sync!(py, agent = self.inner.clone(), {
            let stats = agent.consolidate().await.map_err(to_pyerr)?;
            Python::attach(|py| PyCycleStats::from_rust(py, &stats))
        })
    }

    /// Session ids that have no session-level chunks yet.
    ///
    /// Drives the `finalize_session` backfill loop.
    fn unfinalized_session_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        bridge!(py, agent = self.inner.clone(), {
            let ids = agent.unfinalized_session_ids().await.map_err(to_pyerr)?;
            Ok(ids)
        })
    }

    fn unfinalized_session_ids_sync(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        bridge_sync!(py, agent = self.inner.clone(), {
            let ids = agent.unfinalized_session_ids().await.map_err(to_pyerr)?;
            Ok(ids)
        })
    }

    /// Blocking variant of [`forget_participant`](Self::forget_participant).
    fn forget_participant_sync(
        &self,
        py: Python<'_>,
        participant_id: String,
    ) -> PyResult<Py<PyDeletionReport>> {
        bridge_sync!(py, agent = self.inner.clone(), {
            let report = agent
                .forget_participant(&participant_id)
                .await
                .map_err(to_pyerr)?;
            Python::attach(|py| PyDeletionReport::from_rust(py, &report))
        })
    }

    fn __repr__(&self) -> String {
        format!("Agent(agent_id='{}')", self.inner.agent_id())
    }
}
