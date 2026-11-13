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
use uniko_api::tools::{
    AddObservationParams, Agent, AssertFactParams, InvalidateFactParams, NodeId, RecallOverrides,
    RecordActionParams, RecordEpisodeParams, Scope, Visibility, fact_id_for,
};

use crate::convert::{self, params_to_rust};
use crate::data::PyData;
use crate::errors::to_pyerr;
use crate::goals::PyGoals;
use crate::logic::PyAssumeBuilder;
use crate::macros::{bridge, bridge_sync};
use crate::outputs::{
    PyAbductionResult, PyActionResult, PyAnswer, PyConsolidationReport, PyContextBundle,
    PyDeletionReport, PyFactUpsert, PyFinalizeReport, PyRuleView,
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

    /// Recall with per-call overrides of the instance's recall settings.
    ///
    /// `scope` defaults to an empty `Scope`. Each keyword left as `None`
    /// keeps the configured value. `reranker=True` needs a reranker enabled
    /// at build time; a `limit` above the reranker's `top_n` raises
    /// `ConfigError` while it is on.
    #[pyo3(signature = (query, scope=None, *, limit=None, token_budget=None, min_score=None, reranker=None, reference_ts=None))]
    #[allow(clippy::too_many_arguments)]
    fn recall_with<'py>(
        &self,
        py: Python<'py>,
        query: String,
        scope: Option<&PyScope>,
        limit: Option<usize>,
        token_budget: Option<usize>,
        min_score: Option<f64>,
        reranker: Option<bool>,
        reference_ts: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (scope, overrides) = recall_with_args(
            scope,
            limit,
            token_budget,
            min_score,
            reranker,
            reference_ts,
        )?;
        bridge!(py, agent = self.inner.clone(), {
            let bundle = agent
                .recall_with(&query, scope, &overrides)
                .await
                .map_err(to_pyerr)?;
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
    /// Blocking variant of [`retire_source`](Self::retire_source).
    fn retire_source_sync(&self, py: Python<'_>, source_id: String) -> PyResult<bool> {
        bridge_sync!(py, agent = self.inner.clone(), {
            agent.retire_source(&source_id).await.map_err(to_pyerr)
        })
    }

    /// Blocking variant of [`restore_source`](Self::restore_source).
    fn restore_source_sync(&self, py: Python<'_>, source_id: String) -> PyResult<bool> {
        bridge_sync!(py, agent = self.inner.clone(), {
            agent.restore_source(&source_id).await.map_err(to_pyerr)
        })
    }

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

    /// Blocking variant of [`recall_with`](Self::recall_with).
    #[pyo3(signature = (query, scope=None, *, limit=None, token_budget=None, min_score=None, reranker=None, reference_ts=None))]
    #[allow(clippy::too_many_arguments)]
    fn recall_with_sync(
        &self,
        py: Python<'_>,
        query: String,
        scope: Option<&PyScope>,
        limit: Option<usize>,
        token_budget: Option<usize>,
        min_score: Option<f64>,
        reranker: Option<bool>,
        reference_ts: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyContextBundle>> {
        let (scope, overrides) = recall_with_args(
            scope,
            limit,
            token_budget,
            min_score,
            reranker,
            reference_ts,
        )?;
        bridge_sync!(py, agent = self.inner.clone(), {
            let bundle = agent
                .recall_with(&query, scope, &overrides)
                .await
                .map_err(to_pyerr)?;
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
    /// Retire a logical source: no revision of it grounds a current answer
    /// (issue #41).
    ///
    /// Nothing is deleted, so a past result stays attributable — reach it
    /// with `Scope().include_superseded()`. Retirement is recorded on the
    /// source, not its content, so retiring one source never affects another
    /// that merely shares identical bytes.
    ///
    /// Resolves to `False` when no such source exists.
    fn retire_source<'py>(
        &self,
        py: Python<'py>,
        source_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        bridge!(py, agent = self.inner.clone(), {
            agent.retire_source(&source_id).await.map_err(to_pyerr)
        })
    }

    /// Un-retire a source, making its current revision eligible again.
    fn restore_source<'py>(
        &self,
        py: Python<'py>,
        source_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        bridge!(py, agent = self.inner.clone(), {
            agent.restore_source(&source_id).await.map_err(to_pyerr)
        })
    }

    /// Compiles unprocessed Observations into Facts, reinforcing or
    /// invalidating prior beliefs and flagging entity drift. Always
    /// available — no streaming pipeline required.
    fn consolidate<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        bridge!(py, agent = self.inner.clone(), {
            let stats = agent.consolidate().await.map_err(to_pyerr)?;
            Python::attach(|py| PyConsolidationReport::from_rust(py, &stats))
        })
    }

    fn consolidate_sync(&self, py: Python<'_>) -> PyResult<Py<PyConsolidationReport>> {
        bridge_sync!(py, agent = self.inner.clone(), {
            let stats = agent.consolidate().await.map_err(to_pyerr)?;
            Python::attach(|py| PyConsolidationReport::from_rust(py, &stats))
        })
    }

    /// Assert a fact directly, without an observation to derive it from.
    ///
    /// Facts are keyed by their triple, so asserting one that exists
    /// reinforces it. `visibility` (`"public"`, `"private:<id>"`,
    /// `"team:<id>"`, `"org:<id>"`) scopes it, for every assertion of the
    /// triple.
    #[pyo3(signature = (subject, predicate, object=None, *, observed_at=None, supporting_observation_ids=None, visibility=None))]
    #[allow(clippy::too_many_arguments)]
    fn assert_fact<'py>(
        &self,
        py: Python<'py>,
        subject: String,
        predicate: String,
        object: Option<String>,
        observed_at: Option<&Bound<'_, PyAny>>,
        supporting_observation_ids: Option<Vec<String>>,
        visibility: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (params, visibility) = fact_args(
            subject,
            predicate,
            object,
            observed_at,
            supporting_observation_ids,
            visibility,
        )?;
        let fact_id = fact_id_for(&params.subject, &params.predicate, params.object.as_deref());
        bridge!(py, agent = self.inner.clone(), {
            let upsert = agent.assert_fact(params).await.map_err(to_pyerr)?;
            if let Some(v) = visibility {
                agent
                    .set_visibility(upsert.node_id, v)
                    .await
                    .map_err(to_pyerr)?;
            }
            Python::attach(|py| PyFactUpsert::from_rust(py, &upsert, fact_id))
        })
    }

    /// Invalidate a fact by its `fact_id`, closing its validity interval,
    /// optionally naming the fact that replaces it.
    #[pyo3(signature = (fact_id, *, replacement_fact_id=None, reason=None))]
    fn invalidate_fact<'py>(
        &self,
        py: Python<'py>,
        fact_id: String,
        replacement_fact_id: Option<String>,
        reason: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let params = InvalidateFactParams {
            fact_id,
            replacement_fact_id,
            reason,
            ..InvalidateFactParams::default()
        };
        bridge!(py, agent = self.inner.clone(), {
            agent.invalidate_fact(params).await.map_err(to_pyerr)
        })
    }

    /// Record an action this agent took (a tool call, an API request).
    ///
    /// `input` and `output` are any JSON-shaped value; an output too large
    /// to keep inline goes to an artifact.
    #[pyo3(signature = (action_type, *, action_id=None, input=None, output=None, status=None, started_at=None, ended_at=None, error=None, triggered_by_message_id=None, session_id=None, previous_action_id=None))]
    #[allow(clippy::too_many_arguments)]
    fn record_action<'py>(
        &self,
        py: Python<'py>,
        action_type: String,
        action_id: Option<String>,
        input: Option<&Bound<'_, PyAny>>,
        output: Option<&Bound<'_, PyAny>>,
        status: Option<String>,
        started_at: Option<&Bound<'_, PyAny>>,
        ended_at: Option<&Bound<'_, PyAny>>,
        error: Option<String>,
        triggered_by_message_id: Option<String>,
        session_id: Option<String>,
        previous_action_id: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let params = action_args(
            action_type,
            action_id,
            input,
            output,
            status,
            started_at,
            ended_at,
            error,
            triggered_by_message_id,
            session_id,
            previous_action_id,
        )?;
        bridge!(py, agent = self.inner.clone(), {
            let result = agent.record_action(params).await.map_err(to_pyerr)?;
            Python::attach(|py| PyActionResult::from_rust(py, &result))
        })
    }

    /// Record an episode: an outcome this agent experienced, which
    /// procedure promotion learns from. Returns its node id.
    #[pyo3(signature = (action_type, *, episode_id=None, outcome=None, state=None, delta=None, importance=None, timestamp=None, triggered_by_message_id=None, involved_action_ids=None))]
    #[allow(clippy::too_many_arguments)]
    fn record_episode<'py>(
        &self,
        py: Python<'py>,
        action_type: String,
        episode_id: Option<String>,
        outcome: Option<String>,
        state: Option<&Bound<'_, PyAny>>,
        delta: Option<&Bound<'_, PyAny>>,
        importance: Option<f64>,
        timestamp: Option<&Bound<'_, PyAny>>,
        triggered_by_message_id: Option<String>,
        involved_action_ids: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let params = RecordEpisodeParams {
            episode_id,
            action_type,
            outcome,
            state: state.map(convert::py_to_json).transpose()?,
            delta: delta.map(convert::py_to_json).transpose()?,
            importance,
            timestamp: timestamp.map(convert::py_to_datetime).transpose()?,
            triggered_by_message_id,
            involved_action_ids: involved_action_ids.unwrap_or_default(),
        };
        bridge!(py, agent = self.inner.clone(), {
            agent.record_episode(params).await.map_err(to_pyerr)
        })
    }

    /// Add an observation by hand, anchored to an existing message.
    /// Returns its node id. `visibility` scopes it, as for `assert_fact`.
    #[pyo3(signature = (message_id, content, subject, *, predicate=None, object=None, temporal_phrase=None, temporal_anchor=None, observed_at=None, confidence=None, visibility=None))]
    #[allow(clippy::too_many_arguments)]
    fn add_observation<'py>(
        &self,
        py: Python<'py>,
        message_id: String,
        content: String,
        subject: String,
        predicate: Option<String>,
        object: Option<String>,
        temporal_phrase: Option<String>,
        temporal_anchor: Option<&Bound<'_, PyAny>>,
        observed_at: Option<&Bound<'_, PyAny>>,
        confidence: Option<f64>,
        visibility: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let params = AddObservationParams {
            message_id,
            content,
            subject,
            predicate,
            object,
            temporal_phrase,
            temporal_anchor: temporal_anchor.map(convert::py_to_datetime).transpose()?,
            observed_at: observed_at.map(convert::py_to_datetime).transpose()?,
            confidence,
        };
        let visibility = parse_visibility(visibility)?;
        bridge!(py, agent = self.inner.clone(), {
            let nid = agent.add_observation(params).await.map_err(to_pyerr)?;
            if let Some(v) = visibility {
                agent.set_visibility(nid, v).await.map_err(to_pyerr)?;
            }
            Ok(nid)
        })
    }

    /// Set who may see a Fact or Observation: `"public"`, `"private:<id>"`,
    /// `"team:<id>"` or `"org:<id>"`. Returns False when no node has
    /// `node_id`.
    fn set_visibility<'py>(
        &self,
        py: Python<'py>,
        node_id: NodeId,
        visibility: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let visibility = Visibility::parse(&visibility).map_err(to_pyerr)?;
        bridge!(py, agent = self.inner.clone(), {
            agent
                .set_visibility(node_id, visibility)
                .await
                .map_err(to_pyerr)
        })
    }

    /// Every rule in the store, stdlib ones included.
    fn list_rules<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        bridge!(py, agent = self.inner.clone(), {
            let rules = agent.list_rules().await.map_err(to_pyerr)?;
            Python::attach(|py| {
                rules
                    .iter()
                    .map(|r| PyRuleView::from_rust(py, r))
                    .collect::<PyResult<Vec<_>>>()
            })
        })
    }

    /// A rule's definition and state, or None when no rule has `name`.
    fn explain_rule<'py>(&self, py: Python<'py>, name: String) -> PyResult<Bound<'py, PyAny>> {
        bridge!(py, agent = self.inner.clone(), {
            let rule = agent.explain_rule(&name).await.map_err(to_pyerr)?;
            Python::attach(|py| rule.map(|r| PyRuleView::from_rust(py, &r)).transpose())
        })
    }

    /// Stop a rule running and mark it pruned. Returns False when no rule
    /// has `name`; raises `ConfigError` for a stdlib rule.
    fn delete_rule<'py>(&self, py: Python<'py>, name: String) -> PyResult<Bound<'py, PyAny>> {
        bridge!(py, agent = self.inner.clone(), {
            agent.delete_rule(&name).await.map_err(to_pyerr)
        })
    }

    /// Blocking variant of [`assert_fact`](Self::assert_fact).
    #[pyo3(signature = (subject, predicate, object=None, *, observed_at=None, supporting_observation_ids=None, visibility=None))]
    #[allow(clippy::too_many_arguments)]
    fn assert_fact_sync(
        &self,
        py: Python<'_>,
        subject: String,
        predicate: String,
        object: Option<String>,
        observed_at: Option<&Bound<'_, PyAny>>,
        supporting_observation_ids: Option<Vec<String>>,
        visibility: Option<String>,
    ) -> PyResult<Py<PyFactUpsert>> {
        let (params, visibility) = fact_args(
            subject,
            predicate,
            object,
            observed_at,
            supporting_observation_ids,
            visibility,
        )?;
        let fact_id = fact_id_for(&params.subject, &params.predicate, params.object.as_deref());
        bridge_sync!(py, agent = self.inner.clone(), {
            let upsert = agent.assert_fact(params).await.map_err(to_pyerr)?;
            if let Some(v) = visibility {
                agent
                    .set_visibility(upsert.node_id, v)
                    .await
                    .map_err(to_pyerr)?;
            }
            Python::attach(|py| PyFactUpsert::from_rust(py, &upsert, fact_id))
        })
    }

    /// Blocking variant of [`invalidate_fact`](Self::invalidate_fact).
    #[pyo3(signature = (fact_id, *, replacement_fact_id=None, reason=None))]
    fn invalidate_fact_sync(
        &self,
        py: Python<'_>,
        fact_id: String,
        replacement_fact_id: Option<String>,
        reason: Option<String>,
    ) -> PyResult<()> {
        let params = InvalidateFactParams {
            fact_id,
            replacement_fact_id,
            reason,
            ..InvalidateFactParams::default()
        };
        bridge_sync!(py, agent = self.inner.clone(), {
            agent.invalidate_fact(params).await.map_err(to_pyerr)
        })
    }

    /// Blocking variant of [`record_action`](Self::record_action).
    #[pyo3(signature = (action_type, *, action_id=None, input=None, output=None, status=None, started_at=None, ended_at=None, error=None, triggered_by_message_id=None, session_id=None, previous_action_id=None))]
    #[allow(clippy::too_many_arguments)]
    fn record_action_sync(
        &self,
        py: Python<'_>,
        action_type: String,
        action_id: Option<String>,
        input: Option<&Bound<'_, PyAny>>,
        output: Option<&Bound<'_, PyAny>>,
        status: Option<String>,
        started_at: Option<&Bound<'_, PyAny>>,
        ended_at: Option<&Bound<'_, PyAny>>,
        error: Option<String>,
        triggered_by_message_id: Option<String>,
        session_id: Option<String>,
        previous_action_id: Option<String>,
    ) -> PyResult<Py<PyActionResult>> {
        let params = action_args(
            action_type,
            action_id,
            input,
            output,
            status,
            started_at,
            ended_at,
            error,
            triggered_by_message_id,
            session_id,
            previous_action_id,
        )?;
        bridge_sync!(py, agent = self.inner.clone(), {
            let result = agent.record_action(params).await.map_err(to_pyerr)?;
            Python::attach(|py| PyActionResult::from_rust(py, &result))
        })
    }

    /// Blocking variant of [`record_episode`](Self::record_episode).
    #[pyo3(signature = (action_type, *, episode_id=None, outcome=None, state=None, delta=None, importance=None, timestamp=None, triggered_by_message_id=None, involved_action_ids=None))]
    #[allow(clippy::too_many_arguments)]
    fn record_episode_sync(
        &self,
        py: Python<'_>,
        action_type: String,
        episode_id: Option<String>,
        outcome: Option<String>,
        state: Option<&Bound<'_, PyAny>>,
        delta: Option<&Bound<'_, PyAny>>,
        importance: Option<f64>,
        timestamp: Option<&Bound<'_, PyAny>>,
        triggered_by_message_id: Option<String>,
        involved_action_ids: Option<Vec<String>>,
    ) -> PyResult<NodeId> {
        let params = RecordEpisodeParams {
            episode_id,
            action_type,
            outcome,
            state: state.map(convert::py_to_json).transpose()?,
            delta: delta.map(convert::py_to_json).transpose()?,
            importance,
            timestamp: timestamp.map(convert::py_to_datetime).transpose()?,
            triggered_by_message_id,
            involved_action_ids: involved_action_ids.unwrap_or_default(),
        };
        bridge_sync!(py, agent = self.inner.clone(), {
            agent.record_episode(params).await.map_err(to_pyerr)
        })
    }

    /// Blocking variant of [`add_observation`](Self::add_observation).
    #[pyo3(signature = (message_id, content, subject, *, predicate=None, object=None, temporal_phrase=None, temporal_anchor=None, observed_at=None, confidence=None, visibility=None))]
    #[allow(clippy::too_many_arguments)]
    fn add_observation_sync(
        &self,
        py: Python<'_>,
        message_id: String,
        content: String,
        subject: String,
        predicate: Option<String>,
        object: Option<String>,
        temporal_phrase: Option<String>,
        temporal_anchor: Option<&Bound<'_, PyAny>>,
        observed_at: Option<&Bound<'_, PyAny>>,
        confidence: Option<f64>,
        visibility: Option<String>,
    ) -> PyResult<NodeId> {
        let params = AddObservationParams {
            message_id,
            content,
            subject,
            predicate,
            object,
            temporal_phrase,
            temporal_anchor: temporal_anchor.map(convert::py_to_datetime).transpose()?,
            observed_at: observed_at.map(convert::py_to_datetime).transpose()?,
            confidence,
        };
        let visibility = parse_visibility(visibility)?;
        bridge_sync!(py, agent = self.inner.clone(), {
            let nid = agent.add_observation(params).await.map_err(to_pyerr)?;
            if let Some(v) = visibility {
                agent.set_visibility(nid, v).await.map_err(to_pyerr)?;
            }
            Ok(nid)
        })
    }

    /// Blocking variant of [`set_visibility`](Self::set_visibility).
    fn set_visibility_sync(
        &self,
        py: Python<'_>,
        node_id: NodeId,
        visibility: String,
    ) -> PyResult<bool> {
        let visibility = Visibility::parse(&visibility).map_err(to_pyerr)?;
        bridge_sync!(py, agent = self.inner.clone(), {
            agent
                .set_visibility(node_id, visibility)
                .await
                .map_err(to_pyerr)
        })
    }

    /// Blocking variant of [`list_rules`](Self::list_rules).
    fn list_rules_sync(&self, py: Python<'_>) -> PyResult<Vec<Py<PyRuleView>>> {
        bridge_sync!(py, agent = self.inner.clone(), {
            let rules = agent.list_rules().await.map_err(to_pyerr)?;
            Python::attach(|py| {
                rules
                    .iter()
                    .map(|r| PyRuleView::from_rust(py, r))
                    .collect::<PyResult<Vec<_>>>()
            })
        })
    }

    /// Blocking variant of [`explain_rule`](Self::explain_rule).
    fn explain_rule_sync(&self, py: Python<'_>, name: String) -> PyResult<Option<Py<PyRuleView>>> {
        bridge_sync!(py, agent = self.inner.clone(), {
            let rule = agent.explain_rule(&name).await.map_err(to_pyerr)?;
            Python::attach(|py| rule.map(|r| PyRuleView::from_rust(py, &r)).transpose())
        })
    }

    /// Blocking variant of [`delete_rule`](Self::delete_rule).
    fn delete_rule_sync(&self, py: Python<'_>, name: String) -> PyResult<bool> {
        bridge_sync!(py, agent = self.inner.clone(), {
            agent.delete_rule(&name).await.map_err(to_pyerr)
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

/// The `Scope` and `RecallOverrides` for `recall_with` / `recall_with_sync`.
fn recall_with_args(
    scope: Option<&PyScope>,
    limit: Option<usize>,
    token_budget: Option<usize>,
    min_score: Option<f64>,
    reranker: Option<bool>,
    reference_ts: Option<&Bound<'_, PyAny>>,
) -> PyResult<(Scope, RecallOverrides)> {
    let scope = scope.map(PyScope::snapshot).unwrap_or_default();
    let mut overrides = RecallOverrides::default();
    overrides.limit = limit;
    overrides.token_budget = token_budget;
    overrides.min_score = min_score;
    overrides.reranker = reranker;
    overrides.reference_ts = reference_ts.map(convert::py_to_datetime).transpose()?;
    Ok((scope, overrides))
}

/// A visibility string as a [`Visibility`], or `None`.
fn parse_visibility(visibility: Option<String>) -> PyResult<Option<Visibility>> {
    visibility
        .map(|v| Visibility::parse(&v).map_err(to_pyerr))
        .transpose()
}

/// The params and visibility for `assert_fact` / `assert_fact_sync`.
fn fact_args(
    subject: String,
    predicate: String,
    object: Option<String>,
    observed_at: Option<&Bound<'_, PyAny>>,
    supporting_observation_ids: Option<Vec<String>>,
    visibility: Option<String>,
) -> PyResult<(AssertFactParams, Option<Visibility>)> {
    let params = AssertFactParams {
        subject,
        predicate,
        object,
        observed_at: observed_at.map(convert::py_to_datetime).transpose()?,
        supporting_observation_ids: supporting_observation_ids.unwrap_or_default(),
        ..AssertFactParams::default()
    };
    Ok((params, parse_visibility(visibility)?))
}

/// The params for `record_action` on `Agent` and `Session`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn action_args(
    action_type: String,
    action_id: Option<String>,
    input: Option<&Bound<'_, PyAny>>,
    output: Option<&Bound<'_, PyAny>>,
    status: Option<String>,
    started_at: Option<&Bound<'_, PyAny>>,
    ended_at: Option<&Bound<'_, PyAny>>,
    error: Option<String>,
    triggered_by_message_id: Option<String>,
    session_id: Option<String>,
    previous_action_id: Option<String>,
) -> PyResult<RecordActionParams> {
    Ok(RecordActionParams {
        action_id,
        action_type,
        input: input.map(convert::py_to_json).transpose()?,
        output: output.map(convert::py_to_json).transpose()?,
        status,
        started_at: started_at.map(convert::py_to_datetime).transpose()?,
        ended_at: ended_at.map(convert::py_to_datetime).transpose()?,
        error,
        triggered_by_message_id,
        session_id,
        previous_action_id,
    })
}
