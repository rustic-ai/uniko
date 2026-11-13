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
    RecordActionParams, RecordEpisodeParams, Scope, Visibility, fact_id_for_scoped,
};

use crate::convert::{self, params_to_rust};
use crate::data::PyData;
use crate::errors::to_pyerr;
use crate::goals::PyGoals;
use crate::logic::PyAssumeBuilder;
use crate::macros::pymethods_with_sync;
use crate::outputs::{
    PyAbductionResult, PyActionResult, PyAnswer, PyConsolidationReport, PyContextBundle,
    PyDeletionReport, PyFactUpsert, PyFinalizeReport, PyMatchedProcedure, PyRuleView,
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

pymethods_with_sync! {
    PyAgent;
    plain {
        /// The bound agent's participant id.
        #[getter]
        fn agent_id(&self) -> &str {
            self.inner.agent_id()
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

        fn __repr__(&self) -> String {
            format!("Agent(agent_id='{}')", self.inner.agent_id())
        }

    }
    bridged {
        /// Run the recall cascade for `query`, returning a ranked context bundle.
        fn recall / recall_sync(&self, py, query: String) -> Py<PyContextBundle> {
            prepare {}
            run(agent = self.inner.clone()) {
                let bundle = agent.recall(&query).await.map_err(to_pyerr)?;
                Python::attach(|py| PyContextBundle::from_rust(py, &bundle))

            }
        }

        /// Recall context for `question` and generate an answer with the
        /// configured LLM.
        ///
        /// Raises `ConfigError` if the instance was built without an LLM.
        fn answer / answer_sync(&self, py, question: String) -> Py<PyAnswer> {
            prepare {}
            run(agent = self.inner.clone()) {
                let answer = agent.answer(&question).await.map_err(to_pyerr)?;
                Python::attach(|py| PyAnswer::from_rust(py, &answer))

            }
        }

        /// Run a read-only Cypher query and return the rows as `list[dict]`.
        ///
        /// Rejects any non-read-only Cypher, so this can never mutate the graph.
        fn query / query_sync(&self, py, cypher: String) -> Py<PyAny> {
            prepare {}
            run(agent = self.inner.clone()) {
                let rows = agent.query(&cypher).await.map_err(to_pyerr)?;
                Python::attach(|py| convert::records_to_py(py, &rows))

            }
        }

        /// Recall scoped to `scope` (session / participant / time filters).
        fn recall_in / recall_in_sync(&self, py, query: String, scope: &PyScope) -> Py<PyContextBundle> {
            prepare {
                let scope = scope.snapshot();

            }
            run(agent = self.inner.clone()) {
                let bundle = agent.recall_in(&query, scope).await.map_err(to_pyerr)?;
                Python::attach(|py| PyContextBundle::from_rust(py, &bundle))

            }
        }

        /// Recall with per-call overrides of the instance's recall settings.
        ///
        /// `scope` defaults to an empty `Scope`. Each keyword left as `None`
        /// keeps the configured value. `reranker=True` needs a reranker enabled
        /// at build time; a `limit` above the reranker's `top_n` raises
        /// `ConfigError` while it is on.
        #[pyo3(signature = (query, scope=None, *, limit=None, token_budget=None, min_score=None, reranker=None, reference_ts=None))]
        #[allow(clippy::too_many_arguments)]
        fn recall_with / recall_with_sync(&self, py, query: String, scope: Option<&PyScope>, limit: Option<usize>, token_budget: Option<usize>, min_score: Option<f64>, reranker: Option<bool>, reference_ts: Option<&Bound<'_, PyAny>>) -> Py<PyContextBundle> {
            prepare {
                let (scope, overrides) = recall_with_args(
                    scope,
                    limit,
                    token_budget,
                    min_score,
                    reranker,
                    reference_ts,
                )?;

            }
            run(agent = self.inner.clone()) {
                let bundle = agent
                    .recall_with(&query, scope, &overrides)
                    .await
                    .map_err(to_pyerr)?;
                Python::attach(|py| PyContextBundle::from_rust(py, &bundle))

            }
        }

        /// Generate an answer with recall scoped to `scope`.
        fn answer_in / answer_in_sync(&self, py, question: String, scope: &PyScope) -> Py<PyAnswer> {
            prepare {
                let scope = scope.snapshot();

            }
            run(agent = self.inner.clone()) {
                let answer = agent.answer_in(&question, scope).await.map_err(to_pyerr)?;
                Python::attach(|py| PyAnswer::from_rust(py, &answer))

            }
        }

        /// Run read-only Cypher with `scope`'s allow-set bound as `$allow`.
        ///
        /// Returns raw rows with no visibility filtering — use `recall_in` when
        /// results must be policy-scoped.
        fn query_in / query_in_sync(&self, py, cypher: String, scope: &PyScope) -> Py<PyAny> {
            prepare {
                let scope = scope.snapshot();

            }
            run(agent = self.inner.clone()) {
                let rows = agent.query_in(&cypher, &scope).await.map_err(to_pyerr)?;
                Python::attach(|py| convert::records_to_py(py, &rows))

            }
        }

        /// Define a Locy derivation rule, returning its node id.
        fn define_rule / define_rule_sync(&self, py, name: String, source: String) -> NodeId {
            prepare {}
            run(agent = self.inner.clone()) {
                agent.define_rule(name, source).await.map_err(to_pyerr)

            }
        }

        /// Run a registered Locy rule, returning its `YIELD` rows as `list[dict]`.
        #[pyo3(signature = (name, return_cols, params=None))]
        fn run_rule / run_rule_sync(&self, py, name: String, return_cols: Vec<String>, params: Option<Bound<'_, PyDict>>) -> Py<PyAny> {
            prepare {
                let params: HashMap<_, _> = params_to_rust(params.as_ref())?;

            }
            run(agent = self.inner.clone()) {
                let cols: Vec<&str> = return_cols.iter().map(String::as_str).collect();
                let rows = agent
                    .run_rule(&name, &cols, params)
                    .await
                    .map_err(to_pyerr)?;
                Python::attach(|py| convert::records_to_py(py, &rows))

            }
        }

        /// Run an abductive query: find the minimal facts supporting a conclusion.
        #[pyo3(signature = (program, params=None))]
        fn abduce / abduce_sync(&self, py, program: String, params: Option<Bound<'_, PyDict>>) -> Py<PyAbductionResult> {
            prepare {
                let params = params_to_rust(params.as_ref())?;

            }
            run(agent = self.inner.clone()) {
                let result = agent.abduce(&program, params).await.map_err(to_pyerr)?;
                Python::attach(|py| PyAbductionResult::from_rust(py, &result))

            }
        }

        /// Hard-delete an entire conversation and everything it owns.
        fn delete_session / delete_session_sync(&self, py, session_id: String) -> Py<PyDeletionReport> {
            prepare {}
            run(agent = self.inner.clone()) {
                let report = agent.delete_session(&session_id).await.map_err(to_pyerr)?;
                Python::attach(|py| PyDeletionReport::from_rust(py, &report))

            }
        }

        /// GDPR erasure: remove a participant and the data they authored.
        fn forget_participant / forget_participant_sync(&self, py, participant_id: String) -> Py<PyDeletionReport> {
            prepare {}
            run(agent = self.inner.clone()) {
                let report = agent
                    .forget_participant(&participant_id)
                    .await
                    .map_err(to_pyerr)?;
                Python::attach(|py| PyDeletionReport::from_rust(py, &report))

            }
        }

        /// Build (or refresh) session-level retrieval surfaces for `session_id`
        /// without holding its `Session` handle.
        ///
        /// The backfill entry point for knowledge bases ingested before session
        /// chunking was wired into the facade. Unlike `Session.finalize()` this
        /// does not wait for streamed turns first.
        fn finalize_session / finalize_session_sync(&self, py, session_id: String) -> Py<PyFinalizeReport> {
            prepare {}
            run(agent = self.inner.clone()) {
                let report = agent
                    .finalize_session(&session_id)
                    .await
                    .map_err(to_pyerr)?;
                Python::attach(|py| PyFinalizeReport::from_rust(py, &report))

            }
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
        fn retire_source / retire_source_sync(&self, py, source_id: String) -> bool {
            prepare {}
            run(agent = self.inner.clone()) {
                agent.retire_source(&source_id).await.map_err(to_pyerr)

            }
        }

        /// Un-retire a source, making its current revision eligible again.
        fn restore_source / restore_source_sync(&self, py, source_id: String) -> bool {
            prepare {}
            run(agent = self.inner.clone()) {
                agent.restore_source(&source_id).await.map_err(to_pyerr)

            }
        }

        /// Compiles unprocessed Observations into Facts, reinforcing or
        /// invalidating prior beliefs and flagging entity drift. Always
        /// available — no streaming pipeline required.
        fn consolidate / consolidate_sync(&self, py) -> Py<PyConsolidationReport> {
            prepare {}
            run(agent = self.inner.clone()) {
                let stats = agent.consolidate().await.map_err(to_pyerr)?;
                Python::attach(|py| PyConsolidationReport::from_rust(py, &stats))

            }
        }

        /// Assert a fact directly, without an observation to derive it from.
        ///
        /// Facts are keyed by their triple, so asserting one that exists
        /// reinforces it. `visibility` (`"public"`, `"private:<id>"`,
        /// `"team:<id>"`, `"org:<id>"`) scopes it, for every assertion of the
        /// triple.
        #[pyo3(signature = (subject, predicate, object=None, *, observed_at=None, supporting_observation_ids=None, visibility=None))]
        #[allow(clippy::too_many_arguments)]
        fn assert_fact / assert_fact_sync(&self, py, subject: String, predicate: String, object: Option<String>, observed_at: Option<&Bound<'_, PyAny>>, supporting_observation_ids: Option<Vec<String>>, visibility: Option<String>) -> Py<PyFactUpsert> {
            prepare {
                let params = fact_args(
                    subject,
                    predicate,
                    object,
                    observed_at,
                    supporting_observation_ids,
                    visibility,
                )?;
                let fact_id = fact_id_for_scoped(
                    &params.subject,
                    &params.predicate,
                    params.object.as_deref(),
                    params
                        .visibility
                        .as_ref()
                        .map(Visibility::as_property)
                        .as_deref(),
                );

            }
            run(agent = self.inner.clone()) {
                let upsert = agent.assert_fact(params).await.map_err(to_pyerr)?;
                Python::attach(|py| PyFactUpsert::from_rust(py, &upsert, fact_id))

            }
        }

        /// Invalidate a fact by its `fact_id`, closing its validity interval,
        /// optionally naming the fact that replaces it.
        #[pyo3(signature = (fact_id, *, replacement_fact_id=None, reason=None))]
        fn invalidate_fact / invalidate_fact_sync(&self, py, fact_id: String, replacement_fact_id: Option<String>, reason: Option<String>) -> () {
            prepare {
                let params = InvalidateFactParams {
                    fact_id,
                    replacement_fact_id,
                    reason,
                    ..InvalidateFactParams::default()
                };

            }
            run(agent = self.inner.clone()) {
                agent.invalidate_fact(params).await.map_err(to_pyerr)

            }
        }

        /// Record an action this agent took (a tool call, an API request).
        ///
        /// `input` and `output` are any JSON-shaped value; an output too large
        /// to keep inline goes to an artifact.
        #[pyo3(signature = (action_type, *, action_id=None, input=None, output=None, status=None, started_at=None, ended_at=None, error=None, triggered_by_message_id=None, session_id=None, previous_action_id=None))]
        #[allow(clippy::too_many_arguments)]
        fn record_action / record_action_sync(&self, py, action_type: String, action_id: Option<String>, input: Option<&Bound<'_, PyAny>>, output: Option<&Bound<'_, PyAny>>, status: Option<String>, started_at: Option<&Bound<'_, PyAny>>, ended_at: Option<&Bound<'_, PyAny>>, error: Option<String>, triggered_by_message_id: Option<String>, session_id: Option<String>, previous_action_id: Option<String>) -> Py<PyActionResult> {
            prepare {
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

            }
            run(agent = self.inner.clone()) {
                let result = agent.record_action(params).await.map_err(to_pyerr)?;
                Python::attach(|py| PyActionResult::from_rust(py, &result))

            }
        }

        /// Record an episode: an outcome this agent experienced, which
        /// procedure promotion learns from. Returns its node id.
        #[pyo3(signature = (action_type, *, episode_id=None, outcome=None, state=None, delta=None, importance=None, timestamp=None, triggered_by_message_id=None, involved_action_ids=None))]
        #[allow(clippy::too_many_arguments)]
        fn record_episode / record_episode_sync(&self, py, action_type: String, episode_id: Option<String>, outcome: Option<String>, state: Option<&Bound<'_, PyAny>>, delta: Option<&Bound<'_, PyAny>>, importance: Option<f64>, timestamp: Option<&Bound<'_, PyAny>>, triggered_by_message_id: Option<String>, involved_action_ids: Option<Vec<String>>) -> NodeId {
            prepare {
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

            }
            run(agent = self.inner.clone()) {
                agent.record_episode(params).await.map_err(to_pyerr)

            }
        }

        /// Add an observation by hand, anchored to an existing message.
        /// Returns its node id. `visibility` scopes it, as for `assert_fact`.
        #[pyo3(signature = (message_id, content, subject, *, predicate=None, object=None, temporal_phrase=None, temporal_anchor=None, observed_at=None, confidence=None, visibility=None))]
        #[allow(clippy::too_many_arguments)]
        fn add_observation / add_observation_sync(&self, py, message_id: String, content: String, subject: String, predicate: Option<String>, object: Option<String>, temporal_phrase: Option<String>, temporal_anchor: Option<&Bound<'_, PyAny>>, observed_at: Option<&Bound<'_, PyAny>>, confidence: Option<f64>, visibility: Option<String>) -> NodeId {
            prepare {
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
                    visibility: parse_visibility(visibility)?,
                };

            }
            run(agent = self.inner.clone()) {
                agent.add_observation(params).await.map_err(to_pyerr)

            }
        }

        /// Set who may see a Fact or Observation: `"public"`, `"private:<id>"`,
        /// `"team:<id>"` or `"org:<id>"`. Returns False when no node has
        /// `node_id`.
        fn set_visibility / set_visibility_sync(&self, py, node_id: NodeId, visibility: String) -> bool {
            prepare {
                let visibility = Visibility::parse(&visibility).map_err(to_pyerr)?;

            }
            run(agent = self.inner.clone()) {
                agent
                    .set_visibility(node_id, visibility)
                    .await
                    .map_err(to_pyerr)

            }
        }

        /// Every rule in the store, stdlib ones included.
        fn list_rules / list_rules_sync(&self, py) -> Vec<Py<PyRuleView>> {
            prepare {}
            run(agent = self.inner.clone()) {
                let rules = agent.list_rules().await.map_err(to_pyerr)?;
                Python::attach(|py| {
                    rules
                        .iter()
                        .map(|r| PyRuleView::from_rust(py, r))
                        .collect::<PyResult<Vec<_>>>()
                })

            }
        }

        /// A rule's definition and state, or None when no rule has `name`.
        fn explain_rule / explain_rule_sync(&self, py, name: String) -> Option<Py<PyRuleView>> {
            prepare {}
            run(agent = self.inner.clone()) {
                let rule = agent.explain_rule(&name).await.map_err(to_pyerr)?;
                Python::attach(|py| rule.map(|r| PyRuleView::from_rust(py, &r)).transpose())

            }
        }

        /// Stop a rule running and mark it pruned. Returns False when no rule
        /// has `name`; raises `ConfigError` for a stdlib rule.
        fn delete_rule / delete_rule_sync(&self, py, name: String) -> bool {
            prepare {}
            run(agent = self.inner.clone()) {
                agent.delete_rule(&name).await.map_err(to_pyerr)

            }
        }

        /// Active procedures whose precondition holds in `state` (e.g.
        /// `{"last_action_type": "search"}`): what has tended to work next.
        fn match_procedures / match_procedures_sync(&self, py, state: HashMap<String, String>) -> Vec<Py<PyMatchedProcedure>> {
            prepare {}
            run(agent = self.inner.clone()) {
                let matched = agent.match_procedures(&state).await.map_err(to_pyerr)?;
                Python::attach(|py| {
                    matched
                        .iter()
                        .map(|m| PyMatchedProcedure::from_rust(py, m))
                        .collect::<PyResult<Vec<_>>>()
                })

            }
        }

        /// Record whether following a procedure succeeded.
        fn record_procedure_use / record_procedure_use_sync(&self, py, procedure_id: String, succeeded: bool) -> () {
            prepare {}
            run(agent = self.inner.clone()) {
                agent
                    .record_procedure_use(&procedure_id, succeeded)
                    .await
                    .map_err(to_pyerr)

            }
        }

        /// Session ids that have no session-level chunks yet.
        ///
        /// Drives the `finalize_session` backfill loop.
        fn unfinalized_session_ids / unfinalized_session_ids_sync(&self, py) -> Vec<String> {
            prepare {}
            run(agent = self.inner.clone()) {
                let ids = agent.unfinalized_session_ids().await.map_err(to_pyerr)?;
                Ok(ids)

            }
        }

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

/// The params for `assert_fact` / `assert_fact_sync`.
fn fact_args(
    subject: String,
    predicate: String,
    object: Option<String>,
    observed_at: Option<&Bound<'_, PyAny>>,
    supporting_observation_ids: Option<Vec<String>>,
    visibility: Option<String>,
) -> PyResult<AssertFactParams> {
    Ok(AssertFactParams {
        subject,
        predicate,
        object,
        observed_at: observed_at.map(convert::py_to_datetime).transpose()?,
        supporting_observation_ids: supporting_observation_ids.unwrap_or_default(),
        visibility: parse_visibility(visibility)?,
        ..AssertFactParams::default()
    })
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
