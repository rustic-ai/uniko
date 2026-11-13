// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Industries Inc.

//! The [`PySession`] conversation handle and its [`PyTurn`] input builder.
//!
//! `Session` is the one facade handle that is neither `Clone` nor `&self` for
//! its primary verb (`observe` takes `&mut self`), and it owns cross-turn
//! state. `PySession` therefore holds it behind an `Arc<tokio::sync::Mutex>`
//! and locks across the `await` — serializing turns, which matches the
//! documented single-threaded contract. The external `session_id` is cached
//! alongside so the synchronous getter needs no lock.

use std::sync::Arc;

use pyo3::prelude::*;
use tokio::sync::Mutex;
use uniko_api::tools::{Session, Turn};

use crate::agent::action_args;
use crate::convert;
use crate::errors::to_pyerr;
use crate::ingest::PyIngestSource;
use crate::macros::{bridge, bridge_sync, pymethods_with_sync};
use crate::outputs::{
    PyActionResult, PyDeletionReport, PyFinalizeReport, PyIngestOutcome, PyObserveResult,
    PySummarizeReport, PyUnitResult,
};

/// One conversation turn to feed into a [`PySession`].
///
/// Construct with `Turn(sender_id, content)` and refine with the chainable
/// setters. A turn is reusable: feeding it does not consume the Python object.
#[pyclass(name = "Turn", module = "uniko")]
pub struct PyTurn {
    // `Option` so a consuming builder setter can take-apply-replace; always
    // `Some` between calls. A plain `std::sync::Mutex` is correct — no `.await`
    // is ever held across this lock.
    inner: std::sync::Mutex<Option<Turn>>,
}

impl PyTurn {
    /// Apply a consuming `Turn` builder method in place.
    fn map(&self, f: impl FnOnce(Turn) -> Turn) {
        let mut guard = self.inner.lock().expect("Turn mutex poisoned");
        if let Some(turn) = guard.take() {
            *guard = Some(f(turn));
        }
    }

    /// A cloned snapshot of the current turn, for ingest.
    pub(crate) fn snapshot(&self) -> PyResult<Turn> {
        self.inner
            .lock()
            .expect("Turn mutex poisoned")
            .clone()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Turn is in an invalid state"))
    }
}

#[pymethods]
impl PyTurn {
    /// A text turn from `sender_id` carrying `content`, stamped now.
    #[new]
    fn new(sender_id: String, content: String) -> Self {
        Self {
            inner: std::sync::Mutex::new(Some(Turn::new(sender_id, content))),
        }
    }

    /// Set an explicit message id for idempotent ingest.
    fn id<'py>(slf: PyRef<'py, Self>, message_id: String) -> PyRef<'py, Self> {
        slf.map(|t| t.id(message_id));
        slf
    }

    /// Tag this turn with the caller's own record category (issue #39).
    ///
    /// Typed provenance: recall can filter on it, and it never enters the
    /// searchable text — so a consumer no longer has to encode a class into
    /// the prose and parse it back out of results.
    fn category<'py>(slf: PyRef<'py, Self>, category: String) -> PyRef<'py, Self> {
        slf.map(|t| t.category(category));
        slf
    }

    /// Scope who may see this turn: `"public"`, `"private:<id>"`,
    /// `"team:<id>"` or `"org:<id>"`. Covers its message, chunks,
    /// observations and the facts consolidated from them, not attachments.
    fn visibility<'py>(slf: PyRef<'py, Self>, visibility: &str) -> PyResult<PyRef<'py, Self>> {
        let visibility =
            uniko_api::tools::Visibility::parse(visibility).map_err(crate::errors::to_pyerr)?;
        slf.map(|t| t.visibility(visibility));
        Ok(slf)
    }

    /// Attribute this turn to a stable logical source id (issue #39).
    fn source<'py>(slf: PyRef<'py, Self>, source_id: String) -> PyRef<'py, Self> {
        slf.map(|t| t.source(source_id));
        slf
    }

    /// Declare which revision of that source this turn reflects (issue #41).
    fn revision<'py>(slf: PyRef<'py, Self>, revision_id: String) -> PyRef<'py, Self> {
        slf.map(|t| t.revision(revision_id));
        slf
    }

    /// Override the content type (defaults to `"text"`).
    fn content_type<'py>(slf: PyRef<'py, Self>, content_type: String) -> PyRef<'py, Self> {
        slf.map(|t| t.content_type(content_type));
        slf
    }

    /// Set the send time (a Python `datetime`; defaults to now).
    fn at<'py>(slf: PyRef<'py, Self>, timestamp: &Bound<'_, PyAny>) -> PyResult<PyRef<'py, Self>> {
        let ts = convert::py_to_datetime(timestamp)?;
        slf.map(|t| t.at(ts));
        Ok(slf)
    }

    /// Set explicit recipient participant ids.
    fn addressed_to<'py>(slf: PyRef<'py, Self>, recipients: Vec<String>) -> PyRef<'py, Self> {
        slf.map(|t| t.addressed_to(recipients));
        slf
    }

    /// Attach one arbitrary metadata entry forwarded to ingest.
    fn metadata<'py>(
        slf: PyRef<'py, Self>,
        key: String,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<PyRef<'py, Self>> {
        let json = convert::py_to_json(value)?;
        slf.map(move |t| t.metadata(key, json));
        Ok(slf)
    }

    /// Attach a document/file shared in this turn.
    fn attach<'py>(slf: PyRef<'py, Self>, source: &PyIngestSource) -> PyResult<PyRef<'py, Self>> {
        let src = source.snapshot()?;
        slf.map(|t| t.attach(src));
        Ok(slf)
    }

    /// Attach several documents/files at once.
    fn attachments<'py>(
        slf: PyRef<'py, Self>,
        sources: Vec<Bound<'_, PyIngestSource>>,
    ) -> PyResult<PyRef<'py, Self>> {
        let srcs = sources
            .iter()
            .map(|s| s.borrow().snapshot())
            .collect::<PyResult<Vec<_>>>()?;
        slf.map(|t| t.attachments(srcs));
        Ok(slf)
    }

    fn __repr__(&self) -> String {
        match self.snapshot() {
            Ok(_) => "Turn(...)".to_string(),
            Err(_) => "Turn(<invalid>)".to_string(),
        }
    }
}

/// A conversation scope that feeds turns into memory.
///
/// Obtained from [`Agent.session`](crate::agent::PyAgent). Turns fed through
/// one session preserve cross-turn conversational state, so feed them in order.
#[pyclass(name = "Session", module = "uniko")]
pub struct PySession {
    inner: Arc<Mutex<Session>>,
    session_id: String,
}

impl PySession {
    /// Wrap an owned facade [`Session`], caching its external id.
    pub(crate) fn wrap(session: Session) -> Self {
        let session_id = session.session_id().to_string();
        Self {
            inner: Arc::new(Mutex::new(session)),
            session_id,
        }
    }
}

pymethods_with_sync! {
    PySession;
    plain {
        /// The session's external identifier.
        #[getter]
        fn session_id(&self) -> &str {
            &self.session_id
        }
        // ── Context-manager support ────────────────────────────────────────
        //
        // `async with agent.session("s") as s:` finalizes on exit, so the
        // session-level retrieval surfaces get built without the caller having
        // to remember. Exit is best-effort: a chunking failure must not mask
        // whatever exception is already propagating out of the block.

        fn __aenter__<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            let this: Py<Self> = slf.into();
            pyo3_async_runtimes::tokio::future_into_py(py, async move { Ok(this) })
        }

        #[pyo3(signature = (exc_type=None, exc=None, tb=None))]
        fn __aexit__<'py>(
            &self,
            py: Python<'py>,
            exc_type: Option<Py<PyAny>>,
            exc: Option<Py<PyAny>>,
            tb: Option<Py<PyAny>>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let (_, _, _) = (exc_type, exc, tb);
            bridge!(py, session = self.inner.clone(), {
                let guard = session.lock().await;
                if let Err(e) = guard.finalize().await {
                    eprintln!("uniko: session __aexit__ finalize failed: {e}");
                }
                // Never suppress an in-flight exception.
                Ok(false)
            })
        }

        fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        #[pyo3(signature = (exc_type=None, exc=None, tb=None))]
        fn __exit__(
            &self,
            py: Python<'_>,
            exc_type: Option<Py<PyAny>>,
            exc: Option<Py<PyAny>>,
            tb: Option<Py<PyAny>>,
        ) -> PyResult<bool> {
            let (_, _, _) = (exc_type, exc, tb);
            bridge_sync!(py, session = self.inner.clone(), {
                let guard = session.lock().await;
                if let Err(e) = guard.finalize().await {
                    eprintln!("uniko: session __exit__ finalize failed: {e}");
                }
                Ok(false)
            })
        }

        fn __repr__(&self) -> String {
            format!("Session(session_id='{}')", self.session_id)
        }

    }
    bridged {

        /// Record several related turns as ONE durable, idempotent unit.
        ///
        /// Every turn, and every attachment on them, lands in a single
        /// transaction: either the whole unit is visible to later recall or none
        /// of it is. Two separate `observe` calls cannot give that — an
        /// interruption between them leaves a question with no answer.
        ///
        /// Resolves to a `UnitResult`: one `ObserveResult` per turn, and
        /// `was_replay`, true when every id was already present with identical
        /// content, so nothing was written.
        ///
        /// Reusing an id with different content, repeating an id inside one unit,
        /// or submitting a unit that is only partly recorded raises
        /// `IdConflictError`.
        fn commit_unit / commit_unit_sync(&self, py, turns: Vec<PyRef<'_, PyTurn>>) -> Py<PyUnitResult> {
            prepare {
                // Snapshot before entering the async block: PyO3 refs are not Send.
                let turns: Vec<_> = turns
                    .iter()
                    .map(|t| t.snapshot())
                    .collect::<PyResult<_>>()?;
            }
            run(session = self.inner.clone()) {
                let mut guard = session.lock().await;
                let result = guard.commit_unit(turns).await.map_err(to_pyerr)?;
                Python::attach(|py| PyUnitResult::from_rust(py, &result))
            }
        }

        /// Ingest one turn durably, committing before returning.
        ///
        /// The write is immediately visible to recall (read-after-write).
        fn observe / observe_sync(&self, py, turn: &PyTurn) -> Py<PyObserveResult> {
            prepare {
                let turn = turn.snapshot()?;

            }
            run(session = self.inner.clone()) {
                let mut guard = session.lock().await;
                let result = guard.observe(turn).await.map_err(to_pyerr)?;
                Python::attach(|py| PyObserveResult::from_rust(py, &result))

            }
        }

        /// Ingest a standalone document/blob into this session (not a turn).
        fn ingest / ingest_sync(&self, py, source: &PyIngestSource) -> Py<PyIngestOutcome> {
            prepare {
                let src = source.snapshot()?;

            }
            run(session = self.inner.clone()) {
                let guard = session.lock().await;
                let outcome = guard.ingest(src).await.map_err(to_pyerr)?;
                Python::attach(|py| PyIngestOutcome::from_rust(py, &outcome))

            }
        }

        /// Submit a turn to the streaming pipeline (fire-and-forget).
        ///
        /// Requires the instance to be built with `streaming(True)`.
        fn submit / submit_sync(&self, py, turn: &PyTurn) -> () {
            prepare {
                let turn = turn.snapshot()?;

            }
            run(session = self.inner.clone()) {
                let guard = session.lock().await;
                guard.submit(turn).await.map_err(to_pyerr)?;
                Ok(())

            }
        }

        /// Submit a standalone document/blob to the streaming pipeline.
        fn submit_source / submit_source_sync(&self, py, source: &PyIngestSource) -> () {
            prepare {
                let src = source.snapshot()?;

            }
            run(session = self.inner.clone()) {
                let guard = session.lock().await;
                guard.submit_source(src).await.map_err(to_pyerr)?;
                Ok(())

            }
        }

        /// Wait for all submitted work in this session to finish.
        fn flush / flush_sync(&self, py) -> () {
            prepare {}
            run(session = self.inner.clone()) {
                let guard = session.lock().await;
                guard.flush().await.map_err(to_pyerr)?;
                Ok(())

            }
        }

        /// Build (or refresh) this session's session-level retrieval surfaces.
        ///
        /// Chunks the whole transcript and aggregates the session's observations
        /// into dense chunks wired to the entities and participants they mention.
        /// These are what session-scoped recall and the Phase 1 session boost
        /// retrieve. Cheap and idempotent when the session has not grown.
        /// Waits for streamed turns first when streaming is enabled.
        fn finalize / finalize_sync(&self, py) -> Py<PyFinalizeReport> {
            prepare {}
            run(session = self.inner.clone()) {
                let guard = session.lock().await;
                let report = guard.finalize().await.map_err(to_pyerr)?;
                Python::attach(|py| PyFinalizeReport::from_rust(py, &report))

            }
        }

        /// Generate and persist a summary of the session so far.
        ///
        /// Resolves to a `SummarizeReport`: the new summary node id or `None`
        /// when there was nothing new to summarize, what finalizing built, and
        /// the chunk-refresh failure message or `None` when it succeeded.
        ///
        /// The refresh is best-effort and never fails the call, but its outcome
        /// is reported rather than only logged, so a caller can tell that the
        /// summary was built from stale chunks.
        fn summarize / summarize_sync(&self, py) -> Py<PySummarizeReport> {
            prepare {}
            run(session = self.inner.clone()) {
                let guard = session.lock().await;
                let report = guard.summarize().await.map_err(to_pyerr)?;
                Python::attach(|py| PySummarizeReport::from_rust(py, &report))

            }
        }

        /// Record an action this session's agent took, linked to this session
        /// unless `session_id` names another.
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
            run(session = self.inner.clone()) {
                let guard = session.lock().await;
                let result = guard.record_action(params).await.map_err(to_pyerr)?;
                Python::attach(|py| PyActionResult::from_rust(py, &result))

            }
        }

        /// Soft-forget a single turn (redact content, keep the chain intact).
        fn forget_turn / forget_turn_sync(&self, py, message_id: String) -> Py<PyDeletionReport> {
            prepare {}
            run(session = self.inner.clone()) {
                let guard = session.lock().await;
                let report = guard.forget_turn(&message_id).await.map_err(to_pyerr)?;
                Python::attach(|py| PyDeletionReport::from_rust(py, &report))

            }
        }

        /// Hard-delete a single turn and its cascade.
        fn delete_turn / delete_turn_sync(&self, py, message_id: String) -> Py<PyDeletionReport> {
            prepare {}
            run(session = self.inner.clone()) {
                let guard = session.lock().await;
                let report = guard.delete_turn(&message_id).await.map_err(to_pyerr)?;
                Python::attach(|py| PyDeletionReport::from_rust(py, &report))

            }
        }

        /// Hard-delete a document/artifact and its cascade.
        fn delete_document / delete_document_sync(&self, py, artifact_id: String) -> Py<PyDeletionReport> {
            prepare {}
            run(session = self.inner.clone()) {
                let guard = session.lock().await;
                let report = guard
                    .delete_document(&artifact_id)
                    .await
                    .map_err(to_pyerr)?;
                Python::attach(|py| PyDeletionReport::from_rust(py, &report))

            }
        }

    }
}
