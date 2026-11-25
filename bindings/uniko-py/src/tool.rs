// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Industries Inc.

//! The [`PyToolCall`] input builder: one call of one tool and its result.
//!
//! Recorded with `Turn.tool_call(...)` it commits with the turn that made
//! it; recorded with `Session.observe_tool(...)` it commits on its own.

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use uniko_api::tools::{ToolCall, ToolOutput};

use crate::convert;

/// One call of one tool, to record with `Turn.tool_call` or
/// `Session.observe_tool`.
///
/// Construct with `ToolCall(tool_name)` and refine with the chainable
/// setters. Reusable: recording it does not consume the Python object.
#[pyclass(name = "ToolCall", module = "uniko")]
pub struct PyToolCall {
    // Same take-apply-replace shape as `PyTurn`; no `.await` is held
    // across this lock.
    inner: std::sync::Mutex<Option<ToolCall>>,
}

impl PyToolCall {
    fn map(&self, f: impl FnOnce(ToolCall) -> ToolCall) {
        let mut guard = self.inner.lock().expect("ToolCall mutex poisoned");
        if let Some(call) = guard.take() {
            *guard = Some(f(call));
        }
    }

    /// A cloned snapshot of the current call, for recording.
    pub(crate) fn snapshot(&self) -> PyResult<ToolCall> {
        self.inner
            .lock()
            .expect("ToolCall mutex poisoned")
            .clone()
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("ToolCall is in an invalid state")
            })
    }
}

#[pymethods]
impl PyToolCall {
    /// A call of `tool_name`.
    #[new]
    fn new(tool_name: String) -> Self {
        Self {
            inner: std::sync::Mutex::new(Some(ToolCall::new(tool_name))),
        }
    }

    /// The tool's name.
    #[getter]
    fn tool_name(&self) -> PyResult<String> {
        Ok(self.snapshot()?.tool_name().to_string())
    }

    /// The explicit performer set with ``performer()``, or ``None`` for the
    /// default (the turn's sender, or the session's agent).
    #[getter]
    fn performer_id(&self) -> PyResult<Option<String>> {
        Ok(self.snapshot()?.performer_id().map(str::to_string))
    }

    /// The host's own id for this call. Recording the same id again with
    /// the same result is a no-op; with a different result it raises
    /// `IdConflictError`.
    fn call_id<'py>(slf: PyRef<'py, Self>, call_id: String) -> PyRef<'py, Self> {
        slf.map(|c| c.call_id(call_id));
        slf
    }

    /// The participant that performed the call, when it isn't the default:
    /// the turn's sender inside a ``Turn``, or the session's agent for
    /// ``Session.observe_tool``. The turn keeps its own sender. A performer
    /// that doesn't exist yet is created as kind ``unknown``; naming one never
    /// changes its kind (``register`` makes an agent). The same call id with a
    /// different performer, named or by default, raises ``IdConflictError``.
    fn performer<'py>(slf: PyRef<'py, Self>, participant_id: String) -> PyRef<'py, Self> {
        slf.map(|c| c.performer(participant_id));
        slf
    }

    /// The arguments the tool was called with (any JSON-shaped value).
    fn arguments<'py>(
        slf: PyRef<'py, Self>,
        arguments: &Bound<'_, PyAny>,
    ) -> PyResult<PyRef<'py, Self>> {
        let json = convert::py_to_json(arguments)?;
        slf.map(|c| c.arguments(json));
        Ok(slf)
    }

    /// A JSON-shaped result (dict, list, scalar).
    fn json_output<'py>(
        slf: PyRef<'py, Self>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<PyRef<'py, Self>> {
        let json = convert::py_to_json(value)?;
        slf.map(|c| c.json_output(json));
        Ok(slf)
    }

    /// A text result: a log, test output, a CSV, a diff.
    fn text_output<'py>(slf: PyRef<'py, Self>, text: String) -> PyRef<'py, Self> {
        slf.map(|c| c.text_output(text));
        slf
    }

    /// A binary result with its MIME type, e.g. `"image/png"`.
    fn bytes_output<'py>(
        slf: PyRef<'py, Self>,
        data: &Bound<'_, PyBytes>,
        mime: String,
    ) -> PyRef<'py, Self> {
        let bytes = data.as_bytes().to_vec();
        slf.map(|c| c.output(ToolOutput::Bytes { bytes, mime }));
        slf
    }

    /// Outcome: `"ok"`, `"error"`, `"timeout"`, …
    fn status<'py>(slf: PyRef<'py, Self>, status: String) -> PyRef<'py, Self> {
        slf.map(|c| c.status(status));
        slf
    }

    /// Error text for a failed call.
    fn error<'py>(slf: PyRef<'py, Self>, error: String) -> PyRef<'py, Self> {
        slf.map(|c| c.error(error));
        slf
    }

    /// When the call started (a Python `datetime`; defaults to now).
    fn started_at<'py>(slf: PyRef<'py, Self>, at: &Bound<'_, PyAny>) -> PyResult<PyRef<'py, Self>> {
        let ts = convert::py_to_datetime(at)?;
        slf.map(|c| c.started_at(ts));
        Ok(slf)
    }

    /// When the call ended (defaults to when it started).
    fn ended_at<'py>(slf: PyRef<'py, Self>, at: &Bound<'_, PyAny>) -> PyResult<PyRef<'py, Self>> {
        let ts = convert::py_to_datetime(at)?;
        slf.map(|c| c.ended_at(ts));
        Ok(slf)
    }

    /// The message that made this call, by `message_id`; only needed with
    /// `Session.observe_tool`.
    fn triggered_by<'py>(slf: PyRef<'py, Self>, message_id: String) -> PyRef<'py, Self> {
        slf.map(|c| c.triggered_by(message_id));
        slf
    }

    /// The call this one follows, by call id. Without it a call follows the
    /// session's latest earlier call.
    fn after<'py>(slf: PyRef<'py, Self>, call_id: String) -> PyRef<'py, Self> {
        slf.map(|c| c.after(call_id));
        slf
    }

    /// Who may see the result: `"public"`, `"private:<id>"`, `"team:<id>"`
    /// or `"org:<id>"`. A call added to a turn takes the turn's visibility
    /// unless it sets its own.
    fn visibility<'py>(slf: PyRef<'py, Self>, visibility: &str) -> PyResult<PyRef<'py, Self>> {
        let visibility =
            uniko_api::tools::Visibility::parse(visibility).map_err(crate::errors::to_pyerr)?;
        slf.map(|c| c.visibility(visibility));
        Ok(slf)
    }

    /// The caller's record category for the result (issue #39).
    fn category<'py>(slf: PyRef<'py, Self>, category: String) -> PyRef<'py, Self> {
        slf.map(|c| c.category(category));
        slf
    }

    /// The logical source the result came from (issue #39).
    fn source<'py>(slf: PyRef<'py, Self>, source_id: String) -> PyRef<'py, Self> {
        slf.map(|c| c.source(source_id));
        slf
    }

    /// The revision of that source (issue #41).
    fn revision<'py>(slf: PyRef<'py, Self>, revision_id: String) -> PyRef<'py, Self> {
        slf.map(|c| c.revision(revision_id));
        slf
    }

    fn __repr__(&self) -> String {
        match self.snapshot() {
            Ok(c) => format!("ToolCall(tool_name='{}')", c.tool_name()),
            Err(_) => "ToolCall(<invalid>)".to_string(),
        }
    }
}
