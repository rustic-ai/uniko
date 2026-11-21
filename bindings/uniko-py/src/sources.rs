// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Industries Inc.

//! The [`PySources`] handle — deterministic read tools: lexical search,
//! outlines, verbatim reads by address or range, and tool-call lookups.
//!
//! Like `PyData` it holds an `Agent` clone and rebuilds `agent.sources()`
//! inside each async block, so the borrow lives within one `await`.

use pyo3::prelude::*;
use pyo3::types::PyList;
use uniko_api::tools::{
    Agent, Outline, OutlineEntry, ReadRange, ReadResult, SearchHit, SearchOptions, SourceRef,
    ToolCallView,
};

use crate::convert::{json_to_py, utc_datetime_to_py};
use crate::errors::to_pyerr;
use crate::macros::pymethods_with_sync;

fn source_ref(kind: &str, id: String) -> PyResult<SourceRef> {
    match kind {
        "artifact" => Ok(SourceRef::Artifact(id)),
        "message" => Ok(SourceRef::Message(id)),
        "session" => Ok(SourceRef::Session(id)),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "source kind must be 'artifact', 'message' or 'session', not {other:?}"
        ))),
    }
}

fn source_parts(source: &SourceRef) -> (&'static str, String) {
    match source {
        SourceRef::Artifact(id) => ("artifact", id.clone()),
        SourceRef::Message(id) => ("message", id.clone()),
        SourceRef::Session(id) => ("session", id.clone()),
    }
}

/// One lexical search hit.
#[pyclass(name = "SearchHit", module = "uniko", frozen)]
pub struct PySearchHit {
    /// `"artifact"`, `"message"` or `"session"`.
    #[pyo3(get)]
    source_kind: String,
    /// The source's external id.
    #[pyo3(get)]
    source_id: String,
    /// The chunk's id.
    #[pyo3(get)]
    chunk_id: String,
    /// The chunk's position in its source.
    #[pyo3(get)]
    chunk_index: usize,
    /// Its address in the source, for a structure-aware chunk.
    #[pyo3(get)]
    address: Option<String>,
    /// For a chunk derived from media: where in it the chunk came from.
    #[pyo3(get)]
    position: Option<Py<crate::media::PyMediaPosition>>,
    /// Byte span start in the source.
    #[pyo3(get)]
    start: usize,
    /// Byte span end.
    #[pyo3(get)]
    end: usize,
    /// The chunk's verbatim text.
    #[pyo3(get)]
    text: String,
    /// What the chunk inherits from outside itself, for a tool output.
    #[pyo3(get)]
    context: Option<String>,
    /// Lexical score.
    #[pyo3(get)]
    score: f64,
    /// The session, when known.
    #[pyo3(get)]
    session_id: Option<String>,
    /// The tool call whose result this is.
    #[pyo3(get)]
    call_id: Option<String>,
    /// That call's tool.
    #[pyo3(get)]
    tool_name: Option<String>,
}

impl PySearchHit {
    fn from_rust(py: Python<'_>, h: &SearchHit) -> PyResult<Py<Self>> {
        let (kind, id) = source_parts(&h.source);
        Py::new(
            py,
            Self {
                source_kind: kind.into(),
                source_id: id,
                chunk_id: h.chunk_id.clone(),
                chunk_index: h.chunk_index,
                address: h.address.clone(),
                position: crate::media::PyMediaPosition::from_opt(py, h.position.as_ref())?,
                start: h.start,
                end: h.end,
                text: h.text.clone(),
                context: h.context.clone(),
                score: h.score,
                session_id: h.session_id.clone(),
                call_id: h.call_id.clone(),
                tool_name: h.tool_name.clone(),
            },
        )
    }
}

#[pymethods]
impl PySearchHit {
    fn __repr__(&self) -> String {
        format!(
            "SearchHit(source={}:{}, address={:?}, score={:.3})",
            self.source_kind, self.source_id, self.address, self.score
        )
    }
}

/// One entry of an [`PyOutline`].
#[pyclass(name = "OutlineEntry", module = "uniko", frozen)]
pub struct PyOutlineEntry {
    /// A JSON pointer, `rows a-b`, `lines a-b`, or `chunk n`.
    #[pyo3(get)]
    address: String,
    /// `"object"`, `"array"`, `"scalar"` or `"chunk"`.
    #[pyo3(get)]
    kind: String,
    /// A one-line description.
    #[pyo3(get)]
    summary: String,
    /// Byte span start.
    #[pyo3(get)]
    start: usize,
    /// Byte span end.
    #[pyo3(get)]
    end: usize,
    /// Nesting depth.
    #[pyo3(get)]
    depth: usize,
}

impl PyOutlineEntry {
    fn from_rust(py: Python<'_>, e: &OutlineEntry) -> PyResult<Py<Self>> {
        Py::new(
            py,
            Self {
                address: e.address.clone(),
                kind: e.kind.clone(),
                summary: e.summary.clone(),
                start: e.start,
                end: e.end,
                depth: e.depth,
            },
        )
    }
}

#[pymethods]
impl PyOutlineEntry {
    fn __repr__(&self) -> String {
        format!(
            "OutlineEntry({} {}: {})",
            self.kind, self.address, self.summary
        )
    }
}

/// The shape of a source.
#[pyclass(name = "Outline", module = "uniko", frozen)]
pub struct PyOutline {
    /// `"artifact"`, `"message"` or `"session"`.
    #[pyo3(get)]
    source_kind: String,
    /// The source's external id.
    #[pyo3(get)]
    source_id: String,
    /// Detected format.
    #[pyo3(get)]
    format: String,
    /// Size of the source text in bytes.
    #[pyo3(get)]
    total_bytes: usize,
    /// Number of lines.
    #[pyo3(get)]
    total_lines: usize,
    /// Entries in source order.
    #[pyo3(get)]
    entries: Vec<Py<PyOutlineEntry>>,
    /// More entries existed than were returned.
    #[pyo3(get)]
    truncated: bool,
}

impl PyOutline {
    fn from_rust(py: Python<'_>, o: &Outline) -> PyResult<Py<Self>> {
        let (kind, id) = source_parts(&o.source);
        let entries = o
            .entries
            .iter()
            .map(|e| PyOutlineEntry::from_rust(py, e))
            .collect::<PyResult<Vec<_>>>()?;
        Py::new(
            py,
            Self {
                source_kind: kind.into(),
                source_id: id,
                format: o.format.clone(),
                total_bytes: o.total_bytes,
                total_lines: o.total_lines,
                entries,
                truncated: o.truncated,
            },
        )
    }
}

#[pymethods]
impl PyOutline {
    fn __repr__(&self) -> String {
        format!(
            "Outline({}:{}, format='{}', entries={})",
            self.source_kind,
            self.source_id,
            self.format,
            self.entries.len()
        )
    }
}

/// Verbatim text read from a source.
#[pyclass(name = "ReadResult", module = "uniko", frozen)]
pub struct PyReadResult {
    /// `"artifact"`, `"message"` or `"session"`.
    #[pyo3(get)]
    source_kind: String,
    /// The source's external id.
    #[pyo3(get)]
    source_id: String,
    /// Verbatim text.
    #[pyo3(get)]
    text: String,
    /// Byte span start.
    #[pyo3(get)]
    start: usize,
    /// Byte span end.
    #[pyo3(get)]
    end: usize,
    /// First line, 1-based (0 for a chunk read without the source text).
    #[pyo3(get)]
    first_line: usize,
    /// Last line.
    #[pyo3(get)]
    last_line: usize,
    /// The range was cut at the token limit.
    #[pyo3(get)]
    truncated: bool,
    /// `(start, end)` bytes still to read when the range was cut.
    #[pyo3(get)]
    next_bytes: Option<(usize, usize)>,
}

impl PyReadResult {
    fn from_rust(py: Python<'_>, r: &ReadResult) -> PyResult<Py<Self>> {
        let (kind, id) = source_parts(&r.source);
        let next_bytes = match r.next {
            Some(ReadRange::Bytes(s, e)) => Some((s, e)),
            _ => None,
        };
        Py::new(
            py,
            Self {
                source_kind: kind.into(),
                source_id: id,
                text: r.text.clone(),
                start: r.start,
                end: r.end,
                first_line: r.first_line,
                last_line: r.last_line,
                truncated: r.truncated,
                next_bytes,
            },
        )
    }
}

#[pymethods]
impl PyReadResult {
    fn __repr__(&self) -> String {
        format!(
            "ReadResult({}:{}, bytes {}-{}, truncated={})",
            self.source_kind,
            self.source_id,
            self.start,
            self.end,
            if self.truncated { "True" } else { "False" }
        )
    }
}

/// A recorded tool call.
#[pyclass(name = "ToolCallView", module = "uniko", frozen)]
pub struct PyToolCallView {
    /// The call's id.
    #[pyo3(get)]
    call_id: String,
    /// The tool, for a call recorded as a tool call.
    #[pyo3(get)]
    tool_name: Option<String>,
    /// `"tool_call"`, or the kind given to `record_action`.
    #[pyo3(get)]
    action_type: String,
    /// The arguments it was called with.
    #[pyo3(get)]
    arguments: Py<PyAny>,
    /// Its outcome.
    #[pyo3(get)]
    status: Option<String>,
    /// Error text for a failed call.
    #[pyo3(get)]
    error: Option<String>,
    /// When it started (tz-aware UTC `datetime`), when known.
    #[pyo3(get)]
    started_at: Py<PyAny>,
    /// When it ended.
    #[pyo3(get)]
    ended_at: Py<PyAny>,
    /// The message that made the call.
    #[pyo3(get)]
    triggered_by: Option<String>,
    /// Its session.
    #[pyo3(get)]
    session_id: Option<String>,
    /// Its result artifact's id.
    #[pyo3(get)]
    artifact_id: Option<String>,
}

impl PyToolCallView {
    fn from_rust(py: Python<'_>, v: &ToolCallView) -> PyResult<Py<Self>> {
        let dt = |d: Option<chrono::DateTime<chrono::Utc>>| -> PyResult<Py<PyAny>> {
            match d {
                Some(d) => Ok(utc_datetime_to_py(py, d)?.unbind()),
                None => Ok(py.None()),
            }
        };
        Py::new(
            py,
            Self {
                call_id: v.call_id.clone(),
                tool_name: v.tool_name.clone(),
                action_type: v.action_type.clone(),
                arguments: match &v.arguments {
                    Some(a) => json_to_py(py, a)?,
                    None => py.None(),
                },
                status: v.status.clone(),
                error: v.error.clone(),
                started_at: dt(v.started_at)?,
                ended_at: dt(v.ended_at)?,
                triggered_by: v.triggered_by.clone(),
                session_id: v.session_id.clone(),
                artifact_id: v.artifact_id.clone(),
            },
        )
    }
}

#[pymethods]
impl PyToolCallView {
    fn __repr__(&self) -> String {
        format!(
            "ToolCallView(call_id='{}', tool_name={:?}, artifact_id={:?})",
            self.call_id, self.tool_name, self.artifact_id
        )
    }
}

/// Deterministic read tools over an agent's store.
///
/// Obtained from `Agent.sources`. Every call is a pure function of the
/// store and its arguments and applies the same visibility as recall.
#[pyclass(name = "Sources", module = "uniko")]
pub struct PySources {
    inner: Agent,
}

impl PySources {
    /// Wrap an Agent clone.
    pub(crate) fn wrap(agent: Agent) -> Self {
        Self { inner: agent }
    }
}

fn read_range(
    address: Option<String>,
    bytes: Option<(usize, usize)>,
    lines: Option<(usize, usize)>,
    chunks: Option<(usize, usize)>,
) -> PyResult<ReadRange> {
    let given = [
        address.as_ref().map(|a| ReadRange::Address(a.clone())),
        bytes.map(|(s, e)| ReadRange::Bytes(s, e)),
        lines.map(|(s, e)| ReadRange::Lines(s, e)),
        chunks.map(|(s, e)| ReadRange::Chunks(s, e)),
    ];
    let mut picked = given.into_iter().flatten();
    match (picked.next(), picked.next()) {
        (Some(r), None) => Ok(r),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "give exactly one of address=, bytes=, lines= or chunks=",
        )),
    }
}

pymethods_with_sync! {
    PySources;
    plain {
        fn __repr__(&self) -> String {
            format!("Sources(agent_id='{}')", self.inner.agent_id())
        }
    }
    bridged {
        /// Lexical search; identifiers and figures in `query` match whole.
        #[pyo3(signature = (query, *, limit=10, session_id=None, tool_name=None))]
        fn search / search_sync(&self, py, query: String, limit: usize, session_id: Option<String>, tool_name: Option<String>) -> Py<PyAny> {
            prepare {
                let mut options = SearchOptions::default().limit(limit);
                if let Some(s) = session_id {
                    options = options.session(s);
                }
                if let Some(t) = tool_name {
                    options = options.tool(t);
                }

            }
            run(agent = self.inner.clone()) {
                let hits = agent.sources().search(&query, &options).await.map_err(to_pyerr)?;
                Python::attach(|py| {
                    let items = hits
                        .iter()
                        .map(|h| PySearchHit::from_rust(py, h))
                        .collect::<PyResult<Vec<_>>>()?;
                    Ok(PyList::new(py, items)?.into_any().unbind())
                })

            }
        }

        /// The shape of a source, or `None` when unknown or not visible.
        #[pyo3(signature = (source_kind, source_id, *, depth=3, max_entries=200))]
        fn outline / outline_sync(&self, py, source_kind: String, source_id: String, depth: usize, max_entries: usize) -> Py<PyAny> {
            prepare {
                let source = source_ref(&source_kind, source_id)?;

            }
            run(agent = self.inner.clone()) {
                let outline = agent
                    .sources()
                    .outline(&source, depth, max_entries)
                    .await
                    .map_err(to_pyerr)?;
                Python::attach(|py| match outline {
                    Some(o) => PyOutline::from_rust(py, &o).map(|p| p.into_any()),
                    None => Ok(py.None()),
                })

            }
        }

        /// Read a range of a source verbatim; give exactly one of
        /// `address=`, `bytes=`, `lines=` or `chunks=`.
        #[pyo3(signature = (source_kind, source_id, *, address=None, bytes=None, lines=None, chunks=None, max_tokens=2000))]
        #[allow(clippy::too_many_arguments)]
        fn read / read_sync(&self, py, source_kind: String, source_id: String, address: Option<String>, bytes: Option<(usize, usize)>, lines: Option<(usize, usize)>, chunks: Option<(usize, usize)>, max_tokens: usize) -> Py<PyAny> {
            prepare {
                let source = source_ref(&source_kind, source_id)?;
                let range = read_range(address, bytes, lines, chunks)?;

            }
            run(agent = self.inner.clone()) {
                let read = agent
                    .sources()
                    .read(&source, &range, max_tokens)
                    .await
                    .map_err(to_pyerr)?;
                Python::attach(|py| match read {
                    Some(r) => PyReadResult::from_rust(py, &r).map(|p| p.into_any()),
                    None => Ok(py.None()),
                })

            }
        }

        /// A recorded tool call by id, or `None`.
        fn tool_call / tool_call_sync(&self, py, call_id: String) -> Py<PyAny> {
            prepare {}
            run(agent = self.inner.clone()) {
                let view = agent.sources().tool_call(&call_id).await.map_err(to_pyerr)?;
                Python::attach(|py| match view {
                    Some(v) => PyToolCallView::from_rust(py, &v).map(|p| p.into_any()),
                    None => Ok(py.None()),
                })

            }
        }

        /// Recorded tool calls, newest first.
        #[pyo3(signature = (*, session_id=None, tool_name=None, limit=50))]
        fn tool_calls / tool_calls_sync(&self, py, session_id: Option<String>, tool_name: Option<String>, limit: usize) -> Py<PyAny> {
            prepare {}
            run(agent = self.inner.clone()) {
                let views = agent
                    .sources()
                    .tool_calls(session_id.as_deref(), tool_name.as_deref(), limit)
                    .await
                    .map_err(to_pyerr)?;
                Python::attach(|py| {
                    let items = views
                        .iter()
                        .map(|v| PyToolCallView::from_rust(py, v))
                        .collect::<PyResult<Vec<_>>>()?;
                    Ok(PyList::new(py, items)?.into_any().unbind())
                })

            }
        }
    }
}

/// Register the read-tool classes.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySources>()?;
    m.add_class::<PySearchHit>()?;
    m.add_class::<PyOutline>()?;
    m.add_class::<PyOutlineEntry>()?;
    m.add_class::<PyReadResult>()?;
    m.add_class::<PyToolCallView>()?;
    Ok(())
}
