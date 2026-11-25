//! `DeadLetters`: inspect, retry and resolve streaming-ingest failures.

use pyo3::prelude::*;
use pyo3::types::PyList;
use uniko_api::tools::{
    DeadLetter, DeadLetterCounts, DeadLetterFilter, DeadLetterStatus, DeadLetters, RetryReport,
};

use crate::convert::utc_datetime_to_py;
use crate::errors::to_pyerr;
use crate::macros::pymethods_with_sync;

fn status_from_str(status: &str) -> PyResult<DeadLetterStatus> {
    status.parse().map_err(|e: uniko_api::tools::UnikoError| {
        pyo3::exceptions::PyValueError::new_err(e.to_string())
    })
}

fn opt_time(
    py: Python<'_>,
    dt: Option<chrono::DateTime<chrono::Utc>>,
) -> PyResult<Option<Py<PyAny>>> {
    dt.map(|d| utc_datetime_to_py(py, d).map(Bound::unbind))
        .transpose()
}

/// One dead-lettered streaming item.
#[pyclass(name = "DeadLetter", module = "uniko", frozen)]
pub struct PyDeadLetter {
    /// Stable id; ``"legacy:<node id>"`` for a row from before ids existed.
    #[pyo3(get)]
    id: String,
    /// The step that failed.
    #[pyo3(get)]
    step: String,
    /// The latest failure's message.
    #[pyo3(get)]
    error: String,
    /// The latest failure's kind, e.g. ``"embedding"`` or ``"id_conflict"``.
    #[pyo3(get)]
    error_kind: Option<String>,
    /// Whether the latest failure can succeed on a retry.
    #[pyo3(get)]
    retriable: bool,
    /// ``"pending"``, ``"retrying"``, ``"exhausted"``, ``"resolved"`` or
    /// ``"legacy"``.
    #[pyo3(get)]
    status: &'static str,
    /// Retries attempted so far.
    #[pyo3(get)]
    retry_count: u32,
    /// Retries allowed before the letter is exhausted.
    #[pyo3(get)]
    max_retries: u32,
    /// ``"message"``, ``"artifact"``, ``"pdf"`` or ``"source"``.
    #[pyo3(get)]
    ingest_type: Option<String>,
    /// The agent the item was submitted under, when known.
    #[pyo3(get)]
    agent_id: Option<String>,
    /// The stored payload can be replayed.
    #[pyo3(get)]
    replayable: bool,
    /// When the item was first dead-lettered.
    #[pyo3(get)]
    created_at: Option<Py<PyAny>>,
    /// When the letter last changed.
    #[pyo3(get)]
    updated_at: Option<Py<PyAny>>,
    /// When the next retry is due, for a pending letter.
    #[pyo3(get)]
    next_retry_at: Option<Py<PyAny>>,
    /// When the letter was resolved.
    #[pyo3(get)]
    resolved_at: Option<Py<PyAny>>,
    /// Why it was resolved: ``"retried"``, or the note given to ``resolve``.
    #[pyo3(get)]
    resolution: Option<String>,
}

impl PyDeadLetter {
    fn from_rust(py: Python<'_>, d: &DeadLetter) -> PyResult<Py<Self>> {
        Py::new(
            py,
            Self {
                id: d.id.clone(),
                step: d.step.clone(),
                error: d.error.clone(),
                error_kind: d.error_kind.clone(),
                retriable: d.retriable,
                status: d.status.as_str(),
                retry_count: d.retry_count,
                max_retries: d.max_retries,
                ingest_type: d.ingest_type.clone(),
                agent_id: d.agent_id.clone(),
                replayable: d.replayable,
                created_at: opt_time(py, d.created_at)?,
                updated_at: opt_time(py, d.updated_at)?,
                next_retry_at: opt_time(py, d.next_retry_at)?,
                resolved_at: opt_time(py, d.resolved_at)?,
                resolution: d.resolution.clone(),
            },
        )
    }
}

#[pymethods]
impl PyDeadLetter {
    fn __repr__(&self) -> String {
        format!(
            "DeadLetter(id='{}', status='{}', retries={}/{}, error_kind={:?})",
            self.id, self.status, self.retry_count, self.max_retries, self.error_kind
        )
    }
}

/// How many dead letters are in each status, from ``DeadLetters.counts``.
#[pyclass(name = "DeadLetterCounts", module = "uniko", frozen)]
pub struct PyDeadLetterCounts {
    /// Waiting for a retry.
    #[pyo3(get)]
    pending: u64,
    /// Being retried.
    #[pyo3(get)]
    retrying: u64,
    /// Out of retries, or failed in a way a retry can't fix.
    #[pyo3(get)]
    exhausted: u64,
    /// Retried successfully or resolved.
    #[pyo3(get)]
    resolved: u64,
    /// Written before payloads were kept.
    #[pyo3(get)]
    legacy: u64,
}

impl PyDeadLetterCounts {
    fn from_rust(py: Python<'_>, c: &DeadLetterCounts) -> PyResult<Py<Self>> {
        Py::new(
            py,
            Self {
                pending: c.pending,
                retrying: c.retrying,
                exhausted: c.exhausted,
                resolved: c.resolved,
                legacy: c.legacy,
            },
        )
    }
}

#[pymethods]
impl PyDeadLetterCounts {
    fn __repr__(&self) -> String {
        format!(
            "DeadLetterCounts(pending={}, retrying={}, exhausted={}, resolved={}, legacy={})",
            self.pending, self.retrying, self.exhausted, self.resolved, self.legacy
        )
    }
}

/// A dead letter ``retry_pending`` could not resubmit.
#[pyclass(name = "RetryRefusal", module = "uniko", frozen)]
pub struct PyRetryRefusal {
    /// The letter's id.
    #[pyo3(get)]
    id: String,
    /// Why it was not resubmitted.
    #[pyo3(get)]
    reason: String,
}

#[pymethods]
impl PyRetryRefusal {
    fn __repr__(&self) -> String {
        format!("RetryRefusal(id='{}', reason='{}')", self.id, self.reason)
    }
}

/// What ``DeadLetters.retry_pending`` did.
#[pyclass(name = "RetryReport", module = "uniko", frozen)]
pub struct PyRetryReport {
    /// Letters resubmitted for a retry.
    #[pyo3(get)]
    retried: Vec<String>,
    /// Letters that could not be resubmitted.
    #[pyo3(get)]
    refused: Vec<Py<PyRetryRefusal>>,
}

impl PyRetryReport {
    fn from_rust(py: Python<'_>, r: &RetryReport) -> PyResult<Py<Self>> {
        let refused = r
            .refused
            .iter()
            .map(|x| {
                Py::new(
                    py,
                    PyRetryRefusal {
                        id: x.id.clone(),
                        reason: x.reason.clone(),
                    },
                )
            })
            .collect::<PyResult<Vec<_>>>()?;
        Py::new(
            py,
            Self {
                retried: r.retried.clone(),
                refused,
            },
        )
    }
}

#[pymethods]
impl PyRetryReport {
    fn __repr__(&self) -> String {
        format!(
            "RetryReport(retried={}, refused={})",
            self.retried.len(),
            self.refused.len()
        )
    }
}

/// The streaming pipeline's dead letters, from ``Uniko.dead_letters``.
///
/// Listing, counting and resolving work on any instance. Retrying needs one
/// built with ``streaming(True)``.
#[pyclass(name = "DeadLetters", module = "uniko")]
pub struct PyDeadLetters {
    inner: DeadLetters,
}

impl PyDeadLetters {
    pub(crate) fn wrap(inner: DeadLetters) -> Self {
        Self { inner }
    }
}

pymethods_with_sync! {
    PyDeadLetters;
    plain {
        fn __repr__(&self) -> String {
            "DeadLetters()".to_string()
        }
    }
    bridged {
        /// Dead letters, oldest first, optionally only those in ``status``
        /// and at most ``limit``.
        #[pyo3(signature = (*, status=None, limit=None))]
        fn list / list_sync(&self, py, status: Option<String>, limit: Option<usize>) -> Py<PyAny> {
            prepare {
                let mut filter = DeadLetterFilter::new();
                if let Some(s) = status {
                    filter = filter.status(status_from_str(&s)?);
                }
                if let Some(n) = limit {
                    filter = filter.limit(n);
                }

            }
            run(dl = self.inner.clone()) {
                let letters = dl.list(&filter).await.map_err(to_pyerr)?;
                Python::attach(|py| {
                    let out = PyList::empty(py);
                    for l in &letters {
                        out.append(PyDeadLetter::from_rust(py, l)?)?;
                    }
                    Ok(out.into_any().unbind())
                })

            }
        }

        /// One dead letter by id, or ``None``.
        fn get / get_sync(&self, py, id: String) -> Option<Py<PyDeadLetter>> {
            prepare {}
            run(dl = self.inner.clone()) {
                let letter = dl.get(&id).await.map_err(to_pyerr)?;
                Python::attach(|py| letter.map(|l| PyDeadLetter::from_rust(py, &l)).transpose())

            }
        }

        /// How many letters are in each status.
        fn counts / counts_sync(&self, py) -> Py<PyDeadLetterCounts> {
            prepare {}
            run(dl = self.inner.clone()) {
                let counts = dl.counts().await.map_err(to_pyerr)?;
                Python::attach(|py| PyDeadLetterCounts::from_rust(py, &counts))

            }
        }

        /// Retry a dead letter: count the attempt and resubmit its stored
        /// payload unchanged. Its outcome lands on the letter once processed;
        /// ``Session.flush`` waits for it. ``force`` also retries a letter
        /// whose failure a retry can't normally fix, or one out of retries.
        ///
        /// Raises ``ConfigError`` without streaming or when the letter can't
        /// be retried, and ``PipelineError`` when the ingest queue is full.
        #[pyo3(signature = (id, *, force=false))]
        fn retry / retry_sync(&self, py, id: String, force: bool) -> () {
            prepare {}
            run(dl = self.inner.clone()) {
                dl.retry(&id, force).await.map_err(to_pyerr)

            }
        }

        /// Retry up to ``limit`` pending letters whose failure a retry can
        /// fix, without waiting for their backoff.
        #[pyo3(signature = (limit=100))]
        fn retry_pending / retry_pending_sync(&self, py, limit: usize) -> Py<PyRetryReport> {
            prepare {}
            run(dl = self.inner.clone()) {
                let report = dl.retry_pending(limit).await.map_err(to_pyerr)?;
                Python::attach(|py| PyRetryReport::from_rust(py, &report))

            }
        }

        /// Resolve a letter without retrying it, recording ``note``. Only the
        /// letter changes. Returns ``False`` when there is no such letter or
        /// it was already resolved.
        #[pyo3(signature = (id, note=None))]
        fn resolve / resolve_sync(&self, py, id: String, note: Option<String>) -> bool {
            prepare {}
            run(dl = self.inner.clone()) {
                dl.resolve(&id, note.as_deref()).await.map_err(to_pyerr)

            }
        }
    }
}
