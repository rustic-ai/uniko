// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Industries Inc.

//! The [`PyScope`] per-call recall scope: dimensional hard-filters.
//!
//! Passed to `Agent.recall_in` / `query_in` / `answer_in` to narrow a single
//! call by session / participant / time. Visibility scoping (`as_viewer`)
//! needs an async `Viewer` resolution against the store and is deferred; the
//! instance-level `scope_to_agent` covers the common visibility case.

use pyo3::prelude::*;
use uniko_api::tools::Scope;

use crate::convert::py_to_datetime;

/// A per-call recall scope: session / participant / time filters.
#[pyclass(name = "Scope", module = "uniko")]
pub struct PyScope {
    inner: std::sync::Mutex<Scope>,
}

impl PyScope {
    /// Apply a consuming `Scope` builder method in place.
    fn map(&self, f: impl FnOnce(Scope) -> Scope) {
        let mut guard = self.inner.lock().expect("Scope mutex poisoned");
        let scope = std::mem::take(&mut *guard);
        *guard = f(scope);
    }

    /// A cloned snapshot of the scope, for a recall/query call.
    pub(crate) fn snapshot(&self) -> Scope {
        self.inner.lock().expect("Scope mutex poisoned").clone()
    }
}

#[pymethods]
impl PyScope {
    /// An empty scope (no constraints).
    #[new]
    fn new() -> Self {
        Self {
            inner: std::sync::Mutex::new(Scope::default()),
        }
    }

    /// Restrict recall to these session ids.
    fn sessions<'py>(slf: PyRef<'py, Self>, sessions: Vec<String>) -> PyRef<'py, Self> {
        slf.map(|s| s.sessions(sessions));
        slf
    }

    /// Restrict recall to these record categories (issue #39).
    ///
    /// Pushed into candidate generation, so ranking, the result limit and
    /// the coverage score all describe the permitted evidence rather than
    /// items discarded after ranking. A category with no matches returns an
    /// empty result instead of being padded with other categories.
    fn categories<'py>(slf: PyRef<'py, Self>, categories: Vec<String>) -> PyRef<'py, Self> {
        slf.map(|s| s.categories(categories));
        slf
    }

    /// Restrict recall to these logical source ids (issue #39).
    fn sources<'py>(slf: PyRef<'py, Self>, sources: Vec<String>) -> PyRef<'py, Self> {
        slf.map(|s| s.sources(sources));
        slf
    }

    /// Include superseded revisions and retired sources — historical recall
    /// (issue #41).
    ///
    /// Ordinary recall excludes them, so a replaced page or a retired feed
    /// cannot ground a current answer. Ask for them when the question is
    /// what grounded a result at the time.
    fn include_superseded<'py>(slf: PyRef<'py, Self>) -> PyRef<'py, Self> {
        slf.map(Scope::include_superseded);
        slf
    }

    /// Restrict recall to these participant names (sender or subject).
    fn participants<'py>(slf: PyRef<'py, Self>, participants: Vec<String>) -> PyRef<'py, Self> {
        slf.map(|s| s.participants(participants));
        slf
    }

    /// Set the inclusive lower time bound (a Python `datetime`).
    fn since<'py>(slf: PyRef<'py, Self>, since: &Bound<'_, PyAny>) -> PyResult<PyRef<'py, Self>> {
        let dt = py_to_datetime(since)?;
        slf.map(|s| s.since(dt));
        Ok(slf)
    }

    /// Set the exclusive upper time bound (a Python `datetime`).
    fn until<'py>(slf: PyRef<'py, Self>, until: &Bound<'_, PyAny>) -> PyResult<PyRef<'py, Self>> {
        let dt = py_to_datetime(until)?;
        slf.map(|s| s.until(dt));
        Ok(slf)
    }

    fn __repr__(&self) -> String {
        "Scope(...)".to_string()
    }
}
