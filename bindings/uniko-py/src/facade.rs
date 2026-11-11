// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Industries Inc.

//! The owning [`PyUniko`] handle, its [`PyUnikoBuilder`], and [`PyLlmSpec`].
//!
//! [`Uniko`] is `Clone` (Arc-backed), so async read methods clone it out from
//! under a short lock and `await` on the clone. `shutdown` consumes the
//! instance — Python has no move semantics, so `PyUniko` holds
//! `Mutex<Option<Uniko>>` and `take`s on shutdown, poisoning the handle
//! (subsequent use raises). Outstanding `Agent`/`Session` handles must be
//! dropped before `shutdown`, or the store's reference-count guard fails.

use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use pyo3::types::PyDict;
use uniko_api::tools::{LlmSpec, Uniko, UnikoBuilder, UnikoError};

use crate::agent::PyAgent;
use crate::errors::to_pyerr;
use crate::macros::{bridge, bridge_sync};
use crate::outputs::PyDeletionReport;

/// The `PyErr` raised when a handle is used after `shutdown`.
fn closed() -> PyErr {
    to_pyerr(UnikoError::Config(
        "this Uniko instance has been shut down".to_string(),
    ))
}

/// A cognitive-memory instance: the single end-user entry point.
#[pyclass(name = "Uniko", module = "uniko")]
pub struct PyUniko {
    inner: Arc<Mutex<Option<Uniko>>>,
}

impl PyUniko {
    /// Wrap an owned facade [`Uniko`].
    fn wrap(uniko: Uniko) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Some(uniko))),
        }
    }

    /// Clone the live instance out, or raise if it has been shut down.
    fn cloned(&self) -> PyResult<Uniko> {
        self.inner
            .lock()
            .expect("Uniko mutex poisoned")
            .as_ref()
            .cloned()
            .ok_or_else(closed)
    }
}

#[pymethods]
impl PyUniko {
    /// Open or create a persistent instance at `path` with best defaults.
    #[staticmethod]
    fn open(py: Python<'_>, path: String) -> PyResult<Bound<'_, PyAny>> {
        bridge!(py, _g = (), {
            let uniko = Uniko::open(path).await.map_err(to_pyerr)?;
            Ok(PyUniko::wrap(uniko))
        })
    }

    /// Blocking variant of [`open`](Self::open).
    #[staticmethod]
    fn open_sync(py: Python<'_>, path: String) -> PyResult<PyUniko> {
        bridge_sync!(py, _g = (), {
            let uniko = Uniko::open(path).await.map_err(to_pyerr)?;
            Ok(PyUniko::wrap(uniko))
        })
    }

    /// Open an ephemeral in-memory instance with best defaults.
    #[staticmethod]
    fn in_memory(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        bridge!(py, _g = (), {
            let uniko = Uniko::in_memory().await.map_err(to_pyerr)?;
            Ok(PyUniko::wrap(uniko))
        })
    }

    /// Blocking variant of [`in_memory`](Self::in_memory).
    #[staticmethod]
    fn in_memory_sync(py: Python<'_>) -> PyResult<PyUniko> {
        bridge_sync!(py, _g = (), {
            let uniko = Uniko::in_memory().await.map_err(to_pyerr)?;
            Ok(PyUniko::wrap(uniko))
        })
    }

    /// Start a builder for non-default construction.
    #[staticmethod]
    fn builder() -> PyUnikoBuilder {
        PyUnikoBuilder::new()
    }

    /// A view scoped to `agent_id` exposing the agent-facing tools.
    fn agent(&self, agent_id: String) -> PyResult<PyAgent> {
        let guard = self.inner.lock().expect("Uniko mutex poisoned");
        let uniko = guard.as_ref().ok_or_else(closed)?;
        Ok(PyAgent::wrap(uniko.agent(agent_id)))
    }

    /// A snapshot of the most relevant configuration values, as a dict.
    fn config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let guard = self.inner.lock().expect("Uniko mutex poisoned");
        let uniko = guard.as_ref().ok_or_else(closed)?;
        let config = uniko.config();
        let dict = PyDict::new(py);
        dict.set_item("embedding_model_id", config.embedding.model_id.clone())?;
        dict.set_item("embedding_dimensions", config.embedding.dimensions)?;
        dict.set_item("recall_limit", config.recall_limit)?;
        dict.set_item("recall_token_budget", config.recall_token_budget)?;
        dict.set_item("recall_min_score", config.recall_min_score)?;
        Ok(dict)
    }

    /// Erase the entire graph. Intended for dev/test reset.
    fn purge<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let uniko = self.cloned()?;
        bridge!(py, instance = uniko, {
            let report = instance.purge().await.map_err(to_pyerr)?;
            Python::attach(|py| PyDeletionReport::from_rust(py, &report))
        })
    }

    /// Blocking variant of [`purge`](Self::purge).
    fn purge_sync(&self, py: Python<'_>) -> PyResult<Py<PyDeletionReport>> {
        let uniko = self.cloned()?;
        bridge_sync!(py, instance = uniko, {
            let report = instance.purge().await.map_err(to_pyerr)?;
            Python::attach(|py| PyDeletionReport::from_rust(py, &report))
        })
    }

    /// Shut down the instance and drain the pipeline.
    ///
    /// Consumes the handle: all `Agent` / `Session` objects derived from it
    /// must be dropped first, or the store's reference-count guard fails.
    /// Using the handle afterwards raises.
    fn shutdown<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let taken = self.inner.lock().expect("Uniko mutex poisoned").take();
        bridge!(py, _g = (), {
            let uniko = taken.ok_or_else(closed)?;
            uniko.shutdown().await.map_err(to_pyerr)?;
            Ok(())
        })
    }

    /// Blocking variant of [`shutdown`](Self::shutdown).
    fn shutdown_sync(&self, py: Python<'_>) -> PyResult<()> {
        let taken = self.inner.lock().expect("Uniko mutex poisoned").take();
        bridge_sync!(py, _g = (), {
            let uniko = taken.ok_or_else(closed)?;
            uniko.shutdown().await.map_err(to_pyerr)?;
            Ok(())
        })
    }

    fn __repr__(&self) -> String {
        let live = self.inner.lock().map(|g| g.is_some()).unwrap_or(false);
        if live {
            "Uniko(open)".to_string()
        } else {
            "Uniko(shut down)".to_string()
        }
    }
}

/// Builder for a non-default [`PyUniko`] instance.
///
/// Obtain one with `Uniko.builder()`. Setters are chainable; `build()` is the
/// async terminal. Embedding/config/scope/extractor setters arrive in later
/// phases alongside their own wrapper types.
#[pyclass(name = "UnikoBuilder", module = "uniko")]
pub struct PyUnikoBuilder {
    // `Option` supports the take-apply-replace pattern over the consuming
    // facade builder; `None` after `build()` consumed it.
    inner: Mutex<Option<UnikoBuilder>>,
}

impl PyUnikoBuilder {
    /// Start from the facade's default builder.
    fn new() -> Self {
        Self {
            inner: Mutex::new(Some(Uniko::builder())),
        }
    }

    /// Apply a consuming builder method in place.
    fn map(&self, f: impl FnOnce(UnikoBuilder) -> UnikoBuilder) -> PyResult<()> {
        let mut guard = self.inner.lock().expect("UnikoBuilder mutex poisoned");
        let builder = guard.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("UnikoBuilder already consumed by build()")
        })?;
        *guard = Some(f(builder));
        Ok(())
    }
}

#[pymethods]
impl PyUnikoBuilder {
    /// Use a persistent store rooted at `path`.
    fn path<'py>(slf: PyRef<'py, Self>, path: String) -> PyResult<PyRef<'py, Self>> {
        slf.map(|b| b.path(path))?;
        Ok(slf)
    }

    /// Use an ephemeral in-memory store (the default).
    fn in_memory<'py>(slf: PyRef<'py, Self>) -> PyResult<PyRef<'py, Self>> {
        slf.map(UnikoBuilder::in_memory)?;
        Ok(slf)
    }

    /// Register an LLM, enabling `Agent.answer`.
    fn llm<'py>(slf: PyRef<'py, Self>, spec: &PyLlmSpec) -> PyResult<PyRef<'py, Self>> {
        let spec = spec.spec();
        slf.map(|b| b.llm(spec))?;
        Ok(slf)
    }

    /// Enable the asynchronous streaming ingest pipeline.
    fn streaming<'py>(slf: PyRef<'py, Self>, on: bool) -> PyResult<PyRef<'py, Self>> {
        slf.map(|b| b.streaming(on))?;
        Ok(slf)
    }

    /// Scope every agent's reads to its own visibility.
    fn scope_to_agent<'py>(slf: PyRef<'py, Self>) -> PyResult<PyRef<'py, Self>> {
        slf.map(UnikoBuilder::scope_to_agent)?;
        Ok(slf)
    }

    /// Build the configured instance.
    fn build<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let builder = self
            .inner
            .lock()
            .expect("UnikoBuilder mutex poisoned")
            .take()
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "UnikoBuilder already consumed by build()",
                )
            })?;
        bridge!(py, _g = (), {
            let uniko = builder.build().await.map_err(to_pyerr)?;
            Ok(PyUniko::wrap(uniko))
        })
    }

    /// Blocking variant of [`build`](Self::build).
    fn build_sync(&self, py: Python<'_>) -> PyResult<PyUniko> {
        let builder = self
            .inner
            .lock()
            .expect("UnikoBuilder mutex poisoned")
            .take()
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "UnikoBuilder already consumed by build()",
                )
            })?;
        bridge_sync!(py, _g = (), {
            let uniko = builder.build().await.map_err(to_pyerr)?;
            Ok(PyUniko::wrap(uniko))
        })
    }

    fn __repr__(&self) -> String {
        "UnikoBuilder(...)".to_string()
    }
}

/// An LLM registration for `Agent.answer`.
///
/// Construct via a provider helper (`LlmSpec.openai`,
/// `LlmSpec.llamacpp`).
#[pyclass(name = "LlmSpec", module = "uniko")]
pub struct PyLlmSpec {
    inner: LlmSpec,
}

impl PyLlmSpec {
    /// A clone of the wrapped spec, for the builder.
    pub(crate) fn spec(&self) -> LlmSpec {
        self.inner.clone()
    }
}

#[pymethods]
impl PyLlmSpec {
    /// An OpenAI-compatible chat model (key read from `OPENAI_API_KEY`).
    #[staticmethod]
    #[pyo3(signature = (alias, model_id, base_url=None))]
    fn openai(alias: String, model_id: String, base_url: Option<String>) -> Self {
        Self {
            inner: LlmSpec::openai(alias, model_id, base_url.as_deref()),
        }
    }

    /// An OpenAI-compatible chat model reading its key from `key_env`.
    #[staticmethod]
    #[pyo3(signature = (alias, model_id, key_env, base_url=None))]
    fn openai_with_key_env(
        alias: String,
        model_id: String,
        key_env: String,
        base_url: Option<String>,
    ) -> Self {
        Self {
            inner: LlmSpec::openai_with_key_env(alias, model_id, base_url.as_deref(), key_env),
        }
    }

    /// A chat model served by a llama.cpp `llama-server`.
    ///
    /// `base_url` is the OpenAI-compatible root including `/v1`, e.g.
    /// `http://127.0.0.1:8080/v1`. Pass `key_env` only when the server
    /// was started with `--api-key`.
    #[staticmethod]
    #[pyo3(signature = (alias, model_id, base_url, key_env=None))]
    fn llamacpp(
        alias: String,
        model_id: String,
        base_url: String,
        key_env: Option<String>,
    ) -> Self {
        Self {
            inner: match key_env {
                Some(env) => LlmSpec::llamacpp_with_key_env(alias, model_id, base_url, env),
                None => LlmSpec::llamacpp(alias, model_id, base_url),
            },
        }
    }

    fn __repr__(&self) -> String {
        "LlmSpec(...)".to_string()
    }
}
