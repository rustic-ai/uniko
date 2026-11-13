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
use uniko_api::tools::{
    ExtractRequest, LlmSpec, Modality, RecallScope, TextExtractor, Uniko, UnikoBuilder, UnikoError,
};

use crate::agent::PyAgent;
use crate::config::{config_from_py, config_to_py, embedding_from_py, reranker_from_py};
use crate::errors::to_pyerr;
use crate::macros::{bridge, bridge_sync, pymethods_with_sync};
use crate::outputs::{PyDeletionReport, PyPipelineHealth};
use crate::scope::PyViewer;

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

pymethods_with_sync! {
    PyUniko;
    plain {
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

        /// The full configuration this instance runs with, as nested dicts.
        ///
        /// The shape matches what `UnikoBuilder.config()` accepts, so the result
        /// can be edited and passed back. `embedding.provider_options` is shown
        /// as `"<redacted>"` when set, since it can hold API keys.
        fn config(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            let guard = self.inner.lock().expect("Uniko mutex poisoned");
            let uniko = guard.as_ref().ok_or_else(closed)?;
            config_to_py(py, uniko.config())
        }

        /// A health snapshot of the streaming pipeline, or `None` when the
        /// instance was built without `streaming(True)`.
        fn health(&self, py: Python<'_>) -> PyResult<Option<Py<PyPipelineHealth>>> {
            let guard = self.inner.lock().expect("Uniko mutex poisoned");
            let uniko = guard.as_ref().ok_or_else(closed)?;
            uniko
                .health()
                .map(|h| PyPipelineHealth::from_rust(py, &h))
                .transpose()
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
    bridged {
        /// Record that `participant_id` is a member of team `team_id`, so it
        /// sees items with `team:{team_id}` visibility. Creates the participant
        /// and team when absent. Returns False when already recorded.
        ///
        /// An `Agent` resolves its own memberships on its first scoped read, so
        /// take a fresh one from `agent()` to see a change.
        fn add_team_member / add_team_member_sync(&self, py, participant_id: String, team_id: String) -> bool {
            prepare {
                let uniko = self.cloned()?;

            }
            run(instance = uniko) {
                instance
                    .add_team_member(&participant_id, &team_id)
                    .await
                    .map_err(to_pyerr)

            }
        }

        /// Undo `add_team_member`. Returns False when it was not recorded.
        fn remove_team_member / remove_team_member_sync(&self, py, participant_id: String, team_id: String) -> bool {
            prepare {
                let uniko = self.cloned()?;

            }
            run(instance = uniko) {
                instance
                    .remove_team_member(&participant_id, &team_id)
                    .await
                    .map_err(to_pyerr)

            }
        }

        /// Record that `participant_id` is a member of org `org_id`. Returns False when already recorded.
        fn add_org_member / add_org_member_sync(&self, py, participant_id: String, org_id: String) -> bool {
            prepare {
                let uniko = self.cloned()?;

            }
            run(instance = uniko) {
                instance
                    .add_org_member(&participant_id, &org_id)
                    .await
                    .map_err(to_pyerr)

            }
        }

        /// Undo `add_org_member`. Returns False when it was not recorded.
        fn remove_org_member / remove_org_member_sync(&self, py, participant_id: String, org_id: String) -> bool {
            prepare {
                let uniko = self.cloned()?;

            }
            run(instance = uniko) {
                instance
                    .remove_org_member(&participant_id, &org_id)
                    .await
                    .map_err(to_pyerr)

            }
        }

        /// Record that team `team_id` belongs to org `org_id`, giving its members that org. Returns False when already recorded.
        fn add_team_to_org / add_team_to_org_sync(&self, py, team_id: String, org_id: String) -> bool {
            prepare {
                let uniko = self.cloned()?;

            }
            run(instance = uniko) {
                instance
                    .add_team_to_org(&team_id, &org_id)
                    .await
                    .map_err(to_pyerr)

            }
        }

        /// Undo `add_team_to_org`. Returns False when it was not recorded.
        fn remove_team_from_org / remove_team_from_org_sync(&self, py, team_id: String, org_id: String) -> bool {
            prepare {
                let uniko = self.cloned()?;

            }
            run(instance = uniko) {
                instance
                    .remove_team_from_org(&team_id, &org_id)
                    .await
                    .map_err(to_pyerr)

            }
        }

        /// Erase the entire graph. Intended for dev/test reset.
        fn purge / purge_sync(&self, py) -> Py<PyDeletionReport> {
            prepare {
                let uniko = self.cloned()?;

            }
            run(instance = uniko) {
                let report = instance.purge().await.map_err(to_pyerr)?;
                Python::attach(|py| PyDeletionReport::from_rust(py, &report))

            }
        }

        /// Shut down the instance and drain the pipeline.
        ///
        /// Consumes the handle: all `Agent` / `Session` objects derived from it
        /// must be dropped first, or the store's reference-count guard fails.
        /// Using the handle afterwards raises.
        fn shutdown / shutdown_sync(&self, py) -> () {
            prepare {
                let taken = self.inner.lock().expect("Uniko mutex poisoned").take();

            }
            run(_g = ()) {
                let uniko = taken.ok_or_else(closed)?;
                uniko.shutdown().await.map_err(to_pyerr)?;
                Ok(())

            }
        }

    }
}

/// Builder for a non-default [`PyUniko`] instance.
///
/// Obtain one with `Uniko.builder()`. Setters are chainable; `build()` is the
/// async terminal. The configuration is validated once, at `build()`.
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

pymethods_with_sync! {
    PyUnikoBuilder;
    plain {
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

        /// Set the base configuration from a dict or a `UnikoConfigModel`.
        ///
        /// Partial input is merged onto the defaults, nested sections included.
        /// An unknown key raises `ConfigError` naming it. `embedding()` and
        /// `reranker()` override the matching sections whatever the call order.
        fn config<'py>(slf: PyRef<'py, Self>, config: &Bound<'_, PyAny>) -> PyResult<PyRef<'py, Self>> {
            let config = config_from_py(config)?;
            slf.map(|b| b.raw_config(config))?;
            Ok(slf)
        }

        /// Choose the embedding model: a preset name (`"nomic"`, `"minilm"`,
        /// `"bge-small"`, `"bge-large"`, `"bge-m3"`, `"gemma"`, ...) or a dict
        /// merged onto the default embedding section.
        fn embedding<'py>(
            slf: PyRef<'py, Self>,
            embedding: &Bound<'_, PyAny>,
        ) -> PyResult<PyRef<'py, Self>> {
            let embedding = embedding_from_py(embedding)?;
            slf.map(|b| b.embedding(embedding))?;
            Ok(slf)
        }

        /// Choose the reranker: `"bge-base"`, `"minilm"`, a dict merged onto the
        /// default reranker section, or `None` to turn reranking off.
        fn reranker<'py>(
            slf: PyRef<'py, Self>,
            reranker: &Bound<'_, PyAny>,
        ) -> PyResult<PyRef<'py, Self>> {
            let reranker = reranker_from_py(reranker)?;
            slf.map(|b| b.reranker(reranker))?;
            Ok(slf)
        }

        /// Index image, audio or video sources as the text `extract` returns.
        ///
        /// `modality` is `"image"`, `"audio"` or `"video"`. `extract` is called
        /// with an `ExtractRequest` (the payload `data`, its `mime`, `path` and
        /// `metadata`) and returns the text to index — a caption, a transcript,
        /// OCR output. It runs on a worker thread, holding the GIL only while
        /// it runs; an exception fails that ingest. Registering a modality
        /// again replaces the earlier extractor.
        fn extractor<'py>(
            slf: PyRef<'py, Self>,
            modality: &str,
            extract: Py<PyAny>,
        ) -> PyResult<PyRef<'py, Self>> {
            let modality = match modality {
                "image" => Modality::Image,
                "audio" => Modality::Audio,
                "video" => Modality::Video,
                other => {
                    return Err(to_pyerr(UnikoError::Config(format!(
                        "extractor modality '{other}': expected image, audio or video"
                    ))));
                }
            };
            if !extract.bind(slf.py()).is_callable() {
                return Err(to_pyerr(UnikoError::Config(
                    "extractor: `extract` must be callable".to_string(),
                )));
            }
            let extractor =
                TextExtractor::new(modality, move |request| call_extract(&extract, request));
            slf.map(|b| b.extractor(std::sync::Arc::new(extractor)))?;
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

        /// Set every agent's read scope: a `Viewer` reads as that viewer, and
        /// `None` reads everything (the default). `scope_to_agent()` reads as
        /// each agent itself.
        fn scope<'py>(slf: PyRef<'py, Self>, viewer: Option<&PyViewer>) -> PyResult<PyRef<'py, Self>> {
            let scope = match viewer {
                Some(v) => RecallScope::As(v.viewer()),
                None => RecallScope::Unrestricted,
            };
            slf.map(|b| b.scope(scope))?;
            Ok(slf)
        }

        fn __repr__(&self) -> String {
            "UnikoBuilder(...)".to_string()
        }

    }
    bridged {
        /// Build the configured instance.
        fn build / build_sync(&self, py) -> PyUniko {
            prepare {
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

            }
            run(_g = ()) {
                let uniko = builder.build().await.map_err(to_pyerr)?;
                Ok(PyUniko::wrap(uniko))

            }
        }

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

/// Call a Python extractor with `request`, returning its text.
fn call_extract(extract: &Py<PyAny>, request: ExtractRequest) -> Result<String, UnikoError> {
    Python::attach(|py| {
        let request = Py::new(
            py,
            PyExtractRequest::from_rust(py, request).map_err(py_err)?,
        )
        .map_err(py_err)?;
        let text = extract.call1(py, (request,)).map_err(py_err)?;
        text.extract::<String>(py)
            .map_err(|_| UnikoError::Config("extractor must return a str".to_string()))
    })
}

fn py_err(err: PyErr) -> UnikoError {
    UnikoError::Internal(format!("extractor raised: {err}"))
}

/// What a `UnikoBuilder.extractor` function receives.
#[pyclass(name = "ExtractRequest", module = "uniko", frozen)]
pub struct PyExtractRequest {
    /// `"image"`, `"audio"` or `"video"`.
    #[pyo3(get)]
    modality: &'static str,
    /// The payload.
    #[pyo3(get)]
    data: Py<pyo3::types::PyBytes>,
    /// The resolved MIME type.
    #[pyo3(get)]
    mime: String,
    /// The source path, when known.
    #[pyo3(get)]
    path: Option<String>,
    /// The source's metadata.
    #[pyo3(get)]
    metadata: Py<PyAny>,
}

impl PyExtractRequest {
    fn from_rust(py: Python<'_>, request: ExtractRequest) -> PyResult<Self> {
        let metadata = serde_json::Value::Object(request.metadata.into_iter().collect());
        Ok(Self {
            modality: match request.modality {
                Modality::Image => "image",
                Modality::Audio => "audio",
                Modality::Video => "video",
                _ => "other",
            },
            data: pyo3::types::PyBytes::new(py, &request.bytes).unbind(),
            mime: request.mime.to_string(),
            path: request.path,
            metadata: crate::convert::json_to_py(py, &metadata)?,
        })
    }
}

#[pymethods]
impl PyExtractRequest {
    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "ExtractRequest(modality='{}', mime='{}', bytes={})",
            self.modality,
            self.mime,
            self.data.bind(py).as_bytes().len()
        )
    }
}
