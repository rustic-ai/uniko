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
    ExtractRequest, LlmSpec, Modality, PrefetchPolicy, RecallScope, Requirement, SharedRuntime,
    TextExtractor, Uniko, UnikoBuilder, UnikoError,
};

use crate::agent::PyAgent;
use crate::config::{
    config_from_py, config_to_py, embedding_from_py, nlp_from_py, reranker_from_py,
};
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

        /// What the store at ``path`` was built with, and whether a config
        /// can open it as is. ``config`` is a partial config dict over the
        /// defaults and ``embedding`` a preset name or embedding section, as
        /// on the builder. Reads only: nothing is written and no model loads.
        #[staticmethod]
        #[pyo3(signature = (path, config=None, *, embedding=None))]
        fn inspect_store<'py>(
            py: Python<'py>,
            path: String,
            config: Option<Bound<'py, PyAny>>,
            embedding: Option<Bound<'py, PyAny>>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let target = crate::store::target_config(config.as_ref(), embedding.as_ref())?;
            bridge!(py, _g = (), {
                let report = Uniko::inspect_store(&path, &target).await.map_err(to_pyerr)?;
                Python::attach(|py| crate::store::PyStoreReport::from_rust(py, &report))
            })
        }

        /// Blocking variant of ``inspect_store``.
        #[staticmethod]
        #[pyo3(signature = (path, config=None, *, embedding=None))]
        fn inspect_store_sync<'py>(
            py: Python<'py>,
            path: String,
            config: Option<Bound<'py, PyAny>>,
            embedding: Option<Bound<'py, PyAny>>,
        ) -> PyResult<Py<crate::store::PyStoreReport>> {
            let target = crate::store::target_config(config.as_ref(), embedding.as_ref())?;
            bridge_sync!(py, _g = (), {
                let report = Uniko::inspect_store(&path, &target).await.map_err(to_pyerr)?;
                Python::attach(|py| crate::store::PyStoreReport::from_rust(py, &report))
            })
        }

        /// Bring the store at ``path`` in line with a config in place:
        /// ``"enable_hybrid"`` (add and fill the sparse and ColBERT columns,
        /// same model) or ``"rebuild_vector_indexes"`` (new metric,
        /// algorithm, query prefix or batch size; a document prefix change
        /// needs a fresh store). Raises ``IncompatibleStoreError`` when the
        /// migration can't resolve the differences; a rerun resumes.
        ///
        /// ``progress`` is called with a dict (``migration``, ``phase``,
        /// ``label``, ``done``, ``total``) after each phase and batch. It
        /// can't cancel the migration: an exception it raises
        /// (``KeyboardInterrupt`` and ``SystemExit`` included) or a panic it
        /// sets off is printed to stderr and ignored. An ``async def``
        /// callback is not awaited. The callback must not call other
        /// ``_sync`` uniko methods.
        #[staticmethod]
        #[pyo3(signature = (path, migration, config=None, *, embedding=None, progress=None))]
        fn migrate_store<'py>(
            py: Python<'py>,
            path: String,
            migration: String,
            config: Option<Bound<'py, PyAny>>,
            embedding: Option<Bound<'py, PyAny>>,
            progress: Option<Py<PyAny>>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let target = crate::store::target_config(config.as_ref(), embedding.as_ref())?;
            let migration = crate::store::migration_from_str(&migration)?;
            bridge!(py, _g = (), {
                let report = Uniko::migrate_store(&path, target, migration, crate::store::progress_fn(progress))
                    .await
                    .map_err(to_pyerr)?;
                Python::attach(|py| crate::store::PyMigrationReport::from_rust(py, &report))
            })
        }

        /// Blocking variant of ``migrate_store``, with the same ``progress``
        /// rules: exceptions and panics in the callback are printed to
        /// stderr and ignored, the callback can't cancel the migration, an
        /// ``async def`` callback is not awaited, and the callback must not
        /// call other ``_sync`` uniko methods.
        #[staticmethod]
        #[pyo3(signature = (path, migration, config=None, *, embedding=None, progress=None))]
        fn migrate_store_sync<'py>(
            py: Python<'py>,
            path: String,
            migration: String,
            config: Option<Bound<'py, PyAny>>,
            embedding: Option<Bound<'py, PyAny>>,
            progress: Option<Py<PyAny>>,
        ) -> PyResult<Py<crate::store::PyMigrationReport>> {
            let target = crate::store::target_config(config.as_ref(), embedding.as_ref())?;
            let migration = crate::store::migration_from_str(&migration)?;
            bridge_sync!(py, _g = (), {
                let report = Uniko::migrate_store(&path, target, migration, crate::store::progress_fn(progress))
                    .await
                    .map_err(to_pyerr)?;
                Python::attach(|py| crate::store::PyMigrationReport::from_rust(py, &report))
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

        /// The streaming pipeline's dead letters: list, count, retry and
        /// resolve ingest items that failed.
        #[getter]
        fn dead_letters(&self) -> PyResult<crate::dead_letters::PyDeadLetters> {
            let guard = self.inner.lock().expect("Uniko mutex poisoned");
            let uniko = guard.as_ref().ok_or_else(closed)?;
            Ok(crate::dead_letters::PyDeadLetters::wrap(uniko.dead_letters()))
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

        /// What this instance's store was built with.
        fn store_info / store_info_sync(&self, py) -> Py<crate::store::PyStoreInfo> {
            prepare {
                let uniko = self.cloned()?;

            }
            run(instance = uniko) {
                let info = instance.store_info().await.map_err(to_pyerr)?;
                Python::attach(|py| crate::store::PyStoreInfo::from_rust(py, &info))

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

        /// Fill search text for chunks from before the lexical channel
        /// existed; returns how many were filled. Run once after upgrading.
        fn backfill_search_text / backfill_search_text_sync(&self, py) -> usize {
            prepare {
                let uniko = self.cloned()?;

            }
            run(instance = uniko) {
                instance.backfill_search_text().await.map_err(to_pyerr)

            }
        }

        /// What this build and instance can do, as a dict: compiled
        /// features, registered extractors, each model alias with its load
        /// state, the answer alias, and whether OCR is compiled and enabled.
        ///
        /// Loads nothing. Model entries carry no provider options; an API
        /// key appears only as its environment-variable name.
        fn capabilities / capabilities_sync(&self, py) -> Py<PyAny> {
            prepare {
                let uniko = self.cloned()?;

            }
            run(instance = uniko) {
                let report = instance.capabilities().await;
                let json = serde_json::to_value(&report)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                Python::attach(|py| crate::convert::json_to_py(py, &json))

            }
        }

        /// Check `requirements` now, loading any required model that has
        /// not loaded yet. Returns the unmet ones as dicts (`requirement`,
        /// `code`, `alias`, `error`), empty when all are met. Never raises
        /// for an unmet requirement. `timeout` (seconds) bounds the loading;
        /// a model that failed before is tried again on every call.
        #[pyo3(signature = (requirements, timeout=None))]
        fn check_readiness / check_readiness_sync(&self, py, requirements: Vec<Py<PyRequirement>>, timeout: Option<f64>) -> Py<PyAny> {
            prepare {
                let uniko = self.cloned()?;
                let requirements: Vec<Requirement> =
                    requirements.iter().map(|r| r.get().inner.clone()).collect();
                let timeout = seconds_to_timeout("check_readiness timeout", timeout)?;

            }
            run(instance = uniko) {
                let failures = instance.check_readiness(&requirements, timeout).await;
                let json = serde_json::to_value(&failures)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                Python::attach(|py| crate::convert::json_to_py(py, &json))

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
            pyo3::exceptions::PyRuntimeError::new_err(
                "UnikoBuilder already consumed by build() or build_runtime()",
            )
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

        /// Choose the NLP model behind entity and observation extraction: a
        /// preset (`"v1"`, the default cascade, or a kniv v7 size: `"v7-tiny"`,
        /// `"v7-small"`, `"v7-medium"`, `"v7-large"`, `"v7-xlarge"`,
        /// `"v7-xlarge-d"`) or a dict merged onto the default `nlp` section.
        fn nlp<'py>(slf: PyRef<'py, Self>, nlp: &Bound<'_, PyAny>) -> PyResult<PyRef<'py, Self>> {
            let nlp = nlp_from_py(nlp)?;
            slf.map(|b| b.nlp(nlp))?;
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
        ///
        /// Each call adds one more generation model; the first answers
        /// unless `answer_alias` names another.
        fn llm<'py>(slf: PyRef<'py, Self>, spec: &PyLlmSpec) -> PyResult<PyRef<'py, Self>> {
            let spec = spec.spec();
            slf.map(|b| b.llm(spec))?;
            Ok(slf)
        }

        /// Choose which generation alias `Agent.answer` uses: an `llm()`
        /// alias or one from the `catalog_path` file.
        fn answer_alias<'py>(slf: PyRef<'py, Self>, alias: String) -> PyResult<PyRef<'py, Self>> {
            slf.map(|b| b.answer_alias(alias))?;
            Ok(slf)
        }

        /// Open over a `SharedRuntime` instead of loading this instance's
        /// own models. Build it with `build_runtime()` on a builder with the
        /// same models; `build()` raises `ConfigError` when it does not serve
        /// them. Shutting this instance down leaves the runtime loaded.
        fn runtime<'py>(slf: PyRef<'py, Self>, runtime: &PySharedRuntime) -> PyResult<PyRef<'py, Self>> {
            let runtime = runtime.inner.clone();
            slf.map(|b| b.runtime(runtime))?;
            Ok(slf)
        }

        /// When to load models: `"eager"` (the default) before `build()`
        /// returns, `"lazy"` on first use, or `"background"` on a task.
        fn prefetch<'py>(slf: PyRef<'py, Self>, policy: &str) -> PyResult<PyRef<'py, Self>> {
            let policy = match policy {
                "eager" => PrefetchPolicy::Eager,
                "lazy" => PrefetchPolicy::Lazy,
                "background" => PrefetchPolicy::Background,
                other => {
                    return Err(to_pyerr(UnikoError::Config(format!(
                        "prefetch '{other}': expected eager, lazy or background"
                    ))));
                }
            };
            slf.map(|b| b.prefetch(policy))?;
            Ok(slf)
        }

        /// Require models or stages before `build()` returns. Adds to
        /// earlier calls. `build()` loads each required model whatever the
        /// `prefetch()` policy and raises `ReadinessError` listing every
        /// unmet requirement.
        fn require<'py>(
            slf: PyRef<'py, Self>,
            requirements: Vec<Py<PyRequirement>>,
        ) -> PyResult<PyRef<'py, Self>> {
            let requirements: Vec<Requirement> =
                requirements.iter().map(|r| r.get().inner.clone()).collect();
            slf.map(|b| b.require(requirements))?;
            Ok(slf)
        }

        /// How long `build()` waits, in seconds, for the required models
        /// to load. One still loading then is reported as `timed_out`.
        /// `None` or `math.inf` waits as long as loading takes.
        fn readiness_timeout<'py>(
            slf: PyRef<'py, Self>,
            seconds: Option<f64>,
        ) -> PyResult<PyRef<'py, Self>> {
            let timeout = seconds_to_timeout("readiness_timeout", seconds)?
                .unwrap_or(std::time::Duration::MAX);
            slf.map(|b| b.readiness_timeout(timeout))?;
            Ok(slf)
        }

        /// Also verify, when readiness runs, that each required remote
        /// model can reach its endpoint with a valid credential: one cheap
        /// request per alias, here and on every later `check_readiness`.
        /// A failure is `verification_failed`, its reason in `error`. Off
        /// by default.
        fn verify_remote<'py>(slf: PyRef<'py, Self>, on: bool) -> PyResult<PyRef<'py, Self>> {
            slf.map(|b| b.verify_remote(on))?;
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
                            "UnikoBuilder already consumed by build() or build_runtime()",
                        )
                    })?;

            }
            run(_g = ()) {
                let uniko = builder.build().await.map_err(to_pyerr)?;
                Ok(PyUniko::wrap(uniko))

            }
        }

        /// Build a `SharedRuntime` for this builder's models (embedding,
        /// reranker, NLP, catalog file and LLMs), loading them as
        /// `prefetch()` says. Consumes the builder, like `build()`.
        fn build_runtime / build_runtime_sync(&self, py) -> PySharedRuntime {
            prepare {
                let builder = self
                    .inner
                    .lock()
                    .expect("UnikoBuilder mutex poisoned")
                    .take()
                    .ok_or_else(|| {
                        pyo3::exceptions::PyRuntimeError::new_err(
                            "UnikoBuilder already consumed by build() or build_runtime()",
                        )
                    })?;

            }
            run(_g = ()) {
                let runtime = builder.build_runtime().await.map_err(to_pyerr)?;
                Ok(PySharedRuntime { inner: runtime })

            }
        }

    }
}

/// A model runtime several `Uniko` instances share, so each model loads
/// once. Made by `UnikoBuilder.build_runtime()`; pass it to
/// `UnikoBuilder.runtime()`. It stays loaded while any handle or instance
/// holds it.
#[pyclass(name = "SharedRuntime", module = "uniko", frozen)]
pub struct PySharedRuntime {
    inner: SharedRuntime,
}

#[pymethods]
impl PySharedRuntime {
    /// The model aliases this runtime serves, in catalog order.
    #[getter]
    fn aliases(&self) -> Vec<String> {
        self.inner
            .model_entries()
            .into_iter()
            .map(|e| e.alias)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("SharedRuntime(aliases={:?})", self.aliases())
    }
}

/// Convert a timeout in seconds: `None` or infinity is no timeout; a
/// negative or NaN value is a `ConfigError` naming `what`.
fn seconds_to_timeout(what: &str, seconds: Option<f64>) -> PyResult<Option<std::time::Duration>> {
    match seconds {
        None => Ok(None),
        Some(s) if s == f64::INFINITY => Ok(None),
        Some(s) => std::time::Duration::try_from_secs_f64(s)
            .map(Some)
            .map_err(|_| {
                to_pyerr(UnikoError::Config(format!(
                    "{what} {s}: expected a non-negative number of seconds, or None"
                )))
            }),
    }
}

/// Something an application needs before it can serve, for
/// `UnikoBuilder.require()` and `Uniko.check_readiness()`.
#[pyclass(name = "Requirement", module = "uniko", frozen)]
pub struct PyRequirement {
    inner: Requirement,
}

#[pymethods]
impl PyRequirement {
    /// A model alias from the catalog must load.
    #[staticmethod]
    fn alias(alias: String) -> Self {
        Self {
            inner: Requirement::alias(alias),
        }
    }

    /// The embedding model must load (and the hybrid embedder, if any).
    #[staticmethod]
    fn embedder() -> Self {
        Self {
            inner: Requirement::Embedder,
        }
    }

    /// The configured reranker must load.
    #[staticmethod]
    fn reranker() -> Self {
        Self {
            inner: Requirement::Reranker,
        }
    }

    /// The NLP models must load and the NLP pipeline must start.
    #[staticmethod]
    fn nlp() -> Self {
        Self {
            inner: Requirement::Nlp,
        }
    }

    /// OCR must be compiled in, enabled in the config, and load.
    #[staticmethod]
    fn ocr() -> Self {
        Self {
            inner: Requirement::Ocr,
        }
    }

    /// The answer alias must be set and must load.
    #[staticmethod]
    fn generator() -> Self {
        Self {
            inner: Requirement::Generator,
        }
    }

    /// An extractor must be registered for `modality` (`"image"`,
    /// `"audio"` or `"video"`).
    #[staticmethod]
    fn extractor(modality: &str) -> PyResult<Self> {
        let modality = match modality {
            "image" => Modality::Image,
            "audio" => Modality::Audio,
            "video" => Modality::Video,
            other => {
                return Err(to_pyerr(UnikoError::Config(format!(
                    "requirement extractor '{other}': expected image, audio or video"
                ))));
            }
        };
        Ok(Self {
            inner: Requirement::Extractor(modality),
        })
    }

    /// The alias must load on a non-CPU execution provider it asks for.
    #[staticmethod]
    fn accelerated(alias: String) -> Self {
        Self {
            inner: Requirement::accelerated(alias),
        }
    }

    /// The requirement as reported in a failure, e.g. `"alias:gen/m"`.
    #[getter]
    fn name(&self) -> String {
        self.inner.name()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut h);
        h.finish()
    }

    fn __repr__(&self) -> String {
        format!("Requirement({})", self.inner.name())
    }
}

/// An LLM registration for `Agent.answer`.
///
/// Construct via a provider helper (`LlmSpec.openai`,
/// `LlmSpec.llamacpp`) or, for any other compiled generation provider,
/// `LlmSpec.new`.
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
    /// A generation model served by any compiled provider, such as
    /// `remote/anthropic`, `remote/mistral` or `remote/gemini`.
    ///
    /// `options` are provider options (never credentials: name the
    /// credential's environment variable with `api_key_env`). The builder
    /// validates provider and options when it builds.
    #[staticmethod]
    #[pyo3(signature = (alias, provider, model_id, options=None, api_key_env=None))]
    fn new(
        alias: String,
        provider: String,
        model_id: String,
        options: Option<&Bound<'_, pyo3::types::PyDict>>,
        api_key_env: Option<String>,
    ) -> PyResult<Self> {
        let mut spec = LlmSpec::new(alias, provider, model_id);
        if let Some(options) = options {
            for (key, value) in options.iter() {
                let key: String = key.extract()?;
                spec = spec.option(key, crate::convert::py_to_json(&value)?);
            }
        }
        if let Some(var) = api_key_env {
            spec = spec.api_key_env(var);
        }
        Ok(Self { inner: spec })
    }

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

    /// The alias this model registers under.
    #[getter]
    fn alias(&self) -> &str {
        self.inner.alias()
    }

    fn __repr__(&self) -> String {
        format!("LlmSpec(alias={:?})", self.inner.alias())
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
