// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Industries Inc.

//! Async-first Python bindings for the uniko cognitive memory engine.
//!
//! The crate compiles to a `cdylib` loaded by Python as `uniko._uniko` and
//! wraps the [`Uniko`](uniko_api::Uniko) facade. The whole facade is `async fn`
//! on tokio, so the module owns a single multi-threaded runtime
//! ([`pyo3_async_runtimes`]) and every method returns a Python awaitable.
//!
//! Phase 0 establishes the foundation: the runtime, the typed exception
//! hierarchy ([`errors`]), and the [`Value`](uniko_api::Value) <-> Python
//! converter ([`convert`]). The facade surface (Uniko/Agent/Session, recall,
//! ingest, goals, logic) is layered on in later phases.

// `future_into_py` requires each facade future be `Send + 'static`. Proving
// `Send` for the deeply nested store futures (lance/moka cache types) exceeds
// the default recursion limit of 128.
#![recursion_limit = "512"]

mod agent;
mod config;
mod convert;
mod data;
mod errors;
mod facade;
mod goals;
mod ingest;
mod logic;
mod macros;
mod outputs;
mod scope;
mod session;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use uniko_api::tools::{Record, UnikoError};

/// Round-trip a Python value through the [`Value`](uniko_api::Value) converter.
///
/// Exercises both [`convert::py_to_value`] and [`convert::value_to_py`]; a
/// foundation smoke-test seam used by the Phase 0 test suite.
///
/// # Errors
///
/// Returns a `PyErr` if the value cannot be converted in either direction.
#[pyfunction]
fn _value_roundtrip(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let value = convert::py_to_value(obj)?;
    convert::value_to_py(py, &value)
}

/// Round-trip a list of Python dicts through the [`Record`] row converter.
///
/// Exercises [`convert::params_to_rust`] and [`convert::records_to_py`].
///
/// # Errors
///
/// Returns a `PyErr` if any row cannot be converted.
#[pyfunction]
fn _records_roundtrip(py: Python<'_>, rows: Vec<Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
    let records: Vec<Record> = rows
        .iter()
        .map(|d| convert::params_to_rust(Some(d)))
        .collect::<PyResult<_>>()?;
    convert::records_to_py(py, &records)
}

/// Raise the typed Python exception that [`errors::to_pyerr`] maps a named
/// [`UnikoError`] variant to.
///
/// A Phase 0 smoke-test seam: it lets the test suite assert the exception
/// class and `.kind` attribute for each variant before the facade methods
/// that produce real errors exist.
///
/// # Errors
///
/// Always returns a `PyErr` — that is the function's purpose.
#[pyfunction]
fn _raise_error(kind: &str) -> PyResult<()> {
    let err = match kind {
        "config" => UnikoError::Config("demo".into()),
        "llm" => UnikoError::Llm("demo".into()),
        "timeout" => UnikoError::Timeout(123),
        "conflict" => UnikoError::Conflict("demo".into()),
        "unsupported" => UnikoError::Unsupported("demo".into()),
        "storage" => UnikoError::Storage("demo".into()),
        "search" => UnikoError::Search("demo".into()),
        "schema" => UnikoError::Schema("demo".into()),
        "pipeline" => UnikoError::Pipeline("demo".into()),
        "locy" => UnikoError::Locy("demo".into()),
        "embedding" => UnikoError::Embedding("demo".into()),
        _ => UnikoError::Internal("demo".into()),
    };
    Err(errors::to_pyerr(err))
}

/// Return the current time as a tz-aware UTC `datetime`.
///
/// Exercises [`convert::utc_datetime_to_py`], the shared timestamp bridge.
///
/// # Errors
///
/// Returns a `PyErr` if the Python `datetime` construction fails.
#[pyfunction]
fn _now_utc(py: Python<'_>) -> PyResult<Py<PyAny>> {
    Ok(convert::utc_datetime_to_py(py, chrono::Utc::now())?.unbind())
}

/// The `uniko._uniko` extension module.
///
/// `gil_used = true`: this extension holds the GIL throughout (`Python::attach`
/// at every boundary) and is not audited for free-threaded safety, so it must
/// keep advertising that it requires the GIL.
///
/// # Errors
///
/// Returns a `PyErr` if runtime initialization or class/function registration
/// fails.
#[pymodule(gil_used = true)]
fn _uniko(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // One multi-threaded tokio runtime backs every awaitable. An 8 MB worker
    // stack mirrors the sibling uni-db binding: the recall/ingest pipelines
    // build deeply nested async state machines that overflow the default 2 MB
    // stack in debug builds.
    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.enable_all().thread_stack_size(8 * 1024 * 1024);
    pyo3_async_runtimes::tokio::init(builder);

    // Single-source the runtime version from Cargo: `CARGO_PKG_VERSION` is the
    // crate version, which is `version.workspace = true` -> the workspace
    // version. Exposes `uniko._uniko.__version__` (declared in the .pyi stub) so
    // `uniko.__version__` tracks the workspace version instead of a hardcoded
    // Python fallback.
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_function(wrap_pyfunction!(_value_roundtrip, m)?)?;
    m.add_function(wrap_pyfunction!(_records_roundtrip, m)?)?;
    m.add_function(wrap_pyfunction!(_now_utc, m)?)?;
    m.add_function(wrap_pyfunction!(_raise_error, m)?)?;

    // Facade handles.
    m.add_class::<facade::PyUniko>()?;
    m.add_class::<facade::PyUnikoBuilder>()?;
    m.add_class::<facade::PyLlmSpec>()?;
    m.add_class::<facade::PyExtractRequest>()?;
    m.add_class::<agent::PyAgent>()?;
    m.add_class::<session::PySession>()?;
    m.add_class::<session::PyTurn>()?;
    m.add_class::<data::PyData>()?;
    m.add_class::<goals::PyGoals>()?;
    m.add_class::<ingest::PyIngestSource>()?;
    m.add_class::<scope::PyScope>()?;
    m.add_class::<scope::PyViewer>()?;
    m.add_class::<logic::PyAssumeBuilder>()?;

    // Output value types.
    outputs::register(m)?;

    errors::register_exceptions(py, m)?;

    // `PyModule::add` appends every name to `__all__`, including the
    // underscore test helpers and `__version__`; keep `from uniko import *`
    // to the public API.
    let public: Vec<String> = m
        .index()?
        .iter()
        .map(|name| name.extract::<String>())
        .collect::<PyResult<Vec<_>>>()?
        .into_iter()
        .filter(|name| !name.starts_with('_'))
        .collect();
    m.setattr("__all__", public)?;

    Ok(())
}
