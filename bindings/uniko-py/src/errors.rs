// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Industries Inc.

//! Typed Python exception hierarchy mirroring [`CoreError`].
//!
//! Every binding error is catchable as the base `uniko.UnikoError`. The
//! variants a caller realistically branches on (`config`, `llm`, `timeout`,
//! `conflict`, `unsupported`) get dedicated subclasses; the remaining
//! storage-domain variants fold into the base, all carrying a `.kind` string
//! attribute and the Rust `Display` message. [`to_pyerr`] is the single
//! conversion site.

use pyo3::prelude::*;
use pyo3::{create_exception, exceptions::PyException};
// The core error enum. Aliased so its name does not collide with the
// `UnikoError` Python exception class created below (which must carry that
// exact name for readable Python tracebacks).
use uniko_api::tools::UnikoError as CoreError;

// ============================================================================
// Exception hierarchy
// ============================================================================

create_exception!(
    _uniko,
    UnikoError,
    PyException,
    "Base exception for all uniko errors. Carries a `.kind` string attribute."
);
create_exception!(
    _uniko,
    ConfigError,
    UnikoError,
    "Configuration was invalid or incomplete (e.g. answer() with no LLM alias)."
);
create_exception!(_uniko, LlmError, UnikoError, "An LLM provider call failed.");
create_exception!(
    _uniko,
    TimeoutError,
    UnikoError,
    "An operation exceeded its configured timeout."
);
create_exception!(
    _uniko,
    ConflictError,
    UnikoError,
    "A transient storage conflict (SSI / antidependency abort). Retriable."
);
create_exception!(
    _uniko,
    IdConflictError,
    UnikoError,
    "A stable external id was reused for different content. Not retriable."
);
create_exception!(
    _uniko,
    UnsupportedError,
    UnikoError,
    "The content modality has no registered extractor."
);

// ============================================================================
// Error conversion
// ============================================================================

/// Convert a [`CoreError`] into the matching typed Python exception.
///
/// The returned `PyErr` always carries a `.kind` attribute naming the
/// originating variant, so callers can branch on `e.kind` even for the
/// storage-domain errors that share the base class.
#[must_use]
pub fn to_pyerr(err: CoreError) -> PyErr {
    let msg = err.to_string();
    let (pyerr, kind) = match err {
        CoreError::Config(_) => (ConfigError::new_err(msg), "config"),
        CoreError::Llm(_) => (LlmError::new_err(msg), "llm"),
        CoreError::Timeout(_) => (TimeoutError::new_err(msg), "timeout"),
        CoreError::Conflict(_) => (ConflictError::new_err(msg), "conflict"),
        CoreError::IdConflict(_) => (IdConflictError::new_err(msg), "id_conflict"),
        CoreError::Unsupported(_) => (UnsupportedError::new_err(msg), "unsupported"),
        CoreError::Storage(_) => (UnikoError::new_err(msg), "storage"),
        CoreError::Search(_) => (UnikoError::new_err(msg), "search"),
        CoreError::Schema(_) => (UnikoError::new_err(msg), "schema"),
        CoreError::Pipeline(_) => (UnikoError::new_err(msg), "pipeline"),
        CoreError::Locy(_) => (UnikoError::new_err(msg), "locy"),
        CoreError::Embedding(_) => (UnikoError::new_err(msg), "embedding"),
        CoreError::Internal(_) => (UnikoError::new_err(msg), "internal"),
    };
    attach_kind(pyerr, kind)
}

/// Attach a `kind` attribute to an error value. A failure to set the attribute
/// is suppressed — it is impossible in practice (a plain string on a freshly
/// built exception) and dropping the attribute beats panicking mid-conversion.
fn attach_kind(err: PyErr, kind: &str) -> PyErr {
    Python::attach(|py| {
        let _ = err.value(py).setattr("kind", kind);
    });
    err
}

// ============================================================================
// Module registration
// ============================================================================

/// Register all binding exception types on the Python module.
///
/// # Errors
///
/// Returns a `PyErr` if adding any class to the module fails.
pub fn register_exceptions(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("UnikoError", py.get_type::<UnikoError>())?;
    m.add("ConfigError", py.get_type::<ConfigError>())?;
    m.add("LlmError", py.get_type::<LlmError>())?;
    m.add("TimeoutError", py.get_type::<TimeoutError>())?;
    m.add("ConflictError", py.get_type::<ConflictError>())?;
    m.add("IdConflictError", py.get_type::<IdConflictError>())?;
    m.add("UnsupportedError", py.get_type::<UnsupportedError>())?;
    Ok(())
}
