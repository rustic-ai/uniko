// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Industries Inc.

//! Conversions between the graph [`Value`] type and Python objects.
//!
//! [`Value`] (re-exported from uni-db via the facade) is the type that crosses
//! the boundary in both directions: as `query()` / rule-result rows
//! ([`Value`] -> Python) and as query/rule parameters (Python -> [`Value`]).
//! [`Record`] rows render as `dict[str, Any]`.
//!
//! Graph elements ([`Value::Node`], [`Value::Edge`], [`Value::Path`]) render as
//! plain dicts rather than dedicated pyclasses — a `MATCH (n) RETURN n` row is
//! then a `{"id", "labels", "properties"}` dict, which is enough for the SDK's
//! read surface.
//!
//! Binary payloads are still best retrieved through `data.artifact_bytes(id)`,
//! which reads through the blob backend. The `Bytes` arm here is nonetheless
//! live: the uni-db limitation that prevented `Value::Bytes` from being read
//! back through a Cypher `RETURN` is fixed as of uni-db 3.x (see
//! `crates/uniko-store/tests/unidb_bytes_return_repro.rs`, which now passes).

use chrono::{DateTime, Utc};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use std::collections::HashMap;
use uniko_api::tools::{Record, Value, temporal_epoch_millis};

/// Convert a `chrono::DateTime<Utc>` to a tz-aware Python `datetime` in UTC.
///
/// Used both by the [`Value::Temporal`] arm here and by output pyclasses that
/// carry raw `DateTime<Utc>` fields, so timestamps render uniformly.
///
/// # Errors
///
/// Returns a `PyErr` if the `datetime` module cannot be imported or the
/// `fromtimestamp` construction fails.
pub fn utc_datetime_to_py<'py>(py: Python<'py>, dt: DateTime<Utc>) -> PyResult<Bound<'py, PyAny>> {
    let datetime_module = py.import("datetime")?;
    let utc = datetime_module.getattr("timezone")?.getattr("utc")?;
    let dt_class = datetime_module.getattr("datetime")?;
    let secs = dt.timestamp();
    let micros = f64::from(dt.timestamp_subsec_micros()) / 1_000_000.0;
    dt_class.call_method1("fromtimestamp", (secs as f64 + micros, utc))
}

/// Convert a graph [`Value`] into a Python object.
///
/// # Errors
///
/// Returns a `PyErr` if constructing any nested Python object fails.
pub fn value_to_py(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => b.into_py_any(py),
        Value::Int(i) => i.into_py_any(py),
        Value::Float(f) => f.into_py_any(py),
        Value::String(s) => s.into_py_any(py),
        Value::Bytes(b) => Ok(PyBytes::new(py, b).into_any().unbind()),
        Value::Vector(v) => v.clone().into_py_any(py),
        Value::List(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(value_to_py(py, item)?)?;
            }
            Ok(list.into_any().unbind())
        }
        Value::Map(map) => Ok(map_to_pydict(py, map)?.into_any().unbind()),
        Value::Node(n) => {
            let dict = PyDict::new(py);
            dict.set_item("id", n.vid.as_u64())?;
            dict.set_item("labels", n.labels.clone())?;
            dict.set_item("properties", map_to_pydict(py, &n.properties)?)?;
            Ok(dict.into_any().unbind())
        }
        Value::Edge(e) => {
            let dict = PyDict::new(py);
            dict.set_item("id", e.eid.as_u64())?;
            dict.set_item("type", e.edge_type.clone())?;
            dict.set_item("start", e.src.as_u64())?;
            dict.set_item("end", e.dst.as_u64())?;
            dict.set_item("properties", map_to_pydict(py, &e.properties)?)?;
            Ok(dict.into_any().unbind())
        }
        Value::Path(p) => {
            let nodes = PyList::empty(py);
            for n in p.nodes() {
                nodes.append(value_to_py(py, &Value::Node(n.clone()))?)?;
            }
            let edges = PyList::empty(py);
            for e in p.edges() {
                edges.append(value_to_py(py, &Value::Edge(e.clone()))?)?;
            }
            let dict = PyDict::new(py);
            dict.set_item("nodes", nodes)?;
            dict.set_item("edges", edges)?;
            Ok(dict.into_any().unbind())
        }
        // `Value::Temporal` wraps uni-db's `TemporalValue`. We bridge via
        // `temporal_epoch_millis()` (the same path
        // `uniko_store::datetime_from_value` uses) to a tz-aware UTC datetime.
        // Variants without an epoch (a bare date, a duration, a BTIC interval)
        // have no clean datetime mapping and fall back to `None` rather than
        // guessing.
        Value::Temporal(t) => match temporal_epoch_millis(t) {
            Some(millis) => match DateTime::<Utc>::from_timestamp_millis(millis) {
                Some(dt) => Ok(utc_datetime_to_py(py, dt)?.unbind()),
                None => Ok(py.None()),
            },
            None => Ok(py.None()),
        },
        // `Value` is `#[non_exhaustive]` upstream; surface unknown variants as
        // `None` rather than failing the whole row.
        _ => Ok(py.None()),
    }
}

/// Convert a `HashMap<String, Value>` into a Python `dict`.
fn map_to_pydict<'py>(
    py: Python<'py>,
    map: &HashMap<String, Value>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (k, v) in map {
        dict.set_item(k, value_to_py(py, v)?)?;
    }
    Ok(dict)
}

/// Convert a [`Record`] (a result row) into a Python `dict[str, Any]`.
///
/// # Errors
///
/// Returns a `PyErr` if converting any cell value fails.
pub fn record_to_py(py: Python<'_>, record: &Record) -> PyResult<Py<PyAny>> {
    Ok(map_to_pydict(py, record)?.into_any().unbind())
}

/// Convert a list of [`Record`] rows into a Python `list[dict]`.
///
/// # Errors
///
/// Returns a `PyErr` if converting any row fails.
pub fn records_to_py(py: Python<'_>, records: &[Record]) -> PyResult<Py<PyAny>> {
    let list = PyList::empty(py);
    for record in records {
        list.append(record_to_py(py, record)?)?;
    }
    Ok(list.into_any().unbind())
}

/// Convert a Python object into a graph [`Value`] (the parameter direction).
///
/// Handles `None`, `bool`, `int`, `float`, `str`, `bytes`, `list`, and `dict`.
/// Temporal inputs (`datetime`/`date`/`time`) are not yet supported — they
/// arrive once the logic surface needs date-anchored rule parameters.
///
/// # Errors
///
/// Returns a `PyErr` if extracting a nested element fails.
pub fn py_to_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    // Order matters: `bool` is a subclass of `int` in Python, so test it first.
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(Value::Bool(b));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Value::Int(i));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(Value::Float(f));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Value::String(s));
    }
    if let Ok(b) = obj.cast::<PyBytes>() {
        return Ok(Value::Bytes(b.as_bytes().to_vec()));
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let mut items = Vec::with_capacity(list.len());
        for item in list.iter() {
            items.push(py_to_value(&item)?);
        }
        return Ok(Value::List(items));
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        let mut map = HashMap::with_capacity(dict.len());
        for (k, v) in dict.iter() {
            map.insert(k.extract::<String>()?, py_to_value(&v)?);
        }
        return Ok(Value::Map(map));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "cannot convert Python object of type '{}' to a uniko Value",
        obj.get_type().name()?
    )))
}

/// Convert a Python object into a `serde_json::Value`.
///
/// Used for the free-form `Turn.metadata` values, which the facade carries as
/// `serde_json::Value`. Handles `None`/`bool`/`int`/`float`/`str`/`list`/`dict`;
/// anything else maps to JSON `null`.
///
/// # Errors
///
/// Returns a `PyErr` if extracting a nested element fails.
pub fn py_to_json(obj: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if obj.is_none() {
        return Ok(serde_json::Value::Null);
    }
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(serde_json::Value::Bool(b));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(serde_json::Value::from(i));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(serde_json::Value::from(f));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(serde_json::Value::String(s));
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let mut out = Vec::with_capacity(list.len());
        for item in list.iter() {
            out.push(py_to_json(&item)?);
        }
        return Ok(serde_json::Value::Array(out));
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (k, v) in dict.iter() {
            map.insert(k.extract::<String>()?, py_to_json(&v)?);
        }
        return Ok(serde_json::Value::Object(map));
    }
    Ok(serde_json::Value::Null)
}

/// Convert a `serde_json::Value` into a Python object.
///
/// The inverse of [`py_to_json`], used for free-form JSON fields on output
/// types (e.g. a goal's `metrics`).
///
/// # Errors
///
/// Returns a `PyErr` if constructing any nested Python object fails.
pub fn json_to_py(py: Python<'_>, value: &serde_json::Value) -> PyResult<Py<PyAny>> {
    match value {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => b.into_py_any(py),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_py_any(py)
            } else if let Some(f) = n.as_f64() {
                f.into_py_any(py)
            } else {
                Ok(py.None())
            }
        }
        serde_json::Value::String(s) => s.into_py_any(py),
        serde_json::Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(json_to_py(py, item)?)?;
            }
            Ok(list.into_any().unbind())
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

/// Convert a Python `datetime` into a `chrono::DateTime<Utc>`.
///
/// Uses Python's own `.timestamp()` (a tz-naive datetime is interpreted in the
/// local timezone, matching CPython semantics).
///
/// # Errors
///
/// Returns a `PyErr` if the object is not datetime-like or the instant falls
/// outside the representable range.
pub fn py_to_datetime(obj: &Bound<'_, PyAny>) -> PyResult<DateTime<Utc>> {
    let ts: f64 = obj.call_method0("timestamp")?.extract()?;
    let secs = ts.trunc() as i64;
    let nanos = (ts.fract() * 1_000_000_000.0).round() as u32;
    DateTime::<Utc>::from_timestamp(secs, nanos).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("datetime is outside the representable range")
    })
}

/// Convert an optional Python params dict into Rust params.
///
/// # Errors
///
/// Returns a `PyErr` if any value cannot be converted to a [`Value`].
pub fn params_to_rust(params: Option<&Bound<'_, PyDict>>) -> PyResult<HashMap<String, Value>> {
    let mut out = HashMap::new();
    if let Some(dict) = params {
        for (k, v) in dict.iter() {
            out.insert(k.extract::<String>()?, py_to_value(&v)?);
        }
    }
    Ok(out)
}
