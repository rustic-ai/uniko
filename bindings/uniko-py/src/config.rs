// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2026 Dragonscale Industries Inc.

//! Python <-> [`UnikoConfig`] conversion for `UnikoBuilder.config()`,
//! `.embedding()`, `.reranker()` and `Uniko.config()`.
//!
//! Input is merged onto the serialized default rather than deserialized on
//! its own, for two reasons. `UnikoConfig` is `#[serde(default)]` without
//! `deny_unknown_fields`, so a typo such as `recal_limit` would otherwise be
//! dropped without a word; checking every key against the default catches
//! it. And the nested `EmbeddingConfig` / `RerankerConfig` have required
//! fields, so `{"embedding": {"batch_size": 8}}` only deserializes once the
//! rest of the section is filled in from the default.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::{Map, Value as Json};
use uniko_api::tools::{EmbeddingConfig, RerankerConfig, UnikoConfig, UnikoError};

use crate::convert::json_to_py;
use crate::errors::to_pyerr;

/// What `Uniko.config()` shows in place of `embedding.provider_options`,
/// which can carry API keys.
const REDACTED: &str = "<redacted>";

/// Sections whose value is a tagged enum, so a new variant brings keys the
/// default lacks (`{"kind": "fs", "root": ...}` over `{"kind": "lance"}`).
/// The input replaces these whole instead of being merged key by key.
const REPLACED_WHOLE: &[&str] = &["blob_storage", "vector_algorithm"];

/// Reranker presets accepted by `UnikoBuilder.reranker(name)`.
const RERANKER_PRESETS: &[&str] = &["bge-base", "minilm"];

fn config_error(msg: impl Into<String>) -> PyErr {
    to_pyerr(UnikoError::Config(msg.into()))
}

/// Convert a Python value to JSON, raising on anything that has no JSON
/// form. `path` names the key in errors.
fn to_json(obj: &Bound<'_, PyAny>, path: &str) -> PyResult<Json> {
    if obj.is_none() {
        return Ok(Json::Null);
    }
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(Json::Bool(b));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Json::from(i));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return serde_json::Number::from_f64(f)
            .map(Json::Number)
            .ok_or_else(|| config_error(format!("{path}: {f} is not a finite number")));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Json::String(s));
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let mut out = Vec::with_capacity(list.len());
        for (i, item) in list.iter().enumerate() {
            out.push(to_json(&item, &format!("{path}[{i}]"))?);
        }
        return Ok(Json::Array(out));
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        let mut map = Map::new();
        for (k, v) in dict.iter() {
            let key: String = k
                .extract()
                .map_err(|_| config_error(format!("{path}: config keys must be strings")))?;
            let child = join(path, &key);
            map.insert(key, to_json(&v, &child)?);
        }
        return Ok(Json::Object(map));
    }
    // `os.PathLike` (for `schema_path`, `catalog_path`, blob roots).
    if let Ok(fspath) = obj.call_method0("__fspath__") {
        return Ok(Json::String(fspath.extract()?));
    }
    Err(config_error(format!(
        "{path}: unsupported value of type {}",
        obj.get_type().name()?
    )))
}

fn join(path: &str, key: &str) -> String {
    if path.is_empty() {
        key.to_string()
    } else {
        format!("{path}.{key}")
    }
}

/// Merge `input` onto `base` in place, rejecting keys `base` lacks.
///
/// Recursion stops where the input replaces the value whole: where the
/// default is not an object (`null`, a scalar, a list), is an empty object
/// (a free-form map such as `phase2_graph_edge_weights`), or is one of
/// [`REPLACED_WHOLE`]. That covers sections such as
/// `embedding.provider_options`, whose default is `null`.
///
/// Each value that lands is recorded in `set` with its dotted path, so a
/// later deserialization error can be traced to the key that caused it.
fn merge(base: &mut Json, input: Json, path: &str, set: &mut Vec<(String, Json)>) -> PyResult<()> {
    let replace_whole =
        REPLACED_WHOLE.contains(&path) || base.as_object().is_some_and(Map::is_empty);
    match (base, input) {
        (slot, value) if replace_whole => {
            set.push((path.to_string(), value.clone()));
            *slot = value;
            Ok(())
        }
        (Json::Object(base_map), Json::Object(input_map)) => {
            for (key, value) in input_map {
                let child = join(path, &key);
                match base_map.get_mut(&key) {
                    Some(slot) => merge(slot, value, &child, set)?,
                    None => {
                        return Err(config_error(format!(
                            "unknown config key '{child}'{}",
                            suggestion(base_map, &key)
                        )));
                    }
                }
            }
            Ok(())
        }
        (Json::Object(_), other) if !other.is_object() => Err(config_error(format!(
            "{}: expected a mapping, got {}",
            if path.is_empty() { "config" } else { path },
            kind(&other)
        ))),
        (slot, value) => {
            set.push((path.to_string(), value.clone()));
            *slot = value;
            Ok(())
        }
    }
}

/// Write `value` at the dotted `path` inside `root`.
fn set_at(root: &mut Json, path: &str, value: Json) {
    let mut slot = root;
    for key in path.split('.') {
        match slot {
            Json::Object(map) => slot = map.entry(key.to_string()).or_insert(Json::Null),
            _ => return,
        }
    }
    *slot = value;
}

/// ` (did you mean 'x'?)` for a near-miss key, else empty.
fn suggestion(known: &Map<String, Json>, key: &str) -> String {
    known
        .keys()
        .map(|k| (edit_distance(k, key), k))
        .filter(|(d, _)| *d <= 2)
        .min()
        .map(|(_, k)| format!(" (did you mean '{k}'?)"))
        .unwrap_or_default()
}

fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, cb) in b.iter().enumerate() {
            let cost = usize::from(ca != *cb);
            cur[j + 1] = (prev[j] + cost).min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[b.len()]
}

fn kind(value: &Json) -> &'static str {
    match value {
        Json::Null => "None",
        Json::Bool(_) => "bool",
        Json::Number(_) => "a number",
        Json::String(_) => "str",
        Json::Array(_) => "list",
        Json::Object(_) => "a mapping",
    }
}

/// Serialize a default, merge `input` onto it, and deserialize the result.
fn merged<T>(default: &T, input: Json, path: &str) -> PyResult<T>
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    let default = serde_json::to_value(default)
        .map_err(|e| config_error(format!("serializing the default config: {e}")))?;
    let mut base = default.clone();
    let mut set = Vec::new();
    merge(&mut base, input, path, &mut set)?;
    serde_json::from_value(base).map_err(|err| {
        // serde does not say which field failed; find the key that does by
        // applying each one alone to the default. The input is a handful of
        // keys, so this costs little and only runs on the error path.
        let culprit = set.iter().find_map(|(key, value)| {
            let mut alone = default.clone();
            let relative = key.strip_prefix(&format!("{path}.")).unwrap_or(key);
            set_at(&mut alone, relative, value.clone());
            serde_json::from_value::<T>(alone).err().map(|e| (key, e))
        });
        match culprit {
            Some((key, e)) => config_error(format!("invalid value for '{key}': {e}")),
            None => config_error(format!(
                "invalid {}: {err}",
                if path.is_empty() { "config" } else { path }
            )),
        }
    })
}

/// The JSON of a `dict` or a Pydantic model (anything with `model_dump`).
///
/// A model is dumped with `exclude_unset=True`, so only the fields the
/// caller set override the defaults — the same as passing a partial dict.
fn input_json(obj: &Bound<'_, PyAny>, path: &str) -> PyResult<Json> {
    if obj.cast::<PyDict>().is_ok() {
        return to_json(obj, path);
    }
    if obj.hasattr("model_dump")? {
        let kwargs = PyDict::new(obj.py());
        kwargs.set_item("exclude_unset", true)?;
        let dumped = obj.call_method("model_dump", (), Some(&kwargs))?;
        return to_json(&dumped, path);
    }
    Err(config_error(format!(
        "{}: expected a dict or a UnikoConfigModel, got {}",
        if path.is_empty() { "config" } else { path },
        obj.get_type().name()?
    )))
}

/// `UnikoBuilder.config(...)`: a partial config over the defaults.
pub fn config_from_py(obj: &Bound<'_, PyAny>) -> PyResult<UnikoConfig> {
    let input = input_json(obj, "")?;
    merged(&UnikoConfig::default(), input, "")
}

/// `UnikoBuilder.embedding(...)`: a preset name, or a partial embedding
/// section over the default embedder.
pub fn embedding_from_py(obj: &Bound<'_, PyAny>) -> PyResult<EmbeddingConfig> {
    if let Ok(name) = obj.extract::<String>() {
        return EmbeddingConfig::preset(&name).ok_or_else(|| {
            config_error(format!(
                "unknown embedding preset '{name}'; expected one of: nomic, nomic-q, \
                 minilm, bge-small, bge-large, bge-m3, gemma"
            ))
        });
    }
    let input = input_json(obj, "embedding")?;
    merged(&UnikoConfig::default().embedding, input, "embedding")
}

/// `UnikoBuilder.reranker(...)`: a preset name, a partial reranker section
/// over the default, or `None` to turn reranking off.
pub fn reranker_from_py(obj: &Bound<'_, PyAny>) -> PyResult<RerankerConfig> {
    if obj.is_none() {
        return Ok(RerankerConfig {
            enabled: false,
            ..RerankerConfig::default()
        });
    }
    if let Ok(name) = obj.extract::<String>() {
        let preset = match name.as_str() {
            "bge-base" => RerankerConfig::bge_base(),
            "minilm" => RerankerConfig::minilm_l6(),
            _ => {
                return Err(config_error(format!(
                    "unknown reranker preset '{name}'; expected one of: {}, or None",
                    RERANKER_PRESETS.join(", ")
                )));
            }
        };
        // The presets come disabled; naming one means using it.
        return Ok(RerankerConfig {
            enabled: true,
            ..preset
        });
    }
    let input = input_json(obj, "reranker")?;
    merged(&RerankerConfig::default(), input, "reranker")
}

/// `Uniko.config()`: the full configuration as nested dicts, with
/// `embedding.provider_options` redacted.
pub fn config_to_py(py: Python<'_>, config: &UnikoConfig) -> PyResult<Py<PyAny>> {
    let mut json = serde_json::to_value(config)
        .map_err(|e| config_error(format!("serializing the config: {e}")))?;
    if let Some(options) = json
        .get_mut("embedding")
        .and_then(|e| e.get_mut("provider_options"))
        && !options.is_null()
    {
        *options = Json::String(REDACTED.to_string());
    }
    json_to_py(py, &json)
}
