//! Per-alias model load state and the status report built on it.
//!
//! uni-xervo keeps its load state private (its handle cache and registry
//! are not public), so the store records what it observes itself: the
//! result of prefetching each alias, and the first successful call through
//! [`KnowledgeBase::embed`], [`KnowledgeBase::generate`] and friends. A
//! model loaded behind the store's back (uni-db auto-embedding on write,
//! or NLP through [`KnowledgeBase::model_runtime`]) reads as
//! [`ModelState::NotLoaded`] until one of those paths sees it.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use serde::Serialize;
use uni_db::{ModelAliasSpec, ModelTask};

use crate::storage::KnowledgeBase;

/// Whether a model alias has loaded, as far as this store has seen.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "state", content = "error", rename_all = "snake_case")]
pub enum ModelState {
    /// Not prefetched and not yet used through the store.
    NotLoaded,
    /// Loaded by prefetch or by a successful call.
    Loaded,
    /// Prefetch (or an explicit load check) failed with this error.
    /// URLs in the message have their userinfo and query removed.
    Failed(String),
}

/// Shared per-alias state table. Shared (not copied) between a store and
/// the runtime it was opened over, so every store sees one load history.
pub(crate) type ModelStates = Arc<RwLock<HashMap<String, ModelState>>>;

/// A fresh state table holding every alias in `catalog` as not loaded.
pub(crate) fn new_states(catalog: &[ModelAliasSpec]) -> ModelStates {
    Arc::new(RwLock::new(
        catalog
            .iter()
            .map(|s| (s.alias.clone(), ModelState::NotLoaded))
            .collect(),
    ))
}

/// Record `state` for `alias` when the alias is in the table.
///
/// Aliases outside the catalog are ignored: the table only describes the
/// catalog. A poisoned lock is recovered, since the table holds no
/// invariant a panicking writer could break.
pub(crate) fn record(states: &ModelStates, alias: &str, state: ModelState) {
    let mut table = states.write().unwrap_or_else(|p| p.into_inner());
    if let Some(slot) = table.get_mut(alias) {
        *slot = state;
    }
}

/// Mark `alias` loaded, taking the write lock only on the first success.
pub(crate) fn mark_loaded(states: &ModelStates, alias: &str) {
    let already = states
        .read()
        .unwrap_or_else(|p| p.into_inner())
        .get(alias)
        .is_none_or(|s| *s == ModelState::Loaded);
    if !already {
        record(states, alias, ModelState::Loaded);
    }
}

/// Load every alias in `catalog` through `prefetch`, one at a time,
/// recording each result in `states`. A failure is logged and recorded,
/// and the remaining aliases are still tried.
pub(crate) async fn prefetch_each<F, Fut, E>(
    catalog: &[ModelAliasSpec],
    states: &ModelStates,
    prefetch: F,
) where
    F: Fn(String) -> Fut,
    Fut: std::future::Future<Output = Result<(), E>>,
    E: std::fmt::Display,
{
    let started = std::time::Instant::now();
    let mut failed = 0usize;
    for spec in catalog {
        let alias_started = std::time::Instant::now();
        match prefetch(spec.alias.clone()).await {
            Ok(()) => record(states, &spec.alias, ModelState::Loaded),
            Err(e) => {
                failed += 1;
                let error = redact_urls(&e.to_string());
                tracing::warn!(
                    alias = %spec.alias,
                    provider = %spec.provider_id,
                    error = %error,
                    elapsed_ms = alias_started.elapsed().as_millis() as u64,
                    "model prefetch failed; it may still load on first use"
                );
                record(states, &spec.alias, ModelState::Failed(error));
            }
        }
    }
    tracing::info!(
        aliases = catalog.len(),
        failed,
        elapsed_ms = started.elapsed().as_millis() as u64,
        "model prefetch complete"
    );
}

/// One catalog alias with its load state, safe to log or show.
///
/// Provider `options` are never included. Credentials appear only as the
/// name of the environment variable that holds them, and `base_url` has
/// any userinfo and query string removed.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelStatus {
    /// The alias callers pass to [`KnowledgeBase::embed`] and friends.
    pub alias: String,
    /// What the alias serves.
    pub task: ModelTask,
    /// The uni-xervo provider id, e.g. `local/onnx` or `remote/openai`.
    pub provider_id: String,
    /// The provider's model identifier.
    pub model_id: String,
    /// Pinned model revision, if the catalog names one.
    pub revision: Option<String>,
    /// Whether the model has loaded, as far as the store has seen.
    /// Serialized flat, as `"state"` plus `"error"` when it failed.
    #[serde(flatten)]
    pub state: ModelState,
    /// Name of the environment variable holding the API key, if any.
    pub api_key_env: Option<String>,
    /// The provider endpoint, without userinfo or query.
    pub base_url: Option<String>,
    /// ONNX execution providers the catalog asked for, in priority order.
    pub requested_execution_providers: Option<Vec<String>>,
    /// Execution providers the loaded session reports, when known.
    ///
    /// Only local ONNX embedding and reranking models report these, and
    /// only once loaded. uni-xervo reports the requested providers minus
    /// any not compiled into this build, not the ones that attached, so
    /// `["cuda", "cpu"]` here does not prove the GPU is in use. `None`
    /// means unknown.
    pub active_execution_providers: Option<Vec<String>>,
}

impl KnowledgeBase {
    /// Every catalog alias with its load state and redacted settings.
    ///
    /// Does not load anything: a model that has not loaded reads as
    /// [`ModelState::NotLoaded`]. For loaded local ONNX embedding and
    /// reranking models, the active execution providers are read from the
    /// already-cached handle.
    ///
    /// Returns `None` when the store does not know its catalog (one opened
    /// with [`open_with_runtime`](Self::open_with_runtime)).
    pub async fn model_status(&self) -> Option<Vec<ModelStatus>> {
        let catalog = self.catalog.as_deref()?;
        let states: HashMap<String, ModelState> = self
            .model_states
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .clone();
        let runtime = self.model_runtime();
        let mut out = Vec::with_capacity(catalog.len());
        for spec in catalog {
            let state = states
                .get(&spec.alias)
                .cloned()
                .unwrap_or(ModelState::NotLoaded);
            let active = match (&state, &runtime) {
                (ModelState::Loaded, Some(rt)) => active_eps(rt, spec).await,
                _ => None,
            };
            out.push(ModelStatus {
                alias: spec.alias.clone(),
                task: spec.task,
                provider_id: spec.provider_id.clone(),
                model_id: spec.model_id.clone(),
                revision: spec.revision.clone(),
                state,
                // Vertex AI names its credential variable `api_token_env`.
                api_key_env: string_option(spec, "api_key_env")
                    .or_else(|| string_option(spec, "api_token_env")),
                base_url: string_option(spec, "base_url").map(|u| redact_url(&u)),
                requested_execution_providers: spec
                    .options
                    .get("execution_providers")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(str::to_string))
                            .collect()
                    }),
                active_execution_providers: active,
            });
        }
        Some(out)
    }

    pub(crate) fn mark_model_loaded(&self, alias: &str) {
        mark_loaded(&self.model_states, alias);
    }

    pub(crate) fn mark_model_failed(&self, alias: &str, error: &str) {
        record(
            &self.model_states,
            alias,
            ModelState::Failed(redact_urls(error)),
        );
    }
}

/// Active EPs off the cached handle of a loaded embed or rerank alias.
async fn active_eps(rt: &crate::ModelRuntime, spec: &ModelAliasSpec) -> Option<Vec<String>> {
    let eps = match spec.task {
        ModelTask::Embed => rt
            .embedding(&spec.alias)
            .await
            .ok()?
            .active_execution_providers(),
        ModelTask::Rerank => rt
            .reranker(&spec.alias)
            .await
            .ok()?
            .active_execution_providers(),
        _ => return None,
    };
    (!eps.is_empty()).then_some(eps)
}

fn string_option(spec: &ModelAliasSpec, key: &str) -> Option<String> {
    spec.options
        .get(key)
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

/// `url` without userinfo, query or fragment.
pub(crate) fn redact_url(url: &str) -> String {
    let cut = url.find(['?', '#']).unwrap_or(url.len());
    let url = &url[..cut];
    let Some(scheme_end) = url.find("://") else {
        return url.to_string();
    };
    let rest_start = scheme_end + 3;
    let rest = &url[rest_start..];
    let authority_end = rest.find('/').unwrap_or(rest.len());
    match rest[..authority_end].rfind('@') {
        Some(at) => format!("{}{}", &url[..rest_start], &rest[at + 1..]),
        None => url.to_string(),
    }
}

/// `text` with every `http(s)://` URL (any case) passed through
/// [`redact_url`].
pub(crate) fn redact_urls(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    // ASCII lowercasing keeps byte offsets, so positions carry over.
    while let Some(pos) = rest.to_ascii_lowercase().find("http") {
        let (before, candidate) = rest.split_at(pos);
        out.push_str(before);
        let lower = candidate.get(..8).unwrap_or(candidate).to_ascii_lowercase();
        if !(lower.starts_with("http://") || lower.starts_with("https://")) {
            out.push_str(&candidate[..4]);
            rest = &candidate[4..];
            continue;
        }
        let end = candidate
            .find(|c: char| c.is_whitespace() || matches!(c, '"' | '\'' | '<' | '>' | ')' | ','))
            .unwrap_or(candidate.len());
        out.push_str(&redact_url(&candidate[..end]));
        rest = &candidate[end..];
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redact_url_strips_userinfo_query_and_fragment() {
        assert_eq!(
            redact_url("https://user:pw@api.example.com/v1?key=abc#x"),
            "https://api.example.com/v1"
        );
        assert_eq!(
            redact_url("http://localhost:8080/v1"),
            "http://localhost:8080/v1"
        );
        assert_eq!(redact_url("not a url?q=1"), "not a url");
    }

    #[test]
    fn redact_urls_rewrites_urls_inside_text() {
        let text = "error sending request for url (https://u:secret@h.io/x?token=t): refused";
        let out = redact_urls(text);
        assert!(!out.contains("secret") && !out.contains("token=t"), "{out}");
        assert!(out.contains("https://h.io/x"), "{out}");
        assert_eq!(redact_urls("httpx is fine"), "httpx is fine");
        let shouty = redact_urls("failed: HTTPS://u:secret@h.io/x");
        assert!(!shouty.contains("secret"), "{shouty}");
    }

    #[test]
    fn mark_loaded_ignores_aliases_outside_the_catalog() {
        let states: ModelStates = Arc::new(RwLock::new(HashMap::from([(
            "embed/a".to_string(),
            ModelState::NotLoaded,
        )])));
        mark_loaded(&states, "embed/a");
        mark_loaded(&states, "embed/other");
        let table = states.read().unwrap();
        assert_eq!(table.get("embed/a"), Some(&ModelState::Loaded));
        assert!(!table.contains_key("embed/other"));
    }
}
