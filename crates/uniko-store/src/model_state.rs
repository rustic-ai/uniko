//! Per-alias model load state and the status report built on it.
//!
//! The state comes from the model runtime's own snapshot
//! ([`ModelRuntime::aliases`](crate::ModelRuntime::aliases)), so every load
//! counts, whichever path made it: prefetch, a call through the store,
//! uni-db auto-embedding on write, or NLP and OCR through
//! [`KnowledgeBase::model_runtime`].

use std::sync::Arc;

use serde::Serialize;
use uni_db::{ModelAliasSpec, ModelTask};
use uni_xervo::inspect::{AliasSnapshot, LoadState};

use crate::storage::KnowledgeBase;

/// Whether a model alias has loaded, as the model runtime reports it.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "state", content = "error", rename_all = "snake_case")]
pub enum ModelState {
    /// No load has been attempted yet.
    NotLoaded,
    /// A load is in progress.
    Loading,
    /// Loaded and cached.
    Loaded,
    /// The last load failed (or timed out) with this error, and no later
    /// load has succeeded. URLs in the message have their userinfo and
    /// query removed.
    Failed(String),
}

impl From<&LoadState> for ModelState {
    fn from(state: &LoadState) -> Self {
        match state {
            LoadState::Loaded => Self::Loaded,
            LoadState::Loading => Self::Loading,
            LoadState::Failed { error, .. } => Self::Failed(redact_urls(error)),
            // `Registered`, and any state a later uni-xervo adds.
            _ => Self::NotLoaded,
        }
    }
}

/// Load every alias in `catalog` through `runtime`, one at a time. A
/// failure is logged (the runtime records it) and the remaining aliases
/// are still tried.
pub(crate) async fn prefetch_each(
    catalog: Arc<[ModelAliasSpec]>,
    runtime: Arc<crate::ModelRuntime>,
) {
    let started = std::time::Instant::now();
    let mut failed = 0usize;
    for spec in catalog.iter() {
        let alias_started = std::time::Instant::now();
        if let Err(e) = prefetch_one(&runtime, spec.alias.clone()).await {
            failed += 1;
            tracing::warn!(
                alias = %spec.alias,
                provider = %spec.provider_id,
                error = %redact_urls(&e.to_string()),
                elapsed_ms = alias_started.elapsed().as_millis() as u64,
                "model prefetch failed; it may still load on first use"
            );
        }
    }
    tracing::info!(
        aliases = catalog.len(),
        failed,
        elapsed_ms = started.elapsed().as_millis() as u64,
        "model prefetch complete"
    );
}

/// Load one alias. Takes the alias by value so the future owns everything
/// it borrows across the await, which keeps it `Send` for any lifetime.
pub(crate) async fn prefetch_one(
    runtime: &crate::ModelRuntime,
    alias: String,
) -> Result<(), uni_xervo::error::RuntimeError> {
    let aliases = [alias.as_str()];
    runtime.prefetch(&aliases).await
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
    /// Whether the model has loaded, as the model runtime reports it.
    /// Serialized flat, as `"state"` plus `"error"` when it failed.
    #[serde(flatten)]
    pub state: ModelState,
    /// Name of the environment variable holding the API key, if any.
    pub api_key_env: Option<String>,
    /// Where a remote alias sends requests (its `base_url`, or the
    /// provider's default endpoint), without userinfo or query.
    pub base_url: Option<String>,
    /// ONNX execution providers the catalog asked for, in priority order.
    pub requested_execution_providers: Option<Vec<String>>,
    /// Execution providers the loaded model's sessions attached, when known.
    ///
    /// Every local ONNX model reports these once loaded: a provider that
    /// was asked for but failed to register, or whose output disagreed with
    /// CPU, is left out. Remote models report none. `None` means unknown.
    pub active_execution_providers: Option<Vec<String>>,
}

impl KnowledgeBase {
    /// Every catalog alias with its load state and redacted settings.
    ///
    /// Does not load anything: it reads the model runtime's snapshot, so a
    /// model that has not loaded reads as [`ModelState::NotLoaded`]. For a
    /// store opened with [`open_with_runtime`](Self::open_with_runtime),
    /// which does not know its catalog, every alias of the runtime is
    /// listed.
    ///
    /// Returns `None` when the store has no model runtime.
    pub async fn model_status(&self) -> Option<Vec<ModelStatus>> {
        let runtime = self.model_runtime()?;
        let snapshots = runtime.aliases().await;
        let Some(catalog) = self.catalog.as_deref() else {
            return Some(snapshots.iter().map(|snap| status(snap, None)).collect());
        };
        Some(
            catalog
                .iter()
                .map(
                    |spec| match snapshots.iter().find(|s| s.alias == spec.alias) {
                        Some(snap) => status(snap, Some(spec)),
                        None => not_in_runtime(spec),
                    },
                )
                .collect(),
        )
    }

    /// Kept for callers written against uniko 0.5: the model runtime now
    /// records every load itself, so this does nothing.
    #[deprecated(note = "the model runtime records loads itself; this is a no-op")]
    pub fn note_model_loaded(&self, _alias: &str) {}

    /// The runtime's current state of `alias`, or `None` when the store has
    /// no runtime or the alias is not in it.
    pub(crate) async fn alias_state(&self, alias: &str) -> Option<ModelState> {
        let snapshot = self.model_runtime()?.alias(alias).await?;
        Some(ModelState::from(&snapshot.state))
    }
}

/// A status row from the runtime's snapshot, with the catalog spec (when
/// the store knows it) filling in what the snapshot leaves out.
fn status(snap: &AliasSnapshot, spec: Option<&ModelAliasSpec>) -> ModelStatus {
    let state = ModelState::from(&snap.state);
    let active = match state {
        ModelState::Loaded => snap.execution_providers.clone(),
        _ => None,
    };
    ModelStatus {
        alias: snap.alias.clone(),
        task: snap.task,
        provider_id: snap.provider_id.clone(),
        model_id: snap.model_id.clone(),
        revision: snap.revision.clone(),
        state,
        api_key_env: snap
            .credential_env
            .clone()
            .or_else(|| spec.and_then(credential_env)),
        base_url: snap.endpoint.as_deref().map(redact_url).or_else(|| {
            spec.and_then(|s| string_option(s, "base_url"))
                .map(|u| redact_url(&u))
        }),
        requested_execution_providers: spec.and_then(requested_eps),
        active_execution_providers: active.filter(|eps| !eps.is_empty()),
    }
}

/// A catalog alias the runtime does not hold (it should always hold one).
fn not_in_runtime(spec: &ModelAliasSpec) -> ModelStatus {
    ModelStatus {
        alias: spec.alias.clone(),
        task: spec.task,
        provider_id: spec.provider_id.clone(),
        model_id: spec.model_id.clone(),
        revision: spec.revision.clone(),
        state: ModelState::NotLoaded,
        api_key_env: credential_env(spec),
        base_url: string_option(spec, "base_url").map(|u| redact_url(&u)),
        requested_execution_providers: requested_eps(spec),
        active_execution_providers: None,
    }
}

/// Vertex AI names its credential variable `api_token_env`.
fn credential_env(spec: &ModelAliasSpec) -> Option<String> {
    string_option(spec, "api_key_env").or_else(|| string_option(spec, "api_token_env"))
}

fn requested_eps(spec: &ModelAliasSpec) -> Option<Vec<String>> {
    spec.options
        .get("execution_providers")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
}

fn string_option(spec: &ModelAliasSpec, key: &str) -> Option<String> {
    spec.options
        .get(key)
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

/// `url` without userinfo, query or fragment.
///
/// Userinfo ends at the *last* `@`, so a password holding an unescaped
/// `@`, `/`, `?` or `#` is still removed whole. A legitimate `@` later in
/// the URL costs the host, never a secret.
pub(crate) fn redact_url(url: &str) -> String {
    let Some(scheme_end) = url.find("://") else {
        let cut = url.find(['?', '#']).unwrap_or(url.len());
        return url[..cut].to_string();
    };
    let (scheme, rest) = url.split_at(scheme_end + 3);
    let rest = rest.rfind('@').map_or(rest, |at| &rest[at + 1..]);
    let cut = rest.find(['?', '#']).unwrap_or(rest.len());
    format!("{scheme}{}", &rest[..cut])
}

/// `text` with every `http(s)://` URL (any case) passed through
/// [`redact_url`]. A URL runs to the next whitespace or quote, so
/// punctuation inside a password cannot end it early.
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
            .find(|c: char| c.is_whitespace() || matches!(c, '"' | '\'' | '<' | '>'))
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
    fn runtime_load_states_map_to_model_states() {
        assert_eq!(
            ModelState::from(&LoadState::Registered),
            ModelState::NotLoaded
        );
        assert_eq!(ModelState::from(&LoadState::Loading), ModelState::Loading);
        assert_eq!(ModelState::from(&LoadState::Loaded), ModelState::Loaded);
        let failed = ModelState::from(&LoadState::Failed {
            error: "request to https://u:secret@h.io/v1?key=k failed".into(),
            at: std::time::SystemTime::now(),
        });
        let ModelState::Failed(error) = failed else {
            panic!("{failed:?}");
        };
        assert!(
            !error.contains("secret") && !error.contains("key=k"),
            "{error}"
        );
        assert!(error.contains("https://h.io/v1"), "{error}");
        // Serialized flat, as the capability report shows it.
        let json = serde_json::to_value(ModelState::Loading).unwrap();
        assert_eq!(json["state"], "loading");
    }

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
        for tricky in [
            "https://u:pa#ss@h.io/v1",
            "https://u:pa/ss@h.io/v1",
            "https://u:pa?ss@h.io/v1",
            "(https://u:pa)ss@h.io/x)",
        ] {
            let out = redact_urls(tricky);
            assert!(
                !out.contains("pa") && !out.contains("ss@"),
                "{tricky} -> {out}"
            );
            assert!(out.contains("h.io"), "{tricky} -> {out}");
        }
    }
}
