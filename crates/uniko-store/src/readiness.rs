//! Readiness failures and loading a chosen set of aliases.
//!
//! The facade decides *what* an application requires (an alias, the
//! embedder, NLP, OCR, ...); this module holds the failure type the
//! requirement check reports and the store operation that loads just the
//! required aliases, whatever the store's [`PrefetchPolicy`](crate::PrefetchPolicy).

use std::time::Duration;

use serde::Serialize;

use crate::model_state::{ModelState, prefetch_one, redact_urls};
use crate::storage::KnowledgeBase;

/// Why a requirement is not met. A stable, snake_case identifier.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadinessCode {
    /// The alias is not in the store's model catalog.
    NotInCatalog,
    /// The stage is turned off in the config (e.g. the reranker, or OCR).
    NotConfigured,
    /// The stage needs a Cargo feature this build does not have.
    NotCompiled,
    /// No extractor is registered for the modality.
    NoExtractor,
    /// The alias failed to load.
    LoadFailed,
    /// The alias had not loaded when the readiness timeout ran out.
    TimedOut,
    /// The NLP models loaded but the NLP pipeline could not start.
    NlpUnavailable,
    /// The alias loaded, but none of the accelerators it asked for is
    /// among its active execution providers.
    NotAccelerated,
    /// The alias loaded, but its remote endpoint could not be reached or
    /// rejected its credential. Checked only when asked for, with
    /// `verify_remote`.
    VerificationFailed,
    /// The check cannot be made for this model: only local ONNX models
    /// (other than document extraction) report execution providers.
    NotSupported,
}

impl ReadinessCode {
    /// The snake_case name, as serialized.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotInCatalog => "not_in_catalog",
            Self::NotConfigured => "not_configured",
            Self::NotCompiled => "not_compiled",
            Self::NoExtractor => "no_extractor",
            Self::LoadFailed => "load_failed",
            Self::TimedOut => "timed_out",
            Self::NlpUnavailable => "nlp_unavailable",
            Self::NotAccelerated => "not_accelerated",
            Self::VerificationFailed => "verification_failed",
            Self::NotSupported => "not_supported",
        }
    }
}

impl std::fmt::Display for ReadinessCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One unmet requirement, safe to log or show.
///
/// `error` is the load error with URL userinfo and query removed, the same
/// text [`ModelState::Failed`] holds; it is set for
/// [`ReadinessCode::LoadFailed`] and, with the failed check's reason, for
/// [`ReadinessCode::VerificationFailed`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReadinessFailure {
    /// The requirement as written, e.g. `embedder` or `alias:gen/m`.
    pub requirement: String,
    /// Why it is not met.
    pub code: ReadinessCode,
    /// The model alias involved, when there is one.
    pub alias: Option<String>,
    /// The redacted load error, for [`ReadinessCode::LoadFailed`], or the
    /// reason for [`ReadinessCode::VerificationFailed`].
    pub error: Option<String>,
}

impl ReadinessFailure {
    /// A failure with no alias or error.
    #[must_use]
    pub fn new(requirement: impl Into<String>, code: ReadinessCode) -> Self {
        Self {
            requirement: requirement.into(),
            code,
            alias: None,
            error: None,
        }
    }

    /// Name the alias involved.
    #[must_use]
    pub fn with_alias(mut self, alias: impl Into<String>) -> Self {
        self.alias = Some(alias.into());
        self
    }

    /// Attach the (already redacted) load error.
    #[must_use]
    pub fn with_error(mut self, error: impl Into<String>) -> Self {
        self.error = Some(error.into());
        self
    }
}

impl std::fmt::Display for ReadinessFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.requirement, self.code)?;
        if let Some(alias) = &self.alias {
            write!(f, " (alias '{alias}')")?;
        }
        if let Some(error) = &self.error {
            write!(f, ": {error}")?;
        }
        Ok(())
    }
}

/// Render failures one per line, for [`UnikoError::NotReady`](crate::UnikoError::NotReady).
pub(crate) fn describe(failures: &[ReadinessFailure]) -> String {
    failures
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("; ")
}

impl KnowledgeBase {
    /// Load `aliases` now. The model runtime records each result, which
    /// [`model_status`](Self::model_status) reads.
    ///
    /// Aliases already loaded, and aliases outside the catalog, are
    /// skipped. (For a store that does not know its catalog, one opened
    /// with [`open_with_runtime`](Self::open_with_runtime), that is the
    /// runtime's aliases.) An alias that failed before is tried again. A
    /// failure is logged and the rest are still tried, as in prefetch.
    ///
    /// With `timeout`, loading stops when it runs out. Returns the aliases
    /// whose load did not finish in time, in catalog order; their state
    /// goes back to what it was (a model still loading may load on first
    /// use). Empty when every load finished, successfully or not.
    ///
    /// Works the same over a [`SharedRuntime`](crate::SharedRuntime): the
    /// load is shared with every store over that runtime.
    pub async fn prefetch_aliases(
        &self,
        aliases: &[&str],
        timeout: Option<Duration>,
    ) -> Vec<String> {
        let Some(runtime) = self.model_runtime() else {
            return Vec::new();
        };
        let candidates: Vec<String> = match self.catalog.as_deref() {
            Some(catalog) => catalog
                .iter()
                .filter(|spec| aliases.contains(&spec.alias.as_str()))
                .map(|spec| spec.alias.clone())
                .collect(),
            None => aliases.iter().map(|a| (*a).to_string()).collect(),
        };
        let mut pending = Vec::new();
        for alias in candidates {
            // `None`: not in the runtime, so there is nothing to load.
            match self.alias_state(&alias).await {
                None | Some(ModelState::Loaded) => {}
                Some(_) => pending.push(alias),
            }
        }
        if pending.is_empty() {
            return Vec::new();
        }
        // Each finished load is noted, so a timeout can name the aliases it
        // cut off.
        let finished = std::sync::Mutex::new(Vec::<String>::new());
        let load = async {
            for alias in pending.iter().cloned() {
                if let Err(e) = prefetch_one(&runtime, alias.clone()).await {
                    tracing::warn!(
                        alias = %alias,
                        error = %redact_urls(&e.to_string()),
                        "model load failed"
                    );
                }
                finished
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .push(alias);
            }
        };
        if let Some(limit) = timeout {
            let _ = tokio::time::timeout(limit, load).await;
        } else {
            load.await;
        }
        let finished = finished.into_inner().unwrap_or_else(|p| p.into_inner());
        pending
            .into_iter()
            .filter(|alias| !finished.contains(alias))
            .collect()
    }

    /// Check that each of `aliases` can reach its endpoint with a valid
    /// credential: one cheap authenticated request per remote alias, at
    /// most eight at a time. Loads no model. A local alias passes without
    /// any request.
    ///
    /// Returns each alias that failed, in the order given, with a redacted
    /// reason such as `unauthorized: HTTP 401`. Each check waits at most
    /// the alias's `load_timeout`, capped at 30 seconds. The result is also
    /// kept by the model runtime. Empty when the store has no runtime.
    ///
    /// With `timeout`, the checks are cancelled when it runs out and `None`
    /// is returned.
    ///
    /// Nothing else contacts a remote provider before its first request:
    /// opening a store and prefetching stay offline.
    pub async fn verify_aliases(
        &self,
        aliases: &[&str],
        timeout: Option<Duration>,
    ) -> Option<Vec<(String, String)>> {
        let Some(runtime) = self.model_runtime() else {
            return Some(Vec::new());
        };
        // `ModelRuntime::verify`'s future is not `Send` for every lifetime
        // (its stream closure borrows `&&str`, rustic-ai/uni-xervo#73), so a
        // caller that must be `Send`, like a spawned task, cannot hold it.
        // Run it to completion on a blocking-pool thread instead, which
        // needs no `Send` future.
        // The timeout is applied on that thread too, so a check cut off by
        // it is cancelled rather than left running.
        let owned: Vec<String> = aliases.iter().map(|a| (*a).to_string()).collect();
        let handle = tokio::runtime::Handle::current();
        let results = tokio::task::spawn_blocking(move || {
            let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
            let checks = runtime.verify(&refs);
            match timeout {
                None => Some(handle.block_on(checks)),
                Some(limit) => handle.block_on(tokio::time::timeout(limit, checks)).ok(),
            }
        })
        .await;
        let results = match results {
            Ok(Some(results)) => results,
            Ok(None) => return None,
            Err(e) => {
                return Some(
                    aliases
                        .iter()
                        .map(|a| ((*a).to_string(), format!("failed: {e}")))
                        .collect(),
                );
            }
        };
        let failed = results
            .into_iter()
            .filter_map(|(alias, result)| {
                let error = match result.err()? {
                    uni_xervo::error::RuntimeError::Verification { kind, detail, .. } => {
                        format!("{kind}: {detail}")
                    }
                    other => other.to_string(),
                };
                Some((alias, redact_urls(&error)))
            })
            .collect();
        Some(failed)
    }

    /// Start prefetching the catalog under `policy`, as the open
    /// constructors do. For a store opened [`Lazy`](crate::PrefetchPolicy::Lazy)
    /// so that something (such as a readiness check) could run first.
    /// A background prefetch replaces, and cancels, any earlier one.
    pub async fn start_prefetch(&mut self, policy: crate::PrefetchPolicy) {
        if let Some(task) = self.prefetch_task.take() {
            task.abort();
        }
        self.prefetch_task = self.prefetch_catalog(policy).await;
    }
}
