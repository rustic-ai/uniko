//! Readiness failures and loading a chosen set of aliases.
//!
//! The facade decides *what* an application requires (an alias, the
//! embedder, NLP, OCR, ...); this module holds the failure type the
//! requirement check reports and the store operation that loads just the
//! required aliases, whatever the store's [`PrefetchPolicy`](crate::PrefetchPolicy).

use std::time::Duration;

use serde::Serialize;

use crate::model_state::{self, ModelState};
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
    /// The check cannot be made for this model: only local ONNX embedding,
    /// hybrid-embedding and reranking models report execution providers.
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
/// text [`ModelState::Failed`] holds; it is set only for
/// [`ReadinessCode::LoadFailed`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReadinessFailure {
    /// The requirement as written, e.g. `embedder` or `alias:gen/m`.
    pub requirement: String,
    /// Why it is not met.
    pub code: ReadinessCode,
    /// The model alias involved, when there is one.
    pub alias: Option<String>,
    /// The redacted load error, for [`ReadinessCode::LoadFailed`].
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
    /// Load `aliases` now, recording each result in the load-state table
    /// that [`model_status`](Self::model_status) reads.
    ///
    /// Aliases already loaded, and aliases outside the catalog, are
    /// skipped. An alias that failed before is tried again. A failure is
    /// recorded and the rest are still tried, as in prefetch.
    ///
    /// With `timeout`, loading stops when it runs out. Returns the aliases
    /// whose load did not finish in time, in catalog order; their recorded
    /// state is unchanged (a model still loading may load on first use).
    /// Empty when every load finished, successfully or not.
    ///
    /// Works the same over a [`SharedRuntime`](crate::SharedRuntime): the
    /// load is shared with every store over that runtime.
    pub async fn prefetch_aliases(
        &self,
        aliases: &[&str],
        timeout: Option<Duration>,
    ) -> Vec<String> {
        let (Some(catalog), Some(runtime)) = (self.catalog.as_deref(), self.model_runtime()) else {
            return Vec::new();
        };
        let pending: Vec<_> = {
            let states = self.model_states.read().unwrap_or_else(|p| p.into_inner());
            catalog
                .iter()
                .filter(|spec| {
                    aliases.contains(&spec.alias.as_str())
                        && states.get(&spec.alias) != Some(&ModelState::Loaded)
                })
                .cloned()
                .collect()
        };
        if pending.is_empty() {
            return Vec::new();
        }
        // Each finished load is noted, so a timeout can name the aliases it
        // cut off rather than leaving an older `Failed` to stand for them.
        let finished = std::sync::Mutex::new(Vec::<String>::new());
        let load = model_state::prefetch_each(&pending, &self.model_states, |alias| {
            let runtime = runtime.clone();
            let finished = &finished;
            async move {
                let result = runtime.prefetch(&[alias.as_str()]).await;
                finished
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .push(alias);
                result
            }
        });
        if let Some(limit) = timeout {
            let _ = tokio::time::timeout(limit, load).await;
        } else {
            load.await;
        }
        let finished = finished.into_inner().unwrap_or_else(|p| p.into_inner());
        pending
            .into_iter()
            .map(|spec| spec.alias)
            .filter(|alias| !finished.contains(alias))
            .collect()
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
