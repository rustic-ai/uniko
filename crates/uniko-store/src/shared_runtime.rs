//! One model runtime shared by several stores, and when to prefetch it.

use std::fmt;
use std::sync::Arc;

use uni_db::ModelAliasSpec;

use crate::ModelRuntime;
use crate::config::UnikoConfig;
use crate::error::{Result, UnikoError};
use crate::model::ModelEntry;
use crate::model_state::{self, ModelStates};

/// When a store (or a [`SharedRuntime`]) loads its models.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PrefetchPolicy {
    /// Load every catalog model before returning, so the first call is
    /// warm. A model that fails is recorded and skipped. The default.
    #[default]
    Eager,
    /// Load nothing up front; each model loads on first use. A catalog-file
    /// entry with `warmup: eager` is still loaded by uni-xervo while the
    /// runtime is built, and reads as not loaded until the store uses it.
    Lazy,
    /// Return at once and load every model on a task spawned on the
    /// current tokio runtime, which must outlive the load (a runtime
    /// dropped first cancels it, leaving the models to load on first use).
    Background,
}

impl PrefetchPolicy {
    /// Run this policy over `catalog`, loading through `runtime`.
    pub(crate) async fn run(
        self,
        catalog: Arc<[ModelAliasSpec]>,
        states: ModelStates,
        runtime: Arc<ModelRuntime>,
    ) {
        match self {
            Self::Lazy => {}
            Self::Eager => prefetch_through(&catalog, &states, runtime).await,
            Self::Background => {
                tokio::spawn(async move {
                    prefetch_through(&catalog, &states, runtime).await;
                });
            }
        }
    }
}

async fn prefetch_through(
    catalog: &[ModelAliasSpec],
    states: &ModelStates,
    runtime: Arc<ModelRuntime>,
) {
    model_state::prefetch_each(catalog, states, |alias| {
        let runtime = runtime.clone();
        async move { runtime.prefetch(&[alias.as_str()]).await }
    })
    .await;
}

/// A model runtime several stores can share, so each model loads once.
///
/// Build one with [`SharedRuntime::build`] and open stores over it with
/// [`KnowledgeBase::in_memory_shared`](crate::KnowledgeBase::in_memory_shared)
/// or [`KnowledgeBase::open_shared`](crate::KnowledgeBase::open_shared).
/// Each store keeps its own graph; only inference is shared.
///
/// Cloning is cheap and yields another handle to the same runtime. The
/// runtime lives as long as any handle or store holding it: shutting a
/// store down never unloads it, so other stores keep working.
///
/// The runtime remembers its catalog, so every store over it reports
/// [`model_status`](crate::KnowledgeBase::model_status), and the load
/// state is shared: a model loaded through one store reads as loaded in
/// all of them.
#[derive(Clone)]
pub struct SharedRuntime {
    pub(crate) runtime: Arc<ModelRuntime>,
    pub(crate) catalog: Arc<[ModelAliasSpec]>,
    pub(crate) states: ModelStates,
}

impl fmt::Debug for SharedRuntime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Specs carry provider options, which may hold endpoints or keys.
        let aliases: Vec<&str> = self.catalog.iter().map(|s| s.alias.as_str()).collect();
        f.debug_struct("SharedRuntime")
            .field("aliases", &aliases)
            .finish_non_exhaustive()
    }
}

impl SharedRuntime {
    /// Build a runtime for `config`'s models plus `extra_catalog` (for
    /// example generation models), loading them as `prefetch` says.
    ///
    /// Stores opened over it must use a config whose models it holds;
    /// see [`KnowledgeBase::open_shared`](crate::KnowledgeBase::open_shared).
    ///
    /// # Errors
    ///
    /// [`UnikoError::Config`] when the config is invalid, an alias is
    /// defined twice, or a provider is not compiled in or rejects its
    /// options. A model that fails to *load* is not an error; it is
    /// recorded as [`ModelState::Failed`](crate::ModelState::Failed).
    pub async fn build(
        config: &UnikoConfig,
        extra_catalog: Vec<ModelAliasSpec>,
        prefetch: PrefetchPolicy,
    ) -> Result<Self> {
        config.validate()?;
        let catalog = crate::storage::load_catalog(config, &extra_catalog)?;
        let runtime = uni_db::xervo::build_model_runtime(catalog.clone())
            .await
            .map_err(|e| UnikoError::Config(format!("model catalog is invalid: {e}")))?;
        let catalog: Arc<[ModelAliasSpec]> = catalog.into();
        let states = model_state::new_states(&catalog);
        prefetch
            .run(catalog.clone(), states.clone(), runtime.clone())
            .await;
        Ok(Self {
            runtime,
            catalog,
            states,
        })
    }

    /// The aliases this runtime serves, in catalog order.
    #[must_use]
    pub fn model_entries(&self) -> Vec<ModelEntry> {
        self.catalog.iter().map(ModelEntry::from).collect()
    }

    /// The underlying uni-xervo runtime.
    #[must_use]
    pub fn model_runtime(&self) -> Arc<ModelRuntime> {
        self.runtime.clone()
    }

    /// Check that this runtime holds `spec`'s alias with the same task,
    /// provider, model, revision and options (execution providers aside).
    ///
    /// # Errors
    ///
    /// [`UnikoError::Config`] naming the alias and the first field that
    /// differs. Option values are never printed.
    pub fn check_spec(&self, spec: &ModelAliasSpec) -> Result<()> {
        let Some(held) = self.catalog.iter().find(|s| s.alias == spec.alias) else {
            return Err(UnikoError::Config(format!(
                "model alias '{}' is not in the shared runtime",
                spec.alias
            )));
        };
        match first_difference(spec, held) {
            Some(field) => Err(UnikoError::Config(format!(
                "model alias '{}' differs from the shared runtime's ({field})",
                spec.alias
            ))),
            None => Ok(()),
        }
    }

    /// Check that this runtime serves every model `config` needs, with
    /// the same provider, model, revision and options.
    ///
    /// Execution providers are left out of the comparison: they pick a
    /// device, not a model. Extra aliases in the runtime are fine.
    pub(crate) fn check_serves(&self, config: &UnikoConfig) -> Result<()> {
        let needed = crate::storage::load_catalog(config, &[])?;
        for spec in &needed {
            let Some(held) = self.catalog.iter().find(|s| s.alias == spec.alias) else {
                return Err(UnikoError::Config(format!(
                    "the store config needs model alias '{}', which the shared runtime \
                     does not have; build the runtime from the same config",
                    spec.alias
                )));
            };
            if let Some(field) = first_difference(spec, held) {
                return Err(UnikoError::Config(format!(
                    "model alias '{}' differs between the store config and the shared \
                     runtime ({field}); build the runtime from the same config",
                    spec.alias
                )));
            }
        }
        Ok(())
    }
}

/// The first field that differs between two specs for one alias, as a
/// message safe to show (option values are never printed).
fn first_difference(want: &ModelAliasSpec, held: &ModelAliasSpec) -> Option<String> {
    if want.task != held.task {
        return Some(format!("task {:?} vs {:?}", want.task, held.task));
    }
    if want.provider_id != held.provider_id {
        return Some(format!(
            "provider '{}' vs '{}'",
            want.provider_id, held.provider_id
        ));
    }
    if want.model_id != held.model_id {
        return Some(format!("model '{}' vs '{}'", want.model_id, held.model_id));
    }
    if want.revision != held.revision {
        return Some(format!(
            "revision {:?} vs {:?}",
            want.revision, held.revision
        ));
    }
    if without_eps(&want.options) != without_eps(&held.options) {
        return Some("provider options".to_string());
    }
    None
}

fn without_eps(options: &serde_json::Value) -> serde_json::Value {
    let mut options = options.clone();
    if let Some(map) = options.as_object_mut() {
        map.remove("execution_providers");
        if map.is_empty() {
            return serde_json::Value::Null;
        }
    }
    options
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_providers_do_not_count_as_a_difference() {
        let config = UnikoConfig::default();
        let spec = crate::storage::embed_catalog(&config)
            .into_iter()
            .find(|s| s.alias == "embed/default")
            .unwrap();
        let mut moved = spec.clone();
        moved.options = serde_json::json!({"execution_providers": ["cuda", "cpu"]});
        let mut base = spec.clone();
        base.options = serde_json::Value::Null;
        assert_eq!(first_difference(&base, &moved), None);

        let mut other = spec.clone();
        other.model_id = "something-else".into();
        assert!(
            first_difference(&spec, &other)
                .unwrap()
                .starts_with("model")
        );
    }
}
