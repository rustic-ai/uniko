//! Startup readiness: the models and stages an application requires.
//!
//! [`UnikoBuilder::require`](super::UnikoBuilder::require) names them, and
//! [`build`](super::UnikoBuilder::build) fails with
//! [`UnikoError::NotReady`] when any is unmet.
//! [`Uniko::check_readiness`] runs the same check at any time.
//!
//! The check runs in two steps. The static step looks only at the catalog,
//! the compiled features and the registered extractors, so it contacts no
//! provider and runs before the store opens. The dynamic step loads the
//! required aliases (whatever the [`PrefetchPolicy`](crate::PrefetchPolicy))
//! and checks how they loaded. Aliases that are not required are left to
//! the prefetch policy, as before.

use std::time::Duration;

use uniko_extract::ingest::ModalityRegistry;
use uniko_pipes::content::Modality;
use uniko_store::config::UnikoConfig;
use uniko_store::schema::{EMBED_ALIAS, HYBRID_EMBED_ALIAS, NLP_ALIAS, OCR_ALIAS, RERANK_ALIAS};
use uniko_store::xervo::ModelTask;
use uniko_store::{
    KnowledgeBase, ModelEntry, ModelState, ReadinessCode, ReadinessFailure, UnikoError,
};

use super::Uniko;

/// Something an application needs before it can serve.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Requirement {
    /// A model alias from the catalog must load.
    Alias(String),
    /// The embedding model must load (and the hybrid embedder, when the
    /// config adds one).
    Embedder,
    /// The configured reranker must load: the cross-encoder, or for the
    /// `colbert` style the hybrid embedder that provides its vectors.
    Reranker,
    /// The NLP models must load and the NLP pipeline must start.
    Nlp,
    /// OCR must be compiled in (`pdf-ocr`), enabled in the config, and its
    /// model must load.
    Ocr,
    /// The answer alias must be set and must load.
    Generator,
    /// An extractor must be registered for this modality.
    Extractor(Modality),
    /// The alias must load with one of the non-CPU execution providers it
    /// asks for among the providers its sessions attached.
    ///
    /// The attached providers are what uni-xervo reports after loading. A
    /// provider that is compiled in but fails to register (its library or
    /// driver missing), or whose output disagrees with CPU at load, is left
    /// out, so a GPU build on a host without a usable GPU fails this. Every
    /// `local/onnx` model reports its providers except document
    /// extraction; any other alias (a remote model, whisper.cpp) fails as
    /// `not_supported`.
    Accelerated(String),
}

impl Requirement {
    /// Shorthand for [`Requirement::Alias`].
    #[must_use]
    pub fn alias(alias: impl Into<String>) -> Self {
        Self::Alias(alias.into())
    }

    /// Shorthand for [`Requirement::Accelerated`].
    #[must_use]
    pub fn accelerated(alias: impl Into<String>) -> Self {
        Self::Accelerated(alias.into())
    }

    /// The requirement as reported in a [`ReadinessFailure`].
    #[must_use]
    pub fn name(&self) -> String {
        match self {
            Self::Alias(a) => format!("alias:{a}"),
            Self::Embedder => "embedder".to_string(),
            Self::Reranker => "reranker".to_string(),
            Self::Nlp => "nlp".to_string(),
            Self::Ocr => "ocr".to_string(),
            Self::Generator => "generator".to_string(),
            Self::Extractor(m) => format!("extractor:{}", modality_name(*m)),
            Self::Accelerated(a) => format!("accelerated:{a}"),
        }
    }
}

impl std::fmt::Display for Requirement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name())
    }
}

fn modality_name(m: Modality) -> String {
    serde_json::to_value(m)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_else(|| format!("{m:?}").to_lowercase())
}

/// Whether `alias`'s model reports its execution providers.
fn reports_providers(entry: &ModelEntry) -> bool {
    entry.provider_id == "local/onnx" && entry.task != ModelTask::DocumentExtract
}

pub(crate) fn compiled(feature: &str) -> bool {
    uniko_extract::COMPILED_FEATURES
        .iter()
        .any(|(name, on)| *name == feature && *on)
}

/// What one requirement needs once the static check has passed.
struct Need {
    requirement: String,
    aliases: Vec<String>,
    nlp: bool,
    accelerated: bool,
}

/// The parts of an instance the check reads.
pub(super) struct Context<'a> {
    pub config: &'a UnikoConfig,
    pub entries: &'a [ModelEntry],
    pub extractors: &'a ModalityRegistry,
    pub answer_alias: Option<&'a str>,
    /// Verify required remote aliases after they load.
    pub verify_remote: bool,
}

/// The static step: catalog, features and extractors only.
///
/// Returns the failures, and for each requirement that passed, the
/// aliases the dynamic step must load.
fn plan(requirements: &[Requirement], cx: &Context<'_>) -> (Vec<ReadinessFailure>, Vec<Need>) {
    let mut failures = Vec::new();
    let mut needs = Vec::new();
    let in_catalog = |alias: &str| cx.entries.iter().any(|e| e.alias == alias);
    for req in requirements {
        let name = req.name();
        let fail = |code: ReadinessCode| ReadinessFailure::new(name.clone(), code);
        let mut need = Need {
            requirement: name.clone(),
            aliases: Vec::new(),
            nlp: false,
            accelerated: false,
        };
        let outcome: Result<(), ReadinessFailure> = match req {
            Requirement::Alias(alias) | Requirement::Accelerated(alias) => {
                need.accelerated = matches!(req, Requirement::Accelerated(_));
                match cx.entries.iter().find(|e| &e.alias == alias) {
                    None => Err(fail(ReadinessCode::NotInCatalog).with_alias(alias.clone())),
                    Some(entry) if need.accelerated && !reports_providers(entry) => {
                        Err(fail(ReadinessCode::NotSupported).with_alias(alias.clone()))
                    }
                    Some(_) => {
                        need.aliases.push(alias.clone());
                        Ok(())
                    }
                }
            }
            Requirement::Embedder => {
                need.aliases.push(EMBED_ALIAS.to_string());
                if in_catalog(HYBRID_EMBED_ALIAS) {
                    need.aliases.push(HYBRID_EMBED_ALIAS.to_string());
                }
                if in_catalog(EMBED_ALIAS) {
                    Ok(())
                } else {
                    Err(fail(ReadinessCode::NotInCatalog).with_alias(EMBED_ALIAS))
                }
            }
            Requirement::Reranker => {
                let colbert = cx.config.reranker.style == "colbert";
                let alias = if colbert {
                    HYBRID_EMBED_ALIAS
                } else {
                    RERANK_ALIAS
                };
                if !cx.config.reranker.enabled {
                    Err(fail(ReadinessCode::NotConfigured))
                } else if in_catalog(alias) {
                    need.aliases.push(alias.to_string());
                    Ok(())
                } else if colbert {
                    // The colbert style needs the hybrid embedder, which the
                    // config adds with `sparse_dimensions` or
                    // `multivector_dimensions` (and `validate` requires).
                    Err(fail(ReadinessCode::NotConfigured).with_alias(alias))
                } else {
                    Err(fail(ReadinessCode::NotInCatalog).with_alias(alias))
                }
            }
            Requirement::Nlp => {
                // This crate's own `onnx` feature, which gates the NLP
                // pipeline `nlp_starts` runs (it turns on uniko-extract's).
                if !cfg!(feature = "onnx") {
                    Err(fail(ReadinessCode::NotCompiled))
                } else if in_catalog(NLP_ALIAS) {
                    need.aliases.push(NLP_ALIAS.to_string());
                    need.nlp = true;
                    Ok(())
                } else {
                    Err(fail(ReadinessCode::NotInCatalog).with_alias(NLP_ALIAS))
                }
            }
            Requirement::Ocr => {
                if !compiled("pdf-ocr") {
                    Err(fail(ReadinessCode::NotCompiled))
                } else if !cx.config.ocr.enabled {
                    Err(fail(ReadinessCode::NotConfigured))
                } else if in_catalog(OCR_ALIAS) {
                    need.aliases.push(OCR_ALIAS.to_string());
                    Ok(())
                } else {
                    Err(fail(ReadinessCode::NotInCatalog).with_alias(OCR_ALIAS))
                }
            }
            Requirement::Generator => match cx.answer_alias {
                None => Err(fail(ReadinessCode::NotConfigured)),
                Some(alias) => match cx.entries.iter().find(|e| e.alias == alias) {
                    Some(e) if e.task == ModelTask::Generate => {
                        need.aliases.push(alias.to_string());
                        Ok(())
                    }
                    // `build()` rejects a non-generation answer alias
                    // before readiness runs; this covers `check_readiness`.
                    Some(_) => Err(fail(ReadinessCode::NotConfigured).with_alias(alias)),
                    None => Err(fail(ReadinessCode::NotInCatalog).with_alias(alias)),
                },
            },
            Requirement::Extractor(m) => {
                if cx.extractors.modalities().contains(m) {
                    Ok(())
                } else {
                    Err(fail(ReadinessCode::NoExtractor))
                }
            }
        };
        match outcome {
            Ok(()) => needs.push(need),
            Err(f) => failures.push(f),
        }
    }
    (failures, needs)
}

/// Run the static step only, before a store exists.
pub(super) fn check_static(
    requirements: &[Requirement],
    cx: &Context<'_>,
) -> Result<(), UnikoError> {
    let (failures, _) = plan(requirements, cx);
    if failures.is_empty() {
        Ok(())
    } else {
        Err(UnikoError::NotReady(failures))
    }
}

/// Run both steps against an open store, returning every unmet requirement.
///
/// `timeout` bounds the whole dynamic step: loading the required aliases
/// and starting the NLP pipeline.
pub(super) async fn check(
    kb: &KnowledgeBase,
    requirements: &[Requirement],
    cx: &Context<'_>,
    timeout: Option<Duration>,
) -> Vec<ReadinessFailure> {
    let (mut failures, needs) = plan(requirements, cx);
    let mut aliases: Vec<&str> = needs
        .iter()
        .flat_map(|n| n.aliases.iter().map(String::as_str))
        .collect();
    aliases.sort_unstable();
    aliases.dedup();
    if aliases.is_empty() {
        return failures;
    }
    // A timeout too large to add (e.g. `Duration::MAX`) means no deadline.
    let deadline = timeout.and_then(|t| tokio::time::Instant::now().checked_add(t));
    let mut unfinished = kb.prefetch_aliases(&aliases, timeout).await;
    let mut status = kb.model_status().await.unwrap_or_default();
    // An alias another caller started loading after ours finished reads as
    // `Loading`. Wait for that load once more (a load waits on the one in
    // progress), within what is left of the deadline.
    let loading: Vec<&str> = aliases
        .iter()
        .copied()
        .filter(|a| {
            status
                .iter()
                .any(|s| s.alias == *a && s.state == ModelState::Loading)
        })
        .collect();
    if !loading.is_empty() {
        let remaining =
            deadline.map(|at| at.saturating_duration_since(tokio::time::Instant::now()));
        unfinished.extend(kb.prefetch_aliases(&loading, remaining).await);
        status = kb.model_status().await.unwrap_or_default();
    }

    let verification = if cx.verify_remote {
        verify_loaded(kb, &aliases, &status, deadline).await
    } else {
        Verification::default()
    };
    for need in &needs {
        let mut met = true;
        for alias in &need.aliases {
            let failure = |code| {
                ReadinessFailure::new(need.requirement.clone(), code).with_alias(alias.clone())
            };
            // Fail closed: an alias the store cannot report on is not met.
            let unmet = match status.iter().find(|s| &s.alias == alias) {
                _ if unfinished.contains(alias) => Some(failure(ReadinessCode::TimedOut)),
                None => Some(failure(ReadinessCode::NotInCatalog)),
                Some(model) => match &model.state {
                    ModelState::Loaded if need.accelerated && !accelerated(model) => {
                        Some(failure(ReadinessCode::NotAccelerated))
                    }
                    ModelState::Loaded
                        if verification.timed_out && model.provider_id.starts_with("remote/") =>
                    {
                        Some(failure(ReadinessCode::TimedOut))
                    }
                    ModelState::Loaded => verification.failed.get(alias).map(|reason| {
                        failure(ReadinessCode::VerificationFailed).with_error(reason.clone())
                    }),
                    ModelState::Failed(error) => {
                        Some(failure(ReadinessCode::LoadFailed).with_error(error.clone()))
                    }
                    // Still loading for another caller after the wait above.
                    ModelState::Loading => Some(failure(ReadinessCode::TimedOut)),
                    // Loading finished but nothing recorded it.
                    _ => Some(failure(ReadinessCode::LoadFailed)),
                },
            };
            if let Some(f) = unmet {
                failures.push(f);
                met = false;
            }
        }
        if met && need.nlp {
            let started = match deadline {
                None => Some(nlp_starts(kb).await),
                Some(at) => tokio::time::timeout_at(at, nlp_starts(kb)).await.ok(),
            };
            let code = match started {
                Some(true) => None,
                Some(false) => Some(ReadinessCode::NlpUnavailable),
                None => Some(ReadinessCode::TimedOut),
            };
            if let Some(code) = code {
                failures.push(ReadinessFailure::new(need.requirement.clone(), code));
            }
        }
    }
    failures
}

/// What verifying the loaded aliases found.
#[derive(Default)]
struct Verification {
    /// Each alias whose check failed, with the reason.
    failed: std::collections::HashMap<String, String>,
    /// The deadline ran out before the checks finished.
    timed_out: bool,
}

/// Verify those of `aliases` that loaded and are served by a remote
/// provider, within `deadline`. Local aliases need no check.
async fn verify_loaded(
    kb: &KnowledgeBase,
    aliases: &[&str],
    status: &[uniko_store::ModelStatus],
    deadline: Option<tokio::time::Instant>,
) -> Verification {
    let loaded: Vec<&str> = aliases
        .iter()
        .copied()
        .filter(|alias| {
            status.iter().any(|s| {
                s.alias == *alias
                    && s.state == ModelState::Loaded
                    && s.provider_id.starts_with("remote/")
            })
        })
        .collect();
    if loaded.is_empty() {
        return Verification::default();
    }
    let remaining = deadline.map(|at| at.saturating_duration_since(tokio::time::Instant::now()));
    match kb.verify_aliases(&loaded, remaining).await {
        Some(failed) => Verification {
            failed: failed.into_iter().collect(),
            timed_out: false,
        },
        None => Verification {
            failed: Default::default(),
            timed_out: true,
        },
    }
}

/// Whether the NLP pipeline starts. `plan` only asks on `onnx` builds.
#[cfg(feature = "onnx")]
async fn nlp_starts(kb: &KnowledgeBase) -> bool {
    uniko_extract::nlp::NlpPipeline::try_new(kb).await.is_some()
}

#[cfg(not(feature = "onnx"))]
async fn nlp_starts(_kb: &KnowledgeBase) -> bool {
    false
}

/// Whether a loaded model runs on a non-CPU provider it asked for.
///
/// A model that asked for no accelerator, or whose active providers are
/// unknown (a remote model), does not count as accelerated.
fn accelerated(model: &uniko_store::ModelStatus) -> bool {
    let is_cpu = |ep: &String| ep.eq_ignore_ascii_case("cpu");
    let (Some(requested), Some(active)) = (
        &model.requested_execution_providers,
        &model.active_execution_providers,
    ) else {
        return false;
    };
    requested
        .iter()
        .filter(|ep| !is_cpu(ep))
        .any(|ep| active.iter().any(|a| a.eq_ignore_ascii_case(ep)))
}

impl Uniko {
    /// Check `requirements` against this instance, loading any required
    /// alias that has not loaded yet. Never fails: returns every unmet
    /// requirement, empty when all are met.
    ///
    /// `timeout` bounds the loading, as
    /// [`readiness_timeout`](super::UnikoBuilder::readiness_timeout) does
    /// for `build`. A required alias that failed to load is tried again on
    /// every call, so a health endpoint probing often should pass a
    /// timeout and, if loads are slow, cache the result between probes.
    pub async fn check_readiness(
        &self,
        requirements: &[Requirement],
        timeout: Option<Duration>,
    ) -> Vec<ReadinessFailure> {
        let entries = self.kb.model_entries().unwrap_or_default();
        let cx = Context {
            config: self.kb.config(),
            entries: &entries,
            extractors: &self.extractors,
            answer_alias: self.llm_alias.as_deref(),
            verify_remote: self.verify_remote,
        };
        check(&self.kb, requirements, &cx, timeout).await
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use uniko_pipes::content::Modality;
    use uniko_store::config::UnikoConfig;
    use uniko_store::{ModelState, PrefetchPolicy, ReadinessCode, ReadinessFailure, UnikoError};

    use super::{ModelEntry, ModelTask, Requirement, reports_providers};
    use crate::Uniko;

    fn not_ready(result: Result<Uniko, UnikoError>) -> Vec<ReadinessFailure> {
        match result {
            Err(UnikoError::NotReady(failures)) => failures,
            Err(e) => panic!("expected NotReady, got {e}"),
            Ok(_) => panic!("expected NotReady, got an instance"),
        }
    }

    /// `true` when the default embedder could not load here (no network
    /// and no model cache), so a test that needs it should skip.
    fn embedder_unavailable(failures: &[ReadinessFailure]) -> bool {
        let skip = failures.iter().any(|f| {
            f.code == ReadinessCode::LoadFailed
                && f.alias.as_deref() == Some(uniko_store::schema::EMBED_ALIAS)
        });
        if skip {
            eprintln!("skipping: the default embedder is unavailable: {failures:?}");
        }
        skip
    }

    fn gpu_compiled() -> bool {
        uniko_store::COMPILED_FEATURES
            .iter()
            .any(|(name, on)| name.starts_with("gpu-") && *on)
    }

    fn codes(failures: &[ReadinessFailure]) -> Vec<(String, ReadinessCode)> {
        failures
            .iter()
            .map(|f| (f.requirement.clone(), f.code))
            .collect()
    }

    #[tokio::test]
    async fn static_requirements_fail_before_the_store_is_created() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("store");
        let mut config = UnikoConfig::default();
        config.reranker.enabled = false;
        let result = Uniko::builder()
            .path(&path)
            .raw_config(config)
            .require([
                Requirement::alias("embed/typo"),
                Requirement::Extractor(Modality::Image),
                Requirement::Reranker,
                Requirement::Generator,
                Requirement::Embedder,
            ])
            .build()
            .await;
        let failures = not_ready(result);
        assert_eq!(
            codes(&failures),
            vec![
                ("alias:embed/typo".into(), ReadinessCode::NotInCatalog),
                ("extractor:image".into(), ReadinessCode::NoExtractor),
                ("reranker".into(), ReadinessCode::NotConfigured),
                ("generator".into(), ReadinessCode::NotConfigured),
            ],
            "the met requirement (embedder) is not reported"
        );
        assert_eq!(failures[0].alias.as_deref(), Some("embed/typo"));
        assert!(
            !path.exists(),
            "no store may be created for a static failure"
        );
        let message = UnikoError::NotReady(failures).to_string();
        assert!(
            message.contains("alias:embed/typo: not_in_catalog"),
            "{message}"
        );
    }

    #[tokio::test]
    async fn ocr_needs_the_feature_and_the_config() {
        let failures = not_ready(Uniko::builder().require([Requirement::Ocr]).build().await);
        let expected = if super::compiled("pdf-ocr") {
            ReadinessCode::NotConfigured
        } else {
            ReadinessCode::NotCompiled
        };
        assert_eq!(codes(&failures), vec![("ocr".into(), expected)]);
    }

    #[tokio::test]
    async fn a_lazy_store_loads_only_what_is_required() {
        let result = Uniko::builder()
            .prefetch(PrefetchPolicy::Lazy)
            .require([Requirement::Embedder])
            .build()
            .await;
        let memory = match result {
            Ok(m) => m,
            Err(UnikoError::NotReady(f)) if embedder_unavailable(&f) => return,
            Err(e) => panic!("{e}"),
        };
        let models = memory.capabilities().await.models.unwrap();
        for model in &models {
            let expected = if model.alias == uniko_store::schema::EMBED_ALIAS {
                ModelState::Loaded
            } else {
                ModelState::NotLoaded
            };
            assert_eq!(model.state, expected, "{}", model.alias);
        }
    }

    #[tokio::test]
    async fn a_background_store_waits_for_what_is_required() {
        let result = Uniko::builder()
            .prefetch(PrefetchPolicy::Background)
            .require([Requirement::alias(uniko_store::schema::EMBED_ALIAS)])
            .build()
            .await;
        let memory = match result {
            Ok(m) => m,
            Err(UnikoError::NotReady(f)) if embedder_unavailable(&f) => return,
            Err(e) => panic!("{e}"),
        };
        let models = memory.capabilities().await.models.unwrap();
        let embed = models
            .iter()
            .find(|m| m.alias == uniko_store::schema::EMBED_ALIAS)
            .unwrap();
        assert_eq!(embed.state, ModelState::Loaded);
        memory.shutdown().await.unwrap();
    }

    /// A required model whose artifacts cannot be found fails readiness;
    /// the same model left optional does not stop the build.
    #[tokio::test]
    async fn a_required_model_that_cannot_load_fails_and_an_optional_one_does_not() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = UnikoConfig::default();
        let mut catalog = uniko_store::storage::embed_catalog(&config);
        let mut missing = catalog[0].clone();
        missing.alias = "embed/missing".into();
        missing.model_id = "uniko-test/no-such-model-repo".into();
        catalog.push(missing);
        let path = dir.path().join("catalog.json");
        std::fs::write(&path, serde_json::to_string(&catalog).unwrap()).unwrap();
        config.catalog_path = Some(path);

        let builder = || {
            Uniko::builder()
                .raw_config(config.clone())
                .prefetch(PrefetchPolicy::Lazy)
        };
        let failures = not_ready(
            builder()
                .require([Requirement::alias("embed/missing")])
                .build()
                .await,
        );
        assert_eq!(
            codes(&failures),
            vec![("alias:embed/missing".into(), ReadinessCode::LoadFailed)]
        );
        assert!(failures[0].error.is_some(), "{failures:?}");

        let memory = builder()
            .build()
            .await
            .expect("an optional model that cannot load must not fail build");
        let failures = memory
            .check_readiness(&[Requirement::alias("embed/missing")], None)
            .await;
        assert_eq!(failures.len(), 1, "{failures:?}");
        assert_eq!(failures[0].code, ReadinessCode::LoadFailed);
    }

    /// A remote model reports no execution providers, so `Accelerated` on
    /// one is refused before the store opens.
    #[tokio::test]
    async fn accelerated_is_refused_for_models_that_do_not_report_providers() {
        let failures = not_ready(
            Uniko::builder()
                .llm(crate::LlmSpec::llamacpp(
                    "gen/a",
                    "stub",
                    "http://127.0.0.1:9/v1",
                ))
                .require([Requirement::accelerated("gen/a")])
                .build()
                .await,
        );
        assert_eq!(
            codes(&failures),
            vec![("accelerated:gen/a".into(), ReadinessCode::NotSupported)]
        );
    }

    /// A local HTTP server answering every request with `status`, counting
    /// the requests it gets. Returns its `/v1` base URL.
    async fn stub_server(status: u16) -> (String, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = std::sync::Arc::new(AtomicUsize::new(0));
        let counter = hits.clone();
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                counter.fetch_add(1, Ordering::SeqCst);
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    let _ = socket.read(&mut buf).await;
                    let reply = format!(
                        "HTTP/1.1 {status} X\r\ncontent-length: 2\r\ncontent-type: application/json\r\nconnection: close\r\n\r\n{{}}"
                    );
                    let _ = socket.write_all(reply.as_bytes()).await;
                    let _ = socket.shutdown().await;
                });
            }
        });
        (format!("http://{addr}/v1"), hits)
    }

    fn remote_generator(url: &str) -> crate::UnikoBuilder {
        Uniko::builder()
            .llm(crate::LlmSpec::llamacpp("gen/a", "stub", url))
            .prefetch(PrefetchPolicy::Lazy)
            .require([Requirement::alias("gen/a")])
    }

    /// Without `verify_remote`, a remote alias is ready once configured:
    /// nothing contacts it.
    #[tokio::test]
    async fn remote_aliases_are_not_contacted_unless_asked() {
        let (url, hits) = stub_server(503).await;
        let memory = remote_generator(&url).build().await.unwrap();
        assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), 0);
        memory.shutdown().await.unwrap();
    }

    /// With `verify_remote`, an endpoint that is down or refuses fails as
    /// `verification_failed`, naming why.
    #[tokio::test]
    async fn verify_remote_reports_an_unusable_endpoint() {
        let (url, _) = stub_server(503).await;
        let failures = not_ready(remote_generator(&url).verify_remote(true).build().await);
        assert_eq!(
            codes(&failures),
            vec![("alias:gen/a".into(), ReadinessCode::VerificationFailed)]
        );
        let reason = failures[0].error.as_deref().unwrap_or_default();
        assert!(reason.starts_with("unavailable"), "{reason}");

        // Nothing listens on port 9.
        let failures = not_ready(
            remote_generator("http://127.0.0.1:9/v1")
                .verify_remote(true)
                .build()
                .await,
        );
        let reason = failures[0].error.as_deref().unwrap_or_default();
        assert_eq!(failures[0].code, ReadinessCode::VerificationFailed);
        assert!(reason.starts_with("unreachable"), "{reason}");
    }

    /// A healthy endpoint passes, and later checks on the instance verify
    /// it again.
    #[tokio::test]
    async fn verify_remote_passes_a_healthy_endpoint_and_rechecks() {
        use std::sync::atomic::Ordering;

        let (url, hits) = stub_server(200).await;
        let memory = remote_generator(&url)
            .verify_remote(true)
            .build()
            .await
            .unwrap();
        let after_build = hits.load(Ordering::SeqCst);
        assert!(after_build >= 1);
        let failures = memory
            .check_readiness(&[Requirement::alias("gen/a")], None)
            .await;
        assert!(failures.is_empty(), "{failures:?}");
        assert!(hits.load(Ordering::SeqCst) > after_build);
        memory.shutdown().await.unwrap();
    }

    /// A server that accepts connections and never answers.
    async fn silent_server() -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let mut held = Vec::new();
            while let Ok((socket, _)) = listener.accept().await {
                held.push(socket);
            }
        });
        format!("http://{addr}/v1")
    }

    /// A check still waiting when the readiness timeout runs out is
    /// `timed_out`, and `build` returns promptly rather than after the
    /// check's own (up to 30 s) limit.
    #[tokio::test]
    async fn a_verification_cut_off_by_the_timeout_is_timed_out() {
        let url = silent_server().await;
        let started = std::time::Instant::now();
        let failures = not_ready(
            remote_generator(&url)
                .verify_remote(true)
                .readiness_timeout(Duration::from_millis(300))
                .build()
                .await,
        );
        assert_eq!(
            codes(&failures),
            vec![("alias:gen/a".into(), ReadinessCode::TimedOut)]
        );
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "{:?}",
            started.elapsed()
        );
    }

    /// A local model needs no check: `verify_remote` leaves it ready.
    #[tokio::test]
    async fn verify_remote_passes_local_models_without_a_request() {
        let mut config = UnikoConfig::default();
        config.reranker.enabled = false;
        let built = Uniko::builder()
            .raw_config(config)
            .prefetch(PrefetchPolicy::Lazy)
            .verify_remote(true)
            .require([Requirement::Embedder])
            .build()
            .await;
        match built {
            Ok(memory) => memory.shutdown().await.unwrap(),
            Err(UnikoError::NotReady(failures)) if embedder_unavailable(&failures) => {}
            Err(e) => panic!("{e}"),
        }
    }

    /// Every local ONNX task reports its providers (uni-xervo 0.20), NLP
    /// and OCR included; document extraction and remote models do not.
    #[test]
    fn every_local_onnx_task_but_document_extraction_reports_providers() {
        let entry = |provider: &str, task| {
            ModelEntry::from(&uniko_store::xervo::ModelAliasSpec {
                alias: "a".into(),
                task,
                provider_id: provider.into(),
                model_id: "m".into(),
                revision: None,
                warmup: uniko_store::xervo::WarmupPolicy::Lazy,
                required: false,
                timeout: None,
                load_timeout: None,
                retry: None,
                options: serde_json::Value::Null,
            })
        };
        for task in [
            ModelTask::Embed,
            ModelTask::EmbedHybrid,
            ModelTask::EmbedSparse,
            ModelTask::EmbedMultiVector,
            ModelTask::Rerank,
            ModelTask::Nlp,
            ModelTask::Ocr,
        ] {
            assert!(reports_providers(&entry("local/onnx", task)), "{task:?}");
        }
        assert!(!reports_providers(&entry(
            "local/onnx",
            ModelTask::DocumentExtract
        )));
        assert!(!reports_providers(&entry(
            "remote/openai",
            ModelTask::Embed
        )));
    }

    /// With requirements, the required models load first and the prefetch
    /// policy then runs for the rest: under `Eager`, everything is loaded
    /// when `build` returns.
    #[tokio::test]
    async fn the_prefetch_policy_runs_after_the_required_models() {
        let mut config = UnikoConfig::default();
        config.reranker.enabled = false;
        let memory = match Uniko::builder()
            .raw_config(config)
            .require([Requirement::Embedder])
            .build()
            .await
        {
            Ok(m) => m,
            Err(UnikoError::NotReady(f)) if embedder_unavailable(&f) => return,
            Err(e) => panic!("{e}"),
        };
        let models = memory.capabilities().await.models.unwrap();
        let embed = models
            .iter()
            .find(|m| m.alias == uniko_store::schema::EMBED_ALIAS)
            .unwrap();
        assert_eq!(embed.state, ModelState::Loaded);
        assert!(
            models.iter().all(|m| m.state != ModelState::NotLoaded),
            "Eager prefetch should have tried every alias: {models:?}"
        );
    }

    /// An accelerator-only provider list fails to load on a host without
    /// it; one that allows a CPU fallback loads but is not accelerated.
    #[tokio::test]
    async fn a_cpu_fallback_fails_an_accelerated_requirement() {
        if gpu_compiled() {
            // On a host with a usable GPU the accelerator really attaches.
            eprintln!("skipping: a GPU build may really have the accelerator");
            return;
        }
        let with_eps = |eps: &[&str]| {
            let mut config = UnikoConfig::default();
            config.embedding.execution_providers =
                Some(eps.iter().map(|s| (*s).to_string()).collect());
            Uniko::builder()
                .raw_config(config)
                .prefetch(PrefetchPolicy::Lazy)
        };

        let failures = not_ready(
            with_eps(&["cuda"])
                .require([Requirement::Embedder])
                .build()
                .await,
        );
        assert_eq!(
            codes(&failures),
            vec![("embedder".into(), ReadinessCode::LoadFailed)]
        );

        let failures = not_ready(
            with_eps(&["cuda", "cpu"])
                .require([Requirement::accelerated(uniko_store::schema::EMBED_ALIAS)])
                .build()
                .await,
        );
        if embedder_unavailable(&failures) {
            return;
        }
        assert_eq!(
            codes(&failures),
            vec![(
                "accelerated:embed/default".into(),
                ReadinessCode::NotAccelerated
            )]
        );
    }

    /// A model still loading when the timeout runs out is `timed_out`, the
    /// store is released, and a later check without a timeout loads it.
    #[tokio::test]
    async fn the_readiness_timeout_reports_timed_out_and_releases_the_store() {
        let dir = tempfile::tempdir().unwrap();
        let builder = || {
            Uniko::builder()
                .path(dir.path())
                .prefetch(PrefetchPolicy::Lazy)
                .require([Requirement::Embedder])
        };
        let failures = not_ready(
            builder()
                .readiness_timeout(Duration::from_millis(1))
                .build()
                .await,
        );
        if embedder_unavailable(&failures) {
            return;
        }
        assert_eq!(
            codes(&failures),
            vec![("embedder".into(), ReadinessCode::TimedOut)]
        );

        // The failed build released the path, so it opens again.
        let memory = match builder().build().await {
            Ok(m) => m,
            Err(UnikoError::NotReady(f)) if embedder_unavailable(&f) => return,
            Err(e) => panic!("{e}"),
        };
        let failures = memory
            .check_readiness(&[Requirement::Embedder, Requirement::Nlp], None)
            .await;
        // Without `onnx` there is no NLP pipeline to require.
        let expected: Vec<(String, ReadinessCode)> = if cfg!(feature = "onnx") {
            Vec::new()
        } else {
            vec![("nlp".into(), ReadinessCode::NotCompiled)]
        };
        assert_eq!(codes(&failures), expected, "{failures:?}");
    }

    #[tokio::test]
    async fn requirements_over_a_shared_runtime_use_its_catalog() {
        let Ok(runtime) = Uniko::builder()
            .prefetch(PrefetchPolicy::Lazy)
            .build_runtime()
            .await
        else {
            eprintln!("skipping: runtime unavailable");
            return;
        };
        let failures = not_ready(
            Uniko::builder()
                .runtime(runtime.clone())
                .require([Requirement::alias("gen/absent")])
                .build()
                .await,
        );
        assert_eq!(failures[0].code, ReadinessCode::NotInCatalog);

        let memory = match Uniko::builder()
            .runtime(runtime.clone())
            .require([Requirement::Embedder])
            .build()
            .await
        {
            Ok(m) => m,
            Err(UnikoError::NotReady(f)) if embedder_unavailable(&f) => return,
            Err(e) => panic!("{e}"),
        };
        // The load is recorded in the runtime, so a second store sees it.
        let second = Uniko::builder().runtime(runtime).build().await.unwrap();
        let models = second.capabilities().await.models.unwrap();
        let embed = models
            .iter()
            .find(|m| m.alias == uniko_store::schema::EMBED_ALIAS)
            .unwrap();
        assert_eq!(embed.state, ModelState::Loaded);
        drop(memory);
    }
}
