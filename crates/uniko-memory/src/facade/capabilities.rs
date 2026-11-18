//! [`Capabilities`]: what this build and this instance can do.

use std::collections::BTreeMap;

use serde::Serialize;
use uniko_pipes::content::Modality;
use uniko_store::ModelStatus;

use super::Uniko;

/// A report of what a [`Uniko`] instance can do, from
/// [`Uniko::capabilities`].
///
/// Safe to log or show: model entries carry no provider options, only the
/// name of an API-key environment variable, and endpoints without userinfo
/// or query.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Capabilities {
    /// Every optional Cargo feature of the engine crates, and whether this
    /// build compiled it in (e.g. `pdf-ocr`, `gpu-cuda`, `candle`).
    pub compiled_features: BTreeMap<String, bool>,
    /// Modalities with a user-registered
    /// [`ModalityExtractor`](crate::ModalityExtractor) (image, audio,
    /// video). Built-in text, code, markup, structured and PDF ingest are
    /// always available and are not listed, so this is empty by default.
    pub extractors: Vec<Modality>,
    /// Each catalog model and whether it has loaded; `None` when the
    /// store's catalog is unknown. A hybrid embedder used only by uni-db's
    /// auto-embedding on write reads as not loaded until the first recall.
    pub models: Option<Vec<ModelStatus>>,
    /// The alias [`Agent::answer`](crate::Agent::answer) uses, if any.
    pub answer_alias: Option<String>,
    /// Whether the tiered OCR PDF path is compiled in (`pdf-ocr`).
    pub ocr_compiled: bool,
    /// Whether the config enables OCR (`UnikoConfig::ocr.enabled`).
    ///
    /// PDFs go through OCR only when both this and
    /// [`ocr_compiled`](Self::ocr_compiled) are true.
    pub ocr_configured: bool,
}

impl Uniko {
    /// Report this build's compiled features, the registered extractors,
    /// and each model alias with its load state.
    ///
    /// Loads nothing: a model that was not prefetched and has not been used
    /// reads as [`ModelState::NotLoaded`](uniko_store::ModelState::NotLoaded).
    pub async fn capabilities(&self) -> Capabilities {
        let compiled_features = uniko_store::COMPILED_FEATURES
            .iter()
            .chain(uniko_extract::COMPILED_FEATURES)
            .chain(uniko_cortex::COMPILED_FEATURES)
            .map(|(name, on)| ((*name).to_string(), *on))
            .collect();
        Capabilities {
            compiled_features,
            extractors: self.extractors.modalities(),
            models: self.kb.model_status().await,
            answer_alias: self.llm_alias.clone(),
            ocr_compiled: cfg_ocr(),
            ocr_configured: self.kb.config().ocr.enabled,
        }
    }
}

fn cfg_ocr() -> bool {
    uniko_extract::COMPILED_FEATURES
        .iter()
        .any(|(name, on)| *name == "pdf-ocr" && *on)
}
