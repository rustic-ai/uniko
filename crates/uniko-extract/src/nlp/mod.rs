//! Multi-task NLP via xervo's managed `NlpModel`.
//!
//! [`NlpPipeline`] resolves the `nlp/default` alias to a xervo
//! [`NlpModel`](uni_xervo::traits::NlpModel) (registered as
//! `ModelTask::Nlp`) and turns raw text into uniko's structured
//! [`NlpResult`] via [`adapter::xervo_to_uniko`]. xervo owns
//! tokenization, ONNX inference, and POS/NER/DEP/SRL/CLS decoding; uniko
//! only reconciles output conventions in the adapter.
//!
//! If the runtime or alias is unavailable, [`NlpPipeline::try_new`]
//! returns `None` and callers fall back to rule-based extraction.

pub mod adapter;
pub mod assets;
pub mod decode;
pub mod types;

use std::sync::Arc;

use uni_xervo::traits::{NlpModel, NlpRequest, NlpTasks};

use uniko_store::KnowledgeBase;
use uniko_store::UnikoError;
use uniko_store::schema::NLP_ALIAS;

use types::NlpResult;

/// xervo-backed multi-task NLP pipeline.
///
/// A single `analyze` call runs xervo's `NlpModel` (tokenize → infer →
/// decode) and adapts the result into uniko's [`NlpResult`]. Only the
/// resolved model handle is per-instance; label maps are a global
/// `OnceLock`.
#[derive(Clone)]
pub struct NlpPipeline {
    model: Arc<dyn NlpModel>,
    /// Mirror of `UnikoConfig.nlp_srl_enabled`. When `false`, SRL is not
    /// requested and `NlpResult.srl_frames` stays empty.
    srl_enabled: bool,
}

impl std::fmt::Debug for NlpPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NlpPipeline")
            .field("model", &"<NlpModel>")
            .field("srl_enabled", &self.srl_enabled)
            .finish()
    }
}

impl NlpPipeline {
    /// Try to create a pipeline from the KB's xervo runtime.
    ///
    /// Returns `None` if the runtime is absent or the `nlp/default` alias
    /// cannot be resolved as an NLP model (e.g. the model download
    /// fails) — callers fall back to rule-based extraction.
    pub async fn try_new(kb: &KnowledgeBase) -> Option<Self> {
        let runtime = kb.model_runtime()?;
        let model = match runtime.nlp_model(NLP_ALIAS).await {
            Ok(m) => m,
            Err(e) => {
                tracing::debug!(error = %e, alias = NLP_ALIAS, "xervo NlpModel unavailable");
                return None;
            }
        };
        Some(Self {
            model,
            srl_enabled: kb.config().nlp_srl_enabled,
        })
    }

    /// The NLP task set this pipeline requests (SRL gated by config).
    fn tasks(&self) -> NlpTasks {
        let mut tasks = NlpTasks::POS | NlpTasks::NER | NlpTasks::DEP | NlpTasks::CLS;
        if self.srl_enabled {
            tasks |= NlpTasks::SRL;
        }
        tasks
    }

    /// Run full NLP analysis on a single text string.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Pipeline`] if xervo inference fails or
    /// returns no result.
    pub async fn analyze(&self, text: &str) -> Result<NlpResult, UnikoError> {
        let mut out = self
            .model
            .analyze(vec![NlpRequest::new(text, self.tasks())])
            .await
            .map_err(|e| UnikoError::Pipeline(format!("xervo NLP analyze: {e}")))?;
        let x = out
            .pop()
            .ok_or_else(|| UnikoError::Pipeline("xervo NLP returned no result".into()))?;
        Ok(adapter::xervo_to_uniko(
            &x,
            assets::label_maps(),
            self.srl_enabled,
        ))
    }

    /// Analyze text by splitting into sentences first.
    ///
    /// Returns one [`NlpResult`] per sentence (sentences under 4 words
    /// are dropped by [`split_sentences`]). All sentences go through a
    /// single `analyze` call as separate requests, which keeps xervo's
    /// per-sentence token indexing (NER/DEP/SRL) local to each sentence.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Pipeline`] if xervo inference fails.
    pub async fn analyze_sentences(&self, text: &str) -> Result<Vec<NlpResult>, UnikoError> {
        let sentences = split_sentences(text);
        if sentences.is_empty() {
            return Ok(Vec::new());
        }
        let tasks = self.tasks();
        let requests: Vec<NlpRequest<'_>> = sentences
            .iter()
            .map(|s| NlpRequest::new(s.as_str(), tasks))
            .collect();
        let results = self
            .model
            .analyze(requests)
            .await
            .map_err(|e| UnikoError::Pipeline(format!("xervo NLP analyze: {e}")))?;
        let labels = assets::label_maps();
        Ok(results
            .iter()
            .map(|x| adapter::xervo_to_uniko(x, labels, self.srl_enabled))
            .collect())
    }
}

/// Split text into sentences for per-sentence NLP analysis.
///
/// Splits on sentence-ending punctuation (`.` `!` `?`) followed by
/// whitespace or end-of-string. Filters fragments under 4 words.
/// If no sentences survive filtering, returns the original text
/// (provided it has >= 4 words).
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut start = 0;

    for (byte_pos, ch) in text.char_indices() {
        if ch != '.' && ch != '!' && ch != '?' {
            continue;
        }
        let end = byte_pos + ch.len_utf8();
        // Must be followed by whitespace or end-of-string.
        let at_boundary = end >= text.len()
            || text[end..]
                .chars()
                .next()
                .is_some_and(|c| c.is_whitespace());
        if !at_boundary {
            continue;
        }

        let sentence = text[start..end].trim();
        if sentence.split_whitespace().count() >= 4 {
            sentences.push(sentence.to_string());
        }
        // Advance past punctuation + whitespace.
        start = end;
        while start < text.len() {
            match text[start..].chars().next() {
                Some(ws) if ws.is_whitespace() => start += ws.len_utf8(),
                _ => break,
            }
        }
    }

    // Trailing text without terminal punctuation.
    if start < text.len() {
        let remainder = text[start..].trim();
        if remainder.split_whitespace().count() >= 4 {
            sentences.push(remainder.to_string());
        }
    }

    // Fallback: if nothing survived, use the original text.
    if sentences.is_empty() && text.split_whitespace().count() >= 4 {
        sentences.push(text.trim().to_string());
    }

    sentences
}
