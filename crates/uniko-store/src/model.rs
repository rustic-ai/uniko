//! Model-runtime seam (issue #2): typed access to embedding, text
//! generation, and NLP inference without exposing the raw uni-db
//! `xervo()` handle to higher crates.
//!
//! Before this seam, callers reached the model runtime through
//! `kb.db().xervo()` — the same escape hatch the repository module
//! closes for graph queries. These methods give embedding (`embed`),
//! generation (`generate`), and the NLP runtime handle (`model_runtime`)
//! their own typed surface so `crates/uniko-{memory,extract,cortex}`
//! never name `uni_db::xervo` directly.
//!
//! Prompt-construction types ([`GenerationOptions`], [`Message`]) are
//! re-exported from the crate root (`uniko_store::xervo`); the NLP model
//! traits live in `uni_xervo` (a separate dependency the CI gate does
//! not seal) and are reached via the [`ModelRuntime`](crate::ModelRuntime)
//! returned by [`KnowledgeBase::model_runtime`].

use crate::error::{Result, UnikoError};
use crate::storage::KnowledgeBase;

pub use uni_db::xervo::{GenerationOptions, Message};

impl KnowledgeBase {
    /// True if the Xervo embedding/generation runtime is loaded.
    ///
    /// Callers that want a graceful fallback (rule-based extraction,
    /// skipping a computed embedding) check this before [`embed`] /
    /// [`generate`] to avoid constructing a request that can only fail.
    ///
    /// [`embed`]: Self::embed
    /// [`generate`]: Self::generate
    #[must_use]
    pub fn model_runtime_available(&self) -> bool {
        self.db.xervo().is_available()
    }

    /// Embed `texts` under model `alias`, one vector per input (order
    /// preserved). Applies no prefix — callers that need the model's
    /// document/query prefix add it before calling.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Embedding`] if the runtime is unavailable or
    /// the model fails.
    pub async fn embed(&self, alias: &str, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let xervo = self.db.xervo();
        if !xervo.is_available() {
            return Err(UnikoError::Embedding(
                "Xervo embedding runtime not available".into(),
            ));
        }
        xervo
            .embed(alias, texts)
            .await
            .map_err(|e| UnikoError::Embedding(e.to_string()))
    }

    /// Embed `texts` into per-token multi-vectors (ColBERT) under `alias`,
    /// one `Vec<Vec<f32>>` per input (order preserved).
    ///
    /// Produces the query side of ColBERT MaxSim reranking; document-side
    /// token vectors are auto-embedded into the `colbert_embedding` column
    /// by the same hybrid alias. Applies no prefix.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Embedding`] if the runtime is unavailable or
    /// the model fails (e.g. the alias has no multi-vector head).
    pub async fn embed_multivector(
        &self,
        alias: &str,
        texts: &[&str],
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        let xervo = self.db.xervo();
        if !xervo.is_available() {
            return Err(UnikoError::Embedding(
                "Xervo embedding runtime not available".into(),
            ));
        }
        xervo
            .embed_multivector(alias, texts)
            .await
            .map_err(|e| UnikoError::Embedding(e.to_string()))
    }

    /// Generate a completion for `messages` under model `alias`, returning
    /// the raw response text.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Llm`] if the runtime is unavailable or the
    /// generation call fails.
    pub async fn generate(
        &self,
        alias: &str,
        messages: &[Message],
        options: GenerationOptions,
    ) -> Result<String> {
        let result = self
            .db
            .xervo()
            .generate(alias, messages, options)
            .await
            .map_err(|e| UnikoError::Llm(e.to_string()))?;
        Ok(result.text)
    }

    /// Re-score `docs` against `query` with the cross-encoder reranker
    /// model `alias`. Returns `(doc_index, score)` pairs — `doc_index`
    /// indexes into the input `docs` slice. The raw xervo rerank result
    /// type is collapsed to tuples so callers need not name it.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Search`] if the runtime is unavailable or the
    /// reranker fails.
    pub async fn rerank(
        &self,
        alias: &str,
        query: &str,
        docs: &[&str],
    ) -> Result<Vec<(usize, f32)>> {
        let scored = self
            .db
            .xervo()
            .rerank(alias, query, docs)
            .await
            .map_err(|e| UnikoError::Search(e.to_string()))?;
        Ok(scored.into_iter().map(|s| (s.index, s.score)).collect())
    }

    /// Clone of the underlying [`ModelRuntime`](crate::ModelRuntime), if a
    /// runtime is configured.
    ///
    /// This is the NLP seam: callers resolve `ModelTask::Nlp` aliases to a
    /// `uni_xervo` `NlpModel` off the returned runtime (e.g.
    /// `runtime.nlp_model(alias).await`) without reaching through
    /// `kb.db().xervo()`. Returns `None` when no runtime is loaded, so
    /// callers fall back to rule-based paths.
    #[must_use]
    pub fn model_runtime(&self) -> Option<std::sync::Arc<crate::ModelRuntime>> {
        self.db.xervo().raw_runtime().cloned()
    }

    /// Check that `alias` names a text-generation model this store can
    /// use, loading it if it is not loaded yet.
    ///
    /// For callers that are about to commit to an expensive pass built on
    /// [`generate`](Self::generate) and want a misconfigured alias to fail
    /// up front rather than per call.
    ///
    /// # Errors
    ///
    /// - [`UnikoError::Config`] when no model runtime is configured, when
    ///   `alias` is not in the catalog, or when its catalog entry cannot
    ///   serve generation (wrong task or unknown provider).
    /// - [`UnikoError::Llm`] when the alias is well-formed but the model
    ///   failed to come up (download, weight loading, transport).
    ///
    /// This proves the model loads, not that a remote endpoint answers:
    /// remote providers are not contacted until the first call.
    pub async fn ensure_generator(&self, alias: &str) -> Result<()> {
        use uni_xervo::error::RuntimeError as E;

        let runtime = self.model_runtime().ok_or_else(|| {
            UnikoError::Config(format!(
                "generation alias '{alias}': no model runtime is configured"
            ))
        })?;
        if !runtime.contains_alias(alias).await {
            return Err(UnikoError::Config(format!(
                "generation alias '{alias}' is not registered in the model catalog"
            )));
        }
        runtime
            .generator(alias)
            .await
            .map(|_| ())
            .map_err(|e| match e {
                E::Config(_)
                | E::AliasNotFound { .. }
                | E::ProviderNotFound(_)
                | E::CapabilityMismatch(_)
                | E::ProviderCapabilityMissing { .. } => UnikoError::Config(format!(
                    "alias '{alias}' is not a usable generation model: {e}"
                )),
                other => UnikoError::Llm(format!(
                    "generation alias '{alias}' failed to load: {other}"
                )),
            })
    }
}
