//! [`TextExtractor`]: a [`ModalityExtractor`] built from a function that
//! turns an image, audio or video payload into text — a caption, a
//! transcript, OCR output.
//!
//! The text becomes an ordinary artifact: chunked, embedded and recalled
//! like a document. The function runs on a blocking thread, so it can be a
//! slow model call, or a call back into a host language (the Python binding
//! uses this for `UnikoBuilder.extractor`).

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value as JsonValue;
use uniko_pipes::content::{Mime, Modality};
use uniko_pipes::types::{IngestArtifact, IngestData, IngestSource};
use uniko_store::{KnowledgeBase, UnikoError};

use super::artifact::{
    ArtifactContextNids, ArtifactIngestResult, ArtifactPrep, UnitArtifactSeen,
    ingest_artifact_in_tx, prepare_artifact,
};
use super::modality::{ModalityExtractor, ModalityPrepared};
use super::source::resolve_mime;

/// What a [`TextExtractor`] function receives.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ExtractRequest {
    /// The modality being extracted.
    pub modality: Modality,
    /// The payload bytes.
    pub bytes: Vec<u8>,
    /// The resolved MIME type.
    pub mime: Mime,
    /// The source path, when the payload came from one or the caller set it.
    pub path: Option<String>,
    /// The caller's metadata on the source.
    pub metadata: HashMap<String, JsonValue>,
}

/// The function a [`TextExtractor`] calls: payload in, text out.
pub type ExtractFn = Arc<dyn Fn(ExtractRequest) -> Result<String, UnikoError> + Send + Sync>;

/// A [`ModalityExtractor`] that indexes the text a function extracts.
///
/// Register one per modality with
/// `UnikoBuilder::extractor(Arc::new(TextExtractor::new(Modality::Image, f)))`.
/// The stored artifact's text is the extracted text, with `kind` set to the
/// modality's name; the original bytes are not kept.
#[derive(Clone)]
pub struct TextExtractor {
    modality: Modality,
    extract: ExtractFn,
}

impl std::fmt::Debug for TextExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextExtractor")
            .field("modality", &self.modality)
            .finish_non_exhaustive()
    }
}

impl TextExtractor {
    /// An extractor for `modality` that indexes whatever `extract` returns.
    pub fn new(
        modality: Modality,
        extract: impl Fn(ExtractRequest) -> Result<String, UnikoError> + Send + Sync + 'static,
    ) -> Self {
        Self {
            modality,
            extract: Arc::new(extract),
        }
    }
}

/// The prep a [`TextExtractor`] hands to [`ModalityExtractor::apply_in_tx`]:
/// the extracted text, ready to write as an artifact.
#[derive(Debug)]
struct TextPrep(ArtifactPrep);

impl ModalityPrepared for TextPrep {
    fn content_ids(&self) -> Vec<String> {
        vec![self.0.hash.clone()]
    }

    fn byte_size(&self) -> u64 {
        u64::try_from(self.0.size).unwrap_or(0)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn modality_name(modality: Modality) -> &'static str {
    match modality {
        Modality::Image => "image",
        Modality::Audio => "audio",
        Modality::Video => "video",
        _ => "media",
    }
}

#[async_trait]
impl ModalityExtractor for TextExtractor {
    fn modality(&self) -> Modality {
        self.modality
    }

    async fn prepare(
        &self,
        kb: &KnowledgeBase,
        src: &IngestSource,
    ) -> Result<Box<dyn ModalityPrepared>, UnikoError> {
        let bytes = match &src.data {
            IngestData::Bytes(b) => b.clone(),
            IngestData::Text(t) => t.as_bytes().to_vec(),
            IngestData::Path(p) => tokio::fs::read(p)
                .await
                .map_err(|e| UnikoError::Storage(format!("read {}: {e}", p.display())))?,
        };
        let path = src.path.clone().or_else(|| match &src.data {
            IngestData::Path(p) => Some(p.display().to_string()),
            _ => None,
        });
        let mime = resolve_mime(src.mime.as_ref(), Some(&bytes), path.as_deref(), false);
        let request = ExtractRequest {
            modality: self.modality,
            bytes,
            mime: mime.clone(),
            path: path.clone(),
            metadata: src.metadata.clone(),
        };
        let extract = Arc::clone(&self.extract);
        let content = tokio::task::spawn_blocking(move || extract(request))
            .await
            .map_err(|e| UnikoError::Internal(format!("extractor did not complete: {e}")))??;

        let mut metadata = src.metadata.clone();
        metadata
            .entry("mime".to_string())
            .or_insert_with(|| JsonValue::String(mime.to_string()));
        let caller_supplied_id = src.id.is_some();
        let artifact = IngestArtifact {
            artifact_id: src.id.clone().unwrap_or_else(uniko_store::id::new_id),
            caller_supplied_id,
            content,
            kind: modality_name(self.modality).to_string(),
            path,
            metadata,
            session_id: None,
            triggered_by_message_id: None,
            produced_by_action_id: None,
            category: src.category.clone(),
            source_id: src.source_id.clone(),
            revision_id: src.revision_id.clone(),
        };
        Ok(Box::new(TextPrep(prepare_artifact(kb, &artifact).await?)))
    }

    async fn apply_in_tx(
        &self,
        kb: &KnowledgeBase,
        tx: &uniko_store::Transaction,
        prep: &dyn ModalityPrepared,
        ctx: ArtifactContextNids,
        seen: &mut UnitArtifactSeen,
    ) -> Result<ArtifactIngestResult, UnikoError> {
        let prep = prep
            .as_any()
            .downcast_ref::<TextPrep>()
            .ok_or_else(|| UnikoError::Internal("TextExtractor got a foreign prep".into()))?;
        ingest_artifact_in_tx(kb, tx, &prep.0, ctx, seen).await
    }
}
