//! [`TextExtractor`]: a [`ModalityExtractor`] that turns an image, audio or
//! video payload into text: a caption, a transcript, OCR output.
//!
//! The text comes from a host function ([`TextExtractor::new`]: the Python
//! binding uses this for `UnikoBuilder.extractor`) or from models in the
//! runtime ([`ModelExtractor`](super::model_extractor::ModelExtractor)). It
//! becomes an ordinary artifact: chunked, embedded and recalled like a
//! document. A function can return plain text, or an
//! [`Extraction`](super::derivation::Extraction) whose segments carry their
//! page, time range, region, confidence and model into each chunk.
//!
//! With a [`MediaRetention`] other than `None`, the original media is
//! recorded as a source artifact, and the text artifact links to it with
//! `DERIVED_FROM`. See [`MediaRetention`].

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value as JsonValue;
use uniko_pipes::content::{Mime, Modality};
use uniko_pipes::types::{IngestArtifact, IngestData, IngestSource};
use uniko_store::repository::media::{MediaHeader, MediaSourceSpec};
use uniko_store::storage::blob::MergeContent;
use uniko_store::{KnowledgeBase, ModelEntry, UnikoError};

use super::artifact::{
    ArtifactContextNids, ArtifactIdentity, ArtifactIngestResult, ArtifactPrep, UnitArtifactSeen,
    ingest_artifact_in_tx, link_artifact_context_in_tx, prepare_artifact,
};
use super::chunking::ChunkConfig;
use super::derivation::{ExtractOutput, Extraction, MediaRetention, modality_name};
use super::modality::{ModalityExtractor, ModalityPrepared};
use super::model_extractor::{ModelExtractor, wav_info};
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

/// A plain-text extractor function: payload in, text out. What
/// [`TextExtractor::new`] takes.
pub type ExtractFn = Arc<dyn Fn(ExtractRequest) -> Result<String, UnikoError> + Send + Sync>;

/// An extractor function that returns text or segments: what
/// [`TextExtractor::with_output`] takes.
pub type ExtractOutputFn =
    Arc<dyn Fn(ExtractRequest) -> Result<ExtractOutput, UnikoError> + Send + Sync>;

/// Where a [`TextExtractor`]'s text comes from.
#[derive(Clone)]
enum Deriver {
    /// A host function, run on a blocking thread.
    Callback(ExtractOutputFn),
    /// Models in the store's runtime.
    Models(Arc<ModelExtractor>),
}

/// A [`ModalityExtractor`] that indexes extracted text.
///
/// Register one per modality with
/// `UnikoBuilder::extractor(Arc::new(TextExtractor::new(Modality::Image, f)))`.
/// The stored artifact's text is the extracted text, with `kind` set to the
/// modality's name. The original bytes are kept only under
/// [`with_retention`](Self::with_retention).
#[derive(Clone)]
pub struct TextExtractor {
    modality: Modality,
    deriver: Deriver,
    retention: MediaRetention,
}

impl std::fmt::Debug for TextExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("TextExtractor");
        d.field("modality", &self.modality)
            .field("retention", &self.retention);
        if let Deriver::Models(m) = &self.deriver {
            d.field("models", m);
        }
        d.finish_non_exhaustive()
    }
}

impl TextExtractor {
    /// An extractor for `modality` that indexes the text `extract` returns.
    pub fn new(
        modality: Modality,
        extract: impl Fn(ExtractRequest) -> Result<String, UnikoError> + Send + Sync + 'static,
    ) -> Self {
        Self::with_output(modality, move |req| extract(req).map(ExtractOutput::Text))
    }

    /// An extractor for `modality` that indexes the segments `extract`
    /// returns, each chunk carrying its page, time range, region,
    /// confidence and model.
    pub fn structured(
        modality: Modality,
        extract: impl Fn(ExtractRequest) -> Result<Extraction, UnikoError> + Send + Sync + 'static,
    ) -> Self {
        Self::with_output(modality, move |req| {
            extract(req).map(ExtractOutput::Structured)
        })
    }

    /// An extractor for `modality` whose function returns either plain
    /// text or segments, decided per payload.
    pub fn with_output(
        modality: Modality,
        extract: impl Fn(ExtractRequest) -> Result<ExtractOutput, UnikoError> + Send + Sync + 'static,
    ) -> Self {
        Self {
            modality,
            deriver: Deriver::Callback(Arc::new(extract)),
            retention: MediaRetention::None,
        }
    }

    /// An extractor for `modality` backed by `models`.
    pub(crate) fn from_models(modality: Modality, models: Arc<ModelExtractor>) -> Self {
        Self {
            modality,
            deriver: Deriver::Models(models),
            retention: MediaRetention::None,
        }
    }

    /// Whether to record the original media; see [`MediaRetention`].
    #[must_use]
    pub fn with_retention(mut self, retention: MediaRetention) -> Self {
        self.retention = retention;
        self
    }

    /// The retention this extractor applies.
    #[must_use]
    pub fn retention(&self) -> MediaRetention {
        self.retention
    }
}

/// The source media, ready to record inside the transaction.
#[derive(Debug, Clone)]
struct MediaPrep {
    spec: MediaSourceSpec,
    derivation_kind: String,
}

/// The prep a [`TextExtractor`] hands to [`ModalityExtractor::apply_in_tx`]:
/// the extracted text, ready to write as an artifact, and the media source
/// when it is retained.
#[derive(Debug)]
struct TextPrep {
    text: ArtifactPrep,
    media: Option<MediaPrep>,
}

impl ModalityPrepared for TextPrep {
    fn content_ids(&self) -> Vec<String> {
        let mut ids = vec![self.text.hash.clone()];
        if let Some(m) = &self.media
            && m.spec.hash != self.text.hash
        {
            ids.push(m.spec.hash.clone());
        }
        ids
    }

    fn byte_size(&self) -> u64 {
        let media = self
            .media
            .as_ref()
            .and_then(|m| u64::try_from(m.spec.size).ok())
            .unwrap_or(0);
        u64::try_from(self.text.size).unwrap_or(0).max(media)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// What the header of `bytes` says, for the formats uniko reads itself.
fn media_header(modality: Modality, bytes: &[u8]) -> MediaHeader {
    let mut header = MediaHeader::default();
    if modality == Modality::Audio
        && let Some((_, info)) = wav_info(bytes)
    {
        header.sample_rate = i32::try_from(info.sample_rate).ok();
        header.channels = Some(i32::from(info.channels));
        header.duration_ms = i64::try_from(info.duration_ms).ok();
    }
    header
}

#[async_trait]
impl ModalityExtractor for TextExtractor {
    fn modality(&self) -> Modality {
        self.modality
    }

    fn model_aliases(&self) -> Vec<String> {
        match &self.deriver {
            Deriver::Callback(_) => Vec::new(),
            Deriver::Models(m) => m.aliases(),
        }
    }

    fn validate(&self, entries: &[ModelEntry]) -> Result<(), UnikoError> {
        match &self.deriver {
            Deriver::Callback(_) => Ok(()),
            Deriver::Models(m) => m.validate(entries),
        }
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

        // Identity of the original, before the bytes move into the request.
        let media = (self.retention != MediaRetention::None).then(|| {
            let hash = KnowledgeBase::sha256_hex(&request.bytes);
            let size = i64::try_from(request.bytes.len()).unwrap_or(i64::MAX);
            let header = media_header(self.modality, &request.bytes);
            (hash, size, header)
        });
        let media_bytes = (self.retention == MediaRetention::Bytes).then(|| request.bytes.clone());

        let output = match &self.deriver {
            Deriver::Callback(extract) => {
                let extract = Arc::clone(extract);
                tokio::task::spawn_blocking(move || extract(request))
                    .await
                    .map_err(|e| {
                        UnikoError::Internal(format!("extractor did not complete: {e}"))
                    })??
            }
            Deriver::Models(models) => {
                ExtractOutput::Structured(models.derive(kb, &request).await?)
            }
        };

        let mut metadata = src.metadata.clone();
        metadata
            .entry("mime".to_string())
            .or_insert_with(|| JsonValue::String(mime.to_string()));
        let (content, structured) = match output {
            ExtractOutput::Text(text) => (text, None),
            ExtractOutput::Structured(ex) => {
                if ex.is_empty() {
                    return Err(UnikoError::Unsupported(format!(
                        "{} extractor: no text was found",
                        modality_name(self.modality)
                    )));
                }
                (ex.text(), Some(ex))
            }
        };
        // With a recorded source, the derived artifact is identified by its
        // media, never by its text: two images that happen to get the same
        // caption stay two artifacts, each linked to its own source. Without
        // a caller id, the id is derived from the media hash, so the same
        // bytes re-ingested find the same artifact.
        let (artifact_id, caller_supplied_id) = match (&src.id, &media) {
            (Some(id), _) => (id.clone(), true),
            (None, Some((hash, _, _))) => {
                (format!("{}-{hash}", modality_name(self.modality)), true)
            }
            (None, None) => (uniko_store::id::new_id(), false),
        };
        let artifact = IngestArtifact {
            artifact_id,
            caller_supplied_id,
            content,
            kind: modality_name(self.modality).to_string(),
            path: path.clone(),
            metadata,
            session_id: None,
            triggered_by_message_id: None,
            produced_by_action_id: None,
            category: src.category.clone(),
            source_id: src.source_id.clone(),
            revision_id: src.revision_id.clone(),
        };
        let mut text = prepare_artifact(kb, &artifact).await?;
        if let Some(ex) = &structured {
            text.chunks = ex.chunks(self.modality, &ChunkConfig::from_uniko_config(kb.config()));
        }

        let media = match media {
            None => None,
            Some((hash, size, header)) => {
                let mut spec = MediaSourceSpec::new(
                    hash.clone(),
                    modality_name(self.modality),
                    mime.to_string(),
                    size,
                )
                .with_path(path)
                .with_header(header);
                if let Some(bytes) = media_bytes {
                    // PUT before the transaction, as for any artifact: the
                    // object is content-addressed, so an aborted unit leaves
                    // one the next ingest of these bytes reuses.
                    let put = kb.put_blob(&hash, &bytes).await?;
                    spec = spec.with_content(MergeContent {
                        content_id: hash,
                        bytes: put.bytes_inline,
                        uri: put.uri,
                        mime: mime.to_string(),
                        size,
                        perceptual_hash: None,
                        audio_fingerprint: None,
                    });
                }
                let derivation_kind = structured
                    .as_ref()
                    .map_or("extract", |ex| ex.derivation_kind())
                    .to_string();
                Some(MediaPrep {
                    spec,
                    derivation_kind,
                })
            }
        };
        Ok(Box::new(TextPrep { text, media }))
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
        let Some(media) = &prep.media else {
            return ingest_artifact_in_tx(kb, tx, &prep.text, ctx, seen).await;
        };
        // Same id, same media: a replay. The model may word it differently
        // this time, but the artifact already holds this media's text, so
        // keep it (attaching the bytes if this run retains them).
        let existing = match seen.get(&ArtifactIdentity::Named(prep.text.artifact_id.clone())) {
            Some((nid, _)) => Some(nid),
            None => kb
                .get_node_by_ext_id_in_tx(tx, "Artifact", "artifact_id", &prep.text.artifact_id)
                .await?
                .map(|(nid, _)| nid),
        };
        if let Some(nid) = existing {
            let hashes = kb.media_hashes_in_tx(tx, nid).await?;
            if hashes.contains(&media.spec.hash) {
                kb.merge_media_source_in_tx(tx, &media.spec).await?;
                seen.insert(
                    ArtifactIdentity::Named(prep.text.artifact_id.clone()),
                    nid,
                    prep.text.artifact_id.clone(),
                );
                link_artifact_context_in_tx(kb, tx, nid, &prep.text, ctx).await?;
                return Ok(ArtifactIngestResult {
                    artifact_node_id: nid,
                    artifact_id: prep.text.artifact_id.clone(),
                    chunk_node_ids: Vec::new(),
                    was_deduplicated: true,
                    diagnostics: Vec::new(),
                });
            }
            if !hashes.is_empty() {
                // The id already names text derived from other media.
                return Err(UnikoError::id_conflict(
                    "Artifact",
                    "artifact_id",
                    &prep.text.artifact_id,
                ));
            }
        }
        let result = ingest_artifact_in_tx(kb, tx, &prep.text, ctx, seen).await?;
        let (media_nid, _) = kb.merge_media_source_in_tx(tx, &media.spec).await?;
        kb.link_derived_from_in_tx(
            tx,
            result.artifact_node_id,
            media_nid,
            &media.derivation_kind,
        )
        .await?;
        Ok(result)
    }
}
