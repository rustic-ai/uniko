//! Unified ingest entry: one [`IngestSource`] that resolves its MIME and
//! routes to the right modality pipeline.
//!
//! `Session::observe` stays separate (conversational turns carry speaker /
//! recipient / threading state); this is the front door for *blobs* —
//! documents, PDFs, and future image/audio.

use std::sync::Arc;

use serde_json::Value as JsonValue;
use uniko_pipes::content::{Mime, Modality, modality_for_mime};
use uniko_pipes::types::{IngestArtifact, IngestData, IngestSource};
use uniko_store::{KnowledgeBase, UnikoError};

use super::artifact::{ArtifactContextNids, ArtifactIngestResult, ArtifactPrep, UnitArtifactSeen};
use super::modality::{ModalityExtractor, ModalityPrepared, ModalityRegistry};
use super::pdf::{PdfIngestOptions, PdfIngestResult, PdfInput, PdfPrep};

/// What a unified ingest produced.
#[derive(Debug)]
pub enum IngestOutcome {
    /// A text/code/markup/structured/document/image/audio/video artifact.
    Artifact(ArtifactIngestResult),
    /// A PDF (tiered or legacy extraction).
    Pdf(PdfIngestResult),
}

/// Resolve the effective MIME for a payload.
///
/// Order: explicit override → magic bytes ([`infer`]) → file extension
/// ([`mime_guess`]) → `text/plain` for text payloads → `octet-stream`.
#[must_use]
pub fn resolve_mime(
    explicit: Option<&Mime>,
    bytes: Option<&[u8]>,
    path: Option<&str>,
    is_text: bool,
) -> Mime {
    if let Some(m) = explicit {
        return m.clone();
    }
    if let Some(b) = bytes
        && let Some(kind) = infer::get(b)
        && let Ok(m) = Mime::parse(kind.mime_type())
    {
        return m;
    }
    if let Some(p) = path
        && let Some(guess) = mime_guess::from_path(p).first_raw()
        && let Ok(m) = Mime::parse(guess)
    {
        return m;
    }
    if is_text {
        Mime::text_plain()
    } else {
        Mime::octet_stream()
    }
}

/// Provenance for an ingested blob: the session it was loaded into and/or
/// the message it was attached to.
///
/// `Session::ingest` sets `session_id` only; `Session::observe` sets both so
/// a conversational attachment links `Artifact -ATTACHED_TO-> Message`.
/// [`Default`] (both `None`) is the context-free streaming/corpus case.
#[derive(Debug, Default, Clone)]
pub struct IngestContext {
    /// Session the blob was loaded into.
    pub session_id: Option<String>,
    /// Message the blob was attached to (conversational provenance).
    pub triggered_by_message_id: Option<String>,
}

/// One prepared attachment, ready for in-transaction application.
///
/// There is deliberately no non-transactional variant: every attachment
/// family commits inside the unit's transaction (issue #40). Built by
/// [`prepare_source`] outside any transaction — that is where decoding,
/// model inference, chunking and blob PUTs happen — and applied by
/// [`ingest_source_in_tx`] inside it.
#[derive(Debug)]
pub enum PreparedSource {
    /// A text / code / markup / structured / document artifact.
    Artifact(Box<ArtifactPrep>),
    /// A PDF (native text-only, or tiered Native+OCR).
    Pdf(Box<PdfPrep>),
    /// A host extractor's prep, paired with the extractor that made it and
    /// must apply it.
    Modality {
        /// The extractor that produced `prep` and will write it.
        extractor: Arc<dyn ModalityExtractor>,
        /// Opaque host-defined prepared work.
        prep: Box<dyn ModalityPrepared>,
    },
}

impl PreparedSource {
    /// Every `:ArtifactContent.content_id` this attachment will merge.
    ///
    /// A unit unions these across all its turns and passes them to
    /// [`lock_ingest_unit`](uniko_store::KnowledgeBase::lock_ingest_unit)
    /// BEFORE opening its transaction, so a concurrent unit cannot
    /// interleave its merge of the same content.
    #[must_use]
    pub fn content_ids(&self) -> Vec<String> {
        match self {
            Self::Artifact(p) => vec![p.hash.clone()],
            Self::Pdf(p) => vec![p.hash.clone()],
            Self::Modality { prep, .. } => prep.content_ids(),
        }
    }

    /// Approximate byte weight, for a unit-level attachment budget.
    #[must_use]
    pub fn byte_size(&self) -> u64 {
        match self {
            Self::Artifact(p) => p.size.max(0) as u64,
            Self::Pdf(p) => p.size.max(0) as u64,
            Self::Modality { prep, .. } => prep.byte_size(),
        }
    }

    /// Apply this attachment into the caller's open transaction.
    ///
    /// Does not commit. The caller must already hold the RMW guards for
    /// [`content_ids`](Self::content_ids).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on any write failure; the caller's unit
    /// transaction is aborted and nothing from the unit persists.
    pub async fn apply_in_tx(
        &self,
        kb: &KnowledgeBase,
        tx: &uniko_store::Transaction,
        ctx: ArtifactContextNids,
        seen: &mut UnitArtifactSeen,
    ) -> Result<IngestOutcome, UnikoError> {
        match self {
            Self::Artifact(prep) => Ok(IngestOutcome::Artifact(
                super::artifact::ingest_artifact_in_tx(kb, tx, prep, ctx, seen).await?,
            )),
            Self::Pdf(prep) => Ok(IngestOutcome::Pdf(
                super::pdf::ingest_pdf_in_tx(kb, tx, prep, ctx, seen).await?,
            )),
            Self::Modality { extractor, prep } => Ok(IngestOutcome::Artifact(
                extractor
                    .apply_in_tx(kb, tx, prep.as_ref(), ctx, seen)
                    .await?,
            )),
        }
    }

    /// Best-effort work after the unit commits: mean-pooled artifact
    /// embeddings, plus any host `finish_post_commit`. Never fails the
    /// commit.
    pub async fn finish_post_commit(&self, kb: &KnowledgeBase, outcome: &IngestOutcome) {
        match (self, outcome) {
            (Self::Modality { extractor, .. }, IngestOutcome::Artifact(result)) => {
                if let Err(e) = extractor.finish_post_commit(kb, result).await {
                    tracing::warn!(
                        target: "uniko_extract::ingest",
                        error = %e,
                        "modality finish_post_commit failed; continuing",
                    );
                }
            }
            (_, IngestOutcome::Artifact(r)) if !r.chunk_node_ids.is_empty() => {
                super::artifact::pool_artifact_embeddings_post_commit(kb, &[r.artifact_node_id])
                    .await;
            }
            (_, IngestOutcome::Pdf(r)) if !r.chunk_node_ids.is_empty() => {
                super::artifact::pool_artifact_embeddings_post_commit(kb, &[r.artifact_node_id])
                    .await;
            }
            _ => {}
        }
    }
}

/// Prepare `src` for in-transaction ingest, doing all non-graph work now.
///
/// Resolves the MIME, routes by modality, and returns the work item. No
/// transaction is open and no graph write occurs, so model inference and
/// blob PUTs are never re-paid on a transaction retry.
///
/// Note the Session materialization that `ingest_source` does inline stays
/// with the *caller*: a unit runs `get_or_create_session` once, pre-tx,
/// under the setup-lock domain, and passes the resulting node id through
/// [`ArtifactContextNids`].
///
/// # Errors
///
/// Returns [`UnikoError::Unsupported`] when a non-text modality has no
/// registered extractor, or propagates the underlying preparation error.
pub async fn prepare_source(
    kb: &KnowledgeBase,
    registry: &ModalityRegistry,
    src: IngestSource,
    context: &IngestContext,
) -> Result<PreparedSource, UnikoError> {
    let is_text = matches!(src.data, IngestData::Text(_));
    let sniff_bytes = match &src.data {
        IngestData::Bytes(b) => Some(b.as_slice()),
        _ => None,
    };
    let sniff_path = src.path.as_deref().or(match &src.data {
        IngestData::Path(p) => p.to_str(),
        _ => None,
    });
    let mime = resolve_mime(src.mime.as_ref(), sniff_bytes, sniff_path, is_text);
    let modality = modality_for_mime(&mime);

    match modality {
        Modality::Pdf => {
            let input = match src.data {
                IngestData::Bytes(b) => PdfInput::Bytes(b),
                IngestData::Path(p) => PdfInput::Path(p),
                IngestData::Text(_) => {
                    return Err(UnikoError::Unsupported("text payload typed as PDF".into()));
                }
            };
            let caller_supplied_id = src.id.is_some();
            let options = PdfIngestOptions {
                artifact_id: src.id.unwrap_or_else(uniko_store::id::new_id),
                caller_supplied_id,
                extractor: None,
                source_path: src.path,
                session_id: context.session_id.clone(),
                triggered_by_message_id: context.triggered_by_message_id.clone(),
                category: src.category.clone(),
                source_id: src.source_id.clone(),
                revision_id: src.revision_id.clone(),
            };
            Ok(PreparedSource::Pdf(Box::new(
                super::pdf::prepare_pdf(kb, input, &options).await?,
            )))
        }
        Modality::Text
        | Modality::Code
        | Modality::Markup
        | Modality::Structured
        | Modality::Document => {
            let content = match src.data {
                IngestData::Text(t) => t,
                IngestData::Bytes(b) => String::from_utf8_lossy(&b).into_owned(),
                IngestData::Path(p) => std::fs::read_to_string(&p)
                    .map_err(|e| UnikoError::Storage(format!("read {}: {e}", p.display())))?,
            };
            let mut metadata = src.metadata;
            // Hint the chunker via the legacy content_type token, unless the
            // caller already set one.
            metadata
                .entry("content_type".to_string())
                .or_insert_with(|| JsonValue::String(chunker_hint(modality).to_string()));
            let caller_supplied_id = src.id.is_some();
            let artifact = IngestArtifact {
                artifact_id: src.id.unwrap_or_else(uniko_store::id::new_id),
                caller_supplied_id,
                content,
                kind: "document".to_string(),
                path: src.path,
                metadata,
                session_id: context.session_id.clone(),
                triggered_by_message_id: context.triggered_by_message_id.clone(),
                produced_by_action_id: None,
                category: src.category.clone(),
                source_id: src.source_id.clone(),
                revision_id: src.revision_id.clone(),
            };
            Ok(PreparedSource::Artifact(Box::new(
                super::artifact::prepare_artifact(kb, &artifact).await?,
            )))
        }
        Modality::Image | Modality::Audio | Modality::Video => match registry.get(modality) {
            Some(extractor) => {
                let prep = extractor.prepare(kb, &src).await?;
                Ok(PreparedSource::Modality {
                    extractor: extractor.clone(),
                    prep,
                })
            }
            None => Err(UnikoError::Unsupported(format!("{modality:?}"))),
        },
        // `Modality` is `#[non_exhaustive]`.
        _ => Err(UnikoError::Unsupported(format!("{modality:?}"))),
    }
}

/// Ingest `src`, routing by its resolved [`Modality`].
///
/// Text/Code/Markup/Structured/Document become an artifact; Pdf takes the
/// PDF path; Image/Audio/Video defer to a registered
/// [`ModalityExtractor`](super::modality::ModalityExtractor). `context`
/// carries session/message provenance set on the created artifact.
///
/// A thin wrapper over [`prepare_source`] + [`PreparedSource::apply_in_tx`]:
/// every graph write for one source lands in ONE transaction.
///
/// # Errors
///
/// Returns [`UnikoError::Unsupported`] when a non-text modality has no
/// registered extractor, or propagates the underlying ingest error.
pub async fn ingest_source(
    kb: &KnowledgeBase,
    registry: &ModalityRegistry,
    src: IngestSource,
    context: IngestContext,
) -> Result<IngestOutcome, UnikoError> {
    // Materialize the Session before ingesting into it. Only `observe`
    // created Session rows (via `ensure_session_and_sender`), so a session
    // that *only* ingests documents had no node — and the `ATTACHED_TO`
    // link is best-effort, so it was silently skipped, leaving the artifact
    // unreachable from session-scoped recall.
    if let Some(session_id) = context.session_id.as_deref() {
        super::session::get_or_create_session(kb, session_id, &chrono::Utc::now()).await?;
    }

    let prepared = prepare_source(kb, registry, src, &context).await?;
    let content_ids = prepared.content_ids();
    let _guards = kb.lock_content_ids(&content_ids).await;

    let outcome = kb
        .transact_with_retry(uniko_store::RetryOptions::default(), |tx| {
            let prepared = &prepared;
            async move {
                // Fresh per attempt: a rolled-back attempt's node ids are stale.
                let mut seen = UnitArtifactSeen::new();
                let r = prepared
                    .apply_in_tx(kb, &tx, ArtifactContextNids::default(), &mut seen)
                    .await;
                (tx, r)
            }
        })
        .await?;

    prepared.finish_post_commit(kb, &outcome).await;
    Ok(outcome)
}

/// Legacy chunker hint token for a text-family modality.
fn chunker_hint(modality: Modality) -> &'static str {
    match modality {
        Modality::Code => "code",
        Modality::Markup => "html",
        Modality::Structured => "json",
        _ => "text",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use async_trait::async_trait;
    use uniko_store::config::UnikoConfig;
    use uniko_store::storage::KnowledgeBase;

    use super::super::modality::{ModalityExtractor, ModalityPrepared, ModalityRegistry};

    #[derive(Debug)]
    struct StubImage;

    /// Minimal prep: writes nothing, so it merges no content and weighs
    /// nothing.
    #[derive(Debug)]
    struct StubImagePrep;

    impl ModalityPrepared for StubImagePrep {
        fn content_ids(&self) -> Vec<String> {
            Vec::new()
        }
        fn byte_size(&self) -> u64 {
            0
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[async_trait]
    impl ModalityExtractor for StubImage {
        fn modality(&self) -> Modality {
            Modality::Image
        }
        async fn prepare(
            &self,
            _kb: &KnowledgeBase,
            _src: &IngestSource,
        ) -> Result<Box<dyn ModalityPrepared>, UnikoError> {
            Ok(Box::new(StubImagePrep))
        }
        async fn apply_in_tx(
            &self,
            _kb: &KnowledgeBase,
            _tx: &uniko_store::Transaction,
            _prep: &dyn ModalityPrepared,
            _ctx: ArtifactContextNids,
            _seen: &mut UnitArtifactSeen,
        ) -> Result<ArtifactIngestResult, UnikoError> {
            Ok(ArtifactIngestResult {
                artifact_node_id: 42,
                artifact_id: "stub-image".into(),
                chunk_node_ids: vec![],
                was_deduplicated: false,
            })
        }
    }

    #[tokio::test]
    async fn registered_extractor_handles_its_modality() {
        let kb = KnowledgeBase::in_memory(UnikoConfig::default())
            .await
            .unwrap();
        let mut registry = ModalityRegistry::new();
        registry.register(Arc::new(StubImage));
        let png = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0];
        let outcome = ingest_source(
            &kb,
            &registry,
            IngestSource::bytes(png),
            IngestContext::default(),
        )
        .await
        .unwrap();
        match outcome {
            IngestOutcome::Artifact(r) => assert_eq!(r.artifact_node_id, 42),
            other => panic!("expected stub artifact, got {other:?}"),
        }
        kb.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn unregistered_modality_is_unsupported() {
        let kb = KnowledgeBase::in_memory(UnikoConfig::default())
            .await
            .unwrap();
        let registry = ModalityRegistry::new();
        let src = IngestSource::bytes(vec![1, 2, 3]).with_mime(Mime::parse("audio/mpeg").unwrap());
        let err = ingest_source(&kb, &registry, src, IngestContext::default())
            .await
            .unwrap_err();
        assert!(matches!(err, UnikoError::Unsupported(_)), "got {err:?}");
        kb.shutdown().await.unwrap();
    }

    #[test]
    fn explicit_mime_wins_over_sniffing() {
        let m = resolve_mime(
            Some(&Mime::parse("application/json").unwrap()),
            Some(b"%PDF-1.4"),
            Some("x.png"),
            false,
        );
        assert_eq!(m.essence(), "application/json");
    }

    #[test]
    fn sniffs_pdf_magic_bytes() {
        let m = resolve_mime(None, Some(b"%PDF-1.7\n%docs"), None, false);
        assert_eq!(m.essence(), "application/pdf");
        assert_eq!(modality_for_mime(&m), Modality::Pdf);
    }

    #[test]
    fn sniffs_png_magic_bytes() {
        let png = [0x89u8, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0];
        let m = resolve_mime(None, Some(&png), None, false);
        assert_eq!(m.type_(), "image");
        assert_eq!(modality_for_mime(&m), Modality::Image);
    }

    #[test]
    fn falls_back_to_extension() {
        let m = resolve_mime(None, None, Some("notes.md"), false);
        assert_eq!(modality_for_mime(&m), Modality::Markup);
    }

    #[test]
    fn falls_back_to_text_then_octet_stream() {
        assert_eq!(resolve_mime(None, None, None, true).essence(), "text/plain");
        let bin = resolve_mime(None, None, None, false);
        assert_eq!(bin.essence(), "application/octet-stream");
        assert_eq!(modality_for_mime(&bin), Modality::Text);
    }
}
