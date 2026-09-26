//! Extension point for non-text modalities (image / audio / video).
//!
//! Text, code, markup, structured, and PDF ingest are built in. Image,
//! audio, and video are routed to a [`ModalityExtractor`] registered in a
//! [`ModalityRegistry`]; with no extractor registered, ingest of those
//! modalities returns [`UnikoError::Unsupported`]. No ASR/VLM/captioning
//! model ships here — this is the seam a host plugs one into.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use uniko_pipes::content::Modality;
use uniko_store::{KnowledgeBase, UnikoError};

use uniko_pipes::types::IngestSource;

use super::artifact::{ArtifactContextNids, ArtifactIngestResult, UnitArtifactSeen};

/// An extractor's prepared work: everything computed outside the
/// transaction, ready to be turned into graph writes.
///
/// Opaque to uniko — the extractor defines the concrete type. uniko only
/// needs the two questions below answered before it opens the transaction.
pub trait ModalityPrepared: Send + Sync + std::fmt::Debug {
    /// Every `:ArtifactContent.content_id` (SHA-256 hex) this prep will
    /// merge, including derived content such as a thumbnail or a
    /// transcript.
    ///
    /// Feeds the unit's pre-transaction
    /// [`lock_ingest_unit`](uniko_store::KnowledgeBase::lock_ingest_unit)
    /// call. Omitting one risks a duplicate `:ArtifactContent` row under
    /// concurrent ingest, because the merge is a check-then-create that
    /// SSI does not protect (an insert phantom registers no read-set
    /// conflict).
    fn content_ids(&self) -> Vec<String>;

    /// Approximate byte weight of this attachment. Return the source
    /// payload size when in doubt.
    fn byte_size(&self) -> u64;

    /// Downcast hook so [`ModalityExtractor::apply_in_tx`] can recover its
    /// own concrete prep type. Implement as `fn as_any(&self) -> &dyn
    /// std::any::Any { self }`.
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Extracts a single non-text modality into the graph as an artifact.
///
/// Implement for a concrete modality (e.g. an image captioner) and register
/// it in a [`ModalityRegistry`]; the unified ingest dispatch then routes
/// matching MIME types to it.
///
/// The work is split in two so that a multi-turn unit can commit every
/// message and every attachment in ONE transaction:
///
/// - [`prepare`](Self::prepare) runs **outside** any transaction. Do all
///   decoding, model inference, captioning, transcription, and blob PUTs
///   here.
/// - [`apply_in_tx`](Self::apply_in_tx) runs **inside** the unit's
///   transaction and must perform graph writes only, through
///   `KnowledgeBase`'s `*_in_tx` methods. It must not commit, must not open
///   its own transaction, and may be **re-invoked on a retry**, so it must
///   be a pure function of `prep` + `ctx` + `seen`.
///
/// Splitting it this way is not stylistic: the unit must know every
/// `content_id` it will write *before* it opens the transaction, so it can
/// take the striped locks first. A single `extract(kb, tx, ..)` gives it no
/// point at which to ask.
#[async_trait]
pub trait ModalityExtractor: Send + Sync + std::fmt::Debug {
    /// The modality this extractor handles.
    fn modality(&self) -> Modality;

    /// Do every non-graph step for `src`: decode, run models, chunk, and
    /// PUT blobs via [`KnowledgeBase::put_blob`]. No transaction is open.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on a decode, inference, or blob failure.
    async fn prepare(
        &self,
        kb: &KnowledgeBase,
        src: &IngestSource,
    ) -> Result<Box<dyn ModalityPrepared>, UnikoError>;

    /// Write `prep` into the caller's transaction. Does NOT commit.
    ///
    /// `ctx` carries already-resolved [`NodeId`](uniko_store::NodeId)s for
    /// the Session, Message, and Action this attachment belongs to. Resolve
    /// nothing by external id on a fresh session here: those rows are
    /// created inside this same uncommitted transaction and a committed
    /// read cannot see them, so the edge would be dropped **silently**. Use
    /// [`get_node_by_ext_id_in_tx`](uniko_store::KnowledgeBase::get_node_by_ext_id_in_tx)
    /// if you must look something up.
    ///
    /// `seen` is the unit-wide artifact/content memo: consult and update it
    /// so the same bytes attached to two turns yield one `:Artifact`.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on a write failure. A retriable error aborts
    /// the whole unit and this method is called again on a fresh
    /// transaction with the same `prep`.
    async fn apply_in_tx(
        &self,
        kb: &KnowledgeBase,
        tx: &uniko_store::Transaction,
        prep: &dyn ModalityPrepared,
        ctx: ArtifactContextNids,
        seen: &mut UnitArtifactSeen,
    ) -> Result<ArtifactIngestResult, UnikoError>;

    /// Optional work after the unit has committed — pooled embeddings,
    /// derived-asset backfill.
    ///
    /// This exists because `Artifact.image_embedding` / `audio_embedding` /
    /// `video_embedding` have no `embedding_config`: they are host-computed
    /// and written through a self-committing `update_node`, so they can
    /// never be part of the unit's transaction.
    ///
    /// Failures here must not invalidate the commit; uniko logs them at
    /// `warn` and continues. Defaults to a no-op.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] only if the host wants the failure surfaced.
    async fn finish_post_commit(
        &self,
        _kb: &KnowledgeBase,
        _result: &ArtifactIngestResult,
    ) -> Result<(), UnikoError> {
        Ok(())
    }
}

/// A registry of [`ModalityExtractor`]s keyed by [`Modality`].
///
/// The unified ingest dispatch consults this for Image/Audio/Video; an empty
/// registry (the [`Default`]) yields [`UnikoError::Unsupported`] for them.
#[derive(Debug, Default, Clone)]
pub struct ModalityRegistry {
    extractors: HashMap<Modality, Arc<dyn ModalityExtractor>>,
}

impl ModalityRegistry {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `extractor` for its declared [`Modality`], replacing any
    /// previous entry for that modality.
    pub fn register(&mut self, extractor: Arc<dyn ModalityExtractor>) {
        self.extractors.insert(extractor.modality(), extractor);
    }

    /// The extractor for `modality`, if registered.
    #[must_use]
    pub fn get(&self, modality: Modality) -> Option<&Arc<dyn ModalityExtractor>> {
        self.extractors.get(&modality)
    }
}
