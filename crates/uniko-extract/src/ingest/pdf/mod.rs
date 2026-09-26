//! PDF ingest — text-only path with a pluggable extractor.
//!
//! See the multimodal design notes §5.3.4.
//!
//! Rasterization for VLM is deliberately out of scope: no production-
//! ready pure-Rust PDF rasterizer exists today, and pdfium / mupdf /
//! poppler are C/C++ FFI. The graph shape produced here is forward-
//! compatible with that future path — page chunks already carry
//! `metadata.page_number`, so adding rendered-page derivations later
//! does not require re-chunking.

pub mod chunker;
pub mod extractor;
pub mod input;
#[cfg(feature = "pdf-ocr")]
mod tiered;

pub use chunker::{PdfPageChunker, chunk_pages};
pub use extractor::{ExtractedPage, PdfExtractCrate, PdfExtractError, PdfTextExtractor};
pub use input::{PdfIngestOptions, PdfInput};

use std::collections::HashMap;
use std::sync::Arc;

use uniko_store::Value;

use super::artifact::{ArtifactContextNids, ArtifactIdentity, UnitArtifactSeen};
use uniko_store::storage::blob::MergeContent;
use uniko_store::{KnowledgeBase, NodeId};

use super::chunking::ChunkConfig;
use super::message::create_chunks_in_tx;

/// Result of [`ingest_pdf`].
#[derive(Debug)]
pub struct PdfIngestResult {
    /// Internal node ID of the `:Artifact{kind="pdf"}` row.
    pub artifact_node_id: NodeId,
    /// External artifact id — pass to `agent.data().artifact(..)` to fetch
    /// it back.
    pub artifact_id: String,
    /// Internal node IDs of created chunks (empty on extractor failure).
    pub chunk_node_ids: Vec<NodeId>,
    /// Number of pages reported by the extractor. Zero on failure.
    pub page_count: u32,
    /// `true` if the artifact already existed by hash or `artifact_id`.
    pub was_deduplicated: bool,
    /// `None` on success, `Some(err)` if extraction failed but the
    /// artifact + content were still persisted.
    pub extraction_failure: Option<PdfExtractError>,
    /// Internal node IDs of created `:Page` rows (tiered path only; empty
    /// on the text-only path).
    pub page_node_ids: Vec<NodeId>,
    /// Internal node IDs of created `:Block` rows (tiered path only; empty
    /// on the text-only path).
    pub block_node_ids: Vec<NodeId>,
}

/// Everything about a PDF that is computable before the transaction opens:
/// the bytes, their hash, the extracted pages (native or tiered Native+OCR),
/// the chunk plan, and the blob PUT outcome.
///
/// Built by [`prepare_pdf`] and consumed **by reference** by
/// [`ingest_pdf_in_tx`]. The split is what keeps OCR out of the transaction
/// window: DBNet + CRNN over every page is the dominant cost of this path and
/// must never be re-paid on an SSI retry.
#[derive(Debug)]
pub struct PdfPrep {
    /// External id this artifact will be created under.
    pub artifact_id: String,
    /// Whether `artifact_id` is the caller's identity or an auto-generated one.
    pub caller_supplied_id: bool,
    /// SHA-256 of the PDF bytes, hex-encoded. The `:ArtifactContent` key.
    pub hash: String,
    /// Byte length.
    pub size: i64,
    /// Original path / URL, when recorded.
    pub source_path: Option<String>,
    /// Pages found by extraction (0 when extraction failed).
    pub page_count: u32,
    /// Blob backend PUT outcome. Already persisted — see [`prepare_artifact`]
    /// for why this is pre-transaction and what an aborted unit leaves behind.
    ///
    /// [`prepare_artifact`]: super::artifact::prepare_artifact
    pub put: uniko_store::blob_store::PutOutcome,
    /// Extraction error, when the artifact must persist without doc-IR.
    pub extraction_failure: Option<PdfExtractError>,
    /// Session provenance, resolved in-tx when the caller passes no nid.
    pub session_id: Option<String>,
    /// Message provenance, resolved in-tx when the caller passes no nid.
    pub triggered_by_message_id: Option<String>,
    /// True when the tiered Native+OCR doc-IR path produced the pages.
    pub use_tiered: bool,
    /// Tiered doc-IR pages (`pdf-ocr` builds only).
    #[cfg(feature = "pdf-ocr")]
    pub tiered_pages: Vec<uni_xervo_pdf::TieredPageResult>,
    /// Text-only pages, used when the tiered path is off or unavailable.
    pub legacy_pages: Vec<ExtractedPage>,
    /// Caller's record category (issue #39).
    pub category: Option<String>,
    /// Logical source id (issue #39).
    pub source_id: Option<String>,
    /// Revision identity for these bytes (issue #41).
    pub revision_id: Option<String>,
}

/// The chunk provenance a PDF's chunks inherit from it.
fn pdf_prov(prep: &PdfPrep) -> super::message::ChunkProvenance<'_> {
    super::message::ChunkProvenance {
        category: prep.category.as_deref(),
        source_id: prep.source_id.as_deref(),
        revision_id: prep.revision_id.as_deref(),
    }
}

impl PdfPrep {
    /// How this PDF is identified within a unit — the same split as
    /// [`ingest_artifact`](super::artifact::ingest_artifact).
    fn identity(&self) -> ArtifactIdentity {
        if self.caller_supplied_id {
            ArtifactIdentity::Named(self.artifact_id.clone())
        } else {
            ArtifactIdentity::Content(self.hash.clone())
        }
    }
}

/// Pre-transaction phase of PDF ingest: read, hash, extract, chunk, PUT.
///
/// Performs no graph writes and holds no locks. [`PdfPrep::hash`] is the key
/// the caller passes to
/// [`lock_ingest_unit`](uniko_store::KnowledgeBase::lock_ingest_unit) — or
/// [`lock_content_ids`](uniko_store::KnowledgeBase::lock_content_ids) for a
/// lone PDF — before opening the transaction.
///
/// Extraction failures do **not** propagate: the artifact and its content
/// still persist, and the error lands in [`PdfPrep::extraction_failure`],
/// matching the previous behaviour.
///
/// # Errors
///
/// Returns [`UnikoError::Pipeline`](uniko_store::UnikoError::Pipeline) when
/// `artifact_id` is empty, or
/// [`UnikoError::Storage`](uniko_store::UnikoError::Storage) if the bytes
/// cannot be read or the blob PUT fails.
pub async fn prepare_pdf(
    kb: &KnowledgeBase,
    input: PdfInput,
    opts: &PdfIngestOptions,
) -> uniko_store::Result<PdfPrep> {
    // 1. Materialize bytes.
    let bytes: Vec<u8> = match input {
        PdfInput::Bytes(b) => b,
        PdfInput::Path(p) => tokio::fs::read(&p)
            .await
            .map_err(|e| uniko_store::UnikoError::Storage(format!("read {}: {e}", p.display())))?,
    };

    if opts.artifact_id.is_empty() {
        return Err(uniko_store::UnikoError::Pipeline(
            "PdfIngestOptions.artifact_id must be non-empty".into(),
        ));
    }

    let hash = KnowledgeBase::sha256_hex(&bytes);
    let size = bytes.len() as i64;

    // 2. Extraction, up-front so `page_count` is known for the Artifact row.
    //    Tiered Native+OCR (doc-IR graph) when the `pdf-ocr` feature is built,
    //    OCR is enabled, and a model runtime exists; otherwise pure-Rust
    //    text-only. Both are pure CPU / model work with no graph writes, which
    //    is exactly why they belong out here.
    let use_tiered =
        cfg!(feature = "pdf-ocr") && kb.config().ocr.enabled && kb.model_runtime().is_some();

    let mut extraction_failure: Option<PdfExtractError> = None;

    #[cfg(feature = "pdf-ocr")]
    let mut tiered_pages: Vec<uni_xervo_pdf::TieredPageResult> = Vec::new();
    let mut legacy_pages: Vec<ExtractedPage> = Vec::new();

    #[cfg(feature = "pdf-ocr")]
    if use_tiered {
        match tiered::extract_tiered(kb, bytes.clone()).await {
            Ok(p) => tiered_pages = p,
            Err(e) => {
                tracing::warn!(
                    target: "uniko_extract::ingest::pdf",
                    artifact_id = %opts.artifact_id,
                    error = %e,
                    "tiered pdf extraction failed; persisting artifact without doc-IR",
                );
                extraction_failure = Some(e);
            }
        }
    }

    if !use_tiered {
        let extractor: Arc<dyn PdfTextExtractor> = opts
            .extractor
            .clone()
            .unwrap_or_else(|| Arc::new(PdfExtractCrate));
        match extractor.extract(&bytes) {
            Ok(p) => legacy_pages = p,
            Err(e) => {
                tracing::warn!(
                    target: "uniko_extract::ingest::pdf",
                    artifact_id = %opts.artifact_id,
                    error = %e,
                    "pdf extraction failed; persisting artifact without chunks",
                );
                extraction_failure = Some(e);
            }
        }
    }

    #[cfg(feature = "pdf-ocr")]
    let page_count = if use_tiered {
        tiered_pages.len() as u32
    } else {
        legacy_pages.len() as u32
    };
    #[cfg(not(feature = "pdf-ocr"))]
    let page_count = legacy_pages.len() as u32;

    // 3. Persist the bytes via the blob backend.
    let put = kb.put_blob(&hash, &bytes).await?;

    Ok(PdfPrep {
        artifact_id: opts.artifact_id.clone(),
        caller_supplied_id: opts.caller_supplied_id,
        hash,
        size,
        source_path: opts.source_path.clone(),
        page_count,
        put,
        extraction_failure,
        session_id: opts.session_id.clone(),
        triggered_by_message_id: opts.triggered_by_message_id.clone(),
        use_tiered,
        #[cfg(feature = "pdf-ocr")]
        tiered_pages,
        legacy_pages,
        category: opts.category.clone(),
        source_id: opts.source_id.clone(),
        revision_id: opts.revision_id.clone(),
    })
}

/// Write a prepared PDF into the caller's open transaction. Does NOT commit.
///
/// `ctx` carries already-resolved node ids for the Session / Message / Action
/// this PDF is attached to. In a multi-turn unit the Message is created inside
/// this same uncommitted transaction, so those ids must come from the unit —
/// resolving them beforehand on a fresh session would miss, and a miss here is
/// silent by design.
///
/// `seen` memoizes per-unit identity so the same PDF attached to two turns
/// yields one `:Artifact` while still wiring both `ATTACHED_TO` edges.
///
/// # Preconditions
///
/// The caller holds the `content:<prep.hash>` RMW guard, taken before `tx` was
/// opened and held until after `tx.commit()`.
///
/// # Post-commit
///
/// `Artifact.text_embedding` is not populated here — pooling reads committed
/// Chunk embeddings through a fresh session. Run
/// [`pool_artifact_embeddings_post_commit`](super::artifact::pool_artifact_embeddings_post_commit)
/// after the commit.
///
/// # Errors
///
/// Returns [`UnikoError::IdConflict`](uniko_store::UnikoError::IdConflict)
/// when a caller-supplied id is reused for different bytes, or a storage error
/// on any graph write.
pub async fn ingest_pdf_in_tx(
    kb: &KnowledgeBase,
    tx: &uniko_store::Transaction,
    prep: &PdfPrep,
    ctx: ArtifactContextNids,
    seen: &mut UnitArtifactSeen,
) -> uniko_store::Result<PdfIngestResult> {
    let dedup_result = |nid: NodeId, ext_id: String| PdfIngestResult {
        artifact_node_id: nid,
        artifact_id: ext_id,
        chunk_node_ids: Vec::new(),
        page_count: 0,
        was_deduplicated: true,
        extraction_failure: None,
        page_node_ids: Vec::new(),
        block_node_ids: Vec::new(),
    };

    // 1. Per-unit memo, before any DB read.
    if let Some((nid, under_id)) = seen.get(&prep.identity()) {
        link_pdf_context_in_tx(kb, tx, nid, prep, ctx).await?;
        return Ok(dedup_result(
            nid,
            if prep.caller_supplied_id {
                prep.artifact_id.clone()
            } else {
                under_id
            },
        ));
    }

    // 2. Authoritative identity probe, INSIDE the transaction so it sees both
    //    committed rows and anything this unit already wrote.
    let existing = if prep.caller_supplied_id {
        let found = kb
            .get_node_by_ext_id_in_tx(tx, "Artifact", "artifact_id", &prep.artifact_id)
            .await?;
        if let Some((_, ref props)) = found {
            let stored = match props.get("hash") {
                Some(uniko_store::Value::String(s)) => s.as_str(),
                _ => "",
            };
            if stored != prep.hash {
                return Err(uniko_store::UnikoError::id_conflict(
                    "Artifact",
                    "artifact_id",
                    &prep.artifact_id,
                ));
            }
        }
        found
    } else {
        kb.get_node_by_ext_id_in_tx(tx, "Artifact", "hash", &prep.hash)
            .await?
    };

    if let Some((existing_id, existing_props)) = existing {
        let existing_ext_id = match existing_props.get("artifact_id") {
            Some(uniko_store::Value::String(s)) => s.clone(),
            _ => prep.artifact_id.clone(),
        };
        seen.insert(prep.identity(), existing_id, existing_ext_id.clone());
        link_pdf_context_in_tx(kb, tx, existing_id, prep, ctx).await?;
        return Ok(dedup_result(existing_id, existing_ext_id));
    }

    // 3. MERGE :ArtifactContent, memoized per unit so two turns sharing bytes
    //    under different caller ids still converge on one content row.
    let content_nid = match seen.content_nid(&prep.hash) {
        Some(nid) => nid,
        None => {
            let nid = kb
                .merge_artifact_content_in_tx(
                    tx,
                    MergeContent {
                        content_id: prep.hash.clone(),
                        bytes: prep.put.bytes_inline.clone(),
                        uri: prep.put.uri.clone(),
                        mime: "application/pdf".into(),
                        size: prep.size,
                        perceptual_hash: None,
                        audio_fingerprint: None,
                    },
                )
                .await?;
            seen.insert_content(prep.hash.clone(), nid);
            nid
        }
    };

    // 4. :Artifact{kind="pdf"} metadata node.
    let mut props: HashMap<String, Value> = HashMap::new();
    props.insert(
        "artifact_id".into(),
        Value::String(prep.artifact_id.clone()),
    );
    props.insert("kind".into(), Value::String("pdf".into()));
    if let Some(ref path) = prep.source_path {
        props.insert("path".into(), Value::String(path.clone()));
    }
    props.insert("hash".into(), Value::String(prep.hash.clone()));
    props.insert("size".into(), Value::Int(prep.size));
    props.insert("page_count".into(), Value::Int(i64::from(prep.page_count)));
    let artifact_nid = kb.create_node_in_tx(tx, "Artifact", &props).await?;
    seen.insert(prep.identity(), artifact_nid, prep.artifact_id.clone());

    // 5. HAS_CONTENT edge.
    let mut edge_props: HashMap<String, Value> = HashMap::new();
    edge_props.insert("role".into(), Value::String("primary".into()));
    kb.create_edges_in_tx(
        tx,
        &[("HAS_CONTENT", artifact_nid, content_nid, edge_props)],
    )
    .await?;

    // 6. Conversational provenance.
    link_pdf_context_in_tx(kb, tx, artifact_nid, prep, ctx).await?;

    // 7. Chunks. Tiered builds the :Page/:Block doc-IR (each block owns child
    //    :Chunks); legacy emits one :Chunk per page. Both attach chunks to the
    //    Artifact so mean-pool and recall work uniformly.
    let chunk_cfg = ChunkConfig::from_uniko_config(kb.config());
    #[allow(unused_mut)]
    let mut page_node_ids: Vec<NodeId> = Vec::new();
    #[allow(unused_mut)]
    let mut block_node_ids: Vec<NodeId> = Vec::new();

    #[cfg(feature = "pdf-ocr")]
    let chunk_nids = if prep.use_tiered {
        let mat = tiered::materialize_tiered_in_tx(
            kb,
            tx,
            &prep.artifact_id,
            artifact_nid,
            &prep.tiered_pages,
            &chunk_cfg,
            pdf_prov(prep),
        )
        .await?;
        page_node_ids = mat.page_node_ids;
        block_node_ids = mat.block_node_ids;
        mat.chunk_node_ids
    } else {
        legacy_chunks_in_tx(
            kb,
            tx,
            &prep.artifact_id,
            artifact_nid,
            &prep.legacy_pages,
            &chunk_cfg,
            pdf_prov(prep),
        )
        .await?
    };
    #[cfg(not(feature = "pdf-ocr"))]
    let chunk_nids = legacy_chunks_in_tx(
        kb,
        tx,
        &prep.artifact_id,
        artifact_nid,
        &prep.legacy_pages,
        &chunk_cfg,
        pdf_prov(prep),
    )
    .await?;

    Ok(PdfIngestResult {
        artifact_node_id: artifact_nid,
        artifact_id: prep.artifact_id.clone(),
        chunk_node_ids: chunk_nids,
        page_count: prep.page_count,
        was_deduplicated: false,
        extraction_failure: None,
        page_node_ids,
        block_node_ids,
    })
}

/// Ingest a PDF document into the knowledge graph.
///
/// Thin wrapper over [`prepare_pdf`] + [`ingest_pdf_in_tx`]: every graph write
/// now lands in ONE transaction, so a failure part-way through can no longer
/// leave a committed Artifact with no chunks. For the tiered doc-IR path this
/// also collapses what was O(2·pages + 4·blocks + chunks) separate commits
/// into a single one.
///
/// # Errors
///
/// Returns [`UnikoError::IdConflict`](uniko_store::UnikoError::IdConflict)
/// when a caller-supplied id is reused for different bytes, or a storage error
/// if any graph operation fails.
pub async fn ingest_pdf(
    kb: &KnowledgeBase,
    input: PdfInput,
    opts: PdfIngestOptions,
) -> uniko_store::Result<PdfIngestResult> {
    let prep = prepare_pdf(kb, input, &opts).await?;
    let _guards = kb.lock_content_ids(std::slice::from_ref(&prep.hash)).await;

    let mut result = kb
        .transact_with_retry(uniko_store::RetryOptions::default(), |tx| {
            let prep = &prep;
            async move {
                // Fresh per attempt: a rolled-back attempt's node ids are stale.
                let mut seen = UnitArtifactSeen::new();
                let r = ingest_pdf_in_tx(kb, &tx, prep, ArtifactContextNids::default(), &mut seen)
                    .await;
                (tx, r)
            }
        })
        .await?;

    // Surface the extraction failure the prep recorded, as before.
    if result.extraction_failure.is_none() && !result.was_deduplicated {
        result.extraction_failure = prep.extraction_failure;
    }

    if !result.chunk_node_ids.is_empty() {
        super::artifact::pool_artifact_embeddings_post_commit(kb, &[result.artifact_node_id]).await;
    }
    Ok(result)
}

/// Emit one `:Chunk` per page (text-only path) and link them to the artifact,
/// inside the caller's transaction.
async fn legacy_chunks_in_tx(
    kb: &KnowledgeBase,
    tx: &uniko_store::Transaction,
    artifact_id: &str,
    artifact_nid: NodeId,
    pages: &[ExtractedPage],
    chunk_cfg: &ChunkConfig,
    prov: super::message::ChunkProvenance<'_>,
) -> uniko_store::Result<Vec<NodeId>> {
    let chunks = chunk_pages(pages, chunk_cfg);
    if chunks.is_empty() {
        return Ok(Vec::new());
    }
    create_chunks_in_tx(kb, tx, artifact_id, artifact_nid, &chunks, "Artifact", prov).await
}

/// Link a PDF artifact to the Session/Message/Action it belongs to, inside
/// the caller's transaction.
///
/// Takes already-resolved node ids from `ctx` where available and falls back
/// to an **in-transaction** ext-id lookup otherwise. Two fixes over the
/// previous `link_pdf_context`: a lookup miss is now logged at debug instead
/// of vanishing with no trace at all, and `ATTACHED_TO` carries `attached_at`
/// like [`link_artifact_context_in_tx`](super::artifact) does.
async fn link_pdf_context_in_tx(
    kb: &KnowledgeBase,
    tx: &uniko_store::Transaction,
    artifact_nid: NodeId,
    prep: &PdfPrep,
    ctx: ArtifactContextNids,
) -> uniko_store::Result<()> {
    use uniko_store::schema::{edges, labels};

    let attach_props = || {
        let mut p: HashMap<String, Value> = HashMap::new();
        p.insert(
            "attached_at".into(),
            uniko_store::types::datetime_value(chrono::Utc::now()),
        );
        p
    };

    let mut pending: Vec<(&str, NodeId, NodeId, HashMap<String, Value>)> = Vec::new();

    for (given, ext_id, label, id_field, what) in [
        (
            ctx.session_nid,
            prep.session_id.as_deref(),
            labels::SESSION,
            "session_id",
            "Session",
        ),
        (
            ctx.message_nid,
            prep.triggered_by_message_id.as_deref(),
            labels::MESSAGE,
            "message_id",
            "triggering Message",
        ),
    ] {
        let resolved = match given {
            Some(nid) => Some(nid),
            None => match ext_id {
                Some(id) => match kb.get_node_by_ext_id_in_tx(tx, label, id_field, id).await? {
                    Some((nid, _)) => Some(nid),
                    None => {
                        tracing::debug!(
                            ext_id = id,
                            "ingest_pdf: {what} not found — ATTACHED_TO skipped"
                        );
                        None
                    }
                },
                None => None,
            },
        };
        if let Some(nid) = resolved {
            pending.push((edges::ATTACHED_TO, artifact_nid, nid, attach_props()));
        }
    }

    // PRODUCED: Action → Artifact. Only a caller-supplied nid can drive this —
    // `PdfIngestOptions` carries no action id, so there is nothing to resolve.
    if let Some(action_nid) = ctx.action_nid {
        pending.push((edges::PRODUCED, action_nid, artifact_nid, HashMap::new()));
    }

    if !pending.is_empty() {
        kb.create_edges_in_tx(tx, &pending).await?;
    }
    Ok(())
}
