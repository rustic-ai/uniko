//! Tiered Native+OCR PDF extraction → document-IR graph (`pdf-ocr` feature).
//!
//! Runs `uni-xervo-pdf`'s tiered ladder with the VLM tier disabled
//! (`vlm_alias = None` plus a `Ceiling(Ocr)` policy — nothing to load, nothing
//! to route to), then materializes each [`TieredPageResult`] as typed
//! `:Page`/`:Block` nodes joined by `HAS_PAGE` / `CONTAINS` /
//! `NEXT_IN_READING_ORDER`. The embeddable, recallable text rides on a child
//! `:Chunk` (`chunk_type = "block"`) per block, attached to **both** its
//! `:Block` (structure) and the `:Artifact` (so `mean_pool_artifact_text_embedding`
//! and artifact-scoped recall keep working unchanged).

use std::collections::HashMap;

use serde_json::json;
use uni_xervo::traits::DocBlockKind;
use uni_xervo_pdf::{
    Confidence, DocExtractPolicy, DocInput, Escalation, PdfConfig, PdfExt, Tier, TieredBlock,
    TieredPageResult,
};
use uniko_store::id::{block_id, page_id};
use uniko_store::schema::OCR_ALIAS;
use uniko_store::{KnowledgeBase, NodeId, Value};

use super::super::chunking::{ChunkConfig, ChunkData, Chunker, count_tokens, text::TextChunker};
use super::super::message::{create_chunks, json_to_uni_value};
use super::extractor::PdfExtractError;

/// Node IDs produced by [`materialize_tiered`].
#[derive(Debug, Default)]
pub(super) struct TieredMaterialized {
    /// One `:Page` node per page, in page order.
    pub page_node_ids: Vec<NodeId>,
    /// Every `:Block` node, in (page, reading-order) order.
    pub block_node_ids: Vec<NodeId>,
    /// Every child `:Chunk` node (the embeddable/recall units).
    pub chunk_node_ids: Vec<NodeId>,
}

/// Run the Native+OCR tiered extractor over `bytes`.
///
/// Builds a [`PdfConfig`] bound to the `ocr/default` alias with the VLM tier
/// unbound and a `Ceiling(Ocr)` policy, then extracts every page.
///
/// # Errors
/// Returns [`PdfExtractError::Tiered`] if the model runtime is unavailable, the
/// extractor cannot be built (e.g. the OCR alias does not resolve), or
/// `uni-xervo-pdf` fails to process the document.
pub(super) async fn extract_tiered(
    kb: &KnowledgeBase,
    bytes: Vec<u8>,
) -> Result<Vec<TieredPageResult>, PdfExtractError> {
    let runtime = kb
        .model_runtime()
        .ok_or_else(|| PdfExtractError::Tiered("model runtime unavailable".to_string()))?;

    let config = PdfConfig {
        ocr_alias: Some(OCR_ALIAS.to_string()),
        vlm_alias: None,
        defaults: DocExtractPolicy::ceiling(Tier::Ocr),
    };
    let extractor = runtime
        .pdf_extractor(config)
        .await
        .map_err(|e| PdfExtractError::Tiered(e.to_string()))?;

    extractor
        .extract(
            DocInput::Pdf { bytes, pages: None },
            DocExtractPolicy::ceiling(Tier::Ocr),
        )
        .await
        .map_err(|e| PdfExtractError::Tiered(e.to_string()))
}

/// Materialize tiered pages as the `:Page`/`:Block` document-IR graph.
///
/// Creates, per page, a `:Page` (`Artifact -[:HAS_PAGE]-> Page`) and, per
/// block (sorted by reading order), a `:Block` (`Page -[:CONTAINS]-> Block`,
/// chained by `NEXT_IN_READING_ORDER`) plus its child `:Chunk`s on both the
/// block and the artifact.
///
/// # Errors
/// Returns a storage error if any node or edge write fails.
pub(super) async fn materialize_tiered(
    kb: &KnowledgeBase,
    artifact_id: &str,
    artifact_nid: NodeId,
    pages: &[TieredPageResult],
    chunk_cfg: &ChunkConfig,
) -> uniko_store::Result<TieredMaterialized> {
    let mut out = TieredMaterialized::default();

    for (page_index, page) in pages.iter().enumerate() {
        let page_ext_id = page_id(artifact_id, page.page_number);

        let mut pprops: HashMap<String, Value> = HashMap::new();
        pprops.insert("page_id".into(), Value::String(page_ext_id.clone()));
        pprops.insert(
            "page_number".into(),
            Value::Int(i64::from(page.page_number)),
        );
        pprops.insert(
            "produced_by".into(),
            Value::String(tier_str(page.produced_by).into()),
        );
        pprops.insert(
            "plain_markdown".into(),
            Value::String(page.plain_markdown.clone()),
        );
        pprops.insert("block_count".into(), Value::Int(page.blocks.len() as i64));
        if !page.escalations.is_empty() {
            pprops.insert("escalations".into(), escalations_value(&page.escalations));
        }
        let page_nid = kb.create_node("Page", &pprops).await?;
        out.page_node_ids.push(page_nid);

        let mut hp: HashMap<String, Value> = HashMap::new();
        hp.insert("index".into(), Value::Int(page_index as i64));
        kb.create_edge("HAS_PAGE", artifact_nid, page_nid, &hp)
            .await?;

        // Reading order drives both the CONTAINS index and the chain edges.
        let mut blocks: Vec<&TieredBlock> = page.blocks.iter().collect();
        blocks.sort_by_key(|b| b.reading_order);

        let mut prev_block_nid: Option<NodeId> = None;
        for block in blocks {
            let block_ext_id = block_id(&page_ext_id, block.reading_order);

            let mut bprops: HashMap<String, Value> = HashMap::new();
            bprops.insert("block_id".into(), Value::String(block_ext_id.clone()));
            bprops.insert(
                "kind".into(),
                Value::String(block_kind_str(block.kind).into()),
            );
            bprops.insert("text".into(), Value::String(block.content.clone()));
            bprops.insert(
                "reading_order".into(),
                Value::Int(i64::from(block.reading_order)),
            );
            bprops.insert(
                "produced_by".into(),
                Value::String(tier_str(block.produced_by).into()),
            );
            let (conf_kind, conf_score, conf_signals) = confidence_props(&block.confidence);
            bprops.insert("confidence_kind".into(), Value::String(conf_kind.into()));
            bprops.insert(
                "confidence_score".into(),
                Value::Float(f64::from(conf_score)),
            );
            if let Some(signals) = conf_signals {
                bprops.insert("confidence_signals".into(), signals);
            }
            if let Some([x0, y0, x1, y1]) = block.bbox {
                bprops.insert("bbox_x0".into(), Value::Float(f64::from(x0)));
                bprops.insert("bbox_y0".into(), Value::Float(f64::from(y0)));
                bprops.insert("bbox_x1".into(), Value::Float(f64::from(x1)));
                bprops.insert("bbox_y1".into(), Value::Float(f64::from(y1)));
            }
            let block_nid = kb.create_node("Block", &bprops).await?;
            out.block_node_ids.push(block_nid);

            let mut ce: HashMap<String, Value> = HashMap::new();
            ce.insert(
                "reading_order".into(),
                Value::Int(i64::from(block.reading_order)),
            );
            kb.create_edge("CONTAINS", page_nid, block_nid, &ce).await?;

            if let Some(prev) = prev_block_nid {
                kb.create_edge("NEXT_IN_READING_ORDER", prev, block_nid, &HashMap::new())
                    .await?;
            }
            prev_block_nid = Some(block_nid);

            // Child :Chunk(s): the embeddable/recall unit. Block is atomic, so
            // long content is token-split into several chunks that all hang off
            // this one block.
            let chunks = block_chunks(block, page.page_number, chunk_cfg);
            if chunks.is_empty() {
                continue;
            }
            let chunk_nids = create_chunks(kb, &block_ext_id, block_nid, &chunks, "Block").await?;
            for &chunk_nid in &chunk_nids {
                // Also attach to the Artifact (B1): keeps mean-pool and
                // artifact-scoped recall working without touching their queries.
                let mut hc: HashMap<String, Value> = HashMap::new();
                hc.insert("index".into(), Value::Int(out.chunk_node_ids.len() as i64));
                kb.create_edge("HAS_CHUNK", artifact_nid, chunk_nid, &hc)
                    .await?;
                out.chunk_node_ids.push(chunk_nid);
            }
        }
    }

    Ok(out)
}

/// Build the child `:Chunk`s for one block (one chunk, or token-split parts of
/// a long block — all carrying the same block metadata).
fn block_chunks(block: &TieredBlock, page_number: u32, config: &ChunkConfig) -> Vec<ChunkData> {
    if block.content.trim().is_empty() {
        return Vec::new();
    }
    let meta = json!({
        "page_number": page_number,
        "reading_order": block.reading_order,
        "kind": block_kind_str(block.kind),
        "produced_by": tier_str(block.produced_by),
        "confidence_score": block.confidence.score(),
    });

    let tokens = count_tokens(&block.content);
    if tokens <= config.max_chunk_tokens {
        return vec![block_chunk(
            block.content.clone(),
            0,
            0,
            block.content.len(),
            tokens,
            meta,
        )];
    }

    TextChunker
        .chunk(&block.content, config)
        .into_iter()
        .enumerate()
        .map(|(i, sub)| {
            block_chunk(
                sub.text,
                i,
                sub.start,
                sub.end,
                sub.token_count,
                meta.clone(),
            )
        })
        .collect()
}

/// Build a `chunk_type = "block"` [`ChunkData`] carrying block metadata.
fn block_chunk(
    text: String,
    index: usize,
    start: usize,
    end: usize,
    token_count: usize,
    metadata: serde_json::Value,
) -> ChunkData {
    ChunkData {
        text,
        index,
        start,
        end,
        token_count,
        chunk_type: "block".into(),
        language: None,
        symbol_name: None,
        heading: None,
        metadata: Some(metadata),
    }
}

/// Lowercase string form of a [`Tier`].
fn tier_str(tier: Tier) -> &'static str {
    match tier {
        Tier::Native => "native",
        Tier::Ocr => "ocr",
        Tier::Vlm => "vlm",
    }
}

/// Lowercase string form of a [`DocBlockKind`].
fn block_kind_str(kind: DocBlockKind) -> &'static str {
    match kind {
        DocBlockKind::Text => "text",
        DocBlockKind::Heading => "heading",
        DocBlockKind::List => "list",
        DocBlockKind::Table => "table",
        DocBlockKind::Figure => "figure",
        DocBlockKind::Formula => "formula",
        DocBlockKind::Caption => "caption",
        DocBlockKind::Footer => "footer",
        DocBlockKind::Header => "header",
        // `DocBlockKind` is `#[non_exhaustive]`; treat any future kind as text
        // rather than failing ingest.
        _ => "text",
    }
}

/// Decompose a [`Confidence`] into `(kind, score, signals_json)` for the
/// `:Block` provenance columns.
///
/// `Derived` only occurs for generative (VLM) output, which this Native+OCR
/// path never produces; it is mapped for forward-compatibility.
fn confidence_props(confidence: &Confidence) -> (&'static str, f32, Option<Value>) {
    match confidence {
        Confidence::Deterministic => ("deterministic", 1.0, None),
        Confidence::Measured(score) => ("measured", *score, None),
        Confidence::Derived { score, signals } => (
            "derived",
            *score,
            Some(Value::String(format!("{signals:?}"))),
        ),
    }
}

/// Serialize a page's escalation audit trail into a `CypherValue` JSON array.
fn escalations_value(escalations: &[Escalation]) -> Value {
    let arr: Vec<serde_json::Value> = escalations
        .iter()
        .map(|e| {
            json!({
                "from": tier_str(e.from),
                "to": tier_str(e.to),
                "reason": format!("{:?}", e.reason),
            })
        })
        .collect();
    json_to_uni_value(&serde_json::Value::Array(arr))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::chunking::ChunkConfig;

    fn block(content: &str) -> TieredBlock {
        TieredBlock {
            kind: DocBlockKind::Text,
            content: content.to_string(),
            bbox: None,
            reading_order: 0,
            produced_by: Tier::Ocr,
            confidence: Confidence::Measured(0.9),
        }
    }

    #[test]
    fn tier_str_covers_all_tiers() {
        assert_eq!(tier_str(Tier::Native), "native");
        assert_eq!(tier_str(Tier::Ocr), "ocr");
        assert_eq!(tier_str(Tier::Vlm), "vlm");
    }

    #[test]
    fn block_kind_str_maps_known_kinds() {
        assert_eq!(block_kind_str(DocBlockKind::Text), "text");
        assert_eq!(block_kind_str(DocBlockKind::Heading), "heading");
        assert_eq!(block_kind_str(DocBlockKind::Table), "table");
        assert_eq!(block_kind_str(DocBlockKind::Formula), "formula");
    }

    #[test]
    fn confidence_props_maps_each_source() {
        let (kind, score, signals) = confidence_props(&Confidence::Deterministic);
        assert_eq!(kind, "deterministic");
        assert!((score - 1.0).abs() < f32::EPSILON);
        assert!(signals.is_none());

        let (kind, score, signals) = confidence_props(&Confidence::Measured(0.73));
        assert_eq!(kind, "measured");
        assert!((score - 0.73).abs() < 1e-6);
        assert!(signals.is_none());
    }

    #[test]
    fn block_chunks_skips_blank_content() {
        let cfg = ChunkConfig::default();
        assert!(block_chunks(&block(""), 1, &cfg).is_empty());
        assert!(block_chunks(&block("   \n\t"), 1, &cfg).is_empty());
    }

    #[test]
    fn block_chunks_emits_one_for_short_content() {
        let cfg = ChunkConfig::default();
        let chunks = block_chunks(&block("a short paragraph of text"), 3, &cfg);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_type, "block");
        let meta = chunks[0].metadata.as_ref().expect("metadata");
        assert_eq!(meta["page_number"].as_u64(), Some(3));
        assert_eq!(meta["reading_order"].as_u64(), Some(0));
        assert_eq!(meta["kind"].as_str(), Some("text"));
    }

    #[test]
    fn block_chunks_splits_long_content() {
        let cfg = ChunkConfig {
            max_chunk_tokens: 3,
            min_chunk_tokens: 1,
            overlap_tokens: 0,
        };
        // `chunk_text` splits on paragraph boundaries (`\n\n`), greedily
        // accumulating up to `max_chunk_tokens`; a single paragraph is atomic.
        // Use several short paragraphs so the long block splits.
        let content = "alpha beta gamma\n\ndelta epsilon zeta\n\neta theta iota\n\nkappa lambda mu";
        let chunks = block_chunks(&block(content), 2, &cfg);
        assert!(chunks.len() > 1, "long block must split into many chunks");
        assert!(chunks.iter().all(|c| c.chunk_type == "block"));
    }

    /// Exercise the graph-write path against an in-memory KB with synthetic
    /// pages — no OCR model needed (`materialize_tiered` only writes nodes).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn materialize_tiered_writes_doc_ir_graph() {
        use uniko_store::Value;
        use uniko_store::config::UnikoConfig;
        use uniko_store::storage::KnowledgeBase;

        let kb = KnowledgeBase::in_memory(UnikoConfig::default())
            .await
            .expect("in-memory KB");

        // Minimal :Artifact to hang the doc-IR off.
        let mut aprops: HashMap<String, Value> = HashMap::new();
        aprops.insert("artifact_id".into(), Value::String("art-doc-ir".into()));
        aprops.insert("kind".into(), Value::String("pdf".into()));
        let artifact_nid = kb.create_node("Artifact", &aprops).await.unwrap();

        let pages = vec![TieredPageResult {
            page_number: 1,
            blocks: vec![
                TieredBlock {
                    kind: DocBlockKind::Heading,
                    content: "Quarterly Report".into(),
                    bbox: Some([0.0, 0.0, 1.0, 0.1]),
                    reading_order: 0,
                    produced_by: Tier::Ocr,
                    confidence: Confidence::Measured(0.95),
                },
                TieredBlock {
                    kind: DocBlockKind::Text,
                    content: "Revenue rose across all regions.".into(),
                    bbox: None,
                    reading_order: 1,
                    produced_by: Tier::Ocr,
                    confidence: Confidence::Measured(0.9),
                },
            ],
            plain_markdown: "# Quarterly Report\n\nRevenue rose across all regions.".into(),
            produced_by: Tier::Ocr,
            escalations: Vec::new(),
        }];

        let cfg = ChunkConfig::default();
        let mat = materialize_tiered(&kb, "art-doc-ir", artifact_nid, &pages, &cfg)
            .await
            .expect("materialize");

        assert_eq!(mat.page_node_ids.len(), 1);
        assert_eq!(mat.block_node_ids.len(), 2);
        assert_eq!(mat.chunk_node_ids.len(), 2);

        // Test-only provenance assertions; the seal governs product code.
        let session = kb.db().session(); // ALLOW: test-only

        // Page provenance + HAS_PAGE.
        let rows = session
            .query(
                "MATCH (a:Artifact {artifact_id: 'art-doc-ir'})-[:HAS_PAGE]->(p:Page) \
                 RETURN p.produced_by AS tier, p.block_count AS bc",
            )
            .await
            .unwrap();
        let r = rows.rows().first().expect("page row");
        assert_eq!(r.get::<String>("tier").unwrap(), "ocr");
        assert_eq!(r.get::<i64>("bc").unwrap(), 2);

        // Blocks carry kind + confidence.
        let rows = session
            .query(
                "MATCH (:Page)-[:CONTAINS]->(b:Block) \
                 RETURN b.kind AS kind, b.confidence_kind AS ck ORDER BY b.reading_order",
            )
            .await
            .unwrap();
        let blocks = rows.rows();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].get::<String>("kind").unwrap(), "heading");
        assert_eq!(blocks[0].get::<String>("ck").unwrap(), "measured");
        assert_eq!(blocks[1].get::<String>("kind").unwrap(), "text");

        // bbox lands as 4 scalar columns on the heading, NULL on the text block.
        let rows = session
            .query(
                "MATCH (b:Block {kind: 'heading'}) WHERE b.bbox_x0 IS NOT NULL \
                 RETURN count(b) AS n",
            )
            .await
            .unwrap();
        assert_eq!(
            rows.rows().first().unwrap().get::<i64>("n").unwrap(),
            1,
            "heading block has a bbox"
        );
        let rows = session
            .query("MATCH (b:Block {kind: 'text'}) WHERE b.bbox_x0 IS NULL RETURN count(b) AS n")
            .await
            .unwrap();
        assert_eq!(
            rows.rows().first().unwrap().get::<i64>("n").unwrap(),
            1,
            "text block has no bbox"
        );

        // Reading-order chain links the two blocks.
        let rows = session
            .query("MATCH (:Block)-[r:NEXT_IN_READING_ORDER]->(:Block) RETURN count(r) AS n")
            .await
            .unwrap();
        assert_eq!(rows.rows().first().unwrap().get::<i64>("n").unwrap(), 1);

        // Each block owns a child :Chunk(chunk_type="block"), also linked to the
        // artifact so mean-pool / artifact recall see it.
        let rows = session
            .query(
                "MATCH (b:Block)-[:HAS_CHUNK]->(c:Chunk {chunk_type: 'block'}) \
                 RETURN count(c) AS n",
            )
            .await
            .unwrap();
        assert_eq!(rows.rows().first().unwrap().get::<i64>("n").unwrap(), 2);

        let rows = session
            .query(
                "MATCH (:Artifact {artifact_id: 'art-doc-ir'})-[:HAS_CHUNK]->\
                 (c:Chunk {chunk_type: 'block'}) RETURN count(c) AS n",
            )
            .await
            .unwrap();
        assert_eq!(rows.rows().first().unwrap().get::<i64>("n").unwrap(), 2);

        kb.shutdown().await.unwrap();
    }
}
