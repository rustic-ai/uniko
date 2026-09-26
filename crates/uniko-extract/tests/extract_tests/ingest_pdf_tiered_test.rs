//! Real-model end-to-end test of the tiered (`pdf-ocr`) PDF path.
//!
//! Builds a KB with `ocr.enabled = true`, runs `ingest_pdf` through the
//! `uni-xervo-pdf` tiered extractor, and asserts the document-IR graph
//! (`:Page` / `:Block` / child `:Chunk`) is materialized with provenance and
//! is recallable.
//!
//! Gated by both:
//! - `#[cfg(feature = "pdf-ocr")]` (compiled only with the feature), and
//! - `EXPENSIVE_TESTS=1` (building the extractor eagerly loads the OCR model,
//!   which downloads weights from HuggingFace).
//!
//! The bundled `dummy.pdf` fixture is born-digital, so its pages resolve on
//! the **Native** tier — this exercises the full tiered plumbing (extractor
//! construction, page/block materialization, child-chunk recall). Exercising
//! the OCR *recognition* tier specifically needs a scanned (image-only)
//! fixture, which is a follow-up.

use uniko_extract::ingest::pdf::{PdfIngestOptions, PdfInput, ingest_pdf};
use uniko_store::config::UnikoConfig;
use uniko_store::storage::KnowledgeBase;

fn should_run_expensive() -> bool {
    matches!(
        std::env::var("EXPENSIVE_TESTS").as_deref(),
        Ok("1") | Ok("true") | Ok("yes")
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tiered_ingest_builds_doc_ir_graph() {
    if !should_run_expensive() {
        eprintln!("skipping: set EXPENSIVE_TESTS=1 to run the tiered OCR path");
        return;
    }

    let mut config = UnikoConfig::default();
    config.ocr.enabled = true;
    let kb = KnowledgeBase::in_memory(config)
        .await
        .expect("in-memory KB with OCR enabled");

    let bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/dummy.pdf"
    ))
    .expect("read dummy.pdf fixture");

    let opts = PdfIngestOptions {
        artifact_id: "pdf-tiered-1".into(),
        caller_supplied_id: true,
        extractor: None, // ignored on the tiered path
        source_path: Some("tests/fixtures/dummy.pdf".into()),
        session_id: None,
        triggered_by_message_id: None,
        category: None,
        source_id: None,
        revision_id: None,
    };
    let result = ingest_pdf(&kb, PdfInput::Bytes(bytes), opts)
        .await
        .expect("tiered ingest_pdf");

    assert!(
        result.extraction_failure.is_none(),
        "tiered extraction should succeed, got: {:?}",
        result.extraction_failure
    );
    assert!(result.page_count >= 1, "expected at least one page");
    assert!(
        !result.page_node_ids.is_empty(),
        "tiered path must create :Page nodes"
    );
    assert!(
        !result.block_node_ids.is_empty(),
        "tiered path must create :Block nodes"
    );
    assert!(
        !result.chunk_node_ids.is_empty(),
        "tiered path must create child :Chunk nodes"
    );

    let session = kb.db().session();

    // Doc-IR shape: Artifact -HAS_PAGE-> Page -CONTAINS-> Block, blocks carry
    // a kind + provenance; born-digital fixture => native tier.
    let rows = session
        .query(
            "MATCH (a:Artifact {artifact_id: 'pdf-tiered-1'})-[:HAS_PAGE]->(p:Page) \
             -[:CONTAINS]->(b:Block) \
             RETURN b.kind AS kind, b.produced_by AS tier, b.text AS text \
             ORDER BY b.reading_order",
        )
        .await
        .expect("query blocks");
    assert!(!rows.rows().is_empty(), "expected at least one :Block");
    for r in rows.rows() {
        assert!(!r.get::<String>("kind").unwrap().is_empty());
        assert_eq!(
            r.get::<String>("tier").unwrap(),
            "native",
            "born-digital fixture resolves on the native tier"
        );
    }

    // Each block owns at least one child :Chunk(chunk_type="block").
    let rows = session
        .query(
            "MATCH (:Block)-[:HAS_CHUNK]->(c:Chunk {chunk_type: 'block'}) \
             RETURN count(c) AS n",
        )
        .await
        .expect("query block chunks");
    assert!(
        rows.rows().first().unwrap().get::<i64>("n").unwrap() >= 1,
        "expected child block chunks"
    );

    // The materialized text is recallable via the chunk filter fix.
    let recalled = kb
        .recall_chunk_and_entity_scoped(&[], "dummy", &[], 10, 0.0, 1.0, None)
        .await;
    assert!(
        recalled
            .iter()
            .any(|r| r.content.to_lowercase().contains("dummy")),
        "tiered block chunk should be recallable, got: {:?}",
        recalled.iter().map(|r| &r.content).collect::<Vec<_>>()
    );

    kb.shutdown().await.unwrap();
}
