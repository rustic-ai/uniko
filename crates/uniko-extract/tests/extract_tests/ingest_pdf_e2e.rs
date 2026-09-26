//! Integration tests for `uniko_extract::ingest::pdf::ingest_pdf`.
//!
//! Uses a `MockExtractor` for the success path so the test is
//! independent of the underlying `pdf-extract` crate's parsing quirks.
//! The real-PDF round-trip is covered by extractor unit tests
//! (panic-safety) and by manual sanity checks on real documents.

use std::sync::Arc;

use uniko_extract::ingest::pdf::{
    ExtractedPage, PdfExtractError, PdfIngestOptions, PdfInput, PdfTextExtractor, ingest_pdf,
};
use uniko_store::config::UnikoConfig;
use uniko_store::storage::KnowledgeBase;

async fn test_kb() -> KnowledgeBase {
    KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB")
}

/// Fake extractor that returns hard-coded pages.
struct MockExtractor {
    pages: Vec<ExtractedPage>,
}

impl PdfTextExtractor for MockExtractor {
    fn extract(&self, _bytes: &[u8]) -> Result<Vec<ExtractedPage>, PdfExtractError> {
        Ok(self.pages.clone())
    }
}

/// Fake extractor that always fails.
struct FailingExtractor;

impl PdfTextExtractor for FailingExtractor {
    fn extract(&self, _bytes: &[u8]) -> Result<Vec<ExtractedPage>, PdfExtractError> {
        Err(PdfExtractError::Parse("synthetic failure".into()))
    }
}

fn mock_opts(artifact_id: &str, pages: Vec<ExtractedPage>) -> PdfIngestOptions {
    PdfIngestOptions {
        artifact_id: artifact_id.into(),
        caller_supplied_id: true,
        extractor: Some(Arc::new(MockExtractor { pages })),
        source_path: None,
        session_id: None,
        triggered_by_message_id: None,
        category: None,
        source_id: None,
        revision_id: None,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ingest_pdf_creates_artifact_content_and_chunks() {
    let kb = test_kb().await;
    let pages = vec![
        ExtractedPage {
            page_number: 1,
            text: "First page text.".into(),
        },
        ExtractedPage {
            page_number: 2,
            text: "Second page content.".into(),
        },
        ExtractedPage {
            page_number: 3,
            text: "Third and final.".into(),
        },
    ];
    let bytes = b"fake pdf bytes 1".to_vec();
    let result = ingest_pdf(
        &kb,
        PdfInput::Bytes(bytes.clone()),
        mock_opts("pdf-art-1", pages),
    )
    .await
    .expect("ingest_pdf");

    assert!(!result.was_deduplicated);
    assert_eq!(result.page_count, 3);
    assert!(result.extraction_failure.is_none());
    assert_eq!(result.chunk_node_ids.len(), 3);

    // :Artifact{kind="pdf"} exists with expected page_count.
    let session = kb.db().session();
    let rows = session
        .query(
            "MATCH (a:Artifact {artifact_id: 'pdf-art-1'}) \
             RETURN a.kind AS kind, a.page_count AS pc, a.size AS sz",
        )
        .await
        .expect("query artifact");
    let row = rows.rows().first().expect("artifact row");
    assert_eq!(row.get::<String>("kind").unwrap(), "pdf");
    assert_eq!(row.get::<i64>("pc").unwrap(), 3);
    assert_eq!(row.get::<i64>("sz").unwrap(), bytes.len() as i64);

    // :ArtifactContent exists with application/pdf MIME.
    let rows = session
        .query(
            "MATCH (a:Artifact {artifact_id: 'pdf-art-1'})-[:HAS_CONTENT]->(c:ArtifactContent) \
             RETURN c.mime AS mime",
        )
        .await
        .expect("query content");
    let row = rows.rows().first().expect("content row");
    assert_eq!(row.get::<String>("mime").unwrap(), "application/pdf");

    // :Chunk rows tagged chunk_type="page" with metadata.page_number set.
    let rows = session
        .query(
            "MATCH (a:Artifact {artifact_id: 'pdf-art-1'})-[:HAS_CHUNK]->(c:Chunk) \
             RETURN c.text AS text, c.chunk_type AS ct, c.metadata AS md \
             ORDER BY c.index",
        )
        .await
        .expect("query chunks");
    assert_eq!(rows.rows().len(), 3);
    for (i, r) in rows.rows().iter().enumerate() {
        assert_eq!(r.get::<String>("ct").unwrap(), "page");
        // metadata round-trips as a Map.
        let md = r.value("md").expect("metadata present");
        let map = md.as_object().expect("metadata is a map");
        let page_number = map.get("page_number").and_then(|v| v.as_i64()).unwrap();
        let page_count = map.get("page_count").and_then(|v| v.as_i64()).unwrap();
        assert_eq!(page_number, (i + 1) as i64);
        assert_eq!(page_count, 3);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ingest_pdf_persists_artifact_on_extractor_failure() {
    let kb = test_kb().await;
    let opts = PdfIngestOptions {
        artifact_id: "pdf-fail-1".into(),
        caller_supplied_id: true,
        extractor: Some(Arc::new(FailingExtractor)),
        source_path: None,
        session_id: None,
        triggered_by_message_id: None,
        category: None,
        source_id: None,
        revision_id: None,
    };
    let result = ingest_pdf(&kb, PdfInput::Bytes(b"junk".to_vec()), opts)
        .await
        .expect("ingest_pdf");

    assert!(!result.was_deduplicated);
    assert_eq!(result.page_count, 0);
    assert!(result.chunk_node_ids.is_empty());
    assert!(matches!(
        result.extraction_failure,
        Some(PdfExtractError::Parse(_))
    ));

    // Artifact + ArtifactContent still persisted.
    let session = kb.db().session();
    let rows = session
        .query(
            "MATCH (a:Artifact {artifact_id: 'pdf-fail-1'})-[:HAS_CONTENT]->(c:ArtifactContent) \
             RETURN a.page_count AS pc, c.mime AS mime",
        )
        .await
        .expect("query");
    let row = rows.rows().first().expect("artifact row");
    assert_eq!(row.get::<i64>("pc").unwrap(), 0);
    assert_eq!(row.get::<String>("mime").unwrap(), "application/pdf");

    // No chunks.
    let rows = session
        .query(
            "MATCH (a:Artifact {artifact_id: 'pdf-fail-1'})-[:HAS_CHUNK]->(c:Chunk) \
             RETURN count(c) AS n",
        )
        .await
        .expect("query chunks");
    let row = rows.rows().first().unwrap();
    assert_eq!(row.get::<i64>("n").unwrap(), 0);
}

/// Identical PDF bytes under two ids stay addressable under both — the same
/// contract as `ingest_artifact`. The bytes dedup on `:ArtifactContent`; the
/// ids do not collapse.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ingest_pdf_keeps_distinct_ids_for_identical_bytes() {
    let kb = test_kb().await;
    let pages = vec![ExtractedPage {
        page_number: 1,
        text: "Only page.".into(),
    }];
    let bytes = b"identical bytes".to_vec();

    let first = ingest_pdf(
        &kb,
        PdfInput::Bytes(bytes.clone()),
        mock_opts("pdf-dup-1", pages.clone()),
    )
    .await
    .expect("first ingest");
    assert!(!first.was_deduplicated);

    let second = ingest_pdf(&kb, PdfInput::Bytes(bytes), mock_opts("pdf-dup-2", pages))
        .await
        .expect("second ingest");
    assert!(
        !second.was_deduplicated,
        "a second id is a second artifact, not a dedup hit"
    );
    assert_ne!(second.artifact_node_id, first.artifact_node_id);

    for ext_id in ["pdf-dup-1", "pdf-dup-2"] {
        assert!(
            kb.get_node_by_ext_id("Artifact", "artifact_id", ext_id)
                .await
                .expect("lookup")
                .is_some(),
            "{ext_id} must resolve"
        );
    }
}

/// Replaying a PDF id with the same bytes is idempotent; reusing it for
/// different bytes is rejected.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ingest_pdf_rejects_reused_id_with_different_bytes() {
    let kb = test_kb().await;
    let pages = vec![ExtractedPage {
        page_number: 1,
        text: "Only page.".into(),
    }];

    let first = ingest_pdf(
        &kb,
        PdfInput::Bytes(b"original bytes".to_vec()),
        mock_opts("pdf-conflict", pages.clone()),
    )
    .await
    .expect("first ingest");
    assert!(!first.was_deduplicated);

    let replay = ingest_pdf(
        &kb,
        PdfInput::Bytes(b"original bytes".to_vec()),
        mock_opts("pdf-conflict", pages.clone()),
    )
    .await
    .expect("identical bytes must stay idempotent");
    assert!(replay.was_deduplicated);
    assert_eq!(replay.artifact_node_id, first.artifact_node_id);

    let err = ingest_pdf(
        &kb,
        PdfInput::Bytes(b"different bytes entirely".to_vec()),
        mock_opts("pdf-conflict", pages),
    )
    .await
    .expect_err("a reused id with different bytes must be rejected");
    assert!(
        matches!(err, uniko_store::UnikoError::IdConflict(_)),
        "expected IdConflict, got {err:?}"
    );
    assert!(!err.is_retriable());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ingest_pdf_real_pdf_round_trip() {
    // End-to-end through the default `PdfExtractCrate` against a real
    // PDF fixture — see `tests/fixtures/README.md`.
    let kb = test_kb().await;
    let bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/dummy.pdf"
    ))
    .expect("read fixture");

    let opts = PdfIngestOptions {
        artifact_id: "pdf-real-1".into(),
        caller_supplied_id: true,
        extractor: None, // default = PdfExtractCrate
        source_path: Some("tests/fixtures/dummy.pdf".into()),
        session_id: None,
        triggered_by_message_id: None,
        category: None,
        source_id: None,
        revision_id: None,
    };
    let result = ingest_pdf(&kb, PdfInput::Bytes(bytes), opts)
        .await
        .expect("ingest_pdf");

    assert!(!result.was_deduplicated);
    assert!(
        result.extraction_failure.is_none(),
        "real PDF should extract cleanly, got: {:?}",
        result.extraction_failure
    );
    assert_eq!(result.page_count, 1);
    assert_eq!(result.chunk_node_ids.len(), 1);

    let session = kb.db().session();
    let rows = session
        .query(
            "MATCH (a:Artifact {artifact_id: 'pdf-real-1'})-[:HAS_CHUNK]->(c:Chunk) \
             RETURN c.text AS text, c.metadata AS md",
        )
        .await
        .expect("query chunks");
    let row = rows.rows().first().expect("one chunk row");
    let text = row.get::<String>("text").unwrap();
    assert!(
        text.to_lowercase().contains("dummy"),
        "expected 'dummy' in chunk text, got: {text:?}"
    );
    let md = row.value("md").expect("metadata present");
    let map = md.as_object().expect("metadata is a map");
    assert_eq!(map.get("page_number").and_then(|v| v.as_i64()), Some(1));
    assert_eq!(map.get("page_count").and_then(|v| v.as_i64()), Some(1));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ingest_pdf_real_pdf_via_path() {
    // Same fixture, exercised via `PdfInput::Path` so the read-from-fs
    // branch is also covered.
    let kb = test_kb().await;
    let path = std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/dummy.pdf"
    ));

    let opts = PdfIngestOptions {
        artifact_id: "pdf-real-path-1".into(),
        caller_supplied_id: true,
        extractor: None,
        source_path: Some(path.display().to_string()),
        session_id: None,
        triggered_by_message_id: None,
        category: None,
        source_id: None,
        revision_id: None,
    };
    let result = ingest_pdf(&kb, PdfInput::Path(path), opts)
        .await
        .expect("ingest_pdf via Path");
    assert!(result.extraction_failure.is_none());
    assert_eq!(result.page_count, 1);
    assert_eq!(result.chunk_node_ids.len(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ingest_pdf_rejects_empty_artifact_id() {
    let kb = test_kb().await;
    let opts = PdfIngestOptions {
        artifact_id: String::new(),
        caller_supplied_id: true,
        extractor: Some(Arc::new(MockExtractor { pages: Vec::new() })),
        source_path: None,
        session_id: None,
        triggered_by_message_id: None,
        category: None,
        source_id: None,
        revision_id: None,
    };
    let err = ingest_pdf(&kb, PdfInput::Bytes(b"x".to_vec()), opts)
        .await
        .expect_err("should reject empty artifact_id");
    let msg = format!("{err}");
    assert!(msg.contains("artifact_id"), "got: {msg}");
}
