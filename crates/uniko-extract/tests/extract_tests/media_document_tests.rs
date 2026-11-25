//! Document mode of the model-backed image extractor (issue #61): an image
//! goes to a `document_extract` alias and its blocks become `page_block`
//! chunks in reading order, each block's box kept.
//!
//! The CI test serves a canned DocTags reply from a stub `llama-server`.
//! The opt-in test runs against a real document VLM:
//!
//! ```sh
//! EXPENSIVE_TESTS=1 DOCEXT_BASE_URL=http://localhost:8080/v1 DOCEXT_STYLE=olmocr \
//!   cargo nextest run -p uniko-extract --test integration media_document_tests::
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use uniko_extract::ingest::{
    IngestContext, IngestSource, ModalityRegistry, ModelExtractor, ingest_source,
};
use uniko_pipes::content::Mime;
use uniko_store::config::UnikoConfig;
use uniko_store::storage::KnowledgeBase;
use uniko_store::{UnikoError, Value};

/// A one-pixel PNG.
const PNG: &[u8] = &[
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
    0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
    0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,
    0x42, 0x60, 0x82,
];

/// An OpenAI-compatible chat stub answering every request with `reply`.
async fn chat_stub(reply: String) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let reply = reply.clone();
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let mut chunk = [0u8; 8192];
                let head_end = loop {
                    let Ok(n) = stream.read(&mut chunk).await else {
                        return;
                    };
                    if n == 0 {
                        return;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                    if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        break i + 4;
                    }
                };
                let head = String::from_utf8_lossy(&buf[..head_end]).to_ascii_lowercase();
                let len = head
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .and_then(|v| v.trim().parse::<usize>().ok())
                    .unwrap_or(0);
                while buf.len() < head_end + len {
                    let Ok(n) = stream.read(&mut chunk).await else {
                        break;
                    };
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                }
                let body = serde_json::json!({
                    "choices": [{ "message": { "role": "assistant", "content": reply } }],
                })
                .to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\
                     content-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.shutdown().await;
            });
        }
    });
    format!("http://{addr}/v1")
}

/// A store with a `doc/test` document-extraction alias on `base_url`.
async fn store(base_url: &str, style: &str, model: &str) -> Option<KnowledgeBase> {
    let spec: uniko_store::xervo::ModelAliasSpec = serde_json::from_value(serde_json::json!({
        "alias": "doc/test",
        "task": "document_extract",
        "provider_id": "remote/llamacpp",
        "model_id": model,
        "options": { "base_url": base_url, "style": style },
    }))
    .expect("a valid alias spec");
    match KnowledgeBase::in_memory_with_xervo(UnikoConfig::default(), vec![spec]).await {
        Ok(kb) => Some(kb),
        Err(e) => {
            eprintln!("skipping: store unavailable: {e}");
            None
        }
    }
}

/// Ingest the PNG through a document-mode extractor; `None` when the
/// embedder is unavailable.
async fn ingest(kb: &KnowledgeBase, id: &str) -> Option<()> {
    let extractor = ModelExtractor::image().document("doc/test");
    extractor
        .validate(&kb.model_entries().expect("a known catalog"))
        .expect("document mode validates");
    let mut registry = ModalityRegistry::new();
    registry.register(Arc::new(extractor.into_extractor()));
    let src = IngestSource::bytes(PNG.to_vec())
        .with_mime(Mime::parse("image/png").unwrap())
        .with_id(id);
    match ingest_source(kb, &registry, src, IngestContext::default()).await {
        Ok(_) => Some(()),
        Err(UnikoError::Embedding(e)) => {
            eprintln!("skipping: embedder unavailable: {e}");
            None
        }
        Err(e) => panic!("document ingest failed: {e}"),
    }
}

/// Each chunk of artifact `id`: `(chunk_type, text, model, metadata)`.
async fn chunks(kb: &KnowledgeBase, id: &str) -> Vec<(String, String, String, serde_json::Value)> {
    let rows = kb
        .query_cypher(
            "MATCH (a:Artifact {artifact_id: $id})-[:HAS_CHUNK]->(c:Chunk) \
             RETURN c.chunk_type AS t, c.text AS text, c.source_model_version AS v, \
                    c.metadata AS meta ORDER BY c.index",
            &HashMap::from([("id".to_string(), Value::String(id.into()))]),
        )
        .await
        .expect("chunk query");
    let s = |r: &HashMap<String, Value>, k: &str| match r.get(k) {
        Some(Value::String(v)) => v.clone(),
        _ => String::new(),
    };
    rows.iter()
        .map(|r| {
            let meta = r
                .get("meta")
                .cloned()
                .map(serde_json::Value::from)
                .unwrap_or(serde_json::Value::Null);
            (s(r, "t"), s(r, "text"), s(r, "v"), meta)
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn document_mode_keeps_blocks_in_reading_order_with_their_boxes() {
    let url = chat_stub(
        r#"<heading loc="0.1,0.1,0.9,0.2">Pump Station Log</heading><text>The north pump was replaced on March 12.</text>"#
            .to_string(),
    )
    .await;
    let Some(kb) = store(&url, "granite-docling", "granite-docling-258m").await else {
        return;
    };
    let Some(()) = ingest(&kb, "page").await else {
        return;
    };
    let chunks = chunks(&kb, "page").await;
    assert_eq!(chunks.len(), 1, "two short blocks group: {chunks:?}");
    let (t, text, model, meta) = &chunks[0];
    assert_eq!(t, "page_block");
    assert_eq!(model, "doc/test@granite-docling-258m");
    assert!(
        text.find("Pump Station Log") < text.find("north pump"),
        "blocks keep reading order: {text}"
    );
    let segments = meta["segments"].as_array().expect("each block is kept");
    assert_eq!(segments.len(), 2);
    let heading = &segments[0];
    let (s, e) = (
        heading["start"].as_u64().unwrap() as usize,
        heading["end"].as_u64().unwrap() as usize,
    );
    assert_eq!(&text[s..e], "Pump Station Log");
    let bbox: Vec<f64> = heading["bbox"]
        .as_array()
        .expect("the heading keeps its box")
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert!(
        (bbox[0] - 0.1).abs() < 1e-6 && (bbox[3] - 0.2).abs() < 1e-6,
        "{bbox:?}"
    );
    assert!(
        segments[1].get("bbox").is_none(),
        "an unlocated block has no box"
    );
    kb.shutdown().await.ok();
}

/// A reply with no block structure still lands, as one block of its text.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn document_mode_falls_back_to_the_markdown_of_a_page() {
    let url = chat_stub("Inspector Mira Okafor signed off the repair.".to_string()).await;
    let Some(kb) = store(&url, "olmocr", "olmocr-2").await else {
        return;
    };
    let Some(()) = ingest(&kb, "plain-page").await else {
        return;
    };
    let chunks = chunks(&kb, "plain-page").await;
    assert!(
        chunks
            .iter()
            .any(|(t, text, _, _)| t == "page_block" && text.contains("Mira Okafor")),
        "{chunks:?}"
    );
    kb.shutdown().await.ok();
}

/// Against a real document VLM in `llama-server`. Opt-in: set
/// `EXPENSIVE_TESTS=1` and `DOCEXT_BASE_URL` (and `DOCEXT_STYLE` when the
/// server runs MinerU or Granite-Docling).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn document_mode_against_a_real_vlm() {
    let expensive = matches!(
        std::env::var("EXPENSIVE_TESTS").as_deref(),
        Ok("1" | "true" | "yes")
    );
    let Ok(url) = std::env::var("DOCEXT_BASE_URL") else {
        eprintln!("skipping: set EXPENSIVE_TESTS=1 and DOCEXT_BASE_URL to run");
        return;
    };
    if !expensive {
        eprintln!("skipping: set EXPENSIVE_TESTS=1 to run");
        return;
    }
    let style = std::env::var("DOCEXT_STYLE").unwrap_or_else(|_| "olmocr".into());
    let image = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/scanned-page.jpg"
    ))
    .expect("read scanned-page.jpg");
    let kb = store(&url, &style, "document-vlm")
        .await
        .expect("a store with the document alias");
    let extractor = ModelExtractor::image().document("doc/test");
    let mut registry = ModalityRegistry::new();
    registry.register(Arc::new(extractor.into_extractor()));
    let src = IngestSource::bytes(image)
        .with_mime(Mime::parse("image/jpeg").unwrap())
        .with_id("real-page");
    ingest_source(&kb, &registry, src, IngestContext::default())
        .await
        .expect("document extraction against the server");
    let text = chunks(&kb, "real-page")
        .await
        .into_iter()
        .map(|(_, t, _, _)| t)
        .collect::<Vec<_>>()
        .join("\n")
        .to_lowercase();
    assert!(
        text.contains("pump station"),
        "the VLM read the page: {text}"
    );
    kb.shutdown().await.ok();
}
