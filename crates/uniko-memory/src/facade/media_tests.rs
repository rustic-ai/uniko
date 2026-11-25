//! Media extraction through the facade (issue #61): structured segments,
//! media retention and provenance, model-backed captions over a stub
//! server, video composition, and refusals at build time.
//!
//! Every test that ingests needs the embedder; without it the test skips.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use uniko_pipes::content::Mime;
use uniko_store::{UnikoError, Value};

use crate::{
    Extraction, IngestOutcome, IngestSource, LlmSpec, MediaRetention, Modality, ModelExtractor,
    ModelProvenance, Segment, SegmentKind, TextExtractor, Uniko, VideoFrame, VideoParts,
};

/// A one-pixel PNG.
const PNG: &[u8] = &[
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
    0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
    0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,
    0x42, 0x60, 0x82,
];

fn skip(err: &UnikoError) -> bool {
    matches!(err, UnikoError::Embedding(_))
}

fn artifact_id(outcome: &IngestOutcome) -> String {
    match outcome {
        IngestOutcome::Artifact(r) => r.artifact_id.clone(),
        IngestOutcome::Pdf(r) => r.artifact_id.clone(),
    }
}

/// A 16 kHz mono 16-bit WAV of silence, `ms` long.
fn wav(ms: u32) -> Vec<u8> {
    let data_len = 16_000 * 2 * ms / 1000;
    let mut v = Vec::new();
    v.extend_from_slice(b"RIFF");
    v.extend_from_slice(&(36 + data_len).to_le_bytes());
    v.extend_from_slice(b"WAVEfmt ");
    v.extend_from_slice(&16u32.to_le_bytes());
    v.extend_from_slice(&1u16.to_le_bytes());
    v.extend_from_slice(&1u16.to_le_bytes());
    v.extend_from_slice(&16_000u32.to_le_bytes());
    v.extend_from_slice(&32_000u32.to_le_bytes());
    v.extend_from_slice(&2u16.to_le_bytes());
    v.extend_from_slice(&16u16.to_le_bytes());
    v.extend_from_slice(b"data");
    v.extend_from_slice(&data_len.to_le_bytes());
    v.resize(v.len() + data_len as usize, 0);
    v
}

/// Every chunk of artifact `id`, as `(chunk_type, modality,
/// source_model_version, metadata)`, in index order.
async fn chunks_of(
    agent: &crate::Agent,
    id: &str,
) -> Vec<(String, Option<String>, Option<String>, serde_json::Value)> {
    let rows = agent
        .kb()
        .query_cypher(
            "MATCH (a:Artifact {artifact_id: $id})-[:HAS_CHUNK]->(c:Chunk) \
             RETURN c.chunk_type AS t, c.modality AS m, c.source_model_version AS v, \
                    c.metadata AS meta ORDER BY c.index",
            &HashMap::from([("id".to_string(), Value::String(id.into()))]),
        )
        .await
        .expect("chunk query");
    rows.iter()
        .map(|r| {
            let s = |k: &str| match r.get(k) {
                Some(Value::String(s)) => Some(s.clone()),
                _ => None,
            };
            let meta = r
                .get("meta")
                .cloned()
                .map(serde_json::Value::from)
                .unwrap_or(serde_json::Value::Null);
            (s("t").unwrap_or_default(), s("m"), s("v"), meta)
        })
        .collect()
}

async fn build(builder: crate::UnikoBuilder) -> Option<Uniko> {
    match builder.in_memory().build().await {
        Ok(m) => Some(m),
        Err(e) if matches!(e, UnikoError::Config(_) | UnikoError::Unsupported(_)) => {
            panic!("the test configuration was refused: {e}")
        }
        Err(e) => {
            eprintln!("skipping: instance unavailable: {e}");
            None
        }
    }
}

#[tokio::test]
async fn a_structured_callback_writes_positioned_chunks() {
    let transcript = TextExtractor::structured(Modality::Audio, |_| {
        Ok(Extraction::new(
            vec![
                Segment::new(
                    SegmentKind::Transcript,
                    "the budget review moves to thursday",
                )
                .with_time(0, 2400)
                .with_confidence(0.92)
                .with_speaker("alice"),
                Segment::new(SegmentKind::Transcript, "bring the q3 forecast")
                    .with_time(2400, 4100)
                    .with_confidence(0.81)
                    .with_speaker("alice"),
            ],
            Some(ModelProvenance::new(
                "asr/host",
                "whisper-large",
                "transcribe",
            )),
        ))
    });
    let Some(memory) = build(Uniko::builder().extractor(Arc::new(transcript))).await else {
        return;
    };
    let agent = memory.agent("viewer");
    let session = agent.session("memo");
    let outcome = match session
        .ingest(
            IngestSource::bytes(wav(50))
                .with_mime(Mime::parse("audio/wav").unwrap())
                .with_id("memo-1"),
        )
        .await
    {
        Ok(o) => o,
        Err(e) if skip(&e) => return eprintln!("skipping: {e}"),
        Err(e) => panic!("ingest: {e}"),
    };
    assert_eq!(artifact_id(&outcome), "memo-1");

    let chunks = chunks_of(&agent, "memo-1").await;
    assert_eq!(chunks.len(), 1, "two short segments group into one chunk");
    let (chunk_type, modality, version, meta) = &chunks[0];
    assert_eq!(chunk_type, "transcript_segment");
    assert_eq!(modality.as_deref(), Some("audio"));
    assert_eq!(version.as_deref(), Some("asr/host@whisper-large"));
    assert_eq!(meta["time_start_ms"], 0);
    assert_eq!(meta["time_end_ms"], 4100);
    assert_eq!(meta["speaker"], "alice");
    assert!((meta["confidence"].as_f64().unwrap() - 0.81).abs() < 1e-4);

    let view = agent.data().artifact("memo-1").await.unwrap().unwrap();
    assert!(view.text.contains("q3 forecast"), "{}", view.text);

    // A recall hit and a search hit both say where in the recording the
    // text came from, down to each utterance.
    let check = |p: &crate::MediaPosition| {
        assert_eq!(p.modality, "audio");
        assert_eq!(
            p.source_model_version.as_deref(),
            Some("asr/host@whisper-large")
        );
        assert_eq!((p.time_start_ms, p.time_end_ms), (Some(0), Some(4100)));
        assert_eq!(p.speaker.as_deref(), Some("alice"));
        assert_eq!(p.segments.len(), 2);
        assert_eq!(p.segments[1].time_start_ms, Some(2400));
    };
    let bundle = agent
        .recall("when does the budget review move")
        .await
        .unwrap();
    let hit = bundle
        .items
        .iter()
        .find(|i| i.content.contains("q3 forecast"))
        .expect("the transcript is recalled");
    check(
        hit.position
            .as_ref()
            .expect("a recall hit carries its position"),
    );
    let hits = agent
        .sources()
        .search("q3 forecast", &crate::SearchOptions::default())
        .await
        .unwrap();
    let hit = hits.first().expect("a search hit");
    check(
        hit.position
            .as_ref()
            .expect("a search hit carries its position"),
    );
    assert!(
        agent
            .data()
            .media_sources("memo-1")
            .await
            .unwrap()
            .is_empty(),
        "no retention records no source"
    );
    drop(session);
    drop(agent);
    memory.shutdown().await.unwrap();
}

#[tokio::test]
async fn plain_text_callbacks_are_unchanged() {
    let captioner = TextExtractor::new(Modality::Image, |_| Ok("a red zeppelin".to_string()));
    let Some(memory) = build(Uniko::builder().extractor(Arc::new(captioner))).await else {
        return;
    };
    let agent = memory.agent("viewer");
    let session = agent.session("photos");
    match session
        .ingest(
            IngestSource::bytes(PNG.to_vec())
                .with_mime(Mime::parse("image/png").unwrap())
                .with_id("zep"),
        )
        .await
    {
        Ok(_) => {}
        Err(e) if skip(&e) => return eprintln!("skipping: {e}"),
        Err(e) => panic!("ingest: {e}"),
    }
    let chunks = chunks_of(&agent, "zep").await;
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].0, "text");
    assert_eq!(chunks[0].1, None, "plain text carries no modality");
    assert_eq!(chunks[0].2, None, "or model");
    let view = agent.data().artifact("zep").await.unwrap().unwrap();
    assert_eq!(view.kind, "image");
    let bundle = agent.recall("zeppelin").await.unwrap();
    assert!(
        bundle.items.iter().any(|i| i.content.contains("zeppelin")),
        "the caption is recalled"
    );
    assert!(
        bundle.items.iter().all(|i| i.position.is_none()),
        "plain text carries no position"
    );
    assert!(
        agent
            .data()
            .artifact_bytes("zep")
            .await
            .unwrap()
            .is_some_and(|b| b == b"a red zeppelin")
    );
    drop(session);
    drop(agent);
    memory.shutdown().await.unwrap();
}

#[tokio::test]
async fn identity_retention_records_one_source_without_bytes() {
    let captioner = TextExtractor::new(Modality::Image, |req| {
        Ok(format!("photo {}", req.path.unwrap_or_default()))
    })
    .with_retention(MediaRetention::Identity);
    let Some(memory) = build(Uniko::builder().extractor(Arc::new(captioner))).await else {
        return;
    };
    let agent = memory.agent("viewer");
    let session = agent.session("photos");
    for (id, path) in [("p1", "a.png"), ("p2", "b.png")] {
        match session
            .ingest(
                IngestSource::bytes(PNG.to_vec())
                    .with_mime(Mime::parse("image/png").unwrap())
                    .with_path(path)
                    .with_id(id),
            )
            .await
        {
            Ok(_) => {}
            Err(e) if skip(&e) => return eprintln!("skipping: {e}"),
            Err(e) => panic!("ingest: {e}"),
        }
    }
    let s1 = agent.data().media_sources("p1").await.unwrap();
    let s2 = agent.data().media_sources("p2").await.unwrap();
    assert_eq!(s1.len(), 1);
    assert_eq!(
        s1[0].artifact_id, s2[0].artifact_id,
        "the same bytes are one source"
    );
    let source = &s1[0];
    assert_eq!(source.modality.as_deref(), Some("image"));
    assert_eq!(source.mime.as_deref(), Some("image/png"));
    assert_eq!(source.size, Some(PNG.len() as i64));
    assert_eq!(source.hash, uniko_store::KnowledgeBase::sha256_hex(PNG));
    assert_eq!(source.derivation_kind.as_deref(), Some("extract"));
    assert!(!source.bytes_retained);
    assert_eq!(
        agent
            .data()
            .artifact_bytes(&source.artifact_id)
            .await
            .unwrap(),
        None,
        "identity keeps no bytes"
    );
    let own = agent
        .data()
        .media_sources(&source.artifact_id)
        .await
        .unwrap();
    assert_eq!(own.len(), 1, "a source's own id returns itself");
    assert_eq!(own[0].derivation_kind, None);
    drop(session);
    drop(agent);
    memory.shutdown().await.unwrap();
}

#[tokio::test]
async fn retained_bytes_round_trip_and_go_with_the_last_derived_artifact() {
    let captioner = TextExtractor::structured(Modality::Image, |req| {
        Ok(Extraction::new(
            vec![Segment::new(
                SegmentKind::Caption,
                format!("caption of {}", req.path.unwrap_or_default()),
            )],
            None,
        ))
    })
    .with_retention(MediaRetention::Bytes);
    let Some(memory) = build(Uniko::builder().extractor(Arc::new(captioner))).await else {
        return;
    };
    let agent = memory.agent("viewer");
    let session = agent.session("photos");
    for (id, path) in [("b1", "one.png"), ("b2", "two.png")] {
        match session
            .ingest(
                IngestSource::bytes(PNG.to_vec())
                    .with_mime(Mime::parse("image/png").unwrap())
                    .with_path(path)
                    .with_id(id),
            )
            .await
        {
            Ok(_) => {}
            Err(e) if skip(&e) => return eprintln!("skipping: {e}"),
            Err(e) => panic!("ingest: {e}"),
        }
    }
    let source = agent.data().media_sources("b1").await.unwrap().remove(0);
    assert!(source.bytes_retained);
    assert_eq!(source.derivation_kind.as_deref(), Some("caption"));
    assert_eq!(
        agent
            .data()
            .artifact_bytes(&source.artifact_id)
            .await
            .unwrap()
            .as_deref(),
        Some(PNG),
        "the original bytes come back"
    );

    // The source outlives one derived artifact, and goes with the last.
    session.delete_document("b1").await.unwrap();
    assert!(
        agent
            .data()
            .artifact_bytes(&source.artifact_id)
            .await
            .unwrap()
            .is_some()
    );
    session.delete_document("b2").await.unwrap();
    assert!(
        agent
            .data()
            .media_sources(&source.artifact_id)
            .await
            .unwrap()
            .is_empty(),
        "an orphaned source is deleted"
    );
    assert_eq!(
        agent
            .data()
            .artifact_bytes(&source.artifact_id)
            .await
            .unwrap(),
        None
    );
    drop(session);
    drop(agent);
    memory.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_media_source_is_as_visible_as_its_derived_text() {
    let captioner = TextExtractor::new(Modality::Image, |_| Ok("the launch slide".to_string()))
        .with_retention(MediaRetention::Bytes);
    let Some(memory) = build(Uniko::builder().extractor(Arc::new(captioner))).await else {
        return;
    };
    let agent = memory.agent("viewer");
    let session = agent.session("slides");
    match session
        .ingest(
            IngestSource::bytes(PNG.to_vec())
                .with_mime(Mime::parse("image/png").unwrap())
                .with_id("slide"),
        )
        .await
    {
        Ok(_) => {}
        Err(e) if skip(&e) => return eprintln!("skipping: {e}"),
        Err(e) => panic!("ingest: {e}"),
    }
    let source = agent.data().media_sources("slide").await.unwrap().remove(0);
    assert!(
        agent
            .data()
            .artifact_bytes(&source.artifact_id)
            .await
            .unwrap()
            .is_some()
    );
    // Forgetting the derived text redacts its chunks; the original must
    // not stay readable behind it.
    let chunks = agent
        .kb()
        .query_cypher(
            "MATCH (:Artifact {artifact_id: 'slide'})-[:HAS_CHUNK]->(c:Chunk) RETURN id(c) AS nid",
            &HashMap::new(),
        )
        .await
        .unwrap();
    assert!(!chunks.is_empty());
    for row in chunks {
        let Some(Value::Int(nid)) = row.get("nid") else {
            panic!("chunk id");
        };
        agent
            .kb()
            .update_node(
                *nid,
                &HashMap::from([("redacted".to_string(), Value::Bool(true))]),
            )
            .await
            .unwrap();
    }
    assert_eq!(
        agent
            .data()
            .artifact_bytes(&source.artifact_id)
            .await
            .unwrap(),
        None,
        "the bytes are hidden once the derived text is"
    );
    assert!(
        agent
            .data()
            .artifact(&source.artifact_id)
            .await
            .unwrap()
            .is_none()
    );
    drop(session);
    drop(agent);
    memory.shutdown().await.unwrap();
}

/// An OpenAI-compatible chat stub that records each request body and
/// answers every request with `reply`.
async fn chat_stub(reply: &'static str) -> (String, Arc<Mutex<Vec<String>>>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let seen = bodies.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let seen = seen.clone();
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
                seen.lock()
                    .unwrap()
                    .push(String::from_utf8_lossy(&buf[head_end..]).to_string());
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
    (format!("http://{addr}/v1"), bodies)
}

#[tokio::test]
async fn a_caption_model_receives_the_image_and_is_recorded() {
    let (base_url, bodies) = chat_stub("a lighthouse on a rocky coast at dusk").await;
    let Some(memory) = build(
        Uniko::builder()
            .llm(LlmSpec::llamacpp("vision/stub", "stub-vlm", base_url))
            .model_extractor(
                ModelExtractor::image()
                    .caption("vision/stub")
                    .caption_prompt("Describe it.")
                    .retention(MediaRetention::Identity),
            )
            .require([crate::Requirement::Extractor(Modality::Image)]),
    )
    .await
    else {
        return;
    };
    let agent = memory.agent("viewer");
    let session = agent.session("photos");
    match session
        .ingest(
            IngestSource::bytes(PNG.to_vec())
                .with_mime(Mime::parse("image/png").unwrap())
                .with_id("coast"),
        )
        .await
    {
        Ok(_) => {}
        Err(e) if skip(&e) => return eprintln!("skipping: {e}"),
        Err(e) => panic!("ingest: {e}"),
    }
    let sent = bodies.lock().unwrap().clone();
    assert_eq!(sent.len(), 1, "one caption request");
    assert!(
        sent[0].contains("data:image/png;base64,"),
        "the image is attached: {}",
        sent[0]
    );
    assert!(sent[0].contains("Describe it."), "with the prompt");

    let chunks = chunks_of(&agent, "coast").await;
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].0, "caption");
    assert_eq!(chunks[0].1.as_deref(), Some("image"));
    assert_eq!(chunks[0].2.as_deref(), Some("vision/stub@stub-vlm"));
    let source = agent.data().media_sources("coast").await.unwrap().remove(0);
    assert_eq!(source.derivation_kind.as_deref(), Some("caption"));
    drop(session);
    drop(agent);
    memory.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_video_is_captioned_frame_by_frame_on_its_timeline() {
    let (base_url, bodies) = chat_stub("a whiteboard with an architecture diagram").await;
    let composed = Arc::new(Mutex::new(0usize));
    let calls = composed.clone();
    let Some(memory) = build(
        Uniko::builder()
            .llm(LlmSpec::llamacpp("vision/stub", "stub-vlm", base_url))
            .model_extractor(
                ModelExtractor::video(move |req| {
                    assert_eq!(req.modality, Modality::Video);
                    *calls.lock().unwrap() += 1;
                    Ok(VideoParts::new(
                        vec![
                            VideoFrame::new(5_000, PNG.to_vec(), "image/png"),
                            VideoFrame::new(0, PNG.to_vec(), "image/png"),
                        ],
                        None,
                    ))
                })
                .caption("vision/stub"),
            ),
    )
    .await
    else {
        return;
    };
    let agent = memory.agent("viewer");
    let session = agent.session("standup");
    match session
        .ingest(
            IngestSource::bytes(vec![0u8; 64])
                .with_mime(Mime::parse("video/mp4").unwrap())
                .with_id("standup-video"),
        )
        .await
    {
        Ok(_) => {}
        Err(e) if skip(&e) => return eprintln!("skipping: {e}"),
        Err(e) => panic!("ingest: {e}"),
    }
    assert_eq!(*composed.lock().unwrap(), 1);
    assert_eq!(bodies.lock().unwrap().len(), 2, "one caption per frame");
    let chunks = chunks_of(&agent, "standup-video").await;
    assert_eq!(chunks.len(), 1, "the two captions group");
    let (t, m, _, meta) = &chunks[0];
    assert_eq!(t, "caption");
    assert_eq!(m.as_deref(), Some("video"));
    assert_eq!(
        meta["time_start_ms"], 0,
        "frames are sorted onto the timeline"
    );
    assert_eq!(meta["time_end_ms"], 5_000);
    drop(session);
    drop(agent);
    memory.shutdown().await.unwrap();
}

#[tokio::test]
async fn unsupported_model_extractors_fail_at_build() {
    let refused = |builder: crate::UnikoBuilder| async move {
        match builder.in_memory().build().await {
            Err(e @ (UnikoError::Unsupported(_) | UnikoError::Config(_))) => e.to_string(),
            Ok(_) => panic!("the extractor was accepted"),
            Err(e) => panic!("refused for another reason: {e}"),
        }
    };
    let e = refused(
        Uniko::builder()
            .model_extractor(ModelExtractor::new(Modality::Video).caption("vision/stub")),
    )
    .await;
    assert!(e.contains("no video task"), "{e}");
    let e =
        refused(Uniko::builder().model_extractor(ModelExtractor::image().ocr("ocr/nowhere"))).await;
    assert!(e.contains("not in the model catalog"), "{e}");
    let e = refused(
        Uniko::builder()
            .llm(LlmSpec::new(
                "vision/gemini",
                "remote/gemini",
                "gemini-2.5-flash",
            ))
            .model_extractor(ModelExtractor::image().caption("vision/gemini")),
    )
    .await;
    assert!(e.contains("does not pass images"), "{e}");
    let e =
        refused(Uniko::builder().model_extractor(ModelExtractor::audio().ocr("ocr/default"))).await;
    assert!(e.contains("images only"), "{e}");
}

#[tokio::test]
async fn non_wav_audio_is_refused_with_the_conversion() {
    let (base_url, _) = chat_stub("unused").await;
    // Any audio extractor rejects the payload before a model is called;
    // a stub catalog entry stands in for the transcription alias.
    let extractor = ModelExtractor::audio().transcribe("asr/default");
    let catalog_has_asr = Uniko::builder()
        .in_memory()
        .llm(LlmSpec::llamacpp("vision/stub", "stub", base_url))
        .model_extractor(extractor)
        .build()
        .await;
    match catalog_has_asr {
        // No transcription alias in the default catalog: refused at build,
        // which is the behavior under test for a missing alias.
        Err(UnikoError::Config(m)) => assert!(m.contains("asr/default"), "{m}"),
        Err(UnikoError::Unsupported(m)) => assert!(m.contains("transcribe"), "{m}"),
        Ok(_) => panic!("a transcribe alias outside the catalog was accepted"),
        Err(e) => eprintln!("skipping: {e}"),
    }
    let e = uniko_extract::ingest::check_wav(b"ID3\x03 an mp3").unwrap_err();
    assert!(
        e.to_string().contains("ffmpeg -i in -ar 16000 -ac 1"),
        "{e}"
    );
}

/// Real OCR through the model extractor, on the page image of the scanned
/// PDF fixture. Downloads the OCR model, so it runs only with
/// `EXPENSIVE_TESTS=1`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ocr_reads_an_image_into_located_regions() {
    if !matches!(
        std::env::var("EXPENSIVE_TESTS").as_deref(),
        Ok("1" | "true" | "yes")
    ) {
        eprintln!("skipping: set EXPENSIVE_TESTS=1 to run real OCR");
        return;
    }
    let image = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../uniko-extract/tests/fixtures/scanned-page.jpg"
    ))
    .expect("read scanned-page.jpg");
    let mut config = crate::UnikoConfig::default();
    config.ocr.enabled = true;
    let memory = Uniko::builder()
        .in_memory()
        .raw_config(config)
        .model_extractor(
            ModelExtractor::image()
                .ocr(uniko_store::schema::OCR_ALIAS)
                .retention(MediaRetention::Bytes),
        )
        .build()
        .await
        .expect("an instance with OCR");
    let agent = memory.agent("inspector");
    let session = agent.session("maintenance");
    session
        .ingest(
            IngestSource::bytes(image.clone())
                .with_mime(Mime::parse("image/jpeg").unwrap())
                .with_id("pump-log"),
        )
        .await
        .expect("OCR ingest");

    let view = agent.data().artifact("pump-log").await.unwrap().unwrap();
    let text = view.text.to_lowercase();
    for needle in ["pump station", "mira okafor", "september"] {
        assert!(text.contains(needle), "OCR missed {needle:?}: {text}");
    }
    let chunks = chunks_of(&agent, "pump-log").await;
    assert!(!chunks.is_empty());
    for (t, m, v, meta) in &chunks {
        assert_eq!(t, "ocr_region");
        assert_eq!(m.as_deref(), Some("image"));
        assert!(
            v.as_deref().is_some_and(|v| v.starts_with("ocr/default@")),
            "{v:?}"
        );
        assert!(meta["confidence"].as_f64().is_some(), "{meta}");
    }
    // The detector locates each line, so a chunk of one line has its box.
    assert!(
        chunks
            .iter()
            .any(|(_, _, _, meta)| meta.get("bbox").is_some())
            || chunks.len() == 1,
        "no chunk kept a region: {chunks:?}"
    );
    let source = agent
        .data()
        .media_sources("pump-log")
        .await
        .unwrap()
        .remove(0);
    assert_eq!(source.derivation_kind.as_deref(), Some("ocr"));
    assert_eq!(
        agent
            .data()
            .artifact_bytes(&source.artifact_id)
            .await
            .unwrap(),
        Some(image)
    );
    drop(session);
    drop(agent);
    memory.shutdown().await.unwrap();
}

/// A PNG distinct from [`PNG`]: the same image with a trailing comment
/// chunk, so it hashes differently but decodes the same.
fn other_png(tag: u8) -> Vec<u8> {
    let mut v = PNG[..PNG.len() - 12].to_vec();
    v.extend_from_slice(&[0, 0, 0, 1, b't', b'E', b'X', b't', tag, 0, 0, 0, 0]);
    v.extend_from_slice(&PNG[PNG.len() - 12..]);
    v
}

async fn ingest_png(session: &crate::Session, bytes: Vec<u8>, id: Option<&str>) -> Option<String> {
    let mut src = IngestSource::bytes(bytes).with_mime(Mime::parse("image/png").unwrap());
    if let Some(id) = id {
        src = src.with_id(id);
    }
    match session.ingest(src).await {
        Ok(o) => Some(artifact_id(&o)),
        Err(e) if skip(&e) => {
            eprintln!("skipping: {e}");
            None
        }
        Err(e) => panic!("ingest: {e}"),
    }
}

#[tokio::test]
async fn the_same_caption_on_two_images_stays_two_artifacts() {
    // Temperature-0 captioners (and OCR of the same form) often return the
    // same text for different images. Each must keep its own source.
    let captioner = TextExtractor::new(Modality::Image, |_| Ok("a cat on a sofa".to_string()))
        .with_retention(MediaRetention::Bytes);
    let Some(memory) = build(Uniko::builder().extractor(Arc::new(captioner))).await else {
        return;
    };
    let agent = memory.agent("viewer");
    let session = agent.session("photos");
    let Some(a) = ingest_png(&session, other_png(1), None).await else {
        return;
    };
    let b = ingest_png(&session, other_png(2), None).await.unwrap();
    assert_ne!(a, b, "two images are two artifacts");
    let sa = agent.data().media_sources(&a).await.unwrap();
    let sb = agent.data().media_sources(&b).await.unwrap();
    assert_eq!((sa.len(), sb.len()), (1, 1));
    assert_ne!(sa[0].hash, sb[0].hash);
    assert_eq!(
        agent
            .data()
            .artifact_bytes(&sb[0].artifact_id)
            .await
            .unwrap(),
        Some(other_png(2))
    );

    // A plain document that reads the same never attaches to either.
    let doc = session
        .ingest(IngestSource::text("a cat on a sofa"))
        .await
        .unwrap();
    let doc_id = artifact_id(&doc);
    assert_ne!(doc_id, a);
    assert_ne!(doc_id, b);
    assert!(
        agent
            .data()
            .media_sources(&doc_id)
            .await
            .unwrap()
            .is_empty()
    );
    drop(session);
    drop(agent);
    memory.shutdown().await.unwrap();
}

#[tokio::test]
async fn re_ingesting_the_same_media_is_a_replay_even_if_the_model_rewords() {
    let calls = Arc::new(Mutex::new(0usize));
    let n = calls.clone();
    let captioner = TextExtractor::new(Modality::Image, move |_| {
        let mut n = n.lock().unwrap();
        *n += 1;
        Ok(format!("caption number {n}"))
    })
    .with_retention(MediaRetention::Identity);
    let Some(memory) = build(Uniko::builder().extractor(Arc::new(captioner))).await else {
        return;
    };
    let agent = memory.agent("viewer");
    let session = agent.session("photos");
    // A caller id, then the same bytes with no id: both replay.
    let Some(first) = ingest_png(&session, PNG.to_vec(), Some("named")).await else {
        return;
    };
    assert_eq!(first, "named");
    let again = session
        .ingest(
            IngestSource::bytes(PNG.to_vec())
                .with_mime(Mime::parse("image/png").unwrap())
                .with_id("named"),
        )
        .await
        .expect("the same id over the same media is not a conflict");
    let IngestOutcome::Artifact(r) = again else {
        panic!("artifact")
    };
    assert!(r.was_deduplicated);
    let view = agent.data().artifact("named").await.unwrap().unwrap();
    assert_eq!(view.text, "caption number 1", "the first text is kept");

    let auto1 = ingest_png(&session, other_png(7), None).await.unwrap();
    let auto2 = ingest_png(&session, other_png(7), None).await.unwrap();
    assert_eq!(auto1, auto2, "the same bytes find the same artifact");

    // The id now names text from this image; other media under it conflicts.
    let conflict = session
        .ingest(
            IngestSource::bytes(other_png(9))
                .with_mime(Mime::parse("image/png").unwrap())
                .with_id("named"),
        )
        .await;
    assert!(
        matches!(conflict, Err(UnikoError::IdConflict(_))),
        "{conflict:?}"
    );
    drop(session);
    drop(agent);
    memory.shutdown().await.unwrap();
}

#[tokio::test]
async fn one_hidden_derived_artifact_hides_the_shared_source() {
    let captioner = TextExtractor::new(Modality::Image, |r| {
        Ok(format!("view {}", r.path.unwrap_or_default()))
    })
    .with_retention(MediaRetention::Bytes);
    let Some(memory) = build(Uniko::builder().extractor(Arc::new(captioner))).await else {
        return;
    };
    let agent = memory.agent("viewer");
    let session = agent.session("photos");
    for (id, path) in [("open", "a"), ("secret", "b")] {
        match session
            .ingest(
                IngestSource::bytes(PNG.to_vec())
                    .with_mime(Mime::parse("image/png").unwrap())
                    .with_path(path)
                    .with_id(id),
            )
            .await
        {
            Ok(_) => {}
            Err(e) if skip(&e) => return eprintln!("skipping: {e}"),
            Err(e) => panic!("ingest: {e}"),
        }
    }
    assert_eq!(agent.data().media_sources("open").await.unwrap().len(), 1);
    let rows = agent
        .kb()
        .query_cypher(
            "MATCH (:Artifact {artifact_id: 'secret'})-[:HAS_CHUNK]->(c:Chunk) RETURN id(c) AS nid",
            &HashMap::new(),
        )
        .await
        .unwrap();
    for row in rows {
        let Some(Value::Int(nid)) = row.get("nid") else {
            panic!("chunk id")
        };
        agent
            .kb()
            .update_node(
                *nid,
                &HashMap::from([("redacted".to_string(), Value::Bool(true))]),
            )
            .await
            .unwrap();
    }
    assert!(
        agent.data().media_sources("open").await.unwrap().is_empty(),
        "the source is hidden once any text derived from it is"
    );
    drop(session);
    drop(agent);
    memory.shutdown().await.unwrap();
}

#[tokio::test]
async fn an_extraction_with_no_text_is_refused() {
    let empty = TextExtractor::structured(Modality::Image, |_| {
        Ok(Extraction::new(
            vec![Segment::new(SegmentKind::OcrRegion, "  ")],
            None,
        ))
    });
    let Some(memory) = build(Uniko::builder().extractor(Arc::new(empty))).await else {
        return;
    };
    let agent = memory.agent("viewer");
    let session = agent.session("s");
    let r = session
        .ingest(IngestSource::bytes(PNG.to_vec()).with_mime(Mime::parse("image/png").unwrap()))
        .await;
    assert!(
        matches!(&r, Err(UnikoError::Unsupported(m)) if m.contains("no text was found")),
        "{r:?}"
    );
    drop(session);
    drop(agent);
    memory.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_wav_source_records_its_header_and_goes_with_its_session() {
    let memo = TextExtractor::new(Modality::Audio, |_| {
        Ok("a voice memo about the roadmap".into())
    })
    .with_retention(MediaRetention::Bytes);
    let Some(memory) = build(Uniko::builder().extractor(Arc::new(memo))).await else {
        return;
    };
    let agent = memory.agent("viewer");
    let session = agent.session("memos");
    let outcome = match session
        .ingest(IngestSource::bytes(wav(1500)).with_mime(Mime::parse("audio/wav").unwrap()))
        .await
    {
        Ok(o) => o,
        Err(e) if skip(&e) => return eprintln!("skipping: {e}"),
        Err(e) => panic!("ingest: {e}"),
    };
    let id = artifact_id(&outcome);
    let source = agent.data().media_sources(&id).await.unwrap().remove(0);
    assert_eq!(source.duration_ms, Some(1500));
    assert_eq!(source.sample_rate, Some(16_000));
    assert_eq!(source.channels, Some(1));

    // Ingested into a session, the text and its source go with it.
    agent.kb().delete_session_cascade("memos").await.unwrap();
    assert!(agent.data().artifact(&id).await.unwrap().is_none());
    assert_eq!(
        agent
            .data()
            .artifact_bytes(&source.artifact_id)
            .await
            .unwrap(),
        None
    );
    let left = agent
        .kb()
        .query_cypher(
            "MATCH (m:Artifact {kind: 'media'}) RETURN count(m) AS n",
            &HashMap::new(),
        )
        .await
        .unwrap();
    assert_eq!(
        left[0].get("n"),
        Some(&Value::Int(0)),
        "no source is left behind"
    );
    drop(session);
    drop(agent);
    memory.shutdown().await.unwrap();
}

#[tokio::test]
async fn deleting_a_turn_takes_its_media_unless_another_session_uses_it() {
    let captioner = TextExtractor::new(Modality::Image, |_| Ok("whiteboard sketch".to_string()))
        .with_retention(MediaRetention::Bytes);
    let Some(memory) = build(Uniko::builder().extractor(Arc::new(captioner))).await else {
        return;
    };
    let agent = memory.agent("viewer");
    let mut chat = agent.session("chat");
    let image = |b: Vec<u8>| IngestSource::bytes(b).with_mime(Mime::parse("image/png").unwrap());
    match chat
        .observe(
            crate::Turn::new("user", "here is the sketch")
                .id("t1")
                .attach(image(other_png(3))),
        )
        .await
    {
        Ok(_) => {}
        Err(e) if skip(&e) => return eprintln!("skipping: {e}"),
        Err(e) => panic!("observe: {e}"),
    }
    chat.observe(
        crate::Turn::new("user", "and another")
            .id("t2")
            .attach(image(other_png(4))),
    )
    .await
    .unwrap();
    let id3 = format!(
        "image-{}",
        uniko_store::KnowledgeBase::sha256_hex(&other_png(3))
    );
    let id4 = format!(
        "image-{}",
        uniko_store::KnowledgeBase::sha256_hex(&other_png(4))
    );
    let s3 = agent.data().media_sources(&id3).await.unwrap().remove(0);
    // The second image is also ingested into another session.
    let other = agent.session("archive");
    other.ingest(image(other_png(4))).await.unwrap();
    let s4 = agent.data().media_sources(&id4).await.unwrap().remove(0);

    chat.delete_turn("t1").await.unwrap();
    assert!(
        agent.data().artifact(&id3).await.unwrap().is_none(),
        "the turn's text goes"
    );
    assert!(
        agent
            .data()
            .media_sources(&s3.artifact_id)
            .await
            .unwrap()
            .is_empty(),
        "and its source"
    );
    chat.delete_turn("t2").await.unwrap();
    assert!(
        agent
            .data()
            .artifact_bytes(&s4.artifact_id)
            .await
            .unwrap()
            .is_some(),
        "a source another session still uses stays"
    );
    drop(chat);
    drop(other);
    drop(agent);
    memory.shutdown().await.unwrap();
}
