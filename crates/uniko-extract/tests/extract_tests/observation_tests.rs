//! Integration tests for observation extraction via the atomic ingest path.

use std::collections::HashMap;

use chrono::Utc;

use uniko_extract::ingest::atomic::ingest_message_atomic;
use uniko_extract::ingest::context::SessionContext;
use uniko_pipes::types::IngestMessage;
use uniko_store::config::UnikoConfig;
use uniko_store::storage::KnowledgeBase;
// `Direction` is used only by `test_observation_from_prose`, which is
// onnx-gated below. Gate the import to match so the default (non-onnx)
// build stays clean under `clippy -D warnings`.
#[cfg(feature = "onnx")]
use uniko_store::storage::edges::Direction;

async fn test_kb() -> KnowledgeBase {
    KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB")
}

fn ingest_msg(id: &str, content: &str, session: &str, sender: &str) -> IngestMessage {
    IngestMessage {
        message_id: id.into(),
        content: content.into(),
        content_type: "text".into(),
        sender_id: sender.into(),
        session_id: session.into(),
        addressed_to: None,
        timestamp: Utc::now(),
        metadata: HashMap::new(),
        category: None,
        source_id: None,
        revision_id: None,
    }
}

// Single-token proper-noun subject ("Caroline"). The default-build rule-based
// NER (`ner/rules.rs` `proper_noun` regex) only matches multi-token names
// (2+ capitalized words), so it cannot anchor this observation's subject —
// robust single-token entity detection is an NLP-model capability. This
// end-to-end assertion therefore runs only under `onnx`, and is `#[ignore]`d
// because it drives real ONNX inference requiring a cached model (matching the
// `nlp_observation_integration_test` convention).
#[cfg(feature = "onnx")]
#[tokio::test]
#[ignore] // Requires ONNX model cached in .uni_cache/
async fn test_observation_from_prose() {
    let kb = test_kb().await;
    let text = "Caroline attended an LGBTQ support group yesterday.";
    let msg = ingest_msg("m-obs-1", text, "s-obs", "tester");
    let mut session_ctx = SessionContext::new(msg.session_id.clone(), 0);

    let result = ingest_message_atomic(&kb, &msg, &mut session_ctx)
        .await
        .expect("ingest");
    assert!(
        !result.extracted_observations.is_empty(),
        "should extract at least one observation"
    );

    let obs_nid = result.extracted_observations[0];
    let (label, props) = kb.get_node(obs_nid).await.unwrap().expect("obs node");
    assert_eq!(label, "Observation");
    assert!(props.get("content").and_then(|v| v.as_str()).is_some());
    assert!(props.get("subject").and_then(|v| v.as_str()).is_some());

    let edges = kb
        .get_edges(obs_nid, "OBSERVED_IN", Direction::Outgoing)
        .await
        .unwrap();
    assert!(!edges.is_empty(), "OBSERVED_IN edge should exist");
    assert_eq!(edges[0].to, result.message_node_id);

    let about_edges = kb
        .get_edges(obs_nid, "ABOUT", Direction::Outgoing)
        .await
        .unwrap();
    assert!(!about_edges.is_empty(), "ABOUT edge should exist");
}

#[tokio::test]
async fn test_observation_filtering_greetings() {
    let kb = test_kb().await;
    let msg = ingest_msg("m-greet", "Hey!", "s-greet", "tester");
    let mut session_ctx = SessionContext::new(msg.session_id.clone(), 0);

    let result = ingest_message_atomic(&kb, &msg, &mut session_ctx)
        .await
        .expect("ingest");
    assert!(
        result.extracted_observations.is_empty(),
        "greetings should produce no observations"
    );
}

#[tokio::test]
async fn test_observation_filtering_pure_question() {
    let kb = test_kb().await;
    let msg = ingest_msg("m-q", "Where is the store?", "s-q", "tester");
    let mut session_ctx = SessionContext::new(msg.session_id.clone(), 0);

    let result = ingest_message_atomic(&kb, &msg, &mut session_ctx)
        .await
        .expect("ingest");
    assert!(
        result.extracted_observations.is_empty(),
        "pure questions should produce no observations"
    );
}

#[tokio::test]
async fn test_observation_no_entities_skips() {
    let kb = test_kb().await;
    // No proper nouns — NER yields no entities → observation step gates off.
    let msg = ingest_msg("m-noent", "It rained yesterday.", "s-noent", "tester");
    let mut session_ctx = SessionContext::new(msg.session_id.clone(), 0);

    let result = ingest_message_atomic(&kb, &msg, &mut session_ctx)
        .await
        .expect("ingest");
    assert!(
        result.extracted_observations.is_empty(),
        "no entities → no observations"
    );
}

#[tokio::test]
async fn test_atomic_populates_observations() {
    let kb = test_kb().await;
    let text = "Alice works at the hospital and lives in New York.";
    let msg = ingest_msg("m-pop-obs", text, "s-pop-obs", "tester");
    let mut session_ctx = SessionContext::new(msg.session_id.clone(), 0);

    let result = ingest_message_atomic(&kb, &msg, &mut session_ctx)
        .await
        .expect("ingest");
    assert!(
        !result.extracted_observations.is_empty(),
        "extracted_observations should be populated"
    );

    for &oid in &result.extracted_observations {
        let node = kb.get_node(oid).await.unwrap();
        assert!(node.is_some());
        let (label, _) = node.unwrap();
        assert_eq!(label, "Observation");
    }
}
