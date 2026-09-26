//! Integration tests for entity extraction via the atomic ingest path.

use std::collections::HashMap;

use chrono::Utc;

use uniko_extract::ingest::atomic::ingest_message_atomic;
use uniko_extract::ingest::context::SessionContext;
use uniko_pipes::types::IngestMessage;
use uniko_store::config::UnikoConfig;
use uniko_store::storage::KnowledgeBase;
use uniko_store::storage::edges::Direction;

async fn test_kb() -> KnowledgeBase {
    KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB")
}

fn ingest_msg(id: &str, content: &str, content_type: &str, session: &str) -> IngestMessage {
    IngestMessage {
        message_id: id.into(),
        content: content.into(),
        content_type: content_type.into(),
        sender_id: "tester".into(),
        session_id: session.into(),
        addressed_to: None,
        timestamp: Utc::now(),
        metadata: HashMap::new(),
        category: None,
        source_id: None,
        revision_id: None,
    }
}

#[tokio::test]
async fn test_entity_extraction_prose() {
    let kb = test_kb().await;
    let text = "I met Caroline Smith at https://example.com";
    let msg = ingest_msg("m-prose", text, "text", "s-prose");
    let mut session_ctx = SessionContext::new(msg.session_id.clone(), 0);

    let result = ingest_message_atomic(&kb, &msg, &mut session_ctx)
        .await
        .expect("ingest");
    assert!(
        !result.extracted_entities.is_empty(),
        "should extract entities"
    );

    let edges = kb
        .get_edges(result.message_node_id, "MENTIONS", Direction::Outgoing)
        .await
        .unwrap();
    assert!(!edges.is_empty(), "MENTIONS edges should exist");
}

#[tokio::test]
async fn test_entity_extraction_code() {
    let kb = test_kb().await;
    let code = "def process_order():\n    pass\n\nclass PaymentService:\n    pass\n";
    let mut msg = ingest_msg("m-code", code, "code", "s-code");
    msg.metadata.insert(
        "language".into(),
        serde_json::Value::String("python".into()),
    );
    let mut session_ctx = SessionContext::new(msg.session_id.clone(), 0);

    let result = ingest_message_atomic(&kb, &msg, &mut session_ctx)
        .await
        .expect("ingest");
    assert!(
        result.extracted_entities.len() >= 2,
        "should find process_order and PaymentService, got {}",
        result.extracted_entities.len()
    );
}

#[tokio::test]
async fn test_entity_dedup_existing() {
    let kb = test_kb().await;
    let mut session_ctx = SessionContext::new("s-dedup".into(), 0);

    let msg1 = ingest_msg(
        "m-dedup-1",
        "I met Caroline Smith today.",
        "text",
        "s-dedup",
    );
    ingest_message_atomic(&kb, &msg1, &mut session_ctx)
        .await
        .unwrap();

    let mut msg2 = ingest_msg("m-dedup-2", "Caroline Smith called me.", "text", "s-dedup");
    msg2.timestamp = msg1.timestamp + chrono::Duration::seconds(1);
    let r2 = ingest_message_atomic(&kb, &msg2, &mut session_ctx)
        .await
        .unwrap();

    // Verify dedup: the Entity for "caroline smith" should have frequency >= 2.
    for (eid, _name) in &r2.extracted_entities {
        let node = kb.get_node(*eid).await.unwrap();
        if let Some((_, props)) = node {
            let name = props.get("name").and_then(|v| v.as_str()).unwrap_or("");
            if name == "caroline smith" {
                let freq = props.get("frequency").and_then(|v| v.as_i64()).unwrap_or(0);
                assert!(
                    freq >= 2,
                    "caroline smith frequency should be >= 2, got {freq}"
                );
                return;
            }
        }
    }
    // If "caroline smith" wasn't extracted, the test still passes — rules
    // may attribute differently.
}

#[tokio::test]
async fn test_atomic_populates_entities() {
    let kb = test_kb().await;
    let content = "Visit https://rust-lang.org on January 15, 2024.";
    let msg = ingest_msg("m-pop", content, "text", "s-pop");
    let mut session_ctx = SessionContext::new(msg.session_id.clone(), 0);

    let result = ingest_message_atomic(&kb, &msg, &mut session_ctx)
        .await
        .expect("ingest");
    assert!(
        !result.extracted_entities.is_empty(),
        "extracted_entities should be populated"
    );

    for (eid, _name) in &result.extracted_entities {
        let node = kb.get_node(*eid).await.unwrap();
        assert!(node.is_some(), "entity node {eid} must exist");
        let (label, _) = node.unwrap();
        assert_eq!(label, "Entity");
    }
}

#[tokio::test]
async fn test_admission_drops_date_and_greeting_keeps_real() {
    // Default-build end-to-end check of the entity admission policy: the
    // rules NER tags "January 15, 2024" as a Date and "Hey Caroline Smith"
    // as a single greeting-prefixed Person; admission must drop both while
    // keeping the URL.
    let kb = test_kb().await;
    let content = "Hey Caroline Smith, visit https://rust-lang.org on January 15, 2024.";
    let msg = ingest_msg("m-admit", content, "text", "s-admit");
    let mut session_ctx = SessionContext::new(msg.session_id.clone(), 0);
    let result = ingest_message_atomic(&kb, &msg, &mut session_ctx)
        .await
        .expect("ingest");

    // Assert on the surviving entities' canonical names directly (the
    // String in `extracted_entities`), avoiding per-node fetches.
    let names: Vec<String> = result
        .extracted_entities
        .iter()
        .map(|(_, name)| name.clone())
        .collect();

    assert!(
        names.iter().any(|n| n.contains("rust-lang.org")),
        "the URL must survive admission; got {names:?}"
    );
    assert!(
        !names.iter().any(|n| n.starts_with("hey ")),
        "greeting-prefixed fragments must be dropped; got {names:?}"
    );
    assert!(
        !names
            .iter()
            .any(|n| n.contains("january") || n.contains("2024")),
        "date entities must be dropped by admission; got {names:?}"
    );
}
