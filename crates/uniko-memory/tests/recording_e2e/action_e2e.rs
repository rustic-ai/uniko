//! Integration tests for the `record_action` agent tool (F17–F20).
//!
//! Validates: Action node creation + PERFORMED_BY edge, TRIGGERED_BY
//! wiring to a Message, IN_SESSION wiring, NEXT_ACTION chaining, and
//! the output-overflow path that spills > 256-token outputs into an
//! Artifact via PRODUCED.

use std::collections::HashMap;

use chrono::Utc;
use serde_json::json;
use uni_db::Value;

use uniko_memory::{RecordActionParams, record_action};
use uniko_store::config::UnikoConfig;
use uniko_store::schema::constants::labels;
use uniko_store::types::datetime_value;
use uniko_store::{KnowledgeBase, UnikoError};

async fn kb() -> KnowledgeBase {
    KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB")
}

async fn seed_participant(kb: &KnowledgeBase, agent_id: &str) {
    let mut props: HashMap<String, Value> = HashMap::new();
    props.insert("kind".into(), Value::String("agent".into()));
    props.insert("name".into(), Value::String(format!("agent-{agent_id}")));
    kb.merge_node(labels::PARTICIPANT, "participant_id", agent_id, &props)
        .await
        .expect("participant created");
}

async fn seed_message(kb: &KnowledgeBase, msg_id: &str) -> i64 {
    let mut props: HashMap<String, Value> = HashMap::new();
    props.insert("content".into(), Value::String("trigger".into()));
    props.insert("timestamp".into(), datetime_value(Utc::now()));
    kb.merge_node(labels::MESSAGE, "message_id", msg_id, &props)
        .await
        .expect("message created")
}

async fn seed_session(kb: &KnowledgeBase, session_id: &str) -> i64 {
    let mut props: HashMap<String, Value> = HashMap::new();
    props.insert("topic".into(), Value::String("s".into()));
    props.insert("started_at".into(), datetime_value(Utc::now()));
    kb.merge_node(labels::SESSION, "session_id", session_id, &props)
        .await
        .expect("session created")
}

/// Silent-skip guards are forbidden in this file — if the local
/// embedder breaks we want a loud failure, not a green-bar lie.  The
/// helper below was previously used to swallow `UnikoError::Embedding`;
/// it now exists only to make accidental re-introduction obvious.
#[allow(dead_code)]
fn embedding_must_work() {}

#[tokio::test]
async fn record_action_creates_node_and_performed_by_edge() {
    let kb = kb().await;
    seed_participant(&kb, "agent-a").await;

    let params = RecordActionParams {
        action_type: "shell".into(),
        input: Some(json!("ls -la")),
        output: Some(json!("two files")),
        status: Some("success".into()),
        ..Default::default()
    };
    let result = record_action(&kb, "agent-a", params)
        .await
        .expect("record_action must succeed; the local embedder is required for this test");
    assert!(result.overflow_artifact.is_none());

    // PERFORMED_BY edge present.
    let session = kb.db().session();
    let rows = session
        .query_with(
            "MATCH (a:Action)-[:PERFORMED_BY]->(p:Participant) \
             WHERE id(a) = $aid RETURN p.participant_id AS pid",
        )
        .param("aid", result.action_node)
        .fetch_all()
        .await
        .expect("query");
    let pid: String = rows.rows().first().expect("edge").get("pid").expect("pid");
    assert_eq!(pid, "agent-a");
}

#[tokio::test]
async fn record_action_wires_triggered_by_when_message_exists() {
    let kb = kb().await;
    seed_participant(&kb, "agent-b").await;
    let _msg_node = seed_message(&kb, "msg-1").await;

    let params = RecordActionParams {
        action_type: "edit".into(),
        triggered_by_message_id: Some("msg-1".into()),
        ..Default::default()
    };
    let result = record_action(&kb, "agent-b", params)
        .await
        .expect("record_action");

    let session = kb.db().session();
    let rows = session
        .query_with(
            "MATCH (a:Action)-[:TRIGGERED_BY]->(m:Message) \
             WHERE id(a) = $aid RETURN m.message_id AS mid",
        )
        .param("aid", result.action_node)
        .fetch_all()
        .await
        .expect("query");
    let mid: String = rows.rows().first().expect("edge").get("mid").expect("mid");
    assert_eq!(mid, "msg-1");
}

#[tokio::test]
async fn record_action_wires_in_session_when_session_exists() {
    let kb = kb().await;
    seed_participant(&kb, "agent-c").await;
    let _sess_node = seed_session(&kb, "sess-1").await;

    let params = RecordActionParams {
        action_type: "rpc".into(),
        session_id: Some("sess-1".into()),
        ..Default::default()
    };
    let result = record_action(&kb, "agent-c", params)
        .await
        .expect("record_action");

    let session = kb.db().session();
    let rows = session
        .query_with(
            "MATCH (a:Action)-[:IN_SESSION]->(s:Session) \
             WHERE id(a) = $aid RETURN s.session_id AS sid",
        )
        .param("aid", result.action_node)
        .fetch_all()
        .await
        .expect("query");
    let sid: String = rows.rows().first().expect("edge").get("sid").expect("sid");
    assert_eq!(sid, "sess-1");
}

#[tokio::test]
async fn record_action_chains_next_action() {
    let kb = kb().await;
    seed_participant(&kb, "agent-d").await;

    let first = record_action(
        &kb,
        "agent-d",
        RecordActionParams {
            action_id: Some("a1".into()),
            action_type: "first".into(),
            ..Default::default()
        },
    )
    .await;
    let first = first.expect("record_action (first)");

    let second = record_action(
        &kb,
        "agent-d",
        RecordActionParams {
            action_type: "second".into(),
            previous_action_id: Some("a1".into()),
            ..Default::default()
        },
    )
    .await
    .expect("second action");

    let session = kb.db().session();
    let rows = session
        .query_with(
            "MATCH (a:Action)-[:NEXT_ACTION]->(b:Action) \
             WHERE id(a) = $a AND id(b) = $b RETURN id(b) AS b",
        )
        .param("a", first.action_node)
        .param("b", second.action_node)
        .fetch_all()
        .await
        .expect("query");
    assert_eq!(rows.rows().len(), 1, "NEXT_ACTION edge present");
}

#[tokio::test]
async fn record_action_overflows_large_output_to_artifact() {
    let kb = kb().await;
    seed_participant(&kb, "agent-e").await;

    // 256 token default; build something well over.
    let big = "lorem ipsum dolor sit amet ".repeat(200);
    let params = RecordActionParams {
        action_type: "shell".into(),
        output: Some(json!(big)),
        ..Default::default()
    };
    let result = record_action(&kb, "agent-e", params)
        .await
        .expect("record_action");
    let artifact_node = result.overflow_artifact.expect("overflow artifact created");

    // PRODUCED edge present; the artifact is a `tool_output` whose stored
    // bytes are the full payload and whose chunks reassemble it.
    let session = kb.db().session();
    let rows = session
        .query_with(
            "MATCH (a:Action)-[:PRODUCED]->(art:Artifact)-[:HAS_CONTENT]->(ac:ArtifactContent) \
             WHERE id(a) = $aid AND id(art) = $art \
             RETURN art.kind AS kind, ac.content_id AS cid",
        )
        .param("aid", result.action_node)
        .param("art", artifact_node)
        .fetch_all()
        .await
        .expect("query");
    let row = rows.rows().first().expect("PRODUCED edge");
    let kind: String = row.get("kind").expect("kind");
    assert_eq!(kind, "tool_output");
    let cid: String = row.get("cid").expect("content id");
    let bytes = kb.fetch_blob(&cid).await.expect("blob");
    assert_eq!(
        bytes,
        big.as_bytes(),
        "the artifact stores the payload verbatim"
    );
    let chunks = session
        .query_with(
            "MATCH (art:Artifact)-[:HAS_CHUNK]->(c:Chunk) WHERE id(art) = $art \
             RETURN count(c) AS n",
        )
        .param("art", artifact_node)
        .fetch_all()
        .await
        .expect("query");
    let n: i64 = chunks.rows().first().expect("count").get("n").expect("n");
    assert!(n > 1, "the payload is chunked for recall, got {n} chunk(s)");

    // Action.output is a short stub, not the full payload.
    let rows = session
        .query_with(
            "MATCH (a:Action) WHERE id(a) = $aid \
             RETURN a.output AS out",
        )
        .param("aid", result.action_node)
        .fetch_all()
        .await
        .expect("query");
    let out: String = rows
        .rows()
        .first()
        .expect("out row")
        .get("out")
        .expect("out");
    assert!(out.contains("overflow"), "stub content was: {out}");
}

/// Real Kniv NLP exercise: an Action whose input mentions a named
/// person must end up with a `MENTIONS` edge to an Entity node of
/// type PERSON.  This is the only thing that proves the NER
/// integration is actually wired — without it we'd be claiming
/// "Kniv-backed" Actions without any evidence they bind entities.
#[cfg(feature = "onnx")]
#[tokio::test]
async fn record_action_extracts_entities_via_kniv() {
    let kb = kb().await;
    seed_participant(&kb, "agent-kniv").await;

    let params = RecordActionParams {
        action_type: "shell".into(),
        input: Some(json!(
            "send a status update to Caroline about the refund pipeline"
        )),
        output: Some(json!("ok, sent")),
        status: Some("success".into()),
        ..Default::default()
    };
    let result = record_action(&kb, "agent-kniv", params)
        .await
        .expect("record_action");

    let session = kb.db().session();
    let rows = session
        .query_with(
            "MATCH (a:Action)-[:MENTIONS]->(e:Entity) \
             WHERE id(a) = $aid \
             RETURN e.name AS name, e.entity_type AS etype",
        )
        .param("aid", result.action_node)
        .fetch_all()
        .await
        .expect("MENTIONS query");

    let entities: Vec<(String, String)> = rows
        .rows()
        .iter()
        .map(|r| {
            (
                r.get::<String>("name").unwrap_or_default(),
                r.get::<String>("etype").unwrap_or_default(),
            )
        })
        .collect();
    assert!(
        !entities.is_empty(),
        "Kniv NER must surface at least one entity for the action text"
    );
    assert!(
        entities
            .iter()
            .any(|(n, t)| n.to_lowercase().contains("caroline") && t == "person"),
        "expected a person entity for 'Caroline'; got {entities:?}"
    );
}

#[tokio::test]
async fn record_action_fails_when_agent_missing() {
    let kb = kb().await;
    let err = record_action(
        &kb,
        "missing-agent",
        RecordActionParams {
            action_type: "noop".into(),
            ..Default::default()
        },
    )
    .await
    .expect_err("should fail");
    match err {
        UnikoError::Storage(msg) => assert!(msg.contains("not found")),
        other => panic!("expected Storage error, got {other:?}"),
    }
}
