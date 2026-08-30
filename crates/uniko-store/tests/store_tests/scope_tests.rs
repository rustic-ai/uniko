//! Integration tests for dimensional recall scope (Stage B push-down).
//!
//! Validate the allow-set resolver and that a candidate query honors it.

use std::collections::HashMap;
use std::collections::HashSet;

use chrono::{TimeZone, Utc};
use uni_db::common::Value;
use uniko_store::NodeId;
use uniko_store::config::UnikoConfig;
use uniko_store::repository::recall::ScopeFilter;
use uniko_store::storage::KnowledgeBase;

async fn test_kb() -> KnowledgeBase {
    KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB")
}

async fn mk_session(kb: &KnowledgeBase, id: &str) -> NodeId {
    let mut p = HashMap::new();
    p.insert("session_id".into(), Value::String(id.into()));
    p.insert(
        "started_at".into(),
        Value::String("2024-01-01T00:00:00Z".into()),
    );
    kb.create_node("Session", &p).await.unwrap()
}

async fn mk_msg(kb: &KnowledgeBase, id: &str, content: &str, ts: &str, session: NodeId) -> NodeId {
    let mut p = HashMap::new();
    p.insert("message_id".into(), Value::String(id.into()));
    p.insert("content".into(), Value::String(content.into()));
    p.insert("timestamp".into(), Value::String(ts.into()));
    let m = kb.create_node("Message", &p).await.unwrap();
    kb.create_edge("IN_SESSION", m, session, &HashMap::new())
        .await
        .unwrap();
    m
}

fn since(filter_since: chrono::DateTime<Utc>) -> ScopeFilter {
    ScopeFilter {
        sessions: None,
        participants: None,
        since: Some(filter_since),
        until: None,
    }
}

async fn allow_set(kb: &KnowledgeBase, f: &ScopeFilter) -> HashSet<NodeId> {
    kb.resolve_scope_allow_set(f)
        .await
        .unwrap()
        .into_iter()
        .collect()
}

#[tokio::test]
async fn allow_set_filters_by_session() {
    let kb = test_kb().await;
    let s1 = mk_session(&kb, "s1").await;
    let s2 = mk_session(&kb, "s2").await;
    let m1 = mk_msg(&kb, "m1", "alpha one", "2024-01-01T00:00:00Z", s1).await;
    let m2 = mk_msg(&kb, "m2", "alpha two", "2024-02-01T00:00:00Z", s1).await;
    let m3 = mk_msg(&kb, "m3", "alpha three", "2024-03-01T00:00:00Z", s2).await;

    let f = ScopeFilter {
        sessions: Some(vec!["s1".into()]),
        ..Default::default()
    };
    let allow = allow_set(&kb, &f).await;
    assert!(allow.contains(&m1) && allow.contains(&m2));
    assert!(!allow.contains(&m3));
    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn session_scope_includes_attached_artifact_chunks() {
    let kb = test_kb().await;
    let s1 = mk_session(&kb, "s1").await;
    let s2 = mk_session(&kb, "s2").await;
    let m1 = mk_msg(&kb, "m1", "alpha one", "2024-01-01T00:00:00Z", s1).await;

    let mut chunk_ids = Vec::new();
    for (i, (attachment, target)) in [
        ("direct-session", s1),
        ("through-message", m1),
        ("other-session", s2),
    ]
    .into_iter()
    .enumerate()
    {
        let mut artifact_props = HashMap::new();
        artifact_props.insert(
            "artifact_id".into(),
            Value::String(format!("scope-artifact-{i}")),
        );
        artifact_props.insert("kind".into(), Value::String("text".into()));
        let artifact = kb.create_node("Artifact", &artifact_props).await.unwrap();

        let mut chunk_props = HashMap::new();
        chunk_props.insert("chunk_id".into(), Value::String(format!("scope-chunk-{i}")));
        chunk_props.insert("text".into(), Value::String(attachment.into()));
        chunk_props.insert("chunk_type".into(), Value::String("text".into()));
        let chunk = kb.create_node("Chunk", &chunk_props).await.unwrap();
        kb.create_edge("HAS_CHUNK", artifact, chunk, &HashMap::new())
            .await
            .unwrap();
        kb.create_edge("ATTACHED_TO", artifact, target, &HashMap::new())
            .await
            .unwrap();
        chunk_ids.push(chunk);
    }

    let filter = ScopeFilter {
        sessions: Some(vec!["s1".into()]),
        ..Default::default()
    };
    let allow = allow_set(&kb, &filter).await;
    assert!(allow.contains(&chunk_ids[0]));
    assert!(allow.contains(&chunk_ids[1]));
    assert!(!allow.contains(&chunk_ids[2]));

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn allow_set_filters_by_time() {
    let kb = test_kb().await;
    let s1 = mk_session(&kb, "s1").await;
    let m1 = mk_msg(&kb, "m1", "alpha one", "2024-01-01T00:00:00Z", s1).await;
    let m2 = mk_msg(&kb, "m2", "alpha two", "2024-02-01T00:00:00Z", s1).await;
    let m3 = mk_msg(&kb, "m3", "alpha three", "2024-03-01T00:00:00Z", s1).await;

    let f = since(Utc.with_ymd_and_hms(2024, 1, 15, 0, 0, 0).unwrap());
    let allow = allow_set(&kb, &f).await;
    assert!(!allow.contains(&m1));
    assert!(allow.contains(&m2) && allow.contains(&m3));
    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn empty_sessions_matches_nothing() {
    let kb = test_kb().await;
    let s1 = mk_session(&kb, "s1").await;
    mk_msg(&kb, "m1", "alpha one", "2024-01-01T00:00:00Z", s1).await;

    let f = ScopeFilter {
        sessions: Some(vec![]),
        ..Default::default()
    };
    let allow = allow_set(&kb, &f).await;
    assert!(allow.is_empty());
    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn execute_rule_with_allow_set_filters() {
    // Mirrors what Agent::query_in does: resolve the scope allow-set, bind it
    // as $allow, and run read-only Cypher that references it.
    let kb = test_kb().await;
    let s1 = mk_session(&kb, "s1").await;
    let s2 = mk_session(&kb, "s2").await;
    let m1 = mk_msg(&kb, "m1", "alpha", "2024-01-01T00:00:00Z", s1).await;
    let _m2 = mk_msg(&kb, "m2", "alpha", "2024-02-01T00:00:00Z", s2).await;

    let f = ScopeFilter {
        sessions: Some(vec!["s1".into()]),
        ..Default::default()
    };
    let allow = kb.resolve_scope_allow_set(&f).await.unwrap();
    let mut params = std::collections::HashMap::new();
    params.insert(
        "allow".to_string(),
        Value::List(allow.into_iter().map(Value::Int).collect()),
    );
    let rows = kb
        .query_cypher(
            "MATCH (m:Message) WHERE id(m) IN $allow RETURN id(m) AS nid",
            &params,
        )
        .await
        .unwrap();
    let ids: HashSet<i64> = rows
        .iter()
        .filter_map(|r| match r.get("nid") {
            Some(Value::Int(i)) => Some(*i),
            _ => None,
        })
        .collect();
    assert_eq!(ids, HashSet::from([m1]));
    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn fulltext_respects_allow_set() {
    let kb = test_kb().await;
    let s1 = mk_session(&kb, "s1").await;
    let m1 = mk_msg(&kb, "m1", "alpha bravo", "2024-01-01T00:00:00Z", s1).await;
    let _m2 = mk_msg(&kb, "m2", "alpha bravo", "2024-02-01T00:00:00Z", s1).await;

    // Restrict the candidate set to m1 only.
    let allow = vec![m1];
    let rows = kb
        .recall_fulltext_search("Message", "content", "alpha", 10, Some(&allow))
        .await
        .unwrap();
    assert!(!rows.is_empty());
    assert!(rows.iter().all(|r| r.node_id == m1));
    kb.shutdown().await.unwrap();
}
