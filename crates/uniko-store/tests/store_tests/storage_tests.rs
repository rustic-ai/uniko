//! Integration tests for KnowledgeBase node/edge CRUD and batch operations.

use std::collections::HashMap;
use uni_db::Value;
use uniko_store::config::UnikoConfig;
use uniko_store::storage::KnowledgeBase;
use uniko_store::storage::edges::Direction;
use uniko_store::storage::filter::Filter;

async fn test_kb() -> KnowledgeBase {
    KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB")
}

// ── Node CRUD ──

#[tokio::test]
async fn test_create_and_get_node() {
    let kb = test_kb().await;
    let mut props = HashMap::new();
    props.insert("participant_id".into(), Value::String("p-1".into()));
    props.insert("kind".into(), Value::String("agent".into()));
    props.insert("name".into(), Value::String("TestBot".into()));

    let nid = kb.create_node("Participant", &props).await.unwrap();
    let (label, got) = kb.get_node(nid).await.unwrap().expect("node must exist");
    assert_eq!(label, "Participant");
    assert_eq!(got.get("name").and_then(|v| v.as_str()), Some("TestBot"));

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_get_node_by_ext_id() {
    let kb = test_kb().await;
    let mut props = HashMap::new();
    props.insert("message_id".into(), Value::String("m-1".into()));
    props.insert("content".into(), Value::String("hello".into()));
    props.insert(
        "timestamp".into(),
        Value::String("2024-01-01T00:00:00Z".into()),
    );

    let nid = kb.create_node("Message", &props).await.unwrap();
    let (found_id, found_props) = kb
        .get_node_by_ext_id("Message", "message_id", "m-1")
        .await
        .unwrap()
        .expect("must find by ext_id");
    assert_eq!(found_id, nid);
    assert_eq!(
        found_props.get("content").and_then(|v| v.as_str()),
        Some("hello")
    );

    // Non-existent ext_id returns None.
    assert!(
        kb.get_node_by_ext_id("Message", "message_id", "m-999")
            .await
            .unwrap()
            .is_none()
    );

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_update_node() {
    let kb = test_kb().await;
    let mut props = HashMap::new();
    props.insert("participant_id".into(), Value::String("p-1".into()));
    props.insert("kind".into(), Value::String("agent".into()));
    props.insert("name".into(), Value::String("Bot".into()));
    let nid = kb.create_node("Participant", &props).await.unwrap();

    // Update only the name.
    let mut updates = HashMap::new();
    updates.insert("name".into(), Value::String("UpdatedBot".into()));
    kb.update_node(nid, &updates).await.unwrap();

    let (_, got) = kb.get_node(nid).await.unwrap().unwrap();
    assert_eq!(got.get("name").and_then(|v| v.as_str()), Some("UpdatedBot"));
    // kind is preserved.
    assert_eq!(got.get("kind").and_then(|v| v.as_str()), Some("agent"));

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_delete_node() {
    let kb = test_kb().await;
    let mut props = HashMap::new();
    props.insert("participant_id".into(), Value::String("p-del".into()));
    props.insert("kind".into(), Value::String("human".into()));
    let nid = kb.create_node("Participant", &props).await.unwrap();

    assert!(kb.delete_node(nid).await.unwrap());
    assert!(kb.get_node(nid).await.unwrap().is_none());

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_merge_node_creates() {
    let kb = test_kb().await;
    let mut props = HashMap::new();
    props.insert("name".into(), Value::String("MergeBot".into()));
    props.insert("kind".into(), Value::String("agent".into()));

    let nid = kb
        .merge_node("Participant", "participant_id", "p-merge-1", &props)
        .await
        .unwrap();
    let (_, got) = kb.get_node(nid).await.unwrap().unwrap();
    assert_eq!(got.get("name").and_then(|v| v.as_str()), Some("MergeBot"));

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_merge_node_updates() {
    let kb = test_kb().await;
    let mut props = HashMap::new();
    props.insert("name".into(), Value::String("V1".into()));
    props.insert("kind".into(), Value::String("agent".into()));
    let nid1 = kb
        .merge_node("Participant", "participant_id", "p-merge-2", &props)
        .await
        .unwrap();

    // Merge again with updated name.
    let mut props2 = HashMap::new();
    props2.insert("name".into(), Value::String("V2".into()));
    props2.insert("kind".into(), Value::String("agent".into()));
    let nid2 = kb
        .merge_node("Participant", "participant_id", "p-merge-2", &props2)
        .await
        .unwrap();

    assert_eq!(nid1, nid2, "MERGE should not create a duplicate");
    let (_, got) = kb.get_node(nid2).await.unwrap().unwrap();
    assert_eq!(got.get("name").and_then(|v| v.as_str()), Some("V2"));

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_merge_idempotent() {
    let kb = test_kb().await;
    let mut props = HashMap::new();
    props.insert("name".into(), Value::String("Same".into()));
    props.insert("kind".into(), Value::String("agent".into()));

    let nid1 = kb
        .merge_node("Participant", "participant_id", "p-idem", &props)
        .await
        .unwrap();
    let nid2 = kb
        .merge_node("Participant", "participant_id", "p-idem", &props)
        .await
        .unwrap();
    assert_eq!(nid1, nid2);

    // Only one node should exist.
    let all = kb.query_nodes("Participant", None, None).await.unwrap();
    let count = all
        .iter()
        .filter(|(_, p)| p.get("participant_id").and_then(|v| v.as_str()) == Some("p-idem"))
        .count();
    assert_eq!(count, 1);

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_query_nodes_no_filter() {
    let kb = test_kb().await;
    for i in 0..5 {
        let mut props = HashMap::new();
        props.insert("entity_id".into(), Value::String(format!("ent-q-{i}")));
        props.insert("name".into(), Value::String(format!("Entity{i}")));
        props.insert("entity_type".into(), Value::String("concept".into()));
        kb.create_node("Entity", &props).await.unwrap();
    }

    let all = kb.query_nodes("Entity", None, None).await.unwrap();
    assert_eq!(all.len(), 5);

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_query_nodes_with_eq_filter() {
    let kb = test_kb().await;
    for (i, kind) in ["agent", "human", "agent", "service", "agent"]
        .iter()
        .enumerate()
    {
        let mut props = HashMap::new();
        props.insert("participant_id".into(), Value::String(format!("pf-{i}")));
        props.insert("kind".into(), Value::String(kind.to_string()));
        kb.create_node("Participant", &props).await.unwrap();
    }

    let filter = Filter::Eq("kind".into(), Value::String("agent".into()));
    let agents = kb
        .query_nodes("Participant", Some(&filter), None)
        .await
        .unwrap();
    assert_eq!(agents.len(), 3);

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_invalid_label_rejected() {
    let kb = test_kb().await;
    let result = kb.create_node("BogusLabel", &HashMap::new()).await;
    assert!(result.is_err());
    kb.shutdown().await.unwrap();
}

// ── Edge CRUD ──

#[tokio::test]
async fn test_create_and_get_edge() {
    let kb = test_kb().await;

    let mut mp = HashMap::new();
    mp.insert("message_id".into(), Value::String("m-e1".into()));
    mp.insert("content".into(), Value::String("hi".into()));
    mp.insert(
        "timestamp".into(),
        Value::String("2024-01-01T00:00:00Z".into()),
    );
    let m_id = kb.create_node("Message", &mp).await.unwrap();

    let mut pp = HashMap::new();
    pp.insert("participant_id".into(), Value::String("p-e1".into()));
    pp.insert("kind".into(), Value::String("human".into()));
    let p_id = kb.create_node("Participant", &pp).await.unwrap();

    let mut ep = HashMap::new();
    ep.insert("role".into(), Value::String("user".into()));
    let eid = kb.create_edge("SENT_BY", m_id, p_id, &ep).await.unwrap();

    let edges = kb
        .get_edges(m_id, "SENT_BY", Direction::Outgoing)
        .await
        .unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].id, eid);
    assert_eq!(
        edges[0].properties.get("role").and_then(|v| v.as_str()),
        Some("user")
    );

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_delete_edge() {
    let kb = test_kb().await;

    let mut mp = HashMap::new();
    mp.insert("message_id".into(), Value::String("m-de".into()));
    mp.insert("content".into(), Value::String("x".into()));
    mp.insert(
        "timestamp".into(),
        Value::String("2024-01-01T00:00:00Z".into()),
    );
    let m_id = kb.create_node("Message", &mp).await.unwrap();

    let mut pp = HashMap::new();
    pp.insert("participant_id".into(), Value::String("p-de".into()));
    pp.insert("kind".into(), Value::String("human".into()));
    let p_id = kb.create_node("Participant", &pp).await.unwrap();

    let eid = kb
        .create_edge("SENT_BY", m_id, p_id, &HashMap::new())
        .await
        .unwrap();
    assert!(kb.delete_edge(eid).await.unwrap());

    let edges = kb
        .get_edges(m_id, "SENT_BY", Direction::Outgoing)
        .await
        .unwrap();
    assert!(edges.is_empty());

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_delete_edges_between() {
    let kb = test_kb().await;

    let mut m1 = HashMap::new();
    m1.insert("message_id".into(), Value::String("m-deb1".into()));
    m1.insert("content".into(), Value::String("a".into()));
    m1.insert(
        "timestamp".into(),
        Value::String("2024-01-01T00:00:00Z".into()),
    );
    let id1 = kb.create_node("Message", &m1).await.unwrap();

    let mut m2 = HashMap::new();
    m2.insert("message_id".into(), Value::String("m-deb2".into()));
    m2.insert("content".into(), Value::String("b".into()));
    m2.insert(
        "timestamp".into(),
        Value::String("2024-01-01T00:00:01Z".into()),
    );
    let id2 = kb.create_node("Message", &m2).await.unwrap();

    // Create two NEXT edges between them.
    kb.create_edge("NEXT", id1, id2, &HashMap::new())
        .await
        .unwrap();
    kb.create_edge("NEXT", id1, id2, &HashMap::new())
        .await
        .unwrap();

    let deleted = kb.delete_edges_between("NEXT", id1, id2).await.unwrap();
    assert_eq!(deleted, 2);

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_update_edge() {
    let kb = test_kb().await;

    let mut m1 = HashMap::new();
    m1.insert("message_id".into(), Value::String("m-ue1".into()));
    m1.insert("content".into(), Value::String("a".into()));
    m1.insert(
        "timestamp".into(),
        Value::String("2024-01-01T00:00:00Z".into()),
    );
    let id1 = kb.create_node("Message", &m1).await.unwrap();

    let mut m2 = HashMap::new();
    m2.insert("message_id".into(), Value::String("m-ue2".into()));
    m2.insert("content".into(), Value::String("b".into()));
    m2.insert(
        "timestamp".into(),
        Value::String("2024-01-01T00:00:01Z".into()),
    );
    let id2 = kb.create_node("Message", &m2).await.unwrap();

    let mut ep = HashMap::new();
    ep.insert("gap_ms".into(), Value::Int(100));
    let eid = kb.create_edge("NEXT", id1, id2, &ep).await.unwrap();

    // Update gap_ms.
    let mut updates = HashMap::new();
    updates.insert("gap_ms".into(), Value::Int(500));
    kb.update_edge(eid, &updates).await.unwrap();

    let edges = kb
        .get_edges(id1, "NEXT", Direction::Outgoing)
        .await
        .unwrap();
    assert_eq!(
        edges[0].properties.get("gap_ms").and_then(|v| v.as_i64()),
        Some(500)
    );

    kb.shutdown().await.unwrap();
}

// ── Batch ──

#[tokio::test]
async fn test_batch_create_nodes() {
    let kb = test_kb().await;
    let items: Vec<HashMap<String, Value>> = (0..50)
        .map(|i| {
            let mut props = HashMap::new();
            props.insert("chunk_id".into(), Value::String(format!("art::{i}")));
            props.insert("text".into(), Value::String(format!("chunk {i}")));
            props
        })
        .collect();

    let ids = kb.batch_create_nodes("Chunk", &items).await.unwrap();
    assert_eq!(ids.len(), 50);

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_batch_create_edges() {
    let kb = test_kb().await;

    let mut art = HashMap::new();
    art.insert("artifact_id".into(), Value::String("art-b".into()));
    art.insert("kind".into(), Value::String("file".into()));
    let art_id = kb.create_node("Artifact", &art).await.unwrap();

    let items: Vec<HashMap<String, Value>> = (0..10)
        .map(|i| {
            let mut props = HashMap::new();
            props.insert("chunk_id".into(), Value::String(format!("art-b:{i}")));
            props.insert("text".into(), Value::String(format!("chunk {i}")));
            props
        })
        .collect();
    let chunk_ids = kb.batch_create_nodes("Chunk", &items).await.unwrap();

    let edges: Vec<_> = chunk_ids
        .iter()
        .enumerate()
        .map(|(i, &cid)| {
            let mut props = HashMap::new();
            props.insert("index".into(), Value::Int(i as i64));
            (art_id, cid, props)
        })
        .collect();

    let eids = kb.batch_create_edges("HAS_CHUNK", &edges).await.unwrap();
    assert_eq!(eids.len(), 10);

    kb.shutdown().await.unwrap();
}

// ── Prepared statement & single-tx tests ──

#[tokio::test]
async fn test_prepare_basic() {
    let kb = test_kb().await;

    // Verify tx.prepare() works and doesn't hang.
    let session = kb.db().session();
    let tx = session.tx().await.unwrap();
    let prepared = tx
        .prepare("CREATE (n:Chunk {chunk_id: $cid, text: $txt}) RETURN id(n) AS vid")
        .await
        .unwrap();

    let result = prepared
        .execute(&[
            ("cid", Value::String("test-1".into())),
            ("txt", Value::String("hello".into())),
        ])
        .await
        .unwrap();
    let rows = result.rows();
    eprintln!("prepared execute returned {} rows", rows.len());
    for (i, row) in rows.iter().enumerate() {
        eprintln!("  row {i}: {:?}", row);
    }
    assert!(!rows.is_empty(), "prepared execute returned no rows");
    let vid: i64 = rows.first().unwrap().get("vid").unwrap();
    eprintln!("vid = {vid}");
    assert!(vid >= 0);

    let result2 = prepared
        .execute(&[
            ("cid", Value::String("test-2".into())),
            ("txt", Value::String("world".into())),
        ])
        .await
        .unwrap();
    let vid2: i64 = result2.rows().first().unwrap().get("vid").unwrap();
    assert!(vid2 > vid);

    tx.commit().await.unwrap();
    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_batch_create_nodes_and_edges() {
    let kb = test_kb().await;

    // Create an Artifact to wire chunk edges to.
    let mut art_props = HashMap::new();
    art_props.insert("artifact_id".into(), Value::String("art-x".into()));
    art_props.insert("kind".into(), Value::String("file".into()));
    let art_id = kb.create_node("Artifact", &art_props).await.unwrap();

    // Create chunks + HAS_CHUNK edges in a single transaction.
    let chunk_props: Vec<HashMap<String, Value>> = (0..5)
        .map(|i| {
            let mut props = HashMap::new();
            props.insert("chunk_id".into(), Value::String(format!("cx-{i}")));
            props.insert("text".into(), Value::String(format!("chunk {i}")));
            props
        })
        .collect();

    let node_ids = kb.batch_create_nodes("Chunk", &chunk_props).await.unwrap();
    assert_eq!(node_ids.len(), 5);

    let edge_specs: Vec<_> = node_ids
        .iter()
        .enumerate()
        .map(|(i, &nid)| {
            let mut props = HashMap::new();
            props.insert("index".into(), Value::Int(i as i64));
            (art_id, nid, props)
        })
        .collect();
    kb.batch_create_edges_fast("HAS_CHUNK", Some("Artifact"), Some("Chunk"), &edge_specs)
        .await
        .unwrap();

    // Verify edges exist.
    let edges = kb
        .get_edges(art_id, "HAS_CHUNK", Direction::Outgoing)
        .await
        .unwrap();
    assert_eq!(edges.len(), 5);

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_bump_modality_presence_concurrent_no_lost_update() {
    // Regression: `bump_modality_presence` used to read the current
    // map via a fresh session outside its own transaction, so two
    // concurrent bumps could race and clobber each other. Both flips
    // must persist after the dust settles.
    let kb = test_kb().await;
    kb.init_kb_stats().await.unwrap();

    let kb_a = kb.clone();
    let kb_b = kb.clone();
    let a = tokio::spawn(async move { kb_a.bump_modality_presence("image").await });
    let b = tokio::spawn(async move { kb_b.bump_modality_presence("audio").await });
    a.await.unwrap().unwrap();
    b.await.unwrap().unwrap();

    let presence = kb.read_modality_presence().await.unwrap();
    assert!(
        presence.has_image_content,
        "image flag was lost: {presence:?}"
    );
    assert!(
        presence.has_audio_content,
        "audio flag was lost: {presence:?}"
    );

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_find_stale_open_facts_preserves_upgraded_certainty() {
    // Regression: `find_stale_open_facts` used to reconstruct the
    // returned BTIC from three string-stringified columns (`toString`,
    // `btic_lo_granularity`, `btic_lo_certainty`) and fell back to
    // `Granularity::Day` / `Certainty::Approximate` on any parse hiccup.
    // The fallback could mask a genuine round-trip bug for non-default
    // meta. After switching to bare `f.valid_at AS valid_at` + typed
    // `extract_btic`, the stored meta must survive end-to-end.
    use chrono::Utc;
    use uni_db::common::uni_btic::Certainty;
    use uniko_store::schema::btic::CERTAINTY_THRESHOLD;

    let kb = test_kb().await;

    let subject = "alice";
    let predicate = "likes";
    let object = "tea";
    let observed_at = Utc::now();

    // Cross CERTAINTY_THRESHOLD so `upsert_fact_by_triple` promotes
    // the lo certainty to Definite (default on create is Approximate).
    for _ in 0..(CERTAINTY_THRESHOLD as i64 + 1) {
        kb.upsert_fact_by_triple(subject, predicate, Some(object), 1, observed_at, None)
            .await
            .unwrap();
    }

    // Find the (now stale-from-the-perspective-of-a-new-canonical)
    // fact by asking for "coffee" as the current canonical object.
    let stale = kb
        .find_stale_open_facts(subject, predicate, Some("coffee"))
        .await
        .unwrap();
    assert_eq!(stale.len(), 1, "expected one open Fact: {stale:?}");

    let (_nid, btic) = &stale[0];
    assert_eq!(
        btic.lo_certainty(),
        Certainty::Definite,
        "upgraded certainty did not survive readback: {:?}",
        btic.lo_certainty(),
    );
    assert_eq!(
        btic.lo(),
        observed_at.timestamp_millis(),
        "lo timestamp did not round-trip exactly",
    );

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_upsert_fact_by_triple_concurrent_no_lost_observation_count() {
    // Regression: `upsert_fact_by_triple` read the prior fact via a
    // fresh session and wrote back via a separate transaction.  Two
    // concurrent calls could both observe the same `prior_count` and
    // each set `new_count = prior + observation_count`, losing one
    // call's contribution.  With per-key locking the final count must
    // equal the sum of all contributions.
    use chrono::Utc;

    let kb = test_kb().await;
    let now = Utc::now();
    let subject = "alice";
    let predicate = "likes";
    let object = Some("tea");
    const SPAWNS: i64 = 16;

    let mut handles = Vec::new();
    for _ in 0..SPAWNS {
        let kb_c = kb.clone();
        handles.push(tokio::spawn(async move {
            kb_c.upsert_fact_by_triple(subject, predicate, object, 1, now, None)
                .await
                .unwrap()
        }));
    }
    let mut max_count = 0;
    for h in handles {
        let outcome = h.await.unwrap();
        max_count = max_count.max(outcome.observation_count);
    }
    assert_eq!(
        max_count, SPAWNS,
        "lost observation_count: final visible count {max_count} != {SPAWNS}",
    );

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_record_entity_invalidation_concurrent_no_lost_count() {
    // Regression: `record_entity_invalidation` reads the Entity's
    // `invalidation_count`, increments by 1, and writes back via a
    // separate transaction.  Concurrent calls on the same entity
    // could both observe the same prior count and each set
    // `prior + 1`, losing increments.
    use chrono::Utc;

    let kb = test_kb().await;
    let now = Utc::now();
    let entity = "alice";
    const SPAWNS: i64 = 16;
    const DRIFT_THRESHOLD: i64 = 1_000_000; // never trip "unstable"

    let mut handles = Vec::new();
    for _ in 0..SPAWNS {
        let kb_c = kb.clone();
        handles.push(tokio::spawn(async move {
            kb_c.record_entity_invalidation(entity, now, DRIFT_THRESHOLD)
                .await
                .unwrap()
        }));
    }
    for h in handles {
        let _ = h.await.unwrap();
    }

    // Read back the final invalidation_count. invalidation identifies the
    // entity by `name` (its entity_id is the canonical hash, not the bare
    // name), so look it up by name and assert exactly one row carries the
    // full count.
    let session = kb.db().session();
    let r = session
        .query_with(
            "MATCH (e:Entity {name: $n}) \
             RETURN count(e) AS rows, \
                    sum(coalesce(e.invalidation_count, 0)) AS total",
        )
        .param("n", entity)
        .fetch_all()
        .await
        .unwrap();
    let row = r.rows().first().expect("entity must exist");
    assert_eq!(
        row.get::<i64>("rows").unwrap(),
        1,
        "concurrent invalidations must not create duplicate Entity rows",
    );
    let final_count: i64 = row.get("total").unwrap();
    assert_eq!(
        final_count, SPAWNS,
        "lost invalidation_count: final {final_count} != {SPAWNS}",
    );

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_merge_node_concurrent_no_duplicate_creation() {
    // Regression: `merge_node` does a check-then-create (lookup via
    // `get_node_by_ext_id`, then either `update_node` or `create_node`).
    // Two concurrent callers could both observe "not present" and each
    // create a separate row for the same `ext_id`.  With per-key
    // locking exactly one row must exist after the dust settles.
    let kb = test_kb().await;
    let ext_id = "p-merge-race";
    let label = "Participant";

    const SPAWNS: usize = 16;
    let mut handles = Vec::new();
    for _ in 0..SPAWNS {
        let kb_c = kb.clone();
        handles.push(tokio::spawn(async move {
            let mut props = HashMap::new();
            props.insert("participant_id".into(), Value::String(ext_id.into()));
            props.insert("kind".into(), Value::String("agent".into()));
            kb_c.merge_node(label, "participant_id", ext_id, &props)
                .await
                .unwrap()
        }));
    }
    let mut ids = Vec::new();
    for h in handles {
        ids.push(h.await.unwrap());
    }
    // All merges must resolve to the same node id.
    let first = ids[0];
    assert!(
        ids.iter().all(|&id| id == first),
        "merge_node produced distinct node ids under contention: {ids:?}",
    );

    // Verify only one node was actually created.
    let session = kb.db().session();
    let q = session
        .query_with(&format!(
            "MATCH (n:{label} {{participant_id: $eid}}) RETURN count(n) AS c"
        ))
        .param("eid", ext_id)
        .fetch_all()
        .await
        .unwrap();
    let count: i64 = q.rows().first().unwrap().get("c").unwrap();
    assert_eq!(count, 1, "merge_node created {count} duplicate rows");

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_batch_upsert_facts_concurrent_no_lost_count() {
    // Regression: `batch_upsert_facts` reads prior counts via one
    // session and writes back via a separate transaction.  Two
    // concurrent batches targeting overlapping Facts could both
    // observe the same pre-image and lose increments.  With per-Fact
    // locking the cumulative count must equal the sum of all batches'
    // contributions.
    use chrono::Utc;
    use uniko_store::operations::facts::FactUpsertInput;

    let kb = test_kb().await;
    let now = Utc::now();
    let subject = "alice".to_string();
    let predicate = "likes".to_string();
    let object = "tea".to_string();
    const BATCHES: i64 = 8;
    const PER_BATCH: i64 = 3;

    let mut handles = Vec::new();
    for _ in 0..BATCHES {
        let kb_c = kb.clone();
        let s = subject.clone();
        let p = predicate.clone();
        let o = object.clone();
        handles.push(tokio::spawn(async move {
            let inputs: Vec<FactUpsertInput> = (0..PER_BATCH)
                .map(|_| FactUpsertInput {
                    subject: s.clone(),
                    predicate: p.clone(),
                    object: Some(o.clone()),
                    observation_count: 1,
                    observed_at: now,
                    embedding: None,
                    visibility: None,
                })
                .collect();
            // Each batch internally dedups to a single Fact with
            // observation_count == PER_BATCH; concurrent batches must
            // serialize so the cumulative count is BATCHES * PER_BATCH.
            kb_c.batch_upsert_facts(inputs).await.unwrap()
        }));
    }
    for h in handles {
        let _ = h.await.unwrap();
    }

    // Read back the surviving Fact's observation_count.
    let fact_id = uniko_store::operations::facts::fact_id_for(&subject, &predicate, Some(&object));
    let (_nid, props) = kb
        .get_node_by_ext_id("Fact", "fact_id", &fact_id)
        .await
        .unwrap()
        .expect("Fact must exist");
    let final_count = props
        .get("observation_count")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    assert_eq!(
        final_count,
        BATCHES * PER_BATCH,
        "lost batch upserts: final observation_count {final_count} != {}",
        BATCHES * PER_BATCH,
    );

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_batch_create_edges_fast_rejects_invalid_property_name() {
    // Regression: the fast/bulk path used to silently discard the
    // result of `validate_property_name`, letting invalid keys reach
    // uni-db's `bulk_insert_edges`. It must now surface a Schema error.
    let kb = test_kb().await;

    let mut props_a = HashMap::new();
    props_a.insert("artifact_id".into(), Value::String("art-bad".into()));
    props_a.insert("kind".into(), Value::String("file".into()));
    let art_id = kb.create_node("Artifact", &props_a).await.unwrap();

    let mut props_c = HashMap::new();
    props_c.insert("chunk_id".into(), Value::String("cx-bad".into()));
    props_c.insert("text".into(), Value::String("c".into()));
    let chunk_id = kb.create_node("Chunk", &props_c).await.unwrap();

    let mut bad_edge_props = HashMap::new();
    bad_edge_props.insert("".into(), Value::Int(1));
    let edge_specs = vec![(art_id, chunk_id, bad_edge_props)];

    let err = kb
        .batch_create_edges_fast("HAS_CHUNK", Some("Artifact"), Some("Chunk"), &edge_specs)
        .await
        .expect_err("empty property name must be rejected");
    assert!(
        matches!(err, uniko_store::UnikoError::Schema(_)),
        "expected Schema error, got {err:?}"
    );

    kb.shutdown().await.unwrap();
}

// ── #3: typed SSI errors + reachable retry ──

/// A retriable uni-db SSI conflict must cross the `?` boundary as a
/// `UnikoError::Conflict` (retriable), not an opaque `Storage(String)`.
///
/// Induced via uni-db's read-write antidependency interleave: `tx_a`
/// reads `ctr`, a concurrent `tx_b` writes+commits `ctr`, then `tx_a`
/// writes and commits — which must abort with `SerializationConflict`.
/// uniko runs with `ssi_enabled` (uni-db default), so this is live.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_ssi_conflict_maps_to_retriable_conflict_error() {
    let kb = test_kb().await;
    let mut seed = HashMap::new();
    seed.insert("participant_id".into(), Value::String("ctr".into()));
    seed.insert("kind".into(), Value::String("agent".into()));
    seed.insert("n".into(), Value::Int(0));
    kb.create_node("Participant", &seed).await.unwrap();

    let session_a = kb.db().session();
    let tx_a = session_a.tx().await.unwrap();
    // tx_a reads ctr — recorded in its read-set.
    tx_a.query_with("MATCH (p:Participant {participant_id: 'ctr'}) RETURN p.n")
        .fetch_all()
        .await
        .unwrap();

    // A concurrent transaction writes ctr and commits.
    {
        let session_b = kb.db().session();
        let tx_b = session_b.tx().await.unwrap();
        tx_b.execute_with("MATCH (p:Participant {participant_id: 'ctr'}) SET p.n = 100")
            .run()
            .await
            .unwrap();
        tx_b.commit().await.unwrap();
    }

    // tx_a now writes and commits — a read-write antidependency on ctr
    // forces an abort. Map the raw uni-db error through uniko's `From`.
    tx_a.execute_with("CREATE (:Participant {participant_id: 'ctr2', kind: 'agent'})")
        .run()
        .await
        .unwrap();
    let commit_err = tx_a
        .commit()
        .await
        .expect_err("antidependency commit must abort under SSI");
    let mapped: uniko_store::UnikoError = commit_err.into();
    assert!(
        matches!(mapped, uniko_store::UnikoError::Conflict(_)),
        "expected retriable Conflict, got: {mapped:?}"
    );
    assert!(mapped.is_retriable(), "Conflict must report is_retriable()");

    kb.shutdown().await.unwrap();
}

/// #1 migration: `migrate_entity_ids_to_canonical` recomputes every
/// Entity's id to the canonical scheme, MERGES rows that collapse to the
/// same id (summing frequency, repointing edges onto one survivor), and
/// drops Topics for re-derivation. Idempotent.
#[tokio::test]
async fn test_migrate_entity_ids_merges_divergent_schemes() {
    let kb = test_kb().await;

    // Two legacy rows for the same logical entity (alice/person) under the
    // old divergent schemes: dedup's `type:name` and action's uppercase id.
    let mut a = HashMap::new();
    a.insert("entity_id".into(), Value::String("person:alice".into()));
    a.insert("name".into(), Value::String("alice".into()));
    a.insert("entity_type".into(), Value::String("person".into()));
    a.insert("frequency".into(), Value::Int(2));
    let nid_a = kb.create_node("Entity", &a).await.unwrap();

    let mut b = HashMap::new();
    b.insert("entity_id".into(), Value::String("ent_deadbeef".into()));
    b.insert("name".into(), Value::String("alice".into()));
    b.insert("entity_type".into(), Value::String("PERSON".into())); // legacy upper
    b.insert("frequency".into(), Value::Int(3));
    let nid_b = kb.create_node("Entity", &b).await.unwrap();

    // A distinct entity that must survive untouched (different name).
    let mut c = HashMap::new();
    c.insert("entity_id".into(), Value::String("org:acme".into()));
    c.insert("name".into(), Value::String("acme".into()));
    c.insert("entity_type".into(), Value::String("organization".into()));
    kb.create_node("Entity", &c).await.unwrap();

    // MENTIONS edges onto BOTH rows (from distinct messages) — after the
    // merge both must land on the single survivor. (The repoint code path is
    // identical for ABOUT and BELONGS_TO.)
    let ts = uniko_store::types::datetime_value(chrono::Utc::now());
    let mut m1 = HashMap::new();
    m1.insert("message_id".into(), Value::String("m1".into()));
    m1.insert("content".into(), Value::String("hi alice".into()));
    m1.insert("timestamp".into(), ts.clone());
    let msg1 = kb.create_node("Message", &m1).await.unwrap();
    kb.create_edge("MENTIONS", msg1, nid_a, &HashMap::new())
        .await
        .unwrap();
    let mut m2 = HashMap::new();
    m2.insert("message_id".into(), Value::String("m2".into()));
    m2.insert("content".into(), Value::String("bye alice".into()));
    m2.insert("timestamp".into(), ts);
    let msg2 = kb.create_node("Message", &m2).await.unwrap();
    kb.create_edge("MENTIONS", msg2, nid_b, &HashMap::new())
        .await
        .unwrap();

    // A Topic that must be dropped.
    let mut t = HashMap::new();
    t.insert("topic_id".into(), Value::String("topic_x".into()));
    t.insert("name".into(), Value::String("a topic".into()));
    kb.create_node("Topic", &t).await.unwrap();

    let report = kb.migrate_entity_ids_to_canonical().await.unwrap();
    assert_eq!(report.scanned, 3);

    let session = kb.db().session();
    // alice collapsed to ONE row with summed frequency.
    let canonical = uniko_store::id::entity_id("alice", "person");
    let r = session
        .query_with(
            "MATCH (e:Entity {name: 'alice'}) \
             RETURN count(e) AS c, sum(coalesce(e.frequency,0)) AS f, \
                    collect(DISTINCT e.entity_id) AS ids",
        )
        .fetch_all()
        .await
        .unwrap();
    let row = r.rows().first().unwrap();
    assert_eq!(
        row.get::<i64>("c").unwrap(),
        1,
        "alice rows must collapse to 1"
    );
    assert_eq!(row.get::<i64>("f").unwrap(), 5, "frequency must sum (2+3)");

    // Both edges repointed onto the single survivor.
    let surv = session
        .query_with("MATCH (e:Entity {name: 'alice'}) RETURN id(e) AS vid")
        .fetch_all()
        .await
        .unwrap();
    let survivor: i64 = surv.rows().first().unwrap().get("vid").unwrap();
    assert!(survivor == nid_a || survivor == nid_b);
    let edges = session
        .query_with(
            "MATCH (:Message)-[r:MENTIONS]->(e:Entity {name: 'alice'}) \
             RETURN count(r) AS m",
        )
        .fetch_all()
        .await
        .unwrap();
    let er = edges.rows().first().unwrap();
    assert_eq!(
        er.get::<i64>("m").unwrap(),
        2,
        "both MENTIONS edges must repoint to the single survivor"
    );

    // Canonical id is set; Topics dropped.
    let idrow = session
        .query_with("MATCH (e:Entity {name:'alice'}) RETURN e.entity_id AS eid")
        .fetch_all()
        .await
        .unwrap();
    assert_eq!(
        idrow.rows().first().unwrap().get::<String>("eid").unwrap(),
        canonical
    );
    let topics = session
        .query_with("MATCH (t:Topic) RETURN count(t) AS c")
        .fetch_all()
        .await
        .unwrap();
    assert_eq!(topics.rows().first().unwrap().get::<i64>("c").unwrap(), 0);

    // Idempotent: a second pass migrates nothing new.
    let report2 = kb.migrate_entity_ids_to_canonical().await.unwrap();
    assert_eq!(report2.migrated, 0, "second run must be a no-op");

    kb.shutdown().await.unwrap();
}

/// #6: `merge_artifact_content` must be idempotent under contention.
/// 16 concurrent callers with the same `content_id` must resolve to one
/// `ArtifactContent` row (not N duplicates) and all return the same id.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_merge_artifact_content_concurrent_single_row() {
    use uniko_store::storage::blob::MergeContent;
    let kb = test_kb().await;
    let content_id = KnowledgeBase::sha256_hex(b"the same content");

    const SPAWNS: usize = 16;
    let mut handles = Vec::new();
    for _ in 0..SPAWNS {
        let kb_c = kb.clone();
        let cid = content_id.clone();
        handles.push(tokio::spawn(async move {
            kb_c.merge_artifact_content(MergeContent {
                content_id: cid,
                bytes: Some(b"the same content".to_vec()),
                uri: None,
                mime: "text/plain".into(),
                size: 16,
                perceptual_hash: None,
                audio_fingerprint: None,
            })
            .await
            .unwrap()
        }));
    }
    let mut ids = Vec::new();
    for h in handles {
        ids.push(h.await.unwrap());
    }
    let first = ids[0];
    assert!(
        ids.iter().all(|&id| id == first),
        "merge_artifact_content produced distinct node ids under contention: {ids:?}",
    );

    let session = kb.db().session();
    let q = session
        .query_with("MATCH (c:ArtifactContent {content_id: $cid}) RETURN count(c) AS c")
        .param("cid", content_id)
        .fetch_all()
        .await
        .unwrap();
    let count: i64 = q.rows().first().unwrap().get("c").unwrap();
    assert_eq!(
        count, 1,
        "merge_artifact_content created {count} duplicate rows"
    );

    kb.shutdown().await.unwrap();
}

/// `transact_with_retry` must converge concurrent read-modify-write
/// increments under SSI contention with no lost updates: 16 writers each
/// do `n = n + 1`; conflicting commits are retried until the final value
/// equals the writer count.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_transact_with_retry_converges_under_contention() {
    let kb = test_kb().await;
    let mut seed = HashMap::new();
    seed.insert("participant_id".into(), Value::String("hot".into()));
    seed.insert("kind".into(), Value::String("agent".into()));
    seed.insert("n".into(), Value::Int(0));
    kb.create_node("Participant", &seed).await.unwrap();

    const WRITERS: i64 = 16;
    let mut handles = Vec::new();
    for _ in 0..WRITERS {
        let kb_c = kb.clone();
        handles.push(tokio::spawn(async move {
            kb_c.transact_with_retry(
                uni_db::RetryOptions {
                    max_attempts: 200,
                    ..Default::default()
                },
                move |tx| async move {
                    let r = async {
                        tx.execute_with(
                            "MATCH (p:Participant {participant_id: 'hot'}) SET p.n = p.n + 1",
                        )
                        .run()
                        .await?;
                        Ok(())
                    }
                    .await;
                    (tx, r)
                },
            )
            .await
        }));
    }
    for h in handles {
        h.await.unwrap().unwrap();
    }

    let session = kb.db().session();
    let q = session
        .query_with("MATCH (p:Participant {participant_id: 'hot'}) RETURN p.n AS n")
        .fetch_all()
        .await
        .unwrap();
    let n: i64 = q.rows().first().unwrap().get("n").unwrap();
    assert_eq!(n, WRITERS, "lost updates under contention: n={n}");

    kb.shutdown().await.unwrap();
}
