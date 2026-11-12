//! Isolated test for uni-db UNWIND batch operations.
//!
//! Tests whether UNWIND can be used for bulk node and edge creation,
//! reproducing the observation ingestion pattern:
//!   - Create N Observation nodes per turn (typically 5-30)
//!   - Create N OBSERVED_IN edges (one per observation → source message)
//!   - Create M ABOUT edges (observations → entities)
//!
//! This test uses only uni-db APIs (no KnowledgeBase wrapper) to isolate
//! the database behavior.

use std::collections::HashMap;
use std::time::Instant;

use uni_db::Value;
use uniko_store::config::UnikoConfig;
use uniko_store::storage::KnowledgeBase;

async fn test_kb() -> KnowledgeBase {
    KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB")
}

/// Test basic UNWIND node creation with uni-db directly.
#[tokio::test]
async fn test_unwind_create_nodes() {
    let kb = test_kb().await;
    let db = kb.db();
    let session = db.session();
    let tx = session.tx().await.unwrap();

    // Build 20 observation-like maps (typical per-turn count).
    let items: Vec<Value> = (0..20)
        .map(|i| {
            let mut m = HashMap::new();
            m.insert(
                "observation_id".to_string(),
                Value::String(format!("obs-{i}")),
            );
            m.insert(
                "content".to_string(),
                Value::String(format!("User likes item {i}")),
            );
            m.insert("subject".to_string(), Value::String("user".into()));
            m.insert(
                "observed_at".to_string(),
                Value::String("2024-01-01T00:00:00Z".into()),
            );
            m.insert("confidence".to_string(), Value::Float(0.9));
            Value::Map(m)
        })
        .collect();

    let t0 = Instant::now();
    let result = tx
        .query_with(
            "UNWIND $items AS item \
             CREATE (n:Observation { \
               observation_id: item.observation_id, \
               content: item.content, \
               subject: item.subject, \
               observed_at: item.observed_at, \
               confidence: item.confidence \
             }) RETURN id(n) AS vid",
        )
        .param("items", Value::List(items))
        .fetch_all()
        .await
        .unwrap();
    let elapsed = t0.elapsed();

    let node_ids: Vec<i64> = result
        .rows()
        .iter()
        .map(|row| row.get::<i64>("vid").unwrap())
        .collect();

    assert_eq!(node_ids.len(), 20, "expected 20 nodes created");
    eprintln!(
        "UNWIND CREATE 20 nodes: {:?} ({} nodes returned)",
        elapsed,
        node_ids.len()
    );

    tx.commit().await.unwrap();
    kb.shutdown().await.unwrap();
}

/// Test UNWIND edge creation with MATCH by id().
#[tokio::test]
async fn test_unwind_create_edges() {
    let kb = test_kb().await;
    let db = kb.db();
    let session = db.session();

    // First create a source Message node and some Observation nodes.
    let tx = session.tx().await.unwrap();

    let msg_result = tx
        .query_with(
            "CREATE (m:Message {message_id: $mid, content: $content, timestamp: $ts}) RETURN id(m) AS vid",
        )
        .param("mid", Value::String("msg-1".into()))
        .param("content", Value::String("hello world".into()))
        .param("ts", Value::String("2024-01-01T00:00:00Z".into()))
        .fetch_all()
        .await
        .unwrap();
    let msg_id: i64 = msg_result.rows()[0].get("vid").unwrap();

    // Create 20 observation nodes individually.
    let mut obs_ids = Vec::new();
    for i in 0..20 {
        let r = tx
            .query_with(
                "CREATE (n:Observation { \
                   observation_id: $oid, content: $content, subject: $subj, \
                   observed_at: $at, confidence: $conf \
                 }) RETURN id(n) AS vid",
            )
            .param("oid", Value::String(format!("obs-{i}")))
            .param("content", Value::String(format!("observation {i}")))
            .param("subj", Value::String("user".into()))
            .param("at", Value::String("2024-01-01T00:00:00Z".into()))
            .param("conf", Value::Float(0.9))
            .fetch_all()
            .await
            .unwrap();
        obs_ids.push(r.rows()[0].get::<i64>("vid").unwrap());
    }
    tx.commit().await.unwrap();

    // Now test UNWIND edge creation.
    let tx2 = session.tx().await.unwrap();
    let edges: Vec<Value> = obs_ids
        .iter()
        .map(|&oid| {
            let mut m = HashMap::new();
            m.insert("src".to_string(), Value::Int(oid));
            m.insert("dst".to_string(), Value::Int(msg_id));
            Value::Map(m)
        })
        .collect();

    let t0 = Instant::now();
    let result = tx2
        .query_with(
            "UNWIND $edges AS edge \
             MATCH (a), (b) WHERE id(a) = edge.src AND id(b) = edge.dst \
             CREATE (a)-[r:OBSERVED_IN]->(b) \
             RETURN id(r) AS eid",
        )
        .param("edges", Value::List(edges))
        .fetch_all()
        .await
        .unwrap();
    let elapsed = t0.elapsed();

    let edge_ids: Vec<i64> = result
        .rows()
        .iter()
        .map(|row| row.get::<i64>("eid").unwrap())
        .collect();

    assert_eq!(edge_ids.len(), 20, "expected 20 edges created");
    eprintln!(
        "UNWIND CREATE 20 edges: {:?} ({} edges returned)",
        elapsed,
        edge_ids.len()
    );

    tx2.commit().await.unwrap();
    kb.shutdown().await.unwrap();
}

/// Test UNWIND at realistic scale: 20 nodes + 20 OBSERVED_IN edges + 20 ABOUT
/// edges in a single transaction (simulating one observation extraction turn).
#[tokio::test]
async fn test_unwind_full_observation_turn() {
    let kb = test_kb().await;
    let db = kb.db();
    let session = db.session();

    // Setup: create a Message node and an Entity node.
    let tx = session.tx().await.unwrap();
    let msg_result = tx
        .query_with(
            "CREATE (m:Message {message_id: $mid, content: $c, timestamp: $ts}) RETURN id(m) AS vid",
        )
        .param("mid", Value::String("msg-turn".into()))
        .param("c", Value::String("test message".into()))
        .param("ts", Value::String("2024-01-01T00:00:00Z".into()))
        .fetch_all()
        .await
        .unwrap();
    let msg_id: i64 = msg_result.rows()[0].get("vid").unwrap();

    let ent_result = tx
        .query_with(
            "CREATE (e:Entity {entity_id: $eid, name: $name, entity_type: $t}) RETURN id(e) AS vid",
        )
        .param("eid", Value::String("ent-alice".into()))
        .param("name", Value::String("Alice".into()))
        .param("t", Value::String("person".into()))
        .fetch_all()
        .await
        .unwrap();
    let entity_id: i64 = ent_result.rows()[0].get("vid").unwrap();
    tx.commit().await.unwrap();

    // --- The actual test: UNWIND-based batch creation ---
    let tx2 = session.tx().await.unwrap();

    // Step 1: UNWIND to create 20 Observation nodes.
    let obs_data: Vec<Value> = (0..20)
        .map(|i| {
            let mut m = HashMap::new();
            m.insert("observation_id".into(), Value::String(format!("ot-{i}")));
            m.insert(
                "content".into(),
                Value::String(format!("Alice likes item {i}")),
            );
            m.insert("subject".into(), Value::String("Alice".into()));
            m.insert(
                "observed_at".into(),
                Value::String("2024-01-01T00:00:00Z".into()),
            );
            m.insert("confidence".into(), Value::Float(0.85));
            Value::Map(m)
        })
        .collect();

    let t0 = Instant::now();

    let node_result = tx2
        .query_with(
            "UNWIND $items AS item \
             CREATE (n:Observation { \
               observation_id: item.observation_id, \
               content: item.content, \
               subject: item.subject, \
               observed_at: item.observed_at, \
               confidence: item.confidence \
             }) RETURN id(n) AS vid",
        )
        .param("items", Value::List(obs_data))
        .fetch_all()
        .await
        .unwrap();

    let obs_ids: Vec<i64> = node_result
        .rows()
        .iter()
        .map(|row| row.get::<i64>("vid").unwrap())
        .collect();
    assert_eq!(obs_ids.len(), 20);

    // Step 2: UNWIND to create OBSERVED_IN edges.
    let observed_edges: Vec<Value> = obs_ids
        .iter()
        .map(|&oid| {
            let mut m = HashMap::new();
            m.insert("src".into(), Value::Int(oid));
            m.insert("dst".into(), Value::Int(msg_id));
            Value::Map(m)
        })
        .collect();

    let edge_result = tx2
        .query_with(
            "UNWIND $edges AS e \
             MATCH (a), (b) WHERE id(a) = e.src AND id(b) = e.dst \
             CREATE (a)-[r:OBSERVED_IN]->(b) \
             RETURN id(r) AS eid",
        )
        .param("edges", Value::List(observed_edges))
        .fetch_all()
        .await
        .unwrap();
    assert_eq!(edge_result.rows().len(), 20);

    // Step 3: UNWIND to create ABOUT edges.
    let about_edges: Vec<Value> = obs_ids
        .iter()
        .map(|&oid| {
            let mut m = HashMap::new();
            m.insert("src".into(), Value::Int(oid));
            m.insert("dst".into(), Value::Int(entity_id));
            Value::Map(m)
        })
        .collect();

    let about_result = tx2
        .query_with(
            "UNWIND $edges AS e \
             MATCH (a), (b) WHERE id(a) = e.src AND id(b) = e.dst \
             CREATE (a)-[r:ABOUT]->(b) \
             RETURN id(r) AS eid",
        )
        .param("edges", Value::List(about_edges))
        .fetch_all()
        .await
        .unwrap();
    assert_eq!(about_result.rows().len(), 20);

    tx2.commit().await.unwrap();
    let elapsed = t0.elapsed();

    eprintln!(
        "Full UNWIND turn (20 nodes + 20 OBSERVED_IN + 20 ABOUT, single tx): {:?}",
        elapsed
    );

    // Verify everything persisted.
    let count = session
        .query("MATCH (n:Observation) RETURN count(n) AS c")
        .await
        .unwrap();
    assert_eq!(count.rows()[0].get::<i64>("c").unwrap(), 20);

    let edge_count = session
        .query("MATCH ()-[r:OBSERVED_IN]->() RETURN count(r) AS c")
        .await
        .unwrap();
    assert_eq!(edge_count.rows()[0].get::<i64>("c").unwrap(), 20);

    let about_count = session
        .query("MATCH ()-[r:ABOUT]->() RETURN count(r) AS c")
        .await
        .unwrap();
    assert_eq!(about_count.rows()[0].get::<i64>("c").unwrap(), 20);

    kb.shutdown().await.unwrap();
}

/// Comparison: loop-based creation vs UNWIND at realistic scale.
/// 300 nodes + 800 edges (simulating a session's worth of observations).
#[tokio::test]
async fn test_unwind_vs_loop_performance() {
    let kb = test_kb().await;
    let db = kb.db();
    let session = db.session();
    let n = 300;

    // --- Loop-based: one CREATE per node ---
    let tx = session.tx().await.unwrap();
    let t0 = Instant::now();
    let mut loop_ids = Vec::with_capacity(n);
    for i in 0..n {
        let r = tx
            .query_with(
                "CREATE (n:Observation { \
                   observation_id: $oid, content: $c, subject: $s, \
                   observed_at: $at, confidence: $conf \
                 }) RETURN id(n) AS vid",
            )
            .param("oid", Value::String(format!("loop-{i}")))
            .param("c", Value::String(format!("loop content {i}")))
            .param("s", Value::String("user".into()))
            .param("at", Value::String("2024-01-01T00:00:00Z".into()))
            .param("conf", Value::Float(0.9))
            .fetch_all()
            .await
            .unwrap();
        loop_ids.push(r.rows()[0].get::<i64>("vid").unwrap());
    }
    tx.commit().await.unwrap();
    let loop_elapsed = t0.elapsed();

    // --- UNWIND-based: single query ---
    let tx2 = session.tx().await.unwrap();
    let items: Vec<Value> = (0..n)
        .map(|i| {
            let mut m = HashMap::new();
            m.insert(
                "observation_id".into(),
                Value::String(format!("unwind-{i}")),
            );
            m.insert(
                "content".into(),
                Value::String(format!("unwind content {i}")),
            );
            m.insert("subject".into(), Value::String("user".into()));
            m.insert(
                "observed_at".into(),
                Value::String("2024-01-01T00:00:00Z".into()),
            );
            m.insert("confidence".into(), Value::Float(0.9));
            Value::Map(m)
        })
        .collect();

    let t0 = Instant::now();
    let result = tx2
        .query_with(
            "UNWIND $items AS item \
             CREATE (n:Observation { \
               observation_id: item.observation_id, \
               content: item.content, \
               subject: item.subject, \
               observed_at: item.observed_at, \
               confidence: item.confidence \
             }) RETURN id(n) AS vid",
        )
        .param("items", Value::List(items))
        .fetch_all()
        .await
        .unwrap();
    tx2.commit().await.unwrap();
    let unwind_elapsed = t0.elapsed();

    assert_eq!(result.rows().len(), n);
    assert_eq!(loop_ids.len(), n);

    eprintln!("=== {n} node creation comparison ===");
    eprintln!("  Loop:   {:?}", loop_elapsed);
    eprintln!("  UNWIND: {:?}", unwind_elapsed);
    eprintln!(
        "  Speedup: {:.1}x",
        loop_elapsed.as_secs_f64() / unwind_elapsed.as_secs_f64()
    );

    // --- Now test 800 edges ---
    // Create a target message node.
    let tx3 = session.tx().await.unwrap();
    let msg_r = tx3
        .query_with(
            "CREATE (m:Message {message_id: $mid, content: $c, timestamp: $ts}) RETURN id(m) AS vid",
        )
        .param("mid", Value::String("msg-perf".into()))
        .param("c", Value::String("perf test".into()))
        .param("ts", Value::String("2024-01-01T00:00:00Z".into()))
        .fetch_all()
        .await
        .unwrap();
    let msg_id: i64 = msg_r.rows()[0].get("vid").unwrap();
    tx3.commit().await.unwrap();

    // Collect all observation node IDs from both loop and unwind runs.
    let all_obs = session
        .query("MATCH (n:Observation) RETURN id(n) AS vid")
        .await
        .unwrap();
    let all_obs_ids: Vec<i64> = all_obs
        .rows()
        .iter()
        .map(|r| r.get::<i64>("vid").unwrap())
        .collect();
    // Take up to 800 for the edge test (we have 600 from the node tests).
    let edge_n = 800.min(all_obs_ids.len());
    let obs_for_edges = &all_obs_ids[..edge_n];

    // Loop-based edge creation.
    let tx4 = session.tx().await.unwrap();
    let t0 = Instant::now();
    for &oid in obs_for_edges {
        tx4.query_with(
            "MATCH (a), (b) WHERE id(a) = $src AND id(b) = $dst \
             CREATE (a)-[r:OBSERVED_IN]->(b) RETURN id(r) AS eid",
        )
        .param("src", Value::Int(oid))
        .param("dst", Value::Int(msg_id))
        .fetch_all()
        .await
        .unwrap();
    }
    tx4.commit().await.unwrap();
    let loop_edge_elapsed = t0.elapsed();

    // UNWIND-based edge creation. Same edge type as the loop arm above:
    // these are Observation -> Message edges, which only OBSERVED_IN declares,
    // and uni-db 5.0 enforces declared endpoints on write.
    let tx5 = session.tx().await.unwrap();
    let edge_list: Vec<Value> = obs_for_edges
        .iter()
        .map(|&oid| {
            let mut m = HashMap::new();
            m.insert("src".into(), Value::Int(oid));
            m.insert("dst".into(), Value::Int(msg_id));
            Value::Map(m)
        })
        .collect();

    let t0 = Instant::now();
    let result = tx5
        .query_with(
            "UNWIND $edges AS e \
             MATCH (a), (b) WHERE id(a) = e.src AND id(b) = e.dst \
             CREATE (a)-[r:OBSERVED_IN]->(b) \
             RETURN id(r) AS eid",
        )
        .param("edges", Value::List(edge_list))
        .fetch_all()
        .await
        .unwrap();
    tx5.commit().await.unwrap();
    let unwind_edge_elapsed = t0.elapsed();

    assert_eq!(result.rows().len(), edge_n);

    eprintln!("\n=== {edge_n} edge creation comparison ===");
    eprintln!("  Loop:   {:?}", loop_edge_elapsed);
    eprintln!("  UNWIND: {:?}", unwind_edge_elapsed);
    eprintln!(
        "  Speedup: {:.1}x",
        loop_edge_elapsed.as_secs_f64() / unwind_edge_elapsed.as_secs_f64()
    );

    kb.shutdown().await.unwrap();
}
