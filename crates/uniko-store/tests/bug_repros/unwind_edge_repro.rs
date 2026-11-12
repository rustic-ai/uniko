//! Minimal repro: UNWIND + MATCH WHERE id() = param is ~100x slower
//! than an equivalent loop for edge creation.
//!
//! Filed as https://github.com/rustic-ai/uni-db/issues/53

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

/// Create N source nodes and 1 target node, then compare loop vs UNWIND
/// for creating N edges between them.
#[tokio::test]
async fn repro_unwind_match_id_slow() {
    let kb = test_kb().await;
    let db = kb.db();
    let session = db.session();
    let n = 600;

    // --- Setup: create N source nodes + 1 target node ---
    let tx = session.tx().await.unwrap();

    let target = tx
        .query_with(
            "CREATE (m:Message {message_id: $mid, content: $c, timestamp: $ts}) \
             RETURN id(m) AS vid",
        )
        .param("mid", Value::String("target".into()))
        .param("c", Value::String("target node".into()))
        .param("ts", Value::String("2024-01-01T00:00:00Z".into()))
        .fetch_all()
        .await
        .unwrap();
    let target_id: i64 = target.rows()[0].get("vid").unwrap();

    let items: Vec<Value> = (0..n)
        .map(|i| {
            let mut m = HashMap::new();
            m.insert("observation_id".into(), Value::String(format!("o-{i}")));
            m.insert("content".into(), Value::String(format!("obs {i}")));
            m.insert("subject".into(), Value::String("user".into()));
            m.insert(
                "observed_at".into(),
                Value::String("2024-01-01T00:00:00Z".into()),
            );
            m.insert("confidence".into(), Value::Float(0.9));
            Value::Map(m)
        })
        .collect();

    let nodes = tx
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
    let src_ids: Vec<i64> = nodes
        .rows()
        .iter()
        .map(|r| r.get::<i64>("vid").unwrap())
        .collect();
    assert_eq!(src_ids.len(), n);

    tx.commit().await.unwrap();

    // --- Approach 1: Loop with parameterized query ---
    let tx1 = session.tx().await.unwrap();
    let t0 = Instant::now();
    for &src in &src_ids {
        tx1.query_with(
            "MATCH (a), (b) WHERE id(a) = $src AND id(b) = $dst \
             CREATE (a)-[r:OBSERVED_IN]->(b) RETURN id(r) AS eid",
        )
        .param("src", Value::Int(src))
        .param("dst", Value::Int(target_id))
        .fetch_all()
        .await
        .unwrap();
    }
    tx1.commit().await.unwrap();
    let loop_ms = t0.elapsed().as_millis();

    // --- Approach 2: UNWIND with MATCH WHERE id() ---
    // Same edge type as approach 1: the sources are Observations and the
    // target is a Message, which only OBSERVED_IN declares. uni-db 5.0
    // enforces declared endpoints on write, and holding the type constant is
    // what the loop-vs-UNWIND comparison wants anyway.
    let tx2 = session.tx().await.unwrap();
    let edges: Vec<Value> = src_ids
        .iter()
        .map(|&src| {
            let mut m = HashMap::new();
            m.insert("src".into(), Value::Int(src));
            m.insert("dst".into(), Value::Int(target_id));
            Value::Map(m)
        })
        .collect();

    let t0 = Instant::now();
    let result = tx2
        .query_with(
            "UNWIND $edges AS e \
             MATCH (a), (b) WHERE id(a) = e.src AND id(b) = e.dst \
             CREATE (a)-[r:OBSERVED_IN]->(b) \
             RETURN id(r) AS eid",
        )
        .param("edges", Value::List(edges))
        .fetch_all()
        .await
        .unwrap();
    tx2.commit().await.unwrap();
    let unwind_ms = t0.elapsed().as_millis();

    assert_eq!(result.rows().len(), n);

    eprintln!("=== {n} edge creation: MATCH WHERE id() ===");
    eprintln!("  Loop:   {loop_ms}ms");
    eprintln!("  UNWIND: {unwind_ms}ms");
    eprintln!(
        "  Ratio:  UNWIND is {:.1}x {}",
        if unwind_ms > loop_ms {
            unwind_ms as f64 / loop_ms as f64
        } else {
            loop_ms as f64 / unwind_ms as f64
        },
        if unwind_ms > loop_ms {
            "SLOWER"
        } else {
            "faster"
        }
    );

    // The bug: UNWIND should be faster (or at least comparable),
    // but is observed to be ~100x slower.
    // Uncomment the assertion below once fixed:
    // assert!(unwind_ms < loop_ms * 5, "UNWIND should not be >5x slower than loop");

    // --- Approach 3: Prepared statement loop ---
    let tx3 = session.tx().await.unwrap();
    let prepared = tx3
        .prepare(
            "MATCH (a), (b) WHERE id(a) = $src AND id(b) = $dst \
             CREATE (a)-[r:OBSERVED_IN]->(b) RETURN id(r) AS eid",
        )
        .await
        .unwrap();

    let t0 = Instant::now();
    for &src in &src_ids {
        prepared
            .execute(&[("src", Value::Int(src)), ("dst", Value::Int(target_id))])
            .await
            .unwrap();
    }
    tx3.commit().await.unwrap();
    let prepared_ms = t0.elapsed().as_millis();

    eprintln!("\n=== {n} edge creation: prepared statement loop ===");
    eprintln!("  query_with loop: {loop_ms}ms");
    eprintln!("  prepared loop:   {prepared_ms}ms");
    eprintln!(
        "  Ratio: prepared is {:.1}x {}",
        if prepared_ms < loop_ms {
            loop_ms as f64 / prepared_ms as f64
        } else {
            prepared_ms as f64 / loop_ms as f64
        },
        if prepared_ms < loop_ms {
            "faster"
        } else {
            "SLOWER"
        }
    );

    kb.shutdown().await.unwrap();
}
