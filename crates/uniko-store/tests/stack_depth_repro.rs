//! Isolates which query path consumes ~2 MiB of stack.
//!
//! `uniko-bench::smoke_test test_ingest_and_recall` aborts with a stack
//! overflow in roughly half of full-suite runs, and deterministically at
//! `RUST_MIN_STACK=1048576` while passing at the 2 MiB default. So the recall
//! path sits right at the default thread stack limit and tips over when a
//! slightly deeper call chain runs under load. libtest runs each test on a
//! spawned thread, which gets 2 MiB — not the main thread's 8 MiB.
//!
//! This narrows *which* operation is deep: our cascade, or a single store
//! query. Run each arm under a shrunken stack:
//!
//! ```sh
//! RUST_MIN_STACK=1048576 cargo nextest run -p uniko-store \
//!   --test stack_depth_repro --run-ignored all --no-capture
//! ```
//!
//! FINDING: none of these arms is the deep one — all three pass at 1 MiB,
//! where the full ingest-plus-recall pass aborts. So no single store query is
//! deep; the ~2 MiB is cumulative across a real cascade, and boxing uniko's
//! large ingest/recall futures moved the threshold not at all. Under a
//! 256 KiB stack the overflow lands on uni-db's own `uni-io` thread, which
//! places the depth in the store's query execution rather than in uniko's
//! frames.
//!
//! Kept as a negative result: it rules out the obvious single-query
//! explanations in seconds, so the next person down this path does not have
//! to re-derive them.

use uniko_store::config::UnikoConfig;
use uniko_store::storage::KnowledgeBase;

async fn seeded_kb() -> KnowledgeBase {
    let kb = KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory kb");
    for i in 0..5 {
        let mut props = std::collections::HashMap::new();
        props.insert(
            "message_id".to_string(),
            uniko_store::Value::String(format!("m-{i}")),
        );
        props.insert(
            "content".to_string(),
            uniko_store::Value::String(format!("the telescope readings for night {i}")),
        );
        props.insert(
            "content_type".to_string(),
            uniko_store::Value::String("text".into()),
        );
        props.insert(
            "timestamp".to_string(),
            uniko_store::types::datetime_value(chrono::Utc::now()),
        );
        let _ = kb.create_node("Message", &props).await;
    }
    kb
}

/// Arm A: a single BM25 fulltext query — the operation the bench log shows
/// immediately before the overflow.
#[ignore = "diagnostic: run explicitly with RUST_MIN_STACK set"]
#[tokio::test]
async fn fulltext_query_stack_depth() {
    let kb = seeded_kb().await;
    let hits = kb
        .recall_fulltext_search("Message", "content", "telescope readings", 10, None)
        .await
        .expect("fulltext search");
    eprintln!("fulltext arm OK — {} hits", hits.len());
    kb.shutdown().await.expect("shutdown");
}

/// Arm B: a vector query, for contrast. If this survives a stack that the
/// fulltext arm cannot, the depth is specific to the fulltext path rather
/// than to query execution generally.
#[ignore = "diagnostic: run explicitly with RUST_MIN_STACK set"]
#[tokio::test]
async fn vector_query_stack_depth() {
    let kb = seeded_kb().await;
    let qvec: Vec<f32> = (0..kb.config().embedding.dimensions)
        .map(|i| ((i % 17) as f32) * 0.01)
        .collect();
    let hits = kb
        .recall_vector_search("Message", "embedding", "content", &qvec, 10, None)
        .await
        .expect("vector search");
    eprintln!("vector arm OK — {} hits", hits.len());
    kb.shutdown().await.expect("shutdown");
}

/// Arm C: opening the store alone, as a control. If this overflows too, the
/// depth is in schema/index setup and not in querying at all.
#[ignore = "diagnostic: run explicitly with RUST_MIN_STACK set"]
#[tokio::test]
async fn open_only_stack_depth() {
    let kb = seeded_kb().await;
    eprintln!("open arm OK");
    kb.shutdown().await.expect("shutdown");
}
