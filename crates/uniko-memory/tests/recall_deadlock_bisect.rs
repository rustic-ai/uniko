//! Top-down bisect for the `similar_to` FTS deadlock on uni-db 4.1.0.
//!
//! The LoCoMo bench hangs a few questions in, with this stack:
//!
//! ```text
//! main thread    pthread_join  <- std::thread::scope <- SimilarToExecExpr::evaluate
//! scoped worker  Runtime::block_on(uni_query::…::fts_search_batch)  [never returns]
//! ~50 tokio workers  __psynch_cvwait
//! ```
//!
//! Reproduced 4/4 on 4.1.0 (two differently-built datasets); the same
//! conversation completes on 3.4.0. Bottom-up repros do NOT reproduce it —
//! a plain BM25 query, one against a materialized inverted index, concurrent
//! readers plus a flushing writer, concurrent queries across three labels,
//! more concurrent evaluations than worker threads, and `phase2_expand`'s
//! exact vector+fulltext source fan-out replayed against the failing KB all
//! complete normally.
//!
//! So this starts from the top — the real `recall()` — and removes phases
//! until the hang stops, rather than adding pieces until it starts.
//!
//! ```sh
//! UNIKO_DEADLOCK_KB=/abs/path/to/kb/conv-30 \
//!   cargo nextest run --release -p uniko-memory --test recall_deadlock_bisect \
//!   --run-ignored all --no-capture
//! ```
//!
//! `UNIKO_BISECT` selects the arm:
//!   `full`     — everything on (default): the bench's shipped recall config
//!   `no-graph` — phase2 graph activation off
//!   `no-temporal` — phase2 temporal off
//!   `no-rerank`  — reranker off
//!   `phase1-only` — phase1 strategy only, phases 2/3 suppressed by gating

use std::time::{Duration, Instant};

use uniko_memory::recall::{RecallConfig, recall};
use uniko_store::config::UnikoConfig;
use uniko_store::storage::KnowledgeBase;

const ROUNDS: usize = 25;
const TIMEOUT: Duration = Duration::from_secs(90);

/// Queries with enough lexical variety that the FTS channel is really used.
const QUERIES: [&str; 5] = [
    "what did they say about the dog in the park",
    "where did the trip to the beach happen",
    "who repaired the bicycle and when",
    "what was discussed about work schedules",
    "which photos were shared during the conversation",
];

#[ignore = "diagnostic: needs UNIKO_DEADLOCK_KB pointing at a persistent uniko KB"]
#[tokio::test(flavor = "multi_thread")]
async fn recall_does_not_deadlock() {
    let Ok(kb_path) = std::env::var("UNIKO_DEADLOCK_KB") else {
        eprintln!("SKIP: set UNIKO_DEADLOCK_KB to a persistent uniko KB directory");
        return;
    };
    // Tests run with CWD = the package root, so a relative path would create
    // an empty store and every round would falsely pass.
    let path = std::path::Path::new(&kb_path);
    assert!(path.is_absolute(), "UNIKO_DEADLOCK_KB must be absolute");
    assert!(
        path.join("storage").is_dir(),
        "no store at {kb_path} — refusing to create an empty one"
    );

    let arm = std::env::var("UNIKO_BISECT").unwrap_or_else(|_| "full".into());
    let uniko_config = UnikoConfig::default();
    let kb = KnowledgeBase::open(&kb_path, uniko_config.clone())
        .await
        .expect("open fixture KB");

    // Sanity: an empty fixture would pass vacuously.
    for label in ["Observation", "Message"] {
        let n = kb
            .query_cypher(
                &format!("MATCH (n:{label}) RETURN count(n) AS c"),
                &std::collections::HashMap::new(),
            )
            .await
            .ok()
            .and_then(|r| r.first().and_then(|m| m.get("c").cloned()));
        eprintln!("  fixture {label}: {n:?}");
    }

    let mut config = RecallConfig::from_uniko_config(&uniko_config);
    match arm.as_str() {
        "no-graph" => config.phase2_graph_enabled = false,
        "no-temporal" => config.phase2_temporal_enabled = false,
        "no-rerank" => config.reranker_enabled = false,
        "phase1-only" => {
            config.phase2_graph_enabled = false;
            config.phase2_temporal_enabled = false;
            config.reranker_enabled = false;
        }
        _ => {}
    }
    eprintln!(
        "arm={arm}  graph={} temporal={} rerank={}",
        config.phase2_graph_enabled, config.phase2_temporal_enabled, config.reranker_enabled
    );

    // UNIKO_BISECT_QUERY pins every round to one query, separating
    // "state carried between calls" from "this particular query text".
    let pinned: Option<usize> = std::env::var("UNIKO_BISECT_QUERY")
        .ok()
        .and_then(|v| v.parse().ok());
    // UNIKO_BISECT_ALTERNATE=2 alternates between exactly two query texts,
    // the minimum needed to test whether a CHANGE of query is the trigger.
    let alternate: Option<usize> = std::env::var("UNIKO_BISECT_ALTERNATE")
        .ok()
        .and_then(|v| v.parse().ok());
    for round in 1..=ROUNDS {
        let query = match (pinned, alternate) {
            (_, Some(n)) => QUERIES[(round - 1) % n.max(1)],
            (Some(i), _) => QUERIES[i % QUERIES.len()],
            _ => QUERIES[(round - 1) % QUERIES.len()],
        };
        let start = Instant::now();
        match tokio::time::timeout(TIMEOUT, recall(&kb, query, &config)).await {
            Ok(Ok(bundle)) => eprintln!(
                "  round {round}: OK — {} items in {} ms",
                bundle.items.len(),
                start.elapsed().as_millis()
            ),
            Ok(Err(e)) => panic!("round {round} errored: {e}"),
            Err(_) => panic!(
                "DEADLOCK — round {round} (query {query:?}) exceeded {}s on arm '{arm}'",
                TIMEOUT.as_secs()
            ),
        }
    }
    eprintln!("no deadlock across {ROUNDS} rounds on arm '{arm}'");
}
