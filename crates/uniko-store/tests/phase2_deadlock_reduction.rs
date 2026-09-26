//! Reduction harness for the `similar_to` FTS deadlock seen on uni-db 4.1.0.
//!
//! The LoCoMo bench hangs a few questions in, with this stack:
//!
//! ```text
//! main thread    pthread_join  <- std::thread::scope
//!                              <- SimilarToExecExpr::evaluate
//! scoped worker  Runtime::block_on(uni_query::…::fts_search_batch)  [never returns]
//! ~50 tokio workers  __psynch_cvwait
//! ```
//!
//! `similar_to`'s FTS branch spawns a scoped thread, builds a nested
//! current_thread runtime and blocks the caller in `join()`
//! (`uni-query-4.1.0/src/query/df_graph/similar_to_expr.rs:408-460`).
//!
//! Standalone repros that did NOT reproduce it, each ruling one thing out:
//! a plain BM25 query; one against a materialized inverted index; concurrent
//! readers with a writer flushing throughout; concurrent queries across three
//! labels; and more concurrent evaluations than worker threads. `AutoEmbed`
//! is not involved either — uniko passes a precomputed `$qvec`, which takes
//! `ScoringMode::Vector` and skips the nested runtime.
//!
//! What none of those had is the MIX `phase2_expand` issues concurrently:
//! three vector sources and two fulltext sources against a real uniko KB.
//! This harness replays exactly that against a fixture store, so the trigger
//! can be bisected by dropping sources from the set.
//!
//! Point `UNIKO_DEADLOCK_KB` at a persistent uniko KB (e.g. one the bench
//! left behind) and run:
//!
//! ```sh
//! UNIKO_DEADLOCK_KB=data/kb_sym/conv-30 \
//!   cargo nextest run -p uniko-store --test phase2_deadlock_reduction \
//!   --run-ignored all --no-capture
//! ```
//!
//! `UNIKO_DEADLOCK_SOURCES` selects which sources to fan out:
//! `all` (default), `fts` (fulltext only), `vec` (vector only).

use std::time::{Duration, Instant};

use uniko_store::config::UnikoConfig;
use uniko_store::storage::KnowledgeBase;

/// One phase-2 source: (label, mode, content field, top-k).
const SOURCES: &[(&str, &str, &str, i64)] = &[
    ("Episode", "vector", "action_type", 20),
    ("Observation", "vector", "content", 20),
    ("Message", "vector", "content", 10),
    ("Observation", "fulltext", "content", 20),
    ("Message", "fulltext", "content", 10),
];

const ROUNDS: usize = 20;
const TIMEOUT: Duration = Duration::from_secs(60);

#[ignore = "diagnostic: needs UNIKO_DEADLOCK_KB pointing at a persistent uniko KB"]
#[tokio::test(flavor = "multi_thread")]
async fn phase2_fanout_does_not_deadlock() {
    let Ok(kb_path) = std::env::var("UNIKO_DEADLOCK_KB") else {
        eprintln!("SKIP: set UNIKO_DEADLOCK_KB to a persistent uniko KB directory");
        return;
    };
    let filter = std::env::var("UNIKO_DEADLOCK_SOURCES").unwrap_or_else(|_| "all".into());

    // nextest runs tests with CWD = the package root, not the workspace root,
    // so a relative path here silently CREATES an empty store and every query
    // returns zero rows — a false pass. Require an absolute, existing path.
    let path = std::path::Path::new(&kb_path);
    assert!(
        path.is_absolute(),
        "UNIKO_DEADLOCK_KB must be absolute (tests run with CWD = package root); got {kb_path}"
    );
    assert!(
        path.join("storage").is_dir(),
        "no store at {kb_path} — refusing to create an empty one"
    );

    let kb = match KnowledgeBase::open(&kb_path, UnikoConfig::default()).await {
        Ok(kb) => kb,
        Err(e) => {
            eprintln!("SKIP: cannot open {kb_path}: {e}");
            return;
        }
    };

    let selected: Vec<_> = SOURCES
        .iter()
        .filter(|(_, mode, _, _)| match filter.as_str() {
            "fts" => *mode == "fulltext",
            "vec" => *mode == "vector",
            _ => true,
        })
        .collect();
    eprintln!(
        "fanning out {} sources per round ({}): {:?}",
        selected.len(),
        filter,
        selected
            .iter()
            .map(|(l, m, _, _)| format!("{l}/{m}"))
            .collect::<Vec<_>>()
    );

    // A fixed query vector — uniko precomputes one and passes it as `$qvec`,
    // so the vector sources take `ScoringMode::Vector`, not `AutoEmbed`.
    let qvec: Vec<f32> = (0..384).map(|i| ((i % 17) as f32) * 0.01).collect();

    // Confirm the fixture actually holds rows — 0 hits from an empty store
    // would look like success.
    for label in ["Episode", "Observation", "Message"] {
        let n = kb
            .query_cypher(
                &format!("MATCH (n:{label}) RETURN count(n) AS c"),
                &std::collections::HashMap::new(),
            )
            .await
            .map(|r| format!("{:?}", r.first().and_then(|m| m.get("c"))))
            .unwrap_or_else(|e| format!("ERR {e}"));
        eprintln!("  fixture {label}: {n}");
    }

    for round in 1..=ROUNDS {
        let start = Instant::now();
        let futs: Vec<_> = selected
            .iter()
            .map(|(label, mode, content_field, k)| {
                let kb = kb.clone();
                let qvec = qvec.clone();
                let (label, mode, content_field, k) = (*label, *mode, *content_field, *k);
                tokio::spawn(async move {
                    match mode {
                        "vector" => kb
                            .recall_vector_search(label, "embedding", content_field, &qvec, k, None)
                            .await
                            .map(|r| r.len()),
                        _ => kb
                            .recall_fulltext_search(
                                label,
                                content_field,
                                "lattice tower paris",
                                k,
                                None,
                            )
                            .await
                            .map(|r| r.len()),
                    }
                })
            })
            .collect();

        let joined = async {
            let mut hits = 0usize;
            for (idx, f) in futs.into_iter().enumerate() {
                match f.await {
                    Ok(Ok(n)) => hits += n,
                    // Never swallow: a source that errors looks identical to
                    // one that simply found nothing, and that hides whether
                    // the harness is exercising the path at all.
                    Ok(Err(e)) => panic!("source {idx} errored: {e}"),
                    Err(e) => panic!("source {idx} join failed: {e}"),
                }
            }
            hits
        };

        match tokio::time::timeout(TIMEOUT, joined).await {
            Ok(hits) => {
                eprintln!(
                    "  round {round}: OK — {hits} hits in {} ms",
                    start.elapsed().as_millis()
                );
            }
            Err(_) => {
                panic!(
                    "DEADLOCK — round {round} did not complete within {}s with sources: {filter}",
                    TIMEOUT.as_secs()
                );
            }
        }
    }
    eprintln!("no deadlock across {ROUNDS} rounds with sources: {filter}");
}
