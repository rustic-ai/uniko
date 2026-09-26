//! Isolated repro: a transaction begun AFTER a commit does not always see
//! that commit, while a non-transactional read of the same data does.
//!
//! Found chasing a spurious session-chunk rebuild in uniko. `finalize()` on
//! an unchanged session must report `rebuilt == false`; under load it
//! reported `true` in roughly a third of full-suite runs. The failure was
//! captured with the state printed:
//!
//! ```text
//! first.transcript_chunks  = [130]
//! second.transcript_chunks = [130]
//! read BETWEEN finalizes   = [SessionChunkRow { node_id: 130, text: "alice: ..." }]
//! ```
//!
//! The chunk was committed, and a fresh-session read between the two passes
//! returned it correctly. The second pass's existence check — which reads
//! inside the transaction that would rewrite the chunks — behaved as though
//! the row did not exist. Decode failures were ruled out first: those reads
//! propagate errors now, and none were raised.
//!
//! So the question this isolates is narrow: after `tx.commit()` returns, is
//! the write visible to a transaction begun afterwards?
//!
//! Run:
//! ```sh
//! cargo nextest run -p uniko-store --test unidb_tx_snapshot_lag_repro \
//!   --run-ignored all --no-capture
//! ```
//!
//! `LAG_ROUNDS` sets iterations per task (default 40), `LAG_TASKS` the
//! concurrent tasks (default 8). Concurrency is required — a quiet
//! single-threaded run does not trip it.
//!
//! Depends on `uni_db` alone, so it lifts upstream unchanged.

use std::sync::Arc;

use uni_db::{DataType, Uni};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Count rows matching `tag`, via a fresh session.
async fn count_via_session(db: &Uni, tag: &str) -> usize {
    db.session()
        .query_with("MATCH (d:Doc {tag: $t}) RETURN id(d) AS vid")
        .param("t", tag)
        .fetch_all()
        .await
        .expect("session read")
        .rows()
        .len()
}

/// Count rows matching `tag`, inside a freshly begun transaction.
async fn count_via_tx(db: &Uni, tag: &str) -> usize {
    let tx = db.session().tx().await.expect("begin tx");
    let n = tx
        .query_with("MATCH (d:Doc {tag: $t}) RETURN id(d) AS vid")
        .param("t", tag)
        .fetch_all()
        .await
        .expect("in-tx read")
        .rows()
        .len();
    tx.commit().await.expect("commit read tx");
    n
}

/// A committed write must be visible to BOTH a later session read and a
/// later transaction. The transaction is the one under suspicion.
#[ignore = "repro: needs concurrency; run explicitly"]
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn commit_is_visible_to_a_later_transaction() {
    let rounds = env_usize("LAG_ROUNDS", 40);
    let tasks = env_usize("LAG_TASKS", 8);

    let dir = tempfile::tempdir().expect("tempdir");
    let db = Uni::open(dir.path().join("store").to_string_lossy())
        .build()
        .await
        .expect("open");
    db.schema()
        .label("Doc")
        .property("tag", DataType::String)
        .property("body", DataType::String)
        .done()
        .apply()
        .await
        .expect("schema");
    let db = Arc::new(db);

    let mut handles = Vec::with_capacity(tasks);
    for t in 0..tasks {
        let db = Arc::clone(&db);
        handles.push(tokio::spawn(async move {
            let mut misses = Vec::new();
            for r in 0..rounds {
                let tag = format!("t{t}-r{r}");

                // Write and commit.
                let tx = db.session().tx().await.expect("begin write tx");
                tx.query_with("CREATE (d:Doc {tag: $t, body: 'x'})")
                    .param("t", tag.clone())
                    .fetch_all()
                    .await
                    .expect("create");
                tx.commit().await.expect("commit write");

                // Both reads happen after the commit acknowledged.
                let via_session = count_via_session(&db, &tag).await;
                let via_tx = count_via_tx(&db, &tag).await;

                if via_session != 1 || via_tx != 1 {
                    misses.push(format!(
                        "{tag}: session saw {via_session}, tx saw {via_tx} (expected 1 and 1)"
                    ));
                }
            }
            misses
        }));
    }

    let mut all = Vec::new();
    for h in handles {
        all.extend(h.await.expect("task panicked"));
    }

    let total = rounds * tasks;
    eprintln!(
        "tx snapshot lag: {} misses out of {total} writes",
        all.len()
    );
    for m in all.iter().take(10) {
        eprintln!("  {m}");
    }
    if let Ok(db) = Arc::try_unwrap(db) {
        db.shutdown().await.expect("shutdown");
    }

    assert!(
        all.is_empty(),
        "{} of {total} committed writes were invisible to a later read",
        all.len()
    );
}
