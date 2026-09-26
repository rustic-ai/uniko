//! Isolated repro: a `MATCH` issued **inside** an open transaction appears
//! to be pathologically slower than the identical `MATCH` on a fresh
//! session.
//!
//! Found while folding uniko's artifact ingest into a single transaction
//! (rustic-ai/uniko#40). The committing `merge_artifact_content` runs its
//! existence read on `db.session()` and only the write under a
//! transaction; moving that read onto the caller's `tx` — so it can see the
//! transaction's own uncommitted writes — made
//! `ingest_artifact_creates_node_and_chunks` go from **9.4 s to >360 s**
//! (single-threaded, no concurrency, no contention). Bisected against the
//! same source with the change stashed: 9.4 s pass / >360 s hang, 1/1.
//!
//! A `sample` of the hung process shows the main thread inside
//! `uni_db::api::UniBuilder::build` with leaf frames in `write`/`close` —
//! doing I/O, not blocked on a mutex — so this looks like work amplification
//! rather than a lock cycle.
//!
//! Why it matters: an embedder that wants a check-then-create to be atomic
//! has no alternative. The existence read MUST be on the caller's
//! transaction, because a fresh session cannot observe rows that
//! transaction has written but not yet committed — so the "read on a
//! session, write in a tx" shape silently creates duplicates.
//!
//! Each arm is timed and bounded by `BUDGET`, so a hang fails the test
//! instead of running forever. `flavor = "multi_thread"` is deliberate: a
//! `tokio::time::timeout` cannot fire if the only worker is blocked in a
//! synchronous call.
//!
//! Depends on `uni_db` alone — no uniko types — so it lifts into the
//! upstream repo unchanged.

use std::time::{Duration, Instant};

use uni_db::{DataType, Uni};

/// Generous per-arm bound. The equivalent session read is milliseconds; a
/// arm that needs more than this is the defect, not slow hardware.
const BUDGET: Duration = Duration::from_secs(60);

/// Rows seeded before the read arms, so a "no rows" fast path cannot
/// explain a difference between them.
const SEED_ROWS: usize = 50;

async fn open_with_schema(path: &str) -> Uni {
    let db = Uni::open(path).build().await.expect("open");
    db.schema()
        .label("Doc")
        .property("doc_id", DataType::String)
        .property("body", DataType::String)
        .done()
        .apply()
        .await
        .expect("apply schema");
    db
}

async fn seed(db: &Uni) {
    let tx = db.session().tx().await.expect("begin seed tx");
    for i in 0..SEED_ROWS {
        tx.query_with("CREATE (d:Doc {doc_id: $id, body: $b})")
            .param("id", format!("doc-{i}"))
            .param("b", format!("body of document {i}"))
            .fetch_all()
            .await
            .expect("seed create");
    }
    tx.commit().await.expect("seed commit");
}

/// Run `f`, returning how long it took, or panicking with `label` if it
/// blows the budget.
async fn timed<F, Fut>(label: &str, f: F) -> Duration
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    let start = Instant::now();
    match tokio::time::timeout(BUDGET, f()).await {
        Ok(()) => {
            let e = start.elapsed();
            eprintln!("  {label}: {} ms", e.as_millis());
            e
        }
        Err(_) => panic!("{label} exceeded {}s — this is the bug", BUDGET.as_secs()),
    }
}

/// THE COMPARISON. The same `MATCH`, by fresh session vs inside a tx.
///
/// EXPECTED: comparable. An indexed point lookup should not care which
/// handle issues it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn in_tx_match_is_not_pathologically_slower_than_session_match() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = open_with_schema(&dir.path().join("store").to_string_lossy()).await;
    seed(&db).await;

    // Control: read on a fresh session, exactly as the committing
    // `merge_artifact_content` does today.
    let session_ms = timed("session MATCH x20", || async {
        for i in 0..20 {
            let rows = db
                .session()
                .query_with("MATCH (d:Doc {doc_id: $id}) RETURN id(d) AS vid LIMIT 1")
                .param("id", format!("doc-{i}"))
                .fetch_all()
                .await
                .expect("session match");
            assert_eq!(rows.rows().len(), 1, "seeded row must be found");
        }
    })
    .await;

    // The change under test: the identical read on an open transaction.
    let tx_ms = timed("in-tx MATCH x20", || async {
        let tx = db.session().tx().await.expect("begin tx");
        for i in 0..20 {
            let rows = tx
                .query_with("MATCH (d:Doc {doc_id: $id}) RETURN id(d) AS vid LIMIT 1")
                .param("id", format!("doc-{i}"))
                .fetch_all()
                .await
                .expect("in-tx match");
            assert_eq!(rows.rows().len(), 1, "seeded row must be found");
        }
        tx.commit().await.expect("commit");
    })
    .await;

    db.shutdown().await.expect("shutdown");

    // Deliberately loose: this is about orders of magnitude, not jitter.
    assert!(
        tx_ms < session_ms * 20 + Duration::from_secs(5),
        "in-tx MATCH is {}x the session MATCH ({} ms vs {} ms)",
        tx_ms.as_millis() / session_ms.as_millis().max(1),
        tx_ms.as_millis(),
        session_ms.as_millis(),
    );
}

/// The exact shape uniko needs: interleave reads and writes on ONE
/// transaction — the check-then-create that must be atomic.
///
/// EXPECTED: completes promptly. This is the only correct way to write a
/// check-then-create, since a session read cannot see the tx's own
/// uncommitted rows.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn in_tx_read_then_write_interleaved_completes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = open_with_schema(&dir.path().join("store").to_string_lossy()).await;
    seed(&db).await;

    timed("interleaved read/write x20", || async {
        let tx = db.session().tx().await.expect("begin tx");
        for i in 0..20 {
            let id = format!("new-{i}");
            let rows = tx
                .query_with("MATCH (d:Doc {doc_id: $id}) RETURN id(d) AS vid LIMIT 1")
                .param("id", id.clone())
                .fetch_all()
                .await
                .expect("in-tx probe");
            if rows.rows().is_empty() {
                tx.query_with("CREATE (d:Doc {doc_id: $id, body: $b})")
                    .param("id", id)
                    .param("b", "created inside the tx")
                    .fetch_all()
                    .await
                    .expect("in-tx create");
            }
        }
        tx.commit().await.expect("commit");
    })
    .await;

    db.shutdown().await.expect("shutdown");
}

/// Read-your-own-writes: the property that makes the in-tx read necessary
/// in the first place. If this fails, a check-then-create inside one
/// transaction duplicates rows no later dedup can merge.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn in_tx_read_sees_the_transactions_own_uncommitted_write() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = open_with_schema(&dir.path().join("store").to_string_lossy()).await;

    timed("read-your-own-writes", || async {
        let tx = db.session().tx().await.expect("begin tx");
        tx.query_with("CREATE (d:Doc {doc_id: 'only', body: 'x'})")
            .fetch_all()
            .await
            .expect("create");

        let rows = tx
            .query_with("MATCH (d:Doc {doc_id: 'only'}) RETURN id(d) AS vid LIMIT 1")
            .fetch_all()
            .await
            .expect("in-tx read back");
        assert_eq!(
            rows.rows().len(),
            1,
            "an in-tx read must observe the same tx's uncommitted CREATE — \
             otherwise an atomic check-then-create is impossible"
        );
        tx.commit().await.expect("commit");
    })
    .await;

    db.shutdown().await.expect("shutdown");
}

/// Control: a fresh session must NOT see an uncommitted write. Together
/// with the test above this pins exactly why the read has to move onto the
/// transaction.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn session_read_does_not_see_an_uncommitted_write() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = open_with_schema(&dir.path().join("store").to_string_lossy()).await;

    let tx = db.session().tx().await.expect("begin tx");
    tx.query_with("CREATE (d:Doc {doc_id: 'pending', body: 'x'})")
        .fetch_all()
        .await
        .expect("create");

    let rows = db
        .session()
        .query_with("MATCH (d:Doc {doc_id: 'pending'}) RETURN id(d) AS vid LIMIT 1")
        .fetch_all()
        .await
        .expect("session read");
    assert!(
        rows.rows().is_empty(),
        "a fresh session must not observe another transaction's uncommitted write"
    );

    tx.commit().await.expect("commit");
    db.shutdown().await.expect("shutdown");
}
