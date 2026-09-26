//! Regression guard for a uni-db bug, **fixed upstream in 4.0.0**: a process
//! crash before a brand-new store's first L0→L1 flush left it permanently
//! unopenable, even though the WAL was complete and fsynced.
//!
//! Symptom, on the next `Uni::open`:
//!
//! ```text
//! Internal error: Database has WAL segments but no snapshot manifest.
//! Cannot safely determine version counter -- starting at 0 would cause
//! version conflicts and data corruption. Restore the snapshot manifest or
//! delete WAL to start fresh.
//! ```
//!
//! Root cause: `UniBuilder::build` (`uni-db-3.4.0/src/api/mod.rs:1786-1793`)
//! refuses to open when WAL segments exist and *no* snapshot manifest does.
//! The refusal is only about deriving the version counter — the replay
//! machinery immediately below it
//! (`uni-db-3.4.0/src/api/mod.rs:1925-1932`) is correct and would replay
//! from `wal_high_water_mark = 0`. The WAL segments themselves carry the
//! versions the guard says it cannot determine.
//!
//! The snapshot manifest is written only by a flush. On a fresh store the
//! first flush is whichever comes first: `auto_flush_interval` (default 5s),
//! `auto_flush_threshold` (default 10_000 mutations), or an explicit
//! `flush()` / `shutdown()`. So *every* new store has a multi-second window
//! in which a crash — SIGKILL, power loss, `panic = "abort"`, a host OOM —
//! destroys every committed write in it. No amount of clean-shutdown
//! discipline in the embedder closes that window.
//!
//! This is NOT `rustic-ai/uni-db#251`. That report was `uni-cli` never
//! calling `shutdown`, and commit `36a6053` fixed the CLI; the open guard
//! was deliberately left as it was. A crash cannot be fixed by calling
//! shutdown.
//!
//! **Fixed in uni-db 4.0.0** (`rustic-ai/uni-db#275`). The open path now
//! separates the two situations that shared this shape, using
//! `table_names()` as the discriminator the old guard lacked: a store that
//! never flushed has no L1 data, so nothing can collide with a counter
//! starting at 0 and the WAL is replayed; a store with *lost* manifests
//! over existing L1 data still refuses, because replay assigns fresh
//! versions that would collide silently (`uni-db-4.0.0/src/api/mod.rs:1808`).
//!
//! These tests ran as `#[ignore]`d expected-failures until 2026-09-19, when
//! the 4.0.0 upgrade showed them green. The attribute is gone so they now
//! guard against regression on every run.
//!
//! The two tests isolate the manifest as the trigger: identical crash,
//! identical writes, and the only difference is whether one flush landed
//! first. Reproduced 5/5 against uni-db 3.4.0, with the control green 5/5
//! in the same runs.
//!
//! Depends on `uni_db` alone — no uniko types — so it lifts into the
//! upstream repo unchanged. Reported downstream as `rustic-ai/uniko#38`.

use std::path::Path;
use std::process::Command;

use uni_db::{DataType, Uni};

/// Set on the re-executed child to make it the writer. Value is the store
/// path; `UNIDB_WAL_REPRO_FLUSH=1` additionally asks it to flush first.
const CHILD_ENV: &str = "UNIDB_WAL_REPRO_STORE";
const FLUSH_ENV: &str = "UNIDB_WAL_REPRO_FLUSH";

/// The writer half, run only in a re-executed child (see [`crash_writer`]).
///
/// Creates the store, commits one row, and — with the commit acknowledged —
/// aborts the process. `abort` is the point: it runs no destructor, no
/// `Drop for Uni`, no shutdown hook, exactly like a SIGKILL or a power cut.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn child_writer() {
    // A normal `cargo nextest` run has no child env and this is a no-op.
    let Ok(store) = std::env::var(CHILD_ENV) else {
        return;
    };

    let db = Uni::open(&store).build().await.expect("open store");
    db.schema()
        .label("Person")
        .property("name", DataType::String)
        .done()
        .apply()
        .await
        .expect("register schema");

    let tx = db.session().tx().await.expect("begin tx");
    tx.query_with("CREATE (p:Person {name: 'ada'})")
        .fetch_all()
        .await
        .expect("create");
    tx.commit().await.expect("commit");

    // The control: one flush publishes the snapshot manifest the guard
    // wants. Everything else about the two runs is identical.
    if std::env::var(FLUSH_ENV).is_ok() {
        db.flush().await.expect("flush");
    }

    // Crash with the write committed and the WAL fsynced.
    std::process::abort();
}

/// Re-execute this test binary as `child_writer` against `store` and wait
/// for it to abort. Returns once the child is gone.
fn crash_writer(store: &Path, flush_first: bool) {
    let mut cmd = Command::new(std::env::current_exe().expect("test binary path"));
    cmd.args(["--exact", "child_writer", "--nocapture"])
        .env(CHILD_ENV, store);
    if flush_first {
        cmd.env(FLUSH_ENV, "1");
    }
    let status = cmd.status().expect("spawn child writer");

    // Aborting is how the child is supposed to end; a clean exit means it
    // never reached `abort()` and the repro proved nothing.
    assert!(
        !status.success(),
        "child writer exited cleanly — it never reached abort(), \
         so the store was not crashed and neither assertion below is meaningful"
    );
}

/// THE BUG (fixed in 4.0.0). A crash before the first flush must not make a
/// new store's committed writes unreachable.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn crash_before_first_flush_leaves_store_openable() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = dir.path().join("store");

    crash_writer(&store, false);

    // The WAL is on disk and complete; nothing else is. There is no
    // `storage/` directory at all, because no flush ever ran.
    let wal = store.join("wal");
    assert!(
        wal.is_dir() && wal.read_dir().expect("read wal dir").next().is_some(),
        "no WAL segments were written — the child did not commit"
    );

    // EXPECTED: the open replays the WAL and returns the committed row.
    // ACTUAL (bug): `Internal error: Database has WAL segments but no
    // snapshot manifest.` The row is in the WAL, fsynced, and unreachable —
    // uni-db offers no documented path to it, and the error's own advice
    // ("delete WAL to start fresh") discards it.
    let db = Uni::open(store.to_string_lossy())
        .build()
        .await
        .expect("reopen a store whose only defect is that it never flushed");

    let rows = db
        .session()
        .query_with("MATCH (p:Person) RETURN p.name AS name")
        .fetch_all()
        .await
        .expect("query");
    let name: String = rows
        .rows()
        .first()
        .expect("one row")
        .get("name")
        .expect("name");
    assert_eq!(name, "ada", "committed row did not survive the crash");
}

/// THE CONTROL. Same crash, same writes — but one flush ran first, so a
/// manifest exists and WAL recovery works. This is what isolates the missing
/// manifest, rather than the crash or the WAL, as the trigger.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn crash_after_a_flush_recovers() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = dir.path().join("store");

    crash_writer(&store, true);

    let db = Uni::open(store.to_string_lossy())
        .build()
        .await
        .expect("reopen after a flush");

    let rows = db
        .session()
        .query_with("MATCH (p:Person) RETURN p.name AS name")
        .fetch_all()
        .await
        .expect("query");
    let name: String = rows
        .rows()
        .first()
        .expect("one row")
        .get("name")
        .expect("name");
    assert_eq!(name, "ada");
}
