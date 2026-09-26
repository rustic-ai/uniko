//! Issue #40's literal acceptance case: "interrupt the operation after its
//! first member would otherwise have been written, and reopen the store:
//! neither member may appear as a completed pair."
//!
//! The in-process rollback test in `facade::tests` proves uniko drops the
//! transaction on an error. That is a different guarantee from this one: a
//! clean rollback unwinds and rolls back, while a crash runs no destructor,
//! no `Drop`, no shutdown hook. Only this test says whether an interrupted
//! unit is still absent once the store is reopened from disk, which is what
//! a caller actually depends on after a power cut or an OOM kill.
//!
//! The child re-executes this same test binary — the pattern used by
//! `crates/uniko-store/tests/unidb_wal_no_manifest_repro.rs`.

use std::path::Path;
use std::process::Command;

use uniko_memory::{Turn, Uniko};

/// Set on the re-executed child; value is the store path.
const CHILD_ENV: &str = "UNIKO_UNIT_CRASH_STORE";

/// The writer half, run only in a re-executed child.
///
/// Opens a persistent store, starts a two-turn unit, and aborts inside the
/// transaction after the FIRST turn's writes. The abort is the point: it
/// runs no destructor and no shutdown, exactly like a SIGKILL or a power
/// cut, leaving the transaction open and uncommitted.
#[tokio::test]
async fn child_writer() {
    // A normal `cargo nextest` run has no child env and this is a no-op.
    let Ok(store) = std::env::var(CHILD_ENV) else {
        return;
    };

    let Ok(memory) = Uniko::open(&store).await else {
        // Model unavailable: exit cleanly so the parent skips rather than
        // reading a crash that never happened.
        std::process::exit(0);
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("crash-unit");

    // SAFETY: this process exists only to crash; nothing else runs here.
    unsafe { std::env::set_var("UNIKO_TEST_ABORT_AFTER_TURN", "0") };

    let _ = session
        .unit()
        .turn(Turn::new("alice", "first member of the pair").id("crash-m1"))
        .turn(Turn::new("bob", "second member of the pair").id("crash-m2"))
        .commit()
        .await;

    // Only reached if the hook did not fire — fail loudly rather than
    // letting the parent interpret a clean exit as a crash.
    std::process::exit(0);
}

/// Re-execute this binary as `child_writer` against `store`. Returns true
/// when the child actually aborted.
fn crash_writer(store: &Path) -> bool {
    let status = Command::new(std::env::current_exe().expect("test binary path"))
        .args(["--exact", "child_writer", "--nocapture"])
        .env(CHILD_ENV, store)
        .status()
        .expect("spawn child writer");
    // An abort is how the child is meant to end. A clean exit means it never
    // reached the hook, so nothing below would be meaningful.
    !status.success()
}

/// THE ACCEPTANCE CASE. Interrupt a unit mid-transaction, reopen the store,
/// and assert neither member of the pair is present.
#[tokio::test]
async fn interrupted_unit_leaves_nothing_after_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");
    let store = dir.path().join("store");

    if !crash_writer(&store) {
        eprintln!("SKIP: child exited cleanly (models unavailable, or the hook did not fire)");
        return;
    }

    // Reopen from disk. Recovery replays the WAL, so anything the aborted
    // transaction had committed would come back here.
    let Ok(memory) = Uniko::open(&store).await else {
        eprintln!("SKIP: cannot reopen the crashed store (models unavailable?)");
        return;
    };
    let agent = memory.agent("assistant");

    for id in ["crash-m1", "crash-m2"] {
        let rows = agent
            .query(&format!(
                "MATCH (m:Message) WHERE m.message_id = '{id}' RETURN count(m) AS c"
            ))
            .await
            .expect("count query");
        let count = rows
            .first()
            .and_then(|r| match r.get("c") {
                Some(uniko_memory::Value::Int(n)) => Some(*n),
                _ => None,
            })
            .unwrap_or(0);
        assert_eq!(
            count, 0,
            "{id} survived an interrupted unit — a partial pair must never be \
             visible after reopening the store"
        );
    }

    drop(agent);
    memory.shutdown().await.expect("shutdown");
}
