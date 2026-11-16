//! Regression repro for a uni-db bug: on Windows, opening a store and writing
//! to it fails with `Access is denied. (os error 5)`.
//!
//! Symptom (uniko's Python wheel on Windows, 0.4.0 and 0.5.0 alike):
//!
//! ```text
//! storage error: Internal error: Access is denied. (os error 5)
//! ```
//!
//! Root cause: `uni-store/src/runtime/wal.rs` `sync_file_and_parent` (uni-db
//! 5.1 and 5.2) makes a freshly written WAL segment, snapshot manifest or
//! catalog file durable with
//!
//! ```rust,ignore
//! std::fs::File::open(path)?.sync_all()?;
//! ```
//!
//! `File::open` is read-only. POSIX lets you `fsync` a read-only descriptor, but
//! on Windows `sync_all` is `FlushFileBuffers`, which needs `GENERIC_WRITE` on
//! the handle and otherwise fails with `ERROR_ACCESS_DENIED`. The access check
//! rejects the call before it reaches the file system, so Process Monitor shows
//! every file operation succeeding: the segment is written, reopened with
//! `Generic Read`, and the store is torn down. Opening the file with
//! `OpenOptions::new().write(true)` fixes it. The directory fsync below it is
//! already `#[cfg(unix)]`.
//!
//! Reported upstream as rustic-ai/uni-db#338, with a standalone repro run on
//! Windows Server 2022: stock 5.2.0 panics at the first `commit`, and the
//! one-line `OpenOptions` change makes the same program pass. Fixed in uni-db
//! 5.2.1, which uniko now requires.
//!
//! This test guards the write path on every platform; on Windows it is the
//! regression check for that bug.

use uni_db::{DataType, Uni};

#[tokio::test]
async fn store_opens_and_commits_a_write() {
    let db = Uni::in_memory().build().await.expect("open in-memory db");

    db.schema()
        .label("Note")
        .property("id", DataType::String)
        .done()
        .apply()
        .await
        .expect("register schema");

    let tx = db.session().tx().await.expect("begin tx");
    tx.query("CREATE (:Note {id: 'n1'})")
        .await
        .expect("create node");
    tx.commit().await.expect("commit");

    let rows = db
        .session()
        .query("MATCH (n:Note) RETURN n.id AS id")
        .await
        .expect("read back");
    assert_eq!(rows.len(), 1);
}
