//! Plain uni-db, no uniko code: a second writer on an open store is
//! refused.
//!
//! uni-db once took no lock on a store directory, so a second `Uni::open`
//! of a directory another handle had open succeeded, and the two handles
//! were two writers that didn't see each other's commits
//! (rustic-ai/uni-db#348). uni-db now holds a local store by an OS lock on
//! `writer.lock`. uniko relies on that across processes; within one it
//! refuses the second open itself (`OpenPath` in `storage/mod.rs`).

use std::collections::HashMap;

use uni_db::{DataType, Uni, UniConfig, UniError, Value};

async fn count(db: &Uni) -> i64 {
    db.session()
        .query("MATCH (d:Doc) RETURN count(d) AS n")
        .await
        .unwrap()
        .rows()[0]
        .get::<i64>("n")
        .unwrap()
}

async fn insert(db: &Uni, n: i64) {
    let tx = db.session().tx().await.unwrap();
    tx.bulk_insert_vertices(
        "Doc",
        vec![HashMap::from([("n".to_string(), Value::Int(n))])],
    )
    .await
    .unwrap();
    tx.commit().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_second_handle_on_an_open_store_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store");
    let a = Uni::open(path.to_string_lossy()).build().await.unwrap();
    a.schema()
        .label("Doc")
        .property("n", DataType::Int64)
        .done()
        .apply()
        .await
        .unwrap();
    insert(&a, 1).await;
    a.flush().await.unwrap();

    let mut quick = UniConfig::default();
    quick.writer_lease.acquire_grace = std::time::Duration::from_millis(100);
    let b = Uni::open(path.to_string_lossy())
        .config(quick)
        .build()
        .await;
    assert!(
        matches!(b, Err(UniError::StoreLocked { .. })),
        "a second writer is refused: {:?}",
        b.err()
    );
    insert(&a, 2).await;
    a.shutdown().await.unwrap();

    let reopened = Uni::open(path.to_string_lossy()).build().await.unwrap();
    assert_eq!(
        count(&reopened).await,
        2,
        "the one writer's commits are all there"
    );
    reopened.shutdown().await.unwrap();
}
