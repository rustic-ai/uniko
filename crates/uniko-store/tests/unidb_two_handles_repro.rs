//! Isolated uni-db repro: two `Uni` handles on one store directory.
//!
//! Plain uni-db, no uniko code. uni-db takes no lock on a store directory,
//! so a second `Uni::open` of a directory another handle has open succeeds
//! (once the store has a snapshot manifest), and the two handles are two
//! independent writers on the same files: neither sees the other's commits.
//!
//! The test asserts what a locked directory would give, a refused second
//! open, so it is ignored until uni-db locks it. Run it to see today's
//! behavior:
//!
//! ```sh
//! cargo nextest run -p uniko-store --test integration unidb_two_handles_repro:: \
//!   --run-ignored all --no-capture
//! ```
//!
//! uniko guards against this itself (`OpenPath` in `storage/mod.rs`: one
//! claim per path in the process, and an OS lock on `.uniko.lock`).

use std::collections::HashMap;

use uni_db::{DataType, Uni, Value};

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

#[ignore = "uni-db takes no lock on a store directory: a second handle opens and writes alongside the first"]
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

    let b = Uni::open(path.to_string_lossy()).build().await;
    let Ok(b) = b else {
        println!("second open refused: {}", b.err().unwrap());
        return;
    };
    insert(&b, 2).await;
    insert(&a, 3).await;
    println!(
        "second open succeeded; a sees {} docs, b sees {} (3 were written in all)",
        count(&a).await,
        count(&b).await
    );
    a.flush().await.unwrap();
    b.flush().await.unwrap();
    drop((a, b));
    let reopened = Uni::open(path.to_string_lossy()).build().await.unwrap();
    println!(
        "after both flush and the store reopens: {} docs",
        count(&reopened).await
    );
    panic!("a second handle opened a store another handle had open");
}
