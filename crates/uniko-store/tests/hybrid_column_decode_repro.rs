//! Isolated repro for the hybrid-path `unknown CypherValue tag: 97` decode error.
//!
//! The bge-m3 hybrid e2e run emitted hundreds of
//! `CypherValue decode error: unknown CypherValue tag: 97`
//! (`uni-store/storage/arrow_convert.rs:543`) while reading the hybrid-only
//! columns. Valid CypherValue tags are 0..=20; `97` = ASCII `'a'`, i.e. a value
//! encoded by a NON-canonical path being read by the tagged decoder (the exact
//! class documented in `uni-query/.../plugin_adapter.rs`, where `Int(42)`
//! mis-encoded as `serde_json` surfaced as "tag: 52" = `'4'`).
//!
//! This isolates WHICH column type fails to round-trip through
//! `MATCH ... RETURN n.col` (Arrow -> CypherValue decode), with NO embedder and
//! NO uniko code — plain uni-db only:
//!   * `Vector`         — dense (stored as Arrow FixedSizeList)
//!   * `SparseVector`   — sparse (stored as Arrow Struct)
//!   * `List(Vector)`   — the colbert column (stored as `List(Bytes)` whose
//!     elements are tagged-encoded CypherValues — the suspect)
//!
//! Run:
//!   cargo nextest run -p uniko-store --test integration hybrid_column_decode_repro:: \
//!       --no-capture

use std::collections::HashMap;

use uni_db::{DataType, Uni, Value};

/// Insert one node carrying `val` in a `prop` column of type `dtype`, then read
/// it back via Cypher. Returns the value as it survives the Arrow->CypherValue
/// decode (`Value::Null` if the row/column is absent).
async fn roundtrip(prop: &str, dtype: DataType, val: Value) -> Value {
    let db = Uni::in_memory().build().await.expect("open in-memory db");
    db.schema()
        .label("N")
        .property_nullable(prop, dtype)
        .done()
        .apply()
        .await
        .expect("apply schema");

    let mut m = HashMap::new();
    m.insert(prop.to_string(), val);
    let tx = db.session().tx().await.unwrap();
    tx.bulk_insert_vertices("N", vec![m]).await.unwrap();
    tx.commit().await.unwrap();

    let res = db
        .session()
        .query_with(&format!("MATCH (n:N) RETURN n.{prop} AS v"))
        .fetch_all()
        .await
        .expect("query");
    res.rows()
        .first()
        .and_then(|r| r.value("v"))
        .cloned()
        .unwrap_or(Value::Null)
}

#[tokio::test(flavor = "multi_thread")]
async fn vector_roundtrips() {
    let input = Value::Vector(vec![0.1, 0.2, 0.3, 0.4]);
    let got = roundtrip("vec", DataType::Vector { dimensions: 4 }, input.clone()).await;
    println!("Vector:       in={input:?}\n              out={got:?}");
    assert_eq!(input, got, "plain Vector should round-trip");
}

#[tokio::test(flavor = "multi_thread")]
async fn sparse_vector_roundtrips() {
    let input = Value::SparseVector {
        indices: vec![1, 5, 9],
        values: vec![0.5, 0.3, 0.2],
    };
    let got = roundtrip(
        "sp",
        DataType::SparseVector { dimensions: 100 },
        input.clone(),
    )
    .await;
    println!("SparseVector: in={input:?}\n              out={got:?}");
    assert_eq!(input, got, "SparseVector should round-trip");
}

#[tokio::test(flavor = "multi_thread")]
async fn list_vector_roundtrips() {
    // The colbert column shape: List(Vector).
    let input = Value::List(vec![
        Value::Vector(vec![0.1, 0.2, 0.3, 0.4]),
        Value::Vector(vec![0.5, 0.6, 0.7, 0.8]),
    ]);
    let got = roundtrip(
        "mv",
        DataType::List(Box::new(DataType::Vector { dimensions: 4 })),
        input.clone(),
    )
    .await;
    println!("List(Vector): in={input:?}\n              out={got:?}");
    assert_eq!(input, got, "List(Vector) (colbert) should round-trip");
}
