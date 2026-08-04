//! Standalone repro for a **residual** uni-db bug, filed upstream as
//! rustic-ai/uni-db #100 — a follow-up to #93 ("preserve DataType::Bytes
//! through Cypher projection", CLOSED, fixed in 2.2.4).
//!
//! #93 fixed the *scalar* projection path: `MATCH (b) RETURN b.data` now
//! round-trips a `DataType::Bytes` column on 2.2.4. But the
//! **`collect()` aggregation path was not fixed** — `RETURN collect(b.data)`
//! still feeds the raw bytes to the tagged `CypherValue` MessagePack codec,
//! which reads `byte[0]` as a type tag:
//!
//! ```text
//! CypherValue decode error: Storage error: unknown CypherValue tag: 97
//! ```
//!
//! (`97` = `0x61` = `'a'`, the first byte of the payload below — any
//! unmapped first byte errors; a valid-tag first byte, e.g. `0x00`, would
//! silently corrupt instead.) The value is then dropped, so the aggregate
//! comes back as an **empty list** rather than `[Bytes(..)]`.
//!
//! Root cause (same family as #93): `Bytes`/`CypherValue`/`Duration` all map
//! to Arrow `LargeBinary` and are disambiguated only by the schema
//! `DataType`. #93 threaded that type through scalar projection
//! (`uni-query .../executor/read.rs:797`), but the `collect()` accumulator
//! re-materialises the element without the `Some(DataType::Bytes)` context,
//! so `arrow_to_value(.., None)` runs the codec on raw bytes
//! (`uni-store .../arrow_convert.rs:177`, generic LargeBinary branch).
//!
//! Verified against uni-db 2.2.4 (the latest 2.x). Discovered as 156×
//! `unknown CypherValue tag: 97` during a normal uniko bench ingest/recall,
//! which uses `collect()` over LargeBinary-backed columns.
//!
//! **FIXED upstream.** Both tests assert the correct behaviour (a clean
//! round-trip through `collect()`) and now pass — verified on uni-db 3.0.1 and
//! 3.2.0, so the fix landed in 3.0.0 or earlier. They ran as `#[ignore]`d
//! expected-failures until 2026-08-02; the attribute is removed so they now
//! guard against regression on every run.

use uni_db::{DataType, Uni, Value};

/// Payload whose first byte is `'a'` = `0x61` = 97 — reproduces the exact
/// `unknown CypherValue tag: 97` symptom. Embedded NUL/0xFF bytes ensure it
/// is genuinely binary and must survive verbatim.
fn payload() -> Vec<u8> {
    b"audio-fingerprint-\x00\xff\x01-blob".to_vec()
}

async fn db_with_one_blob() -> Uni {
    let db = Uni::in_memory().build().await.expect("open in-memory db");
    db.schema()
        .label("Blob")
        .property("id", DataType::String)
        .property_nullable("data", DataType::Bytes)
        .done()
        .apply()
        .await
        .expect("register schema");

    let tx = db.session().tx().await.expect("begin tx");
    tx.query_with("CREATE (b:Blob {id: 'b1', data: $d})")
        .param("d", Value::Bytes(payload()))
        .fetch_all()
        .await
        .expect("create node with bytes");
    tx.commit().await.expect("commit");
    db
}

/// BUG: `collect()` over a `Bytes` column does not round-trip on uni-db
/// 2.2.4 — the value is mis-decoded (logs `unknown CypherValue tag: 97`)
/// and dropped, yielding an empty list.
#[tokio::test]
async fn collect_over_bytes_column_round_trips() {
    let db = db_with_one_blob().await;

    let result = db
        .session()
        .query_with("MATCH (b:Blob {id: 'b1'}) RETURN collect(b.data) AS items")
        .fetch_all()
        .await
        .expect("query itself should not fail");
    let got = result.rows().first().and_then(|r| r.value("items"));

    // EXPECTED: Some(List([Bytes(payload)])).
    // ACTUAL (bug): Some(List([])) — the Bytes element was fed to the
    // CypherValue codec, errored on tag 97, and was discarded.
    let ok = matches!(
        got,
        Some(Value::List(items))
            if items.len() == 1
            && matches!(&items[0], Value::Bytes(b) if b == &payload())
    );
    assert!(
        ok,
        "collect() dropped the Bytes value; got {got:?} \
         (raw bytes mis-decoded as a tagged CypherValue: byte[0]=0x61=97)"
    );
}

/// CONTROL: the scalar projection path #93 fixed *does* round-trip on
/// 2.2.4 — included to show the residual gap is specific to `collect()`
/// (this test passes; the one above fails).
#[tokio::test]
async fn scalar_return_of_bytes_column_round_trips_post_93() {
    let db = db_with_one_blob().await;

    let result = db
        .session()
        .query_with("MATCH (b:Blob {id: 'b1'}) RETURN b.data AS data")
        .fetch_all()
        .await
        .expect("query");
    let got = result.rows().first().and_then(|r| r.value("data"));

    assert!(
        matches!(got, Some(Value::Bytes(b)) if b == &payload()),
        "scalar RETURN of a Bytes column should round-trip post-#93; got {got:?}"
    );
}
