//! Regression repro for a uni-db bug: a `DataType::Bytes` property cannot be
//! read back through a Cypher `RETURN`.
//!
//! Symptom (observed via `KnowledgeBase::fetch_blob` on the Lance backend):
//!
//! ```text
//! CypherValue decode error: Storage error: unknown CypherValue tag: 35
//! ```
//!
//! Root cause: `DataType::Bytes`, `DataType::CypherValue`, and
//! `DataType::Duration` all map to Arrow `LargeBinary`
//! (`uni-common/src/core/schema.rs:171-172`), distinguished only by the schema
//! `DataType`. `Bytes` columns store **raw** bytes; `CypherValue` columns store
//! **tagged** MessagePack. On read, `arrow_to_value(col, row, data_type)`
//! (`uni-store/src/storage/arrow_convert.rs:177`) returns raw bytes only when
//! `data_type == Some(DataType::Bytes)` (line 262); with `None` it falls into
//! the generic `LargeBinaryArray` branch (line 536) that runs
//! `cypher_value_codec::decode`, reading `byte[0]` as a type tag. The Cypher
//! `RETURN` projection drops the `DataType::Bytes` context, so the codec is
//! applied to raw bytes — erroring when `byte[0]` is an unmapped tag, or
//! silently corrupting when it is a valid one (0..=19).
//!
//! **FIXED upstream.** These tests assert the correct behavior (a clean
//! round-trip) and now pass — verified on uni-db 3.0.1 and 3.2.0, so the fix
//! landed in 3.0.0 or earlier. They ran as `#[ignore]`d expected-failures until
//! 2026-08-02, when re-running the ignored set showed them green; the attribute
//! is removed so they now guard against regression on every run. Full write-up
//! of the original defect: `crates/uniko-store/docs/unidb-bytes-cypher-return-bug.md`.

use uni_db::{DataType, Uni, Value};

#[tokio::test]
async fn bytes_column_round_trips_through_cypher_return() {
    let db = Uni::in_memory().build().await.expect("open in-memory db");

    db.schema()
        .label("Blob")
        .property("id", DataType::String)
        .property_nullable("data", DataType::Bytes)
        .done()
        .apply()
        .await
        .expect("register schema");

    // A markdown document — first byte is '#' = 0x23 = 35, which is not a
    // valid CypherValue type tag.
    let payload = b"# Spec\n\n- requirement one".to_vec();

    let tx = db.session().tx().await.expect("begin tx");
    tx.query_with("CREATE (b:Blob {id: 'b1', data: $data})")
        .param("data", Value::Bytes(payload.clone()))
        .fetch_all()
        .await
        .expect("create node with bytes");
    tx.commit().await.expect("commit");

    // Read the bytes back via a plain Cypher RETURN.
    let session = db.session();
    let result = session
        .query_with("MATCH (b:Blob {id: 'b1'}) RETURN b.data AS data")
        .fetch_all()
        .await
        .expect("query itself should not fail");

    let row = result.rows().first().expect("one row");
    let got = row.value("data");

    // EXPECTED: Some(Value::Bytes(payload)). ACTUAL (bug): the decode logs
    // "CypherValue decode error: unknown CypherValue tag: 35" to stderr and
    // yields Value::Null.
    assert!(
        matches!(got, Some(Value::Bytes(b)) if b == &payload),
        "Bytes column did not round-trip through RETURN; got {got:?} \
         (mis-decoded as a tagged CypherValue: byte[0]=0x23=35)"
    );
}

/// The same root cause, but worse: when the payload's first byte *is* a valid
/// CypherValue tag (here `0x00` = TAG_NULL), the decode succeeds and **silently
/// returns the wrong value** — no error is logged. Any binary blob is at risk.
#[tokio::test]
async fn bytes_with_tag_valued_first_byte_silently_corrupts() {
    let db = Uni::in_memory().build().await.expect("open in-memory db");
    db.schema()
        .label("Blob")
        .property("id", DataType::String)
        .property_nullable("data", DataType::Bytes)
        .done()
        .apply()
        .await
        .expect("register schema");

    // First byte 0x00 == TAG_NULL; the rest is real data that must survive.
    let payload = vec![0x00, 0xDE, 0xAD, 0xBE, 0xEF];

    let tx = db.session().tx().await.expect("begin tx");
    tx.query_with("CREATE (b:Blob {id: 'b2', data: $data})")
        .param("data", Value::Bytes(payload.clone()))
        .fetch_all()
        .await
        .expect("create");
    tx.commit().await.expect("commit");

    let result = db
        .session()
        .query_with("MATCH (b:Blob {id: 'b2'}) RETURN b.data AS data")
        .fetch_all()
        .await
        .expect("query");
    let got = result.rows().first().expect("one row").value("data");

    // EXPECTED: the 5-byte payload. ACTUAL (bug): Value::Null — byte[0]=0x00
    // was read as TAG_NULL and the remaining bytes discarded, with NO error.
    assert!(
        matches!(got, Some(Value::Bytes(b)) if b == &payload),
        "Bytes column silently corrupted; got {got:?} (byte[0]=0x00 read as TAG_NULL)"
    );
}
