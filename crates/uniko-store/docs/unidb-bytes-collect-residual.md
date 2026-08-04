# uni-db bug (follow-up to #93): `collect()` over a `DataType::Bytes` column drops the value

**Component:** uni-db (`uni-query` aggregation / `collect()` projection + `uni-store` arrow conversion)
**Severity:** High — silent data loss for any binary property read through a `collect()` aggregate.
**Status:** Filed upstream as [rustic-ai/uni-db #100](https://github.com/rustic-ai/uni-db/issues/100). Verified with a standalone public-API repro against uni-db **2.2.4** (latest 2.x).
**Relationship:** Residual of [#93](https://github.com/rustic-ai/uni-db/issues/93) (CLOSED, fixed scalar projection).

> **UPDATE 2026-08-02 — FIXED.** The aggregation path round-trips too as of uni-db 3.x
> (verified on 3.0.1 and 3.2.0). `crates/uniko-store/tests/unidb_bytes_residual_repro.rs`
> is no longer `#[ignore]`d and runs as a regression guard. The analysis below describes
> the original 2.2.4 behaviour.
**Repro test:** `crates/uniko-store/tests/unidb_bytes_residual_repro.rs` (`#[ignore]`d; run with `cargo nextest run -p uniko-store --test unidb_bytes_residual_repro --run-ignored all`).

## Summary

After #93, `MATCH (b) RETURN b.data` correctly round-trips a `DataType::Bytes` property on 2.2.4. But `RETURN collect(b.data)` still feeds the **raw** bytes to the tagged `CypherValue` MessagePack codec, which reads `byte[0]` as a type tag. The value is mis-decoded and dropped, so the aggregate returns an **empty list** instead of `[Bytes(..)]`. When `byte[0]` is an unmapped tag, stderr also logs `unknown CypherValue tag: {byte0}`.

## Reproduction (pure public API)

```rust
use uni_db::{DataType, Uni, Value};

#[tokio::test]
async fn collect_over_bytes_column_round_trips() {
    let db = Uni::in_memory().build().await.unwrap();
    db.schema()
        .label("Blob")
        .property("id", DataType::String)
        .property_nullable("data", DataType::Bytes)
        .done()
        .apply().await.unwrap();

    // First byte 'a' = 0x61 = 97 (not a valid CypherValue tag).
    let payload = b"audio-fingerprint-\x00\xff\x01-blob".to_vec();

    let tx = db.session().tx().await.unwrap();
    tx.query_with("CREATE (b:Blob {id: 'b1', data: $d})")
        .param("d", Value::Bytes(payload.clone()))
        .fetch_all().await.unwrap();
    tx.commit().await.unwrap();

    // Scalar projection (fixed by #93) — round-trips OK.
    let scalar = db.session()
        .query_with("MATCH (b:Blob {id:'b1'}) RETURN b.data AS data")
        .fetch_all().await.unwrap();
    assert!(matches!(
        scalar.rows()[0].value("data"),
        Some(Value::Bytes(b)) if b == &payload
    ));

    // Aggregation projection — BROKEN: returns Some(List([])), logs
    // "CypherValue decode error: Storage error: unknown CypherValue tag: 97".
    let agg = db.session()
        .query_with("MATCH (b:Blob {id:'b1'}) RETURN collect(b.data) AS items")
        .fetch_all().await.unwrap();
    let got = agg.rows()[0].value("items");
    assert!(
        matches!(got, Some(Value::List(items)) if items.len()==1
            && matches!(&items[0], Value::Bytes(b) if b==&payload)),
        "collect() dropped the Bytes value; got {got:?}"
    );
}
```

## Observed vs expected

| Read shape | Expected | Actual on 2.2.4 |
|---|---|---|
| `RETURN b.data` (scalar) | `Bytes(payload)` | `Bytes(payload)` ✅ (#93) |
| `RETURN collect(b.data)` | `List([Bytes(payload)])` | `List([])` ❌ + `unknown CypherValue tag: 97` |

Other shapes checked and **OK** on 2.2.4: `WITH b.data AS d RETURN d`, multi-column `RETURN b.id, b.data`, `RETURN b.data ORDER BY ...`, `RETURN DISTINCT b.data`, and edge-property `RETURN r.data`. Only `collect()` regresses.

## Root cause (same family as #93)

`Bytes`/`CypherValue`/`Duration` all map to Arrow `LargeBinary`, disambiguated only by the schema `DataType`. #93 threaded that type into the scalar projection so `arrow_to_value(col, row, Some(DataType::Bytes))` takes the raw-bytes branch (`uni-store .../arrow_convert.rs:262`; `uni-query .../executor/read.rs:797`). The **`collect()` accumulator** re-materialises each element **without** that context (`arrow_to_value(.., None)`), so raw `Bytes` are run through `cypher_value_codec::decode`, which treats `byte[0]` as a tag (`uni-common/src/cypher_value_codec.rs`).

## Suggested fix

Thread the column's schema `DataType` into the aggregation/`collect()` element materialisation, the same way #93 did for scalar projection — so list elements built from a `LargeBinary` column tagged `DataType::Bytes` take the raw-bytes branch. Add a regression test that `collect()`s a `Bytes` property and asserts it round-trips.

## How it surfaced

Discovered as **156× `unknown CypherValue tag: 97`** during a normal uniko bench ingest/recall on uni-db 2.2.4, where `collect()` runs over LargeBinary-backed columns. Non-fatal (the affected reads return empty), but it is silent binary data loss for any consumer that aggregates a `Bytes` column.
