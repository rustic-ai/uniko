# uni-db bug: `DataType::Bytes` property cannot be read via Cypher `RETURN`

**Component:** uni-db (`uni-store` arrow conversion + `uni-common` cypher value codec)
**Severity:** High — silent data corruption / data loss for any binary property read through Cypher.
**Status:** **FIXED upstream** (verified 2026-08-02 on uni-db 3.0.1 and 3.2.0, so the fix
landed in 3.0.0 or earlier). Originally verified with a standalone repro against the uni-db
public API; the document is kept for the root-cause analysis.
**Repro test:** `crates/uniko-store/tests/unidb_bytes_return_repro.rs` — no longer `#[ignore]`d;
it now runs on every `cargo nextest run` as a regression guard.

## Summary

A node/edge property declared `DataType::Bytes` stores raw bytes, but reading that property back
through a Cypher `RETURN` mis-decodes it: the projection drops the `DataType::Bytes` schema context,
so the read path applies the **tagged `CypherValue` MessagePack codec** to the **raw** bytes, treating
`byte[0]` as a type tag. Depending on the first byte the result is either an error or a silently wrong
value. A `Bytes` value never round-trips through a Cypher query.

## Reproduction

Pure `uni_db` public API — no consumer-side code involved:

```rust
use uni_db::{DataType, Uni, Value};

let db = Uni::in_memory().build().await?;
db.schema()
    .label("Blob")
    .property("id", DataType::String)
    .property_nullable("data", DataType::Bytes)
    .done()
    .apply()
    .await?;

// First byte '#' = 0x23 = 35.
let payload = b"# Spec\n\n- requirement one".to_vec();

let tx = db.session().tx().await?;
tx.query_with("CREATE (b:Blob {id: 'b1', data: $data})")
    .param("data", Value::Bytes(payload.clone()))
    .fetch_all().await?;
tx.commit().await?;

let result = db.session()
    .query_with("MATCH (b:Blob {id: 'b1'}) RETURN b.data AS data")
    .fetch_all().await?;
let got = result.rows()[0].value("data");
assert!(matches!(got, Some(Value::Bytes(b)) if b == &payload)); // FAILS
```

### Observed behavior

Two failure modes, both confirmed:

1. **Error + data loss** — `byte[0]` is an unmapped tag (here `0x23` = 35). stderr logs
   `CypherValue decode error: Storage error: unknown CypherValue tag: 35` and the value comes back
   `Value::Null`.
2. **Silent corruption** — `byte[0]` is a *valid* tag (e.g. `0x00` = `TAG_NULL`). No error is logged;
   the decode "succeeds" and returns the wrong value (`Null`, discarding the real bytes).

Expected: `Value::Bytes(payload)` in both cases.

## Root cause

`DataType::Bytes`, `DataType::CypherValue`, and `DataType::Duration` all map to the **same** Arrow
type, `LargeBinary`:

- `uni-common/src/core/schema.rs:171` — `DataType::Duration => ArrowDataType::LargeBinary`
- `uni-common/src/core/schema.rs:172` — `DataType::CypherValue => ArrowDataType::LargeBinary`
- `uni-store/src/storage/arrow_convert.rs:1891` (test) — `DataType::Bytes.to_arrow() == LargeBinary`

They differ only in **payload encoding**:

- `DataType::Bytes` columns are built with **raw** bytes — `build_bytes_column`
  (`arrow_convert.rs:1092`, `:1456`) → `LargeBinaryBuilder.append_value(b)`.
- `DataType::CypherValue` / `Duration` columns are built with **tagged MessagePack** via
  `cypher_value_codec` (`arrow_convert.rs:1091`, `:1301`).

On read, `arrow_to_value(col, row, data_type: Option<&DataType>)`
(`arrow_convert.rs:177`) branches on the schema type:

- With `Some(DataType::Bytes)` → returns raw bytes directly (`arrow_convert.rs:262-271`). **Correct.**
- With `None` (no schema context) → falls through to the generic `LargeBinaryArray` branch
  (`arrow_convert.rs:536-546`), which calls `uni_common::cypher_value_codec::decode(bytes)` —
  expecting a tagged blob.

The `CypherValue` codec reads `byte[0]` as the tag (`uni-common/src/cypher_value_codec.rs`, tag table
`TAG_NULL=0 … TAG_BYTES=7 … TAG_BTIC=19`); an unmapped tag returns
`unknown CypherValue tag: {tag}` (`cypher_value_codec.rs:242-245`).

**The Cypher `RETURN` projection path calls `arrow_to_value` without the column's `DataType::Bytes`
context** (`data_type = None`), so raw `Bytes` columns are fed to the codec. (The runtime decode
attempt — the `unknown tag` error — proves the projection reaches the generic branch.)

### Why it's latent

uni-db's own `arrow_convert` tests exercise only the **with-context** path —
`arrow_to_value(arr, i, Some(&DataType::Bytes))` (`arrow_convert.rs:1846-1891`), including the comment
*"A LargeBinary column tagged as DataType::Bytes must NOT be decoded [as CypherValue]"*. The
`cypher_value_codec` round-trip test (`cypher_value_codec.rs:587`) also passes. No test stores a
`Bytes` property and reads it back through a **Cypher query**, so the missing-context path is never
exercised.

## Impact

Any `DataType::Bytes` property is unreadable via Cypher `RETURN`. In uniko this is
`:ArtifactContent.bytes` (inline blob storage on the Lance backend): `KnowledgeBase::fetch_blob`
(`crates/uniko-store/src/storage/blob.rs:170`) reads `RETURN c.bytes`, gets `Null` instead of the
bytes, and falls over to `LanceBlobStore::get`, which is intentionally not callable — so
`agent.data().artifact_bytes(..)` cannot return inline bytes on the in-memory/Lance backend. The
Fs/S3 backends use the `uri` path and are unaffected. More broadly, any consumer storing binary blobs
in a `Bytes` column and reading them via Cypher gets silent corruption.

## Suggested fix (in uni-db)

Thread the column's schema `DataType` through the Cypher projection so `arrow_to_value` receives
`Some(DataType::Bytes)` and takes the raw-bytes branch (`arrow_convert.rs:262`). Because `Bytes`,
`CypherValue`, and `Duration` are indistinguishable at the Arrow `LargeBinary` level, the schema type
is the only disambiguator and must be preserved through projection. Alternatives: attach Arrow field
metadata marking a `LargeBinary` column as raw-`Bytes` vs `CypherValue`-encoded; or give raw `Bytes`
a distinct Arrow type. A regression test should store a `Bytes` property and assert it round-trips
through a Cypher `RETURN` (the repro above).
