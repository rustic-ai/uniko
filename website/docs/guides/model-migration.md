# Changing the Embedding Model

A store's vectors are tied to the model and settings that made them. uni-db fixes each vector
column's type when the store is created and can't change it later, so some config changes need a
fresh store, some can be made in place, and uniko tells you which before it touches anything.

## Inspect a store

`Uniko::inspect_store` reports what a store was built with and how a config compares. It reads
the stored schema and the stats row's model record. It applies no schema, changes no index or
data, and loads no model. uni-db's own open and close of the store may still write its
bookkeeping files, such as a manifest and its WAL.

```rust
use uniko_memory::Uniko;
use uniko_store::config::UnikoConfig;

let report = Uniko::inspect_store("./memory", &UnikoConfig::default()).await?;
println!("{:?}", report.info.embedding);   // model, dimensions, hybrid columns, index settings
for d in &report.compatibility.differences {
    println!("{d}");   // e.g. "dimensions: store has 384, config asks for 1024 (... fresh store)"
}
```

```python
report = uniko.Uniko.inspect_store_sync("./memory", embedding="bge-m3")
report.compatible            # False
report.differences[0].field  # "dimensions"
report.migrations            # [] — nothing in place can fix it
```

`migrations` lists the migrations that would make the store open. It is empty whenever any
difference is incompatible, since no migration would help then. So a BGE-small store checked
against `bge-m3` reports `[]`, even though it also lacks the hybrid columns.

What inspecting does and doesn't touch:

- A local path that isn't a uni-db store (neither `catalog/schema.json` nor a root
  `schema.json`), such as a missing or empty directory, reports `exists: false` with no
  differences. It isn't touched at all: nothing is created there.
- A store in an object store is inspected and migrated like a local one: through its local
  directory with [`remote_storage`](configuration.md#remote-storage) set (hybrid), or by its
  `s3://` or `gs://` URL (remote-only). Any other URI is refused with an `Unsupported` error.
- A store that is open, or being inspected or migrated, in this process or another, is refused
  with a `Config` error. Ask the open instance instead: `memory.store_info()` reports its own store.
  `store_info().hybrid_backfill_pending` counts the rows a hybrid backfill still has to fill.
- Stores whose model is served by a `remote/*` provider or `llamacpp` can be inspected and
  migrated too. To open a hybrid store for inspection, uni-db needs an `embed/hybrid` alias in the
  catalog. When the config doesn't provide one, uniko adds a lazy `local/onnx` stub that only has
  to exist and never loads.

## What opening checks

Opening a store runs the same comparison **before the schema is applied and before any model
loads**. Any difference is refused with `UnikoError::IncompatibleStore`, and the store is left
exactly as it was. In Python it's `IncompatibleStoreError`, a subclass of `ConfigError`, so
`except ConfigError` still catches it. Its `.kind` is `"incompatible_store"`, not `"config"`. It
carries `.differences` (dicts with `field`, `stored`, `requested`, `verdict` and `migration`),
`.compatible` and `.migrations`.

| Difference | Verdict | What to do |
|---|---|---|
| Another model, or other dense dimensions | Incompatible | Open with the original config, or use a fresh store |
| A hybrid store opened with a dense-only config | Incompatible | Use the hybrid config |
| Hybrid columns asked for over a different model | Incompatible | Fresh store |
| Hybrid columns asked for over a store with no recorded model id | Incompatible | Reopen once with the original config to record it, then migrate |
| Another `document_prefix` | Incompatible | The stored vectors carry the old prefix: fresh store |
| Same model and dimensions, sparse and/or ColBERT columns asked for | `enable_hybrid` | Migrate in place |
| Another vector index algorithm or metric, `query_prefix` or `batch_size` | `rebuild_vector_indexes` | Migrate in place |

`enable_hybrid` also covers a sparse-only store gaining the ColBERT column.

A store from before the model id was recorded is checked on its shape only: its dimensions,
hybrid columns and index settings. Another model with the same dimensions shows no difference,
so the check can't catch it, and opening the store with that model records it as the store's
model. Open such a store once with its original config, which records the right model id. A PQ algorithm with
`sub_vectors: 0` (automatic) is compared the way uni-db stores it, with the count resolved, so the
store reopens with the same config.

The open-time check reads only the schema and the stats row. Counting the rows a backfill still
has to fill scans chunks and observations, so only `inspect_store`, `store_info` and
`migrate_store` do it.

A config with `schema_path` set skips the check, since its column types come from that file. A
changed model or dimensions is still refused at open, with a `Config` "embedding mismatch" error.
`migrate_store` refuses a `schema_path` config with a `Config` error, since the schema would come
from that file and not from the config being migrated to.

## Migrate in place

```rust
use uniko_store::Migration;

let report = Uniko::migrate_store("./memory", config, Migration::EnableHybrid, |p| {
    println!("{} {:?} {}/{}", p.phase, p.label, p.done, p.total);
})
.await?;
assert!(report.completed);
```

```python
uniko.Uniko.migrate_store_sync(
    "./memory", "enable_hybrid", embedding="bge-m3", progress=print
)
```

- **`enable_hybrid`** applies to a store built with the hybrid model in dense-only mode, for
  example BGE-M3 with its sparse and ColBERT outputs off:
  - it adds whichever of the `sparse_embedding` and `colbert_embedding` columns are missing, and
    their indexes;
  - then it re-embeds existing chunks and observations so the new columns are filled. The
    embedding model loads for this. Every auto-embedded column is rewritten in that pass, the
    existing dense `embedding` included, which the hybrid model then produces: the backfill costs
    a full re-embed, not one sized to the new columns.

  The backfill walks the Chunk and Observation rows once each, in node-id order, a batch per
  transaction. A row whose new columns still come back empty, such as a text the model returns no
  sparse terms for, is counted in `MigrationReport.rows_unfilled` and passed over in that run. It
  doesn't loop or raise an error. The row still counts as pending, so
  `hybrid_backfill_pending` doesn't reach 0 and a rerun tries it again. The migration still
  reports `completed` when the only pending rows left are the ones it couldn't fill.

  The backfill is resumable: a rerun picks up only the rows whose hybrid columns are still empty.
  While it's unfinished the store already opens with the hybrid config, and
  `store_info().hybrid_backfill_pending` says how many rows are left.

  There's no way back. Once the schema step has added the columns, the original dense-only
  config is refused, because a hybrid store opened dense-only is incompatible. A cut-off
  `enable_hybrid` is finished by rerunning it, never undone.
- **`rebuild_vector_indexes`** rebuilds the vector indexes for a new algorithm, metric,
  `query_prefix` or `batch_size`. No stored vector is rewritten and no model loads.

A migration refuses to start unless it resolves every difference. `enable_hybrid` also applies
any index change.

A run marks the stats row until it finishes, and `store_info().migration_in_progress` reports
the marker. The marker is set before the schema step when the store already has the stats row's
`migration` column, and after it otherwise: an older store gets the column from that step. Only
the migration kind that set a marker clears it. For example, a `rebuild_vector_indexes` run
leaves an unfinished `enable_hybrid` marker in place. `MigrationReport.migration_in_progress`
names a migration still marked unfinished after the run, which is why `completed` can be false
after a run that did its own part: rerun that migration to finish it.

A normal open checks the marker too, and the marker records which step a run was in. A run cut off
in its schema step, which includes building indexes, refuses the next open with `IncompatibleStore`,
its difference `migration_in_progress`, and `migrations` naming the migration to rerun: uni-db may
hold the new index definitions with indexes only partly built, which would quietly miss results.
This applies to `rebuild_vector_indexes` and to `enable_hybrid`, whose schema step adds and builds
indexes too. An `enable_hybrid` cut off in its backfill opens, with a warning in the log: the data is
correct, the rows it didn't reach just take no part in sparse or ColBERT recall, and
`store_info().hybrid_backfill_pending` counts them.

Recover by rerunning the same migration with the same config, with the store closed. It sees its own
marker and picks up where it stopped. Resuming a cut-off schema step rebuilds every vector-indexed
label outright, because uni-db saves a new index definition before building it and skips the build
when the definition already matches. That rebuild covers every index on those labels, so it costs a
full re-index. `enable_hybrid` covers a cut-off `rebuild_vector_indexes` too, so when both are
needed `migrations` lists `enable_hybrid` alone. uni-db logs a failed index build as a warning
rather than an error, so a run can report `completed` over an index whose build failed: check the
log after a migration.

The report's `completed` is true when the store needs nothing further for this config.
`rows_backfilled` counts the rows filled and `rows_unfilled` the rows passed over.

A store takes one writer at a time: one open handle, or one inspection or migration.

- Within the process, a registry of claimed paths refuses a second open, and refuses inspecting
  or migrating an open store, at once with a `Config` error. Share one instance instead: its
  clones (and every `Agent` and `Session` from it) use the same store.
- Across processes, uni-db holds the store for its writer: an OS lock on `writer.lock` in a local
  store's directory, or, for a store in an object store
  ([`remote_storage`](configuration.md#remote-storage)), a lease at `catalog/writer_lease.json`
  under the store's prefix. Another process opening, inspecting or migrating the store waits a
  few seconds for it, then is refused with a `Config` error naming the holder. The lock goes when
  the handle closes or its process exits; the lease is released at shutdown, and one a crashed
  holder left lapses 30 seconds after its last renewal.
- A writer whose lease was taken over (it stalled past the lease, a paused VM) is fenced: uni-db
  refuses its commits from then on, with a `Config` error. Shut that instance down.
- Inspecting a directory that isn't a store opens nothing and creates nothing there.

The lease is tested against two local S3 implementations that honor conditional writes,
[LocalStack](https://github.com/localstack/localstack) and the
[AWS Emulator](https://github.com/aws/aws-emulator). It hasn't been tested against AWS S3 or GCS
themselves.

On a network file system that doesn't support file locks, the cross-process check may not hold.

`progress` is called after each phase and backfill batch. Each event carries:

- `migration`;
- `phase`: `"schema"`, `"backfill"` or `"done"`;
- `label`: the label being backfilled, during `"backfill"`;
- `done` and `total`: during `"backfill"`, the rows done so far in that label (filled plus
  unfilled) and the rows it had to do when its backfill started. The final `"done"` event has
  `done == total == rows_backfilled + rows_unfilled`.

In Python, `progress` gets a dict with those keys (typed as `uniko.types.MigrationProgressEvent`),
and the argument order is `migrate_store(path, migration, config=None, ...)`, where Rust's is
`(path, config, migration, progress)`. About the callback:

- an exception or panic in it is printed and ignored;
- it can't cancel a migration;
- an `async def` callback isn't awaited, so its body doesn't run;
- it must not call other `_sync` uniko methods.

## Changing the model or dimensions

A different model, or the same model at another dimension size, can't be migrated in place.
Its vectors aren't comparable with the stored ones, and uni-db can't change a column's type. Open
a new store with the new config and re-ingest your sources into it. The old store keeps working
with its original config until you retire it. The same applies to a new `document_prefix`.

## Shared runtimes

Over a host-built `SharedRuntime`, the runtime's models may already be loaded before uniko sees
the store, so "before any model loads" applies only to models the store itself would load. The
store is still refused before it changes.
