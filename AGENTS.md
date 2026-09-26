# AGENTS.md

Guidance for AI coding agents working in this repository. Humans should read
[`CONTRIBUTING.md`](CONTRIBUTING.md); this file is the agent-facing distillation
plus the invariants that are *not* discoverable from the code alone.

uniko is an embedded, Rust-native cognitive memory engine for AI agents, built
as a Cargo workspace on top of [uni-db](https://crates.io/crates/uni-db). It
links into a host process like SQLite — graph, vector, full-text, and Locy logic
in one in-process engine, with `$0` LLM-free ingest.

---

## Build

```sh
cargo build
```

- **Rust:** stable channel (pinned in `rust-toolchain.toml`), **edition 2024**,
  **MSRV `1.91`** (`rust-version` in the workspace `Cargo.toml`).
- `cargo build` pulls `uni-db` (`^4` — the latest 4.x) and `uni-xervo` (`0.18.1`)
  straight from crates.io — **no token, private repo, or credentials required.**
  Those are the requirements in the workspace `Cargo.toml`; `Cargo.lock` has the
  exact resolved versions.
- **System deps:** `protobuf-compiler` (`protoc`), a C/C++ toolchain (the stack
  statically links ONNX Runtime via `ort`), and — on Linux — `mold`, which
  `.cargo/config.toml` forces as the link backend. CI installs
  `protobuf-compiler mold`.
- First builds are slow (ONNX Runtime, tokenizers compile under `opt-level = 3`).

## Test

Use **`cargo nextest`**, never `cargo test` — nextest is the runner of record in CI.

```sh
cargo nextest run --workspace            # full suite
cargo nextest run -p uniko-memory        # one crate
cargo nextest run -E 'test(recall_cascade)'   # filter by name
```

### Recall needs more than a 2 MiB stack

An ingest-plus-recall pass needs **between 1 and 2 MiB of stack**, which sits
right at the default for a spawned thread. libtest gives each test a 2 MiB
thread (not the main thread's 8 MiB), and tokio's workers default to 2 MiB
too, so anything running recall on a spawned thread is close to the edge.

The failure mode is a `SIGABRT` stack overflow, not an error: the process dies,
so whatever else that test run reported becomes untrustworthy. It reproduces
deterministically with `RUST_MIN_STACK=1048576` and disappeared in isolation,
which is why it read as a ~50% flake under suite load for a long time.

The depth is in the store's query execution, not in uniko's own frames —
boxing the large ingest and recall futures did not move the threshold, no
single store query is deep (see
`crates/uniko-store/tests/stack_depth_repro.rs`), and under a 256 KiB stack
the overflow lands on uni-db's `uni-io` thread. So give it headroom rather
than trying to shrink it:

- a test that drives ingest + recall should run on an explicit thread — see
  `crates/uniko-bench/tests/smoke_test.rs`, which uses 16 MiB;
- a host embedding uniko should size the thread that calls `recall` (for tokio,
  `Builder::thread_stack_size`), or call it from the main thread.

### macOS: put `TMPDIR` on a RAM disk

`KnowledgeBase::in_memory()` is not actually in memory — uni-db materializes
it as a store directory under `TMPDIR`, and standing one up writes the whole
schema (26 node types, 55 edge types, plus indexes) as many small files. One
store per test, N tests in parallel.

On macOS that lands on the APFS Data volume, where small-file writes get
pathologically slow as the volume fills. Measured on a volume at 87%: **200
small files + `sync` took 5.45 s** (~27 ms per file) versus **0.06 s** on a RAM
disk — about 90x. The visible symptom is not an error: tests sit at ~3% CPU
with ~60 MB RSS and no swap, making no progress, and `cargo nextest` reports
them only as `SLOW [>60s]` forever. It looks exactly like a deadlock and is
not one.

```sh
diskutil erasevolume HFS+ unikoram $(hdiutil attach -nomount ram://8388608)
export TMPDIR=/Volumes/unikoram
```

The same nine `ingest_atomic_tests` go from never finishing in >120 s to
**2.5 s total**. Linux CI runs on fast storage and never sees this, so a green
CI says nothing about whether your local run will hang. Eject with
`hdiutil detach /Volumes/unikoram` when done; re-create it after a reboot.

## The check loop (mirrors CI exactly — run before declaring work done)

```sh
cargo fmt --all --check                  # format (use without --check to fix)
cargo clippy --workspace -- -D warnings  # lint — warnings are errors
cargo check --workspace                  # compile
cargo nextest run --workspace            # tests
cargo deny check                         # license + advisory policy (deny.toml)
# + the uni-db seal (below)
```

CI lives in `.github/workflows/ci.yml`. If a change touches dependencies, expect
`cargo deny check` to gate the license allow-list.

---

## Architectural invariants (do not break these)

### 1. The uni-db seal

**The product crates `uniko-memory`, `uniko-extract`, `uniko-cortex`, and
`uniko-pipes` must reach the graph only through `uniko-store`'s typed API.** They
must never `use uni_db` or call the `.db()` escape hatch. `uniko-store` *is* the
boundary; `uniko-api` only composes the product crates. CI enforces this with a
ripgrep gate:

```sh
rg -n -e 'use uni_db' -e '\.db\(\)' \
  crates/uniko-memory/src crates/uniko-extract/src \
  crates/uniko-cortex/src crates/uniko-pipes/src \
  | grep -vE ':[0-9]+:[[:space:]]*//' | grep -v 'ALLOW:'
```

No output = intact. Comments are exempt; `tests/` and `uniko-bench` are out of
scope; a reviewed exception may be tagged `// ALLOW:` on the same line. If you
need a graph op that `uniko-store` doesn't expose, **add it to `uniko-store`** and
call it from there — never reach past the boundary.

### 2. Crate layering

Layer numbers rank *meaning*, not dependency direction. Most crates depend only
on lower layers, with one deliberate exception: `uniko-memory` (L4) depends on
`uniko-cortex` (L5) because cortex's P5/P6 sweeps subscribe to memory's
consolidation.

| Crate | Layer | Responsibility |
|---|---|---|
| `uniko-store` | 1 | Graph storage, search (vector/fulltext/hybrid), Locy runtime. **Only crate that touches uni-db.** |
| `uniko-pipes` | 2 | Pipeline infra — `Step` trait, circuit breaker, retry, DLQ, metrics. |
| `uniko-extract` | 3 | NER, observations, chunking, ingest, embeddings. |
| `uniko-memory` | 4 | Recall cascade, consolidation, rules, orchestration. |
| `uniko-cortex` | 5 | Procedures, topics, planning. |
| `uniko-api` | facade | Public facade: builders + re-exports, no logic. |

Plus `uniko-bench` (`publish = false`) and `bindings/uniko-py`.

### 3. uni-db is a SEPARATE project — never edit it

uni-db is consumed from crates.io (`uni-db = "4"`). A local checkout may exist
at `../uni/` or `../uni-db/` for reference only — **never edit it directly**, and
never leave a `[patch.crates-io]` pointing at it in a commit. When you hit a
uni-db bug: build a *minimal isolated repro* (see
`crates/uniko-store/tests/unidb_bytes_return_repro.rs` for the pattern), file it
upstream against `rustic-ai/uni-db`, and submit a PR there rather than working
around it silently in uniko.

---

## Python bindings (`bindings/uniko-py`)

Async-first PyO3 SDK over the `Uniko` facade, built with
[maturin](https://www.maturin.rs/) and managed with [uv](https://docs.astral.sh/uv/).

```sh
cd bindings/uniko-py
uv run maturin develop   # compile the extension into the uv venv
uv run pytest            # run the Python test suite
```

Needs `protoc` + a C/C++ toolchain on `PATH`. No prebuilt wheels yet.

## Documentation site (`website/`)

Static site built with [Zensical](https://zensical.org/), **uv-managed** (migrated
off poetry on 2026-06-19):

```sh
cd website
uv sync
uv run zensical serve    # local preview with hot reload
uv run zensical build    # build into ./site
```

When editing docs, verify claims against source — the site was audited
file-by-file against the code. Schema counts, benchmark numbers, and API
signatures must trace to a source `file:line`, not to design docs.

---

## Source-of-truth gotchas

- **Schema** lives in `crates/uniko-store/src/schema/constants.rs`
  (`labels::ALL` and `edges::ALL`). Current counts: **26 node types, 55 edge
  types.** Ignore stale `schema/mod.rs` doc-comments with lower numbers.
- **Effective config defaults** come from `UnikoConfig::default()` — the pipeline
  builds `RecallConfig::from_uniko_config` / `ChunkConfig::from_uniko_config` from
  it. The standalone `RecallConfig::default()` / `ChunkConfig::default()` differ;
  **do not quote those as the runtime defaults.**
- **Fact visibility** scheme is `null` / `public` / `private:{id}` / `team:{id}` /
  `org:{id}` (in `policy.rs`), not the older `agent`/`global` strings.

---

## Commit & PR conventions

- **Do not commit or push without explicit maintainer approval.**
- **Never add `Co-Authored-By` or other attribution/trailer lines** to commits.
- Conventional-commit prefixes consistent with history: `feat:`, `fix:`,
  `refactor:`, `docs:`, `deps:`, `test:`, optionally scoped (`feat(api): …`).
- Imperative subject under ~72 chars; explain *why* in the body. One logical
  change per commit; keep PRs scoped.
- Branch off `main` (e.g. `feat/recall-decay`, `fix/ingest-ssi-conflict`).
- A behavioral change ships with tests; CI must be green before merge.
