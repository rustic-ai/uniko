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
- `cargo build` pulls `uni-db` (`^3` — the latest 3.x) and `uni-xervo` (`0.17.0`)
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

uni-db is consumed from crates.io (`uni-db = "3"`). A local checkout exists at
`../uni/` for reference only. **Never edit `../uni/` directly.** When you hit a
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
  (`labels::ALL` and `edges::ALL`). Current counts: **24 node types, 53 edge
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
