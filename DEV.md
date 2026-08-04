<!-- SPDX-License-Identifier: Apache-2.0 -->
# Developing uniko

How to build, test, and run everything in this repository from scratch. This
consolidates the scattered instructions in [`CONTRIBUTING.md`](CONTRIBUTING.md),
[`AGENTS.md`](AGENTS.md), the binding/website READMEs, and CI.

The repo has three buildable surfaces:

| Surface | Path | Toolchain |
| --- | --- | --- |
| Rust workspace (the engine) | `crates/*` | `cargo` |
| Python bindings (PyO3 SDK) | `bindings/uniko-py` | `maturin` + `uv` |
| Documentation site | `website` | `uv` + `zensical` |

---

## 1. Prerequisites

Install once, for any surface that touches the native stack:

- **Rust ≥ 1.91** (edition 2024) — `rustup toolchain install stable`. The MSRV is
  pinned by `rust-version = "1.91"` in the root `Cargo.toml`.
- **A C/C++ toolchain** — required by native ML dependencies (`ort` / ONNX
  Runtime, `tokenizers`).
- **`protobuf-compiler` (`protoc`)** — must be on `PATH`. The stack statically
  links ONNX Runtime; the build shells out to `protoc`.
  - Debian/Ubuntu: `sudo apt-get install -y protobuf-compiler`
- **`mold`** (Linux only) — **required, not optional.** `.cargo/config.toml`
  forces `-C link-arg=-fuse-ld=mold` for every Linux build; linking the ~500
  statically linked dependency crates is the slowest step of an incremental
  rebuild and mold saves ~20-60s on a cold link. Without it the build fails at
  link time with ``error: linker `cc` failed … cannot find -fuse-ld=mold``.
  - Debian/Ubuntu: `sudo apt-get install -y mold`
  - Fedora/RHEL: `sudo dnf install mold`
- **[`uv`](https://docs.astral.sh/uv/)** — only for the Python bindings and the
  docs site (manages their virtualenvs and tooling).

Optional, only for GPU builds:

- **CUDA toolkit** — for the `gpu-cuda` feature (NVIDIA).
- **macOS / CoreML** — for the `gpu-metal` feature (Apple, macOS only).

> **First build is slow.** The native ML dependencies (`ort`, `tokenizers`,
> `half`) are compiled at `opt-level = 3` even in dev profile (see
> `[profile.dev.package.*]` in `Cargo.toml`). Expect a long first `cargo build`.

---

## 2. Repository layout

Cargo workspace members (`Cargo.toml`):

| Crate | Role |
| --- | --- |
| `crates/uniko-store` | Graph storage, search, and Locy reasoning over uni-db. **The only crate allowed to touch `uni-db` directly.** |
| `crates/uniko-pipes` | Pipeline infrastructure. |
| `crates/uniko-extract` | NER, observations, chunking, ingest, embeddings. |
| `crates/uniko-memory` | Recall cascade, consolidation, rules, orchestration. |
| `crates/uniko-cortex` | Procedures and topics. |
| `crates/uniko-api` | The public facade. |
| `crates/uniko-bench` | Benchmark harness (`publish = false`). |
| `bindings/uniko-py` | Async-first PyO3 Python SDK (alpha, `publish = false`). |

`uni-db` (`^3` — the latest 3.x) and `uni-xervo` (`0.17.0`) are pulled from
crates.io — there is nothing external to install or run for them. Those are the
requirements declared in the workspace `Cargo.toml`; `Cargo.lock` pins the exact
resolved versions. Refresh uni-db within the range with
`cargo update -p uni-db`.

---

## 3. Rust workspace

### Build

```sh
git clone https://github.com/rustic-ai/uniko.git
cd uniko
cargo build
```

### The local check loop (mirrors CI exactly)

Run these before pushing — they are the same gates CI enforces, in order:

```sh
# 1. uni-db seal — only uniko-store may import uni_db / call .db()
rg -n -e 'use uni_db' -e '\.db\(\)' \
  crates/uniko-memory/src crates/uniko-extract/src \
  crates/uniko-cortex/src crates/uniko-pipes/src \
  | grep -vE ':[0-9]+:[[:space:]]*//' \
  | grep -v 'ALLOW:'

# 2. compile
#    `uniko-cuda` / `uniko-metal` are workspace members for dependency
#    inheritance only: `-cuda` downloads the ORT CUDA sidecar and `-metal`
#    compiles solely on macOS (on Linux it fails in `objc2`). CI excludes
#    both; so must you. They are covered by the release workflow's GPU jobs.
cargo check --workspace --exclude uniko-cuda --exclude uniko-metal

# 3. lint (warnings are errors)
cargo clippy --workspace --exclude uniko-cuda --exclude uniko-metal -- -D warnings

# 4. format check (use `cargo fmt --all` to auto-fix)
cargo fmt --all --check

# 5. tests
cargo nextest run --workspace --exclude uniko-cuda --exclude uniko-metal

# 6. dependency policy
cargo deny check
```

The uni-db seal should print **nothing**. Any output is a layering violation —
a crate other than `uniko-store` importing `uni_db` or calling `.db()`.

### One-time tool installs

```sh
cargo install cargo-nextest --locked   # or: cargo binstall cargo-nextest
cargo install cargo-deny --locked
```

### Running a subset of tests

```sh
cargo nextest run -p uniko-memory             # single crate
cargo nextest run -E 'test(recall_cascade)'   # filter by test name
```

### Cargo feature flags

Local inference is **off by default** — opt in with `onnx`:

| Crate | Feature | Default | Effect |
| --- | --- | --- | --- |
| `uniko-extract` | `code-parse` | **on** | Tree-sitter parsers (Python, Rust, JS, TS) for structure-aware code chunking. |
| `uniko-extract` | `onnx` | off | Pulls in `ort`, `tokenizers`, `ndarray` for the local ONNX inference path. |
| `uniko-memory` | `onnx` | off | Forwards to `uniko-extract/onnx`. |
| `uniko-memory` | `llm` | off | Enables the abstractive (LLM-rewritten) summary path. Without it, summaries stay deterministic/extractive and fully offline. |
| `uniko-store` | `gpu-cuda` | off | NVIDIA CUDA acceleration in uni-db (needs CUDA toolkit at build time). |
| `uniko-store` | `gpu-metal` | off | Apple Metal / CoreML acceleration (macOS only). |
| `uniko-store` | `batch-record` | off | Diagnostic-only batch capture for benchmarks. Never enable in production. |

---

## 4. Python bindings (`bindings/uniko-py`)

An async-first PyO3 SDK over the `Uniko` facade, built with `maturin` and managed
with `uv`. You only need this if you are changing the Python surface.

> Needs `protoc` + a C/C++ toolchain on `PATH` (the engine statically links ONNX
> Runtime). **Status: alpha — no prebuilt wheels yet; build from source.**

### Build the extension

```sh
cd bindings/uniko-py
uv run maturin develop                              # compile _uniko into the uv venv
uv run python -c "import uniko; print(uniko.__file__)"   # smoke test
```

Re-run `uv run maturin develop` whenever the Rust side of the binding changes.
(Editing `pyproject.toml` or the pure-Python `python/uniko/` overlay does not
require a rebuild, but `uv run` will rebuild automatically when needed.)

### The Python dev loop

All commands run from `bindings/uniko-py`:

```sh
uv run pytest python/tests/ -n auto   # test suite, parallel (pytest-xdist)
uv run ruff format                    # auto-format
uv run ruff check                     # lint
uv run ty check                       # type-check (library + tests)
```

Notes:

- **Tests need the extension built first** (`uv run maturin develop`). The live
  tests spin up `uniko.Uniko.in_memory()`, which downloads the default models on
  first run (see [§6](#6-models--offline-mode)).
- `uv run ty check` is scoped to `python/` (library + tests) via `[tool.ty.src]`
  in `pyproject.toml`; the native engine's types come from
  `python/uniko/_uniko.pyi`.
- The Pydantic IO layer (`uniko.models`) and the demo notebooks live under
  `python/uniko/models/` and `examples/`. Execute a notebook end-to-end with:
  ```sh
  uv run --group notebook jupyter nbconvert --to notebook --execute \
      --output /tmp/out.ipynb examples/uniko_pydantic_io_demo.ipynb
  ```

---

## 5. Documentation site (`website`)

A `uv`-managed [Zensical](https://github.com/squidfunk/zensical) static site.

```sh
cd website
uv sync
uv run zensical serve   # live preview at the printed localhost URL
uv run zensical build   # build the static site into ./site
```

---

## 6. Models & offline mode

Out of the box — BGE-small embeddings, the INT8 NLP cascade, the MiniLM
reranker, and **no** `llm` feature — uniko runs entirely on CPU with zero
external API calls (a fit for edge, air-gapped, and privacy-sensitive
deployments).

| Alias | Task | Default model |
| --- | --- | --- |
| `embed/default` | Embedding | `BAAI/bge-small-en-v1.5` |
| `nlp/default` | NLP | `dragonscale-ai/kniv-deberta-nlp-base-en-xsmall` (INT8, `onnx/cascade-int8.onnx`) |
| `rerank/default` | Rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

The default models are pulled from their repositories on first use (and
pre-warmed on `open`). A fresh machine downloads them once, then caches the
weights. Summary generation stays extractive and offline unless you opt into the
`llm` feature.

---

## 7. Conventions

**Branching** — branch off `main` with a descriptive name, e.g.
`feat/recall-decay` or `fix/ingest-ssi-conflict`.

**Commits** — Conventional-Commit-style prefixes consistent with history
(`feat:`, `fix:`, `refactor:`, `docs:`, `deps:`, `test:`), optionally scoped
(`feat(mvp): …`). Imperative subject under ~72 chars; put the *why* in the body;
one logical change per commit where practical.

- **Do not commit or push without explicit maintainer approval.**
- **Never add `Co-Authored-By` or other attribution/trailer lines** to commits.

**PRs** — keep them scoped to one logical change; CI must be green and the change
reviewed before merge.

---

## 8. CI reference

CI (`.github/workflows/ci.yml`, `ubuntu-xlarge`) installs `protobuf-compiler`
and `mold` plus a Rust stable toolchain with `clippy` + `rustfmt`, then runs the uni-db seal,
`cargo check --workspace`, `cargo clippy --workspace -- -D warnings`,
`cargo fmt --all --check`, and `cargo nextest run --workspace`. A separate job
runs `cargo deny check`. Keeping the [§3 check loop](#the-local-check-loop-mirrors-ci-exactly)
green locally is sufficient to pass CI.

---

## 9. Releasing

Publishing the crates to crates.io and the `uniko` wheels to PyPI is driven by
`.github/workflows/release.yml` — push a `v*` tag, then approve the gated
`release` environment in the Actions UI. The full process, including the
one-time Trusted Publishing setup for crates.io and PyPI, is documented in
[`RELEASING.md`](RELEASING.md).
