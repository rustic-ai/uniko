# Contributing to uniko

Thanks for your interest in contributing to **uniko**, an embedded, Rust-native
cognitive memory system for AI agents built on [uni-db](https://crates.io/crates/uni-db).

This guide covers how to set up your environment, the checks your change must
pass, and the process for getting a pull request merged. By participating you
agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## 1. Development setup

### Prerequisites

- **Rust ≥ 1.91** (edition 2024). Install via [rustup](https://rustup.rs/):

  ```sh
  rustup toolchain install stable
  ```

  The repo pins the `stable` channel in `rust-toolchain.toml`, so rustup will
  select the right toolchain automatically when you build inside the workspace.

- A recent C toolchain (for native ML dependencies such as `ort`/ONNX Runtime).

- **`mold`**, on Linux — this one is required, not a nicety. `.cargo/config.toml`
  forces it as the link backend for all Linux builds, so a machine without it
  fails at link time rather than falling back:

  ```sh
  sudo apt-get install -y mold   # Debian/Ubuntu
  sudo dnf install mold          # Fedora/RHEL
  ```

  Linking dominates incremental rebuilds here — roughly 500 dependency crates
  are statically linked into a single artifact — which is why it is mandatory
  rather than left to individual preference.

### Clone and build

```sh
git clone https://github.com/rustic-ai/uniko.git
cd uniko
cargo build
```

`cargo build` pulls **uni-db** (`^3` — the latest 3.x) and **uni-xervo**
(`0.17.0`) straight from crates.io. **No access token, private repository, or
special credentials are required** — a plain `cargo build` works out of the box.

> First builds compile inference-heavy dependencies (ONNX Runtime,
> tokenizers). The dev profile already bumps `opt-level = 3` for `ort`,
> `tokenizers`, and `half` so test runs aren't slow; the trade-off is a longer
> initial compile.

### Workspace layout

uniko is a Cargo workspace. The product crates are:

- `uniko-store` — graph storage, search, and Locy reasoning over uni-db. **The
  only crate allowed to touch uni-db directly.**
- `uniko-pipes` — pipeline infrastructure.
- `uniko-extract` — NER, observations, chunking, ingest, embeddings.
- `uniko-memory` — recall cascade, consolidation, rules, orchestration.
- `uniko-cortex` — procedures and topics.
- `uniko-api` — the public facade.

Plus `uniko-bench` (benchmark harness, `publish = false`) and
`bindings/uniko-py` (async-first PyO3 Python SDK, alpha, `publish = false`).

### Python bindings (optional)

You only need this if you are changing the Python surface. The
`bindings/uniko-py` crate is an async-first [PyO3](https://pyo3.rs) SDK over the
`Uniko` facade, built with [maturin](https://www.maturin.rs/) and managed with
[uv](https://docs.astral.sh/uv/):

```sh
cd bindings/uniko-py
uv run maturin develop   # compile the extension into the uv venv
uv run pytest            # run the Python test suite
```

`maturin` needs `protobuf-compiler` (`protoc`) and a C/C++ toolchain on `PATH`
— the stack statically links ONNX Runtime. The documentation site is likewise
uv-based: `cd website && uv sync && uv run zensical serve`.

---

## 2. The local check loop

Run these before pushing. They mirror the CI gates exactly, so passing locally
means passing in CI:

```sh
# 1. Format
cargo fmt --all --check

# 2. Lint (warnings are errors)
cargo clippy --workspace -- -D warnings

# 3. Compile the whole workspace
cargo check --workspace

# 4. Tests (see §3)
cargo nextest run --workspace

# 5. Dependency policy (licenses + advisories)
cargo deny check

# 6. uni-db seal (see §4)
```

To auto-fix formatting before committing, drop the `--check`:

```sh
cargo fmt --all
```

---

## 3. Running tests

We use [cargo-nextest](https://nexte.st/) as the test runner. Install it once:

```sh
cargo install cargo-nextest --locked
# or, faster: cargo binstall cargo-nextest
```

Run the full suite:

```sh
cargo nextest run --workspace
```

Run a single crate or filter by test name:

```sh
cargo nextest run -p uniko-memory
cargo nextest run -E 'test(recall_cascade)'
```

CI runs `cargo nextest run --workspace`, so that is the command of record. Add
or update tests alongside any behavioral change.

---

## 4. CI gates (must pass before merge)

Every pull request must pass the checks in `.github/workflows/ci.yml`. They are:

| Gate | Command |
| --- | --- |
| Format | `cargo fmt --all --check` |
| Lint | `cargo clippy --workspace -- -D warnings` |
| Compile | `cargo check --workspace` |
| Tests | `cargo nextest run --workspace` |
| Dependency policy | `cargo deny check` |
| uni-db seal | see below |

### Dependency policy (cargo-deny)

Dependencies are checked against `deny.toml` for license allow-listing and
security advisories. Install and run locally with:

```sh
cargo install cargo-deny --locked
cargo deny check
```

If you add a dependency under a license that isn't already allow-listed, the
check will fail — discuss it in your PR before extending the allow-list.

### The uni-db seal

This is uniko's most important architectural invariant. **The product crates
`uniko-memory`, `uniko-extract`, `uniko-cortex`, and `uniko-pipes` must reach
the graph only through `uniko-store`'s typed API.** They must never:

- `use uni_db` (import the database crate directly), or
- call the `.db()` escape hatch.

CI enforces this with a `ripgrep` gate over those crates' `src/` directories.
You can reproduce it locally:

```sh
rg -n -e 'use uni_db' -e '\.db\(\)' \
  crates/uniko-memory/src crates/uniko-extract/src \
  crates/uniko-cortex/src crates/uniko-pipes/src \
  | grep -vE ':[0-9]+:[[:space:]]*//' \
  | grep -v 'ALLOW:'
```

No output means the seal is intact. Notes on scope:

- **Comments are exempt**, and intentional, reviewed exceptions may be tagged
  with a `// ALLOW:` marker on the same line.
- **Tests** (`tests/` directories) and the **`uniko-bench`** crate are out of
  scope.
- `uniko-store` and `uniko-api` are not subject to the seal — `uniko-store`
  *is* the boundary, and `uniko-api` composes the product crates.

If you need a graph operation that `uniko-store` doesn't expose yet, add it to
`uniko-store` and call it from there rather than reaching past the boundary.

---

## 5. Branching and pull requests

1. Branch off `main` with a descriptive name, e.g. `feat/recall-decay` or
   `fix/ingest-ssi-conflict`.
2. Make focused commits (see §6).
3. Run the full local check loop (§2).
4. Open a pull request against `main`. Describe **what** changed and **why**,
   and link any related issue.
5. CI must be green and the change reviewed before merge.

Keep PRs scoped to one logical change where possible — smaller PRs are reviewed
faster.

---

## 6. Commit message hygiene

- Write clear, imperative subject lines: *"add temporal recall filter"*, not
  *"added stuff"*.
- Use Conventional-Commit-style prefixes consistent with the existing history
  (`feat:`, `fix:`, `refactor:`, `docs:`, `deps:`, `test:`), optionally scoped
  (`feat(mvp): …`, `refactor(arch): …`).
- Keep the subject under ~72 characters; put detail in the body.
- Explain *why* in the body when the change isn't self-evident.
- One logical change per commit where practical.

---

## 7. Code of Conduct

This project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). Please
report unacceptable behavior to **conduct@dragonscale.ai**.

---

## 8. Getting help

- General questions: **dev@dragonscale.ai**
- Security issues: **security@dragonscale.ai** (please do not open public issues
  for vulnerabilities)

uniko is licensed under **Apache-2.0**. By contributing, you agree that your
contributions are licensed under the same terms.
