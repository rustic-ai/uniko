# Python Bindings for uniko (PyO3)

**Status:** Design plan (draft — pre-implementation)
**Date:** 2026-06-18
**Scope:** A first-class, async-first Python SDK over the full `Uniko` facade, built with PyO3.
**Constraint:** Pure Rust → Python via PyO3. No C ABI, no subprocess/IPC, no HTTP. Embedded, in-process — same model as the Rust library.
**Audience:** internal eng. Not website content (yet).

---

## 0. Decisions locked in

These were settled before drafting; the rest of the document follows from them.

| Decision | Choice | Consequence |
|---|---|---|
| **Async model** | **Native async first, sync skins second** | Primary methods return Python awaitables (`asyncio`-drivable). Sync `*_sync` skins land in Phase 4 over the same impls. |
| **Surface breadth** | **Full facade** | Core loop, Goals/tasks, ingest + streaming, and the Locy logic surface (`assume`/`abduce`/rules) are all in scope. |
| **Output marshalling** | **Typed `#[pyclass]` wrappers (premium)** | Every output type gets a hand-written pyclass with getters, `__repr__`, and a `.pyi` stub. Feels like a first-class SDK (à la `polars`/`qdrant-client`), not an RPC shim. |
| **Python-authored `ModalityExtractor`** | **Deferred** (post-v1) | v1 registers *built-in* extractors only. Rationale in §6. |

---

## 1. What exists today

- **Crate:** `bindings/uniko-py/` — already a workspace member (`bindings/uniko-py` in root `Cargo.toml` members).
- **Cargo.toml:** `name = "uniko"`, `crate-type = ["cdylib"]`, deps `uniko-api` + `pyo3` (workspace `pyo3 = "0.29"`, `extension-module`). `publish = false`.
- **src/lib.rs:** a no-op `#[pymodule] fn uniko(...)` with a `TODO`. Nothing wired.
- **No** `pyproject.toml`, **no** maturin config, **no** `.pyi` stubs, **zero** Python consumers in the repo.

The public surface we bind against is `uniko-api`, which re-exports the facade from `uniko-memory` (handles), `uniko-extract` (ingest), and `uniko-store` (errors, `Value`, Locy, config). The full method inventory is catalogued in §4.

---

## 2. The two structural forces

Everything non-obvious in this design traces to two facts about the Rust facade.

### 2.1 The whole facade is `async fn` on tokio

Python has no ambient tokio runtime, and a `#[pyclass]` must be `'static`. So the binding **owns** a single multi-threaded tokio runtime and bridges every call across the sync/async boundary.

**One impl, two skins.** Each method has exactly one async body that calls the Arc-cloned `inner`. Two thin wrappers expose it:

```rust
// NATIVE ASYNC (Phase 1–3): returns a Python awaitable
fn recall<'py>(&self, py: Python<'py>, query: String) -> PyResult<Bound<'py, PyAny>> {
    let inner = self.inner.clone(); // Agent is Arc-backed, clone is cheap
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let bundle = inner.recall(&query).await.map_err(to_pyerr)?;
        Python::with_gil(|py| PyContextBundle::from_rust(py, bundle))
    })
}

// SYNC SKIN (Phase 4): same body, blocks on the shared runtime, releases the GIL
fn recall_sync(&self, py: Python<'_>, query: String) -> PyResult<Py<PyContextBundle>> {
    let inner = self.inner.clone();
    let bundle = py
        .allow_threads(|| RUNTIME.block_on(inner.recall(&query)))
        .map_err(to_pyerr)?;
    PyContextBundle::from_rust(py, bundle)
}
```

The logic is never duplicated — only the ~4-line bridge is, and a declarative macro (`bridge!`) collapses even that. `allow_threads` releases the GIL during the Rust work so other Python threads make progress; embedding/NLP/graph work is exactly the long-running kind that benefits.

**Runtime ownership.** `pyo3-async-runtimes` (the successor to `pyo3-asyncio`, version-matched to PyO3 0.29) needs one registered tokio runtime. uniko depends on `tokio` with `"full"` and spawns tasks internally, so it must be **multi-threaded** (`Runtime::new()`, not `current_thread`). The sync skins `block_on` the *same* runtime, so submitted/streaming work shares one executor.

### 2.2 Borrows cannot cross into pyclasses

Three handles borrow the Agent with a lifetime: `Data<'_>`, `Goals<'_>`, `AssumeBuilder<'_>`. A `'static` pyclass can't hold a borrow. Resolution per handle:

- **`Data` / `Goals`** → `PyData` / `PyGoals` hold an **Arc-clone of the Agent**, and rebuild `agent.data()` / `agent.goals()` *inside* each async block. The borrow lives and dies within one `await`, never inside a struct field. The ergonomic `agent.data.message(id)` / `agent.goals.active()` shape is preserved.
- **`assume`** returns a borrowing builder that forks state → mutate → query → roll back. Modelled as a **Python-side builder that accumulates strings** (the assume block, mutation clauses, the query) and does the borrow + execute in a single async block only at the terminal `.query()`.

This is sound precisely because the report confirms the handles are Arc-backed and cheap to clone — rebuilding per call costs nothing.

---

## 3. Type marshalling (the "premium" path)

### 3.1 Output types → typed pyclasses

Every type that crosses the boundary outward gets a `#[pyclass]` with `#[pyo3(get)]` fields (or getter methods for computed/nested ones), a `__repr__`, and a `.pyi` entry. We do **not** lean on `Serialize`/`pythonize`, because the derive coverage is split ~50/50 and dict-based output is precisely the RPC-shim feel we rejected.

| Has `#[derive(Serialize)]` | Missing `Serialize` |
|---|---|
| `ContextBundle`, `RecallItem`, `RecallSource`, `RecallKind`, `RecallTier`, `GoalView`, `TaskView`, `GoalContext`, `DeletionReport` | `Answer`, `GeneratedAnswer`, `MessageView`, `ArtifactView`, `ObserveResult`, `IngestOutcome`, `AtomicIngestResult`, `AbductionResult`, `DerivationTree` |

Each wrapper implements a `from_rust(py, value) -> Py<Self>` constructor that walks the Rust struct field-by-field. This is mechanical but it is the bulk of the binding code; it is budgeted explicitly in the phases. (`Serialize` on the right column is *not* required for this approach, so we don't add derives purely for the binding.)

### 3.2 `Value` ↔ Python — mandatory bidirectional converter

`Value` (re-exported from uni-db via `uniko-store`) is the graph value type used for **query params (Python → `Value`)** and **`Record` rows (`Value` → Python)**. It must round-trip both ways; `pythonize` cannot do the input direction cleanly, and several variants need deliberate mapping:

| `Value` variant | Python type |
|---|---|
| `Null` | `None` |
| `Bool(bool)` | `bool` |
| `Int(i64)` | `int` |
| `String(String)` | `str` |
| `List(Vec<Value>)` | `list` |
| `Map(HashMap<String, Value>)` | `dict[str, Any]` |
| `Temporal(DateTime)` | `datetime.datetime` (tz-aware, UTC) |
| `Bytes(Vec<u8>)` | `bytes` |
| `Vector(Vec<f32>)` | `list[float]` |

`Record` (`HashMap<String, Value>`) → `dict[str, Any]`. This converter is foundational and lands in Phase 0; `query`/`query_in`/`run_rule`/`abduce` all depend on it.

> **Note on `Value::Bytes` via Cypher.** A known uni-db limitation: `DataType::Bytes` can't be read back through a Cypher `RETURN` (arrow `LargeBinary` type-ambiguity — see `bugs/` repro in `uniko-store`). The converter handles the `Bytes` variant for completeness, but binary payloads are retrieved through `data.artifact_bytes(id)` (blob store), not through `query()`. Documented as a binding caveat, not worked around.

### 3.3 Errors → Python exception hierarchy

`UnikoError` (variants: `Storage`, `Search`, `Schema`, `Pipeline`, `Locy`, `Config`, `Embedding`, `Llm`, `Timeout`, `Conflict`, `Internal`, `Unsupported`) maps to:

```
UnikoError(Exception)            # base — every binding error is catchable here
├── ConfigError
├── LlmError
├── TimeoutError
├── ConflictError                # SSI / antidependency aborts surface here
├── UnsupportedError
└── (Storage/Search/Schema/Pipeline/Locy/Embedding/Internal → base UnikoError with .kind)
```

We subclass only the variants a caller realistically branches on; the rest fold into the base carrying a `.kind` string and the Rust `Display` message. A single `to_pyerr(UnikoError) -> PyErr` helper is the one conversion site.

---

## 4. API surface inventory (full facade)

Mapped 1:1 unless a lifetime forces the flattening from §2.2. Async unless marked sync.

### `uniko.Uniko` (handle)
- `Uniko.open(path)` *(async, classmethod)* · `Uniko.in_memory()` *(async, classmethod)* · `Uniko.builder()` *(sync → `UnikoBuilder`)*
- `agent(agent_id) -> Agent` *(sync)* · `config() -> UnikoConfig` *(sync)*
- `purge()` *(async)* · `shutdown()` *(async)*

### `uniko.UnikoBuilder` (builder, sync setters + async `build`)
- `path` · `in_memory` · `embedding(EmbeddingConfig)` · `llm(LlmSpec)` · `raw_config(UnikoConfig)` · `streaming(bool)` · `scope_to_agent()` · `scope(RecallScope)` · `extractor(...)` *(built-in only — §6)* · `build()` *(async)*

### `uniko.LlmSpec` (sync constructors)
- `LlmSpec.openai(alias, model_id, base_url=None)` · `openai_with_key_env(...)` · `mistralrs(alias, model_id)`

### `uniko.Agent` (handle)
- `agent_id` *(property)*
- `recall(query)` · `recall_in(query, scope)` · `answer(question)` · `answer_in(question, scope)`
- `query(cypher)` · `query_in(cypher, scope)`
- `define_rule(name, source)` · `run_rule(name, return_cols, params)`
- `assume(block) -> AssumeBuilder` *(sync; terminal `.query()` async)* · `abduce(program, params)`
- `delete_session(session_id)` · `finalize_session(session_id)` · `unfinalized_session_ids()` · `forget_participant(participant_id)`
- `session(session_id) -> Session` *(sync)*
- `data -> Data` *(sync property; §2.2 flatten)* · `goals -> Goals` *(sync property; §2.2 flatten)*

### `uniko.Session` (handle)
- `session_id` *(property)*
- `observe(turn)` · `submit(turn)` · `submit_source(source)` · `flush()` · `ingest(source)` · `finalize()` · `summarize()`
- `forget_turn(id)` · `delete_turn(id)` · `delete_document(id)`

### `uniko.Turn` (sync builder)
- `Turn(sender_id, content)` then `.id` · `.content_type` · `.at(datetime)` · `.addressed_to(list)` · `.metadata(key, value)` · `.attach(source)` · `.attachments(sources)`

### `uniko.Data`
- `message(id) -> MessageView | None` · `artifact(id) -> ArtifactView | None` · `artifact_bytes(id) -> bytes | None`

### `uniko.Goals`
- reads: `all()` · `in_phase(p)` · `active()` · `planned()` · `completed()` · `get(id)` · `tasks()` · `tasks_in(p)` · `tasks_of(goal_id)` · `context(goal_id)`
- transitions: `start/complete/abandon(goal_id)` · `start_task/complete_task/block_task/unblock_task(task_id)`

### Inputs / value objects (sync constructors)
- `IngestSource` — `from_path`, `from_bytes(data, mime)`, `from_text(...)` constructors
- `Scope`, `RecallScope`, `Dimensions`, `Viewer`, `GoalPhase`, `TaskPhase` enums/builders
- `EmbeddingConfig`, `UnikoConfig` (presets exposed as classmethods)

### Output pyclasses (typed, §3.1)
`ContextBundle`, `RecallItem`, `RecallSource`, `Answer`, `ObserveResult`, `IngestOutcome`, `AtomicIngestResult`, `MessageView`, `ArtifactView`, `GoalView`, `TaskView`, `GoalContext`, `DeletionReport`, `AbductionResult`, `DerivationTree`, `Record`(→dict)

---

## 5. Phased implementation plan

Each phase ends green: `maturin develop` builds, the module imports, and the phase's surface is exercised by a `pytest`.

- **Phase 0 — Foundation.** Global multi-thread tokio runtime + `pyo3-async-runtimes` wiring; the `bridge!` macro (async skin now, sync skin slot reserved); `to_pyerr` + exception hierarchy (§3.3); `Value` ↔ Python + `Record` → dict (§3.2); `pyproject.toml` + maturin (`abi3-py39` → one wheel per platform across CPython ≥3.9); first importable `#[pymodule]`.
- **Phase 1 — Core loop.** `Uniko`, `UnikoBuilder`, `LlmSpec`, `Agent` (recall/answer/query families), `Session.observe/summarize/forget+delete`, `Turn`. Output pyclasses: `ContextBundle`, `RecallItem`, `RecallSource`, `Answer`, `ObserveResult`.
- **Phase 2 — Data, Goals, ingest.** `PyData` + `PyGoals` (full surface, §2.2); `IngestSource` constructors + `Session.ingest`; streaming `submit`/`submit_source`/`flush`; `delete_session`/`forget_participant`. Output pyclasses: `MessageView`, `ArtifactView`, `GoalView`, `TaskView`, `GoalContext`, `IngestOutcome`, `DeletionReport`.
- **Phase 3 — Logic surface.** `define_rule`/`run_rule`; `assume` builder; `abduce` + `AbductionResult`/`DerivationTree`; `Scope`/`RecallScope`/`Dimensions`/`Viewer` construction.
- **Phase 4 — Sync skins + polish.** `*_sync` variants across the surface via the reserved `bridge!` slot; complete `.pyi` stubs; docstrings; example scripts; wheel-build CI (maturin-action). Python-authored extractors remain deferred.

---

## 6. Deferred: Python-authored `ModalityExtractor`

`ModalityExtractor` is an `#[async_trait]` with `fn modality(&self) -> Modality` (trivial) and `async fn extract(&self, kb: &KnowledgeBase, src: &IngestSource) -> Result<ArtifactIngestResult, UnikoError>` (hard). A Python implementation would mean a Rust shim holding a `Py<PyAny>` that calls back into Python from inside a tokio task that is *mid graph-write*, against a borrowed live `KnowledgeBase` — GIL-across-`await` reentrancy with a borrowed KB. That is a real feature with real deadlock surface, not a wrapper.

**v1:** `builder.extractor(...)` registers *built-in* extractors only (selected by `Modality`/name). Python-authored extractors are a deliberate post-v1 item, flagged here so the gap is a decision and not a silent omission.

---

## 7. Packaging & distribution

- **Build backend:** `maturin`. `pyproject.toml` with `[build-system] requires = ["maturin>=1.5,<2"]`, `build-backend = "maturin"`, and `[tool.maturin]` pointing at the cdylib.
- **ABI:** `abi3-py39` — a single wheel per platform spanning CPython ≥3.9, instead of one per minor version. Compatible with `pyo3-async-runtimes`.
- **Module name:** `import uniko`.
- **Typing:** ship `uniko.pyi` (or `py.typed` + stubs) so the typed pyclasses and awaitable signatures land in editors/mypy.
- **CI:** `maturin-action` builds linux/macos/windows wheels; not auto-published to PyPI until the facade is declared stable (README currently says bindings are "not yet available").
- **Heavy deps:** the wheel statically links the uniko stack (ONNX via `ort`, tokenizers, tree-sitter). Wheel size and the `ort` runtime/`LD_LIBRARY_PATH` story are a packaging risk to validate early (Phase 0 smoke build on all three OSes).

---

## 8. Open risks

1. **`ort` / ONNX runtime in a Python wheel** — native lib loading across platforms is the likeliest packaging failure. Validate in Phase 0.
2. **`shutdown()` consumes `self` and errors if handles are held.** Python has no move semantics; `Uniko.shutdown()` must invalidate the handle (poison-on-use) and document that outstanding `Agent`/`Session` objects must be dropped first.
3. **Runtime lifecycle vs. interpreter shutdown** — the global tokio runtime must not outlive `Py_Finalize`. Register an `atexit`/module-free hook to drain.
4. **`ConflictError` surfacing** — uni-db SSI antidependency aborts (seen on concurrent same-session ingest) reach Python as `ConflictError`; examples should show the retry pattern rather than hiding it.
