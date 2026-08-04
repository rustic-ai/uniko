# Python SDK

## The same engine, from Python — no server, no IPC

The `uniko` Python package is a PyO3 binding over the exact same Rust engine that powers the
core library. It links into your Python process like SQLite: one in-process handle, no sidecar
service, no network hop, no serialization tax on the hot path. You get $0-cost local ingest,
graph-native reasoning, and provenance-backed recall — all from `import uniko`.

The SDK is **async-first**. Every facade verb returns an `asyncio` awaitable and runs the real
Rust work on a shared tokio runtime. When you need to call from synchronous code, every verb
also ships a `*_sync` twin that blocks the calling thread and releases the GIL across the Rust
work — so a blocking call never starves the rest of your interpreter.

<div class="cta-row" markdown>
[Quickstart](quickstart.md){ .cta-primary }
[Python API reference](api.md){ .cta-secondary }
</div>

---

## When to reach for the Python SDK

Reach for it when your agent lives in Python and you want cognitive memory without standing up
infrastructure:

- You're building on a Python agent framework and want recall, facts, goals, and reasoning in the
  same process — not behind an HTTP boundary.
- You want the in-process, embedded model (no Neo4j, no Qdrant, no Postgres) with a Pythonic
  async surface.
- You need provenance: every recalled item traces back to the message that grounds it, and you
  want those handles in Python.

If you're writing Rust, use the [core library](../getting-started/installation.md) directly — the
Python SDK is a thin skin over the same types.

!!! note "The Python SDK does not re-explain the memory model"
    Entities, Facts, Procedures, Topics, and Episodes — and how they derive from Messages with
    full provenance — are the same here as in the core engine. This page documents the Python
    surface; for the model itself, read the Rust [Concepts](../concepts/index.md) and
    [Pipelines](../pipelines/index.md) pages. They apply verbatim.

---

## Package and module layout

The distribution is named `uniko` (import `uniko`). The native extension is an `abi3` cdylib at
`uniko._uniko`; the package re-exports its public surface, so you import everything straight from
the top-level package:

```python
import uniko

uni = await uniko.Uniko.in_memory()
agent = uni.agent("assistant")
```

- **`uniko`** — the package you import. Re-exports the full public API (30 names).
- **`uniko._uniko`** — the compiled native extension. You don't import this directly.
- Ships **`py.typed`** and a hand-written **type stub** (`_uniko.pyi`), so editors and type
  checkers see the full signature surface, including the `*_sync` twins. Requires **Python ≥ 3.10**.

The public surface is one `Uniko` handle from which you mint `Agent`s, open `Session`s, observe
`Turn`s, and read memory back through `Data` and `Goals`. Builders (`Turn`, `IngestSource`,
`Scope`, `LlmSpec`, `UnikoBuilder`, `AssumeBuilder`) construct inputs; frozen view types
(`ContextBundle`, `Answer`, `MessageView`, `GoalView`, …) carry results back.

---

## Alpha status — read this first

The Python SDK is in active development. Here's the honest state of the world:

!!! warning "Shipped now, and what isn't"
    **Shipped:** the full async facade, the synchronous `*_sync` twins (GIL released across Rust
    work), `py.typed` + a complete type stub, and the structured exception hierarchy.

    **Also shipped:** prebuilt wheels on PyPI (CPython 3.10+, abi3) across three distributions,
    and a public Pydantic overlay at `uniko.models`.

Install one of the three distributions — they all provide the `uniko` import and cannot coexist:

```bash
pip install uniko          # CPU
pip install uniko-cuda     # NVIDIA CUDA, Linux x86_64
pip install uniko-metal    # Apple Silicon, macOS arm64
```

`uniko.__version__` mirrors the native `_uniko.__version__`, which is single-sourced from the
crate version (currently `0.2.0`) rather than hardcoded.

A handful of design-doc surfaces did not ship in this alpha — notably a `UnikoConfig` object
(`config()` returns a plain `dict`), Python-authored extractors, and visibility scoping on
`Scope`. The [API reference](api.md) documents only what actually ships.

---

## A minimal example

The core loop is three calls: open an instance, observe a turn, recall a context bundle. Pick
your skin — the async surface and the synchronous `*_sync` twins are line-for-line equivalent.

=== "Async"

    ```python
    import asyncio
    import uniko


    async def main() -> None:
        uni = await uniko.Uniko.in_memory()
        agent = uni.agent("assistant")
        session = agent.session("conversation-1")

        # Observe a message. Local ONNX extraction runs, then one atomic
        # transaction commits the Message, Entities, Observations, and edges
        # — durable and read-after-write before this returns.
        result = await session.observe(
            uniko.Turn("alice", "I love hiking in the mountains.")
        )
        assert result.message_node_id > 0

        # Recall compiled knowledge. No LLM in this path; each item carries
        # its kind, score, and the sources it traces back to.
        bundle = await agent.recall("hiking")
        for item in bundle.items:
            print(f"[{item.kind}] {item.content}")


    asyncio.run(main())
    ```

=== "Sync"

    ```python
    import uniko

    uni = uniko.Uniko.in_memory_sync()
    agent = uni.agent("assistant")
    session = agent.session("conversation-1")

    result = session.observe_sync(
        uniko.Turn("alice", "I love hiking in the mountains.")
    )
    assert result.message_node_id > 0

    bundle = agent.recall_sync("hiking")
    for item in bundle.items:
        print(f"[{item.kind}] {item.content}")

    # Shut down cleanly: drop derived handles first, then shut down the store.
    del session, agent
    uni.shutdown_sync()
    ```

!!! tip "Construction helpers stay synchronous"
    Only the verbs that do real engine work have `_sync` twins. `agent(...)`, `session(...)`,
    builder setters, and the value-object constructors (`Turn`, `Scope`, `IngestSource`,
    `LlmSpec`) are plain synchronous calls in both skins.

---

## Errors are structured

Every failure raises a `uniko.UnikoError` (or a subclass), and every instance carries a `.kind`
string you can branch on. Subclasses cover the cases you'll catch by type — `ConfigError`,
`LlmError`, `TimeoutError`, `ConflictError` (a transient, retriable SSI abort), and
`UnsupportedError`. Storage-domain failures fold into the base `UnikoError` with distinct kinds
(`"storage"`, `"search"`, `"schema"`, `"pipeline"`, `"locy"`, `"embedding"`, `"internal"`), so
branch on `e.kind` for those.

```python
try:
    answer = await agent.answer("Where does Alice like to hike?")
except uniko.ConfigError as e:
    # No LLM configured — answer()/answer_in() need an LlmSpec on the builder.
    print(f"configure an LLM first ({e.kind})")
```

---

## What you can do

<div class="feature-grid" markdown>
<div class="feature-card" markdown>
### Observe and recall
Feed turns with `session.observe(...)`, pull a `ContextBundle` with `agent.recall(...)`, and get
grounded answers from `agent.answer(...)` once an LLM is configured. Every recalled item carries
its `sources`.
</div>
<div class="feature-card" markdown>
### Goals and tasks
`agent.goals` is a full lifecycle surface — create goals and tasks, drive them through
`planned → active → completed`, and pull a `GoalContext` that gathers the goal, its tasks,
sessions, facts, and entities.
</div>
<div class="feature-card" markdown>
### Reasoning in the database
Define and run Locy rules, run `abduce(...)` for hypothesis generation, and use `agent.assume(...)`
to test hypothetical mutations that roll back — the real graph is never touched.
</div>
<div class="feature-card" markdown>
### By-id retrieval and Cypher
`agent.data` resolves messages, artifacts, and raw binary payloads by id (unknown id → `None`,
never raises). `agent.query(...)` runs read-only Cypher and rejects mutations.
</div>
</div>

---

## Next steps

<div class="feature-grid" markdown>
<div class="feature-card" markdown>
### [Quickstart](quickstart.md)
Build the SDK, open an instance, and run the observe → recall loop end to end.
</div>
<div class="feature-card" markdown>
### [Python API reference](api.md)
Every handle, builder, and view type — with the exact async and `*_sync` signatures.
</div>
<div class="feature-card" markdown>
### [Concepts](../concepts/index.md)
The five kinds of memory, the data model, and how facts carry bitemporal provenance.
</div>
<div class="feature-card" markdown>
### [Pipelines](../pipelines/index.md)
What actually happens inside `observe` and `recall` — the ingest, consolidation, and recall paths.
</div>
</div>
