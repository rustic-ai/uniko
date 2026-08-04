# Python Quickstart

The `uniko` Python SDK is the same in-process cognitive memory engine you get in Rust, wrapped in an
idiomatic `asyncio` package. You observe a conversation, recall grounded context, answer questions,
run goals and tasks, and reason hypothetically — all from Python, with the heavy lifting (storage,
vector search, local NLP, the recall cascade) running natively in-process.

Every facade verb ships in two skins: the bare name returns an awaitable for `asyncio`, and the
`*_sync` twin blocks the calling thread on the shared runtime and returns the resolved value. The
sync twins release the GIL across the native work, so you can call them from threads without
serializing your whole program. The examples below show both, side by side.

## Install

Prebuilt wheels ship on PyPI for CPython 3.10+ (abi3). Install **one** of the three distributions —
they all provide the same `uniko` import and cannot coexist:

```bash
pip install uniko          # CPU: Linux x86_64/aarch64, macOS arm64, Windows x64
pip install uniko-cuda     # NVIDIA CUDA, Linux x86_64
pip install uniko-metal    # Apple Silicon, macOS arm64
```

The package is `uniko` (importable as `uniko`, native extension at `uniko._uniko`). The ONNX
runtime is linked statically into the wheel, so there is nothing else to install for the local
inference path.

!!! tip "If a GPU wheel is too large for PyPI"
    `uniko-cuda` and `uniko-metal` bundle GPU runtimes and may exceed PyPI's per-file limit. The
    project publishes a PEP 503 index serving the same artifacts from GitHub Release assets:

    ```bash
    pip install uniko-cuda --extra-index-url https://rustic-ai.github.io/uniko/packages/
    ```

??? note "Building from source instead"
    Contributors, or anyone on an unsupported platform, can build the extension with
    [maturin](https://www.maturin.rs/):

    ```bash
    cd bindings/uniko-py
    uv run maturin develop --release
    ```

    This needs a working **C/C++ toolchain** and **`protoc`** on your `PATH` (plus `mold` on
    Linux). `maturin develop` installs into the currently-activated environment.

!!! note "Ships typed"
    The package includes `py.typed` and a hand-written `_uniko.pyi` stub, so your editor and type
    checker see full signatures for every handle, builder, and value object.

## A guided tour

The walk-through below is one continuous program. Each step shows the async and sync forms; pick the
style that fits your application.

### 1. Open an instance, mint an agent and a session

A `Uniko` instance owns the whole memory system. From it you mint an `Agent` (an identity), and from
the agent a `Session` (a conversation thread). `in_memory()` is ephemeral — perfect for tests and
notebooks; `open(path)` gives you a durable, file-backed store.

=== "Async"

    ```python
    import uniko

    uni = await uniko.Uniko.in_memory()      # or: await uniko.Uniko.open("./memory.db")
    agent = uni.agent("assistant")
    session = agent.session("conversation-1")
    ```

=== "Sync"

    ```python
    import uniko

    uni = uniko.Uniko.in_memory_sync()       # or: uniko.Uniko.open_sync("./memory.db")
    agent = uni.agent("assistant")
    session = agent.session("conversation-1")
    ```

`agent(...)` and `session(...)` are synchronous constructors — no `await`, no `_sync` twin. For
tuned setups (an LLM for `answer()`, streaming ingest, agent-scoped recall) reach for
`uniko.Uniko.builder()` instead; see the [`UnikoBuilder` reference](api.md#unikobuilder).

### 2. Observe a turn

`observe` runs the full ingest pipeline — chunking, entity extraction, observation extraction — and
**commits before returning**, so the very next read sees it (read-after-write). A `Turn` is a
plain constructor, `Turn(sender_id, content)`, reusable and chainable.

=== "Async"

    ```python
    result = await session.observe(
        uniko.Turn("alice", "I love hiking in the mountains.")
    )
    assert result.message_node_id > 0
    ```

=== "Sync"

    ```python
    result = session.observe_sync(
        uniko.Turn("alice", "I love hiking in the mountains.")
    )
    assert result.message_node_id > 0
    ```

The returned `ObserveResult` tells you exactly what landed: `message_node_id`, `chunk_node_ids`,
`session_node_id`, `extracted_entities` (a list of `(node_id, entity_name)` tuples),
`extracted_observations`, and `attachment_count`.

!!! note "Sessions are serialized"
    `observe` mutates the session in order — feed turns sequentially, not concurrently, within one
    `Session`.

A `Turn` carries more than text. Chain `.id(message_id)` for idempotency, `.at(datetime)`,
`.addressed_to([...])`, `.content_type("...")`, `.metadata(key, value)`, and
`.attach(IngestSource.from_path("handbook.md"))` to ride a document along with the message.

### 3. Recall a context bundle

`recall` returns a `ContextBundle` — the ranked, deduplicated context for a query, with no LLM
involved. Iterate `.items`; each `RecallItem` carries its `kind`, `score`, `content`, and the
`sources` it traces back to.

=== "Async"

    ```python
    bundle = await agent.recall("hiking")
    for item in bundle.items:
        print(item.kind, item.score, item.content)
        for src in item.sources:
            print("   from", src.kind, src.message_id)
    print(f"{len(bundle)} items, {bundle.total_tokens} tokens, coverage {bundle.coverage}")
    ```

=== "Sync"

    ```python
    bundle = agent.recall_sync("hiking")
    for item in bundle.items:
        print(item.kind, item.score, item.content)
        for src in item.sources:
            print("   from", src.kind, src.message_id)
    print(f"{len(bundle)} items, {bundle.total_tokens} tokens, coverage {bundle.coverage}")
    ```

`ContextBundle` supports `len(bundle)` and exposes `phase1_only`, `phase2_only`, and `coverage`
alongside the items — everything you need to format your own prompt or audit what grounded a result.

### 4. Answer a question

`answer` runs the recall cascade, hands the ranked context to the configured LLM, and returns an
`Answer` with the synthesized text plus the context that grounded it. **This requires an LLM** — if
none is configured, it raises `ConfigError`.

=== "Async"

    ```python
    answer = await agent.answer("What does Alice love?")
    print(answer.text)
    print("model:", answer.model, "tokens out:", answer.output_tokens)
    for source in answer.citations():
        print("  cited:", source.kind, source.message_id)
    ```

=== "Sync"

    ```python
    answer = agent.answer_sync("What does Alice love?")
    print(answer.text)
    print("model:", answer.model, "tokens out:", answer.output_tokens)
    for source in answer.citations():
        print("  cited:", source.kind, source.message_id)
    ```

`answer.context` is the same `ContextBundle` you'd get from `recall`, and `answer.citations()`
lists the `RecallSource`s the reply drew on. To configure a model, build with an `LlmSpec`:

```python
spec = uniko.LlmSpec.openai("llm/default", "gpt-4o-mini")   # key from OPENAI_API_KEY
uni = await uniko.Uniko.builder().in_memory().llm(spec).build()
```

`LlmSpec` also offers `openai_with_key_env(alias, model_id, key_env)` and `mistralrs(alias,
model_id)` for a local, in-situ quantized model.

!!! warning "No LLM, no answer"
    ```python
    with pytest.raises(uniko.ConfigError) as info:
        await agent.answer("anything")
    assert info.value.kind == "config"
    ```
    `recall` never needs an LLM — only `answer`/`answer_in` do.

### 5. Run a raw Cypher query

`query` runs read-only Cypher and returns plain `list[dict]` rows. Mutations are rejected.

=== "Async"

    ```python
    rows = await agent.query("MATCH (n) RETURN n LIMIT 3")
    for row in rows:
        print(row)
    ```

=== "Sync"

    ```python
    rows = agent.query_sync("MATCH (n) RETURN n LIMIT 3")
    for row in rows:
        print(row)
    ```

### 6. Goals and tasks

Every agent has a `goals` surface for structured planning. Reads are scoped to the agent's owned
goals and assigned tasks; transitions return `bool` (`False` when the id doesn't resolve); creation
methods return the new node id.

!!! note "The agent must exist first"
    Goals attach to the agent's `Participant`, which is created on first sight. Send one turn before
    creating goals.

=== "Async"

    ```python
    await agent.session("setup").observe(uniko.Turn("assistant", "kicking off"))

    goals = agent.goals
    gid = await goals.create(
        "Ship the Python SDK",
        goal_id="g1",
        description="PyO3 bindings",
        metrics={"target_phase": 3},
    )
    await goals.start("g1")

    tid = await goals.create_task("write tests", goal_id="g1", task_id="t1", priority=0.8)

    active = await goals.active()
    tasks = await goals.tasks_of("g1")
    ctx = await goals.context("g1")

    await goals.complete("g1", {"done": True})
    ```

=== "Sync"

    ```python
    agent.session("setup").observe_sync(uniko.Turn("assistant", "kicking off"))

    goals = agent.goals
    gid = goals.create_sync(
        "Ship the Python SDK",
        goal_id="g1",
        description="PyO3 bindings",
        metrics={"target_phase": 3},
    )
    goals.start_sync("g1")

    tid = goals.create_task_sync("write tests", goal_id="g1", task_id="t1", priority=0.8)

    active = goals.active_sync()
    tasks = goals.tasks_of_sync("g1")
    ctx = goals.context_sync("g1")

    goals.complete_sync("g1", {"done": True})
    ```

Reads include `all()`, `active()`, `planned()`, `completed()`, `in_phase(phase)`, `get(goal_id)`,
`tasks()`, `tasks_in(phase)`, `tasks_of(goal_id)`, and `context(goal_id)`. Goal phases are the plain
strings `planned`/`active`/`completed`/`abandoned`; task phases are
`planned`/`active`/`completed`/`blocked`. Task transitions are `start_task`, `block_task`,
`complete_task`, and `set_task_status` — to leave the blocked state, call `start_task`,
`complete_task`, or `set_task_status`.

!!! tip "Invalid phases raise early"
    Passing an unknown phase to `in_phase` or `tasks_in` raises `ValueError` before any query runs.

### 7. Reason hypothetically with `assume`

`agent.assume(...)` opens a hypothetical transaction: you stage mutations, query against them, and
the changes **roll back** when the builder runs — the real graph is never touched. Build it with
`.then_query(...)`, optionally `.param(key, value)`, then terminate with `run()`/`run_sync()`.

=== "Async"

    ```python
    rows = await (
        agent.assume(
            "ASSUME { CREATE (:Fact {fact_id: 'srv-port', subject: 'srv', "
            "predicate: 'port', object: '9090'}) }"
        )
        .then_query("MATCH (f:Fact {subject: 'srv'}) RETURN f")
        .run()
    )
    assert len(rows) == 1

    # The hypothetical mutation never hit the real graph:
    after = await agent.query("MATCH (f:Fact {subject: 'srv'}) RETURN f")
    assert after == []
    ```

=== "Sync"

    ```python
    rows = (
        agent.assume(
            "ASSUME { CREATE (:Fact {fact_id: 'srv-port', subject: 'srv', "
            "predicate: 'port', object: '9090'}) }"
        )
        .then_query("MATCH (f:Fact {subject: 'srv'}) RETURN f")
        .run_sync()
    )
    assert len(rows) == 1

    after = agent.query_sync("MATCH (f:Fact {subject: 'srv'}) RETURN f")
    assert after == []
    ```

For the inverse — abductive reasoning that proposes what *would* make a query hold — define a rule
and call `abduce`:

=== "Async"

    ```python
    await agent.define_rule(
        "reachable", "CREATE RULE reachable AS MATCH (a:Episode) YIELD KEY a"
    )
    result = await agent.abduce("ABDUCE reachable WHERE a.kind = 'query'")
    for mod in result.modifications:
        print(mod["modification"], mod["validated"], mod["cost"])
    ```

=== "Sync"

    ```python
    agent.define_rule_sync(
        "reachable", "CREATE RULE reachable AS MATCH (a:Episode) YIELD KEY a"
    )
    result = agent.abduce_sync("ABDUCE reachable WHERE a.kind = 'query'")
    for mod in result.modifications:
        print(mod["modification"], mod["validated"], mod["cost"])
    ```

`AbductionResult` supports `len(result)`; each entry in `result.modifications` is a dict with keys
`modification`, `validated`, and `cost`.

### 8. Shut down cleanly

`shutdown` drains the workers and **consumes the handle**. Because the store is reference-counted,
you must drop every derived `Agent`/`Session` first — otherwise the guard raises `ConfigError`. Use
`purge()` to wipe the whole graph for a dev/test reset.

=== "Async"

    ```python
    await uni.purge()        # erase everything (dev/test reset)
    del session, agent       # MUST drop derived handles first
    await uni.shutdown()     # else raises ConfigError
    ```

=== "Sync"

    ```python
    uni.purge_sync()
    del session, agent       # MUST drop derived handles first
    uni.shutdown_sync()      # else raises ConfigError
    ```

Any use of a `Uniko` handle after `shutdown` raises `ConfigError`.

## Handling conflicts

`uniko` raises a small, predictable exception hierarchy. The base is `UnikoError`, and **every**
instance carries a `.kind` string. Subclasses include `ConfigError` (`"config"`), `LlmError`
(`"llm"`), `TimeoutError` (`"timeout"`), `UnsupportedError` (`"unsupported"`, no extractor for the
modality), and `ConflictError` (`"conflict"`). Storage-domain failures fold into the base
`UnikoError` but still carry distinct `.kind` values (`"storage"`, `"search"`, `"schema"`,
`"pipeline"`, `"locy"`, `"embedding"`, `"internal"`) — branch on `e.kind` for those.

`ConflictError` is a **transient** serialization abort (SSI / antidependency) and is safe to retry:

```python
import asyncio
import uniko

async def observe_with_retry(session, turn, attempts=5):
    for i in range(attempts):
        try:
            return await session.observe(turn)
        except uniko.ConflictError:
            if i == attempts - 1:
                raise
            await asyncio.sleep(0.05 * (2 ** i))   # back off and try again
```

!!! tip "Branch on `.kind`, not just type"
    Because storage-domain errors all surface as `UnikoError`, the robust pattern is
    `except uniko.UnikoError as e: ...` then `match e.kind:`. The named subclasses
    (`ConfigError`, `ConflictError`, …) are there when you want to catch one specifically.

## Next steps

<div class="feature-grid" markdown>
<div class="feature-card" markdown>
### [Python API Reference](api.md)
The full facade surface — every handle, builder, value object, and the config keys behind `config()`.
</div>
<div class="feature-card" markdown>
### [Recall & Retrieval](../guides/retrieval.md)
How `recall`/`answer` rank context, what each `RecallItem` `kind` means, and tracing `sources`.
</div>
<div class="feature-card" markdown>
### [Reasoning with Locy](../guides/reasoning-with-locy.md)
The logic behind `assume`, `abduce`, `define_rule`, and `run_rule`.
</div>
<div class="feature-card" markdown>
### [Memory Model](../concepts/memory-model.md)
The node types — Message, Observation, Fact, Entity, Episode — and how memory is organized.
</div>
</div>
