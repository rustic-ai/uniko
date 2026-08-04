# Python SDK Reference

The `uniko` Python package is a thin, typed binding over the same Rust engine the
[facade](../guides/facade.md) exposes. You program against the same verbs — *observe*, *recall*,
*answer*, *forget* — from one owning `Uniko` handle, with native `asyncio` support and a blocking
twin for every call. Ships `py.typed` and a hand-written stub, so your editor and type checker see
the full surface.

```python
import uniko

uni = await uniko.Uniko.in_memory()
agent = uni.agent("assistant")
session = agent.session("conversation-1")

await session.observe(uniko.Turn("alice", "I love hiking in the mountains."))
bundle = await agent.recall("hiking")
for item in bundle.items:
    print(item.score, item.content)
```

!!! note "Install"
    Requires Python ≥ 3.10. From the binding crate, `maturin develop` builds the native extension
    (`uniko._uniko`). A C/C++ toolchain and `protoc` are needed at build time; ONNX is statically
    linked. See [Installation](../getting-started/installation.md).

---

## The async / sync contract

Every facade verb ships **two skins**:

- the **bare name** (`recall`, `observe`, `answer`, …) returns an awaitable you `await` inside an
  event loop;
- the **`*_sync` twin** (`recall_sync`, `observe_sync`, …) blocks the calling thread on the shared
  tokio runtime and returns the resolved value directly. `*_sync` releases the GIL across the Rust
  work, so other Python threads keep running.

=== "Async"

    ```python
    uni = await uniko.Uniko.in_memory()
    bundle = await agent.recall("rockets")
    ```

=== "Sync"

    ```python
    uni = uniko.Uniko.in_memory_sync()
    bundle = agent.recall_sync("rockets")
    ```

**Construction helpers and builder setters are synchronous** — there is no `_sync` twin for
`agent()`, `session()`, `Turn(...)`, `Scope().sessions(...)`, or any `UnikoBuilder` setter. Only the
async I/O verbs (and terminal `build`) get the twin. Throughout this page, tables mark which methods
have a `_sync` twin in the **Sync** column; assume the async signature shown and append `_sync` to
the name for the blocking form.

!!! tip "`__version__`"
    `uniko.__version__` tracks the crate version (currently `0.2.0`), single-sourced from Cargo
    via `env!("CARGO_PKG_VERSION")` rather than hardcoded. It falls back to the deliberately
    invalid sentinel `"0+unknown"` if the native module fails to expose it.

---

## `Uniko` — the instance

`Uniko` owns the store, the model runtime, and the ingest pipeline. Build one, mint agents from it,
and `shutdown` when done. `repr(uni)` is `Uniko(open)` or `Uniko(shut down)`.

=== "Zero-config"

    ```python
    uni = await uniko.Uniko.in_memory()             # ephemeral
    uni = await uniko.Uniko.open("./data/kb")       # persistent
    ```

=== "Builder"

    ```python
    uni = await (uniko.Uniko.builder()
        .path("./data/kb")
        .llm(uniko.LlmSpec.openai("answerer", "gpt-4o-mini"))
        .streaming(True)
        .scope_to_agent()
        .build())
    ```

| Method | Signature | Sync | Description |
| --- | --- | --- | --- |
| `open` | `@staticmethod open(path: str) -> Awaitable[Uniko]` | `open_sync` | Persistent, zero-config defaults. |
| `in_memory` | `@staticmethod in_memory() -> Awaitable[Uniko]` | `in_memory_sync` | Ephemeral, zero-config defaults. |
| `builder` | `@staticmethod builder() -> UnikoBuilder` | — (sync) | Configure capabilities, then `build()`. |
| `agent` | `agent(agent_id: str) -> Agent` | — (sync) | Mint an agent identity (cheap). |
| `config` | `config() -> dict[str, Any]` | — (sync) | The resolved configuration as a dict. |
| `purge` | `purge() -> Awaitable[DeletionReport]` | `purge_sync` | Erase the whole graph (dev/test reset). |
| `shutdown` | `shutdown() -> Awaitable[None]` | `shutdown_sync` | Drain workers and close. **Consumes the handle.** |

`config()` returns a plain dict with keys: `embedding_model_id`, `embedding_dimensions`,
`recall_limit`, `recall_token_budget`, `recall_min_score`.

!!! warning "`shutdown` needs sole ownership"
    `shutdown` consumes the handle and requires that all derived `Agent`/`Session` objects are
    dropped first (a store ref-count guard), or it raises. Drop them explicitly, then shut down:

    ```python
    del session, agent      # MUST drop derived handles first
    uni.shutdown_sync()     # else raises ConfigError
    ```

    Any subsequent use of the `Uniko` handle after `shutdown` raises `ConfigError`.

---

## `UnikoBuilder`

Chainable, synchronous setters; `build` is the terminal async step. Returns `self` from every setter
so you can fluently compose, and re-using a builder after `build()` raises
`RuntimeError("UnikoBuilder already consumed by build()")`.

| Method | Signature | Sync | Description |
| --- | --- | --- | --- |
| `path` | `path(path: str) -> UnikoBuilder` | — (sync) | File-backed store at `path`. |
| `in_memory` | `in_memory() -> UnikoBuilder` | — (sync) | Ephemeral store (default). |
| `llm` | `llm(spec: LlmSpec) -> UnikoBuilder` | — (sync) | Generation model — **required for `answer()`**. |
| `streaming` | `streaming(on: bool) -> UnikoBuilder` | — (sync) | Enables the fire-and-forget `submit`/`flush` lane. |
| `scope_to_agent` | `scope_to_agent() -> UnikoBuilder` | — (sync) | Reads default to the agent's visibility. |
| `build` | `build() -> Awaitable[Uniko]` | `build_sync` | Terminal: produce the `Uniko` instance. |

---

## `LlmSpec`

Describes the generation model that powers `answer()` / `answer_in()`. All constructors are static
and synchronous.

| Constructor | Signature | Notes |
| --- | --- | --- |
| `openai` | `@staticmethod openai(alias: str, model_id: str, base_url: str \| None = None)` | Key read from `OPENAI_API_KEY`. |
| `openai_with_key_env` | `@staticmethod openai_with_key_env(alias: str, model_id: str, key_env: str, base_url: str \| None = None)` | Key read from the named env var. |
| `mistralrs` | `@staticmethod mistralrs(alias: str, model_id: str)` | Local model, in-situ Q4K quantization. |

```python
spec = uniko.LlmSpec.openai("answerer", "gpt-4o-mini")
uni = await uniko.Uniko.builder().llm(spec).build()
```

---

## `Agent`

An `Agent` binds an identity to the store. It is `Arc`-backed and clonable, so passing it around is
cheap. Read through the agent (`recall` / `answer` / `query`), retrieve by id through `agent.data`,
and plan through `agent.goals`.

```python
agent = uni.agent("assistant")
bundle = await agent.recall("what does the user prefer?")
answer = await agent.answer("when is the deadline?")   # needs .llm(...)
```

### Properties

| Property | Type | Description |
| --- | --- | --- |
| `agent_id` | `str` | The agent's identity. |
| `data` | `Data` | By-id retrieval surface. |
| `goals` | `Goals` | Goals & tasks surface. |

### Recall, answer, query

| Method | Signature | Sync | Description |
| --- | --- | --- | --- |
| `recall` | `recall(query: str) -> Awaitable[ContextBundle]` | `recall_sync` | Ranked, prompt-ready context. |
| `recall_in` | `recall_in(query: str, scope: Scope) -> Awaitable[ContextBundle]` | `recall_in_sync` | Recall confined to a `Scope`. |
| `answer` | `answer(question: str) -> Awaitable[Answer]` | `answer_sync` | Recall + the configured LLM. |
| `answer_in` | `answer_in(question: str, scope: Scope) -> Awaitable[Answer]` | `answer_in_sync` | Scoped `answer`. |
| `query` | `query(cypher: str) -> Awaitable[list[dict[str, Any]]]` | `query_sync` | Read-only Cypher; mutations rejected. |
| `query_in` | `query_in(cypher: str, scope: Scope) -> Awaitable[list[dict[str, Any]]]` | `query_in_sync` | Binds the scope allow-set as `$allow`. |

!!! note "`query_in` is a raw escape hatch"
    `query_in` binds the scope's allow-set as the `$allow` parameter for you to reference in your
    Cypher, then returns the raw rows **with no visibility filtering applied**. You enforce the
    filter in your query.

!!! warning "`answer()` requires an LLM"
    Calling `answer()` / `answer_in()` on an instance built without `.llm(...)` raises `ConfigError`
    (`e.kind == "config"`).

### Logic & reasoning

| Method | Signature | Sync | Description |
| --- | --- | --- | --- |
| `define_rule` | `define_rule(name: str, source: str) -> Awaitable[int]` | `define_rule_sync` | Register a Locy rule; returns its node id. |
| `run_rule` | `run_rule(name: str, return_cols: list[str], params: dict \| None = None) -> Awaitable[list[dict]]` | `run_rule_sync` | Evaluate a named rule. |
| `abduce` | `abduce(program: str, params: dict \| None = None) -> Awaitable[AbductionResult]` | `abduce_sync` | Run an abductive program. |
| `assume` | `assume(assume_block: str) -> AssumeBuilder` | — (sync) | Open a hypothetical-mutation builder. |

`assume` is synchronous and returns an [`AssumeBuilder`](#assumebuilder); the awaitable lives on the
builder's terminal `run`/`run_sync`. See [Reasoning with Locy](../guides/reasoning-with-locy.md).

### Sessions & lifecycle

| Method | Signature | Sync | Description |
| --- | --- | --- | --- |
| `session` | `session(session_id: str) -> Session` | — (sync) | Open a conversation thread. |
| `delete_session` | `delete_session(session_id: str) -> Awaitable[DeletionReport]` | `delete_session_sync` | Hard-delete a session and cascade. |
| `finalize_session` | `finalize_session(session_id: str) -> Awaitable[FinalizeReport]` | `finalize_session_sync` | Build/refresh a session's retrieval surfaces without its `Session` handle. |
| `consolidate` | `consolidate() -> Awaitable[CycleStats]` | `consolidate_sync` | Run one consolidation cycle now: Observations → Facts. |
| `unfinalized_session_ids` | `unfinalized_session_ids() -> Awaitable[list[str]]` | `unfinalized_session_ids_sync` | Sessions with no session-level chunks yet — drives a backfill loop. |
| `forget_participant` | `forget_participant(participant_id: str) -> Awaitable[DeletionReport]` | `forget_participant_sync` | Forget a participant. |

---

## `Session`

A `Session` is a conversation thread you write through. `observe` takes a mutable, serialized
borrow — **feed turns in order, one at a time**. Each `observe` is durable and committed before it
returns, so reads see the write immediately.

```python
session = agent.session("conversation-1")
result = await session.observe(uniko.Turn("alice", "I love hiking in the mountains."))
assert result.message_node_id > 0
```

### Properties

| Property | Type |
| --- | --- |
| `session_id` | `str` |

### Write & ingest

| Method | Signature | Sync | Description |
| --- | --- | --- | --- |
| `observe` | `observe(turn: Turn) -> Awaitable[ObserveResult]` | `observe_sync` | Durable, committed-before-return turn; read-after-write. |
| `ingest` | `ingest(source: IngestSource) -> Awaitable[IngestOutcome]` | `ingest_sync` | Standalone doc/blob, not a turn. |
| `submit` | `submit(turn: Turn) -> Awaitable[None]` | `submit_sync` | Fire-and-forget streaming turn. |
| `submit_source` | `submit_source(source: IngestSource) -> Awaitable[None]` | `submit_source_sync` | Fire-and-forget streaming doc/blob. |
| `flush` | `flush() -> Awaitable[None]` | `flush_sync` | Wait for the streaming pipeline to drain. |
| `finalize` | `finalize() -> Awaitable[FinalizeReport]` | `finalize_sync` | Build/refresh the session-level retrieval surfaces. |
| `summarize` | `summarize() -> Awaitable[int \| None]` | `summarize_sync` | New summary node id, or `None` if nothing new. |

!!! note "Streaming lane requires `streaming(True)`"
    `submit` / `submit_source` are fire-and-forget — they return immediately and process in the
    background. They require the instance was built with `.streaming(True)`. Use `flush()` to wait
    for the backlog to clear.

!!! tip "Or just use `async with`"
    `Session` is a context manager — the exit calls `finalize()` for you, best-effort:

    ```python
    async with agent.session("conversation-1") as session:
        await session.observe(uniko.Turn("alice", "..."))
    # finalized here
    ```

    A synchronous `with` block works the same way. Exit never suppresses an exception
    already propagating out of the block.

!!! tip "Call `finalize()` at the end of a conversation"
    `finalize()` chunks the whole transcript and aggregates the session's observations into dense
    chunks wired to the entities and participants they mention. These are what session-scoped
    recall and the Phase 1 session boost retrieve — a session that is never finalized contributes
    only its per-turn chunks.

    It is cheap and idempotent when the session has not grown (nothing is rewritten or
    re-embedded), so calling it periodically during a long conversation is fine. `summarize()`
    calls it for you on a best-effort basis. When streaming is on, `finalize()` flushes first.

### Forget & delete

| Method | Signature | Sync | Description |
| --- | --- | --- | --- |
| `forget_turn` | `forget_turn(message_id: str) -> Awaitable[DeletionReport]` | `forget_turn_sync` | Soft-forget: redact, keep the chain intact. |
| `delete_turn` | `delete_turn(message_id: str) -> Awaitable[DeletionReport]` | `delete_turn_sync` | Hard-delete a turn + cascade. |
| `delete_document` | `delete_document(artifact_id: str) -> Awaitable[DeletionReport]` | `delete_document_sync` | Hard-delete an artifact + cascade. |

---

## `Data`

By-id retrieval, reached via `agent.data`. An unknown id returns `None` — these methods never raise
for a missing id.

| Method | Signature | Sync | Description |
| --- | --- | --- | --- |
| `message` | `message(message_id: str) -> Awaitable[MessageView \| None]` | `message_sync` | Fetch a stored message. |
| `artifact` | `artifact(artifact_id: str) -> Awaitable[ArtifactView \| None]` | `artifact_sync` | Fetch an artifact's metadata + text. |
| `artifact_bytes` | `artifact_bytes(artifact_id: str) -> Awaitable[bytes \| None]` | `artifact_bytes_sync` | Fetch the raw binary payload. |

!!! warning "Binary payloads: use `artifact_bytes`"
    Raw bytes cannot be read back through a Cypher `RETURN`. `artifact_bytes` is the supported path
    for binary payloads — reach for it instead of `query`/`query_in` when you need the bytes.

---

## `Goals`

Goals & tasks, reached via `agent.goals`. Reads are scoped to the goals the agent owns and the tasks
assigned to it. Transition methods return `bool` — `False` when the id does not resolve. Creation
methods return the new node id (`int`).

!!! note "The Participant must exist first"
    Create the agent's Participant by sending at least one turn before creating goals:

    ```python
    await agent.session("setup").observe(uniko.Turn("assistant", "kicking off"))
    goals = agent.goals
    gid = await goals.create("Ship the Python SDK", goal_id="g1",
                             description="PyO3 bindings", metrics={"target_phase": 3})
    await goals.start("g1")
    tid = await goals.create_task("write tests", goal_id="g1", task_id="t1", priority=0.8)
    ctx = await goals.context("g1")
    await goals.complete("g1", {"done": True})
    ```

### Reads

Each read has a `_sync` twin (`all_sync`, `get_sync`, …).

| Method | Signature | Returns |
| --- | --- | --- |
| `all` | `all()` | `list[GoalView]` |
| `in_phase` | `in_phase(phase: str)` | `list[GoalView]` |
| `active` | `active()` | `list[GoalView]` |
| `planned` | `planned()` | `list[GoalView]` |
| `completed` | `completed()` | `list[GoalView]` |
| `get` | `get(goal_id: str)` | `GoalView \| None` |
| `tasks` | `tasks()` | `list[TaskView]` |
| `tasks_in` | `tasks_in(phase: str)` | `list[TaskView]` |
| `tasks_of` | `tasks_of(goal_id: str)` | `list[TaskView]` |
| `context` | `context(goal_id: str)` | `GoalContext \| None` |

### Transitions

Each returns `Awaitable[bool]` (`False` when the id doesn't resolve) and has a `_sync` twin.

| Method | Signature |
| --- | --- |
| `start` | `start(goal_id: str)` |
| `abandon` | `abandon(goal_id: str)` |
| `set_status` | `set_status(goal_id: str, status: str)` |
| `complete` | `complete(goal_id: str, result: Any \| None = None)` |
| `start_task` | `start_task(task_id: str)` |
| `block_task` | `block_task(task_id: str)` |
| `complete_task` | `complete_task(task_id: str)` |
| `set_task_status` | `set_task_status(task_id: str, status: str)` |

### Creation

Both return `Awaitable[int]` (the new node id) and have a `_sync` twin. All arguments after `title`
are keyword-only.

```python
create(title, *, goal_id=None, description=None, status=None, metrics=None,
       guardrails=None, deadline: datetime | None = None, parent_goal_id=None)

create_task(title, *, task_id=None, description=None, status=None,
            priority: float | None = None, goal_id=None,
            depends_on_task_id=None, subtask_of_task_id=None)
```

!!! note "Valid phase strings"
    Phases are plain lowercase strings, not enums.

    - **Goal phases:** `planned`, `active`, `completed`, `abandoned`
    - **Task phases:** `planned`, `active`, `completed`, `blocked`

    Passing an invalid phase to `in_phase` / `tasks_in` raises `ValueError`. To leave a `blocked`
    task, use `start_task`, `complete_task`, or `set_task_status`.

---

## `AssumeBuilder`

Returned by `Agent.assume`. Lets you run a hypothetical mutation plus a query against it; the
mutation **rolls back** and never touches the real graph.

| Method | Signature | Description |
| --- | --- | --- |
| `then_query` | `then_query(query: str) -> AssumeBuilder` | Cypher to run against the hypothetical state. |
| `param` | `param(key: str, value: Any) -> AssumeBuilder` | Bind a query parameter. |
| `run` | `run() -> Awaitable[list[dict]]` | Execute; returns rows. Has `run_sync`. |

```python
rows = await (agent.assume(
    "ASSUME { CREATE (:Fact {fact_id: 'srv-port', subject: 'srv', predicate: 'port', object: '9090'}) }")
    .then_query("MATCH (f:Fact {subject: 'srv'}) RETURN f")
    .run())
assert len(rows) == 1

after = await agent.query("MATCH (f:Fact {subject: 'srv'}) RETURN f")
assert after == []   # hypothetical mutation rolled back
```

---

## Input & value-object builders

All input builders are synchronous constructors with chainable setters.

### `Turn`

```python
Turn(sender_id: str, content: str)
```

A `Turn` is reusable — feeding it to `observe`/`submit` does not consume it.

| Setter | Signature | Default / Notes |
| --- | --- | --- |
| `.id` | `.id(message_id: str)` | Auto-generated if unset. |
| `.content_type` | `.content_type(content_type: str)` | `"text"` |
| `.at` | `.at(timestamp: datetime)` | now |
| `.addressed_to` | `.addressed_to(recipients: list[str])` | — |
| `.metadata` | `.metadata(key: str, value: Any)` | Add one metadata entry. |
| `.attach` | `.attach(source: IngestSource)` | Attach one source. |
| `.attachments` | `.attachments(sources: list[IngestSource])` | Attach many. |

```python
turn = (uniko.Turn("alice", "see the attached report")
        .addressed_to(["bob"])
        .metadata("channel", "email")
        .attach(uniko.IngestSource.from_path("report.pdf")))
```

### `IngestSource`

Factories build a source; chainable setters refine it.

| Member | Signature | Notes |
| --- | --- | --- |
| `from_text` | `@staticmethod from_text(content: str)` | Text payload. |
| `from_bytes` | `@staticmethod from_bytes(data: bytes)` | Bytes payload (one arg — set MIME via `.with_mime`). |
| `from_path` | `@staticmethod from_path(path: str)` | Load from a file path. |
| `.with_mime` | `.with_mime(mime: str)` | Skip sniffing; invalid MIME raises `ValueError`. |
| `.with_id` | `.with_id(id: str)` | Set the artifact id. |
| `.with_path` | `.with_path(path: str)` | Set the logical path. |

```python
src = uniko.IngestSource.from_bytes(raw).with_mime("application/pdf").with_id("doc-1")
```

### `Scope`

```python
Scope()
```

Confines reads to a subset of the graph. Chainable.

| Setter | Signature | Notes |
| --- | --- | --- |
| `.sessions` | `.sessions(sessions: list[str])` | Restrict to these sessions. |
| `.participants` | `.participants(participants: list[str])` | Restrict to these participants. |
| `.since` | `.since(since: datetime)` | Inclusive lower bound. |
| `.until` | `.until(until: datetime)` | Exclusive upper bound. |

```python
scope = uniko.Scope().sessions(["s1"]).since(cutoff)
bundle = await agent.recall_in("budget", scope)
```

---

## Output value types

All output types are frozen — read-only getters, no setters — and implement `__repr__`.

### `RecallSource`

The provenance of a recalled item.

| Field | Type |
| --- | --- |
| `kind` | `str` |
| `message_id` | `str \| None` |
| `artifact_id` | `str \| None` |
| `chunk_id` | `str \| None` |

### `RecallItem`

One ranked piece of context.

| Field | Type |
| --- | --- |
| `node_id` | `int` |
| `kind` | `str` |
| `score` | `float` |
| `content` | `str` |
| `sources` | `list[RecallSource]` |

### `ContextBundle`

The result of `recall` / `recall_in`. Implements `__len__`.

| Field | Type | Description |
| --- | --- | --- |
| `items` | `list[RecallItem]` | Ranked context items. |
| `total_tokens` | `int` | Token count of the bundle. |
| `phase1_only` | `bool` | Only phase-1 retrieval ran. |
| `phase2_only` | `bool` | Only phase-2 retrieval ran. |
| `coverage` | `float` | Retrieval coverage estimate. |

### `Answer`

The result of `answer` / `answer_in`.

| Field | Type |
| --- | --- |
| `text` | `str` |
| `model` | `str \| None` |
| `input_tokens` | `int \| None` |
| `output_tokens` | `int \| None` |
| `recorded_episode` | `int \| None` |
| `context` | `ContextBundle` |

| Method | Signature | Returns |
| --- | --- | --- |
| `citations` | `citations()` | `list[RecallSource]` |

### `ObserveResult`

The result of `observe`.

| Field | Type |
| --- | --- |
| `message_node_id` | `int` |
| `chunk_node_ids` | `list[int]` |
| `session_node_id` | `int` |
| `sender_node_id` | `int \| None` |
| `sender_id` | `str \| None` |
| `extracted_entities` | `list[tuple[int, str]]` |
| `extracted_observations` | `list[int]` |
| `attachment_count` | `int` |

### `DeletionReport`

Returned by every delete/forget/purge verb.

| Field | Type |
| --- | --- |
| `nodes_deleted` | `int` |
| `edges_deleted` | `int` |
| `facts_invalidated` | `int` |
| `chains_repaired` | `int` |
| `nodes_redacted` | `int` |
| `root_existed` | `bool` |

### `FinalizeReport`

Returned by `Session.finalize` and `Agent.finalize_session`.

| Field | Type | Meaning |
| --- | --- | --- |
| `transcript_chunks` | `list[int]` | Node ids of the transcript chunks (`chunk_type = "session"`). |
| `observation_chunks` | `list[int]` | Node ids of the observation chunks (`chunk_type = "observation"`). |
| `rebuilt` | `bool` | `True` when this call wrote to the graph; `False` when the surfaces were already current. |
| `ended_at` | `datetime \| None` | The Session's latest message time, stamped by `finalize`. A Session is *open* while this is null, so finalizing removes it from the inactivity auto-close sweep. |

### `CycleStats`

Returned by `Agent.consolidate`.

| Field | Type |
| --- | --- |
| `observations_processed` | `int` |
| `facts_created` | `int` |
| `facts_reinforced` | `int` |
| `facts_invalidated` | `int` |
| `drift_alerts` | `int` |

### `MessageView`

Returned by `Data.message`.

| Field | Type |
| --- | --- |
| `message_id` | `str` |
| `sender_id` | `str` |
| `content` | `str` |
| `timestamp` | `datetime` (tz-aware) |
| `session_id` | `str` |
| `addressed_to` | `list[str]` |
| `attachments` | `list[str]` |

### `ArtifactView`

Returned by `Data.artifact`.

| Field | Type |
| --- | --- |
| `artifact_id` | `str` |
| `kind` | `str` |
| `mime` | `str \| None` |
| `path` | `str \| None` |
| `text` | `str` |
| `attached_to_message` | `str \| None` |

### `GoalView`

| Field | Type |
| --- | --- |
| `goal_id` | `str` |
| `title` | `str` |
| `description` | `str \| None` |
| `status` | `str` |
| `phase` | `str` |
| `created_at` | `datetime \| None` |
| `deadline` | `datetime \| None` |
| `completed_at` | `datetime \| None` |
| `metrics` | `Any \| None` |

### `TaskView`

| Field | Type |
| --- | --- |
| `task_id` | `str` |
| `title` | `str` |
| `description` | `str \| None` |
| `status` | `str` |
| `phase` | `str` |
| `priority` | `float \| None` |
| `goal_id` | `str \| None` |
| `created_at` | `datetime \| None` |
| `completed_at` | `datetime \| None` |

### `GoalContext`

Returned by `Goals.context`.

| Field | Type |
| --- | --- |
| `goal` | `GoalView` |
| `tasks` | `list[TaskView]` |
| `sessions` | `list[str]` |
| `recent_messages` | `list[str]` |
| `facts` | `list[str]` |
| `entities` | `list[str]` |

### `IngestOutcome`

Returned by `Session.ingest`.

| Field | Type |
| --- | --- |
| `kind` | `str` |
| `artifact_id` | `str` |
| `artifact_node_id` | `int` |
| `chunk_node_ids` | `list[int]` |
| `was_deduplicated` | `bool` |
| `page_count` | `int \| None` |
| `extraction_failed` | `bool \| None` |

### `AbductionResult`

Returned by `Agent.abduce`. Implements `__len__`.

| Field | Type |
| --- | --- |
| `modifications` | `list[dict[str, Any]]` |

Each modification dict has keys `modification`, `validated: bool`, and `cost: float`.

```python
result = await agent.abduce("ABDUCE reachable WHERE a.kind = 'query'")
for mod in result.modifications:
    print(mod["modification"], mod["validated"], mod["cost"])
```

---

## `uniko.models` — the Pydantic overlay

The native classes above are frozen PyO3 snapshots: fast, immutable, and not Pydantic models. For
codebases that want validation, JSON Schema, or FastAPI response models, the package also ships a
pure-Python overlay in the `uniko.models` namespace. `pydantic>=2` is an unconditional runtime
dependency, so it is always available — nothing extra to install.

The overlay never shadows the native names: `uniko.ContextBundle` stays the native class,
`uniko.models.ContextBundle` is the Pydantic mirror.

```python
from uniko.models import TypedUniko, ContextBundle

memory = await TypedUniko.open("./memory.db")
agent = memory.agent("assistant")

bundle: ContextBundle = await agent.recall("hiking")   # a real BaseModel
print(bundle.model_dump_json())                        # serializable
print(ContextBundle.model_json_schema())               # OpenAPI-ready
```

| Module | What it holds |
| --- | --- |
| `uniko.models.outputs` | Pydantic mirrors of the frozen snapshots — `ContextBundle`, `RecallItem`, `Answer`, `ObserveResult`, `FinalizeReport`, `DeletionReport`, the views, and more. Each has a `from_native(native)` classmethod. |
| `uniko.models.inputs` | Validated input payloads (`extra="forbid"`) — `GoalSpec`, `TaskSpec`, `ScopeSpec`, `TurnSpec`, `IngestSourceSpec`, `LlmSpecModel` — each with `to_native()`. |
| `uniko.models.config` | `UnikoConfigModel`, mirroring the subset `Uniko.config()` returns. |
| `uniko.models.adapters` | Free functions (`observe`, `ingest`, `recall_in`, `answer_in`, …) that take a native handle, accept specs, and return models. Each has a `*_sync` twin. |
| `uniko.models.typed` | `TypedUniko`, `TypedUnikoBuilder`, `TypedAgent`, `TypedSession`, `TypedData`, `TypedGoals` — wrapper handles whose snapshot-returning methods hand back Pydantic models. |

!!! note "Where the overlay passes through"
    Surfaces with no fixed shape — Cypher result rows, rule parameters, raw artifact bytes — pass
    through untyped. The overlay types what has a schema, not everything.

---

## Value ↔ Python type mapping

Cypher rows (`query`, `query_in`, `run_rule`, `AssumeBuilder.run`) come back as plain
`list[dict[str, Any]]`. Engine values map to native Python types as follows.

| Engine `Value` | Python type |
| --- | --- |
| Null | `None` |
| Bool | `bool` |
| Int | `int` |
| Float | `float` |
| String | `str` |
| Bytes | `bytes` — **not** readable via Cypher `RETURN`; use [`Data.artifact_bytes`](#data) |
| List | `list` |
| Map / record | `dict` |
| DateTime | `datetime` (tz-aware) |

---

## Exceptions

The base is `UnikoError(Exception)`. **Every** instance carries a `.kind: str` attribute, so you can
branch on the kind even when the concrete class is the base.

```python
try:
    await agent.observe(turn)
except uniko.ConflictError:
    ...                          # transient — retry
except uniko.UnikoError as e:
    if e.kind == "storage":
        ...                      # a storage-domain error folded into the base
```

### Dedicated subclasses

Each is an `issubclass` of `UnikoError` and carries a fixed `.kind`.

| Class | `.kind` | When |
| --- | --- | --- |
| `ConfigError` | `"config"` | `answer()` without an LLM; using a handle after `shutdown`; bad configuration. |
| `LlmError` | `"llm"` | The generation model failed. |
| `TimeoutError` | `"timeout"` | An operation timed out. |
| `ConflictError` | `"conflict"` | Transient SSI / antidependency abort — **retriable**. |
| `UnsupportedError` | `"unsupported"` | No extractor for the modality. |

!!! note "`uniko.TimeoutError`"
    `uniko.TimeoutError` is a `UnikoError` subclass, distinct from the built-in
    `builtins.TimeoutError`. Catch `uniko.TimeoutError` (or the base) for engine timeouts.

### Storage-domain errors

These fold into the **base** `UnikoError` (they are *not* dedicated subclasses) but still carry a
distinct `.kind` — branch on it:

`"storage"` · `"search"` · `"schema"` · `"pipeline"` · `"locy"` · `"embedding"` · `"internal"`

```python
with pytest.raises(uniko.ConfigError) as info:
    await agent.answer("anything")   # no LLM configured
assert info.value.kind == "config"
```

---

## Next steps

<div class="feature-grid" markdown>

<div class="feature-card" markdown>
### [Quick Start](../getting-started/quickstart.md)
The 15-minute taste — install, observe a turn, recall it back.
</div>

<div class="feature-card" markdown>
### [Working with the Facade](../guides/facade.md)
The full mental model behind these verbs, in Rust and Python.
</div>

<div class="feature-card" markdown>
### [Recall & Retrieval](../guides/retrieval.md)
How `recall` ranks context, and how `Scope` narrows it.
</div>

<div class="feature-card" markdown>
### [Reasoning with Locy](../guides/reasoning-with-locy.md)
Rules, `assume`, and `abduce` — the logic surface in depth.
</div>

<div class="feature-card" markdown>
### [Configuration](../guides/configuration.md)
Embedders, models, recall budgets, and the defaults behind `config()`.
</div>

<div class="feature-card" markdown>
### [Data Model](../concepts/data-model.md)
The nodes and edges your turns and goals become.
</div>

</div>
