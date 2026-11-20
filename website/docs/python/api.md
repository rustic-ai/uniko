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
    `uniko.__version__` tracks the crate version (currently `0.5.1`), single-sourced from Cargo
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
        .config({"recall_limit": 10, "recall_token_budget": 4096})
        .embedding("bge-small")
        .reranker(None)
        .llm(uniko.LlmSpec.openai("generate/answer", "gpt-4o-mini"))
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
| `config` | `config() -> dict[str, Any]` | — (sync) | The full configuration this instance runs with, as nested dicts. |
| `health` | `health() -> PipelineHealth \| None` | — (sync) | Streaming-pipeline health; `None` without `streaming(True)`. |
| `inspect_store` | `@staticmethod inspect_store(path, config=None, *, embedding=None) -> Awaitable[StoreReport]` | `inspect_store_sync` | What the store at `path` was built with and how a config compares (`compatible`, `differences`, `migrations`). `migrations` is empty whenever a difference is incompatible. Applies no schema, changes no index or data and loads no model (uni-db's own open and close may still write bookkeeping files, such as a manifest and its WAL). A local path that isn't a store isn't touched at all and reports `info.exists == False`. Raises `ConfigError` while the store is open, inspected or migrated in this process (use `store_info` on an open handle), and `UnsupportedError` for a remote URI (a path containing `://`). See [Changing the Embedding Model](../guides/model-migration.md). |
| `migrate_store` | `@staticmethod migrate_store(path, migration, config=None, *, embedding=None, progress=None) -> Awaitable[MigrationReport]` | `migrate_store_sync` | Run `"enable_hybrid"` or `"rebuild_vector_indexes"` in place. The argument order differs from Rust's `migrate_store(path, config, migration, progress)`. `progress` gets a dict, typed as `uniko.types.MigrationProgressEvent` (`migration`, `phase`: `"schema"` \| `"backfill"` \| `"done"`, `label`, `done`, `total`); the final `"done"` event has `done == total == rows_backfilled + rows_unfilled`. An exception or panic in the callback is printed and ignored, and it can't cancel the run. An `async def` callback isn't awaited, and the callback must not call other `_sync` uniko methods. Resumable. Raises `IncompatibleStoreError` when the migration can't resolve the differences; `ConfigError` when there is no store at `path`, the config sets `schema_path`, or the store is open, inspected or migrated in this process; and `UnsupportedError` for a remote URI. While a migration runs, opening the store in this process raises `ConfigError`. |
| `store_info` | `store_info() -> Awaitable[StoreInfo]` | `store_info_sync` | What this instance's store was built with, including `hybrid_backfill_pending`. |
| `dead_letters` | property `-> DeadLetters` | — | Failed streaming items: list, count, retry, resolve. See [`DeadLetters`](#deadletters). |
| `add_team_member` | `add_team_member(participant_id: str, team_id: str) -> Awaitable[bool]` | `add_team_member_sync` | Record a team membership; `False` when already recorded. |
| `remove_team_member` | `remove_team_member(participant_id: str, team_id: str) -> Awaitable[bool]` | `remove_team_member_sync` | Remove it; `False` when not recorded. |
| `add_org_member` | `add_org_member(participant_id: str, org_id: str) -> Awaitable[bool]` | `add_org_member_sync` | Record a direct org membership. |
| `remove_org_member` | `remove_org_member(participant_id: str, org_id: str) -> Awaitable[bool]` | `remove_org_member_sync` | Remove it. |
| `add_team_to_org` | `add_team_to_org(team_id: str, org_id: str) -> Awaitable[bool]` | `add_team_to_org_sync` | Put a team in an org; its members get the org. |
| `remove_team_from_org` | `remove_team_from_org(team_id: str, org_id: str) -> Awaitable[bool]` | `remove_team_from_org_sync` | Take it out again. |
| `capabilities` | `capabilities() -> Awaitable[dict[str, Any]]` | `capabilities_sync` | Compiled features, extractors, each model's load state, the answer alias, OCR. Loads nothing. |
| `check_readiness` | `check_readiness(requirements: list[Requirement], timeout: float \| None = None) -> Awaitable[list[dict[str, Any]]]` | `check_readiness_sync` | The unmet requirements, loading any required model not yet loaded (a failed one is retried each call). Never raises for an unmet one. |
| `purge` | `purge() -> Awaitable[DeletionReport]` | `purge_sync` | Erase the whole graph (dev/test reset). |
| `shutdown` | `shutdown() -> Awaitable[None]` | `shutdown_sync` | Drain workers and close. **Consumes the handle.** |

`config()` returns every configuration field, nested the way the engine holds them
(`embedding`, `reranker`, `nlp` and `ocr` are sub-dicts), in the same shape
`UnikoBuilder.config()` accepts, so you can read it, change a value and build with it.
`embedding.provider_options` is shown as `"<redacted>"` when set, because it can hold API keys.
The fields and their defaults are listed in the [configuration guide](../guides/configuration.md).

`capabilities()` reports what this build and instance can do:

- `compiled_features`: each optional engine feature (`pdf-ocr`, `onnx`, `code-parse`, `gpu-cuda`,
  `gpu-metal`, `candle`, `llm`, `batch-record`) and whether this build has it;
- `extractors`: the modalities with a registered extractor;
- `models`: one entry per catalog alias, with `alias`, `task`, `provider_id`, `model_id`,
  `revision`, `state` (`"not_loaded"`, `"loading"`, `"loaded"` or `"failed"`, with `error` when failed),
  `api_key_env`, `base_url`, and the requested and active ONNX execution providers;
- `answer_alias`, `ocr_compiled` and `ocr_configured`.

Every published wheel compiles OCR in, so `ocr_compiled` is `True`; `ocr_configured` stays `False`
until the config sets `{"ocr": {"enabled": True}}`. With it on, a scanned PDF's pages are OCR'd
into `:Page`/`:Block` nodes with bounding boxes and confidence, and its text is recallable.

Prefetch at open (with the default `prefetch("eager")`) tries every model and keeps going past one
that fails, so a single bad alias shows up here as `"failed"` instead of stopping the others
loading; a later successful use clears it. `extractors` lists only extractors registered with
`extractor()`; built-in text, code, markup, structured and PDF ingest are always available. Model entries never include
provider options; an API key appears only as its environment-variable name, and `base_url` has
any userinfo and query string removed. `active_execution_providers` is known once a local ONNX
model has loaded, and lists the providers its sessions attached: one that failed to register, or
whose output disagreed with CPU, is left out. `TypedUniko.capabilities()` returns the same as a
`Capabilities` Pydantic model.

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
| `config` | `config(config: dict \| UnikoConfigModel) -> UnikoBuilder` | — (sync) | Base configuration, merged onto the defaults. |
| `embedding` | `embedding(embedding: str \| dict) -> UnikoBuilder` | — (sync) | Embedding model: a preset name or an `embedding` section. |
| `reranker` | `reranker(reranker: str \| dict \| None) -> UnikoBuilder` | — (sync) | Reranker: `"bge-base"`, `"minilm"`, a `reranker` section, or `None` for off. |
| `nlp` | `nlp(nlp: str \| dict) -> UnikoBuilder` | — (sync) | NLP model for entity and observation extraction: `"v1"` (default), a kniv v7 size (`"v7-tiny"`, `"v7-small"`, `"v7-medium"`, `"v7-large"`, `"v7-xlarge"`, `"v7-xlarge-d"`), or an `nlp` section. Wins over `config()` in either order. |
| `llm` | `llm(spec: LlmSpec) -> UnikoBuilder` | — (sync) | Generation model — **required for `answer()`**. Each call adds one; the first answers by default. |
| `answer_alias` | `answer_alias(alias: str) -> UnikoBuilder` | — (sync) | Which generation alias `answer()` uses: an `llm()` alias or one from the `catalog_path` file. `build()` raises `ConfigError` if it is missing or not a generation model. |
| `prefetch` | `prefetch(policy: "eager" \| "lazy" \| "background") -> UnikoBuilder` | — (sync) | When models load: before `build()` returns (default), on first use, or on a background task. |
| `runtime` | `runtime(runtime: SharedRuntime) -> UnikoBuilder` | — (sync) | Open over a runtime shared with other instances; `build()` raises `ConfigError` if it does not serve this builder's models. |
| `require` | `require(requirements: list[Requirement]) -> UnikoBuilder` | — (sync) | Models or stages `build()` must have ready; it raises `ReadinessError` otherwise. Adds to earlier calls. |
| `readiness_timeout` | `readiness_timeout(seconds: float \| None) -> UnikoBuilder` | — (sync) | How long `build()` waits for the required models to load; `None` or `math.inf` waits as long as loading takes. |
| `verify_remote` | `verify_remote(on: bool) -> UnikoBuilder` | — (sync) | Also check that each required remote model is reachable and accepts its credential, at `build()` and every later `check_readiness`; a failure is `verification_failed`. Off by default. |
| `streaming` | `streaming(on: bool) -> UnikoBuilder` | — (sync) | Enables the fire-and-forget `submit`/`flush` lane. |
| `scope_to_agent` | `scope_to_agent() -> UnikoBuilder` | — (sync) | Reads default to the agent's visibility. |
| `scope` | `scope(viewer: Viewer \| None) -> UnikoBuilder` | — (sync) | Every agent reads as `viewer`; `None` reads everything (the default). |
| `extractor` | `extractor(modality: str, extract: Callable[[ExtractRequest], str]) -> UnikoBuilder` | — (sync) | Index `"image"`, `"audio"` or `"video"` sources as the text `extract` returns. |
| `build` | `build() -> Awaitable[Uniko]` | `build_sync` | Terminal: produce the `Uniko` instance. |
| `build_runtime` | `build_runtime() -> Awaitable[SharedRuntime]` | `build_runtime_sync` | Terminal: a `SharedRuntime` for this builder's models, loaded as `prefetch()` says. |

To run several instances without loading each model several times, build one runtime and pass it
to each builder. Each instance keeps its own graph, and shutting one down leaves the runtime loaded
for the others. The LLMs belong to the runtime, so the instance builders need not repeat `llm()`:
the answer alias of the builder that built the runtime (its `answer_alias()`, else its first
`llm()`) answers unless `answer_alias()` names another.
`SharedRuntime.aliases` lists the aliases it serves.

```python
runtime = await uniko.Uniko.builder().build_runtime()
alice = await uniko.Uniko.builder().path("./alice").runtime(runtime).build()
bob = await uniko.Uniko.builder().path("./bob").runtime(runtime).build()
```

### Readiness

`Requirement` names what an application needs: `Requirement.alias(alias)`,
`.embedder()`, `.reranker()`, `.nlp()`, `.ocr()`, `.generator()`,
`.extractor("image" | "audio" | "video")` and `.accelerated(alias)`. `build()` loads each
required model whatever the `prefetch()` policy and raises `ReadinessError`, whose `.failures` is a
list of dicts (`requirement`, `code`, `alias`, `error`). A requirement the catalog, compiled
features or registered extractors cannot meet fails before the store is created.
`Uniko.check_readiness(requirements, timeout=None)` (and `check_readiness_sync`) returns the same
list without raising, for health endpoints; `TypedUniko.check_readiness` returns
`uniko.models.ReadinessFailure` models. The codes are listed in the
[facade guide](../guides/facade.md#require-models-at-startup).

```python
R = uniko.Requirement
try:
    uni = await (uniko.Uniko.builder()
        .prefetch("lazy")
        .require([R.embedder(), R.nlp()])
        .readiness_timeout(120)
        .build())
except uniko.ReadinessError as e:
    for f in e.failures:
        print(f["requirement"], f["code"], f["alias"], f["error"])
```

### Configuration

`config()` takes a dict, or a `uniko.models.UnikoConfigModel`, with any subset of the fields.
What you leave out keeps its default, nested sections included:

```python
uni = (uniko.Uniko.builder()
    .config({"recall_limit": 10, "reranker": {"top_n": 40}})
    .build_sync())
```

- **Unknown keys raise `ConfigError`** naming the key, with a suggestion for a near miss
  (`'recal_limit' (did you mean 'recall_limit'?)`). A wrong value type raises `ConfigError`
  naming its key, too.
- **`embedding()` and `reranker()` win** over the matching section of `config()`, whichever is
  called first.
- **Validation runs once, at `build()`**: a `reranker.top_n` below `recall_limit`, a queue
  capacity of `0`, and similar constraints raise `ConfigError` there, not in the setter.
- **Embedding presets**: `nomic`, `nomic-q`, `minilm`, `bge-small` (the default), `bge-large`,
  `bge-m3`, `gemma`.
- **Reopening a store with a different embedding model raises `IncompatibleStoreError`**, a
  `ConfigError` subclass. Its vectors were made with the original model and are not comparable
  with another's. A changed index setting raises it too, until a migration runs; see
  [Changing the Embedding Model](../guides/model-migration.md).
- `blob_storage` and `vector_algorithm` are replaced whole rather than merged, since a different
  variant has different keys: `{"blob_storage": {"kind": "fs", "root": "/data/blobs"}}`.

---

### Extractors

`extractor` lets a model you run index images, audio or video. Uniko calls `extract` with an
`ExtractRequest` (`modality`, `data` as bytes, `mime`, `path`, `metadata`) and stores the
returned text as the artifact's content, chunked and embedded like a document; the original
bytes are not kept. The function runs on a worker thread and holds the GIL only while it
runs. An exception in it fails that ingest.

```python
def caption(request: uniko.ExtractRequest) -> str:
    return my_vision_model.describe(request.data)

memory = uniko.Uniko.builder().extractor("image", caption).build_sync()
memory.agent("a").session("s").ingest_sync(uniko.IngestSource.from_path("photo.png"))
```

---

## `LlmSpec`

Describes the generation model that powers `answer()` / `answer_in()`. All constructors are static
and synchronous.

| Constructor | Signature | Notes |
| --- | --- | --- |
| `openai` | `@staticmethod openai(alias: str, model_id: str, base_url: str \| None = None)` | Key read from `OPENAI_API_KEY`. |
| `openai_with_key_env` | `@staticmethod openai_with_key_env(alias: str, model_id: str, key_env: str, base_url: str \| None = None)` | Key read from the named env var. |
| `llamacpp` | `@staticmethod llamacpp(alias: str, model_id: str, base_url: str, key_env: str \| None = None)` | A model held by a llama.cpp `llama-server`. `base_url` includes `/v1`; pass `key_env` only if the server runs with `--api-key`. |
| `new` | `@staticmethod new(alias: str, provider: str, model_id: str, options: dict[str, Any] \| None = None, api_key_env: str \| None = None)` | Any compiled generation provider: `remote/anthropic`, `remote/azure-openai`, `remote/cohere`, `remote/gemini`, `remote/llamacpp`, `remote/mistral`, `remote/openai`, `remote/vertexai`. `options` are provider options; credentials go only through `api_key_env` (for `remote/vertexai` it sets `api_token_env`). |

`alias` is a read-only property. Aliases take the `task/name` form. `build()` checks every spec
before opening the store and raises `ConfigError` naming the alias for a provider that cannot
generate, an option the provider rejects, an inline credential (`api_key`, `token`, ...), or an
alias registered twice. `repr()` shows only the alias.

```python
spec = uniko.LlmSpec.openai("generate/answer", "gpt-4o-mini")
claude = uniko.LlmSpec.new(
    "generate/claude", "remote/anthropic", "claude-sonnet-4-5", api_key_env="ANTHROPIC_API_KEY"
)
uni = await uniko.Uniko.builder().llm(spec).llm(claude).answer_alias("generate/claude").build()
```

The Pydantic layer's `LlmSpecModel` takes the same spec as `{"provider": "generic", "alias": ...,
"provider_id": ..., "model_id": ..., "options": ..., "api_key_env": ...}`.

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
| `sources` | `Sources` | Deterministic read tools: search, outline, read, tool calls. |
| `goals` | `Goals` | Goals & tasks surface. |

### Recall, answer, query

| Method | Signature | Sync | Description |
| --- | --- | --- | --- |
| `recall` | `recall(query: str) -> Awaitable[ContextBundle]` | `recall_sync` | Ranked, prompt-ready context. |
| `recall_in` | `recall_in(query: str, scope: Scope) -> Awaitable[ContextBundle]` | `recall_in_sync` | Recall confined to a `Scope`. |
| `recall_with` | `recall_with(query: str, scope: Scope \| None = None, *, limit=None, token_budget=None, min_score=None, reranker=None, reference_ts=None, strict=None) -> Awaitable[ContextBundle]` | `recall_with_sync` | Recall with per-call overrides; each `None` keeps the configured value. `strict=True` raises `DegradedError` when a stage falls back or fails. |
| `answer` | `answer(question: str) -> Awaitable[Answer]` | `answer_sync` | Recall + the configured LLM. |
| `answer_in` | `answer_in(question: str, scope: Scope) -> Awaitable[Answer]` | `answer_in_sync` | Scoped `answer`. |
| `query` | `query(cypher: str) -> Awaitable[list[dict[str, Any]]]` | `query_sync` | Read-only Cypher; mutations rejected. Dense vectors come back as `list[float]`, sparse vectors as `{"indices": list[int], "values": list[float]}`, binary vectors as `bytes`, and a date, duration or other value with no Python equivalent as its string form. Only a database null is `None`. |
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
| `list_rules` | `list_rules() -> Awaitable[list[RuleView]]` | `list_rules_sync` | Every rule, stdlib ones included. |
| `explain_rule` | `explain_rule(name: str) -> Awaitable[RuleView \| None]` | `explain_rule_sync` | A rule's source and state. |
| `delete_rule` | `delete_rule(name: str) -> Awaitable[bool]` | `delete_rule_sync` | Stop a rule and mark it pruned. Stdlib rules raise `ConfigError`. |

`assume` is synchronous and returns an [`AssumeBuilder`](#assumebuilder); the awaitable lives on the
builder's terminal `run`/`run_sync`. See [Reasoning with Locy](../guides/reasoning-with-locy.md).

### Sessions & lifecycle

| Method | Signature | Sync | Description |
| --- | --- | --- | --- |
| `session` | `session(session_id: str) -> Session` | — (sync) | Open a conversation thread. |
| `register` | `register(*, name: str \| None = None) -> Awaitable[Registration]` | `register_sync` | Record this agent as a participant of kind `agent`, writing nothing else. Returns a `Registration` with `participant_id` and `outcome` (`"created"`, `"upgraded"` or `"existing"`). Idempotent; a participant of kind `unknown` (for example from `add_org_member`) becomes `agent`, keeping its name and memberships. `name` is set only when the participant has none. Optional: goals, tasks, actions and tool calls create the participant themselves. |
| `delete_session` | `delete_session(session_id: str) -> Awaitable[DeletionReport]` | `delete_session_sync` | Hard-delete a session and cascade. |
| `finalize_session` | `finalize_session(session_id: str) -> Awaitable[FinalizeReport]` | `finalize_session_sync` | Build/refresh a session's retrieval surfaces without its `Session` handle. |
| `retire_source` | `retire_source(source_id: str) -> Awaitable[bool]` | `retire_source_sync` | Hide everything recorded under a source id from recall, reversibly. |
| `restore_source` | `restore_source(source_id: str) -> Awaitable[bool]` | `restore_source_sync` | Undo `retire_source`. |
| `consolidate` | `consolidate(*, llm_triples_alias: str \| None = None) -> Awaitable[ConsolidationReport]` | `consolidate_sync` | Consolidate now: Observations → Facts, then procedure promotion and topic detection. Pass `llm_triples_alias` (a registered generation alias) to refine each triple with that model first; an unknown or non-generation alias raises `ConfigError`. |
| `unfinalized_session_ids` | `unfinalized_session_ids() -> Awaitable[list[str]]` | `unfinalized_session_ids_sync` | Sessions with no session-level chunks yet — drives a backfill loop. |
| `forget_participant` | `forget_participant(participant_id: str) -> Awaitable[DeletionReport]` | `forget_participant_sync` | Forget a participant. |

### Procedures

| Method | Signature | Sync | Description |
| --- | --- | --- | --- |
| `match_procedures` | `match_procedures(state: dict[str, str]) -> Awaitable[list[MatchedProcedure]]` | `match_procedures_sync` | Active procedures whose precondition holds, e.g. `{"last_action_type": "search"}`. |
| `record_procedure_use` | `record_procedure_use(procedure_id: str, succeeded: bool) -> Awaitable[None]` | `record_procedure_use_sync` | Score how following one worked out. |

`consolidate` learns procedures from repeated successful sequences of recorded episodes;
`MatchedProcedure` has `procedure_id`, `name` and `effectiveness`.

### Facts, actions, episodes, observations

| Method | Signature | Sync | Description |
| --- | --- | --- | --- |
| `assert_fact` | `assert_fact(subject, predicate, object=None, *, observed_at=None, supporting_observation_ids=None, visibility=None) -> Awaitable[FactUpsert]` | `assert_fact_sync` | Write a fact directly; asserting an existing triple reinforces it. |
| `invalidate_fact` | `invalidate_fact(fact_id, *, replacement_fact_id=None, reason=None) -> Awaitable[None]` | `invalidate_fact_sync` | Close a fact's validity interval. |
| `record_action` | `record_action(action_type, *, action_id, input, output, status, started_at, ended_at, error, triggered_by_message_id, session_id, previous_action_id) -> Awaitable[ActionResult]` | `record_action_sync` | Record a tool call or other action. `input` / `output` are any JSON-shaped value. |
| `record_episode` | `record_episode(action_type, *, episode_id, outcome, state, delta, importance, timestamp, triggered_by_message_id, involved_action_ids) -> Awaitable[int]` | `record_episode_sync` | Record an outcome; procedure promotion learns from these. |
| `add_observation` | `add_observation(message_id, content, subject, *, predicate, object, temporal_phrase, temporal_anchor, observed_at, confidence, visibility) -> Awaitable[int]` | `add_observation_sync` | Add an observation anchored to a stored message. |
| `set_visibility` | `set_visibility(node_id: int, visibility: str) -> Awaitable[bool]` | `set_visibility_sync` | Scope a Fact or Observation. |

All keyword arguments default to `None`. The agent's participant is created if it has not
spoken yet.

### Visibility

Memory can be scoped to `"public"`, `"private:<participant_id>"`, `"team:<team_id>"` or
`"org:<org_id>"`: a whole turn with `Turn.visibility`, a single item with the `visibility`
argument of `assert_fact` and `add_observation`, or an existing Fact or Observation with
`set_visibility`. Scoped reads then leave out what the reader may not see:

```python
memory = uniko.Uniko.builder().scope_to_agent().build_sync()
memory.add_team_member_sync("alice", "eng")

writer = memory.agent("writer")
writer.add_observation_sync("msg-1", "launch moves to march", "launch", visibility="team:eng")

memory.agent("alice").recall_sync("launch")   # includes it
memory.agent("bob").recall_sync("launch")     # does not
```

- **Unscoped reads see everything.** Scope them with `scope_to_agent()`, `scope(Viewer(...))`,
  or per call with `Scope.as_viewer` / `Scope.as_participant`.
- **A scoped turn scopes everything drawn from it:** its message, its chunks, its
  observations, and the facts consolidated from them. Its attachments are not scoped.
- **Session-level chunks and summaries are built from public turns only,** since one chunk
  spans many turns. A forgotten turn is left out of them too.
- **A scoped fact is a separate fact.** The same triple asserted or observed under
  `team:eng` and publicly gives two facts, so neither reinforces nor contradicts the other.
  `set_visibility` on an existing fact changes that fact in place.
- **Forgotten items stay hidden:** `set_visibility` refuses to change them.
- **An `Agent` resolves its own memberships once,** on its first scoped read. Take a fresh one
  from `memory.agent(...)` to see a membership change.

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
| `commit_unit` | `commit_unit(turns: Sequence[Turn]) -> Awaitable[UnitResult]` | `commit_unit_sync` | Several turns as one atomic, idempotent write. |
| `summarize` | `summarize() -> Awaitable[SummarizeReport]` | `summarize_sync` | Finalize, then write a summary of anything new. |
| `observe_tool` | `observe_tool(call: ToolCall) -> Awaitable[ToolCallResult]` | `observe_tool_sync` | Record a tool call on its own, apart from any turn. |
| `record_action` | `record_action(action_type, *, ...) -> Awaitable[ActionResult]` | `record_action_sync` | `Agent.record_action`, linked to this session. |

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
for a missing id. They apply recall's policy: forgotten content, and on an agent scoped to a viewer
anything that viewer may not read, also returns `None`; `artifact_bytes` returns `None` when any of
the artifact's chunks is forgotten or hidden.

| Method | Signature | Sync | Description |
| --- | --- | --- | --- |
| `message` | `message(message_id: str) -> Awaitable[MessageView \| None]` | `message_sync` | Fetch a stored message. |
| `artifact` | `artifact(artifact_id: str) -> Awaitable[ArtifactView \| None]` | `artifact_sync` | Fetch an artifact's metadata + text. |
| `artifact_bytes` | `artifact_bytes(artifact_id: str) -> Awaitable[bytes \| None]` | `artifact_bytes_sync` | Fetch the raw binary payload. |

!!! warning "Binary payloads: use `artifact_bytes`"
    Raw bytes cannot be read back through a Cypher `RETURN`. `artifact_bytes` is the supported path
    for binary payloads — reach for it instead of `query`/`query_in` when you need the bytes.

---

## `Sources`

Deterministic read tools, reached via `agent.sources`; see the [Tool Output guide](../guides/tool-output.md).
A source is named by `source_kind` (`"artifact"`, `"message"` or `"session"`) and `source_id`. An
unknown or invisible source returns `None`.

| Method | Signature | Sync | Description |
| --- | --- | --- | --- |
| `search` | `search(query, *, limit=10, session_id=None, tool_name=None) -> Awaitable[list[SearchHit]]` | `search_sync` | Lexical search; identifiers and figures match whole. |
| `outline` | `outline(source_kind, source_id, *, depth=3, max_entries=200) -> Awaitable[Outline \| None]` | `outline_sync` | A JSON payload's objects and arrays, or a source's chunks. |
| `read` | `read(source_kind, source_id, *, address=None, bytes=None, lines=None, chunks=None, max_tokens=2000) -> Awaitable[ReadResult \| None]` | `read_sync` | Verbatim text; give exactly one range. |
| `tool_call` | `tool_call(call_id) -> Awaitable[ToolCallView \| None]` | `tool_call_sync` | A recorded call. |
| `tool_calls` | `tool_calls(*, session_id=None, tool_name=None, limit=50) -> Awaitable[list[ToolCallView]]` | `tool_calls_sync` | Recorded calls, newest first. |

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
| `task` | `task(task_id: str)` | `TaskView \| None` |
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

`create` and `create_task` return `Awaitable[int]` (the new node id). `create_view` and
`create_task_view` take the same arguments and return the created `GoalView` / `TaskView`, which
carries the `goal_id` / `task_id` every other method takes, including one generated because you
omitted it. Each has a `_sync` twin. All arguments after `title` are keyword-only.

```python
create(title, *, goal_id=None, description=None, status=None, metrics=None,
       guardrails=None, deadline: datetime | None = None, parent_goal_id=None)
create_view(...)        # same arguments -> GoalView

create_task(title, *, task_id=None, description=None, status=None,
            priority: float | None = None, goal_id=None,
            depends_on_task_id=None, subtask_of_task_id=None)
create_task_view(...)   # same arguments -> TaskView
```

Omitting `goal_id` / `task_id` generates a new id on every call, so two calls create two goals or
tasks. Pass an id when you need to know it up front.

Creating again with an id that already exists updates that goal or task in place: its properties
are overwritten and it keeps a single owner, assignee and goal link, so retrying with the same id
is safe. If only the embedding step fails, the goal or task is still stored and the `_view` forms
return it instead of raising, so a generated id is never lost.

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
| `.tool_call` | `.tool_call(call: ToolCall)` | Record a tool call this turn made; it commits with the turn. |
| `.visibility` | `.visibility(visibility: str)` | Scope the turn and what derives from it; see [Visibility](#visibility). |

```python
turn = (uniko.Turn("alice", "see the attached report")
        .addressed_to(["bob"])
        .metadata("channel", "email")
        .attach(uniko.IngestSource.from_path("report.pdf")))
```

### `ToolCall`

```python
ToolCall(tool_name: str)
```

`call.tool_name` and `call.performer_id` (`None` unless `.performer(...)` was set) read the call back.

One call of one tool. Reusable, like `Turn`.

| Setter | Signature | Default / Notes |
| --- | --- | --- |
| `.call_id` | `.call_id(call_id: str)` | Derived from the session, turn, tool, arguments and result if unset. |
| `.performer` | `.performer(participant_id: str)` | The turn's sender, or the session's agent for `observe_tool`. The turn keeps its sender; the performer is created as kind `agent` if needed. The same call id under a different performer raises `IdConflictError`. |
| `.arguments` | `.arguments(arguments: Any)` | Any JSON-shaped value. |
| `.json_output` / `.text_output` / `.bytes_output` | `.json_output(value)`, `.text_output(text)`, `.bytes_output(data, mime)` | The result, stored verbatim. |
| `.status` / `.error` | `.status(status: str)`, `.error(error: str)` | — |
| `.started_at` / `.ended_at` | `.started_at(at: datetime)` | now / the start time |
| `.triggered_by` | `.triggered_by(message_id: str)` | Only for `Session.observe_tool`. |
| `.after` | `.after(call_id: str)` | The session's previous call. |
| `.visibility` | `.visibility(visibility: str)` | The turn's visibility. |
| `.category` / `.source` / `.revision` | — | Provenance of the result. |

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
| `.as_participant` | `.as_participant(participant_id: str)` | Read as this participant, memberships looked up at recall time. |
| `.as_viewer` | `.as_viewer(viewer: Viewer)` | Read as this explicit viewer. |

`Viewer(participant_id, teams=None, orgs=None)` is a viewer identity with the memberships given
rather than looked up; it has `participant_id`, `teams` and `orgs` properties.

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
| `kind` | `"message"`, `"attachment"`, `"document"`, `"session"` or `"tool"` |
| `message_id` | `str \| None` |
| `session_id` | `str \| None` — for a session transcript chunk |
| `call_id` | `str \| None` — for a tool call's result |
| `tool_name` | `str \| None` |
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
| `token_count` | `int` — tokens in `content`, as the budget counts them |
| `context` | `str \| None` — for a tool-output chunk, the header of what it inherits |
| `address` | `str \| None` — where the chunk sits in its source |

### `ContextBundle`

The result of `recall` / `recall_in`. Implements `__len__`.

| Field | Type | Description |
| --- | --- | --- |
| `items` | `list[RecallItem]` | Ranked context items. |
| `total_tokens` | `int` | Token count of the bundle. |
| `phase1_only` | `bool` | Only phase-1 retrieval ran. |
| `phase2_only` | `bool` | Only phase-2 retrieval ran. |
| `coverage` | `float` | Retrieval coverage estimate. |
| `diagnostics` | `list[Diagnostic]` | What each requested recall stage did. |

### `Diagnostic`

One stage outcome of a recall or an ingest. No query or content text is included.

| Field | Type | Description |
| --- | --- | --- |
| `stage` | `str` | Recall: `query_embedding`, `query_analysis`, `sparse`, `colbert`, `rerank`, `source`, `provenance`. Ingest: `nlp`, `observation_model`, `observation_rules`, `pdf_extraction`, `artifact_embedding`, `extractor_post_commit`. |
| `status` | `str` | `applied`, `skipped`, `fallback` or `failed`. |
| `code` | `str \| None` | Why, when not `applied`, e.g. `embed_failed`, `rerank_failed`, `query_failed`, `not_reached`, `nlp_unavailable`. |
| `target` | `str \| None` | The node label, model alias or source the stage ran on. |
| `count` | `int \| None` | How many items the stage contributed, when it counts them. |

The stages and codes are described in the facade guide, for
[recall](../guides/facade.md#diagnostics-and-strict-recall) and
[ingest](../guides/facade.md#ingest-diagnostics). `uniko.models.Diagnostic` adds a `degraded`
property.

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
| `attachments` | `list[IngestOutcome]` — one per attachment, in order |
| `tool_calls` | `list[ToolCallResult]` — one per `Turn.tool_call`, in order |
| `diagnostics` | `list[Diagnostic]` — what each ingest stage did; filled by `observe`, empty for a turn of a multi-turn unit (see `UnitResult`) |

### `ToolCallResult`

The result of recording one tool call.

| Field | Type |
| --- | --- |
| `call_id` | `str` |
| `action_node` | `int` |
| `artifact_id` | `str \| None` |
| `artifact_node` | `int \| None` |
| `was_replay` | `bool` |
| `diagnostics` | `list[Diagnostic]` — filled by `observe_tool` |

### `SearchHit`, `Outline`, `OutlineEntry`, `ReadResult`, `ToolCallView`

Returned by `Sources`. A `SearchHit` has `source_kind`, `source_id`, `chunk_id`, `chunk_index`,
`address`, `start`, `end`, `text`, `context`, `score`, `session_id`, `call_id` and `tool_name`. An
`Outline` has `format`, `total_bytes`, `total_lines`, `truncated` and `entries`, each an
`OutlineEntry` with `address`, `kind`, `summary`, `start`, `end` and `depth`. A `ReadResult` has the
verbatim `text`, its byte span `start`/`end`, `first_line`/`last_line`, `truncated`, and
`next_bytes` — the `(start, end)` still to read when it was cut. A `ToolCallView` has `call_id`,
`tool_name`, `action_type`, `arguments`, `status`, `error`, `started_at`, `ended_at`,
`triggered_by`, `session_id` and `artifact_id`.

### `UnitResult`

Returned by `Session.commit_unit`. Implements `__len__`.

| Field | Type | Meaning |
| --- | --- | --- |
| `turns` | `list[ObserveResult]` | One result per turn, in the unit's order. |
| `was_replay` | `bool` | The unit was already committed with identical content, so nothing was written. |
| `diagnostics` | `list[Diagnostic]` | What each ingest stage did across the unit. |

`message_node_ids()` returns each turn's `message_node_id`, in order.

### `SummarizeReport`

Returned by `Session.summarize`.

| Field | Type | Meaning |
| --- | --- | --- |
| `summary` | `int \| None` | Node id of the new summary, or `None` when there was nothing new. |
| `finalize` | `FinalizeReport \| None` | What finalizing the session built first, when that succeeded. |
| `finalize_error` | `str \| None` | Why finalizing failed, when it did; the summary is still attempted. |

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

### `ConsolidationReport`

Returned by `Agent.consolidate`: `cycle` (`CycleStats`), `procedures` (`PromotionReport`:
`created`, `reinforced`, `promoted`) and `topics` (`TopicReport`: `created`, `updated`,
`entities_assigned`). Topic detection covers the whole store, not only the agent.

### `FactUpsert`, `ActionResult`, `RuleView`

- `FactUpsert` — from `assert_fact`: `node_id` (for `set_visibility`), `fact_id` (for
  `invalidate_fact`; a scoped fact's id ends in its visibility), `was_created`,
  `observation_count`.
- `ActionResult` — from `record_action`: `action_node`, `overflow_artifact` when the output
  was too large to keep inline, and `diagnostics` (its NLP entity-linking stage).
- `RuleView` — from `list_rules` / `explain_rule`: `name`, `source_type` (`stdlib`, `authored`,
  `derived`), `status`, `confidence`, and from `explain_rule` the `source` and
  `natural_language`.

### `CycleStats`

The `cycle` part of a `ConsolidationReport`.

| Field | Type |
| --- | --- |
| `observations_processed` | `int` |
| `facts_created` | `int` |
| `facts_reinforced` | `int` |
| `facts_invalidated` | `int` |
| `drift_alerts` | `int` |
| `triples_requested` | `int` — Observations sent for LLM refinement; `0` without `llm_triples_alias` |
| `triples_refined` | `int` — of those, how many were replaced by a parsed model reply; the rest kept the rule-based triple |
| `triples_failed` | `int` — of those, how many model calls failed outright (endpoint down, timeout) |

### `MessageView`

Returned by `Data.message`.

| Field | Type |
| --- | --- |
| `message_id` | `str` |
| `sender_id` | `str` |
| `content` | `str` |
| `content_type` | `str` |
| `category` | `str \| None` |
| `visibility` | `str \| None` — `None` is public |
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
| `page_count` | `int \| None` (PDFs) |
| `extraction_failed` | `bool \| None` (PDFs) |
| `extraction_error` | `str \| None` — why PDF extraction failed |
| `page_node_ids` | `list[int]` — the PDF's pages, in order |
| `block_node_ids` | `list[int]` — the PDF's layout blocks |
| `diagnostics` | `list[Diagnostic]` — what each ingest stage did (filled by `ingest`) |

### `PipelineHealth`

Returned by `Uniko.health`.

| Field | Type |
| --- | --- |
| `ingest` | `WorkerHealth` |
| `consolidation` | `WorkerHealth` |
| `llm_circuit` | `"closed" \| "open" \| "half_open"` |
| `uptime_secs` | `int` |

`WorkerHealth` has `status` (`"healthy"`, `"degraded"`, `"backpressured"` or `"stalled"`),
`queue_depth`, `queue_capacity`, `items_processed`, `items_failed`, `avg_latency_ms`, and
`degradations`: degraded ingest stages counted since start, keyed `stage:code`
(e.g. `"pdf_extraction:extraction_failed"`). Streaming ingest has no per-call result, so this is
where its diagnostics surface. `dead_lettered`, `retries` and `retries_succeeded` count dead
letters stored, retries resubmitted and retries that succeeded since start.

### `DeadLetters`

From `Uniko.dead_letters`. Listing, counting and resolving work on any instance; retrying needs
`streaming(True)` and raises `ConfigError` otherwise. Drop the handle before `shutdown()`, as with
an `Agent` or `Session`.

| Method | Signature | Sync | Description |
| --- | --- | --- | --- |
| `list` | `list(*, status=None, limit=None) -> Awaitable[list[DeadLetter]]` | `list_sync` | Oldest first. `status` is `"pending"`, `"retrying"`, `"exhausted"`, `"resolved"` or `"legacy"`. |
| `get` | `get(id: str) -> Awaitable[DeadLetter \| None]` | `get_sync` | One letter. |
| `counts` | `counts() -> Awaitable[DeadLetterCounts]` | `counts_sync` | Letters per status: `pending`, `retrying`, `exhausted`, `resolved`, `legacy`. |
| `retry` | `retry(id: str, *, force=False) -> Awaitable[None]` | `retry_sync` | Resubmit the stored payload; the outcome lands on the letter, and `Session.flush` waits for it. `force` retries a permanent failure or one out of retries. Raises `ConfigError` when it can't be retried, and `UnikoError` with `kind == "pipeline"` when the ingest queue is full (the attempt isn't counted). |
| `retry_pending` | `retry_pending(limit=100) -> Awaitable[RetryReport]` | `retry_pending_sync` | Retry pending, retriable letters now, at most `limit`. `RetryReport.retried` lists the ids resubmitted; `refused` lists `RetryRefusal(id, reason)`. |
| `resolve` | `resolve(id: str, note=None) -> Awaitable[bool]` | `resolve_sync` | Close a letter without retrying it. Only the letter changes, and its payload is dropped. Raises `ConfigError` while this process is retrying it. |

A `DeadLetter` has these fields:
- `id`, `step`, `error`, `error_kind` (e.g. `"internal"`, `"id_conflict"`);
- `retriable`, `status`, `retry_count`, `max_retries`;
- `ingest_type`, `agent_id`, `replayable`;
- `created_at`, `updated_at`, `next_retry_at`, `resolved_at`, `resolution`.

The typed `uniko.models.DeadLetter` mirrors them. Automatic background retries are opt-in through
the `dead_letter_auto_retry` config key; see the facade guide.

### `StoreReport`

Returned by `Uniko.inspect_store`. Rust's `StoreReport` holds a `Compatibility`; Python flattens
it into `compatible`, `differences` and `migrations`, and represents verdicts and migrations as
strings.

| Field | Type |
| --- | --- |
| `info` | `StoreInfo` |
| `differences` | `list[StoreDifference]` — empty when the config opens the store as is |
| `compatible` | `bool` |
| `migrations` | `list["enable_hybrid" \| "rebuild_vector_indexes"]` — empty when any difference is incompatible |

### `StoreInfo`

Returned by `Uniko.store_info`, and as `StoreReport.info`.

| Field | Type |
| --- | --- |
| `exists` | `bool` — `False` for a path that isn't a store |
| `embedding` | `StoredEmbedding` |
| `hybrid_backfill_pending` | `int \| None` — chunks and observations whose hybrid columns aren't filled yet, on a hybrid store |
| `migration_in_progress` | `"enable_hybrid" \| "rebuild_vector_indexes" \| None` — a migration that started and hasn't finished |

### `StoredEmbedding`

The embedding setup a store was built with, as stored. Every field is `None` when the store
doesn't record it.

| Field | Type |
| --- | --- |
| `model_id` | `str \| None` — `None` on a store from before the model was recorded |
| `dimensions` | `int \| None` |
| `sparse_dimensions` | `int \| None` |
| `multivector_dimensions` | `int \| None` |
| `algorithm` | `str \| None` — e.g. `"hnsw_sq(m=16, ef_construction=100)"` |
| `metric` | `str \| None` — `"cosine"`, `"l2"` or `"dot"` |
| `embed_alias` | `str \| None` — `"embed/default"`, or `"embed/hybrid"` on a hybrid store |
| `document_prefix` | `str \| None` |
| `query_prefix` | `str \| None` |
| `batch_size` | `int \| None` |
| `is_hybrid` | `bool` (property) — the store has the sparse or ColBERT column |

### `StoreDifference`

One entry of `StoreReport.differences`.

| Field | Type |
| --- | --- |
| `field` | `str` — e.g. `"dimensions"`, `"model_id"`, `"vector_metric"` |
| `stored` | `str` |
| `requested` | `str` |
| `verdict` | `"incompatible" \| "needs_migration" \| "unknown"` — `"unknown"` for a verdict this build doesn't know; the alias is `uniko.types.VerdictName` |
| `migration` | `"enable_hybrid" \| "rebuild_vector_indexes" \| None` — set when `verdict` is `"needs_migration"` |

### `MigrationReport`

Returned by `Uniko.migrate_store`.

| Field | Type |
| --- | --- |
| `migration` | `"enable_hybrid" \| "rebuild_vector_indexes"` |
| `rows_backfilled` | `int` — existing rows whose new columns were filled |
| `rows_unfilled` | `int` — rows re-embedded whose new columns still came back empty; passed over in this run, but still counted as pending, so a rerun tries them again |
| `completed` | `bool` — the store needs nothing further for this config. Still true when the only pending rows are the unfilled ones |
| `migration_in_progress` | `"enable_hybrid" \| "rebuild_vector_indexes" \| None` — a migration still marked unfinished on the store after the run, which is why `completed` can be false after a run that did its own part; rerun that migration to finish it |
| `elapsed_secs` | `float` |

The closed sets of strings the API passes (`GoalView.phase`, `RecallItem.kind`,
`WorkerHealth.status`, and others) have `Literal` aliases in `uniko.types` for type checking.

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
| `uniko.models.config` | `UnikoConfigModel`, the full configuration `Uniko.config()` returns and `UnikoBuilder.config()` accepts, with nested `EmbeddingConfigModel`, `RerankerConfigModel`, `NlpConfigModel` and `OcrConfigModel`. Unknown keys are rejected. |
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
| `UnikoTimeoutError` | `"timeout"` | An operation timed out. |
| `ConflictError` | `"conflict"` | Transient SSI / antidependency abort — **retriable**. |
| `UnsupportedError` | `"unsupported"` | No extractor for the modality. |
| `ReadinessError` | `"not_ready"` | A `require()`d model or stage is unavailable; `.failures` lists each. |
| `DegradedError` | `"degraded"` | A `strict=True` recall had a stage fall back or fail; `.diagnostics` lists them as dicts. |
| `IncompatibleStoreError` | `"incompatible_store"` | The store can't be opened (or migrated) with this config as is. A subclass of `ConfigError`, so `except ConfigError` catches it, but its `.kind` is `"incompatible_store"`. `.differences` lists dicts with `field`, `stored`, `requested`, `verdict` (`"incompatible"`, `"needs_migration"` or `"unknown"`; `uniko.types.VerdictName`) and `migration`; `.compatible` is a bool and `.migrations` the migration names that would fix it (empty when any difference is incompatible). |

!!! note "`UnikoTimeoutError`, formerly `uniko.TimeoutError`"
    The class was renamed so that `from uniko import *` no longer shadows the built-in
    `TimeoutError`. `uniko.TimeoutError` still resolves to it, with a `DeprecationWarning`,
    until the next minor release.

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
