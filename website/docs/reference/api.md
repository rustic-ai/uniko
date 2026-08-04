# API Reference

uniko is a Rust library, so its API is the set of crates you link and the methods you call — there
is no service to wire up. The public surface is the **`Uniko` facade**: one owning handle from which
you mint agents, open sessions, observe turns, and read memory back. Every signature below is
reproduced from the source.

One crate matters for callers:

- **`uniko-api`** — the public facade. It contains no logic; its `tools` module re-exports the
  curated surface (`Uniko`, `Agent`, `Session`, the view/result types). Depend on this crate and you
  get the whole reachable surface. The same types are re-exported from `uniko-memory`, which is where
  they're defined.

The lower-level engine (`KnowledgeBase`, `PipelineSystem`, the `recall`/`answer_query` free
functions) is still available for embedded/advanced use — see [Engine internals](#engine-internals).

!!! note
    Parameter structs use `..Default::default()` heavily — most fields are optional and documented
    inline in the crate's rustdoc. Only the load-bearing shape is shown here.

---

## Uniko — the instance

`Uniko` owns the store, the model runtime, and the ingest pipeline. Build one, mint agents from it,
and `shutdown` when done.

=== "Zero-config"

    ```rust
    use uniko_memory::Uniko;

    let memory = Uniko::in_memory().await?;        // ephemeral
    let memory = Uniko::open("./data/kb").await?;  // persistent
    ```

=== "Builder"

    ```rust
    use uniko_memory::{EmbeddingConfig, LlmSpec, Uniko};

    let memory = Uniko::builder()
        .path("./data/kb")
        .embedding(EmbeddingConfig::bge_small_en_v15())
        .llm(LlmSpec::openai("llm/default", "gpt-4o-mini", None))
        .streaming(true)
        .scope_to_agent()
        .build()
        .await?;
    ```

| Method | Signature | Description |
| --- | --- | --- |
| `open` | `async fn open(path: impl AsRef<Path>) -> Result<Self, UnikoError>` | Persistent, zero-config defaults. |
| `in_memory` | `async fn in_memory() -> Result<Self, UnikoError>` | Ephemeral, zero-config defaults. |
| `builder` | `fn builder() -> UnikoBuilder` | Configure capabilities, then `build()`. |
| `agent` | `fn agent(&self, agent_id: impl Into<String>) -> Agent` | Mint an agent identity (cheap). |
| `config` | `fn config(&self) -> &UnikoConfig` | The resolved configuration. |
| `purge` | `async fn purge(&self) -> Result<DeletionReport, UnikoError>` | Wipe all data (dev/test). |
| `shutdown` | `async fn shutdown(self) -> Result<(), UnikoError>` | Drain workers and close. Needs sole ownership — drop agents/sessions first. |

**`UnikoBuilder`** (chainable): `.path(p)` / `.in_memory()` (location) · `.embedding(EmbeddingConfig)`
· `.llm(LlmSpec)` (enables `answer()`) · `.raw_config(UnikoConfig)` (full override) ·
`.streaming(bool)` (enables `submit`/`flush`) · `.scope_to_agent()` / `.scope(RecallScope)`
(read visibility) · `.extractor(Arc<dyn ModalityExtractor>)` (pluggable modality) · `.build()`.

**`LlmSpec`** constructors: `openai(alias, model_id, base_url: Option<&str>)` (reads
`OPENAI_API_KEY`) · `openai_with_key_env(alias, model_id, base_url, key_env)` ·
`mistralrs(alias, model_id)` (local model).

---

## Agent

An `Agent` binds an identity to the store. Reads can be scoped to what that agent may see.

```rust
let agent = memory.agent("assistant");
let bundle = agent.recall("what does the user prefer?").await?;
let answer = agent.answer("when is the deadline?").await?;   // needs .llm(...)
```

| Method | Signature | Description |
| --- | --- | --- |
| `recall` | `async fn recall(&self, query: &str) -> Result<ContextBundle, UnikoError>` | Ranked, prompt-ready context. |
| `recall_in` | `async fn recall_in(&self, query: &str, scope: Scope) -> Result<ContextBundle, UnikoError>` | Recall confined to a session/participant/time scope. |
| `answer` | `async fn answer(&self, question: &str) -> Result<Answer, UnikoError>` | Recall + the configured LLM. |
| `answer_in` | `async fn answer_in(&self, question: &str, scope: Scope) -> Result<Answer, UnikoError>` | Scoped `answer`. |
| `query` | `async fn query(&self, cypher: &str) -> Result<Vec<Record>, UnikoError>` | Read-only Cypher against the graph. |
| `query_in` | `async fn query_in(&self, cypher: &str, scope: &Scope) -> Result<Vec<Record>, UnikoError>` | Binds the scope's allow-set as `$allow`. |
| `session` | `fn session(&self, session_id: impl Into<String>) -> Session` | Open a conversation thread. |
| `data` | `fn data(&self) -> Data<'_>` | Addressed retrieval — see [Data](#data-retrieval). |
| `goals` | `fn goals(&self) -> Goals<'_>` | Goal/task lifecycle — see [Goals](#goals-tasks). |
| `delete_session` | `async fn delete_session(&self, session_id: &str) -> Result<DeletionReport, UnikoError>` | Delete a whole conversation. |
| `forget_participant` | `async fn forget_participant(&self, participant_id: &str) -> Result<DeletionReport, UnikoError>` | GDPR erasure of a person's data. |
| `agent_id` | `fn agent_id(&self) -> &str` | The bound participant id. |

**Logic surface:** `define_rule(name, source)` → `NodeId` · `run_rule(name, &cols, params)` →
`Vec<Record>` · `assume(block) -> AssumeBuilder` (what-if, rolled back) ·
`abduce(program, params) -> AbductionResult`. See [Reasoning with Locy](../guides/reasoning-with-locy.md).

---

## Session & Turn

A `Session` is a conversation thread. `observe` is durable (commits before returning); the streaming
verbs need `.streaming(true)`.

```rust
let mut session = agent.session("chat-42");
let result = session
    .observe(Turn::new("alice", "here's the spec").attach(IngestSource::path("spec.pdf")))
    .await?;
session.ingest(IngestSource::path("handbook.md")).await?;   // standalone document
```

| Method | Signature | Description |
| --- | --- | --- |
| `observe` | `async fn observe(&mut self, turn: Turn) -> Result<ObserveResult, UnikoError>` | Ingest a conversational turn (+ attachments); read-after-write. |
| `submit` | `async fn submit(&self, turn: Turn) -> Result<(), UnikoError>` | Streaming fire-and-forget turn. |
| `submit_source` | `async fn submit_source(&self, source: IngestSource) -> Result<(), UnikoError>` | Streaming blob. |
| `flush` | `async fn flush(&self) -> Result<(), UnikoError>` | Barrier: drain the streaming pipeline. |
| `ingest` | `async fn ingest(&self, source: IngestSource) -> Result<IngestOutcome, UnikoError>` | Load a standalone document. |
| `summarize` | `async fn summarize(&self) -> Result<Option<NodeId>, UnikoError>` | Build/refresh the session Summary. |
| `forget_turn` | `async fn forget_turn(&self, message_id: &str) -> Result<DeletionReport, UnikoError>` | Soft-redact (hidden from recall, lineage kept). |
| `delete_turn` | `async fn delete_turn(&self, message_id: &str) -> Result<DeletionReport, UnikoError>` | Hard-delete + cascade. |
| `delete_document` | `async fn delete_document(&self, artifact_id: &str) -> Result<DeletionReport, UnikoError>` | Delete an ingested document. |

**`Turn`** (builder): `Turn::new(sender, content)` · `.id(message_id)` (idempotency) ·
`.content_type(s)` · `.at(DateTime<Utc>)` · `.addressed_to(Vec<String>)` ·
`.metadata(k, serde_json::Value)` · `.attach(IngestSource)` · `.attachments(iter)`.

**`ObserveResult`** `{ message: AtomicIngestResult, attachments: Vec<IngestOutcome> }` — each
attachment's `IngestOutcome::Artifact(_).artifact_id` is the handle for later retrieval.

---

## Recall results

`recall`/`recall_in` return a `ContextBundle` of ranked items, each carrying **what** it is (`kind`)
and **where** it came from (`sources`).

```rust
for item in &bundle.items {
    println!("[{:?}] {}", item.kind, item.content);
    for src in &item.sources { /* RecallSource::Message | Attachment | Document */ }
}
```

| Type | Shape | Description |
| --- | --- | --- |
| `ContextBundle` | `{ items: Vec<RecallItem>, total_tokens, coverage, phase1_only, phase2_only }` | Ranked result set + accounting. |
| `RecallItem` | `{ node_id: NodeId, kind: RecallKind, score: f64, content: String, sources: Vec<RecallSource> }` | One recalled node. `primary_source()` returns the first. |
| `RecallKind` | `Chunk · Observation · Fact · Procedure · Topic · Episode · Message · Other` | What it is. `is_derived()` ⇒ true for Fact/Procedure/Topic/Observation; `tier()` ⇒ its `RecallTier`. |
| `RecallSource` | `Message { message_id, chunk_id? }` · `Attachment { message_id, artifact_id, chunk_id? }` · `Document { artifact_id, chunk_id? }` | Lineage — 1 for a chunk, many for a Fact. |
| `RecallTier` | `Semantic · Procedural · Episodic · KnowledgeBase · Provenance` | Scoring weight band. |

**Scoping (`Scope`)** — chainable, passed to the `_in` variants: `.sessions([...])` ·
`.participants([...])` · `.since(DateTime)` · `.until(DateTime)` · `.as_viewer(Viewer)`.
`Dimensions` holds the resolved filters; `ViewerScope` is `Unrestricted | As(Viewer)`.

!!! warning
    Unscoped reads default to `ViewerScope::Unrestricted` — recall does **not** filter
    Fact/Observation visibility unless you scope to a viewer (`recall_in(.., Scope::default().as_viewer(v))`
    or build with `.scope_to_agent()`).

---

## Answer

`agent.answer(...)` returns an `Answer` — the generated text plus the grounding context, citable.

```rust
let ans = agent.answer("when is the deadline?").await?;
println!("{}", ans.text);
for src in ans.citations() { /* deduped grounding sources */ }
```

| Field / method | Type | Description |
| --- | --- | --- |
| `text` | `String` | The synthesized answer. |
| `context` | `ContextBundle` | The recall bundle that grounded it (items carry `kind` + `sources`). |
| `model` / `input_tokens` / `output_tokens` | `Option<…>` | Generation telemetry. |
| `recorded_episode` | `Option<NodeId>` | Set when query-episode recording was opted in and succeeded. |
| `citations()` | `fn citations(&self) -> Vec<&RecallSource>` | Deduped grounding sources, in bundle order. |

!!! note
    `citations()` reports the grounding context (what the model was shown), not model-attributed
    citations — the model isn't asked which items it used.

---

## Data — retrieval

`agent.data()` dereferences the ids that recall hands back.

```rust
let view = agent.data().artifact("spec.pdf").await?;        // Option<ArtifactView>
let bytes = agent.data().artifact_bytes("spec.pdf").await?; // Option<Vec<u8>>
let msg = agent.data().message("m-12").await?;              // Option<MessageView>
```

| Method | Signature | Description |
| --- | --- | --- |
| `message` | `async fn message(&self, message_id: &str) -> Result<Option<MessageView>, UnikoError>` | A turn: sender, content, session, recipients, attachments. |
| `artifact` | `async fn artifact(&self, artifact_id: &str) -> Result<Option<ArtifactView>, UnikoError>` | Metadata + reassembled text + the message it was attached to. |
| `artifact_bytes` | `async fn artifact_bytes(&self, artifact_id: &str) -> Result<Option<Vec<u8>>, UnikoError>` | The original blob. |

**`MessageView`** `{ message_id, sender_id, content, timestamp, session_id, addressed_to: Vec<String>,
attachments: Vec<String> }` · **`ArtifactView`** `{ artifact_id, kind, mime: Option<String>,
path: Option<String>, text: String, attached_to_message: Option<String> }`.

---

## Goals & tasks

`agent.goals()` is the goal/task lifecycle surface — create, transition, read sliced by phase, and
expand a goal's working context.

```rust
let g = agent.goals().create(CreateGoalParams { title: "Ship the release".into(), ..Default::default() }).await?;
agent.goals().complete("goal-1", Some(serde_json::json!({ "shipped": true }))).await?;
let active = agent.goals().active().await?;              // Vec<GoalView>
let ctx = agent.goals().context("goal-1").await?;        // Option<GoalContext>
```

| Group | Methods |
| --- | --- |
| **Read** | `all()` · `in_phase(GoalPhase)` · `active()` · `planned()` · `completed()` · `get(id)` · `tasks()` · `tasks_in(TaskPhase)` · `tasks_of(goal_id)` · `context(goal_id)` |
| **Transition** (→ `Result<bool>`, `false` = unknown id) | `start(id)` · `complete(id, Option<Json>)` · `abandon(id)` · `set_status(id, &str)` · `start_task` · `complete_task` · `block_task` · `set_task_status` |
| **Create** | `create(CreateGoalParams) -> NodeId` · `create_task(CreateTaskParams) -> NodeId` |

**`GoalView`** `{ goal_id, title, description?, status, phase: GoalPhase, created_at?, deadline?,
completed_at?, metrics: Option<Json> }` · **`TaskView`** `{ task_id, title, description?, status,
phase: TaskPhase, priority?, goal_id?, created_at?, completed_at? }` · **`GoalContext`**
`{ goal: GoalView, tasks: Vec<TaskView>, sessions, recent_messages, facts, entities }`.

**`GoalPhase`** `Planned · Active · Completed · Abandoned` · **`TaskPhase`** `Planned · Active ·
Completed · Blocked` — derived from the free-form `status` (+ `completed_at`); the raw `status` stays
exposed.

!!! note "Participant must exist"
    `create`/`create_task` require the agent's `Participant` to exist — `observe()` creates it on
    first sight. A pure-planning agent should observe at least once, or seed the participant.

---

## Ingest sources

A single blob type drives both `Turn::attach` and `Session::ingest`.

| Constructor | Description |
| --- | --- |
| `IngestSource::text(s)` / `::bytes(b)` / `::path(p)` | Build from text, bytes, or a file. |
| `.with_mime(m)` / `.with_id(id)` / `.with_path(p)` | Override MIME, set the artifact id, set a display path. |

`IngestOutcome` is `Artifact(ArtifactIngestResult)` · `Pdf(PdfIngestResult)`; the
result types carry `artifact_id` + node ids. Unsupported modalities return `Err(UnikoError::Unsupported(...))`. MIME is resolved explicit → magic-bytes → extension →
text, then routed (text/markdown/code/json → artifact, PDF → tiered extractor, image/audio/video →
a registered `ModalityExtractor`).

---

## Recording primitives (low-level)

Beyond `agent.goals()`, the subjective-state **free functions** are available at the `uniko_memory`
crate root for fine-grained recording. Each takes `(&kb, agent_id, params)` (except `assert_fact` /
`invalidate_fact`, which take no `agent_id`) and the Participant must already exist.

| Function | Returns | Description |
| --- | --- | --- |
| `create_goal` / `create_task` | `NodeId` | The primitives behind `agent.goals().create*`. |
| `assert_fact` | `FactUpsert` | Upsert a `(subject, predicate, object)` Fact. |
| `invalidate_fact` | `()` | Close a Fact's bitemporal validity interval. |
| `add_observation` | `NodeId` | An explicit Observation anchored to a Message. |
| `record_episode` | `NodeId` | An Episode (action + outcome + state). |
| `record_action` | `RecordActionResult` | An Action node; large output overflows to an Artifact. |

See [Agent Tools](../guides/agent-tools.md) for the parameter structs and edge wiring.

---

## Access-control policy

Facts/Observations carry a `visibility` (`public` / `private:{id}` / `team:{id}` / `org:{id}`).
Scoped reads filter automatically; you can also apply it directly.

| Symbol | Signature | Description |
| --- | --- | --- |
| `Viewer::new` | `async fn new(kb, participant_id) -> Result<Self, UnikoError>` | Resolve team/org memberships from the graph. |
| `Viewer::from_parts` | `fn from_parts(participant_id, teams, orgs) -> Self` | Build from already-resolved memberships. |
| `visibility_admits` | `fn visibility_admits(visibility: Option<&str>, viewer: &Viewer) -> bool` | Whether a tag admits the viewer. Unknown schemes fail closed. |

---

## Errors & ids

| Re-export | Use |
| --- | --- |
| `UnikoError` | The crate-wide error type every method returns. |
| `NodeId` | Internal node id (`i64`) returned by create/ingest. |
| `DeletionReport` | What a delete/forget changed. |
| `Value`, `Record` | Graph value type and a query result row (`query`/`run_rule`). |
| `FinalizeReport` | What `Session::finalize` built or refreshed. |
| `temporal_epoch_millis` | Epoch milliseconds for a datetime-shaped temporal `Value` — the supported way to read a BTIC/timestamp column out of a query row. |

---

## Engine internals

`Uniko` wraps the low-level engine, which remains public for embedded/advanced use. These are the
mechanisms the facade exists to hide — reach for them only when you need to.

| Symbol | Where | Wrapped by |
| --- | --- | --- |
| `KnowledgeBase` (`uniko_store`) | the graph + model runtime handle | `Uniko` |
| `PipelineSystem` (`uniko_memory`) | ingest/consolidation workers, channels, circuit breaker | `Uniko` (streaming) |
| `recall(&kb, q, &RecallConfig)` | the 3-phase cascade free function | `Agent::recall` |
| `answer_query(&kb, q, &cfg, generator, record)` | recall + a generator closure (returns `GeneratedAnswer`) → `Answer` | `Agent::answer` |
| `ingest_message_atomic(&kb, &msg, &mut ctx)` | the single-transaction ingest path | `Session::observe` |

See [Pipelines](../pipelines/index.md) for the engine's ingest/consolidation/recall internals.

---

## See also

- [Recall & Retrieval](../guides/retrieval.md) — `recall`/`answer`, provenance, and `data()`.
- [Agent Tools](../guides/agent-tools.md) — goals/tasks and the recording primitives.
- [Architecture](../concepts/architecture.md) — how the layers fit together.
- [Schema](schema.md) — the node and edge catalog.
