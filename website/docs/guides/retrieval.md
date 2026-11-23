# Recall & Retrieval

Reading memory back is three altitudes — and every result is traceable to where it came from. This
guide is the *usage* side; for the engine internals of the cascade see
[Pipelines: Recall](../pipelines/recall.md).

## Three altitudes

| You want… | Use | Returns |
|---|---|---|
| Relevant context for a prompt | `agent.recall(q)` | `ContextBundle` (ranked items) |
| A finished answer | `agent.answer(q)` | `Answer` (generated text + grounding) |
| Exact nodes you specify | `agent.query(cypher)` | `Vec<Record>` (raw rows) |

```rust
let bundle = agent.recall("what's the deadline?").await?;        // ranked context
let answer = agent.answer("when is the deadline?").await?;       // needs .llm(...)
let rows   = agent.query("MATCH (m:Message) RETURN m.content").await?;  // read-only
```

`recall`/`answer`/`query` each have an `_in` variant taking a [`Scope`](#scoping) to confine the read
to a session, participant, or time window.

## What a recalled item carries

Each `RecallItem` answers two orthogonal questions: **what** it is (`kind`) and **where** it came from
(`sources`).

```rust
for item in &bundle.items {
    println!("[{:?}{}] {}",
        item.kind,
        if item.kind.is_derived() { " derived" } else { "" },
        item.content);
    for src in &item.sources {
        match src {
            RecallSource::Message { message_id, .. }            => println!("   ← {message_id}"),
            RecallSource::Attachment { artifact_id, message_id, .. } => println!("   ← {artifact_id} (in {message_id})"),
            RecallSource::Document { artifact_id, .. }          => println!("   ← {artifact_id}"),
            RecallSource::Session { session_id, .. }            => println!("   ← session {session_id}"),
            RecallSource::Tool { call_id, artifact_id, .. }     => println!("   ← {artifact_id} (tool call {call_id})"),
        }
    }
}
```

- **`kind`** — `Chunk` / `Observation` / `Fact` / `Procedure` / `Topic` / `Episode` / `Message` /
  `Other`.
  `kind.is_derived()` distinguishes synthesized memory (Fact/Procedure/Topic/Observation) from raw
  input (Chunk/Message) — *"the system concluded X"* vs *"alice literally said X."*
- **`sources`** — the lineage. A chunk has one source; a derived `Fact` lists the messages/attachments
  its supporting observations came from; Topics and Procedures carry none. A chunk built by
  `Session::finalize()` traces to its `Session`, and a tool call's result to its `Tool` call.
  `item.primary_source()` returns the first.

!!! note "Provenance is captured, not inferred"
    Sources are the graph's own edges (`HAS_CHUNK`, `ATTACHED_TO`, `OBSERVED_IN`, `SUPPORTED_BY`,
    `PRODUCED`)
    projected onto the result — so a chunk from a shared document reports the `Attachment` it came
    from, and the message that shared it.

## Cited answers

`agent.answer(...)` returns an `Answer` that carries the grounding context, so answers are citable
without a second call — attribution is pure data:

```rust
let ans = agent.answer("when is the deadline?").await?;
println!("{}", ans.text);
for src in ans.citations() {                 // deduped grounding sources
    // "per spec.pdf, shared in m-att" / "alice said in m-12"
}
println!("(model {:?}, {} out-tok)", ans.model, ans.output_tokens.unwrap_or(0));
```

`citations()` reports the grounding context — what the model was shown — not model-attributed
citations. `ans.context` is the full ranked bundle (`items`, `coverage`, `total_tokens`).

## Dereferencing a source — `agent.data()`

`sources` hand back *ids*; `agent.data()` turns them into content. Recall is the index lookup; `data`
is the document fetch — keep them separate so you can cite over many sources but fetch only the few
you open.

```rust
// the turn a source points at
let msg = agent.data().message("m-12").await?;          // Option<MessageView>
// the attachment/document: metadata + reassembled text
let art = agent.data().artifact("spec.pdf").await?;     // Option<ArtifactView>
// the original bytes (e.g. to re-open the PDF)
let bytes = agent.data().artifact_bytes("spec.pdf").await?;  // Option<Vec<u8>>
```

`MessageView` gives `{ message_id, sender_id, content, content_type, timestamp, session_id,
addressed_to, attachments, category, visibility }`; `ArtifactView` gives `{ artifact_id, kind, mime,
path, text, attached_to_message }`. Unknown ids return `Ok(None)`.

The id is the one `observe`/`ingest` handed you:

```rust
let result = session.observe(Turn::new("alice", "see attached").attach(IngestSource::path("spec.pdf"))).await?;
if let IngestOutcome::Artifact(a) = &result.attachments[0] {
    let view = agent.data().artifact(&a.artifact_id).await?;   // round-trips
}
```

## Scoping

Confine any read to a session, participant, or time window with `Scope` + the `_in` variants:

```rust
use chrono::{Duration, Utc};

let scope = Scope::default()
    .sessions(["chat-42"])
    .participants(["alice"])
    .since(Utc::now() - Duration::days(7));

let bundle = agent.recall_in("deadline", scope.clone()).await?;
let rows   = agent.query_in("MATCH (n:Message) WHERE id(n) IN $allow RETURN n", &scope).await?;
```

`recall_in`/`answer_in` apply the scope (and viewer visibility) for you; `query_in` binds the
in-scope node-ids as the `$allow` list for your Cypher to filter on. Build the instance with
`.scope_to_agent()` to default every read to the agent's own visibility.

## See also

<div class="feature-grid" markdown>
<div class="feature-card" markdown>
### [Pipelines: Recall](../pipelines/recall.md)
The 3-phase cascade, coverage gates, and tier weights behind `recall`.
</div>
<div class="feature-card" markdown>
### [API Reference](../reference/api.md)
Full signatures for `recall`/`answer`/`query`, `Answer`, `Data`, and the recall types.
</div>
</div>
