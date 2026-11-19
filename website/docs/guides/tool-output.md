# Tool Output

Most of what an agent knows precisely comes back from its tools: an order list, a query result, a
service log, a test run. That output is structured, and a record in it rarely means much on its own.
An order's `amount` needs the `currency` declared once in `meta`; a CSV row needs its header; a
failing test needs the commit the suite ran on, which may appear only in the tool's arguments.
Chunk such a payload like prose and each chunk keeps the record but loses that context, and a
similarity search returns whichever chunk shares the most words.

uniko treats tool output as its own kind of memory:

- each call is **recorded as an `Action`** with its tool, arguments and status, linked to the turn
  that made it;
- its result is **stored verbatim** as an `Artifact` of kind `tool_output`, and **chunked along its
  own structure**, each chunk carrying a header of what it inherits from outside itself;
- recall gains a **lexical channel** that matches identifiers and figures whole;
- **read tools** let the agent search, outline and read a result deterministically.

## Recording a call

Add a [`ToolCall`](../reference/api.md) to the turn that made it. The call commits in the same
transaction as the turn: if the turn fails, neither is written.

=== "Rust"

    ```rust
    use serde_json::json;
    use uniko_api::tools::{ToolCall, Turn};

    let call = ToolCall::new("list_orders")
        .call_id("call_01")                               // the host's own id
        .arguments(json!({"customer_id": "C-1042", "page": 2}))
        .json_output(orders)                              // what came back
        .status("ok");

    let result = session
        .observe(Turn::new("assistant", "Checking the orders API.").tool_call(call))
        .await?;
    let artifact_id = result.tool_calls[0].artifact_id.clone();
    ```

=== "Python"

    ```python
    import uniko

    call = (
        uniko.ToolCall("list_orders")
        .call_id("call_01")
        .arguments({"customer_id": "C-1042", "page": 2})
        .json_output(orders)
        .status("ok")
    )
    result = session.observe_sync(
        uniko.Turn("assistant", "Checking the orders API.").tool_call(call)
    )
    artifact_id = result.tool_calls[0].artifact_id
    ```

A call recorded apart from any turn uses `Session::observe_tool` (`session.observe_tool` in Python),
naming its triggering message with `triggered_by` when there is one.

### Who performed the call

A call inside a turn is performed by the turn's sender, and one recorded with `observe_tool` by the
session's agent. When someone else made the call, name them with `.performer(...)`. For example, a
user's turn can carry a call the reporting agent made:

```rust
let call = ToolCall::new("search").call_id("call").text_output("Evidence").performer("reporter");
session.commit_unit(vec![Turn::new("user", "Question").id("question").tool_call(call)]).await?;
// Message -[:SENT_BY]-> user;  Action -[:PERFORMED_BY]-> reporter
```

The turn keeps its sender. The performer is created as kind `agent` if it doesn't exist yet. It's
also part of the call's replay key: recording the same call id again under a different performer is
refused with `UnikoError::IdConflict`, rather than replaying with the old `PERFORMED_BY` edge. Naming
the default performer explicitly is the same as naming none.

What gets written:

| Node / edge | What it holds |
|---|---|
| `Action` | `action_type = "tool_call"`, `tool_name`, `input` (the arguments), `status`, `error`, timing, and `payload_hash` |
| `TRIGGERED_BY` → `Message` | the turn that made the call |
| `IN_SESSION` → `Session`, `PERFORMED_BY` → `Participant` | the session, and the turn's sender (or the call's `performer`) |
| `NEXT_ACTION` | from the session's previous call (or the one named with `after`) |
| `PRODUCED` → `Artifact {kind: "tool_output"}` | the result, verbatim, with its chunks |

The call id is the Action's `action_id`. Recording the same id again with the same result is a no-op
(`was_replay` is true); with a different result it is refused with `UnikoError::IdConflict`. Without
a call id, one is derived from the session, the turn, the tool, its arguments and its result, so an
identical retry is still recognised.

A call takes the turn's [visibility](retrieval.md) unless it sets its own, and its result's chunks
are scoped with it.

!!! note "Results as plain turns"
    A turn with `content_type` `"tool_result"` (or `json`, `csv`, a JSON or CSV MIME) is also chunked
    along its structure, but it has no tool name or arguments to carry, and it stays a `Message`.
    Prefer `ToolCall` when the call's arguments matter, which they usually do.

## How a result is chunked

The payload's format is detected from its content — JSON, NDJSON, CSV/TSV, a Markdown table, a
unified diff, a log, or text — and it is split along that structure:

| Format | A chunk is | Its address |
|---|---|---|
| JSON | one value, or a run of array elements, found by descending into whatever is too big for one chunk | `/data/items[3-9]`, `/meta` |
| NDJSON | a run of records | `records 1-40` |
| CSV, Markdown table | a run of rows; the header is context, not text | `rows 10-19` |
| Diff | a hunk, or part of one | `hunk 2, lines 40-61` |
| Log, text | a run of whole lines | `lines 120-180` |

Every chunk's `text` is an exact slice of the payload — never rewritten. Its **context header** is a
few lines, capped at `chunk_context_max_tokens` (default 64):

```text
tool: list_orders {"customer_id":"C-1042","page":2}
at: /data/items[3-9]
meta.currency=EUR; meta.amount_unit=cents; meta.as_of=2026-10-03; meta.page=2
```

For JSON the inherited values are the scalar fields of every object enclosing the chunk, and of their
small sibling objects, nearest first; for a table, its column names. The header is embedded together
with the chunk's text, and indexed for search, so a question about the amount's currency finds the
chunk that holds both. Structured data gets no sentence NLP, dialogue acts or observations; a prose
result, such as a build summary, still does.

## Recall

Recall's Phase 3 gains a lexical channel over each chunk's normalised `search_text`: thousands
separators are removed and identifiers kept whole as well as split into parts, so `ORD-48213`,
`req-9f3a2b1c`, `1,234.50` and `src/pay.rs` match as written. The channel searches the query's
**literals** — quoted phrases, identifiers, figures, paths, acronyms — and they count double in
fusion; a query without literals is searched by its keywords. Turn it off with
`recall_lexical_enabled: false`.

A recalled tool-output chunk carries its header and address:

```rust
for item in &bundle.items {
    if let Some(RecallSource::Tool { call_id, tool_name, .. }) = item.primary_source() {
        println!("{tool_name:?} {call_id}: {} @ {:?}", item.content, item.address);
        println!("  inherits: {:?}", item.context);
    }
}
```

`RecallItem.token_count` is the item's real size, and the bundle respects `recall_token_budget` in
those tokens.

## Reading around a hit

Recall ranks; the read tools read. `agent.sources()` returns a handle whose calls are deterministic —
the same store and arguments give the same answer, so a sequence of them can be replayed — and apply
the same visibility as recall.

=== "Rust"

    ```rust
    use uniko_api::tools::{ReadRange, SearchOptions};

    let sources = agent.sources();
    let hits = sources.search("status of ORD-48213", &SearchOptions::default().limit(5)).await?;
    let hit = &hits[0];                       // source, address, verbatim text, context, call id

    let outline = sources.outline(&hit.source, 3, 0).await?;           // keys, arrays, chunks
    let record = sources
        .read(&hit.source, &ReadRange::Address("/data/items/17".into()), 2_000)
        .await?;                                                       // exact JSON value
    let call = sources.tool_call(hit.call_id.as_deref().unwrap()).await?;  // tool, arguments, timing
    ```

=== "Python"

    ```python
    src = agent.sources
    hits = src.search_sync("status of ORD-48213", limit=5)
    hit = hits[0]
    outline = src.outline_sync(hit.source_kind, hit.source_id)
    record = src.read_sync(hit.source_kind, hit.source_id, address="/data/items/17")
    call = src.tool_call_sync(hit.call_id)
    ```

| Call | Returns |
|---|---|
| `search(query, options)` | lexical hits, each with its source, chunk, address, byte span, verbatim text, context header, session and producing call; ordered by score, then chunk id |
| `outline(source, depth, max_entries)` | for JSON, its objects and arrays down to `depth`; otherwise its chunks with their addresses |
| `read(source, range, max_tokens)` | the verbatim text at an address (`/data/items[3-9]`, `rows a-b`, `lines a-b`, `records a-b`), a byte or line range, or a chunk range — cut at `max_tokens` on a line break, with the rest of the range in `next` |
| `tool_call(call_id)`, `tool_calls(session, tool, limit)` | the call's tool, arguments, status, timing, triggering message, session and result artifact |

A source is an artifact, a message or a session, so the same tools work on ingested documents and long
turns as well as tool results.

## Upgrading a store

The new properties and indexes are added when an existing store is opened. Chunks written before the
upgrade have no `search_text`, so the lexical channel does not see them until it is filled:

```rust
let filled = memory.backfill_search_text().await?;   // `memory.backfill_search_text_sync()` in Python
```

Existing chunk embeddings are kept; only new chunks are embedded with their context header.

!!! warning "Full-text indexes from before uni-db 5.2"
    uni-db 5.2 moves to Lance 14, whose English stop-word list no longer drops negations and
    contractions (`no`, `not`, `nor`, `don't`, …). A full-text index built by an earlier release has
    no postings for them, so content indexed before the upgrade is not found by those words, while
    content added since is. Nothing errors: uni-db names the stale indexes in a warning when the
    store opens. uniko does not rebuild them yet, nor expose uni-db's switch for it; until it does,
    open the store once directly with uni-db and run `db.indexes().rebuild(label)` for each label
    named, or open it with `index_rebuild.auto_rebuild_enabled`. `Chunk.search_text` keeps stop
    words, so the lexical index described here is unaffected.
