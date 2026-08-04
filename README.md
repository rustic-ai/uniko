# uniko

**Embedded, Rust-native cognitive memory for AI agents — compiled knowledge in, fast recall out, zero infrastructure.**

License: Apache-2.0 · Docs: https://rustic-ai.github.io/uniko/ · Crates: uniko-api, uniko-memory, uniko-store (crates.io)

---

## What is uniko

AI agents are stateless. They can pull text snippets out of a vector store, but they can't
track who said what across sessions, notice when a fact changes, learn reusable procedures
from repeated experience, or explain why they believe something. Existing memory systems each
solve a piece — and all of them bolt memory onto the agent as **external infrastructure**
(Neo4j, Qdrant, PostgreSQL) and call an LLM on every ingested turn.

uniko takes a different shape. Memory is a typed knowledge graph organized around
communication, linked into your agent process like SQLite — no separate services, no network
hop, no vector store to keep consistent. **Messages between participants are the atomic unit,
and everything else derives from them with full provenance:**

```
Messages -> Entities / Observations -> Facts -> Procedures
who said what  ->  what was observed  ->  what was learned  ->  what works
```

The core insight is *when* the work happens. Raw messages are like source code; consolidation
"compiles" them into reusable knowledge. Extraction runs a **local ONNX model cascade**
(kniv-deberta, INT8-quantized) at write-time — a single encoder pass producing POS, NER, SRL,
DEP, and CLS labels, with **no LLM called per message**. Recall then queries the compiled
Entities, Observations, and Facts instead of re-deriving them — so there is **no LLM in the
recall hot path**. Compile once, query forever.

## Why uniko

- **Zero infrastructure.** A single in-process database (uni-db: graph + vector + full-text +
  Locy logic, in one embedded engine). Nothing to deploy or operate.
- **No LLM in the hot path.** Write-time extraction is local ONNX only; recall hits compiled
  knowledge. Ingest cost is predictable, offline-capable, and $0 in API spend by default.
- **Interaction-first schema.** Message / Session / Participant are first-class. Provenance
  (`who said what`) is structural, not reconstructed.
- **Goal-oriented working memory.** Live graph traversal from Goal -> Task -> Session ->
  Messages / Facts / Entities, assembled per query under a token budget.
- **Formal reasoning via Locy.** Database-native rule execution (the `sequence_detector` rule
  drives procedure promotion today) rather than an LLM at query time.

## Benchmarks

Full LoCoMo10 (10 conversations, 1,986 questions). Numbers quoted exactly from the source
benchmark reports.

| Metric | uniko |
|---|---|
| LLM-judge (gemini-3.1) | **81.2%** |
| Retrieval hit | 85.6% |
| F1 | 0.321 |
| Total LLM cost (1,986 q, incl. judging) | **$3.55** |
| Ingest (5,882 turns, local ONNX only) | **7.5 min at $0** (~76 ms/turn) |
| Mean Q&A latency | **4.04s** (fastest of 6 systems compared) |

The full 5,882-turn corpus ingests in 7.5 minutes with zero LLM calls. Mean query wall-time of
4.04s is the fastest of six systems measured (vs Mem0, Graphiti, Cognee, RAG, Full-Context),
all while running fully in-process on consumer hardware. See the
[benchmarks](https://rustic-ai.github.io/uniko/benchmarks/) for the full comparison.

## Install

uniko is a Rust library. Requires **Rust >= 1.91** (edition 2024). Add the crates you need
from crates.io:

```toml
[dependencies]
uniko-api = "*"      # public facade — re-exports the surface below
uniko-memory = "*"   # recall cascade, consolidation, orchestration
uniko-store = "*"    # graph storage / search over uni-db
uniko-extract = "*"  # NER / observations / chunking / atomic ingest
```

The `uni-db` dependency (`^3` — the latest 3.x) is pulled in transitively from
crates.io — there is nothing external to install or run.

## Quick start

A `KnowledgeBase` is the single in-process handle over uni-db. Open one, ingest
`IngestMessage`s atomically, then `recall` a `ContextBundle` for a query — no LLM in the recall
path.

```rust
use uniko_store::KnowledgeBase;
use uniko_store::config::UnikoConfig;
use uniko_extract::ingest::atomic::ingest_message_atomic;
use uniko_extract::ingest::context::SessionContext;
use uniko_memory::recall::{recall, RecallConfig};
use uniko_pipes::types::IngestMessage;
use chrono::Utc;
use std::collections::HashMap;

async fn demo() -> uniko_store::Result<()> {
    // 1. Open an embedded knowledge base — no external services.
    let kb = KnowledgeBase::open("./agent-memory", UnikoConfig::default()).await?;

    // 2. Ingest a message. Local extraction (NER + NLP cascade + observations)
    //    runs first, then one atomic transaction writes the Message, Entities,
    //    Observations, edges, and chunks — idempotent on `message_id`.
    let mut session_ctx = SessionContext::new("session-1".to_string(), 0);
    let msg = IngestMessage {
        message_id: "m-1".to_string(),
        content: "Caroline researched coral reef restoration in Belize.".to_string(),
        content_type: "text".to_string(),
        sender_id: "melanie".to_string(),
        session_id: "session-1".to_string(),
        addressed_to: None,
        timestamp: Utc::now(),
        metadata: HashMap::new(),
    };
    ingest_message_atomic(&kb, &msg, &mut session_ctx).await?;

    // 3. Recall compiled knowledge — no LLM in this path.
    let bundle = recall(&kb, "What did Caroline research?", &RecallConfig::default()).await?;
    for item in &bundle.items {
        println!("{:?}: {}", item.kind, item.content);
    }
    Ok(())
}
```

`recall` is a three-phase cascade (compiled Facts/Topics/Procedures -> hybrid vector + BM25
over Episodes/Observations/Messages -> Chunk/Artifact fallback), gated by coverage thresholds
and filtered by visibility policy. See the
[quick start guide](https://rustic-ai.github.io/uniko/getting-started/quickstart/) for the full
version.

## Documentation

Full documentation lives at **https://rustic-ai.github.io/uniko/**:

- [Getting Started](https://rustic-ai.github.io/uniko/getting-started/installation/) — add
  uniko and ingest your first messages.
- [Concepts](https://rustic-ai.github.io/uniko/concepts/architecture/) — the cognitive model,
  the typed graph schema, and the recall cascade.
- [Pipelines](https://rustic-ai.github.io/uniko/pipelines/recall/) — ingest, consolidation, and
  recall internals.
- [Benchmarks](https://rustic-ai.github.io/uniko/benchmarks/) — full LoCoMo results and the
  head-to-head cost / latency comparison.
- [Python SDK](https://rustic-ai.github.io/uniko/python/) — the same engine from Python:
  async-first, with blocking `*_sync` twins.
- [Why uniko](https://rustic-ai.github.io/uniko/why-uniko/) — differentiators in depth.

## Project status

uniko is a **shipped Rust library** covering Phases 1–3: the typed schema (24 node types, 53
edge types), atomic ingest with local NLP extraction, async consolidation
(observations -> Facts), three-phase recall, procedure promotion, topic detection, and formal
Locy reasoning (`sequence_detector` live).

**Python bindings (alpha):** an async-first PyO3 SDK over the same in-process engine ships in
`bindings/uniko-py` — full async surface plus blocking `*_sync` twins and type stubs. Build from
source with `maturin` (no prebuilt wheels yet); see the
[Python SDK docs](https://rustic-ai.github.io/uniko/python/).

**Not yet available** (do not depend on these): an HTTP / MCP server, a CLI, prebuilt Python
wheels, and Phase 4–6 features (multimodal ingest, rule induction, MCTS planning, cross-agent
sharing). See the docs for the honest roadmap.

## License

Apache-2.0 © Dragonscale Industries Inc.

Security: security@dragonscale.ai · Conduct: conduct@dragonscale.ai · General:
dev@dragonscale.ai
