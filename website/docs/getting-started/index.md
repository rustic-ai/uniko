# Getting Started

By the end of this section you will have a working memory layer running inside your own Rust
process: a `Uniko` instance open on disk, a conversation observed at **$0 in LLM cost**, and a
question answered from compiled knowledge — with no LLM in the recall path. It takes about fifteen
minutes, most of which is a one-time model download.

uniko links into your process like SQLite. There is no service to deploy, no vector store to keep
in sync, no network hop between your agent and its memory. You feed it `Turn`s; every turn is
compiled into a typed knowledge graph with full provenance — `Entity` and `Observation` on the
write path, then `Fact`, `Procedure` and `Topic` from consolidation — and answers queries
against that compiled knowledge.

<div class="feature-grid" markdown>
<div class="feature-card" markdown>
### [Installation](installation.md)
Add uniko to your Cargo workspace and pull in the uni-db engine it builds on.
</div>
<div class="feature-card" markdown>
### [Quick Start](quickstart.md)
Build a `Uniko` instance, observe a `Turn`, and run your first recall — end to end in Rust.
</div>
</div>

!!! note "uniko is a Rust library"
    You use uniko by depending on its crates and calling its async APIs from Rust. Ingest runs
    entirely locally — entity and observation extraction goes through an ONNX model cascade, with
    **zero LLM tokens per message** by default.

## How to get started (5–15 minutes)

1. **[Install](installation.md)** — add the `uniko-api` crate. Nothing to deploy. uniko targets the
   Rust 2024 edition on the stable toolchain.
2. **[Quick Start](quickstart.md)** — observe a few `Turn`s and answer a question end to end,
   seeing the compile-once / query-forever flow in action.
3. **[Learn the model](../concepts/architecture.md)** — understand how `Message`s become
   `Observation`s, `Fact`s, and `Procedure`s, and how the recall cascade assembles a
   `ContextBundle`.

!!! tip "One runtime, many knowledge bases"
    A single `KnowledgeBase` is enough to follow the Quick Start. When you scale to many
    KnowledgeBases in one process, they share a single ONNX `ModelRuntime` via
    `KnowledgeBase::build_shared_runtime` and `open_with_runtime` — model weights stay resident
    exactly once, so VRAM use stays flat as you add tenants.

## Where to go next

- **[Concepts: Architecture](../concepts/architecture.md)** — the layered crate stack and the
  P1–P7 pipelines that turn messages into knowledge.
- **[Concepts: Memory Model](../concepts/memory-model.md)** — the five memory types (working,
  episodic, semantic, procedural, meta) and the graph nodes behind them.
