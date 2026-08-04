# The uniko Blackbook

*An embedded, Rust-native cognitive-memory engine that compiles conversations into a queryable knowledge graph.*

**Version 0.2.0 · Generated 2026-08-03**

---

## Executive Summary

uniko is an embedded, Rust-native cognitive-memory library for AI agents. It links into a host process like SQLite — no external services, no sidecar databases, no network hops — and turns raw conversation and agent activity into a single typed property graph that spans graph structure, vector embeddings, full-text (BM25) indexes, and the Locy logic runtime, all inside one `uni-db` store. Where the conventional stack stitches together a vector store, a graph database, a rules/LLM layer, and consistency glue — and calls an LLM on every ingested turn — uniko collapses all of that into one in-process engine and pays for enrichment exactly once, at write time.

The central thesis is **write-time compile: "compile once, query forever."** Raw messages are treated as source code; consolidation compiles them into reusable knowledge. Ingest runs a deterministic, LLM-free cascade — regex, tree-sitter, and a small quantized DeBERTa model (kniv-deberta, INT8) that emits POS/NER/SRL/DEP/CLS in a single shared-encoder ONNX pass — so ingestion is cheap, reproducible, and fully offline. Every message is written atomically: all CPU work and read-only lookups happen before a transaction opens, then one transaction commits the Message plus its edges, chunks, entities, and observations. Recall then queries the already-compiled Entities, Observations, and Facts rather than re-deriving them, so there is no LLM in the recall hot path.

Architecturally, uniko is a strict stack of layered crates. **L1 uniko-store** is the sole boundary to `uni-db` — it owns the schema (25 node labels, 54 edge types), vector/fulltext/hybrid/sparse/ColBERT search, graph traversal and Personalized PageRank, bitemporal Facts, and SSI-aware concurrency with striped locks. **L2 uniko-pipes** supplies content-free pipeline machinery (the `Step` trait, circuit breaker, dead-letter queue, cancellation, content taxonomy). **L3 uniko-extract** turns messages, artifacts, and PDFs into graph nodes via the NLP cascade and atomic ingest. **L4 uniko-memory** is the public `Uniko` facade — the 3-phase recall cascade, P4 fact consolidation with drift and contradiction detection, the async pipeline, and the agent-tool surface. **L5 uniko-cortex** promotes recurring successful episode sequences into Procedures (P5) and clusters co-occurring entities into Topics (P6). A thin `uniko-api` facade plus an async-first PyO3 Python SDK form the outermost, logic-free surface. Layer numbers denote cognitive altitude, not build order.

Knowledge in uniko is **derived, never merely stored**. Only Messages and Actions are observed directly; Entities, Observations, Facts, Topics, and Procedures are all derived and wired back to their evidence, so the schema itself *is* the provenance spine — every belief is walk-back-able to "why do we believe this?" Facts are bitemporal (a single BTIC valid-time interval with per-bound certainty), reinforced by Laplace-smoothed confidence, and subject to contradiction detection (F38) and entity drift (F39) that forces volatile-entity queries into deeper recall. Per-claim visibility scoping gates Fact and Observation access by viewer, team, and org.

The headline benchmark proof comes from the repository's own harness on LoCoMo10 (1,986 questions, gemini-3.1 judge, 22-core CPU + 8 GB consumer GPU): **81.2% LLM-judge accuracy, 85.6% retrieval hit rate, 0.321 token-F1** — competitive on quality — while ingesting 5,882 turns in **7.5 minutes at $0** (no LLM on the write path) and answering at **4.04 s mean latency**. Against graph-backed peers measured in the KTH dmas-memory study, uniko ingests 33–76× faster at zero token cost and posts the fastest end-to-end Q&A wall-time of six systems. These are self-measured internal figures, not a third-party leaderboard, and the cost/latency tables use the 1,540-question non-adversarial subset — the two question sets should not be conflated.

uniko is for founders and engineering leads shipping agents in their own process who want zero operational footprint, where ingest cost and offline capability matter, where conversation and provenance are central (tracking who-said-what across sessions and explaining beliefs), who organize memory around goals, and who prefer graph-native reasoning compiled at ingest over inference paid on every query. It ships today as six crates on crates.io (facade: `uniko-api = "0.2.0"`) and three interchangeable Python wheels (`uniko` CPU, `uniko-cuda`, `uniko-metal`). The Python SDK is alpha; HTTP/MCP servers, a CLI, multimodal ingest, rule induction, and cross-agent sharing are on the roadmap, not yet shipped.

---

## Table of Contents

1. Overview, Purpose & Use Cases
2. Architecture & Layered Design
3. Data Model & Knowledge Graph Schema
4. Ingestion & Extraction Pipeline (L3)
5. Storage, Search & Locy Reasoning (L1)
6. Memory, Recall & Consolidation (L4)
7. Higher Reasoning — Procedures & Topics (L5)
8. Public API & Usage (Rust + Python)
9. Benchmarks & Performance
10. Configuration, Build, Testing & Operations
11. Development, Contributing & Roadmap
- Appendix A: Glossary

---

## 1. Overview, Purpose & Use Cases

### 1.1 What uniko is, in one page

uniko is an **embedded, Rust-native cognitive-memory engine for AI agents**. It links into your host process like SQLite — no server to run, no cluster to operate, no external services to keep in sync. Inside that one process it runs a single [uni-db](https://crates.io/crates/uni-db) store that fuses a property graph, vector indexes, a BM25 full-text index, and the Locy logic runtime into one engine, and it turns raw conversation into a typed, queryable, reasoned-over knowledge graph.

Everything an agent "remembers" lives in **one property graph** — 25 node labels and 54 edge types (the source of truth is `crates/uniko-store/src/schema/constants.rs`: `labels::ALL` and `edges::ALL`). The organizing principle is that **knowledge is derived, never directly stored**: only `Message` and `Action` nodes are observed directly; every other node — `Entity`, `Observation`, `Fact`, `Topic`, `Procedure`, `Summary` — is *derived* from those observations and wired back to its evidence with graph edges. The schema *is* the provenance spine. You can always walk back from a belief to the message that produced it and answer "why do we believe this?"

The public entry point is one owning handle:

```rust
use uniko_memory::{LlmSpec, Turn, Uniko};

let memory = Uniko::builder()
    .path("./data/kb")                                        // or .in_memory()
    .llm(LlmSpec::openai("llm/default", "gpt-4o-mini", None)) // optional; enables answer()
    .scope_to_agent()                                         // access-control filtering
    .build().await?;

let agent   = memory.agent("assistant");
let mut s   = agent.session("chat-1");
s.observe(Turn::new("alice", "I adopted a rescue greyhound named Biscuit.")).await?;

let answer  = agent.answer("What pet does Alice have?").await?;  // recall + LLM
println!("{}", answer.text);
for src in answer.citations() { println!("  {src:?}"); }        // provenance-backed
```

That `Uniko` handle hands out agent-scoped `Agent` / `Session` / `Goals` / `Data` views and hides the entire engine — `KnowledgeBase`, the pipeline, catalogs — behind them.

### 1.2 The core thesis

uniko rests on one architectural bet, from which every differentiator follows: **compile messages into knowledge at write-time, in-process, in one typed graph.** The mental model is a compiler.

```
   WRITE TIME (ingest)                              READ TIME (recall)
   ───────────────────                              ──────────────────
   Messages (source code)                           query the COMPILED
        │  local ONNX NLP cascade                    Entities/Observations/
        │  (POS·NER·SRL·DEP·CLS, one pass)           Facts/Procedures/Topics
        ▼                                                  ▲
   Entities · Observations                            no LLM in the
        │  async consolidation (P4)                   recall hot path
        ▼                                                  │
   Facts · Procedures · Topics  ───────────────────────────┘
        "compiled knowledge"          compile once, query forever
```

Four claims make up the thesis:

**1. Compile at write-time, not query-time.** Raw messages are treated as source code; consolidation "compiles" them into reusable knowledge. Extraction runs a small quantized DeBERTa cascade (`dragonscale-ai/kniv-deberta-nlp-base-en-xsmall`, INT8) that produces POS / NER / SRL / DEP / CLS labels in a **single shared-encoder forward pass** per sentence. Observations are *reconstructed from the dependency/semantic-role parse*, not copied from raw text — "I'm starting a dance studio" becomes the clean, speaker-attributed, pronoun-resolved claim "Jon is starting a dance studio."

**2. No LLM in the recall hot path.** All write-time enrichment is deterministic CPU work — regex + tree-sitter + the ONNX cascade. Recall then queries the already-compiled `Fact`/`Observation`/`Entity`/`Procedure`/`Topic` nodes rather than re-deriving them with an LLM on every query. An LLM is *optional at every layer* (answer synthesis, triple refinement, topic naming, NL→Cypher) and **never on the write path**.

**3. Zero infrastructure.** One in-process uni-db store holds graph + vectors + full-text + logic. There is nothing to deploy and nothing to keep consistent across services. It is a library, not a stack.

**4. Interaction-first schema.** Memory is organized around communication. `Message` between `Participant`s is the atomic unit, anchored by four load-bearing edges — `SENT_BY→Participant` (carrying `role`), `IN_SESSION→Session`, `NEXT→Message` (carrying `gap_ms`), and `MENTIONS→Entity`. Who-said-what across sessions is *structural*, not something you reconstruct after the fact.

The conventional alternative pays three times: infrastructure cost for the bolted-together services, per-message ingest cost from calling an LLM on every turn, and inference cost paid again on every query. uniko removes all three by moving the intelligence to a local model that runs once at ingest.

### 1.3 What lives where (the layered engine)

uniko is a Cargo workspace of six layered crates over `uni-db` (the engine) and `uni-xervo` (the model runtime). Layer numbers denote cognitive altitude, **not** build/dependency order.

| Layer | Crate | Responsibility |
|------|-------|----------------|
| L1 | `uniko-store` | Typed façade over uni-db: schema, search (vector/FTS/hybrid/sparse/ColBERT), graph traversal + PPR, bitemporal Facts, blob storage, the Locy seam, SSI-aware concurrency. **The sole boundary to uni-db.** |
| L2 | `uniko-pipes` | Content-free pipeline machinery: the `Step` trait, circuit breaker, dead-letter queue, cancellation, health, the MIME/`Modality` taxonomy. |
| L3 | `uniko-extract` | Content intelligence: the write-time ONNX NLP cascade, NER, observation reconstruction, chunking, PDF/artifact ingest, atomic per-message transactions, embeddings. |
| L4 | `uniko-memory` | The public `Uniko` facade, the 3-phase recall cascade, P4 fact consolidation (drift + contradiction), the async pipeline, Locy rule lifecycle, agent-tool functions. |
| L5 | `uniko-cortex` | Higher reasoning: P5 procedure promotion, P6 topic detection. |
| — | `uniko-api` | Logic-free public re-export facade; the lean developer surface. |

An architectural invariant — the **uni-db seal** — keeps the boundary clean: product crates (`uniko-memory`/`extract`/`cortex`/`pipes`) must never `use uni_db` or call `.db()`; they reach the graph only through `uniko-store`'s typed API. A CI ripgrep gate enforces it.

A Python SDK (`import uniko`, alpha) wraps the same in-process engine via PyO3, shipping as three interchangeable wheels (`uniko` CPU / `uniko-cuda` / `uniko-metal`) that differ only by Cargo feature flags.

### 1.4 What the graph knows

The derivation chain is one-directional and always walk-back-able:

```
Message ──SENT_BY──▶ Participant           (who said it, with role)
   │
   │ write-time extraction
   ▼
Observation ──OBSERVED_IN──▶ Message       (an atomic claim, anchored to its turn)
   │        ──ABOUT───────▶ Entity
   │ async consolidation (P4)
   ▼
Fact ──SUPPORTED_BY──▶ Observation         (bitemporal, weighted by cosine)
   │  ──DERIVED_FROM─▶ Episode/Action
   │  ──INVALIDATES──▶ Fact                (supersession, never deletion)
   ▼
Procedure ──DERIVED_FROM──▶ Episode        ("action A→B works", promoted from repetition)
```

Two properties of this model are worth calling out for a new user:

- **Bitemporal Facts.** Every `Fact` carries a single `valid_at` property of `DataType::Btic` — a half-open interval `[lo, hi)` with per-bound certainty and granularity (*not* a split `valid_from`/`valid_until`). An active fact is `[observed_at, +∞)`; when a newer observation contradicts it, the interval's `hi` bound is closed at invalidation time and an `INVALIDATES` edge is drawn. Facts and Procedures are never deleted, preserving the audit trail. Certainty upgrades from `approximate` to `definite` once a fact accumulates ≥ 10 supporting observations.

- **Entity drift.** When an entity accrues more than 4 invalidations in a rolling 30-day window, its `unstable` flag flips true (F39). Recall reads this: a query that touches an unstable entity is forced past the cheap indexed Phase 1 into deeper episodic expansion (F58 drift override), so volatile facts get re-checked against recent evidence.

### 1.5 Target users and concrete use cases

uniko is for teams shipping agents **in their own process** who want a memory layer with zero operational footprint and where ingest cost, offline capability, and provenance matter. Three concrete use cases it is built for:

**Agent long-term memory across sessions.** The interaction-first schema tracks who-said-what over arbitrarily many sessions. `observe()` writes a durable, read-after-write turn; the 3-phase recall cascade (below) then retrieves compiled facts, episodic evidence, and raw chunks, scoped by session / participant / time window via `Scope`. Streamed turns (`submit`/`flush`) exist for throughput but do not preserve cross-turn conversational context — `observe()` is the fidelity path.

**Provenance and "why do we believe this?"** Because every derived node links back to its evidence, an `Answer` carries `citations()` and every `RecallItem` carries a `sources` lineage (`Message` / `Attachment` / `Document`, with chunk ids). Per-claim **visibility scoping** (`null`/`public`/`private:{id}`/`team:{id}`/`org:{id}`) gates `Fact` and `Observation` reads by `Viewer`; unknown visibility schemes fail closed. This suits agents that must explain or defend their outputs, or enforce access boundaries between users/teams.

**Procedural learning.** Repeated successful `Episode` sequences (`FOLLOWED_BY`-adjacent, both `outcome='success'`, both recorded by the agent) are counted by the `sequence_detector` Locy rule and promoted into reusable `Procedure` nodes once a pair recurs ≥ 3 times (P5). A confidence-driven lifecycle then demotes/repromotes/prunes procedures by observed effectiveness. Agents that repeat multi-step workflows accumulate reusable know-how instead of re-planning from scratch.

Adjacent fits: **goal-oriented working memory** (a `Goal→Task→Session→Message→Fact→Entity` traversal computed on demand — there is no stored working-memory node, so it is never stale) and **thematic organization** via `Topic` communities detected with weighted Label Propagation (P6).

### 1.6 How recall works (why the compile bet pays off at read-time)

`agent.recall(query)` runs a **three-phase, coverage-gated cascade** with no LLM in the loop:

```
Phase 1  Compact   vector search over COMPILED tier (Fact/Procedure/Topic)
                   coverage gate 0.75 & ≥3 items → early-exit
                        │ (miss, or unstable-entity drift override)
                        ▼
Phase 2  Expand    Episode/Observation/Message vector + BM25, fused by RRF (k=60),
                   + optional temporal (BTIC) + graph PPR channels, MMR-deduped
                   coverage gate 0.65 & ≥3 items → early-exit
                        │
                        ▼
Phase 3  Broaden   per-variant hybrid fan-out over Chunk/Artifact,
                   optional cross-encoder or ColBERT rerank, token-budgeted assembly
```

Fusion uses Reciprocal Rank Fusion (`RRF_K=60`); results are weighted by a semantic tier (`Fact`=1.0 → `Message`/provenance=0.4); the default reranker is a MiniLM cross-encoder over the top 50 candidates. Because Phase 1 hits *pre-compiled* facts, the common case answers from a small consolidated tier without ever expanding — this is the "query forever" half of the thesis.

### 1.7 Competitive positioning

uniko's claim is not the top judge score — it is being **the only embedded, zero-infrastructure memory system in its class**, competitive on accuracy while an order of magnitude cheaper and faster to operate. Every listed competitor requires at least one external service; uniko runs entirely in-process.

| System | Infrastructure required | LoCoMo10 judge (self-reported comparison) |
|--------|------------------------|-------------------------------------------|
| **uniko** | **None — embedded, in-process** | **81.2%** |
| Mem0 | External vector store (Qdrant) | 91.6% |
| Graphiti (Zep) | External graph DB (Neo4j / FalkorDB) | 75–84% |
| Letta (MemGPT) | External services | 74.0% |
| LangMem | External services | 58.1% |
| Cognee | Graph + Vector + Relational stores | — |

The Neo4j+Qdrant-style "bolt-on stack" is exactly the shape uniko is arguing against: 2–4 services stitched together, an LLM called on every ingested turn, and inference paid again on every query. uniko folds graph, vectors, full-text, and logic into one uni-db store and moves extraction to a local ONNX model that runs once.

Where uniko demonstrably wins is **operations**. From the KTH *dmas-memory* comparison (Wolff & Bennati, arXiv:2601.07978, measured 2026-06-14):

| Ingest (full 5,882-turn corpus) | Cost | Wall time |
|---|---|---|
| **uniko** | **~$0 / 0 tokens** | **~7.5 min (~76 ms/turn)** |
| cognee | $1.32 | 493 min |
| mem0 | $4.82 | 251 min |
| graphiti | $5.49 | 569 min |

That is **33–76× faster ingest at $0** versus the graph backends. On the write path specifically, uniko's bulk graph API is measured at ~980× faster per row than the Cypher executor, and record→replay microbenchmarks show up to **524× on edges** and **49.6× on nodes** (embedder-bound when embeddings are computed). Per-question Q&A wall time is the fastest of the six systems at **4.04 s** (2.84 s recall + 1.20 s generation), against ~6.2 s (graphiti), ~7.0 s (cognee), and ~9.5 s for full-context.

### 1.8 Headline benchmark numbers (proof)

Benchmark of record: **LoCoMo10, gemini-3.1 judge, 2026-05-26, Mem0's verbatim judge prompt, on a 22-core CPU + one 8 GB consumer GPU.** These are figures from uniko's own harness (`crates/uniko-bench`), not a third-party leaderboard.

| Metric | Value |
|--------|-------|
| LLM-judge accuracy (full 1,986-q set) | **0.8117 (81.2%)** |
| Retrieval hit rate | **0.8555 (85.6%)** |
| Token-F1 | **0.321** |
| Total LLM cost (answer + judge) | **$3.55** |
| Ingest (5,882 turns) | **7.5 min at $0** (~76 ms/turn) |
| Mean Q&A latency | **4.04 s** (2.84 s recall + 1.20 s generation) |
| Perf journey (ingest) | Same 369-turn LoCoMo conversation **2h7m → ~22–28s (≈300×)**; separately the full 5,882-turn corpus ingests in **~7.5 min at $0** |

Two caveats the docs insist on, repeated here so they are not conflated: the **81.2%** figure is the full 1,986-question run, whereas the KTH cost/latency tables use the 1,540-question non-adversarial subset — different denominators. And these are self-measured internal-harness numbers.

The proof point that ties back to the thesis: **ingest costs $0 and needs no network** because the intelligence is a local INT8 DeBERTa cascade, not an LLM API. You compile once, offline, for free — then query forever with no LLM in the recall path.

### 1.9 What is shipped, and what is not

Shipped (Phases 1–3): the full Rust engine (six crates on crates.io, `uniko-api = "0.2.0"` as the facade), the 3-phase recall cascade, P4 consolidation with drift/contradiction, P5 procedures, P6 topics, the Locy rule lifecycle, bitemporal Facts, visibility scoping, PDF/artifact ingest, and an alpha Python SDK (three wheels).

Not yet shipped — do not build on these: HTTP/MCP server, CLI, cross-agent sharing, rule induction, and MCTS planning (Phases 4–6). The Python API is alpha and may change before 1.0. The largest known recall weakness is date-anchored/temporal questions, where retrieval tuning is ongoing.

---

## 2. Architecture & Layered Design

uniko is not a service you deploy — it is a Rust library you link into your agent's process, the way you link SQLite. Everything it does happens in-process against a single embedded engine. That single decision shapes the entire architecture: there is no vector store to keep in sync with a graph DB, no rules engine sitting behind an RPC boundary, no LLM in the recall hot path. Instead, uniko is a stack of five layered crates plus a thin facade (`uniko-api`), (plus a facade, Python bindings, and a benchmark harness) sitting on top of two foundational dependencies — `uni-db` (the embedded multi-model graph engine) and `uni-xervo` (the model runtime).

This chapter explains that stack: what each crate is responsible for, which direction dependencies flow, what the `uni-db` foundation actually provides, and how the Layer-2 pipeline infrastructure (the `Step` trait, circuit breaker, dead-letter queue, and the retry story) ties the whole ingest pipeline together.

### 2.1 The layer stack at a glance

uniko's crates are numbered L1 through L5 by *cognitive altitude* — how far a crate sits from raw bytes and how close it sits to reasoning. This is the single most important thing to internalize, and also the single most common source of confusion:

> **Layer numbers describe cognitive altitude, not build order.** L4 `uniko-memory` depends on L5 `uniko-cortex`, not the other way around. The altitude number tells you what a crate *means*; the dependency arrows tell you what it *links against*.

Here is the full picture. The left column is altitude (meaning); the arrows show actual compile-time dependency direction (each crate depends on everything below and to its left that it points at).

```
   ALTITUDE                      CRATE                         WHAT IT OWNS
   (meaning)                 (build artifact)

  ┌───────────────────────────────────────────────────────────────────────────┐
  │  BINDINGS / TOOLS   uniko-py ── uniko-cuda ── uniko-metal   (PyO3 wheels)   │
  │                     uniko-bench            (LoCoMo / LME harness, publish=false)
  └───────────────────────────────┬───────────────────────────────────────────┘
                                   │
  ┌────────────────────────────────▼──────────────────────────────────────────┐
  │  FACADE            uniko-api      thin, logic-free public re-export surface │
  └────────────────────────────────┬──────────────────────────────────────────┘
                                   │
  ┌────────────────────────────────▼──────────────────────────────────────────┐
  │  L4  MEMORY        uniko-memory   Uniko facade, 3-phase recall, P4          │
  │                                   consolidation, pipeline workers, rules    │
  │                        │                                                    │
  │                        │  ── deliberate reverse-altitude edge ──►           │
  │                        ▼                                                    │
  │  L5  CORTEX        uniko-cortex   P5 procedure promotion, P6 topic detection│
  └────────┬───────────────────────────────────┬───────────────────────────────┘
           │                                    │
  ┌────────▼──────────────┐         ┌───────────▼──────────────────────────────┐
  │  L3  EXTRACT          │         │  L2  PIPES                                │
  │  uniko-extract        │         │  uniko-pipes                              │
  │  NER, observations,   │────────►│  Step trait, PipelineContext,             │
  │  chunking, NLP        │         │  CircuitBreaker, DeadLetterQueue,         │
  │  cascade, ingest, PDF │         │  ShutdownCoordinator, content taxonomy    │
  └────────┬──────────────┘         └───────────┬──────────────────────────────┘
           │                                    │
  ┌────────▼────────────────────────────────────▼──────────────────────────────┐
  │  L1  STORE         uniko-store    typed façade over uni-db: schema (25       │
  │                                   labels / 54 edges), search, Locy runtime,  │
  │                                   Facts/BTIC, blob store, striped locks      │
  └────────────────────────────────┬───────────────────────────────────────────┘
                                   │
  ┌────────────────────────────────▼──────────────────────────────────────────┐
  │  FOUNDATION   uni-db (graph + vector + FTS + Locy + SSI + BTIC, embedded)   │
  │               uni-xervo (ONNX embed / NLP / rerank / OCR + LLM providers)   │
  └───────────────────────────────────────────────────────────────────────────┘
```

The linear build order — the order in which crates must be published and compiled — is:

```
uni-db, uni-xervo
      │
      ▼
uniko-store ──► uniko-pipes ──► uniko-extract ──► uniko-cortex ──► uniko-memory ──► uniko-api
    (L1)          (L2)             (L3)             (L5)            (L4)          (facade)
```

Note the two things that violate a naive "L1→L2→…→L5" reading:

1. **`uniko-cortex` (L5) publishes before `uniko-memory` (L4).** Cortex is a *sibling* of extract in the dependency graph — it depends only on `uniko-store`. Memory depends on cortex because P4 consolidation (the "heartbeat" that lives in memory's consolidation worker) is what *triggers* the P5/P6 cortex sweeps. Cortex is a subscriber to P4, so memory owns the trigger policy and therefore takes the dependency. If cortex ever needed memory's runtime APIs at execution time this would become a true cycle; the documented escape is to invert via a sweep trait defined in memory and injected at the composition root.
2. **`uniko-pipes` (L2) depends only on `uniko-store`** and contains *no content logic*. It sits "above" store but "below" extract purely so that the `Step` trait can be declared in one place and implemented by extract, then executed by memory's workers, without either end knowing the other's concrete types (classic dependency inversion).

### 2.2 Crate-by-crate responsibilities

| Crate | Altitude | Depends on (intra-project) | Owns | Must **not** |
|---|---|---|---|---|
| **uniko-store** | L1 | `uni-db`, `uni-xervo` | The typed `KnowledgeBase` façade over uni-db: the full domain schema (25 node labels / 54 edge types in `schema/constants.rs`), all search (vector/FTS/hybrid/sparse/ColBERT), graph traversal + PPR, bitemporal Facts (BTIC), blob storage, the Locy runtime seam, the model-runtime seam, and the concurrency primitives (`StripedLocks`, `transact_with_retry`) that reconcile uni-db's SSI with check-then-create hot paths. | — (it *is* the uni-db boundary) |
| **uniko-pipes** | L2 | `uniko-store` only | Content-free pipeline machinery: the `Step` trait + `PipelineContext`, the lock-free `CircuitBreaker`, the graph-backed `DeadLetterQueue`, the `ShutdownCoordinator` cancellation tree, `HealthTracker`/`WorkerStatus`, Prometheus metric helpers, the MIME/`Modality` taxonomy, and the wire task/outcome types (`IngestTask`, `StepOutcome`, `StepErrorPolicy`). | contain any content-processing logic |
| **uniko-extract** | L3 | `uniko-store`, `uniko-pipes`, `uni-xervo` (direct) | Content intelligence: the write-time ONNX NLP cascade (POS/NER/DEP/SRL/CLS via xervo's shared DeBERTa encoder), entity extraction (regex + tree-sitter + ONNX NER), DEP/SRL-reconstructed observations, chunking, embedding helpers, PDF ingest, and the atomic per-message ingest transaction (`ingest_message_atomic`). Implements `Step` (`IngestStep`). | `use uni_db` / call `.db()` |
| **uniko-cortex** | L5 | `uniko-store` only | Higher reasoning: P5 procedure promotion (`promote_procedures_once`, driven by the Locy `sequence_detector` rule) and P6 topic detection (`detect_topics_once`, weighted Label Propagation over the entity co-occurrence graph). Deliberately thin — algorithms only; all graph I/O delegated to store repository helpers. | `use uni_db` / call `.db()`; depend on `uniko-memory` |
| **uniko-memory** | L4 | `uniko-store`, `uniko-extract`, `uniko-pipes`, `uniko-cortex` | The `Uniko` facade and everything an end user touches: the 3-phase recall cascade, P4 fact consolidation (drift/contradiction), the `PipelineSystem` (ingest + consolidation workers, the latter triggering cortex sweeps), the Locy rule lifecycle, and the agent-tool free functions (goal/task/episode/action/observation/fact/summary/query). | `use uni_db` / call `.db()` |
| **uniko-api** | facade | `uniko-cortex`, `uniko-memory` | The logic-free public surface: a wildcard re-export of `uniko_cortex` (`lib.rs`) plus a hand-curated ~55-type re-export list from `uniko_memory` (`tools.rs`). Negative surface (engine internals stay hidden) is enforced by `compile_fail` doctests; positive surface by a `surface.rs` guard test. | contain *any* logic |
| **uniko-py / -cuda / -metal** | bindings | `uniko-api` | Async-first PyO3 SDK (`import uniko`). One shared multi-thread tokio runtime; every async verb has a blocking `*_sync` twin; a Pydantic overlay under `uniko.models`. cuda/metal are the *same source* (`[lib] path` → `../uniko-py/src/lib.rs`), differing only in the forwarded Cargo feature set. | — |
| **uniko-bench** | tools | all uniko crates + `uni-db`, `uni-xervo` | The internal (`publish=false`) measurement harness: LoCoMo + LongMemEval benchmarks, recall + LLM answer + judge scoring, per-token USD cost tracking, and write-path microbenches (insert/update/bulk-vs-unwind). One of only two places (with `tests/`) permitted to call `.db()`. | — (out of scope for the seal) |

### 2.3 The dependency direction and the uni-db seal

The most load-bearing invariant in the whole codebase is **"issue #2": `uni-db` is an implementation detail, and `uniko-store` is its only gateway.** Every higher crate reaches the graph through store's typed API — `repository/` reads that return decoded Rust structs, `operations/`/`storage/` writes, `model.rs` for embeddings/LLM, `search/` for retrieval, or `begin_tx` driven by validated `*_in_tx` helpers. Store re-exports only the handful of uni-db types callers legitimately need:

```rust
// crates/uniko-store/src/lib.rs — the sealed re-export surface
pub use uni_db::{Value, Transaction, RetryOptions};
pub use uni_xervo::runtime::ModelRuntime;            // from uni-xervo, not uni-db
pub mod temporal { pub use uni_db::common::{TemporalValue, uni_btic::Btic}; }
pub mod xervo    { pub use uni_db::xervo::{GenerationOptions, Message};
                   pub use uni_db::{ModelAliasSpec, ModelTask, WarmupPolicy}; }
```

This seal is **CI-enforced by a ripgrep gate**: product crates (`uniko-{memory,extract,cortex,pipes}`) must not contain `use uni_db` or `kb.db()` anywhere in `src/`. The gate must print nothing; comment lines are exempt, and reviewed exceptions are tagged `// ALLOW:` on the same line. `KnowledgeBase::db() -> &Uni` remains `pub` but is a documented escape hatch **for tests and the benchmark crate only**.

```sh
# The seal check, exactly as CI runs it — must print nothing:
rg -n -e 'use uni_db' -e '\.db\(\)' \
  crates/uniko-memory/src crates/uniko-extract/src \
  crates/uniko-cortex/src crates/uniko-pipes/src \
  | grep -vE ':[0-9]+:[[:space:]]*//' | grep -v 'ALLOW:'
```

The practical consequence: if a higher crate needs a new graph operation, you add a typed method to `uniko-store` rather than reaching past it. This is what keeps the graph schema, the concurrency discipline, and the bitemporal machinery in exactly one place.

### 2.4 The uni-db foundation

Everything uniko does rests on `uni-db` (3.2.0 from crates.io, tracked as `^3`), an **embedded multi-model engine that combines graph, vector, full-text, and a logic runtime in a single in-process store** — nothing to keep in sync, nothing to deploy. `uniko-store` is a typed façade over it. Concretely, uni-db provides:

- **A property graph with OpenCypher.** uniko's entire memory is one property graph — 25 node labels, 54 edge types. Generic CRUD in store builds parameterized Cypher (`build_inline_props`/`build_set_clause`), with `validate_label`/`validate_edge_type`/`validate_property_name` gating every interpolated identifier while values are always `$pN`-bound.
- **Vector, full-text (BM25), sparse, and multi-vector (ColBERT) indexes** in the same store as the graph. Store wraps `uni.vector.query`, `uni.fts.query`, `uni.sparse.query`, and a ColBERT MaxSim path. A hybrid embedder (BGE-M3) fuses dense + sparse + per-token ColBERT columns into a *single* `EmbedHybrid` forward pass. Default vector index is `HnswSq{m:16, ef_construction:100}`, Cosine metric, 384 dimensions.
- **The Locy logic runtime** — rules, `ASSUME` (hypothetical fork-query-rollback), and `ABDUCE` (ranked modifications). Store exposes `create_rule`/`query_rule`/`execute_rule`/`assume`/`abduce`. Cortex's `sequence_detector` and memory's four stdlib rules run here. A recurring gotcha: **Locy is not Cypher** — a single comma-joined `MATCH` (a second `MATCH` is a parse error), `expr AS name` aggregates, and no `$param` in a post-FOLD `HAVING` (RC12), so threshold logic is pushed into Rust consumers.
- **A native bitemporal type, `DataType::Btic`.** A `Fact`'s `valid_at` is one half-open interval `[lo, hi)` with per-bound granularity and certainty — *not* a `valid_from`/`valid_until` pair. Certainty upgrades `Approximate → Definite` once cumulative observation count crosses `CERTAINTY_THRESHOLD = 10`. Recall matches validity via the `btic_overlaps(f.valid_at, $window)` scalar Cypher function (Allen algebra: `a.lo < b.hi && b.lo < a.hi`).
- **Serializable Snapshot Isolation (SSI) transactions** with retriable conflict classification, plus `bulk_insert_vertices`/`bulk_insert_edges` fast paths that bypass the Cypher executor. The bulk path is measured ~980× faster per row than Cypher (bulk ~150µs/edge vs Cypher ~147ms/edge at sess=24) because the VIDs are already known — but it doesn't return EIDs and doesn't re-validate property names, so ingest hot paths validate keys up front.

Above uni-db sits **`uni-xervo`** (0.17.0), the model runtime: ONNX embedding/NLP/rerank/OCR plus remote (`openai`, `vertexai`), `mistralrs`, and `candle` providers. Store reaches it internally via `db.xervo()` and exposes it through the `model.rs` seam (`embed`, `embed_multivector`, `generate`, `rerank`, `model_runtime`). The default stack: BGE-small-en-v1.5 (384d) embedder, `ms-marco-MiniLM-L-6-v2` cross-encoder reranker (enabled by default), and `kniv-deberta-xsmall` INT8 for the NLP cascade.

#### Concurrency: why SSI alone isn't enough

SSI guarantees two concurrent read-modify-write callers won't lose an update — the second committer aborts with a retriable `SerializationConflict`, surfaced as `UnikoError::Conflict` and retried by `transact_with_retry` (fresh tx per attempt, capped jitterless exponential backoff). **But SSI does not catch insert-phantoms**: an empty `MATCH` registers no read-set, so a bare check-then-create on a *non-unique* index (`entity_id`, `content_id`, `session_id`, `fact_id`) lets two callers both read "absent" and both `CREATE` a duplicate.

Store's fix is `StripedLocks` (`locks.rs`): 256 tokio async mutexes keyed by canonical byte-prefixed keys (`entity:`, `content:`, `session:`, `fact:`, `node:`). A guard must be held across **both** the existence re-read **and** the commit. `lock_many` sorts and dedups by *stripe index* (not key bytes) to give a global acquisition order and avoid AB/BA deadlock — and to avoid self-deadlock when two distinct keys collide on the same non-reentrant stripe (a hand-rolled per-key loop once deadlocked `batch_upsert_facts` past ~50 facts).

```rust
// The canonical write pattern: locks acquired BEFORE tx-open, held across commit
let guards = kb.lock_entity_ids(&entity_ids).await;      // striped RMW guards
let out = kb.transact_with_retry(RetryOptions::default(), |tx| async move {
    // authoritative existence read happens INSIDE the tx, under the locks
    let existing = kb.fetch_entities_for_upsert_in_tx(&tx, &entity_ids).await;
    // ... create_node_in_tx / batch_update_entity_counters_in_tx ...
    (tx, Ok(()))
}).await?;
drop(guards);                                            // release after commit
```

### 2.5 The L2 pipeline infrastructure

`uniko-pipes` is the "infrastructure spine" of the ingest and consolidation system. It defines the *vocabulary* and the *reliability primitives* that extract and memory plug into, and — this is important — it contains **no runner**. The `Step` trait is declared here; the executor (`run_step_chain`) lives downstream in `uniko-memory`'s `ingest_worker.rs`. Anyone reasoning about execution semantics must read both.

#### The `Step` trait and `PipelineContext`

A pipeline is an ordered `Vec<Box<dyn Step>>`. Each step reads and mutates a per-item `PipelineContext` threaded down the chain:

```rust
#[async_trait::async_trait]
pub trait Step: Send + Sync {
    fn name(&self) -> &str;
    fn should_run(&self, ctx: &PipelineContext) -> bool { true }
    async fn execute(&self, ctx: &mut PipelineContext) -> Result<StepOutcome, UnikoError>;
}
```

`PipelineContext` carries `node_id`, `content`, `content_type`, a per-item `CancellationToken`, an `Arc<KnowledgeBase>`, an `Arc<CircuitBreaker>`, and the accumulating results (`extracted_entities`, `sender`, `extracted_observations`, plus a free-form `metadata` map that carries the typed ingest payload from step to step). The metadata-driven dispatch is how `IngestStep` deserializes its typed `IngestMessage`/`IngestArtifact`/`IngestPdf`/`IngestSource` payload without pipes knowing extract's concrete types.

#### Error isolation: `StepOutcome` and `StepErrorPolicy`

A single item's failure at one step must not poison the batch. That is expressed entirely as data:

```rust
pub enum StepOutcome {
    Completed,
    Skipped { reason: String },
    Failed  { error: String, policy: StepErrorPolicy },
}

#[derive(Copy, Clone)]
pub enum StepErrorPolicy {
    Skip,        // log + continue the chain
    DeadLetter,  // persist to the DLQ + continue the chain
    Abort,       // stop remaining steps for THIS item
}
```

The downstream runner iterates steps in order and, for each: checks cancellation (cooperative, only *between* steps), honors `should_run`, then matches `execute`'s result. `Completed`/`Skipped` continue; `Failed` dispatches on policy. A transport-level `Err` (as opposed to an explicit `Failed`) is defensively coerced to `StepErrorPolicy::DeadLetter` — an `Err` has no policy to declare, so it never aborts. The key semantic to remember: **`Skip` and `DeadLetter` both continue the remaining chain; only `Abort` short-circuits.** A dead-lettered step still lets later steps run on a partially-populated context.

#### The circuit breaker

External LLM calls are wrapped in a lock-free 3-state `CircuitBreaker` (`AtomicU8`/`U32`/`U64`) so a flaky provider trips to local fallbacks instead of cascading:

```
   record_success                         count >= failure_threshold
        ┌──────────────────┐            ┌────────────────────────────┐
        ▼                  │            ▼                            │
   ┌────────┐   failure   ┌┴────────┐  now - last_failure          ┌─┴──────┐
   │ Closed │────────────►│ (count) │  >= recovery_ms   ┌─────────►│  Open  │
   └────────┘             └─────────┘  (checked lazily  │          └────────┘
        ▲                              inside call())    │              │
        │ success on probe                               │ probe granted │
        │                          ┌──────────┐          │ to first      │
        └──────────────────────────┤ HalfOpen │◄─────────┘ caller        │
                                   └──────────┘  failure on probe ───────┘
```

`call()` loads the state; if `Open` and not yet recovered, it returns `Err(UnikoError::Llm("circuit breaker open"))` immediately; if `Open` and recovered, it stores `HalfOpen` and lets exactly the probe call through, then records success (→ `Closed`) or failure (→ `Open`). Two caveats worth knowing: `HalfOpen` has **no single-probe mutual exclusion** (two concurrent callers after recovery can both be admitted — best-effort, not strict), and recovery timing uses `SystemTime` wall clock, so NTP steps can affect it. `failure_count` is a plain monotonic counter reset only on success — "consecutive failures" really means "failures since last success," not a sliding window.

```rust
// Guarding an LLM call through the breaker on the context:
let out = ctx.llm_breaker.call(|| async { provider.complete(prompt).await }).await;
// Err(UnikoError::Llm("circuit breaker open")) when tripped.
```

#### The dead-letter queue — and the truth about "retry"

The chapter spec, and the pipes crate description, mention "retry." **There is no retry module and no retry loop in `uniko-pipes`.** This matters, so state it plainly:

- `DeadLetterQueue` is a thin wrapper over `Arc<KnowledgeBase>` with a **single `store()` method** that creates a `DeadLetter` graph node (`{step, error, node_ref, retry_count, max_retries}`). There is no `list`, `retry`, or `clear` — the module doc says these surfaces are intentionally not provided.
- `retry_count` is **always written `0`** and is never read or incremented anywhere in the crate. "Retry" exists only as *data* (`PipelineConfig::dead_letter_max_retries = 3`, and the two node properties). Retry orchestration is deferred/unimplemented.
- The **only automatic recovery mechanism actually implemented** is the circuit breaker's `Open → HalfOpen` probe.

The `DeadLetter` node is standalone with no edges — it records a failed pipeline step for offline triage, and is not wired into the memory graph.

Separately, the *store* layer does own a real retry loop — `transact_with_retry` retries retriable SSI/commit conflicts with capped exponential backoff — but that is transaction-conflict recovery inside L1, a different concern from pipeline step retry.

#### Shutdown and health

`ShutdownCoordinator` builds a `CancellationToken` tree: a root with `ingest` and `consolidation` children. `shutdown()` is phased — cancel ingest (drain ≤5s) → cancel consolidation (drain ≤10s) → cancel root → bounded join-or-abort of the worker handles within `total_timeout`. (Each drain sleep is itself capped at `total_timeout`, and the join deadline restarts after them, so worst-case wall time is `min(5s, T) + min(10s, T) + T`.) `HealthTracker` keeps an EMA latency (α=0.1) and classifies each worker, priority-ordered: `circuit_open → Degraded`, else idle>300s with a non-empty queue → `Stalled`, else queue ratio>0.8 → `Backpressured`, else `Healthy`.

#### The content taxonomy

Pipes also owns the content-type taxonomy (`content.rs`) because it is the single routing key shared by *both* ends of the pipeline — ingest (which extractor/chunker handles a blob) and recall (per-modality channels). It has a validated `Mime` newtype (parsed through the third-party `mime` crate but not leaked) and a `#[non_exhaustive]` `Modality` enum (`Text | Code | Markup | Structured | Document | Pdf | Image | Audio | Video`). `modality_for_mime` is an ordered `essence` match with fallthrough guards; unknown/arbitrary binary routes to `Modality::Text` (the text chunker) unless an explicit modality is supplied.

### 2.6 How a message flows through the layers

Putting the pieces together, here is the end-to-end path of one observed turn, and which layer owns each stage:

There are **two** write paths, and they differ in whether L2/L4's pipeline
machinery is involved at all. `observe()` — the default — goes straight to L3
and commits before returning; the `PipelineSystem` is only constructed when the
instance was built with `.streaming(true)`, and is reachable only via
`submit()` / `submit_source()`.

**(a) The durable default — `session.observe(turn)`:**

```
  user code
     │  session.observe(Turn::new("alice", "I love hiking"))
     │  (no pipeline, no step chain, no DLQ — a direct call)
     ▼
  L3  uniko-extract       ingest_message_atomic:
     │                      1. idempotency check (get_node_by_ext_id)
     │                      2. ensure session/sender (lock_session_setup)
     │                      3. PURE CPU: rule NER + code AST + ONNX cascade
     │                         (POS/NER/DEP/SRL/CLS in one xervo encoder pass)
     │                      4. prepare entity upserts (canonical entity_id)
     │                      5. lock_entity_ids  ── held across commit ──┐
     ▼                                                                  │
  L1  uniko-store         6-9. ONE retriable tx writes:                 │
     │                       Message + edges + Chunks + Entity upserts  │
     │                       + MENTIONS + Observations + OBSERVED_IN     │
     │                       + ABOUT, then commit()  ◄── all-or-nothing │
     ▼                                                                  │
  Foundation  uni-db      bulk_insert_vertices/edges (hot path),  ◄─────┘
                          SSI + StripedLocks serialize concurrent
                          same-entity writes
```

**(b) The opt-in streaming path — `session.submit(turn)`**, which requires
`.streaming(true)` and otherwise returns `UnikoError::Config`. This is where
the L4 runner, semaphore, and dead-letter queue live:

```
  user code
     │  session.submit(Turn::new("alice", "I love hiking"))   // fire-and-forget
     ▼
  L4  uniko-memory        PipelineSystem.submit_ingest → ingest_worker
     │                    (biased select, Semaphore(concurrency=8),
     │                     per-item child cancel token, InflightGuard)
     │                    run_step_chain(steps, ctx, dlq)   ◄── THE runner
     ▼
  L3  uniko-extract       IngestStep.execute → dispatches on `ingest_type`:
     │                    "message" → ingest_message_atomic (as in (a)),
     │                    "artifact" → ingest_artifact, "pdf" → ingest_pdf,
     │                    "source" → ingest_source; unknown → Skipped
     ▼
  L1/uni-db               ...same write path as (a) from here on.
```

At the end of a conversation, `session.finalize()` (also folded into
`summarize()`) builds the **session-level** retrieval surfaces: a chunked
transcript and deduplicated observation chunks wired `ABOUT` the entities and
participants involved. These are what session-scoped recall and the Phase 1
session boost walk; see §6.

```
  ... later, asynchronously ...

  L4  consolidation_worker  P4: Observations → Facts (BTIC), F38
     │                      contradiction, F39 drift → then triggers:
     ▼
  L5  uniko-cortex          P5 promote_procedures_once (sequence_detector)
                            P6 detect_topics_once (weighted LPA)
```

The recall path runs the reverse direction and, critically, calls **no LLM in the hot path**: L4's 3-phase cascade queries the *already-compiled* Facts/Observations/Entities in L1's graph, optionally reranks via xervo, and assembles a token-budgeted `ContextBundle`. This is the "compile once, query forever" thesis expressed as an architecture: the expensive extraction work is paid once at write-time by the L3 ONNX cascade, and every subsequent recall is a graph + vector query.

### 2.7 Architectural invariants to carry forward

These are the load-bearing rules the rest of this book assumes:

1. **The uni-db seal** — product crates reach the graph only through `uniko-store`'s typed API; the CI ripgrep gate enforces it.
2. **Altitude ≠ build order** — `uniko-memory` (L4) depends on `uniko-cortex` (L5); the one deliberate reverse-altitude edge.
3. **`schema/constants.rs` is the schema source of truth** — 25 node labels, 54 edge types. `config/schema.json` is a generated snapshot (`cargo run --bin export-schema`), loaded only when `UnikoConfig::schema_path` is set; regenerate it alongside any schema change.
4. **`uni-db` is a separate project** — never edit the local `../uni/` checkout; on a suspected uni-db bug, build a minimal isolated repro (pattern: `crates/uniko-store/tests/unidb_bytes_return_repro.rs`) and file upstream.
5. **SSI does not catch insert-phantoms** — every check-then-create on a non-unique index must hold the correct `StripedLocks` key across both the existence re-read and the commit.
6. **"Retry" in the pipeline is data-only** — the circuit breaker's `Open→HalfOpen` probe is the sole automatic recovery mechanism; the DLQ persists but does not re-drive.

---

## 3. Data Model & Knowledge Graph Schema

uniko keeps the entire memory of an agent in **one uni-db property graph** — graph structure, dense/sparse vectors, ColBERT multi-vectors, full-text/BM25 indexes, and the Locy logic runtime all live in the same embedded store, so there is nothing to keep in sync between a vector DB, a graph DB, and a rules engine. This chapter is the reference for the shape of that graph: every node label, every edge type, how knowledge is *derived* rather than stored, how Facts carry bitemporal validity and drift, and how visibility scoping is enforced.

The single source of truth for the catalog is `crates/uniko-store/src/schema/constants.rs` — the two arrays `labels::ALL` (25 node labels) and `edges::ALL` (54 edge types), grouped by cognitive layer. Per-node property and index definitions live in sibling files (`schema/facts.rs`, `schema/entities.rs`, `schema/observations.rs`, `schema/messages.rs`, `schema/chunks.rs`, `schema/procedures.rs`, `schema/topics.rs`, …) and are installed by the idempotent, two-phase `register_schema` (`schema/mod.rs`).

> **Counts drift easily — `constants.rs` is authoritative.** `labels::ALL` = 25 node labels, `edges::ALL` = 54 edge types. The downstream artifacts (`website/docs/reference/schema.md`, `concepts/data-model.md`, and the `config/schema.json` snapshot) currently agree with it, but they are *downstream*: regenerate the snapshot with `cargo run --bin export-schema` and update the docs in the same change as any schema edit. Two additions are the usual casualties — the `Pattern` label and the `CONTRADICTED_BY` edge, both Locy-consumer additions (`episode_pattern_detector`, `contradiction_detector`).

### 3.1 The central design rule: knowledge is derived, never directly stored

Only two node kinds are *observed* directly:

- **`Message`** — a communication turn between Participants.
- **`Action`** — a concrete tool call / agent operation.

Everything else — `Entity`, `Observation`, `Fact`, `Topic`, `Procedure`, `Summary`, `Episode` — is **derived** from those observations and wired back to its evidence with a provenance edge. The schema *is* the provenance spine: you can always walk from a belief back to the raw turns that justify it, so "why do we believe this?" is a graph query, not an afterthought.

This is why the derivation is one-directional:

```
Participant
   ▲ SENT_BY (role=user|assistant|system|tool)
   │
Message ── IN_SESSION ─▶ Session
   │  └─ NEXT (gap_ms) ─▶ Message
   │  └─ MENTIONS (count) ─▶ Entity
   │
   │  (write-time NLP cascade: POS/NER/SRL/DEP/CLS)
   ▼
Observation ── OBSERVED_IN ─▶ Message | Chunk
   │  ├─ OBSERVED_DURING ─▶ Episode
   │  └─ ABOUT ─▶ Entity | Participant
   │
   │  (async P4 consolidation: group by (subject,predicate), cluster, vote)
   ▼
Fact ── SUPPORTED_BY (weight=cos) ─▶ Observation
   │  ├─ DERIVED_FROM (derivation_kind, derived_at) ─▶ Episode | Action
   │  ├─ DERIVED_BY ─▶ Rule
   │  ├─ ABOUT ─▶ Entity
   │  └─ INVALIDATES (reason, invalidated_at) ─▶ Fact   (older, contradicted)
   │
   │  (P5 procedure promotion from repeated success chains)
   ▼
Procedure ── DERIVED_FROM ─▶ Episode
```

Two consequences worth internalizing:

1. **Facts and Procedures are never deleted.** Contradiction closes a Fact's validity interval and records an `INVALIDATES` edge; the audit trail survives. Hard deletion of a Message re-evaluates dependent Facts and soft-invalidates (BTIC-closes) any that lose their last supporting Observation — it never silently drops the belief.
2. **Ingest is LLM-free and atomic.** All write-time enrichment is deterministic CPU work (regex + tree-sitter + a small quantized DeBERTa cascade). Per-message ingest is *prep-then-commit*: all NLP/NER/SRL/dedup and read-only lookups happen before a transaction opens, then **one** transaction writes `Message` + edges + `Chunk`s + `Entity` upserts + `MENTIONS` + `Observation`s + `OBSERVED_IN`/`ABOUT` and commits once (`ingest_message_atomic`, `crates/uniko-extract/src/ingest/atomic.rs`). There is never a half-state like "a Message with no entities." LLM work (triple refinement, topic naming, answer synthesis) is async and optional and never touches the write path.

### 3.2 Node label catalog (25)

`labels::ALL` is numbered by cognitive layer in source order. The layer numbers describe *cognitive altitude*, not build order or dependency direction.

| Layer | Labels | Role |
|---|---|---|
| L0 | `Participant` | The actor: user, assistant, system, tool. First-class identity. |
| L1 (working/task) | `Goal`, `Task`, `Session` | Goal-oriented working memory; `Session` groups Messages. |
| L2 (episodic) | `Message`, `Action`, `Episode` | The **observed** substrate + agent-recorded experience. |
| L3 (artifacts / document-IR) | `Artifact`, `ArtifactContent`, `Chunk`, `Page`, `Block` | Attachments, blob content, retrieval chunks, and the PDF Page/Block document-IR. |
| L4 (semantic) | `Entity`, `Observation`, `Fact`, `Topic`, `Summary` | Derived knowledge. |
| L5 (procedural) | `Procedure`, `Rule`, `Pattern` | What works; Locy rules; detected patterns. |
| L6 (meta) | `ConsolidationCycle`, `DeadLetter` | Audit node for P4; failed-pipeline-step records. |
| L7 (org) | `Organization`, `Team` | Membership for team/org visibility scoping. |
| L8 (stats) | `KnowledgeBaseStats` | Singleton: persists blob-storage backend + modality-presence flags. |

Notes on the less-obvious labels:

- **`Working memory has no stored node.`** There is deliberately no `WorkingMemory` label — a goal's working set is *recomputed on demand* by traversing `Goal → Task → Session → Message → Fact → Entity`. It is always current, never cached (see `Goals::context` in `crates/uniko-memory/src/facade/goals.rs`).
- **`ArtifactContent`** holds the content-addressed bytes for the default Lance blob backend. It is shared/deduped: dropped only when its reference count reaches 1. `Page`/`Block` form the PDF document-IR (`Artifact → HAS_PAGE → Page → CONTAINS → Block`, blocks chained by `NEXT_IN_READING_ORDER`).
- **`Pattern`** and the `CONTRADICTED_BY` edge back the `episode_pattern_detector` and `contradiction_detector` stdlib Locy rules. They are the two additions most often missed when the downstream catalogs are updated.
- **`DeadLetter`** is a *standalone* node with **no edges** — it records a failed pipeline step (`step`, `error`, `node_ref`, `retry_count`, `max_retries`) and is not wired into the graph. `retry_count` is always written `0` and never read; DLQ retry is data-only and unimplemented (`crates/uniko-pipes/src/dead_letter.rs`).

#### Session / Participant / Message as first-class

Memory in uniko is organized around **communication**, not around opaque documents. The three interaction primitives are first-class nodes with structural (not inferred) relationships:

- **`Participant`** is the stable identity of an actor. Facts, Observations, and MENTIONS all resolve to it. Team/org membership hangs off it (`MEMBER_OF`, `PART_OF_TEAM`) and is what visibility scoping consults.
- **`Session`** groups a conversation. Messages carry `IN_SESSION → Session`; Participants carry `PARTICIPATED_IN → Session`.
- **`Message`** is the atomic observed unit and anchors everything with four load-bearing edges:
  - `SENT_BY → Participant` — carries `role` (`user`/`assistant`/`system`/`tool`),
  - `IN_SESSION → Session`,
  - `NEXT → Message` — the conversational chain, carrying `gap_ms`,
  - `MENTIONS → Entity` — carrying a `count`.

Because who-said-what is structural, uniko can answer cross-session provenance questions ("when did Alice first mention X?", "which turn justifies this Fact?") that a flat vector store cannot.

Ensuring the Session and sender exist is a cold-path, first-sight-only operation guarded by `lock_session_setup` (a striped RMW lock) to avoid duplicate `Session`/`Participant`/`PARTICIPATED_IN` and SSI antidependency aborts under concurrent ingest. On the warm path (both ids cached in the in-memory `SessionContext`), it does no DB work.

### 3.3 Edge type catalog (54)

`edges::ALL` grouped by layer, with endpoints and edge properties. Many edges are **polymorphic** (multiple valid source or target labels) — noted below.

**L1 — Working memory / task graph**

| Edge | Endpoints | Props |
|---|---|---|
| `OWNED_BY` | Goal → Participant | |
| `PARENT_GOAL` | Goal → Goal | |
| `PART_OF` | Task → Goal | |
| `ASSIGNED_TO` | Task → Participant | |
| `DEPENDS_ON` | Task → Task | |
| `SUBTASK_OF` | Task → Task | |
| `FOR_TASK` | Session, Episode → Task | |
| `FOR_GOAL` | Session → Goal | |
| `PARTICIPATED_IN` | Participant → Session | |

**L2 — Episodic**

| Edge | Endpoints | Props |
|---|---|---|
| `SENT_BY` | Message → Participant | `role` |
| `ADDRESSED_TO` | Message → Participant | |
| `IN_SESSION` | Message, Action, Episode → Session | |
| `NEXT` | Message → Message | `gap_ms` |
| `PERFORMED_BY` | Action → Participant | |
| `TRIGGERED_BY` | Action, Episode → Message | |
| `PRODUCED` | Action → Artifact | |
| `NEXT_ACTION` | Action → Action | |
| `RECORDED_BY` | Episode → Participant | |
| `INVOLVES` | Episode → Action | |
| `FOLLOWED_BY` | Episode → Episode | `gap_ms` |

**L3 — Artifacts / chunks / document-IR**

| Edge | Endpoints | Props |
|---|---|---|
| `HAS_CHUNK` | Artifact, Message, Session, Block → Chunk | `index` |
| `HAS_CONTENT` | Artifact → ArtifactContent | `role` |
| `HAS_PAGE` | Artifact → Page | `index` |
| `CONTAINS` | Page → Block | `reading_order` |
| `NEXT_IN_READING_ORDER` | Block → Block | |
| `CREATED_BY` | Artifact → Action | |
| `MODIFIED_BY` | Artifact → Action | `diff_summary` |
| `ATTACHED_TO` | Artifact → Session, Message | `attached_at` |

**L4 — Semantic / derivation**

| Edge | Endpoints | Props |
|---|---|---|
| `MENTIONS` | Message, Chunk, Action, Artifact, Episode → Entity | `count` |
| `OBSERVED_IN` | Observation → Message, Chunk | |
| `OBSERVED_DURING` | Observation → Episode | |
| `ABOUT` | Observation, Fact, Chunk → Entity, Participant | |
| `SUPPORTED_BY` | Fact → Observation | `weight` |
| `DERIVED_BY` | Fact → Rule | |
| `DERIVED_FROM` | Fact, Procedure, Artifact → Episode, Action, Artifact | `derivation_kind`, `derived_at` |
| `INVALIDATES` | Fact → Fact | `reason`, `invalidated_at` |
| `CONTRADICTED_BY` | Fact → Episode | `detected_at` *(drawn by `contradiction_detector`)* |
| `SHARED_FROM` | Fact → Fact | `shared_by`, `shared_at` |
| `BELONGS_TO` | Entity, Fact → Topic | |
| `SUMMARIZES` | Summary → Session, Task, Goal, Artifact, Entity, Topic | |

**L5 — Procedural**

| Edge | Endpoints | Props |
|---|---|---|
| `OPERATES_ON` | Procedure → Entity | |
| `USED_IN` | Procedure → Task | |
| `SUPERSEDES` | Rule → Rule | |
| `COVERS` | Rule → Episode | `correct` |

**L6 — Meta (ConsolidationCycle audit)**

| Edge | Endpoints |
|---|---|
| `PROCESSED` | ConsolidationCycle → Observation *(idempotency anchor)* |
| `INVOLVED` | ConsolidationCycle → Episode |
| `CREATED` | ConsolidationCycle → Fact |
| `REINFORCED` | ConsolidationCycle → Fact |
| `INVALIDATED` | ConsolidationCycle → Fact |
| `PROMOTED` | ConsolidationCycle → Procedure |
| `APPLIED_RULE` | ConsolidationCycle → Rule |

**L7 — Organization**

| Edge | Endpoints | Props |
|---|---|---|
| `MEMBER_OF` | Participant → Organization | `role`, `joined_at` |
| `PART_OF_TEAM` | Participant → Team | |
| `TEAM_IN_ORG` | Team → Organization | |

The `PROCESSED` edge deserves emphasis: it is what makes consolidation **idempotent**. An Observation with an inbound `PROCESSED` edge is skipped by the next cycle, so `run_cycle` is safe to call repeatedly. Each `(Fact, Observation)` pair gets exactly one `SUPPORTED_BY` edge.

### 3.4 Key node properties and indexes

Reference: `website/docs/reference/schema.md`. Property/index specifics from the schema submodules:

**`Fact`** (`schema/facts.rs`): `subject` (String, notnull, Hash + FullText), `predicate` (String, notnull, Hash), `object` (String, null), `confidence` (Float64, BTree), `observation_count` (Int64), `valid_at` (**Btic**, null), `source_rule` (String), `visibility` (String, null), `invalidation_reason`, `invalidated_at`, `fact_id` (String, Hash), `embedding` (Vector 384).

**`Entity`** (`schema/entities.rs`): `entity_id` (Hash), `name` (Hash + FullText), `entity_type` (Hash), `first_seen`/`last_seen`, `frequency` (Int64), `confidence` (Float64), plus the drift fields `invalidation_count` (Int64), `last_invalidation_at` (DateTime), `unstable` (Bool, **hash-indexed**), and `embedding` (Vector 384).

**`Observation`** (`schema/observations.rs`): `observation_id`, `content` (String, notnull; FullText + auto-embed), the structured triple `subject`/`predicate`/`object`, `temporal_phrase`, `temporal_anchor` (DateTime, **BTree**), `observed_at`, `confidence`, `visibility` (String, null), `embedding` (Vector 384). When ingested with a hybrid embedder (BGE-M3), Chunk/Observation additionally carry a `sparse_embedding` (SparseVector) and `colbert_embedding` (List\<Vector\>) column, all pointing at the same `embed/hybrid` alias so uni-db fuses them into one `EmbedHybrid` forward pass.

**Index families** (`schema.md`):

- **Hash** — every id prop except `Chunk.chunk_id` (unindexed — chunks are reached via `HAS_CHUNK`, vector, or BM25, never by id; `DeadLetter` has no id at all), plus `Entity.name`, `Fact.subject`/`predicate`, `Entity.unstable`.
- **BTree** — `Message.timestamp`, `Session.started_at`, `Fact.confidence`, `Observation.temporal_anchor`.
- **FullText/BM25** — `Message.content`, `Chunk.text`, `Observation.content`, `Summary.text`, `Fact.subject`.
- **Vector** — `HnswSq`, Cosine metric, 384-d by default.

**Entity identity is canonical and single-sourced** (`id.rs`): `entity_id = "ent_" + hex64(SHA256(lower(name) \0 canonical_type))`. Every NER writer (regex rules, tree-sitter AST, ONNX NER) must pass a type drawn from **one shared lowercase vocabulary** (`person`/`organization`/`location`/…). Mixing vocabularies duplicates entities. Because `entity_id` is *non-unique* in uni-db, concurrent ingest of the same entity is serialized by a striped RMW lock (`lock_entity_ids`) held across the existence re-read and the commit — otherwise two callers both read "absent" and both `CREATE`.

### 3.5 Fact drift & temporal semantics

#### Bitemporal Facts (BTIC)

Every `Fact` carries a **single** `valid_at` property of `DataType::Btic` — a half-open **valid-time** interval `[lo, hi)`, *not* a split `valid_from`/`valid_until` pair. Do not try to decompose it; recall matches validity with the scalar Cypher UDF `btic_overlaps(f.valid_at, $window)`, never a DateTime range join.

Two temporal axes coexist:

- **Valid time** — the BTIC `lo`/`hi` (when the fact is true in the world).
- **Transaction time** — uni-db's own `_created_at`/`_updated_at` (when we recorded it).

BTIC bounds also carry **granularity** and **certainty** metadata (`schema/btic.rs`). A fact is *active* as `[observed_at, +inf)` (open `hi` = `POS_INF`); invalidation closes `hi` at the invalidation time. Certainty starts `Approximate` and upgrades to `Definite` once cumulative supporting `observation_count ≥ CERTAINTY_THRESHOLD = 10` (idempotent, `btic.rs`). Reinforcement recomputes a Laplace-smoothed confidence `(n+1)/(n+2)` (→ 0.5 at n=0, asymptotes to 1.0).

A critical gotcha: BTIC columns **must** be read via a bare column projection (`RETURN f.valid_at`), never `RETURN n` — a Node-wrapped projection stringifies Temporals lossily. This readback pattern is reused across `facts.rs`, `deletion.rs`, and `recall.rs`.

Allen-algebra predicates (`btic_contains`/`overlaps`/`before`) back the temporal recall channel: overlap is `a.lo < b.hi && b.lo < a.hi`.

#### Consolidation (P4) — how Observations become Facts

`run_cycle` / `run_cycle_with` (`crates/uniko-memory/src/consolidation.rs`) does five steps:

1. Fetch unprocessed, triple-bearing Observations (no inbound `PROCESSED` edge), capped at `DEFAULT_BATCH_SIZE = 500`. Optional LLM triple refinement (`TripleSource::Llm{alias}`, falls back to `SrlDep` on failure; default is `SrlDep`, no LLM).
2. **Group** by `(normalize_canonical(subject), predicate)`.
3. **Paraphrase-collapse then vote** on a canonical object: cluster object surface forms by cosine similarity (`COSINE_THRESHOLD = 0.88`, greedy single-pass agglomeration with running-mean centroids in sorted key order for stability); canonical object = mode over clusters, then mode over surface forms, recency tiebreak.
4. **Upsert** one `Fact` per group (`batch_upsert_facts`, within-batch dedup by `fact_id`), reusing/embedding as needed; wire `SUPPORTED_BY` edges with per-edge `weight = clamp(cos(Fact, Obs), 0, 1)` (fallback 1.0).
5. Record a `ConsolidationCycle` audit node with `PROCESSED`→Observation, `CREATED`/`REINFORCED`/`INVALIDATED`→Fact, and `APPLIED_RULE`→Rule edges. (`INVOLVED` is registered in the schema but no code path writes it today.)

`CycleStats` returns `observations_processed`, `facts_created`, `facts_reinforced`, `facts_invalidated`, `drift_alerts`.

#### F38 — contradiction detection

Within a `(subject, predicate)` group, votes are tallied in **cluster space**. If the fraction of votes falling **outside** the prior open Fact's cluster exceeds `CONTRADICTION_THRESHOLD = 0.40`, `invalidate_fact` closes the prior Fact's BTIC `hi` at *now* and wires `INVALIDATES` with `reason = "consolidation contradiction"`. The old Fact is soft-invalidated, never deleted. (`find_stale_open_facts` uses `btic_is_unbounded` to locate still-open Facts.)

#### F39 — entity drift

`record_entity_invalidation` counts `INVALIDATES` edges within a rolling `DRIFT_WINDOW_DAYS = 30` window. When the windowed count exceeds `DRIFT_THRESHOLD = 4`, it flips `Entity.unstable = true`. The important subtlety: the flag is **windowed**, not lifetime — the cumulative `invalidation_count` keeps incrementing even while `unstable` stays false.

#### F58 — drift override in recall

`unstable` is consumed by recall. Normally Phase 1 (cheap indexed vector search over the consolidated Semantic tier) can early-exit if coverage is sufficient. But if a query entity-ref resolves to an `Entity` with `unstable = true` (`kb.any_unstable_entities`), the cascade is **forced past Phase 1 into Phase 2+** episodic expansion, so volatile-entity queries consult recent evidence instead of a possibly-stale consolidated Fact.

#### Rule confidence decay

Locy rules carry a confidence lifecycle (`RuleLifecycleConfig`): `confidence *= 0.95^missed_cycles`; a match resets `missed_cycles` and adds `+0.05` (cap 1.0); demote below `0.40`, re-promote at/above `0.60` (hysteresis gap), prune after 90 days without a match. The 4 stdlib rules are pinned at confidence 1.0 and **exempt** from decay/demote/prune.

### 3.6 Visibility & scoping

Two distinct filters run at recall time and must not be conflated:

**Redaction** (`filter_redacted`, runs on *every* recall) drops soft-forgotten items: Facts/Observations whose `visibility` starts with a `redacted:` scheme, and Messages/Chunks with a `redacted` flag. This is the soft-delete/GDPR tombstone path, not access control.

**Access control** (`filter_bundle`, `crates/uniko-memory/src/policy.rs`) enforces per-claim visibility against a `Viewer`. The scheme strings on a node's `visibility` property:

| `visibility` value | Admits |
|---|---|
| `null` / `""` / `"public"` | everyone |
| `"private:{participant_id}"` | that participant only |
| `"team:{team_id}"` | members of that team |
| `"org:{org_id}"` | members of that org |
| anything else (e.g. a typo `"secret:42"`) | **no one — fails closed** |

`visibility_admits(Option<&str>, &Viewer) -> bool` is the decision function. A `Viewer { participant_id, teams: HashSet, orgs: HashSet }` is resolved from the graph via `Viewer::new(kb, pid)` (walks `MEMBER_OF`/`PART_OF_TEAM`) or supplied directly with `Viewer::from_parts`.

Four load-bearing invariants:

1. **Only `Fact` and `Observation` are policy-checked.** Every other node type — `Message`, `Chunk`, `Entity`, `Topic`, `Summary`, `Goal`, `Task`, `Episode`, … — carries no scope and passes `filter_bundle` untouched. Do not assume recall hides structural nodes.
2. **Unknown schemes fail closed.** A typo permanently locks a claim down; it can never accidentally widen access.
3. **Recall is fail-open by default.** With `ViewerScope::Unrestricted` (the default for unscoped reads), policy-scoped Facts/Observations are returned **unfiltered** with only a `WARN`. Production callers serving a specific participant **must** set `RecallConfig.viewer`, call `recall_in(.., Scope::default().as_viewer(v))`, or build the facade with `.scope_to_agent()` — otherwise data leaks across visibility boundaries.
4. `filter_bundle` batch-fetches visibilities, retains admitted items, and recomputes `total_tokens = items.len() * 50`.

Beyond visibility, per-call **Dimensions** (`Scope`) apply *hard* filters orthogonal to access control — `sessions`, `participants`, `since`, `until`. These are resolved once into an id allow-set (`resolve_scope_allow_set`) that every candidate query then intersects (`AND id(n) IN $allow`).

### 3.7 Reference: the schema doc vs. the code

- **Authoritative catalog:** `crates/uniko-store/src/schema/constants.rs` (`labels::ALL`, `edges::ALL`).
- **Human-readable catalog:** `website/docs/reference/schema.md` (flat node/edge listing, key fields, polymorphic edges, index families).
- **Drift/temporal semantics narrative:** `website/docs/concepts/facts-and-drift.md` (BTIC, cosine 0.88 clustering, F38 0.40, F39 >4/30d, rule decay 0.95^missed).
- **Visibility narrative:** `website/docs/concepts/visibility.md`.
- **Provenance/derivation narrative:** `website/docs/concepts/data-model.md` and `memory-model.md`.
- **Installed snapshot:** `config/schema.json` — generated by `cargo run --bin export-schema` (`crates/uniko-bench/src/export_schema_main.rs`) and loaded only when a caller sets `UnikoConfig::schema_path`.

When any two of these disagree, the Rust source in `constants.rs` and the per-node `schema/*.rs` registration modules are the ground truth; the docs and the JSON snapshot are downstream artifacts that must be regenerated to match.

---

## 4. Ingestion & Extraction Pipeline (L3)

This chapter covers the *write-time* path: how a raw message, artifact, or PDF becomes a set of typed graph nodes and edges. In uniko this is the work of **Layer 3 — `crates/uniko-extract`**, driven by the **Layer 2 pipeline machinery** in `crates/uniko-pipes`, and persisted through the **Layer 1 `KnowledgeBase`** in `crates/uniko-store`. The governing principle is stated bluntly in the crate docs: **ingest is LLM-free**. Every write-time enrichment — entity recognition, observation reconstruction, chunking, dependency/semantic-role parsing — is deterministic CPU work (regex + tree-sitter + a single small quantized DeBERTa forward pass). The LLM never touches the write path; Facts, Topics, and Procedures are derived later in async consolidation/cortex workers.

This is the "compile" step in uniko's "compile once, query forever" model: raw messages are source code, and extraction compiles them into structured knowledge that recall queries directly, so there is no LLM in the recall hot path either.

### 4.1 Why no LLM per message

The conventional memory stack calls an LLM on every ingested turn to extract entities and facts. uniko rejects this for three reasons made concrete throughout the codebase:

- **Cost and reproducibility.** LoCoMo's 5,882-turn corpus ingests in ~7.5 minutes at **$0** (~76 ms/turn) precisely because no tokens are spent at write time. A deterministic cascade also makes ingest reproducible and offline-capable.
- **Latency and atomicity.** Removing the LLM lets the entire per-message write collapse into one prep-then-commit transaction (§4.4), which cut commits-per-message from 3→1 and is the backbone of the ~300× ingest speedup recorded in the perf journey.
- **Quality is not sacrificed for observations.** Observations are *reconstructed from the parse tree*, not paraphrased by an LLM. `"I'm starting a dance studio"` becomes the clean, speaker-attributed, pronoun-resolved declarative `"Jon is starting a dance studio"` via dependency/SRL patterns (§4.6), which is both cheaper and more predictable than an LLM rewrite.

Where an LLM *can* help — refining `(subject, predicate, object)` triples during P4 consolidation, naming topics in P6, synthesizing cited answers, translating NL→Cypher — it is always **async, optional, and off the write path** (`TripleSource::Llm{alias}` falls back to `SrlDep` on failure).

### 4.2 Where L3 sits in the pipeline

L3 does not own the pipeline runner. The vocabulary and reliability primitives live in `uniko-pipes` (L2); the actual executor lives *downstream* in `uniko-memory`'s `ingest_worker.rs`. L3 supplies the content-processing `Step` implementations.

```
                 submit(IngestTask)
                        │
   ┌────────────────────▼─────────────────────────────┐
   │ uniko-memory::pipeline::ingest_worker (L4)        │
   │  biased select · Semaphore(concurrency=8)         │
   │  per-item child CancellationToken + InflightGuard │
   │  run_step_chain(steps, ctx, dlq)                  │
   └────────────────────┬─────────────────────────────┘
                        │  Vec<Box<dyn Step>>
   ┌────────────────────▼─────────────────────────────┐
   │ uniko-extract::ingest::IngestStep (L3)            │
   │  dispatch on ctx.metadata["ingest_type"]:         │
   │    "message" → ingest_message_atomic  ← HOT PATH  │
   │    "artifact"→ ingest_artifact                    │
   │    "pdf"     → ingest_pdf                          │
   │    "source"  → ingest_source (MIME-sniff route)   │
   └────────────────────┬─────────────────────────────┘
                        │  &KnowledgeBase (L1)
                        ▼
   uni-db  (graph + vector + FTS + BTIC, SSI transactions)
```

`Step` (defined in `uniko-pipes/src/step.rs`) is a small async trait — `name`, `should_run`, `execute(&mut PipelineContext) -> Result<StepOutcome>`. The `PipelineContext` is a per-item mutable bag threaded between steps: `node_id`, `content`, `content_type`, `cancel`, `kb`, `llm_breaker`, and — critically for extraction — `extracted_entities`, `extracted_observations`, and a free-form `metadata` map. `IngestStep` reads its typed payload out of `metadata["ingest_payload"]` and dispatches on `metadata["ingest_type"]`, setting `ctx.node_id`. There is no downstream embedding step — chunks are embedded inside the same atomic write — and the step chain the facade builds is a single element, `vec![Box::new(IngestStep)]`.

Error isolation is per-step (`StepErrorPolicy::{Skip, DeadLetter, Abort}`), but note the atomic message path is itself all-or-nothing internally: a mid-transaction failure rolls the whole message back.

### 4.3 Modality dispatch: `ingest_source`

For arbitrary blobs, `ingest_source` (`ingest/source.rs`) is the front door. It resolves a MIME type through a fixed precedence and routes on the resulting `Modality`:

```
resolve_mime:  explicit override
             → magic bytes (infer crate)
             → file extension (mime_guess)
             → text/plain (if it sniffs as text)
             → application/octet-stream
```

`modality_for_mime` (`uniko-pipes/src/content.rs`) then maps the essence to a coarse `Modality`, which is the single routing key shared by *both* ends of the system — ingest (which chunker/extractor) and recall (per-modality channels). This is why the taxonomy lives in the shared L2 crate.

| `Modality` | Route |
|---|---|
| `Text`, `Code`, `Markup`, `Structured`, `Document` | `ingest_artifact` (with the matching chunker) |
| `Pdf` | `ingest_pdf` |
| `Image`, `Audio`, `Video` | a registered `ModalityExtractor` (empty registry → `UnikoError::Unsupported`) |

Gotcha: unknown/arbitrary binary maps to `Modality::Text` (preserving legacy behavior), so an unrecognized binary blob silently routes to the text chunker unless you supply an explicit `Mime`/`Modality`.

### 4.4 The hot path: `ingest_message_atomic`

`ingest_message_atomic` (`ingest/atomic.rs`) is the core of write-time processing. Its design is **prep-then-commit**: all CPU work and read-only lookups happen *before* a transaction opens; then a single retriable transaction writes everything and commits once. Its result carries the message/chunk/session/sender ids, extracted entities and observations, and an `AtomicTimings` breakdown.

```
ingest_message_atomic(kb, msg, session_ctx):

  (1) idempotency: get_node_by_ext_id("Message", message_id) → early return if present
  (2) ensure_session_and_sender          [cold path only; own commits]
        - guarded by kb.lock_session_setup (avoids dup Session/Participant + SSI aborts)
        - warm path (ids cached in SessionContext) skips all DB work
        resolve_recipients from the in-memory participant cache (no PARTICIPATED_IN query)
  (3) extract_entities_and_nlp           [PURE CPU, no DB]  ← §4.5, §4.6
        rule NER (always)
        + code AST NER (if content_type == "code")
        + ONNX cascade → per-sentence entities + observation prep
        → suppress_onnx_over_structured → admit_entities → deduplicate_raw
  (4) prepare_entity_upsert: canonical entity_ids = id::entity_id(name, type); snapshot now
  (5) kb.lock_entity_ids(entity_ids)     ← acquired BEFORE tx-open, held across commit
  ─── retry loop (SSI-conflict backoff) ───────────────────────────────
  (6) begin_tx
  (7) apply_message_writes_in_tx: Message node + create_message_edges_in_tx + in-tx chunking
  (8) apply_entity_upsert: authoritative existence read UNDER lock → batched
        CREATE(new) / UPDATE(counters) / MENTIONS edges (Message→Entity)
  (9) prepare_observations + apply_observations: Observation nodes + OBSERVED_IN + ABOUT
 (10) tx.commit()
  ─────────────────────────────────────────────────────────────────────
 (11) post-commit ONLY: advance session_ctx.prev_message_nid/ts and sentence_ctx
```

Two concurrency invariants are load-bearing:

- **`entity_id` is non-unique in uni-db**, and uni-db's SSI does *not* catch insert-phantoms (an empty `MATCH` registers no read-set). Two concurrent ingests of the same entity would both read "absent" and both `CREATE` a duplicate. The fix is `lock_entity_ids` (256-stripe async mutexes in `uniko-store/src/locks.rs`) acquired **before** the tx opens and held across the commit, with the *authoritative existence read performed inside the tx under those locks* (`dedup.rs`). Do not move the existence read back into prep.
- **`session_ctx` must not be mutated until after a successful commit** (step 11). Because the retry loop re-runs the closure per attempt, `entity_prep` is cloned per attempt and `prepare_observations` re-seeds from the same `sentence_ctx` each time; mutating context mid-attempt would corrupt pronoun resolution on retry.

`ensure_session_and_sender` is deliberately *not* folded into the atomic tx yet (a documented follow-up); it takes `lock_session_setup` to serialize first-sight creation of `Session`/`Participant`/`PARTICIPATED_IN` and dodge SSI antidependency aborts.

The message edges — `SENT_BY`, `ADDRESSED_TO`, `IN_SESSION`, `NEXT` (carrying `gap_ms`) — are written through `create_message_edges_in_tx`, which uses uni-db's `bulk_insert_edges` fast path rather than the Cypher executor (the measured gap is ~980×: ~150 µs/edge bulk vs ~147 ms/edge Cypher at concurrency 24).

### 4.5 Entity extraction: three sources, then admission + dedup

Entities come from three independently-run sources that are then merged:

| Source | File | When | Confidence | Produces |
|---|---|---|---|---|
| Regex rules | `ner/rules.rs` | always (<10 ms) | per-rule | url, email, date_iso, date_informal, measurement, preference, quoted, proper_noun |
| Code AST (tree-sitter) | `ner/code.rs` | `content_type == "code"` | 0.9 | `CodeSymbol`, `CodeImport` (python/rust/js/ts/tsx) |
| ONNX NER | `ner/onnx.rs` | `onnx` feature | model | NER spans → `RawEntity` via byte-offset `text.find` |

The `EntityType` vocabulary written to the graph is: `Person, Url, Email, Date, Measurement, Preference, QuotedString, Organization, Location, CodeSymbol, CodeImport, Other`. `ExtractionSource` records provenance: `RuleBased, CodeAst, OnnxModel, Llm`.

The three lists pass through a fixed merge/filter pipeline:

1. **`suppress_onnx_over_structured`** — drop ONNX guesses whose byte spans overlap a rule-matched Email/URL (the structured rule is authoritative).
2. **`admit_entities`** (gated by `cfg.entity_strict_admission`, default true) — drop `Date`/`Measurement`/`Preference`/`QuotedString` (noise, not durable entities); gate `Other` by `entity_other_min_confidence` (default 0.9); drop greeting-fragment Persons. With strict admission off, everything is admitted.
3. **`deduplicate_raw`** — collapse by `canonical_name`, keep the max-confidence variant, sum mention counts.

**Canonical identity is the single most important invariant here.** Every NER writer must converge on the same id, computed as:

```
entity_id = "ent_" + hex64( SHA256( lower(name) \0 canonical_type ) )
```

`text::normalize_canonical` (lowercase, strip punctuation, collapse whitespace) is the one name normalizer, and the type must come from the shared lowercase vocabulary. Mixing vocabularies duplicates entities (this is the class behind issue #1). In the graph, `MENTIONS` is multi-source (Message/Chunk/Action/Artifact/Episode → Entity) and carries a `count`; on upsert, existing entities get batched counter UPDATEs (`frequency += mentions`, `confidence = max`, `last_seen`) while new ones get batched CREATEs, and the `MENTIONS` edges are safe plain CREATEs because the Message node is fresh.

### 4.6 The write-time ONNX NLP cascade

This is the analytical engine of L3. The critical architectural fact: **uniko-extract does not own the model.** `NlpPipeline` (`nlp/mod.rs`) resolves the `nlp/default` alias (`NLP_ALIAS`) to a `uni_xervo::NlpModel` via `kb.model_runtime()`. uni-xervo owns tokenization, the **single shared-encoder ONNX forward pass**, and per-head decoding. If the runtime/alias is unavailable, `NlpPipeline::try_new` returns `None` and the whole thing falls back to rule-based extraction.

**The model.** The default is `dragonscale-ai/kniv-deberta-nlp-base-en-xsmall`, **INT8-quantized** (`onnx/cascade-int8.onnx`, in `.uni_cache/onnx-nlp`). One encoder pass feeds all requested task heads, selected via `NlpTasks` bitflags (`tasks()` in `nlp/mod.rs`):

```
        ┌──────────────────────────────────────────────┐
 text → │  SentencePiece tokenizer                      │
        │            │                                   │
        │   ┌────────▼─────────┐  ← ONE shared DeBERTa   │
        │   │  encoder (INT8)  │    forward pass         │
        │   └──┬──┬──┬──┬──┬───┘                         │
        │     POS NER DEP SRL CLS   ← per-head decode    │
        └──────┼──┼──┼──┼──┼────────────────────────────┘
               ▼  ▼  ▼  ▼  ▼
             NlpResult { words, pos_indices, ner spans,
                         dep_arcs, srl_frames, sentence_class, … }
```

- **POS** — 17 UPOS tags.
- **NER** — 37 OntoNotes BIO labels → `NerEntityType` (`Person, Organization, Location, Date, Numeric, Event, Product, WorkOfArt, Group, Misc`).
- **DEP** — 53 Universal-Dependencies relation labels; `dep_arcs` carry a head token index.
- **CLS** — 8 dialog-act labels → `SentenceClass` (`Statement, Question, Command, Greeting, Acknowledgment`).
- **SRL** — 42 PropBank BIO labels; **enabled by default** (`nlp_srl_enabled = true`). SRL costs one extra ONNX forward per VERB per sentence (per-verb fan-out), producing `SrlFrame { predicate_idx, predicate_word, args: [SrlArg{role, text, start_word, end_word}] }`.

The 37/17/53/8/42 label arrays live in an embedded `label_maps.json` (`nlp/assets.rs`, behind a `OnceLock`) and are vendored verbatim from the upstream model's `student_loader.py` — they are the source of truth and must be mirrored, not invented.

**Sentence handling.** `analyze_sentences` splits text (dropping fragments <4 words), then batches *all* sentences into **one** `xervo.analyze()` call while keeping per-sentence token indexing local.

**The adapter — uniko's only real job here.** `adapter::xervo_to_uniko` (`nlp/adapter.rs`) reconciles the three representation gaps that the NLP-parity bench proved are the *only* divergences between uniko's decode and xervo's:

1. **SentencePiece metaspace → words.** A token starting with a space or `U+2581` opens a new word; otherwise it continues the previous (`reconstruct_words`). POS/NER use first-subword-wins.
2. **DEP head token → word.** The head is a 0-based global token index mapped back to a word; a root or self-reference becomes the sentinel `usize::MAX`. Downstream loops filter `arc.head == anchor_idx` and must **never** treat `MAX` as a valid index.
3. **NER BIO → spans.** Raw per-token BIO tags are merged (`merge_word_bio`: `B` or orphan-`I` opens, matching `I` extends, `O` closes) then `build_span`/`parse_ner_type` map OntoNotes+CoNLL tags to `NerEntityType`; SRL inclusive token spans become exclusive word spans.

A known degradation: xervo's `NlpModel` returns only the **top** CLS act + confidence, not a full softmax, so `cls_probs` stays empty and the CLS informativeness gate falls back to top-1 logic rather than the intended multi-label distribution.

### 4.7 From parse to observations

Observations are the structured, speaker-attributed claims that later consolidate into Facts. They are produced in `observations/mod.rs` via a prep/apply split so the whole thing fits inside the atomic transaction.

**`prepare_observations`** resolves the sender, applies a message-level `is_informative` pre-gate *only when there are no per-sentence NLP results* (the model path gates per sentence instead), and then:

- **Model path.** For each sentence's `NlpResult`, `cls_gate_admits` decides admission by reconstructing the informativeness gate from the top-1 CLS signal. Non-admitted sentences still update `SentenceContext` (speaker / other-speakers / last noun subject / last noun object) so pronoun antecedents stay correct. Admitted sentences run through `extract_with_rules` (`rules_engine/matcher.rs`), which runs **both**:
  - **DEP-anchored patterns** — walk each word, match an anchor POS, fill child captures by DEP relation, render a template.
  - **SRL-anchored patterns** — for each `SrlFrame` × each `SrlPattern`, capture by PropBank role and pick the longest template whose variables are all non-empty.

  Output is deduped by rendered text. Subject pronouns are resolved: 1st person → speaker, 2nd → addressee, 3rd/demonstrative → last noun antecedent, unresolvable → drop.

- **Rule fallback** runs only when the model is unavailable *and* there are entity references. Crucially, this fallback sets `predicate = None`/`object = None`, so **only the SRL/DEP model path yields structured triples** — Fact derivation (which filters `predicate IS NOT NULL`) skips fallback observations.

Each `DepObservation` then gets:

- **Predicate normalization** → `normalize_predicate` (snake_case) → `lemmatize_predicate` (strip aux prefixes like `is_`/`has_been_`, irregular-verb table `got→get`, longest-first regular suffix rules `ing→''`, `ed→''`, `ies→y`, with a min-stem guard). This can produce non-word stems (`was_pursuing → pursu`) — intentional and conservative; callers group by the stem.
- **Object cleanup** → `clean_object_phrase` (iteratively strip leading articles/prepositions, reject pure pronoun/stopword/temporal-deictic residues). Light-verb-only triples (`do/make/get/have/be/go` with no specifying complement) are rejected unless they carry a temporal/location/destination/recipient/source complement.
- **Temporal rewrite** → `resolve_temporal_in_content` substitutes the absolute ISO date into the content and fills `temporal_anchor`. It uses `resolve_temporal_with_granularity`, which returns `None` on unparseable phrases — so `"next Whitsun"` is left alone rather than being rewritten to the message timestamp. Temporal families handled: `yesterday`, `last <weekday>`, `N <unit> ago`, `last <month>`, each mapping to a `DateTime` + granularity (Day/Week/Month/Year) → a `[lo, hi)` window.

**`apply_observations`** batch-creates `Observation` nodes (subject `normalize_canonical`'d for grouping, content left human-readable), wires `OBSERVED_IN` (Obs → Message), and wires `ABOUT` edges to the speaker plus any entity whose normalized name subset/superset-matches the normalized subject. That `ABOUT` match is a loose substring test, so it can over- or under-link.

### 4.8 Chunking

Message chunking triggers only when `count_tokens(content) > message_chunk_threshold` (default 1024). Token counting uses tiktoken `cl100k_base` with a fallback. `select_chunker` routes by `Modality`:

| Modality | Chunker |
|---|---|
| `Code` | `CodeChunker` (tree-sitter, `code-parse` feature; else text) |
| `Markup` | `HtmlChunker` (dom_smoothie) |
| `Structured` | `StructuredChunker` (csv/json) |
| everything else | `TextChunker` |

`TextChunker` (`chunking/text.rs`) splits recursively — paragraphs (`\n\n`) → sentences → words — greedily accumulating to `max_chunk_tokens` (default 256), merges sub-`min_chunk_tokens` (default 32) chunks with a neighbor, applies a **char-approximate** (~4 chars/token) overlap at word boundaries, and extracts a markdown heading. Because overlap is prepended *after* the greedy pass and `token_count` is recomputed, final chunks can slightly exceed `max_chunk_tokens`.

`create_chunks_in_tx` batch-creates `:Chunk` nodes with a deterministic `chunk_id(parent_ext_id, index)` and `HAS_CHUNK` edges (carrying `index`) from a dynamic parent label. `ChunkData` carries `text, index, start, end, token_count, chunk_type, language, symbol_name, heading, metadata`.

**Session-level chunking.** Two chunkers in `ingest/session_chunk.rs` build the session-granularity retrieval surfaces: `chunk_session` (transcript chunks, `chunk_type = "session"`) and `chunk_session_observations` (deduped observations → dense chunks, `chunk_type = "observation"`, plus `ABOUT` edges to entities and participants). Both hang their output off the `Session` via `HAS_CHUNK`.

These run **at end of session, not per turn** — `Session::finalize()` invokes both, and `Session::summarize()` calls `finalize()` best-effort, so the common end-of-conversation verb builds them. `Agent::finalize_session(&id)` does the same without a `Session` handle and is the backfill entry point for older KBs (`Agent::unfinalized_session_ids()` enumerates them).

Each chunker takes a `ChunkMode`. `Once` is build-once: it skips entirely when chunks of that type already exist — the semantics `uniko-bench` relies on. `Refresh` (what `finalize` uses) re-chunks and compares the result against the stored text; identical means no write and, crucially, **no re-embedding**, which is what makes it cheap enough to call unconditionally. When the session *has* grown, the old generation is deleted and the replacement written in **one transaction** — chunk writes are plain inserts and `chunk_id` has no uniqueness constraint, so rebuilding without deleting would silently duplicate every chunk.

`finalize` also stamps `Session.ended_at` from the latest message. A Session counts as *open* while `ended_at` is null, so a finalized Session is skipped by the inactivity auto-close sweep; re-finalizing after more turns re-stamps it.

Under `Refresh` the rebuild is **incremental**: chunks are compared index by index and only the suffix from the first mismatch is deleted and re-created, so appending turns re-embeds the tail rather than the whole transcript.

!!! note
    Before 0.2.x these chunkers existed but were called only by `uniko-bench`. Since the Phase 1 session boost walks `Session-[:HAS_CHUNK]->Chunk`, and `phase1_strategy` defaults to `"boost"`, a facade-ingested KB got **no** Phase 1 contribution at all. See §6.

### 4.9 Artifacts and PDFs

**Artifacts.** `ingest_artifact` hashes the content, dedups (by hash, then by `artifact_id`), PUTs the bytes to the blob store, MERGEs an `:ArtifactContent` node, creates an `:Artifact` node with a `HAS_CONTENT` edge, wires provenance/context edges, chunks the content, and computes a **mean-pool `text_embedding`** (element-wise average of child chunk embeddings; dim-mismatch → `None`).

**PDFs.** `ingest_pdf` (`ingest/pdf/mod.rs`) always persists the artifact + blob **even when extraction fails** — the failure is surfaced in `PdfIngestResult.extraction_failure`, *not* as an `Err`, so a caller can re-extract later with a different backend. It chooses a path:

```
use_tiered = cfg!(feature = "pdf-ocr") && ocr.enabled && runtime present
```

- **Legacy (default) text-only path** — `PdfTextExtractor` / `PdfExtractCrate` wraps `pdf-extract` in `catch_unwind` (pdf-extract has ~50 known panics on adversarial input, issue #141 → `PdfExtractError::Panic`). Emits per-page `chunk_type = "page"` chunks with `{page_number, page_count}` metadata and a global index across pages.

- **Tiered doc-IR path** (`pdf/tiered.rs`, `pdf-ocr` feature) — runs `uni-xervo-pdf` with the VLM tier disabled and a `Ceiling(Ocr)` policy (Native+OCR ladder), then materializes a document-IR graph:

  ```
  Artifact ──HAS_PAGE──▶ :Page {plain_markdown, produced_by, block_count, escalations}
                            │
                            └─CONTAINS (reading_order)─▶ :Block {kind, text,
                                 reading_order, confidence_kind+score, bbox_x0..y1}
                            :Block ──NEXT_IN_READING_ORDER──▶ :Block  (sorted)
                            :Block ──HAS_CHUNK──▶ :Chunk {chunk_type="block"}
                                                    └── also HAS_CHUNK from Artifact
  ```

  Each per-block child `:Chunk` is **dual-attached** to both the Block and the Artifact, so `mean_pool_artifact_text_embedding` and artifact-level recall keep working. `Confidence` is `Deterministic | Measured | Derived`; `Tier` is `Native | Ocr | Vlm` (VLM disabled here).

Nodes written across the L3 write paths: `Message, Chunk, Entity, Observation, Session, Participant, Artifact, ArtifactContent, Page, Block`. Edges: `SENT_BY, ADDRESSED_TO, IN_SESSION, NEXT, PARTICIPATED_IN, MENTIONS, OBSERVED_IN, ABOUT, HAS_CHUNK, HAS_CONTENT, ATTACHED_TO, PRODUCED, HAS_PAGE, CONTAINS, NEXT_IN_READING_ORDER`.

### 4.10 Embedding

`embedding/mod.rs` computes vectors over the uni-db xervo aliases (`EMBED_ALIAS` dense, `HYBRID_EMBED_ALIAS` for bge-m3). Helpers: `embed_document`, `embed_query`, `embed_raw`, `embed_batch`, `embed_batch_chunked`, `embed_multivector_query` (ColBERT), `embed_entity` (embeds the string `"name (type)"`), `embed_episode`, `episode_topic_text`.

`embed_batch_chunked` exists because the ONNX BFC arena can OOM on a single large batched forward (~6000 inputs → ~1.3 GB); the default `chunk_size` is 64. A measurement-only escape hatch, `UNIKO_BENCH_NO_MSG_EMBED=1`, pre-populates a zero embedding to skip auto-embed — it invalidates recall and is for benchmarking write throughput only.

The default embedder is `BGESmallENV15` (384-d, query-side prefix only). When a hybrid embedder (bge-m3) is configured, the schema adds `sparse_embedding` (`SparseVector`) and `colbert_embedding` (`List<Vector>`) columns on Chunk/Observation, all pointing at the same `embed/hybrid` alias so uni-db fuses them into one `EmbedHybrid` forward pass. The embedding dimension is fixed at DB creation (it is part of the on-disk vector index), so switching embedders requires a fresh KB.

### 4.11 Build features and testing

L3's behavior is gated by Cargo features:

| Feature | Default | Enables |
|---|---|---|
| `code-parse` | on | tree-sitter code AST NER + code chunking (python/rust/js/ts) |
| `onnx` | off* | the NLP cascade + ONNX NER (`NlpPipeline`, `entities_from_nlp_result`) |
| `pdf-ocr` | off | tiered Native+OCR PDF doc-IR (`uni-xervo-pdf`) |

\* Note the layering subtlety: uni-db's `provider-onnx` is statically linked regardless, so the embedding/NLP *runtime* is always present; the `onnx` feature only adds uniko-extract's ort-backed extraction adapters. Without `onnx`, `NlpPipeline::try_new` returns `None` and extraction falls back to rules.

Run L3 tests with the cascade enabled:

```sh
cargo nextest run -p uniko-extract --features onnx
```

### 4.12 Practical notes and invariants

- **The `onnx` seam is graceful, not fatal.** No model runtime means rule-based NER + rule-fallback observations (no structured triples), which still produces a valid graph — just fewer Facts downstream.
- **Rule-fallback observations carry no triple**, so they never derive Facts. If you need Facts, the SRL/DEP model path must run.
- **`entity_id` locking is non-negotiable.** Acquire `lock_entity_ids` before opening the tx and keep the existence read inside the tx under those locks.
- **`session_ctx` is post-commit only.** Never mutate cross-turn context until the commit succeeds, or retries corrupt pronoun resolution.
- **BTIC/temporal columns** must be read via a bare projection (`RETURN f.valid_at`), never `RETURN n` — Node-wrapped projection stringifies temporals lossily. `temporal_anchor` (the SRL `ARGM-TMP`-resolved date) is a plain `DateTime` on `Observation`, BTree-indexed for temporal recall.
- **`suppress_onnx_over_structured` before `admit_entities`** — the ordering matters: structured rule matches win over model guesses on the same span.
- **PDF failures don't propagate.** Check `PdfIngestResult.extraction_failure`; the artifact and blob are already durably stored.
- **There is a stray `eprintln!` (`RECALL_PROF`)** noted in the recall read layer (L1) — not L3, but it shares the ingest-adjacent read path; treat stderr noise there as known.

---

## 5. Storage, Search & Locy Reasoning (L1)

`uniko-store` (`crates/uniko-store`) is the lowest layer of the uniko stack and the single sanctioned gateway to the graph. Everything above it — `uniko-extract`, `uniko-pipes`, `uniko-cortex`, `uniko-memory` — reaches persistence exclusively through the typed `KnowledgeBase` API defined here. This chapter covers how that façade wraps the embedded `uni-db` engine: the repository/write/search modules, the concurrency discipline that reconciles `uni-db`'s serializable isolation with uniko's hot paths, the bitemporal Fact machinery, pluggable blob storage, and the in-database Locy logic runtime including the `sequence_detector` rule.

### 5.1 The "issue #2" boundary

`uni-db` is meant to be an implementation detail. `uniko-store` enforces that by wrapping it behind `KnowledgeBase` and re-exporting only the handful of `uni-db` types a caller legitimately needs (`lib.rs:54-80`):

| Re-exported | Purpose |
|---|---|
| `Value` | Graph value type at the API boundary |
| `Transaction` | Multi-statement write unit |
| `RetryOptions` | Passed to `transact_with_retry` |
| `temporal::{Btic, TemporalValue}` | Bitemporal interval type |
| `xervo::{GenerationOptions, Message}` | LLM generation seam |
| `ModelAliasSpec`, `ModelTask`, `WarmupPolicy` | Naming a model alias when registering an extra model (what the facade's `LlmSpec` builds) |
| `ModelRuntime` | Shared ONNX/model runtime handle |

A CI grep gate forbids `use uni_db` and `KnowledgeBase::db()` in the `src/` of product crates. `.db() -> &Uni` remains a `pub` escape hatch but is documented as tests/benchmark-only (`storage/mod.rs:315-326`). Reviewed exceptions are tagged `// ALLOW:` on the same line. If a product crate needs a new graph operation, the operation is added *here* rather than by reaching past the seal.

The crate is deliberately **policy-free for scoring**: recall/working-memory/consolidation reads return *decoded* Rust structs (never raw `Record`s), and the ranking math — RRF fusion, tier weights, MMR, coverage — lives one layer up in `uniko-memory`. What lives here is the schema, the graph walks, bitemporal Facts, idempotent id derivation, and the concurrency primitives.

### 5.2 `KnowledgeBase`: the Layer-1 handle

```rust
pub struct KnowledgeBase {
    db: Arc<Uni>,
    config: UnikoConfig,
    kb_stats_lock: Arc<Mutex<()>>,   // serializes :KnowledgeBaseStats RMW
    rmw_locks: Arc<StripedLocks>,    // check-then-create serialization
}
```
(`storage/mod.rs:80-99`). It is `Clone` (the `Arc<Uni>` is shared).

**Constructors** (all validate config, build the xervo catalog via `load_catalog`→file or `embed_catalog`, open `Uni`, `apply_schema`, optionally `prefetch_models`, then `finalize_init`→`init_kb_stats`):

| Constructor | Use |
|---|---|
| `in_memory(cfg)` / `in_memory_with_xervo(cfg, extra)` | Ephemeral KB (tests, benches) |
| `open(path, cfg)` | Persistent KB |
| `open_with_xervo[_no_prefetch]` | Persistent + explicit catalog |
| `build_shared_runtime(cfg, specs)` + `open_with_runtime(...)` | Share ONE `Arc<ModelRuntime>` (one ONNX session / VRAM arena) across many KBs |

`build_shared_runtime` bootstraps a throwaway in-memory `Uni` just to run `uni-db`'s `#[cfg(provider-*)]` gates, extracts the `Arc<ModelRuntime>`, and hands it to many `open_with_runtime` KBs. This is mandatory for concurrent benchmark harnesses: per-KB ONNX sessions OOM an 8 GB GPU past `--question-concurrency 3`, and it is the only open path that lets a BGE-M3 hybrid KB reopen (the catalog-open path rejects non-`Embed` vector-index aliases, `uni-db #130`) (`mod.rs:214-299`).

Persistent-open honors env perf knobs that override `UniConfig`: `UNIKO_WAL_DISABLED`, `UNIKO_AUTOFLUSH_THRESHOLD`, `UNIKO_AUTOFLUSH_INTERVAL_OFF` (`mod.rs:27-50, 196-198`).

### 5.3 Concurrency: SSI + striped locks + retry

This is the central correctness story of L1. `uni-db` runs with **Serializable Snapshot Isolation (SSI)** enabled. Two things follow.

**SSI protects read-modify-write, and `transact_with_retry` handles the aborts.** Two concurrent RMW callers won't lose updates: the second committer aborts with a retriable `SerializationConflict`, surfaced as `UnikoError::Conflict`. `UnikoError::is_retriable()` is true *only* for `Conflict`, and `From<uni_db::UniError>` preserves `uni-db`'s own retriability classification (`error.rs:77-89`).

```rust
pub async fn transact_with_retry<T, F, Fut>(
    &self, opts: RetryOptions, f: F,
) -> Result<T>
where F: FnMut(Transaction) -> Fut,
      Fut: Future<Output = (Transaction, Result<T>)>;
```
It threads the `Transaction` *by value* through the closure to stay `Send` and free of higher-ranked trait bounds. Each attempt gets a **fresh** transaction; commit errors are also retried; backoff is capped exponential with **no jitter** — `base * 2^min(attempt-2, 20)` clamped to `max_backoff` (`mod.rs:379-418, 505-512`). No jitter is deliberate because the hot paths are already pre-serialized by locks. Because the closure re-runs per attempt, any consumed input must be re-usable across attempts (capture by `Copy` or re-clone inside).

**SSI does *not* catch insert-phantoms — this is the load-bearing subtlety.** An empty `MATCH` registers no read-set, so a check-then-create on a **non-unique** index (`entity_id`, `content_id`, `session_id`, `fact_id`, and `merge_node`) lets two callers both read "absent" and both `CREATE` a duplicate. This is the bug class behind issue #1 (entity duplication). The fix is `StripedLocks`.

```
StripedLocks: 256 × tokio::Mutex, indexed by DefaultHasher(key) % 256
canonical keys (byte-prefixed):
  entity:<id>  content:<id>  session:<id>  participant:<id>  fact:<id>  node:<id>
```
(`locks.rs`). Guards must be held across **both** the existence re-read **and** the commit. Every row family has ONE canonical lock-key builder; a divergent namespace silently breaks serialization. External guards are acquired **outside** `transact_with_retry` so a single writer per key holds across *all* retry attempts:

```rust
let guards = kb.lock_entity_ids(&[eid.clone()]).await;      // held across the whole tx
let out = kb.transact_with_retry(RetryOptions::default(), |tx| async move {
    let existing = kb.fetch_entities_for_upsert_in_tx(&tx, &[eid.clone()]).await;
    // create_node_in_tx / batch_update_entity_counters_in_tx ...
    (tx, Ok(()))
}).await?;
drop(guards);
```

`lock_many` sorts and dedups by **stripe index** (not key bytes), giving both a global acquisition order (prevents AB/BA deadlock) and self-deadlock safety when two distinct keys collide on one non-reentrant stripe (`locks.rs:110-142`). The mutexes are **non-reentrant**: acquiring the same stripe twice self-deadlocks — always use `lock_many` for multi-key acquisition, never a hand-rolled per-key loop. `batch_upsert_facts` learned this the hard way; a per-fact loop self-deadlocked past ~50 facts (`operations/facts.rs:330-338`).

`bump_modality_presence` is a separate case: it uses the single shared `kb_stats_lock` because `uni-db` `Map` props are atomic-per-column (no per-key `SET`), so the full 4-entry map is round-tripped under lock (`kb_stats.rs:178-240`).

```
concurrent same-entity ingest, no lock:
  A: MATCH (e{entity_id:X}) -> ∅   B: MATCH (e{entity_id:X}) -> ∅
  A: CREATE e{X}  commit OK        B: CREATE e{X}  commit OK   ← DUP (SSI blind: empty read-set)

with lock_entity_ids([X]):
  A holds stripe(X) ─ re-read ∅ ─ CREATE ─ commit ─ drop
                                   B waits ────────────► re-read HIT ─ UPDATE ─ commit
```

### 5.4 Write paths: Cypher vs bulk

Generic CRUD builds **parameterized** Cypher via `build_inline_props` / `build_set_clause` (`mod.rs:829-879`). Every interpolated *identifier* (label, edge type, property name) is gated by `validate_label` / `validate_edge_type` / `validate_property_name`; *values* are always `$pN`-bound.

| API | Path | Notes |
|---|---|---|
| `create_node[_in_tx]`, `get_node`, `get_node_by_ext_id`, `update_node`, `delete_node`, `merge_node`, `query_nodes(label, Filter, limit)` | Cypher | `merge_node` is locked check-then-create |
| `create_edge`, `create_edges[_in_tx]`, `get_edges[_filtered]`, `delete_edge[s_between]`, `update_edge` | Cypher | |
| `batch_create_nodes[_in_tx]` → `tx.bulk_insert_vertices` | **Bulk** | bypasses Cypher executor |
| `batch_create_edges_fast[_in_tx]` → `tx.bulk_insert_edges` | **Bulk** | |
| `create_message_edges_in_tx` | **Bulk** | hand-writes SENT_BY / IN_SESSION / ADDRESSED_TO / NEXT (`edges.rs:193-279`) |

The bulk fast paths bypass the Cypher executor entirely because VIDs are already known. The comment at `batch.rs:225-235` records a measured **~980×** gap (bulk ~150 µs/edge vs Cypher ~147 ms/edge at session-concurrency 24) — pure per-row Cypher-executor overhead. Trade-offs of the bulk path: it **does not** return allocated EIDs and **does not** re-validate property names, so callers must validate keys up front (`batch.rs` does). When EIDs are needed, `create_edges_in_tx`'s `return_ids` arm falls back to UNWIND-Cypher; it groups mixed edge types by `BTreeMap` and consults a static `edge_type_label_hints` map (only relevant on the return_ids Cypher arm; inert on bulk).

`NodeId`/`EdgeId` are `i64` aliases; conversion to `uni-db` `Vid(u64)`/`Eid(u64)` happens at the storage boundary (`types.rs:1-11`).

### 5.5 Schema

`schema/constants.rs` is the single source of truth: **25 node labels** (`labels::ALL`) and **54 edge types** (`edges::ALL`), organized by cognitive layer. `config/schema.json` is a generated snapshot of exactly this (`cargo run --bin export-schema`) and is only consulted when `UnikoConfig::schema_path` points at it.

**Node labels (25), by layer:**

```
L0  Participant
L1  Goal  Task  Session
L2  Message  Action  Episode
L3  Artifact  ArtifactContent  Chunk  Page  Block
L4  Entity  Observation  Fact  Topic  Summary
L5  Procedure  Rule  Pattern
L6  ConsolidationCycle  DeadLetter
L7  Organization  Team
L8  KnowledgeBaseStats
```

**Edge types (54), by layer:**

```
L1  OWNED_BY PARENT_GOAL PART_OF ASSIGNED_TO DEPENDS_ON SUBTASK_OF FOR_TASK FOR_GOAL PARTICIPATED_IN
L2  SENT_BY ADDRESSED_TO IN_SESSION NEXT PERFORMED_BY TRIGGERED_BY PRODUCED NEXT_ACTION
    RECORDED_BY INVOLVES MENTIONS FOLLOWED_BY
L3  HAS_CHUNK HAS_CONTENT HAS_PAGE CONTAINS NEXT_IN_READING_ORDER CREATED_BY MODIFIED_BY ATTACHED_TO
L4  CONTRADICTED_BY OBSERVED_IN OBSERVED_DURING ABOUT SUPPORTED_BY DERIVED_BY DERIVED_FROM
    INVALIDATES SHARED_FROM BELONGS_TO SUMMARIZES
L5  OPERATES_ON USED_IN SUPERSEDES COVERS
L6  PROCESSED INVOLVED CREATED REINFORCED INVALIDATED PROMOTED APPLIED_RULE
L7  MEMBER_OF PART_OF_TEAM TEAM_IN_ORG
```

`register_schema` (`schema/mod.rs:180-232`) is **two-phase and idempotent**: Phase 1 registers every node type's properties + indexes; Phase 2 registers all edge types; then a single `.apply()`. Each node type is a submodule exposing `register_labels(builder, config)` / `register_edges(builder)`.

**Embed aliases** (defined as consts in `schema/mod.rs`): `EMBED_ALIAS`, `HYBRID_EMBED_ALIAS`, `NLP_ALIAS`, `RERANK_ALIAS`, `OCR_ALIAS`.

**Vector-index variants** are built from config:

- `vector_index` (no auto-embed) — for app-computed embeddings (Entity, Fact): the crate computes the vector and writes it.
- `auto_embed_vector_index` — for `uni-db`-computed columns (Message, Summary): `uni-db` runs the embedder on write.
- Hybrid (BGE-M3) additionally adds a `sparse_embedding` (`SparseVector`) column and a `colbert_embedding` (`List<Vector>`) column on Chunk/Observation, **all three pointing at the same `embed/hybrid` alias and the same source property** so `uni-db` fuses them into ONE `EmbedHybrid` forward pass. ColBERT uses `VectorAlgo::Flat` — per-token vectors feed only MaxSim rerank, never first-stage search (`schema/chunks.rs:56-104`, `schema/mod.rs:94-169`).

A hybrid model implements only `HybridEmbeddingModel`, so it **cannot** back lone-dense columns. This is why a hybrid deployment needs **two** aliases: `embed/default` (dense, `ModelTask::Embed`) and `embed/hybrid` (`ModelTask::EmbedHybrid`), both backed by the same model — the model loads twice (~2× VRAM). Sparse/ColBERT columns exist only when `config.embedding.{sparse,multivector}_dimensions` is `Some`; `config.validate()` rejects `recall_sparse_enabled` or `reranker.style = colbert` without them.

### 5.6 IDs (`id.rs`)

The single identity source (issue #1 — divergent id derivation duplicates rows):

| Function | Derivation |
|---|---|
| `new_id()` | UUIDv7 (time-sortable) |
| `entity_id(name, type)` | `"ent_" + hex64(SHA256(lower(name) \0 canonical_type))` (`id.rs:114-120`) |
| `stable_hex64(prefix, bytes)` | `"{prefix}_{u64:016x}"` over the **leading 64 bits** of the SHA-256 digest — a 16-hex-char id ("hex64" = 64-bit, not 64 hex chars) (`id.rs:66-98`) |
| chunk/page/block ids | deterministic from `(parent_ext_id, index)` |

`entity_id` is the canonical entity identity: **all** NER writers must pass a type from the ONE shared lowercase vocabulary (`person`/`organization`/`location`/…). Mixing vocabularies duplicates entities. `text::normalize_canonical` (lowercase, strip punctuation, collapse whitespace) is the single name-key normalizer.

### 5.7 Search

Two families live here: the generic `SearchResult`-returning methods and the richer decoded **recall** reads (`repository/recall.rs`).

**Generic search:**

```rust
pub struct SearchResult { node_id: i64, node_type: String, score: f64, properties: HashMap<..> }
```

`vector_search` / `fulltext_search` wrap the `uni.vector.query` / `uni.fts.query` procedures and decode `(node, score, vid)` rows via `rows_to_search_hits` (`search/mod.rs:35-56`). Also: `multi_type_vector_search`, `multi_field_fulltext_search`.

**Hybrid search + RRF.** `hybrid_search` fetches `2 × top_k` candidates from each channel and fuses via Reciprocal Rank Fusion (`search/hybrid.rs:161-196`):

```
score(node) = Σ_lists  weight_i / (RRF_K + rank_in_list_i)      RRF_K = 60, rank 1-indexed
```
Items appearing in multiple lists accumulate. `hybrid_search_weighted(&[SearchTarget])` additionally scales fused scores by a per-target **Tier** weight (`hybrid.rs:33-45`):

| Tier | Node classes | Weight |
|---|---|---|
| Semantic | Fact, Topic | 1.0 |
| Procedural | Procedure | 0.9 |
| Episodic | Episode | 0.7 |
| KnowledgeBase | Observation, Chunk | 0.5 |
| Provenance | Action, Message | 0.4 |

**Traversal + PPR** (`search/traversal.rs`): `traverse` (BFS, variable-length relationships), `shortest_path`, and a Rust-side **weighted Personalized PageRank**. `personalized_pagerank_weighted` pulls the whole adjacency in one Cypher `MATCH (a)-[r]->(b)` then runs power iteration in Rust: each source splits its score by the sum of edge weights, where each edge contributes `μ(ℓ)` (a per-edge-type multiplier) × `r.weight` (Hindsight-style), teleporting to seeds, converging at `max-Δ < 1e-6` or `max_iter` (`traversal.rs:208-333`). Semantic edges (ABOUT/MENTIONS/SUPPORTED_BY) are weighted above structural edges (IN_SESSION/FOLLOWED_BY).

**Recall reads** (`repository/recall.rs`, the largest read module) return decoded `ScoredRow`/`TemporalRow`/`NodeContentRow`:

- `resolve_scope_allow_set(ScopeFilter)` — unions per-anchor-type MATCH arms (Message/Observation/Episode/Chunk) into an id allow-set; each candidate query then filters with `AND id(n) IN $allow`.
- `recall_vector_search`, `recall_fulltext_search`, `recall_sparse_search` (`uni.sparse.query`), `recall_colbert_maxsim`. Sparse/ColBERT restrict at the *index* level via a `_vid IN (...)` SQL filter string — `_vid` equals Cypher `id(n)` (`recall.rs:98-105, 368-450`).
- `temporal_window_hits` — `btic_overlaps(f.valid_at, $qbtic)` UDF for Facts; BTree range for Observation/Episode.
- `recall_chunk_and_entity_scoped`, `resolve_entity_seeds`, `any_unstable_entities` (drift override signal), `fetch_node_contents`, `query_cypher` (read-only decoded rows).

> Gotcha: `recall_chunk_and_entity_scoped` still ships an `eprintln!` `RECALL_PROF` profiling line (`recall.rs:857`) that writes to stderr in production. The recall sub-queries are best-effort/infallible — failures are logged at debug and skipped.

### 5.8 Bitemporal Facts (BTIC)

A Fact carries a single `valid_at` column of `DataType::Btic` — a half-open interval `[lo, hi)` with per-bound `Granularity` and `Certainty` packed into the meta; `POS_INF` hi means an open interval (still-valid) (`schema/btic.rs`). Two time axes: **valid time** (BTIC lo/hi) and **transaction time** (`uni-db` `_created_at`/`_updated_at`).

BTIC helpers (`schema/btic.rs`): `btic_active`, `btic_query_window`, `btic_invalidate`, `btic_contains`/`overlaps`/`before` (Allen algebra), `btic_upgrade_certainty`. `CERTAINTY_THRESHOLD = 10`.

```
Allen overlap (btic_overlaps UDF):  a.lo < b.hi  &&  b.lo < a.hi
```

**Fact upsert** (`operations/facts.rs`):

- `upsert_fact_by_triple` locks `fact:<id>`, looks up by hash-indexed `fact_id`, and on reinforce sums `observation_count`, recomputes Laplace-smoothed confidence `(n+1)/(n+2)` (n=0 → 0.5, asymptotes to 1.0 without reaching), and upgrades the lo-bound certainty `Approximate → Definite` once cumulative count ≥ 10 (idempotent).
- `batch_upsert_facts(Vec<FactUpsertInput>)` does within-batch dedup-by-`fact_id` (first-occurrence wins for fields, counts summed) then 3 phases: batched MATCH → `batch_create_nodes` for new → UNWIND `SET` with `COALESCE(u.valid_at, n.valid_at)` for reinforce.
- `attach_supported_by`, `write_consolidation_cycle` (audit node), `fact_valid_at`, `fact_id_for`, `laplace_confidence`.

**F38 contradiction / F39 drift:** `find_stale_open_facts[_batched]` (via `btic_is_unbounded`) closes stale *open* Facts and wires `INVALIDATES`. `record_entity_invalidation` counts `INVALIDATES` edges in a rolling 30-day window and flips `Entity.unstable = true`; recall's `any_unstable_entities` reads that flag to force a deeper recall phase (the drift override).

> **Critical readback rule:** BTIC columns must be read via a **bare projection** (`RETURN f.valid_at`), never `RETURN n`. Node-wrapped projection stringifies Temporals *lossily*. `extract_btic` and this readback pattern are reused across facts/deletion/recall.

### 5.9 Blob storage (`blob_store/`)

Content-addressed backends for `:ArtifactContent` bytes, behind an async `BlobStore` trait selected by the serde-tagged `BlobStorage` enum:

| Backend | Storage | Notes |
|---|---|---|
| `Lance` (default) | inline in `:ArtifactContent.bytes` | `LanceBlobStore::get` is deliberately an **error** — bytes are fetched via Cypher (`fetch_blob`), not the backend |
| `Fs` | local content-addressed store | two-level hex fanout `<aa>/<bb>/<content_id>` |
| `S3` | S3/R2/GCS/MinIO via `object_store` | same key layout |

The content-addressed key is the SHA-256 hex; PUT is idempotent on a matching size (`blob_store/mod.rs:161-168`). API: `sha256_hex`, `put_blob`, `merge_artifact_content(MergeContent)`, `fetch_blob`. The chosen backend is persisted in `:KnowledgeBaseStats`; **reopening a KB with a different `blob_storage.kind` is a HARD error** — there is no implicit migration (`kb_stats.rs`).

### 5.10 Deletion & GDPR (`storage/deletion.rs`)

Two families, each run as one `transact_with_retry` unit:

- **Soft-forget** sets tombstones: `redacted = true` on Message/Chunk, or `visibility = 'redacted:forgotten'` on Observation/Fact-supported-only-by-this-turn. Recall post-filters these out (`fetch_redacted`). Verbs: `forget_message`, `forget_participant`.
- **Hard delete** DETACH-deletes the owned subtree, splices the `NEXT` chain (summing `gap_ms`), and re-evaluates any Fact that loses its *last* supporting Observation — those Facts get BTIC-closed (soft-invalidated), **never deleted**, preserving audit. Verbs: `delete_message`, `delete_session_cascade`, `delete_artifact`, `purge_all` → `DeletionReport`.

Shared nodes (Entity, Participant, Session, Topic, deduped ArtifactContent) are never cascaded; a deduped ArtifactContent is dropped only when `refs == 1`. GDPR `forget_participant` force-invalidates Facts *about* the participant regardless of remaining support (`subject_erased`).

### 5.11 The Locy logic runtime

Locy is `uni-db`'s in-database logic-programming layer. `uniko-store` exposes it as a thin seam over `db.rules()` and `session.locy_with(program)`:

| Method | Behavior |
|---|---|
| `create_rule(program)` | Idempotent registration; duplicate program is a no-op |
| `execute_rule(program, params) -> Vec<Record>` | Runs a program body |
| `query_rule(name, cols, params)` | Builds `QUERY <name> RETURN ...` — the supported way to invoke a *registered* rule by name |
| `list_rules`, `delete_rule`, `explain_rule` | Lifecycle/introspection |
| `assume(block) -> AssumeBuilder` | Hypothetical fork → query → rollback |
| `abduce(program, params) -> AbductionResult` | Ranked candidate modifications |

**ASSUME** (hypothetical reasoning) splices the caller's query into the `THEN { }` body of the assume block, forks the graph, runs the query, and rolls back — the graph is never mutated:

```rust
let rows = kb.assume("ASSUME { CREATE (:Fact {subject:'server',predicate:'port',object:'9090'}) }")
    .then_query("MATCH (f:Fact {subject:'server'}) RETURN f")
    .run().await?;
```
Rows are pulled from `CommandResult::Assume | Query` — *not* `rows()`, which only finds the `Query` goal.

**ABDUCE** decodes `CommandResult::Abduce` into an `AbductionResult` holding ranked `AbducedModification`s (field `modifications: Vec<AbducedModification>`) with fixed costs: `RemoveEdge 1.0` / `ChangeProperty 0.5` / `AddEdge 1.5`.

**Locy is not Cypher.** Several documented constraints shape how rules must be authored:

- A rule uses a **single comma-joined MATCH** — a second `MATCH` clause is a parse error.
- Aggregates are `expr AS name` (no `VALUE` keyword).
- A `$param` in a **post-FOLD HAVING** does not resolve (RC12) — threshold classification must be pushed into the Rust consumer.
- Passing a bare rule *name* to `execute_rule` fails (it tries to parse the name as a program); use `query_rule` to invoke a registered rule (RC12).

Other worked-around `uni-db` quirks surfaced through this crate: `close_inactive_sessions` avoids the Cypher `max(DateTime)` aggregate (serialized as `LargeBinary`, breaks readback); `mean_episode_importance` is computed in Rust because `FOLD AVG` returns `0.0` under alias rename (`uni-db #145`); `datetime_value` is preferred over string coercion for DateTime writes.

### 5.12 The `sequence_detector` rule (P5 procedure promotion)

The canonical example of a rule running in-database is `sequence_detector`, which drives procedure promotion. Its source string is defined **once** (in `uniko-cortex`, `SEQUENCE_DETECTOR_RULE`) and referenced by both `uniko-memory`'s startup registration and the P5 sweep so the two cannot drift, but it is *executed* through this L1 Locy seam:

```
CREATE RULE sequence_detector AS
  MATCH (e1:Episode)-[:FOLLOWED_BY]->(e2:Episode),
        (e1)-[:RECORDED_BY]->(p:Participant {participant_id: $agent_id})
  WHERE e1.outcome = 'success' AND e2.outcome = 'success'
  FOLD n = COUNT(*)
  YIELD KEY e1.action_type AS action_a, KEY e2.action_type AS action_b,
        n AS success_count
```

It matches every `FOLLOWED_BY`-adjacent pair of Episodes both recorded by the target agent and both with `outcome = 'success'`, groups by the `(action_a, action_b)` action-type pair (the two `KEY` columns), and counts occurrences into `success_count`. Three Locy constraints are visible here and were once latent bugs that kept the rule from registering:

1. The two relationships are **one comma-joined MATCH** (a second `MATCH` would be a parse error).
2. Aggregates use `expr AS name` (`n AS success_count`), no `VALUE`.
3. There is **deliberately no HAVING/threshold** in the rule — it surfaces *all* recurring pairs, and the Rust consumer (`upsert_procedure`) applies the `promote_threshold` (default 3), because a `$promotion_threshold` in a post-FOLD HAVING will not resolve (RC12).

Invocation, from L1's perspective, is:

```rust
kb.create_rule(SEQUENCE_DETECTOR_RULE).await?;               // idempotent
let rows = kb.query_rule(
    "sequence_detector",
    &["action_a", "action_b", "success_count"],
    &params!{ "agent_id": agent_id },
).await?;                                                    // -> QUERY sequence_detector RETURN ...
```

`query_rule` is the *only* correct entry point: it builds `QUERY sequence_detector RETURN action_a, action_b, success_count`. Each returned row `(action_a: String, action_b: String, success_count: i64)` becomes/reinforces a `:Procedure`; the count is the repetition of that success-pair across all episode chains, and a pair first crosses `candidate → active` once that count reaches the threshold. The producer side of this graph is `uniko-memory`'s `record_episode`, which wires `RECORDED_BY → Participant` and `FOLLOWED_BY` (within a 1-hour window, carrying `gap_ms`) between consecutive same-agent episodes — exactly the `(e1)-[:FOLLOWED_BY]->(e2)` + `RECORDED_BY` shape the rule reads.

### 5.13 Configuration (`config.rs`)

`UnikoConfig` is the root runtime config (`Serialize`/`Deserialize`, struct-level `#[serde(default)]` so empty `{}` deserializes to `Default` — guarded by a regression test), with `validate()`. Its opinionated, spec-mandated defaults:

| Area | Default |
|---|---|
| Embedding | `BGESmallENV15`, 384-d, query-side prefix only (presets: nomic 768d, minilm 384d, bge-small 384d **default**, bge-large, bge-m3 hybrid, embeddinggemma) |
| Reranker | **enabled**, `cross-encoder/ms-marco-MiniLM-L-6-v2`, top_n 50, sigmoid on; style ∈ {cross-encoder, generative, colbert} |
| NLP | kniv-deberta xsmall INT8; `nlp_srl_enabled = true` |
| Vector index | `HnswSq{m:16, ef_construction:100}`, metric Cosine (`VectorAlgorithm` ∈ HnswSq/Flat/Pq, IvfSq/Pq/Rq) |
| Blob storage | `Lance` |
| Recall | limit 15, token_budget 8192, vector/bm25 0.5/0.5, rrf_k 60, phase1_strategy `boost` α=0.6, phase1/phase2 coverage gates 0.75/0.65, phase2 MMR λ 0.7, duplicate 0.85 |
| Entities | `entity_strict_admission = true`, `entity_other_min_confidence = 0.9` |
| Consolidation | threshold 20, interval 900 s |
| Chunking | message_chunk_threshold 1024, max/min chunk tokens 256/32 |
| Sessions | `session_inactivity_secs = 3600` |

External overrides: `catalog_path` (xervo model catalog) and `schema_path` (installed-schema snapshot). Note that the MMR/coverage/phase-cascade *thresholds* are configured here but the algorithms that consume them run one layer up in `uniko-memory`.

### 5.14 Errors

`UnikoError` is the unified error: `Storage`, `Search`, `Schema`, `Pipeline`, `Locy`, `Config`, `Embedding`, `Llm`, `Timeout`, `Conflict`, `Internal`, `Unsupported`. `is_retriable()` is true **only** for `Conflict`; `From<uni_db::UniError>` preserves `uni-db`'s retriability classification so the retry loop behaves correctly.

### 5.15 Invariants cheat-sheet

- **The seal is CI-enforced.** Product crates reach the graph only through repository/operations/model/search methods or `begin_tx` driven by validated `*_in_tx` helpers. `.db()` is tests/bench only.
- **SSI misses insert-phantoms.** Hold the correct `StripedLocks` key across *both* the existence re-read *and* the commit; every row family has ONE canonical lock-key builder. Use `lock_many` (dedups by stripe index); the mutexes are non-reentrant.
- **BTIC → bare projection only** (`RETURN f.valid_at`), never `RETURN n`.
- **Entity identity is `entity_id(name, canonical_type)`** from the ONE lowercase vocabulary; mixing vocabularies duplicates entities.
- **Hybrid needs two aliases** (`embed/default` dense + `embed/hybrid`) backed by one model; sparse/ColBERT columns exist only with `sparse/multivector_dimensions` set.
- **Bulk writes skip the executor** (~980× faster) but return no EIDs and skip property-name validation — validate keys up front.
- **Locy ≠ Cypher:** single comma-joined MATCH, `expr AS name` aggregates, no `$param` in post-FOLD HAVING, `query_rule` (not `execute_rule`) to invoke a registered rule by name.
- **Acquire external locks outside `transact_with_retry`** so a single writer per key holds across all attempts; the closure re-runs per attempt, so re-clone consumed inputs.

---

## 6. Memory, Recall & Consolidation (L4)

`uniko-memory` is the crate an end user actually holds. Everything below it — `uniko-store` (the sole uni-db boundary), `uniko-pipes` (the `Step`/worker/circuit-breaker machinery), `uniko-extract` (ingest, NLP, embeddings), and `uniko-cortex` (P5 procedures, P6 topics) — exposes primitives. Driving them directly means hand-assembling a `KnowledgeBase`, a `PipelineSystem`, catalogs, and an agent. This crate hides all of that behind one owning `Uniko` handle and four cognitive subsystems:

1. **Recall** — a 3-phase Compact→Expand→Broaden cascade with coverage gating, RRF fusion, optional cross-encoder/ColBERT rerank, temporal + graph-PPR + cross-modal channels, MMR dedup, and access-control/redaction filtering (`recall/mod.rs`, ~2289 lines).
2. **Consolidation** — P4 derivation of Facts from Observations by grouping on `(subject, predicate)`, cosine-clustering object surface forms, mode-voting a canonical object, upserting bitemporal Facts, and running F38 contradiction + F39 entity drift (`consolidation.rs`).
3. **The pipeline** — a `PipelineSystem` orchestrating an ingest worker and a consolidation worker whose loop also fires the cortex sweep (P5/P6), rule execution, rule decay, memory decay, and session maintenance (`pipeline/`).
4. **Rules** — registration of 4 stdlib Locy rules plus a confidence-driven lifecycle for authored/induced rules (`rules/`).

A layering note (`lib.rs:6-18`): `uniko-memory` depends on `uniko-cortex` even though cortex is nominally "Layer 5". Layer numbers describe **cognitive altitude, not build order**. Consolidation (P4) is the heartbeat that triggers cortex sweeps, so the trigger policy lives in the consolidation worker here. `recursion_limit` is bumped to 256 to clear an `E0275` on the consolidation worker future's `Send` check.

The remaining modules are the **agent-tool free functions** — each a `(&KnowledgeBase, agent_id, params) -> Result` function that the `Agent`/`Goals`/`Session` facades bind and re-expose as methods. These cover explicit subjective acts (goals, tasks, episodes, actions, direct fact assert/invalidate) that cannot be inferred from message content, plus session summaries, query-episode recording, NL→Cypher translation, and LLM-refined triple extraction.

---

### 6.1 The `Uniko` facade — orchestration

`Uniko` (`facade/uniko.rs`) is the single owning handle. It is clone-cheap (an `Arc` store inside), and it registers the 4 stdlib rules on open.

```rust
use uniko_memory::{Uniko, Turn, LlmSpec, Scope};

let memory = Uniko::builder()
    .path("./agent-memory")                       // or .in_memory()
    .llm(LlmSpec::openai("answerer", "gpt-4o-mini", None))  // enables answer()
    .scope_to_agent()                             // recall filtered by viewer
    .streaming(true)                              // wire the async pipeline
    .build().await?;

let agent = memory.agent("assistant-1");
```

`UnikoBuilder` is chainable: `.path`/`.in_memory`, `.embedding(EmbeddingConfig)`, `.llm(LlmSpec)`, `.raw_config`, `.streaming(bool)`, `.scope_to_agent`/`.scope(RecallScope)`, `.extractor(Arc<dyn ModalityExtractor>)`, then `.build().await`.

`LlmSpec` registers the generation model for the answer path. It lowers to a uni-db `ModelAliasSpec` (task `Generate`, `Lazy` warmup):

| Constructor | Purpose |
|---|---|
| `LlmSpec::openai(alias, model, base_url)` | OpenAI-compatible endpoint |
| `LlmSpec::openai_with_key_env(alias, model, key_env, base_url)` | Same, with API key from an env var |
| `LlmSpec::mistralrs(alias, model)` | On-device local LLM (GPU wheels only) |

**Handle graph.** `Uniko::agent(id)` returns an `Agent` bound to `(kb, agent_id)`, which caches its resolved `Viewer` for `AsAgent` scope. From `Agent` you reach `session(id) -> Session`, `goals() -> Goals`, `data() -> Data`, plus the read verbs and Locy surface.

**Shutdown ordering is load-bearing.** `Uniko::shutdown` skips the pipeline drain (with a `WARN`) if any `Agent`/`Session` clone still shares the `Arc<PipelineSystem>` (`uniko.rs:140-161`). Drop all handles first, or the pipeline is never drained.

`RecallScope` (Unrestricted / AsAgent / As) is resolved to a `ViewerScope` at recall time (`facade/mod.rs`).

---

### 6.2 The recall cascade

`recall(kb, query, &RecallConfig) -> ContextBundle` (`recall/mod.rs:688`) is a thin wrapper that runs `recall_unfiltered()` and then applies, in order:

1. `filter_redacted` — unconditional soft-forget drop.
2. viewer access-control filter — **fail-open** with a `WARN` when `ViewerScope::Unrestricted` still returns `Fact`/`Observation` items.
3. `populate_sources` — stamps lineage onto the small final bundle.

`recall_unfiltered` (`:925`) short-circuits on an empty query, resolves `Dimensions` into an id allow-set once via `resolve_scope_allow_set`, builds an `IntentProfile` via `build_intent_at`, then runs the three phases.

```
                    ┌──────────────────────────────────────────┐
   query ──▶ Intent │  variants · entity_refs · answer_type ·   │
                    │  temporal_window · query_modalities        │
                    └──────────────────────────────────────────┘
                                     │
        ┌────────────────────────────┼───────────────────────────────┐
        ▼                                                             │
  PHASE 1  COMPACT  (consolidated Semantic/Procedural tier)          │
   Fact top-20 · Procedure top-10 · Topic top-5   [vector]          │
   coverage ≥ 0.75  AND  ≥3 items ?                                  │
        │  yes ─ and NOT drift-forced ─▶ early-exit  phase1_only ────┤
        │  no / F58 drift override                                   │
        ▼                                                             │
  PHASE 2  EXPAND  (episodic, parallel fan-out)                      │
   Episode/Obs/Msg vector + Obs/Msg fulltext                        │
   + sparse-Obs + cross-modal + temporal(BTIC) + graph-PPR          │
   per-source min-max norm ─▶ RRF ─▶ tier-weight ─▶ MMR dedup       │
   coverage ≥ 0.65  AND  ≥3 items ?                                  │
        │  yes ─▶ early-exit phase2_only (+ up to 3 Phase-1 Facts) ──┤
        │  no                                                        │
        ▼                                                             │
  PHASE 3  BROADEN  (per-variant hybrid fan-out)                     │
   recall_chunk_and_entity_scoped (vec+BM25+entity-scoped) ×variants │
   + optional sparse Chunk ─▶ RRF ─▶ RecallItems (tier weights)     │
   ─▶ optional rerank (ColBERT MaxSim OR cross-encoder)             │
   ─▶ answer-type boost ─▶ Phase-1 contribution (Merge/Boost/Off)   │
        ▼                                                             │
   finalize_bundle: sort desc · truncate to limit · sum 50 tok/item ◀┘
```

#### Phase 1 — Compact (`phase1_compact`, `:1394`)

Vector search over the **consolidated** tier only: `Fact.embedding` top-20, `Procedure.embedding` top-10, `Topic.embedding` top-5. (Procedure/Topic return 0 rows until P5/P6 have run.) Coverage is computed against `COVERAGE_GATE_PHASE1 = 0.75` **and** requires `>= 3` items.

**F58 drift override** (`:975`): if Phase 1 would exit early but a query `entity_ref` resolves to an `unstable` Entity (`kb.any_unstable_entities`), the cascade is forced past Phase 1 into Phase 2+. This makes queries about volatile entities consult recent episodic evidence rather than trusting a stale consolidated Fact.

#### Phase 2 — Expand (`phase2_expand`, `:1472`)

Fans out **in parallel** (`join_all`):

- Episode / Observation / Message vector search
- Observation / Message fulltext (BM25)
- optional sparse-Observation channel (`uni.sparse.query`)
- cross-modal image/audio/multimodal channels — gated (`phase2_activation`, `:1635`) on `modality_presence` AND the config toggle AND the query carrying a per-modality vector (dormant in text-only corpora)
- temporal channel (`phase2_temporal`, `:1855`) via `temporal_window_hits` (BTIC overlap for Facts, BTree range for Observation/Episode)
- graph channel (`phase2_graph_activation`, `:1918`) — `personalized_pagerank_weighted` from resolved entity seeds, seed-excluded

Each source is min-max normalized (`normalize_scores_in_place`), RRF-fused (`rrf_fuse`, `1/(k+rank)`), tier-weighted (`fuse_and_score_phase2`), then MMR-deduped. If coverage ≥ `phase2_coverage_gate = 0.65` and `>= 3` items, it early-exits with `phase2_only = true`, merging up to 3 Phase-1 Facts under the Merge strategy.

#### Phase 3 — Broaden (`:1047+`)

Per-variant fan-out: each `QueryVariant` runs `run_recall_for_variant` (`kb.recall_chunk_and_entity_scoped` — hybrid vector + BM25 + entity-scoped), concurrently; optional sparse Chunk channels; RRF-fused across variants + sparse. If the scored set is empty, it falls back to the Phase-2 items.

`RecallItem`s are then built with tier weights, and optionally reranked (`:1160`):

- **ColBERT** — in-process MaxSim over the `colbert_embedding` column, rescaled into the top-window score band, OR
- **xervo cross-encoder** — `top_n` candidates rescored, optional sigmoid mapping.

An **answer-type boost** (`:1197`) runs one Cypher per item over `top_n` (default multiplier 1.0 = no-op). Finally a **Phase-1 contribution strategy** (`:1230`) folds consolidated Facts back in.

#### Kinds, tiers and coverage

`RecallKind` maps to a `RecallTier` and a weight:

| Kind | Tier | Weight |
|---|---|---|
| Fact, Topic | Semantic | 1.0 |
| Procedure | Procedural | 0.9 |
| Episode | Episodic | 0.7 |
| Observation, Chunk | KnowledgeBase | 0.5 |
| Message, Other | Provenance | 0.4 |

```
coverage = 0.4·facet_coverage + 0.3·mean_score + 0.3·diversity(distinct_tiers/5)
facet_coverage = min(#semantic+procedural items, facet_count) / facet_count
```

`finalize_bundle` (`:657`) sorts descending, truncates to `limit`, and sums `TOKENS_PER_ITEM = 50` until the token budget is hit.

#### Intent profiling (`recall/intent.rs`)

`build_intent_at` (`:147`) runs `analyze_query` (the NLP cascade under `onnx`, else rule-based NER) and builds up to **4 query variants**:

- **keywords** — POS-filtered NOUN/VERB/PROPN/ADJ/NUM
- **original** — the raw query
- **declarative** — an SVO reconstruction from the DEP tree
- **type_anchored** — entity + predicted answer-type

The default (empty `enabled_variants`) is **keywords-only**. Enabling all four measured **−2.1pt evidence% and ~3× latency** on LoCoMo, so multi-variant reformulation is opt-in (`:165-174`). Variant embeddings are computed concurrently; failures fall back to an empty vector (BM25-only for that variant).

`predict_answer_type` (`:350`) is a precision-tuned ordered regex (`who → person`, `where → location`, `when → date`, `how many → measurement`, …). The temporal window is resolved via `resolve_temporal_with_granularity` against `reference_ts` (else `now`).

#### `RecallConfig`

A ~40-field tuning struct, normally built from the KB config via `RecallConfig::from_uniko_config()` so recall honors the reranker/recall knobs set at ingest rather than standalone defaults. Salient fields: `limit`, `token_budget`, `min_score`, vector/bm25 weights, reranker (`enabled`/`top_n`/`style`/`sigmoid`), sparse toggles, `query_variants`, `rrf_k`, `per_variant_limit`, `phase1_strategy` + `alpha`, phase2 gates/mmr/temporal/graph/edge_weights, cross-modal toggles, `viewer`, `dimensions`, `drift_override`, and `reference_ts`.

`ContextBundle` returns `items: Vec<RecallItem>`, `total_tokens`, `phase1_only`, `phase2_only`, `coverage`. Each `RecallItem` is `{ node_id, kind: RecallKind, score, content, sources: Vec<RecallSource> }`, where `RecallSource` is the lineage tag:

- `Message { message_id, chunk_id? }`
- `Attachment { message_id, artifact_id, chunk_id? }`
- `Document { artifact_id, chunk_id? }`

**MMR dedup** (`recall/mmr.rs`) maximizes `λ·rel(i) − (1−λ)·max_sim(i, selected)` using Jaccard token overlap, with a hard-drop for candidates over `duplicate_threshold = 0.85` (default `λ = 0.7`).

#### Phase-1 contribution strategies

| `Phase1Strategy` | Behavior |
|---|---|
| `Merge` | cap-3 interleave of Facts by score, dedup keeping the higher-scored copy (conv-26 0.750) |
| `Boost` (default) | `session_boost_signals` walks `Fact -SUPPORTED_BY-> Obs -OBSERVED_IN-> Msg -IN_SESSION-> Session -HAS_CHUNK-> Chunk` and adds `alpha·fact_score` to chunk scores — keeps the bundle 100% chunks. The final hop needs the session-level chunks built by `Session::finalize` (§4.8); an unfinalized session contributes nothing. |
| `Off` | disables Phase-1 contribution entirely |

#### Recall gotchas

- **Recall is FAIL-OPEN by default.** `ViewerScope::Unrestricted` returns policy-scoped `Fact`/`Observation` items **unfiltered** with only a `WARN` (`:704-714`). A production caller serving a specific participant MUST set `RecallConfig.viewer` or build with `.scope_to_agent()`, or data leaks across visibility. (Unknown visibility schemes fail **closed** — see §6.6.)
- **`reference_ts` is a per-query anchor, never a KB setting.** Recalling a historical corpus without setting it computes the temporal window around *now*, which never overlaps old data and silently disables the Phase-2 temporal channel (`:424-431`).
- **`answer_type_boost` defaults to 1.0 (no-op) deliberately** — the naive rule measured −0.149 R@5 / −0.186 NDCG@5 because common predicted types (e.g. `measurement` for "how many") swamp top-K (`:492-500`).
- **Leftover debug prints.** Three `eprintln!("RECALL_PROF …")` statements ship in the hot path (phase2 gate `:1528-1531`, `run_sparse_source` `:1748-1752`, `run_phase2_source` `:1831-1835`); they write to stderr on every recall.
- `populate_sources` and the viewer/redaction filters run on the **small final bundle**, not the candidate set — a `populate_sources` failure is swallowed (items keep empty `sources`) rather than failing the recall.

---

### 6.3 Goal-oriented working memory

There is **no stored working-memory node**. Working memory is recomputed on demand by traversing `Goal → Task → Session → Messages/Facts/Entities`, so it is always current and never cached.

The expander is `Goals::context(goal_id)` (`facade/goals.rs:278`):

```
GoalContext {
    goal,             // GoalView
    tasks,            // Vec<TaskView>
    sessions,
    recent_messages,
    facts,
    entities,
}
```

`context()` resolves `goal_scope_ids(goal_id, include_descendants = true)` (walking `PARENT_GOAL`), then calls the store helpers `fetch_wm_sessions`/`fetch_wm_messages`/`fetch_wm_facts`/`fetch_wm_entities`. The token budget for the assembled context lives in those `fetch_wm_*` helpers in the store layer, not in the facade.

Goal and task **phases are derived from free-form status strings** via a fixed match table (`goal_phase`/`task_phase`), with `completed_at` winning:

- `GoalPhase`: Planned / Active / Completed / Abandoned
- `TaskPhase`: Planned / Active / Completed / Blocked

An unrecognized status defaults to **Active** — arbitrary status strings won't map to Planned/Blocked unless they're in the known set.

The `Goals<'a>` surface exposes reads (`all`/`active`/`planned`/`completed`/`in_phase`, `get`, `tasks`/`tasks_in`/`tasks_of`, `context`), transitions (`start`/`abandon`/`complete`/`set_status`, `start_task`/`block_task`/`complete_task`), and creation (`create`, `create_task`). Creation binds the `create_goal` (F8) and `create_task` (F9) free functions.

```rust
use uniko_memory::CreateGoalParams;
let gid = agent.goals()
    .create(CreateGoalParams { title: "ship v1".into(), ..Default::default() }).await?;
agent.goals().start(&gid).await?;
let wm = agent.goals().context(&gid).await?;   // Goal→Task→Session→Messages/Facts/Entities
```

`Data<'a>` is the addressed-retrieval sibling: `message(id) -> MessageView`, `artifact(id) -> ArtifactView`, `artifact_bytes(id) -> Vec<u8>` (the supported binary path, since `Value::Bytes` cannot round-trip through a Cypher `RETURN`).

---

### 6.4 Consolidation (P4) — Facts, drift & contradiction

Consolidation is the heartbeat that turns accumulated Observations into bitemporal Facts. `run_cycle_with(kb, agent_id, opts, &TripleSource)` (`consolidation.rs:154`) does:

1. **Fetch unprocessed Observations** — those with no inbound `PROCESSED` edge (the idempotency anchor), capped at `DEFAULT_BATCH_SIZE = 500`.
2. **Optional LLM triple refinement** (`:183`, `refine_triples`) when `TripleSource::Llm { alias }` is chosen; default is `SrlDep` (no LLM), which falls back to `SrlDep` on any LLM failure.
3. **Group** by `(normalize_canonical(subject), predicate)` (`:240`).
4. **Batched prior-Fact lookup** via `find_stale_open_facts_batched` (`:278`).
5. **Collect all unique object surface forms**, embed them in one batched, chunked call (`EMBED_BATCH_CHUNK_SIZE = 64` — do not remove; larger single batches OOM the ORT BFC arena at ~1.3 GB / ~6k inputs).
6. **Per group:**
   - `build_clusters` (`:717`) — greedy single-pass agglomeration with running-mean centroids over sorted keys, `COSINE_THRESHOLD = 0.88`; a missing embedding becomes a singleton.
   - `canonical_object` (`:812`) — mode over clusters, then mode over surface forms, with a recency tie-break.
   - **F38 contradiction** — `vote_tallies` in cluster space vs `CONTRADICTION_THRESHOLD = 0.40`.
   - `compose_embed_text` (`:868`) — optional `[Month Year]` date prefix, doc-side only.
7. **Single batched Fact embed + `batch_upsert_facts`** (`:463`).
8. **`SUPPORTED_BY` edges** batched, each with `weight = clamp(cosine(Fact, Obs), 0, 1)` (fallback 1.0).
9. **Per stale prior Fact:** `invalidate_fact` + `record_entity_invalidation`; `DRIFT_THRESHOLD = 4` flips `Entity.unstable = true` (F39).
10. **`write_consolidation_cycle`** audit node with `PROCESSED`/`CREATED`/`REINFORCED`/`INVALIDATED`/`APPLIED_RULE` edges.

`CycleStats` returns `{ observations_processed, facts_created, facts_reinforced, facts_invalidated, drift_alerts }`. Phase timing is logged under target `p4_prof`.

**Bitemporality.** Each Fact carries one atomic `valid_at` of `DataType::Btic` — a half-open interval `[lo, hi)`, **not** split `valid_from`/`valid_until`. Active = `[observed_at, +inf)`; invalidation closes `hi` at now. Certainty is `approximate` below `CERTAINTY_THRESHOLD = 10` supporting observations and upgrades to `definite` at ≥ 10. Recall matches validity via the `btic_overlaps(valid_at, $window)` scalar Cypher fn.

**F38 contradiction detection.** Within a `(subject, predicate)` group, tally votes in cluster space; if the fraction of votes *outside* the prior Fact's cluster exceeds 0.40, close the prior open-BTIC Fact (`hi` = now) and wire an `INVALIDATES` edge (reason `consolidation contradiction`).

**F39 entity drift.** `record_entity_invalidation` counts `INVALIDATES` edges in a rolling `DRIFT_WINDOW_DAYS = 30` window; when the windowed count exceeds `DRIFT_THRESHOLD = 4` it flips `Entity.unstable = true`. The cumulative `invalidation_count` keeps rising even when the flag stays false — the flag is windowed, the counter is lifetime. Recall's F58 override then forces volatile-entity queries past Phase 1.

```rust
// Usually driven by the worker; callable directly:
let stats = uniko_memory::consolidation::run_cycle(kb, "assistant-1", None).await?; // SrlDep
let stats = run_cycle_with(kb, "assistant-1", None,
    &TripleSource::Llm { alias: "triple-extractor".into() }).await?;               // LLM triples
```

Consolidation is idempotent via `PROCESSED` edges — safe to call repeatedly; each `(Fact, Observation)` pair gets exactly one `SUPPORTED_BY` edge.

---

### 6.5 The pipeline — ingest & consolidation workers

`PipelineSystem::new(config, Arc<kb>, steps)` (`pipeline/mod.rs:69`) spawns an `IngestWorker` and a `ConsolidationWorker` on bounded mpsc channels, sharing a `CircuitBreaker` and a `DeadLetterQueue`. Submit work via `submit_ingest`/`submit_consolidation`; `quiesce()` is a barrier; `health()` and `shutdown()` complete the surface.

**Ingest worker** (`pipeline/ingest_worker.rs`). A `biased` `tokio::select!` (shutdown > recv). Each item is spawned under a `Semaphore(concurrency)`. An `InflightGuard` (RAII) decrements an `AtomicUsize` on every exit path, incremented at submit; `quiesce()` polls it to 0. `populate_ingest_metadata` serializes the typed payload into the `PipelineContext` so the generic `IngestStep` can deserialize it, then `run_step_chain` executes the chain honoring `StepErrorPolicy` (Skip / DeadLetter / Abort). Note: `run_step_chain` is the actual `Step` executor and lives *here*, not in `uniko-pipes` — that crate only defines the trait and outcome vocabulary. A transport-level `Err` from a step is defensively coerced to `DeadLetter`; Skip and DeadLetter both **continue** the chain, only Abort short-circuits; cancellation is cooperative and checked only *between* steps.

**Consolidation worker** (`pipeline/consolidation_worker.rs`). A `biased` select (shutdown > task > timer). Triggers:

- `ObservationsReady` — a per-agent counter reaching `consolidation_threshold`. Emitted by the ingest path as Observations land: `Session::observe` sends it directly, and the ingest worker sends it for streamed `submit`s, attributing them to the agent carried on the task's reserved `agent_id` metadata key. Both are best-effort — a full consolidation queue is logged and dropped rather than failing a committed ingest.
- `ForceConsolidate { agent_id }`
- `RunCycle { agent_id }`
- a periodic timer, firing cycles for any agent with `count > 0`

This worker only exists on a `.streaming(true)` instance. The always-available path is the facade verb `Agent::consolidate()`, which calls `consolidation::run_cycle` on the caller's task and returns `CycleStats`.

On a successful cycle, `maybe_run_cortex_sweep` (`:227`) runs when a per-agent cycle counter reaches `cortex_every_n`, executing in order:

1. `run_procedure_sweep` — P5 `promote_procedures_once` (per-agent throttle)
2. `run_topic_sweep` — P6 `detect_topics_once` (global throttle)
3. `run_decay_sweep` — F50 memory decay via `consume_relevance_decay`
4. `run_rule_execution_sweep` — **before** decay, so matches reset `missed_cycles`
5. `run_rule_decay_sweep` — global
6. `run_session_maintenance_sweep` — F14 auto-close + F59 summarize (global)

All sweeps are cadence-gated by `cortex_min_interval`; all failures are logged and dropped. This is the crate's one deliberate reverse-altitude edge: P5/P6 (cortex, "Layer 5") are *subscribers* to P4, so the trigger policy correctly lives in the memory worker.

---

### 6.6 Rules — stdlib + confidence lifecycle

Four stdlib Locy rules are registered idempotently on open (`rules/stdlib.rs`) with confidence fixed at 1.0, exempt from decay/demote/prune:

| Rule | Runtime | Role |
|---|---|---|
| `sequence_detector` | Locy + Rust consumer | drives P5 procedure promotion; the Rust consumer is `upsert_procedure` in `uniko-cortex` (rule source imported from cortex to avoid drift) |
| `episode_pattern_detector` | Locy + Rust consumer | recurring-pattern detection; the Rust consumer (`consume_episode_patterns`) computes `mean_importance` in Rust (uni-db #145 workaround) |
| `contradiction_detector` | Locy + Rust consumer | fact contradiction |
| `relevance_decay` | Locy + Rust consumer | F50 memory decay; the Locy rule **is** registered and is the single source of truth for the decay formula (`stdlib.rs:31`), while the Rust consumer (`consume_relevance_decay`) performs only the DELETE the rule cannot. It runs in its own `run_decay_sweep`, so `run_active_rules` skips it |

Authored/induced rules follow a confidence lifecycle (`rules/lifecycle.rs`): `add_rule` validates the Locy up-front (no node created on syntax error) → `candidate@0.5`. `apply_decay_cycle` applies `confidence *= 0.95`, `missed_cycles++`, demote `<0.40`, repromote/promote `>=0.60`, and prunes `demoted|candidate` rules `>= 90d` since last scored. `record_rule_match` adds `+0.05`, resets `missed_cycles`, and promotes `candidate/demoted → active` at `>= 0.60`.

`run_active_rules` (`rules/execution.rs`) runs active **and** candidate rules (candidates must run to bind and promote), skipping dedicated-sweep rules. **It must pass the UNION of all stdlib rules' params on every `query_rule` call** — uni-db evaluates all registered rules together, so an unresolved `$decay_rate` in `relevance_decay` would fail every other rule (`:69-71`). `consume_episode_patterns` computes `mean_importance` in Rust (working around uni-db #145, where a FOLD value-aggregate returns 0.0 when YIELD renames it — only COUNT is trusted in Locy FOLDs).

The `Agent` also exposes the interactive Locy surface: `define_rule`, `run_rule`, `assume`, `abduce`.

---

### 6.7 Query episodes — feeding P5 from production traffic

`query.rs` closes the loop: production recall traffic becomes durable `Episode` nodes that P5 can promote into Procedures.

`answer_query(kb, q, &config, generator_closure, Option<QueryRecordOptions>) -> Answer` orchestrates three steps:

1. **recall** — the full 3-phase cascade
2. **generation** — via a caller-supplied `generator_closure` (not a trait, so the caller owns the LLM entirely)
3. **opt-in recording** — `record_query_episode`

Recording failure never breaks the answer (`episode = None` on failure). The Episode's `state` JSON duplicates the question into `topic` so the embedding path picks it up. `finite_or_zero` guards a NaN coverage.

`record_query_episode` is also the standalone primitive that feeds P5 — any recall consumer can call it. `Agent::answer` wires `answer_query` with a **synchronous** prompt-build closure to keep the future's captures owned (`agent.rs:374-406`).

```rust
let answer = agent.answer("what does alice like?").await?;
println!("{} — {} citations", answer.text, answer.citations().len());
```

`Answer` carries `{ text, model: Option<String>, input_tokens: Option<u64>, output_tokens: Option<u64>, recorded_episode, context: ContextBundle }` plus `citations() -> Vec<&RecallSource>` (`query.rs:181-205`).

---

### 6.8 `nl_to_cypher` and the optional LLM-offline paths

`uniko-memory` keeps LLM strictly optional — the default offline path uses local ONNX NER + rule-based observation extraction + SrlDep triples. Three modules provide the optional LLM enrichments; none is ever on the write path.

#### `nl_to_cypher.rs` (F61) — NL → read-only Cypher

`translate(...)` turns a natural-language question into a read-only Cypher query for `Agent::query`. It uses a process-global LRU cache, a schema-snapshot prompt, and LLM retry. Every candidate passes `is_safe_read_only`:

```rust
// Conservative word-boundary regex rejecting mutating keywords, even inside string literals:
//   CREATE | MERGE | DELETE | DETACH | SET | REMOVE | DROP | LOAD
```

The guard is deliberately conservative — it rejects legitimate reads that contain those words inside string parameters (the caller must re-quote via parameters). `CALL` is intentionally **not** blocked, because read-only procedures exist.

#### `llm_triples.rs` — LLM triple refinement for P4

When consolidation runs with `TripleSource::Llm { alias }`, `llm_triples` refines `(subject, predicate, object)` triples with a concurrent batch of LLM calls, parsing a `SUBJECT|PREDICATE|OBJECT` line format and handling `SKIP`. Any failure silently falls back to the deterministic SrlDep triples.

#### `summary.rs` (F59) — session summaries

`generate_session_summary` is **extractive by default**, with an optional LLM-abstractive path behind the `llm` feature. The `Summary` node is idempotent.

---

### 6.9 Agent-tool free functions

Each subjective act that cannot be inferred from message content is a free function over `&KnowledgeBase`, bound as an `Agent`/`Goals`/`Session` method. Each takes a `*Params` struct.

| Function | Feature | Notes |
|---|---|---|
| `create_goal` (F8) | — | Goal + `OWNED_BY`/`PARENT_GOAL` + embedding |
| `create_task` (F9) | — | Task + `ASSIGNED_TO`/`PART_OF`/`DEPENDS_ON`/`SUBTASK_OF` + embedding |
| `record_episode` | — | Episode + `RECORDED_BY`/`FOLLOWED_BY`(1-hr window)/`TRIGGERED_BY`/`INVOLVES` + topic embedding |
| `record_action` (F17–F20) | — | Action node + edges; string output over a token threshold spills to an Artifact + `PRODUCED` (F20), Action keeps a stub; NER `MENTIONS` linking |
| `add_observation` (F34) | — | via the shared `apply_observations` writer, anchored `OBSERVED_IN` a Message |
| `assert_fact` (F33) / `invalidate_fact` (F37) | — | direct bitemporal Fact assert/close |
| `generate_session_summary` (F59) | `llm` (abstractive) | idempotent Summary |

**Bootstrap invariant:** `record_episode`/`record_action`/`create_goal`/`create_task`/`add_observation` all require the agent's `Participant` node to already exist (they fail with `UnikoError::Storage` otherwise). An embedding failure still creates the node (back-fillable) but returns the error.

---

### 6.10 Access control & redaction (F66)

`policy.rs` implements two distinct filters that both run over the final bundle.

`Viewer { participant_id, teams, orgs }` is resolved via `resolve_participant_memberships` (`Viewer::new(kb, pid)`) or supplied directly (`Viewer::from_parts`). `visibility_admits(Option<&str>, &Viewer)`:

| Scheme | Admits |
|---|---|
| `null` / `""` / `public` | everyone |
| `private:{pid}` | that participant |
| `team:{tid}` | members of that team |
| `org:{oid}` | members of that org |
| unknown scheme | **no one (fail-closed)** |

`filter_bundle` checks **only** `Fact` and `Observation` items (batch `fetch_visibilities`); every other node type passes through untouched — recall does not hide structural nodes. `filter_redacted` runs on *every* recall (redaction ≠ access control): it drops `redacted:`-scheme Facts/Observations and `redacted`-flag Messages/Chunks. Both recompute `total_tokens = items.len() * 50`.

An unrecognized visibility string like `secret:42` locks a claim down permanently — it can never accidentally widen access.

---

### 6.11 Invariants & gotchas summary

- **Fail-open recall.** Unscoped reads default to `ViewerScope::Unrestricted`; scope to a viewer (`.scope_to_agent()` or `Scope::default().as_viewer(v)`) before serving a specific participant, or `Fact`/`Observation` visibility is not enforced.
- **Streamed vs observed turns.** `submit`/`submit_source` do **not** advance the Session's cross-turn context and are not session-linked; only `observe()` preserves conversational fidelity and links attachments. `await flush()` before a recall that must see streamed turns, and use one path per session.
- **Multi-query is opt-in and usually a loss** (−2.1pt / 3× latency). Keep `query_variants` empty (keywords-only) unless you have evidence otherwise.
- **`reference_ts` is per-query** — never a KB setting; forgetting it silently disables the temporal channel on historical corpora.
- **Consolidation embed batching is chunked at 64** to dodge the ORT BFC arena OOM — do not remove `EMBED_BATCH_CHUNK_SIZE`.
- **`run_active_rules` must pass the union of all stdlib params** every call, or one unresolved param fails every rule.
- **Shutdown ordering:** drop all `Agent`/`Session` clones before `Uniko::shutdown`, or the pipeline never drains.
- **Leftover `RECALL_PROF` stderr prints** ship in the recall hot path (three sites).
- Facade re-exports surface `uniko_store` types directly (`FactUpsert`, `*IngestResult`, `FactUpsertInput`, …) so the public API names no un-nameable types.

---

## 7. Higher Reasoning — Procedures & Topics (L5)

Layers 1 through 4 turn raw messages into a searchable, provenance-backed knowledge graph: entities, observations, bitemporal facts. Layer 5 — the **cortex** — is where uniko stops recording *what happened* and starts distilling *what works* and *what things are about*. It has exactly two jobs, implemented by exactly two source modules in `crates/uniko-cortex/`:

| Sweep | Module | Feature | Output node | Driver |
|-------|--------|---------|-------------|--------|
| **P5 — Procedure promotion** | `procedures.rs` | F41/F42/F43 | `:Procedure` | Locy `sequence_detector` rule |
| **P6 — Topic detection** | `topics.rs` | F40 | `:Topic` | Weighted Label Propagation |

The crate is deliberately thin. It owns the *algorithms and lifecycle logic* only; every graph read and write is delegated to `uniko-store` repository helpers (`repository/procedures.rs`, `repository/topics.rs`). Cortex holds **no `uni_db` session code of its own** — all graph I/O goes through `uniko-store`. Its declared dependency set is `uniko-store` and `uni-db` (the latter declared in `Cargo.toml` but not used for session code in `src/`), plus `chrono`, `serde`, `tracing`.

### 7.1 "Layer 5" is altitude, not build order

The name is a lie about the dependency graph, and the crate root says so explicitly (`lib.rs:6-14`). In the actual build DAG, `uniko-cortex` is a *sibling* of `uniko-extract` and sits *below* `uniko-memory`. `uniko-memory` depends on `uniko-cortex`, not the other way round.

The reason is the trigger relationship. Cortex's sweeps are **subscribers to P4 consolidation**. Consolidation (the P4 "heartbeat" in `uniko-memory`) is the sole production caller: after a successful consolidation cycle, the consolidation worker invokes `promote_procedures_once` (P5) and `detect_topics_once` (P6). So memory-depends-on-cortex is the natural direction — the publisher pulls in the subscriber's entry points.

```
   uniko-memory (L4)  ── consolidation worker (the "heartbeat")
        │  depends on
        ▼
   uniko-cortex (L5)  ── promote_procedures_once / detect_topics_once
        │  depends on
        ▼
   uniko-store (L1)   ── repository::{procedures,topics}, Locy runtime, schema
```

If cortex ever needs memory's *runtime* APIs (e.g. an MCTS planner that calls `recall`), the intended fix is **not** a new dependency edge (that would be a true cycle). It is a sweep trait defined in `uniko-memory` and injected at the composition root (`lib.rs:11-14`).

### 7.2 The producer: how the FOLLOWED_BY chain gets built

P5 reads a graph shape it does not create. The producer is `uniko-memory`'s `record_episode` (`episode.rs:90-215`), an agent-tool free function. Every `record_episode` call writes an `:Episode` node carrying `action_type`, `outcome` (`'success'`, …), `importance`, and `timestamp`, and wires two edges that are load-bearing for procedure learning:

- **`RECORDED_BY` → `Participant`** (`episode.rs:146-152`) — ties the episode to the agent whose experience it is.
- **`FOLLOWED_BY` → previous Episode** — created *only* if the same agent's previous episode was within `FOLLOWED_BY_WINDOW_MS = 3_600_000` (one hour, `episode.rs:69`). The edge carries the actual `gap_ms` between the two episodes.

```
(e1:Episode {action_type:"investigate", outcome:"success"})
    │  FOLLOWED_BY {gap_ms: 42000}
    ▼
(e2:Episode {action_type:"summarize",  outcome:"success"})
    │
    ├── e1 -[:RECORDED_BY]-> (p:Participant {participant_id:"agent-1"})
    └── e2 -[:RECORDED_BY]-> (p)
```

This is exactly the `(e1)-[:FOLLOWED_BY]->(e2)` + `RECORDED_BY` shape the rule matches. **Repetition** — the same `(action_a, action_b)` success-pair recurring across many episode chains — is the signal. The Locy rule's `COUNT(*)` *is* that repetition count.

### 7.3 The `sequence_detector` Locy rule

The canonical rule string lives in exactly one place — `SEQUENCE_DETECTOR_RULE` in `procedures.rs:64-70` — and is referenced (not copied) by `uniko-memory`'s `register_stdlib_rules` (`stdlib.rs:71`) so startup registration and sweep-time registration can never drift:

```
CREATE RULE sequence_detector AS
  MATCH (e1:Episode)-[:FOLLOWED_BY]->(e2:Episode),
        (e1)-[:RECORDED_BY]->(p:Participant {participant_id: $agent_id})
  WHERE e1.outcome = 'success' AND e2.outcome = 'success'
  FOLD n = COUNT(*)
  YIELD KEY e1.action_type AS action_a, KEY e2.action_type AS action_b,
        n AS success_count
```

Read it as: *for every `FOLLOWED_BY`-adjacent pair of episodes, both recorded by the target agent and both with `outcome = 'success'`, group by the `(action_a, action_b)` action-type pair and count the occurrences into `success_count`.* The two `KEY` columns are the grouping key; `n` is the per-group count.

**This rule is authored against Locy's grammar, which is not Cypher.** Three constraints are baked in, and each was once a latent bug that kept the rule from registering (`procedures.rs:57-63`):

1. **One comma-joined `MATCH`.** The two relationships (`FOLLOWED_BY` and `RECORDED_BY`) are joined with a comma inside a *single* `MATCH`. A second `MATCH` clause is a Locy parse error.
2. **`expr AS name` aggregates, no `VALUE` keyword.** `FOLD n = COUNT(*)` … `YIELD KEY … AS …, n AS success_count`.
3. **No threshold in the rule.** There is deliberately no `HAVING`/threshold filter. The rule surfaces *all* recurring pairs; the promotion threshold is applied in Rust. This is not a stylistic choice — a `$param` in a post-`FOLD` `HAVING` **does not resolve** in Locy (the RC12 bug). Pushing the threshold to the consumer is the workaround.

### 7.4 P5 entry point: `promote_procedures_once`

```rust
pub async fn promote_procedures_once(
    kb: &KnowledgeBase,
    agent_id: &str,
    cfg: LifecycleConfig,
) -> Result<PromotionReport>
```

The flow (`procedures.rs:125-166`):

1. **Register the rule** — `kb.create_rule(SEQUENCE_DETECTOR_RULE)`. Idempotent: registering a duplicate program is a no-op (`procedures.rs:130-134`). Note the rule is therefore registered in *two* places (startup via `register_stdlib_rules`, and here on every sweep), both idempotent.
2. **Run it by name** — `kb.query_rule("sequence_detector", [action_a, action_b, success_count], {agent_id})`. This is critical: `query_rule` (`locy/rules.rs:70-91`) builds a *goal query* `QUERY sequence_detector RETURN action_a, action_b, success_count`. That is the supported way to invoke a **registered** rule by name. Handing the bare name to `execute_rule` would try to parse `"sequence_detector"` as a program and fail (`locy/rules.rs:54-62`).
3. **Upsert per row** — each result row is destructured into `action_a: String`, `action_b: String`, `success_count: i64`; missing/mistyped fields are skipped via `let-else` (`procedures.rs:147-153`). Each surviving row calls `upsert_procedure`.

The returned `PromotionReport { created, reinforced, promoted }` tallies new procedures, reinforced ones, and the subset of reinforcements that flipped a `candidate` to `active` on this call.

### 7.5 Where lifecycle classification happens: `upsert_procedure`

`upsert_procedure` (`procedures.rs:276-347`) is the heart of F41. It is the only place the sequence-count path can promote `candidate → active`.

**Deterministic id.** `stable_procedure_id(agent_id, a, b)` = `stable_hex64("proc", agent \0 a \0 b)` (`procedures.rs:356-360`). The hasher is fed in order **agent, then a, then b**, NUL-separated. This is order-sensitive on purpose: `(a → b)` and `(b → a)` hash to distinct ids, and the NUL prevents aliasing across component boundaries. Do not "simplify" the ordering — it would break every persisted id.

**Status decision** (`procedures.rs:291-305`), given `support_count` (the rule's `success_count`) and the existing snapshot from `read_procedure_snapshot`:

| Situation | Resulting status |
|-----------|------------------|
| New procedure, `support_count ≥ promote_threshold` | `active` |
| New procedure, `support_count < promote_threshold` | `candidate` |
| Existing, currently `candidate`, `support_count ≥ threshold` | `active` (**promoted_this_call**) |
| Existing, status empty | `candidate` |
| Existing, otherwise | keep existing status |

Demotion and re-promotion are **owned exclusively** by `record_procedure_use` — `upsert_procedure` never demotes. `promoted_this_call` is true precisely when an existing `candidate` just became `active` (`procedures.rs:306-307`), which is what feeds `PromotionReport.promoted`.

**Counters.** Both `success_count` and `use_count` use `max()` as a floor: `success_count = max(snapshot.success_count, support_count)` (`procedures.rs:310`), same for `use_count` (`:333`). The rule's occurrence count can *raise* these but never *lower* them, so failure history accumulated by `record_procedure_use` is preserved across sweeps.

**Effectiveness.** `effectiveness = success / (success + failure)`, defaulting to **1.0** when the denominator is 0 (a fresh procedure, `procedures.rs:311-316`). Note this default is asymmetric with `record_procedure_use`, which defaults effectiveness to **0.0** when it has no uses (`procedures.rs:205`). A never-used but promoted `active` procedure therefore reads effectiveness `1.0`.

**Write.** `kb.merge_node(PROCEDURE, …)` (idempotent upsert) persists `name = "{a} → {b}"`, a human description, `precondition_rule = "last_action_type={action_a}"`, status, counters, effectiveness, `created_at` (only when new), and `last_used_at`.

### 7.6 F42 — `record_procedure_use` and the demote/repromote state machine

When an active procedure is actually executed, the caller records the outcome:

```rust
record_procedure_use(kb, &procedure_id, /*succeeded=*/ true, cfg).await?;
```

The flow (`procedures.rs:175-226`): look up the node (error if absent), read a `ProcedureSnapshot`, increment `use_count` and one of `success_count`/`failure_count` per the bool, recompute effectiveness, then run the state machine (`procedures.rs:208-215`):

```
        effectiveness < demote_effectiveness (0.4)
   ACTIVE ─────────────────────────────────────────► DEPRECATED
      ▲                                                    │
      └──────────────────────────────────────────────────┘
        effectiveness ≥ repromote_effectiveness (0.6)

   CANDIDATE ──(stays CANDIDATE — usage never promotes it)──►
```

The gap between demote (`< 0.4`) and repromote (`≥ 0.6`) is intentional **hysteresis**: a single failure that dips effectiveness to, say, 0.5 will not flap an active procedure into deprecated and back. Two rules to internalize:

- **Usage never promotes a candidate.** `CANDIDATE` only advances to `ACTIVE` via the sequence-count path in `upsert_procedure`. Using a candidate just bumps its counters.
- **Demotion/repromotion apply only to `ACTIVE`/`DEPRECATED`.**

### 7.7 F43 — `match_procedures`: retrieving "what works" for the current state

```rust
let mut state = HashMap::new();
state.insert("last_action_type".into(), "investigate".into());
let matches = match_procedures(&kb, &state).await?; // Vec<MatchedProcedure>
```

`match_procedures` (`procedures.rs:242-259`) fetches all `ACTIVE` procedures (`fetch_procedures_by_status` → `MATCH (p:Procedure) WHERE p.status=$st … ORDER BY eff DESC`, `repository/procedures.rs:33-61`) and keeps those whose precondition is satisfied by `state`.

The MVP matcher, `precondition_matches` (`procedures.rs:368-386`):

- Split `precondition_rule` on `,`; each clause must be `key=value`.
- A clause with no `=` **fails closed** (the whole match fails).
- Every key must be present in `state` with the exact value.
- An **empty** precondition matches **anything** (fails open).

Because `upsert_procedure` writes `precondition_rule = "last_action_type={action_a}"`, a caller passing `state = {"last_action_type": X}` retrieves every procedure whose antecedent action just fired — i.e. "given I just did X, here are the follow-ups that have historically succeeded." The returned `MatchedProcedure { procedure_id, name, effectiveness }` is ranked by effectiveness (from the store query's `ORDER BY`).

### 7.8 End-to-end: from repeated experience to a reusable procedure

```
 record_episode (memory)         promote_procedures_once (cortex,
 ──────────────────────           called by P4 consolidation worker)
  e1 success                       ──────────────────────────────────
   │ FOLLOWED_BY                    kb.create_rule(SEQUENCE_DETECTOR_RULE)  (idempotent)
   ▼                               kb.query_rule("sequence_detector", …, {agent_id})
  e2 success   ── repeats ──►         → rows: (inv:→sum:, count=1)
  e1' success                          → rows: (inv:→sum:, count=2)
   │ FOLLOWED_BY                        → rows: (inv:→sum:, count=3)
   ▼                               upsert_procedure(agent, "investigate","summarize", 3)
  e2' success                        count 3 ≥ promote_threshold(3) → status=ACTIVE
  …                                  merge_node(:Procedure {name:"investigate → summarize", …})
```

A pair first seen once becomes a **CANDIDATE**; it only crosses to **ACTIVE** once its `FOLLOWED_BY` co-occurrence count reaches `promote_threshold` (default 3). Since the P4 worker runs `promote_procedures_once` periodically (throttled by `cortex_min_interval`, per-agent), promotion is incremental: the same procedure is re-evaluated each sweep, and the `max()`-floored counters make that monotonic.

**`LifecycleConfig`** (`procedures.rs`, defaults `3 / 0.4 / 0.6`):

| Field | Default | Meaning |
|-------|---------|---------|
| `promote_threshold: i64` | 3 | min `FOLLOWED_BY` count for `candidate → active` |
| `demote_effectiveness: f64` | 0.4 | `active → deprecated` below this |
| `repromote_effectiveness: f64` | 0.6 | `deprecated → active` at/above this |

**`:Procedure` schema** (`uniko-store/src/schema/procedures.rs:13-39`): `procedure_id`, `name`, `description`, `precondition_rule`, `effectiveness`, `use_count`, `success_count`, `failure_count`, `status` (`candidate|active|deprecated`), `created_at`, `last_used_at`, `embedding`. Edges: `OPERATES_ON` (Procedure → Entity), `USED_IN` (Procedure → Task).

### 7.9 P6 — Topic detection via weighted Label Propagation

Topics are the *thematic* counterpart to procedures: communities of entities that co-occur. P6 is a single global sweep (topics span all entities, unlike per-agent procedures).

```rust
pub async fn detect_topics_once(kb, cfg: TopicConfig) -> Result<TopicReport>
// == detect_topics_once_with_llm(kb, cfg, None)
pub async fn detect_topics_once_with_llm(kb, cfg, llm_alias: Option<&str>) -> Result<TopicReport>
```

Flow (`topics.rs:101-148`):

1. **`fetch_all_entities`** (`repository/topics.rs:39-60`) — load every `:Entity` (`node_id`, `entity_id`, `name`, `entity_type`, optional `embedding`); empty → empty report.
2. Build an `entity_id → index` map.
3. **`fetch_cooccurrence_pairs(max_pairs)`** (`repository/topics.rs:70-96`) — Cypher joining two `MENTIONS` edges from a common source to distinct entities `id(a) < id(b)`, counting shared sources as weight `w`, `ORDER BY w DESC LIMIT max_pairs`. "Sources" are any node with a `MENTIONS` edge: `Message`, `Chunk`, `Action`, `Artifact`, `Episode`. So the co-occurrence weight of two entities = the number of distinct nodes that mention both.
4. Build a symmetric weighted adjacency list, skipping pairs whose endpoints aren't in the entity set and self-loops (`topics.rs:118-128`).
5. **`run_lpa`** → per-node labels.
6. **`group_by_label`** → communities.
7. For each community with size `≥ min_community_size`, compute `stable_topic_id` and `upsert_topic`; tally `created` / `updated` / `entities_assigned`.

**`TopicConfig`** defaults: `min_community_size = 2`, `max_iterations = 10`, `max_pairs = 100_000`.

#### 7.9.1 `run_lpa` — the propagation kernel

`run_lpa` (`topics.rs:155-190`) is weighted Label Propagation:

- Each node starts as its own label (`0..n`).
- Each sweep, for every non-isolated node, tally neighbour label weights and adopt the highest-weight label. Ties break by **smaller label id** (`(*w-best).abs() < 1e-12 && *lbl < best_label`, `topics.rs:174-179`) — this is what makes convergence deterministic.
- Isolated nodes (empty adjacency) keep their own label.
- Stop early when a full sweep makes no change; otherwise run `max_iterations`.

Note the labels vector is **mutated in place during a sweep**, so a node can see its neighbours' updated labels within the same sweep — this is the *asynchronous/sequential* LPA variant, not synchronous. Determinism comes from the smaller-label-id tie-break, not from synchronous updates. It converges in ~5–10 sweeps for up to ~10K entities.

#### 7.9.2 `upsert_topic` and membership-derived ids

`upsert_topic` (`topics.rs:216-253`): check existence by `topic_id`; compute `name` (`resolve_topic_name`), `summary` (`community_summary` buckets members by `entity_type` via a `BTreeMap`, falling back to a name-join when there are no types), `entity_count`, and a mean-pooled `embedding` (`pool_embedding` averages members with equal-dimension embeddings, returns `None` if none qualify, `topics.rs:390-410`). Then `merge_node` upserts the `:Topic` (adds `created_at` only when new), and `merge_belongs_to_edges` wires each member `Entity → Topic` with an idempotent, retried edge-`MERGE` (`repository/topics.rs:105-140`, uses `transact_with_retry`).

`stable_topic_id` (`topics.rs:205-214`) hashes the **sorted member `entity_id`s**, each terminated by `|`. So the id is order-invariant but **membership-sensitive**: if a community gains or loses even one entity, its `topic_id` changes and a *new* `:Topic` node is created rather than the old one being updated. Old topics are not garbage-collected here (see §7.11).

**`:Topic` schema** (`schema/topics.rs:13-28`): `topic_id`, `name`, `summary`, `entity_count`, `fact_count`, `embedding`. Edge: `BELONGS_TO` (Entity/Fact → Topic). Cortex only ever writes `entity_count` and only wires `Entity` members, though the schema permits `Fact → Topic` and declares `fact_count`.

#### 7.9.3 Optional LLM naming

Behind the `llm` feature (default off, `topics.rs:288-339`): build a prompt from the type breakdown plus up to 8 sorted member names, make **one** `generate()` call, strip wrapping quotes/markdown. Any failure — or an empty result — returns `None` and the code falls back to the deterministic `community_name` (up to 3 sorted names joined). Names are explicitly **cosmetic and must never block a sweep** (`topics.rs:90-95`). No retry.

### 7.10 The trigger: how the P4 worker drives both sweeps

Both entry points are called from `uniko-memory`'s consolidation worker (`pipeline/consolidation_worker.rs`), specifically `maybe_run_cortex_sweep`. After a successful consolidation cycle, a per-agent cycle counter reaches `cortex_every_n` and the worker runs, in order:

- `run_procedure_sweep` → `promote_procedures_once` (**per-agent**, per-agent throttle),
- `run_topic_sweep` → `detect_topics_once` (**global**, global throttle),
- then decay/rule-execution/session-maintenance sweeps.

All cortex sweeps are cadence-gated by `cortex_min_interval`; **all failures are logged and dropped** — a cortex error never breaks consolidation. The `agent_id` label on the P6 topic metric is only the *trigger* label; the sweep itself is global (`consolidation_worker.rs:466-503`). The metric surface (registered in `uniko-pipes::metrics`) exposes `uniko.cortex.{procedure_cycles_total, procedure_cycle_ms, procedures_promoted_total, topic_cycles_total, topic_cycle_ms, topics_created_total}`.

### 7.11 How learned knowledge re-enters recall

Procedures and Topics are first-class recall citizens. In the recall cascade (`uniko-memory/src/recall/mod.rs`):

- **Phase 1 (Compact)** vector-searches the consolidated tier: `Fact.embedding` (top-20), `Procedure.embedding` (top-10), `Topic.embedding` (top-5). Until P5/P6 have run, Procedure and Topic return **0 rows** — so a fresh KB simply never surfaces them, degrading gracefully.
- **Tier weighting** ranks a matched `Procedure` at `RecallTier::Procedural = 0.9`, just below `Semantic` (Fact/Topic = 1.0) and above `Episodic` (0.7). `Topic` maps to the `Semantic` tier (1.0). This is the payoff of "compile once, query forever": once a procedure or topic exists, recall retrieves it directly rather than re-deriving it.

`match_procedures` is the complementary *structured* retrieval path — given the agent's current `last_action_type`, fetch the ranked follow-ups that have worked before, without going through the vector cascade at all.

### 7.12 Gotchas and invariants

- **Locy ≠ Cypher (RC12).** The `sequence_detector` rule must use a single comma-joined `MATCH` (a second `MATCH` is a parse error), `expr AS name` aggregates (no `VALUE` keyword), and **no `$param` in a post-`FOLD` `HAVING`** (it won't resolve). The promotion threshold is therefore applied in Rust (`upsert_procedure`), not in the rule. All three were latent bugs that once kept the rule from registering (`procedures.rs:57-63`).
- **Single source of truth for the rule string.** `SEQUENCE_DETECTOR_RULE` lives only in cortex but is referenced by `register_stdlib_rules` (`stdlib.rs:71`), so startup and sweep-time registration can't drift. It *is* registered in two places (startup + every `promote_procedures_once`), both idempotent.
- **`record_procedure_use` never promotes candidates.** `CANDIDATE → ACTIVE` happens only via the sequence-count path in `upsert_procedure`. Usage of a candidate just bumps counters. Demotion/repromotion apply only to `ACTIVE`/`DEPRECATED`.
- **`max()` floors preserve failure history.** `support_count` and `use_count` are floored via `max()` in `upsert_procedure` (`:310,:333`): a sweep can raise `success_count`/`use_count` but never lower them, so a procedure's accumulated failures survive re-promotion sweeps.
- **Asymmetric effectiveness default.** `1.0` for a brand-new procedure in `upsert_procedure`; `0.0` in `record_procedure_use` when the denominator is 0. A never-used `active` procedure reads `1.0`.
- **`precondition_matches` fails closed on malformed clauses** (a clause with no `=`) but **open on an empty rule** (matches everything). Deliberate, but easy to trip.
- **`stable_procedure_id` ordering is (agent, a, b) with NUL separators** — order-sensitive, so `a→b ≠ b→a`. Don't "simplify" it.
- **`run_lpa` mutates labels in place** (sequential LPA), so results can depend on node index order for pathological graphs; determinism comes from the smaller-label-id tie-break.
- **Topic ids are membership-derived.** Gaining/losing one entity changes `topic_id` and creates a *new* `:Topic` node; the old one is not garbage-collected. `BELONGS_TO` edges are only added (idempotent merge), never removed — usually moot because the id changes, but a Topic can accumulate members if ids are ever reused.
- **`fetch_cooccurrence_pairs` caps at `max_pairs` (100K), ordered by weight desc** — on very dense graphs, low-weight pairs are silently dropped, which can fragment communities. A comment suggests switching to `CALL uni.algo.louvain` past ~50K entities (`topics.rs:17-21`).
- **LLM topic naming is best-effort single-shot with swallowed failures** — cosmetic, never blocks a sweep, only compiled with the `llm` feature.
- **Schema declares `fact_count` and permits `Fact → Topic`**, but cortex writes only `entity_count` and wires only `Entity` members.

---

## 8. Public API & Usage (Rust + Python)

This chapter is the operator's manual for the two surfaces that external consumers actually touch: the Rust facade (`uniko-api` re-exporting the `Uniko` handle from `uniko-memory`) and the async-first Python SDK (`uniko-py`, plus its `uniko-cuda`/`uniko-metal` GPU siblings). Everything below either compiles or runs as written; where a call needs a feature flag, a viewer scope, or a warm-up step to behave correctly, it is called out inline.

### 8.1 The layering, and what "public" means

uniko is a stack of layered crates. The outermost, `uniko-api`, is **logic-free by contract** — it exists only to expose *intent types* and hide *engine internals*:

```
uniko-py / uniko-cuda / uniko-metal   (PyO3 wheels — import uniko)
        │
   uniko-api          (facade re-export: Uniko/Agent/Session + value types)
        │
   uniko-memory       (the real facade impl: recall, consolidation, agent tools)
        │
   uniko-extract  ─┐
   uniko-pipes    ─┤  (Layer 2/3: pipeline machinery, NLP, ingest, embeddings)
   uniko-cortex   ─┘
        │
   uniko-store        (KnowledgeBase — the ONLY crate that touches uni-db)
        │
   uni-db  +  uni-xervo   (embedded graph+vector+FTS+Locy engine; model runtime)
```

`uniko-api/src/lib.rs` is a single wildcard re-export (`pub use uniko_cortex::*`) plus `pub mod tools;`. `tools.rs` is a hand-curated, explicit re-export of ~55 intent types from `uniko_memory`. The negative surface — what must *never* leak — is enforced at compile time: nine `compile_fail` doctests in `tools.rs` prove that `KnowledgeBase`, `PipelineSystem`, `IngestMessage`, `Document`, `PdfSource`, the free functions `create_goal`/`working_memory`, and internal result types like `QueryOutcome`/`GeneratedAnswer` are unreachable through `uniko_api::tools`. A positive guard, `tests/surface.rs::facade_entry_points_are_exported`, asserts ~45 facade types stay reachable.

The practical rule for a Rust consumer: **depend on `uniko-api` and import from `uniko_api::tools`.** The quickstart examples in the repo import from `uniko_memory` directly (that also works and exposes the same types), but `uniko-api::tools` is the sanctioned lean surface.

The `tools` module re-exports (from `crates/uniko-api/src/tools.rs`):

| Category | Types / functions |
|---|---|
| Handles | `Uniko`, `UnikoBuilder`, `Agent`, `Session`, `Turn`, `LlmSpec` |
| Recall | `ContextBundle`, `RecallItem`, `RecallKind`, `RecallTier`, `RecallSource`, `Scope`, `Dimensions`, `RecallScope`, `ViewerScope` |
| Answer / query | `Answer`, `nl_to_cypher::is_safe_read_only` |
| Ingest | `IngestSource`, `IngestData`, `IngestContext`, `IngestOutcome`, `ingest_source`, `resolve_mime`, `Mime`, `Modality`, `ContentType`, `ModalityRegistry`, `ModalityExtractor` |
| Goals | `GoalContext`, `GoalView`, `TaskView`, `GoalPhase`, `TaskPhase`, `CreateGoalParams`, `CreateTaskParams` |
| Results | `AtomicIngestResult`, `ObserveResult`, `ArtifactIngestResult`, `PdfIngestResult`, `DeletionReport` |
| Views | `MessageView`, `ArtifactView` |
| Reasoning | `AssumeBuilder`, `AbductionResult`, `AbducedModification`, `DerivationTree`, `DerivationNode` |
| Policy | `policy::Viewer`, `policy::visibility_admits` |
| Config / primitives | `UnikoConfig`, `EmbeddingConfig`, `UnikoError`, `NodeId`, `Value`, `Record` |

### 8.2 The `Uniko` handle and `UnikoBuilder`

`Uniko` is the single owning handle. It is clone-cheap (an `Arc`-backed store inside) and hands out agent-scoped views. Construction is via one of three entry points, all `async`:

```rust
use uniko_api::tools::{Uniko, EmbeddingConfig, LlmSpec};

// 1. Persistent, opinionated best config
let memory = Uniko::open("./data/kb").await?;

// 2. Ephemeral (tests, notebooks)
let memory = Uniko::in_memory().await?;

// 3. Builder — the configurable path
let memory = Uniko::builder()
    .path("./data/kb")                                   // or .in_memory()
    .embedding(EmbeddingConfig::bge_small_en_v15())      // 384-d default
    .llm(LlmSpec::openai("llm/default", "gpt-4o-mini", None))
    .streaming(true)                                     // enable submit/flush path
    .scope_to_agent()                                    // recall filters by viewer
    .build().await?;
```

`UnikoBuilder` chain methods: `.path(p)` / `.in_memory()`, `.embedding(EmbeddingConfig)`, `.llm(LlmSpec)`, `.raw_config(UnikoConfig)`, `.streaming(bool)`, `.scope_to_agent()` / `.scope(RecallScope)`, `.extractor(Arc<dyn ModalityExtractor>)`, and the terminal `.build().await`. Opening registers the four stdlib Locy rules idempotently.

`memory.agent(id) -> Agent` mints an agent-scoped handle. `memory.config()` returns the effective `UnikoConfig`. `memory.purge()` clears the store. `memory.shutdown().await` drains the pipeline and closes the DB.

> **Shutdown ordering (load-bearing).** `Uniko::shutdown` *skips* the pipeline drain (with a `WARN`) if any `Agent`/`Session` clone still shares the internal `Arc<PipelineSystem>`. Drop all agent and session handles **before** calling `shutdown`, or the pipeline isn't drained and in-flight ingest work can be lost.

### 8.3 `Agent`, `Session`, `Turn` — the write and read verbs

`Agent` binds `(kb, agent_id)`. It is the read/reason surface; `Session` under it is the write surface.

```rust
let agent = memory.agent("assistant-1");
let mut session = agent.session("chat-1");
```

**Writing — durable `observe` vs streaming `submit`.** `Turn` is a reusable builder:

```rust
use uniko_api::tools::Turn;

session.observe(
    Turn::new("alice", "I adopted a rescue greyhound named Biscuit")
        .addressed_to(vec!["bob".into()])
        .content_type("text/plain")
        .metadata("source", serde_json::json!("crm"))
).await?;   // returns ObserveResult, synchronous, read-after-write
```

`observe` runs the **atomic per-message ingest** (`ingest_message_atomic`): all NLP/NER/observation extraction happens before a single transaction that writes the `Message` + edges + chunks + entities + `MENTIONS` + observations, committing once. It advances the session's cross-turn context (pronoun resolution, the `NEXT` chain) and links attachments. The Rust `ObserveResult` (`facade/session.rs:276-282`) is **not** flat — it is `{ message: AtomicIngestResult, attachments: Vec<IngestOutcome> }`, so you reach `result.message.message_node_id`, `result.message.chunk_node_ids`, `result.message.extracted_entities: Vec<(i64, String)>`, `result.message.extracted_observations`, etc. The *flattened* view (`message_node_id`, `chunk_node_ids`, `session_node_id`, `sender`, `extracted_entities`, `extracted_observations`, `attachment_count`) is the Python wrapper `PyObserveResult` (`bindings/uniko-py/src/outputs.rs:307-333`).

The streaming path — `session.submit(Turn)` / `session.submit_source(IngestSource)` then `session.flush()` — requires `.streaming(true)` on the builder. It is fire-and-forget onto the pipeline.

> **Do not mix the two paths on one session.** Streamed turns (`submit`) do **not** advance the session's cross-turn context and are **not** session-linked; only `observe` preserves conversational fidelity. If you must `recall` after a `submit`, `await flush()` first. Use one path per session.

**Reading — three altitudes:**

```rust
// 1. recall  -> ContextBundle (ranked evidence, NO LLM)
let bundle = agent.recall("what pet does alice have?").await?;

// 2. answer  -> Answer (recall + LLM; requires .llm(); records a query Episode)
let answer = agent.answer("what pet does alice have?").await?;
println!("{}", answer.text);
for src in answer.citations() { println!("  {src:?}"); }

// 3. query   -> Vec<Record> (read-only Cypher; NL->Cypher via translate happens under query())
let rows = agent.query("MATCH (m:Message) RETURN count(m) AS n").await?;
```

Each read verb has a scoped twin — `recall_in(query, Scope)`, `answer_in`, `query_in` — that confines the read to sessions/participants/time-window and optionally a viewer:

```rust
use uniko_api::tools::Scope;

let ctx = agent.recall_in(
    "what did alice say",
    Scope::default().sessions(["chat-1"]).since(some_ts).until(now)
).await?;
```

`Agent` also carries the Locy surface (`define_rule`, `run_rule`, `assume`, `abduce`), goal/data sub-handles (`agent.goals()`, `agent.data()`), and lifecycle verbs (`delete_session`, `forget_participant`).

> **Recall is FAIL-OPEN by default.** `ViewerScope::Unrestricted` returns policy-scoped `Fact`/`Observation` items **unfiltered** (only a `WARN` is logged). A production caller serving a specific participant **must** either build with `.scope_to_agent()` or pass `Scope::default().as_viewer(v)` on the `_in` variant, or Fact/Observation data leaks across visibility boundaries. Unknown visibility schemes, by contrast, fail *closed*.

### 8.4 The `ContextBundle` / `Answer` shapes

`recall` returns a `ContextBundle`:

```
ContextBundle {
    items: Vec<RecallItem>,   // ranked, truncated to limit, token-budget-capped
    total_tokens: usize,
    phase1_only: bool,        // cascade exited at Phase 1
    phase2_only: bool,
    coverage: f64,
}

RecallItem {
    node_id: NodeId,
    kind: RecallKind,         // Chunk|Observation|Fact|Procedure|Topic|Episode|Message|Other
    score: f64,
    content: String,
    sources: Vec<RecallSource>,  // provenance lineage stamped from graph edges
}

RecallSource =
    Message   { message_id, chunk_id? }
  | Attachment{ message_id, artifact_id, chunk_id? }
  | Document  { artifact_id, chunk_id? }
```

`RecallKind` maps to a `RecallTier` weight during fusion: `Semantic`(Fact/Topic)=1.0, `Procedural`=0.9, `Episodic`(Episode)=0.7, `KnowledgeBase`(Observation/Chunk)=0.5, `Provenance`(Message/Other)=0.4. (There is no `Artifact` `RecallKind` — the enum is Chunk/Observation/Fact/Procedure/Topic/Episode/Message/Other, `recall/mod.rs:88-97`.)

`answer` returns:

```
Answer {
    text: String,
    model: String,
    input_tokens, output_tokens,
    recorded_episode: Option<...>,   // the query Episode fed to P5
    context: ContextBundle,          // the evidence the answer was grounded in
    // citations() -> Vec<&RecallSource>
}
```

### 8.5 Goals — goal-oriented working memory

`agent.goals()` returns a `Goals<'a>` view (it borrows the store, so it is rebuilt per call in the bindings). Working memory is *computed on demand* — there is no stored `WorkingMemory` node; `context(goal_id)` traverses Goal → Task → Session → Messages/Facts/Entities.

```rust
use uniko_api::tools::CreateGoalParams;

let gid = agent.goals().create(CreateGoalParams {
    title: "ship v1".into(),
    ..Default::default()
}).await?;

agent.goals().start(&gid).await?;
let wm: uniko_api::tools::GoalContext = agent.goals().context(&gid).await?;
// wm.goal, wm.tasks, wm.sessions, wm.recent_messages, wm.facts, wm.entities
```

Reads: `all` / `active` / `planned` / `completed` / `in_phase` / `get` / `tasks` / `tasks_in` / `tasks_of` / `context`. Transitions: `start` / `abandon` / `complete` / `set_status` / `start_task` / `block_task` / `complete_task`. Creation: `create` / `create_task`.

> Goal/Task phases (`GoalPhase::{Planned,Active,Completed,Abandoned}`, `TaskPhase::{...,Blocked}`) are derived from **free-form** status strings via a fixed match table; an unrecognized status defaults to `Active`. The recording tools (`create_goal`, `record_episode`, etc.) also **require the agent's `Participant` to already exist** — bootstrap it once (e.g. via a first `observe`).

### 8.6 End-to-end Rust example

```rust
use uniko_api::tools::{Uniko, Turn, LlmSpec, Scope};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Open with an answer model.
    let memory = Uniko::builder()
        .path("./data/kb")
        .llm(LlmSpec::openai("llm/default", "gpt-4o-mini", None))
        .scope_to_agent()                     // never fail-open in prod
        .build().await?;

    let agent = memory.agent("assistant");
    let mut session = agent.session("session-1");

    // 2. Ingest a turn (atomic: message + entities + observations, one commit).
    session.observe(
        Turn::new("alice", "I adopted a rescue greyhound named Biscuit.")
    ).await?;

    // 3. Recall (3-phase cascade, no LLM).
    let bundle = agent.recall("pets").await?;
    println!("top item: {}", bundle.items[0].content);

    // 4. Answer (recall + LLM + records a query Episode for P5).
    let answer = agent.answer("What pet does Alice have?").await?;
    println!("{}", answer.text);
    for src in answer.citations() { println!("  {src:?}"); }

    // 5. Goal working memory.
    // (create/start a goal, then agent.goals().context(&gid).await? ...)

    // 6. Scoped read.
    let _scoped = agent.recall_in("what did alice say",
        Scope::default().sessions(["session-1"])).await?;

    // 7. Drop session before shutdown so the pipeline drains.
    drop(session);
    memory.shutdown().await?;
    Ok(())
}
```

Add to `Cargo.toml`:

```toml
[dependencies]
uniko-api = "0.2.0"
tokio = { version = "1", features = ["full"] }
```

Feature flags worth knowing (each forwards down the stack from `uniko-api`): `onnx` (local NLP cascade + ONNX NER in `uniko-extract`), `gpu-cuda` / `gpu-metal`, `mistralrs` / `candle` (local-LLM providers). Note: `uniko-api` declares **no** `llm` feature — LLM abstractive summaries live behind the `llm` feature on `uniko-memory`/`uniko-cortex` and are not re-exposed by the facade (`uniko-api/llm` does not compile). Note: the ONNX embedding/NLP *runtime* is statically linked regardless of the `onnx` facade feature; `onnx` only adds the extraction adapters.

### 8.7 The Python SDK (`uniko-py`)

The Python SDK compiles to a `cdylib` loaded as `uniko._uniko`, wrapping the same `uniko_api::Uniko` facade. Because the entire facade is `async fn` on tokio, the module owns **one multi-threaded tokio runtime** (8 MB worker stack, `recursion_limit=512`) and **every method returns a Python awaitable**, with a blocking `*_sync` twin for every async verb.

**Install** (the three wheels are mutually exclusive — all import as `uniko`, only one installable at a time):

```sh
pip install uniko          # CPU, self-contained (~113 MiB bundled ONNX Runtime)
pip install uniko-cuda     # NVIDIA CUDA (Linux x86_64), on-GPU local LLM
pip install uniko-metal    # Apple Silicon (macOS arm64)
```

Requires Python ≥ 3.10 (abi3), `pydantic ≥ 2` for the typed overlay. Ships `py.typed` stubs.

**Async quickstart (native surface):**

```python
import uniko, asyncio

async def main():
    uni = await uniko.Uniko.in_memory()
    agent = uni.agent("assistant")
    s = agent.session("s1")

    await s.observe(uniko.Turn("alice", "I love hiking on weekends"))

    bundle = await agent.recall("hobbies")        # native ContextBundle pyclass
    print(bundle.items[0].content, bundle.coverage)
    print(len(bundle))                             # ContextBundle.__len__

    await uni.shutdown()

asyncio.run(main())
```

**Blocking twin** — every awaitable verb has a `*_sync` form driving the same shared runtime:

```python
uni = uniko.Uniko.in_memory_sync()
uni.agent("a").session("s1").observe_sync(uniko.Turn("alice", "hi"))
```

**Builder with an LLM and answer:**

```python
spec = uniko.LlmSpec.openai("gpt", "gpt-4o-mini")           # alias, model_id, base_url?
uni  = await uniko.Uniko.builder().path("/data/kb").llm(spec).streaming(True).build()
ans  = await uni.agent("a").answer("where does alice like to hike?")
print(ans.text, ans.citations())
```

`LlmSpec` static constructors: `openai(alias, model_id, base_url=None)`, `openai_with_key_env(alias, model_id, key_env, base_url=None)`, `mistralrs(alias, model_id)`.

> `LlmSpec.mistralrs()` *constructs* a spec on all three wheels, but the mistralrs provider only *compiles* into the `uniko-cuda`/`uniko-metal` variants. Building with a mistralrs spec on the base CPU wheel fails **at runtime**, not at construction.

### 8.8 Python module map

**Native PyO3 classes** (`uniko.<Name>`):

| Class | Key methods (each has a `*_sync` twin unless noted) |
|---|---|
| `Uniko` | `open`/`in_memory`/`builder` (staticmethods); `agent(id)`, `config()->dict`, `purge()`, `shutdown()` |
| `UnikoBuilder` | `path`/`in_memory`/`llm`/`streaming`/`scope_to_agent`; terminal `build()` |
| `LlmSpec` | `openai` / `openai_with_key_env` / `mistralrs` |
| `Agent` | `recall`/`answer`/`query` (+ `_in` scoped), `define_rule`/`run_rule`/`abduce`/`assume`, `session(id)`, `data`/`goals` getters, `delete_session`/`forget_participant` |
| `Session` | `observe(turn)`, `ingest(source)`, `submit`/`submit_source`/`flush`, `summarize`, `forget_turn`/`delete_turn`/`delete_document` |
| `Turn` | ctor `Turn(sender_id, content)` + `id`/`content_type`/`at(datetime)`/`addressed_to`/`metadata(k,v)`/`attach`/`attachments` |
| `IngestSource` | `from_text`/`from_bytes`/`from_path` + `with_mime`/`with_id`/`with_path` |
| `Scope` | ctor + `sessions`/`participants`/`since`/`until(datetime)` |
| `AssumeBuilder` | `then_query(q)`/`param(k,v)`/`run()->list[dict]` (hypothetical, never mutates) |
| `Data` | `message(id)->MessageView?`, `artifact(id)->ArtifactView?`, `artifact_bytes(id)->bytes?` |
| `Goals` | reads/transitions/creation mirroring the Rust `Goals` surface |

**Frozen output wrappers** (immutable snapshots, `__repr__` + `from_rust`): `RecallSource`, `RecallItem`, `ContextBundle` (`__len__`), `Answer` (`citations()`), `ObserveResult`, `DeletionReport`, `MessageView`, `ArtifactView`, `GoalView`, `TaskView`, `GoalContext`, `IngestOutcome`, `AbductionResult` (`__len__`).

**Exceptions** — `UnikoError` (base, carries a `.kind` string attribute) plus `ConfigError`, `LlmError`, `TimeoutError`, `ConflictError`, `UnsupportedError`. The five dedicated subclasses map from the corresponding `UnikoError` variants; the seven storage-domain variants (Storage/Search/Schema/Pipeline/Locy/Embedding/Internal) fold into the base — always distinguishable via `.kind`.

**The Pydantic overlay** — `uniko.models.*` — a pure-Python layer that never clobbers the native top-level names (`uniko.ContextBundle` stays the native class; `uniko.models.ContextBundle` is the Pydantic model). It adds validation, JSON-schema/serialization (FastAPI-ready), and "typed handles" that auto-accept specs and auto-return models:

```python
from uniko.models import TypedUniko, TurnSpec, GoalSpec, ContextBundle

uni   = await TypedUniko.in_memory()
agent = uni.agent("assistant")
await agent.session("s1").observe(TurnSpec(sender_id="alice", content="I love hiking"))

bundle: ContextBundle = await agent.recall("hobbies")   # Pydantic model
print(bundle.model_dump_json())

gid = await agent.goals.create(GoalSpec(title="Plan trip"))
```

Input specs (`GoalSpec`/`TaskSpec`/`ScopeSpec`/`IngestSourceSpec`/`TurnSpec`/`LlmSpecModel`) use `extra="forbid"` and **lazy native imports** inside `to_native()`, so the models import even with no compiled extension present — you can generate OpenAPI schemas from `uniko.models` without the engine installed. `IngestSourceSpec` and `LlmSpecModel` are discriminated unions producing clean `oneOf` schemas. Output models use `from_attributes=True, frozen=True` so `model_validate` reads straight off the frozen native pyclass.

### 8.9 Python behavioral gotchas

- **`Uniko.shutdown()` poisons the handle.** It `take()`s the internal `Option`, modeling the consuming Rust `shutdown`. Any `Agent`/`Session` objects derived from it must be dropped first; subsequent use raises `ConfigError("shut down")`.
- **`Session` serializes turns.** `PySession` holds `Arc<tokio::Mutex<Session>>` and locks *across* the await because `observe` takes `&mut self` and owns cross-turn state. A stuck `observe` blocks the whole session — this matches the documented single-threaded contract.
- **Binary payloads go through `Data.artifact_bytes`.** `Value::Bytes` cannot round-trip through a Cypher `RETURN` (a uni-db limitation), so `artifact_bytes(id)` reads the blob store directly. Do not expect bytes to come back through `query`.
- **Pass tz-aware datetimes.** `py_to_datetime` uses Python's `.timestamp()`; a tz-naive datetime is interpreted in the *local* timezone per CPython.
- **Graph elements render as plain dicts.** `Node`/`Edge`/`Path` values from `query`/`run` come back as `{id, labels, properties}` / `{id, type, start, end, properties}` / `{nodes, edges}` — not dedicated classes.
- **Scope visibility (`as_viewer`) is deferred in Python** — only session/participant/time dimensional filters exist on `PyScope`; instance-level `scope_to_agent` covers the common visibility case.

### 8.10 The three wheel variants and feature flags

`uniko-cuda` and `uniko-metal` are **not separate source trees**. Both point their `[lib] path` at `../uniko-py/src/lib.rs`, and their Python package is *copied* (not symlinked — maturin won't follow symlinks) from `bindings/uniko-py/python` by `scripts/bootstrap-wheel-variants.sh`. All three set `module-name = uniko._uniko` and `python-source = python`, so all import as `uniko`. **The only difference is the `uniko-api` feature set forwarded down the stack:**

| Wheel | Platform | `uniko-api` features | GPU stack |
|---|---|---|---|
| `uniko` (base) | Linux x86_64/aarch64, macOS arm64, Windows x64 | `[onnx]` | none (CPU ONNX Runtime, statically bundled) |
| `uniko-cuda` | Linux x86_64 | `[onnx, gpu-cuda, mistralrs, candle]` | ONNX Runtime CUDA EP + mistralrs/candle CUDA kernels |
| `uniko-metal` | macOS arm64 | `[onnx, gpu-metal, mistralrs, candle]` | ONNX Runtime CoreML + mistralrs/candle Metal kernels |

All wheels are `abi3` for CPython ≥ 3.10 — one wheel per platform spans every supported minor version. GPU wheels resolve host CUDA/cuDNN at load time rather than bundling them.

**Building the wheels:**

```sh
# Base CPU wheel:
maturin build --profile dist -m bindings/uniko-py/Cargo.toml

# GPU variants — copy the python package in FIRST, then build:
scripts/bootstrap-wheel-variants.sh
CUDA_COMPUTE_CAP=80 maturin build --profile dist --auditwheel skip \
    -m bindings/uniko-cuda/Cargo.toml          # Ampere PTX, fwd-JITs to newer archs
maturin build --profile dist -m bindings/uniko-metal/Cargo.toml   # macOS/aarch64
```

`dist` is the release profile plus whole-graph thin LTO at `codegen-units = 1`
(root `Cargo.toml`); it is what CI publishes. Use `--release` for a faster local
build when artifact size does not matter. On a machine with less than ~16 GB of
RAM, export `CARGO_PROFILE_DIST_CODEGEN_UNITS=16` first — cgu=1 peaks around
11-12 GiB of rustc RSS on a cdylib this size.

`bootstrap-wheel-variants.sh` **must** run before building a GPU variant (the `python/` dirs are gitignored and populated by it). The CUDA build requires the CUDA toolkit (`nvcc`) at build time; `--auditwheel skip` is used so CUDA runtime libs are resolved at load, not bundled.

The corresponding Cargo feature flags, and where they land:

| Feature | On `uniko-store` effect | Set by |
|---|---|---|
| `onnx` | (facade) adds `uniko-extract` ort extraction adapters; NLP runtime is always linked | all wheels |
| `gpu-cuda` / `gpu-metal` | forwards uni-db GPU acceleration | cuda / metal wheels |
| `mistralrs` / `candle` | local-LLM provider kernels via uni-xervo | cuda / metal wheels |
| `llm` (`uniko-memory`) | abstractive session summaries | off by default |
| `batch-record` (`uniko-store`) | diagnostic bulk-batch capture (benchmarks only) | off |

### 8.11 The agent-tools surface (subjective vs inferred writes)

Beyond `observe` (which *infers* entities/observations from message content), the facade exposes explicit **subjective** acts that cannot be inferred — goals, tasks, episodes, actions, direct fact assertions, summaries, query-episode recording. In Rust these are free functions in `uniko-memory` of the shape `(&KnowledgeBase, agent_id, Params) -> Result<...>`, each bound as an `Agent`/`Goals`/`Session` method. They are deliberately **not** in `uniko_api::tools` as free functions (that would leak the `&KnowledgeBase` store handle) — they surface only as methods:

| Tool (feature) | Bound as | Effect |
|---|---|---|
| `create_goal` (F8) | `Goals::create` | `Goal` + `OWNED_BY`/`PARENT_GOAL` + embedding |
| `create_task` (F9) | `Goals::create_task` | `Task` + `ASSIGNED_TO`/`PART_OF`/`DEPENDS_ON`/`SUBTASK_OF` |
| `record_episode` | `Agent`/`Session` | `Episode` + `RECORDED_BY`/`FOLLOWED_BY`(1-hr window)/`TRIGGERED_BY`/`INVOLVES` — feeds P5 procedure promotion |
| `record_action` (F17–20) | | `Action` node + edges; output overflow spills to an `Artifact` via `PRODUCED` |
| `add_observation` (F34) | | `Observation` anchored `OBSERVED_IN` a `Message` |
| `assert_fact` (F33) / `invalidate_fact` (F37) | | direct bitemporal `Fact` assert / BTIC-close |
| `generate_session_summary` (F59) | `Session::summarize` | extractive by default; LLM-abstractive behind the `llm` feature |
| `answer_query` / `record_query_episode` | `Agent::answer` | recall + generator closure + opt-in query `Episode` (feeds P5 from production traffic) |

All the recording tools require the agent's `Participant` to already exist. Embedding failure still creates the node (back-fillable) but returns the error.

**Locy reasoning** is reachable through `Agent` in both languages:

```python
rows = await agent.assume(
    "ASSUME { CREATE (:Fact {subject:'server',predicate:'port',object:'9090'}) }"
).then_query("MATCH (f:Fact {subject:'server'}) RETURN f").run()
```

`assume` forks the graph, runs the `THEN` query, and rolls back — it never mutates. `abduce` returns an `AbductionResult` of ranked `AbducedModification`s. `define_rule`/`run_rule` register and invoke Locy rules through the confidence-driven lifecycle.

### 8.12 Notes and caveats for API consumers

- **`reference_ts` is a per-query anchor, never a KB setting.** Recalling a historical corpus without setting a reference timestamp computes the temporal window around *now*, which never overlaps old data and silently disables the Phase-2 temporal channel. (In the Rust `RecallConfig` this is `reference_ts`; the LME bench sets it to the question date.)
- **Multi-query variant reformulation is opt-in.** The default is keywords-only. Enabling all four variants measured −2.1 pt evidence and 3× latency on LoCoMo — don't turn it on expecting a win.
- **The reranker is on by default** (`cross-encoder/ms-marco-MiniLM-L-6-v2`, top_n=50, ~50–100 ms/query). Disable via `RerankerConfig { enabled: false, .. }` if latency matters more than precision. (The config *doc comment* saying "disabled by default" is stale — the `Default` impl enables it. Trust the code.)
- **Embedding dimension is fixed at DB creation** — it's baked into the on-disk vector index. Switching embedders (e.g. to a 1024-d model) requires a fresh KB; `validate()` catches zero dims but not a model/dimension mismatch on reopen.
- **The negative-surface guards are the API contract.** If you extend `uniko-api::tools`, the nine `compile_fail` doctests and `tests/surface.rs` will break on drift — treat them as load-bearing, not incidental.
- **Not yet shipped** (do not build against): HTTP/MCP server, CLI, cross-agent sharing, rule induction, MCTS planning. The Python SDK is alpha and its API may change before 1.0.

---

## 9. Benchmarks & Performance

uniko ships with its own measurement crate — `crates/uniko-bench` (`publish = false`, an internal tool, never released to crates.io). It exists to answer three questions with committed artifacts:

1. **How accurate** is uniko's recall + answer pipeline on public long-conversation QA benchmarks (LoCoMo, LongMemEval)?
2. **What does a query cost** in LLM tokens/USD, and how long does it take (recall latency vs generation latency)?
3. **Where does write-path time go** during ingest, so uni-db and NLP regressions can be isolated into minimal repros and filed upstream?

The crate is **CPU-by-default** (`features = default = []`) so `cargo check/clippy/nextest --workspace` builds on CUDA-less CI. GPU is opt-in via `--features gpu-cuda` / `gpu-metal`, which `crates/uniko-bench/run.sh` supplies along with the ORT CUDA execution provider, cuDNN-13, and a linker shim. It defines **eleven `[[bin]]` targets**: two full benchmark harnesses, a Cypher console, an NLP comparison + parity pair, and five write-path microbenches.

| Binary | Kind | Purpose |
| --- | --- | --- |
| `uniko-bench` | Harness | LoCoMo10 accuracy + cost + latency |
| `longmemeval-bench` | Harness | LongMemEval retrieval + judge |
| `uniko-cypher` | Tool | Cypher console over an ingested KB |
| `nlp-compare` | Tool | NLP cascade latency across model sizes |
| `nlp-parity` | Tool | A/B in-crate NLP decode vs xervo |
| `insert-microbench` | Microbench | Node/edge insert: Cypher vs bulk |
| `update-microbench` | Microbench | UNWIND-SET per-row cost repro |
| `mutation-set-microbench` | Microbench | Which Entity index drives MutationSet cost |
| `bulk-vs-unwind` | Microbench | Bulk API vs UNWIND on real batch distributions |
| `profile-writes` | Microbench | No-contention per-call write floor |

Everything the harnesses need — model, device, recall, and cost knobs — lives in a single `--bench-config <path>.json` (`crates/uniko-bench/src/bench_config.rs`). The CLI carries only what varies per invocation (data file, output path, conversation/category filters, KB reuse). The legacy flat-flag surface was retired; a pre-parse pass, `reject_retired_flags` (`main.rs:154`), emits migration hints pointing each old flag at its new JSON location.

---

### 9.1 The LoCoMo10 harness (`uniko-bench`)

**LoCoMo** (Long Conversational Memory) is a benchmark of long multi-session dialogues between two speakers, each carrying a set of QA pairs across five categories. The loader (`crates/uniko-bench/src/data.rs`) parses the flattened dynamic `session_N` / `session_N_date_time` keys of each conversation via regex and classifies questions (illustrative variant→category-number mapping; the real enum has bare variants and the `1..=5` numbering comes from a match at `data.rs:93-97`):

```text
QuestionCategory          category #
  MultiHop     1   // answer spans multiple turns/sessions
  Temporal     2   // date-anchored ("when did …")
  OpenDomain   3
  SingleHop    4   // single evidence turn
  Adversarial  5   // the answer is "not mentioned"
```

#### Control flow

The per-conversation loop (`main.rs:299-512`) is the spine of the whole harness:

```
reject_retired_flags(argv)              # migration errors before clap
  → BenchConfig::load
  → data::load_locomo → filter by --conversations / --categories
  → UnikoConfig::default() + bench_cfg.apply_to_uniko_config    (:214)
  → build_catalog_specs
  → KnowledgeBase::build_shared_runtime                          (:234)   ← ONE xervo runtime
  → load Pricing + open EventWriter
  for each conversation:
      parse_sessions
      → open_kb_with_runtime(kb_dir, config, shared_runtime)
      → reuse-or-ingest      (reuse_existing checked BEFORE open, :321)
      → run_post_ingest_sweep    (P4 consolidation + P5 procedures + P6 topics)
      → ensure_bench_agent
      for each question:
          resolve_evidence → run_query → token_f1 → llm_judge (skip Adversarial)
          → write Query event (answer_cost + judge_cost)
          → record_query_episode      (feeds P5; f1 as importance)
      shutdown_kb + checkpoint write_json_with_pricing    (aggregate :506, checkpoint write :508)
```

Two structural details matter. First, the **shared `ModelRuntime`** (`main.rs:234`) is built once and passed to every per-conversation KB so N conversations reuse one set of ONNX sessions instead of OOMing an 8 GB GPU — and it is the *only* open path that lets a bge-m3 hybrid KB reopen (the catalog-open path rejects non-`Embed` vector-index aliases per uni-db #130). Second, the **per-conversation checkpoint** (`write_json_with_pricing`) is written after each conversation completes, so a `SIGKILL` mid-run preserves finished conversations — the checkpoint-batch discipline the project follows for all long jobs.

`RecallConfig` is derived from `kb.config()` via `RecallConfig::from_uniko_config` (`query.rs:100`), so the bench honors the reranker/recall knobs set at ingest rather than falling back to `RecallConfig::default()`.

#### Scoring

Three metrics run per question (`crates/uniko-bench/src/eval.rs`):

- **Token-F1** — the original LoCoMo metric. `token_f1` (`eval.rs:21`) dispatches on category: Porter-stemmed bag-of-words F1 for the common case (lowercase, strip punctuation, drop articles `a/an/the/and`, harmonic mean of precision/recall); multi-hop splits the gold answer on commas and averages the per-sub-answer F1; adversarial is binary over a negation-phrase whitelist that accepts both past-participle ("not mentioned") and verb-form ("does not mention") abstentions.
- **Evidence hit-rate** — `evidence_hit` (`eval.rs:177`) is a two-stage retrieval metric: build three fingerprints per evidence text (head-50 chars / skip-first-word / middle-slice, all UTF-8-boundary-safe via `floor/ceil_char_boundary`), substring-match against recalled contents, then a gold-answer word-boundary fallback that fires only when zero evidence matched and the gold length ≥ 4.
- **LLM judge** — `llm_judge_with_usage` (`eval.rs:313`; the verbatim prompt `format!` begins at `eval.rs:327`) runs the **verbatim Mem0 CORRECT/WRONG prompt** so uniko's numbers are directly comparable to published Mem0 figures. `parse_mem0_judge_label` prefers the JSON `label` field, then last-occurrence `CORRECT/WRONG`, defaulting to `WRONG` on ambiguity. Adversarial questions are excluded from the judge (`main.rs:395`) and scored only by the negation whitelist.

#### Cost model

`report.rs` deliberately separates two costs:

- `total_answer_cost_usd` — **customer usage**, the headline figure. `cost_per_question_usd` is answer-only.
- `total_judge_cost_usd` — **bench-only evaluator overhead**.
- `total_query_cost_usd` (answer + judge) is retained but **deprecated** — never use it for headline numbers.

Pricing (`crates/uniko-bench/src/pricing.rs`) reads `pricing.csv` into `Rates { input_per_m, output_per_m, embedding_per_m: Option<f64> }`. A blank CSV cell is `None`, never a fabricated zero; unknown models warn once. Embedding tokens are **estimated as `chars/4`** because uni-db's `xervo.embed()` discards the provider usage struct (`events.rs:8`, `ingest.rs:272`) — treat embedding cost as approximate, and note it is nonzero only for remote embedders (local models are `$0` by construction).

Two cost gotchas are load-bearing:

- Pricing keys are HF model ids, but `run_query` sets `answer_model` to the *alias* (`llm/gen`); `main.rs:388` substitutes the real `gen_model_id` back before the cost lookup — forget it and costs silently read `$0`.
- `gemini-3.1-pro-preview` **thoughts-tokens are not captured** (uni-xervo drops `thoughtsTokenCount`); `pricing.csv` warns the real Vertex bill may be **5–50× the reported cost**. The `$3.55` headline understates true reasoning-judge spend.

---

### 9.2 The LongMemEval harness (`longmemeval-bench`)

**LongMemEval (LME)** tests memory over a "haystack" of many sessions, asking questions of six types. The loader (`crates/uniko-bench/src/longmemeval/data.rs`). The real `LmeQuestionType` has **bare variants**; the string codes below come from an `as_str()` match (`data.rs:73-78`), not enum discriminants (Rust forbids `&str` discriminants) — read this as an illustrative variant→code mapping:

```text
LmeQuestionType          code (via as_str)
  SingleSessionUser        "SSU"
  SingleSessionAssistant   "SSA"
  SingleSessionPreference  "SSP"
  MultiSession             "MS"
  TemporalReasoning        "TR"
  KnowledgeUpdate          "KU"
```

Abstention questions carry an `_abs` suffix. Structurally the LME harness (`longmemeval_main.rs`) differs from LoCoMo in two ways:

- **Items run concurrently** via `stream::for_each_concurrent(question_concurrency)` (`:198`), each holding its own KB over the shared runtime.
- **Within an item, sessions ingest concurrently** via `buffer_unordered(session_concurrency)` (`ingest.rs:246`) with atomic counters and a `Mutex<EvidenceMap>`. Pronoun resolution stays causal *within* a session — concurrency is across sessions, not turns.

Because per-turn timing differs (LoCoMo ingests a conversation single-threaded in a sequential turn loop; LME ingests sessions concurrently), **per-turn ms is not comparable across the two harnesses**.

The critical LME correctness detail is `reference_ts = question_date` (`query.rs:70`): setting the recall temporal anchor to ask-time (not wall-clock) lets the Phase-2 temporal channel resolve "last May" against a corpus that is two years old. Without it the temporal window centers on `now`, never overlaps old data, and silently disables the channel.

LME metrics (`longmemeval/eval.rs`):

- `context_contains_answer` — normalized substring of the gold answer in the top-k recalled contents; the **Phase-1 retrieval gate** metric.
- `recall_at_k` — `|answer_sessions ∩ retrieved_top_k| / |answer_sessions|` (session-level).
- `ndcg_at_k` — `DCG = Σ 1/log2(rank+2)` over relevant retrieved sessions, normalized by ideal DCG.
- `lme_judge` — category-specific yes/no prompt (TR allows off-by-one days, KU checks the updated value is present, SSP checks personal info is used).
- `abstention_score` — yes/no over a refusal-phrase whitelist.

A dedicated `--phase1` mode runs only SSU+SSA+MS in retrieval-only mode and gates **PASS at overall `context_contains@R5 ≥ 0.90`** (`:396`) — the fast retrieval-quality check used during recall tuning before spending on generation.

---

### 9.3 Headline numbers (with context)

The numbers of record are **LoCoMo10, gemini-3.1 judge, measured 2026-05-26**, using Mem0's verbatim judge prompt, on a 22-core CPU + 8 GB consumer GPU. The canonical source artifact is `data/locomo_gemini31_merged.json`.

| Metric | Value | Notes |
| --- | --- | --- |
| LLM-judge accuracy | **0.8117 (81.2%)** | full **1,986-question** set |
| Retrieval hit rate | **0.8555 (85.6%)** | `evidence_hit` |
| Token-F1 | **0.321** | Porter-stemmed bag-of-words |
| Total LLM cost (answer + judge) | **$3.55** | understated — Vertex thoughts-tokens uncaptured |
| Ingest | **5,882 turns in 7.5 min at $0** | ~76 ms/turn, LLM-free |
| Mean Q&A latency | **4.04 s** | 2.84 s recall + 1.20 s generation |

**Two crucial caveats** that the docs themselves flag (`website/docs/benchmarks/index.md`, `why-uniko.md:142-144`):

1. **Two question sets.** The `81.2%` judge figure is the **full 1,986-q** run. The KTH cost/latency tables below use the **1,540-q non-adversarial subset** (adversarial questions bypass the judge). *Do not conflate them.*
2. These are **self-measured internal harness figures**, not a third-party leaderboard.

#### Comparison vs six systems (KTH dmas-memory study)

The competitive tables come from an independent KTH study (Wolff & Bennati, arXiv:2601.07978, measured 2026-06-14). uniko's claimed wins are **operational, not top judge accuracy**.

LoCoMo10 judge accuracy (1,986 q), for orientation:

| System | Judge |
| --- | --- |
| Mem0 | 91.6% |
| **uniko** | **81.2%** |
| Graphiti (Zep) | 75–84% |
| Letta (MemGPT) | 74.0% |
| LangMem | 58.1% |
| Cognee | — |

**Ingest** the full 5,882-turn corpus:

| System | Cost | Tokens | Wall time |
| --- | --- | --- | --- |
| **uniko** | **~$0** | **0** | **7.5 min** (~76 ms/turn) |
| cognee | $1.32 | — | 493 min |
| mem0 | $4.82 | — | 251 min |
| graphiti | $5.49 | — | 569 min |

uniko is **33–76× faster to ingest** than the graph backends, at **$0**, because ingest is LLM-free: a local ONNX cascade (kniv-deberta INT8) does POS/NER/SRL/DEP/CLS in one encoder pass at write-time.

**Q&A per question** (1,540-q non-adversarial subset):

| System | Wall time | Tokens | Total cost |
| --- | --- | --- | --- |
| **uniko** | **4.04 s** | **2,468** | **$1.01** |
| graphiti | 6.20 s | 4,546 | — |
| cognee | 6.99 s | 4,780 | — |
| full_context | 9.51 s | 45,708 | — |

uniko posts the **fastest Q&A wall-time of the six systems**, and every competitor requires an external service (Graphiti→Neo4j/FalkorDB, Mem0→Qdrant, Cognee→graph+vector+relational) while uniko runs entirely in-process. The architectural reason recall is fast and cheap is that there is **no LLM in the recall hot path** — recall queries the *compiled* Entities/Observations/Facts rather than re-deriving them ("compile once, query forever").

> **Provenance note.** The reports `initial-docs/perf-journey.md` and `initial-docs/kth-dmas-comparison.md` referenced elsewhere are **not present in the tree** (`initial-docs/` is absent). Their numbers survive in `website/docs/benchmarks/index.md` and the auto-memory. The canonical headline source is `data/locomo_gemini31_merged.json`.

---

### 9.4 Write-path microbenchmarks

The microbenches exist to attribute ingest time precisely — most of it lands in uni-db's Cypher executor, and these binaries isolate exactly where. They need no GPU and run against `tempfile` scratch DBs (or persistent Lance).

#### `insert-microbench` — Cypher vs bulk, contention probe

Sweeps three axes: **api** (Cypher vs `bulk_insert_vertices`/`bulk_insert_edges`) × **label-mode** (`Same` = one table vs `PerWorker` = `Item0..N`, to distinguish per-table from global write contention) × **concurrency** (`--sess`). Each worker owns its transaction and commit; the binary reports `QueryMetrics` parse/plan/exec µs.

```sh
cargo build -p uniko-bench --bin insert-microbench --release
./target/release/insert-microbench --nodes 510 --sess 1,8,24 --reps 3 --op both
```

This is where the **~980× bulk-vs-Cypher gap** was measured (bulk ~150 µs/edge vs Cypher ~147 ms/edge at sess=24), which drove uniko-store's ingest hot paths to bypass the Cypher executor entirely (`batch_create_nodes_in_tx → tx.bulk_insert_vertices`, `create_message_edges_in_tx` → direct bulk calls).

#### `update-microbench` — the UNWIND-SET repro

Reproduces a **non-monotonic** per-row cost in `UNWIND … MATCH WHERE id(n)=u.nid SET …`: 1.9 ms @ batch 1 → 12 ms @ batch 3 → amortizes at large batches. It dumps `.profile()` per-operator stats and shows that **~98% of UPDATE time is attributed to `MutationSetExec`/`GraphScanExec`, both of which report `time = 0 ms`** — a uni-db profiling blind spot (not a measurement bug), captured as a minimal upstream repro.

#### `mutation-set-microbench` — which index costs 17 ms/row

Runs persistent-Lance variants (`noindex`, `hash_only`, `fulltext_name`, `vector_only`, …) to isolate which Entity index drives the ~17 ms/row `MutationSetExec` cost during entity counter updates.

#### `bulk-vs-unwind` — record → replay on real batches

The most faithful write-path bench. It **records the real LoCoMo batch-size distribution** during a live conv-26 ingest+sweep (`enable_batch_recording` behind the `batch-record` feature), then **replays each captured batch** through both the bulk API and a hand-built UNWIND, rolling back (no mutation), taking the **median of `--reps`** with alternating arm order to cancel warmup bias.

```sh
./target/release/bulk-vs-unwind --data data/locomo10.json --conversations conv-26 \
    --bench-config crates/uniko-bench/bench-configs/locomo-arm0-bge-small-baseline.json --reps 5
```

Measured speedups: **524× on edges**, **49.6× on nodes-no-embed**, and only **1.4× on nodes-with-embed** — the last because at ingest the embedder dominates, so the write-API gap is invisible on the embed-bound node path. (The binary needs `#![recursion_limit = 512]` because the chained-await replay future exceeds the default type-layout recursion depth.)

#### `profile-writes` — the no-contention floor

Runs `ExecuteBuilder::profile()` on per-edge-type UNWIND-MATCH-CREATE patterns against a populated KB to isolate the single-writer per-call write floor.

#### `uniko-cypher`, `nlp-compare`, `nlp-parity`

`uniko-cypher` is a one-shot/stdin/REPL Cypher console over an existing KB (`open_with_xervo`) with pretty/json/csv output and `--profile` — a data-inspection tool:

```sh
./target/release/uniko-cypher --kb data/kb/conv-26 \
    --query "MATCH (m:Message) RETURN count(m) AS n" --format json
```

`nlp-compare` runs the NLP cascade through xsmall/small/base variants over LME messages measuring per-call latency and entity/observation counts (no graph writes). `nlp-parity` A/B-tests uniko's in-crate NLP decode against `uni_xervo`'s `NlpModel::analyze` over the same ONNX artifact, reporting per-axis agreement (tokenization / POS / NER-Jaccard / DEP arcs / CLS / SRL) — the parity proof that justified migrating decode into xervo, and which pinned the three representation gaps the adapter reconciles (SentencePiece metaspace word reconstruction, DEP head→word mapping, BIO span merging).

---

### 9.5 The perf journey (~300× ingest speedup)

The ≈300× number is one *369-turn* LoCoMo conversation going from **2h 7m (2026-04-21) to ~22–28s** at ~62 ms/turn (`website/docs/benchmarks/index.md:145-146`). Do **not** conflate it with the 7.5-min figure, which is the *full 5,882-turn corpus* (a different, ~16× larger workload) at the same per-turn rate. The speedup is the compounding of several changes surfaced by the microbenches above:

1. **Atomic per-message ingest.** `ingest_message_atomic` (`crates/uniko-extract/src/ingest/atomic.rs`) folded the three legacy per-message steps (message create, entity extraction, observation extraction) into a single prep-then-commit flow. All CPU work and read-only lookups happen *before* the transaction opens; then **one** transaction writes Message + edges + chunks + Entity upserts + MENTIONS + Observations + OBSERVED_IN + ABOUT and commits once. This cut **commits-per-message from 3 → 1**.

2. **Bulk write paths.** The ~980× Cypher-executor per-row overhead (from `insert-microbench`) was bypassed on the hot paths: `batch_create_nodes_in_tx`/`batch_create_edges_fast_in_tx` route through `tx.bulk_insert_vertices`/`bulk_insert_edges`, and `create_message_edges_in_tx` hand-writes SENT_BY/IN_SESSION/ADDRESSED_TO/NEXT as direct bulk calls. Bulk skips the Cypher executor (VIDs are already known), doesn't return EIDs, and doesn't re-validate property names — so callers validate keys up front.

3. **LLM-free ingest.** All write-time enrichment is deterministic CPU work — regex + tree-sitter + a small quantized DeBERTa cascade — so ingest is `$0`, offline, and reproducible. Facts/Procedures/Topics are derived later in async consolidation workers, off the write path.

4. **Concurrency-safe hot paths.** uni-db runs with SSI; two concurrent RMW callers don't lose updates (the second aborts with a retriable conflict, retried by `transact_with_retry`). But SSI does not catch insert-phantoms, so entity dedup relies on `StripedLocks` (256 tokio mutexes) keyed by canonical prefixes, acquired *before* the transaction opens and held across commit. A hand-rolled per-fact lock loop once self-deadlocked `batch_upsert_facts` past ~50 facts; `lock_many` dedups by *stripe index* to prevent that.

> **A caveat the memory carries:** the 205 ms/turn figure once quoted was LoCoMo single-process; LME/GPU/parallel costs differ wildly. Per-turn timing is context-dependent — re-measure per config, and never compare LoCoMo's sequential per-turn ms against LME's concurrent ingest.

The shared library (`crates/uniko-bench/src/lib.rs`) holds KB lifecycle, catalog construction, context formatting, and the post-ingest P4/P5/P6 sweep (`run_post_ingest_sweep`) precisely so a future `uniko-api` HTTP surface can reuse the exact validated behavior the benchmarks exercise.

---

### 9.6 Running the benchmarks

**Full LoCoMo run (GPU)** via `run.sh`, which supplies `--features gpu-cuda` plus the CUDA runtime env:

```sh
./crates/uniko-bench/run.sh \
    --bench-config crates/uniko-bench/bench-configs/locomo-bge-gemini31.json \
    --data data/locomo10.json --conversations conv-26 \
    --output data/results.json          # per-conv checkpoints + _events.jsonl alongside
```

**CPU-only** (no CUDA toolkit): `NO_GPU=1 ./crates/uniko-bench/run.sh …`.

**All 10 conversations, one invocation each** — `scripts/run-all-convs.sh` wraps each conversation in its own `run.sh` leg so the Vertex OAuth token (which expires ~60 min) refreshes per conversation; a single `gemini-3.1-pro-preview` judge conversation runs ~32 min:

```sh
./crates/uniko-bench/scripts/run-all-convs.sh \
    crates/uniko-bench/bench-configs/locomo-bge-gemini31.json data/locomo10.json data/locomo_gemini31
./crates/uniko-bench/scripts/summarize-multi-conv.py data/locomo_gemini31   # answer-only cost rollup
```

**LongMemEval Phase-1 gate:**

```sh
BIN=longmemeval-bench ./crates/uniko-bench/run.sh \
    --bench-config crates/uniko-bench/configs/lme_default.json \
    --data data/longmemeval_s_cleaned.json --phase1        # PASS if contains@R5 ≥ 90%
```

Add `--reuse` to skip ingest and re-query an existing KB (default to this — don't `rm -rf data/kb*` unless the ingest pipeline changed or a `kill -9` broke the WAL).

**Bench-config profiles.** `crates/uniko-bench/bench-configs/` holds ~15 LoCoMo profiles (baseline bge-small + gpt-4o-mini, gemini31, embeddinggemma, the arm0/A/B/C sparse+ColBERT sweep, retrieval-only variants); `configs/` holds the LME twins. Every model/device/recall/cost knob is a JSON field; the CLI stays minimal.

Two operational invariants worth internalizing before trusting a run:

- `reuse_existing` **must be computed before** `open_kb_with_runtime`, because `open` creates `kb_dir` and a naive `exists()` check afterward always returns true (`main.rs:321`).
- The **shared `ModelRuntime` is mandatory** at `question_concurrency ≥ 3` — per-KB ONNX sessions OOM an 8 GB GPU, and the prebuilt-runtime open path is the only way a bge-m3 hybrid KB reopens.

---

## 10. Configuration, Build, Testing & Operations

This chapter is the operator's and integrator's reference for uniko: what you can configure, how the code is built and linked, how it is tested and released, what it depends on from the host system, and how the GPU variants differ from the default CPU build. Everything here is grounded in the actual configuration types, build manifests, CI gates, and release runbook that ship in the repository.

### 10.1 The configuration surface

uniko is configured in three layers, in increasing order of specificity:

1. **Compiled-in defaults** — `UnikoConfig::default()` (`crates/uniko-store/src/config.rs`). This fixes the embedding model, reranker, NLP cascade, chunking, recall, and consolidation parameters for a KB that is opened with no overrides.
2. **External catalog / schema overrides** — `config/catalog.json` (the xervo model catalog) and `config/schema.json` (a persisted installed-schema snapshot), selected via `UnikoConfig.catalog_path` / `UnikoConfig.schema_path`.
3. **Programmatic overrides** — build a `UnikoConfig` (or drive the `UnikoBuilder` in `uniko-memory`) and set fields, then `validate()`.

`UnikoConfig` is `Serialize`/`Deserialize` with struct-level `#[serde(default)]`, so an empty JSON object `{}` must deserialize to `Default` — a regression test guards this invariant. Do not rely on this behavior loosely: a missing field falls back to its opinionated default, which is not always the "off" value you might expect (the reranker is the classic trap — see below).

#### 10.1.1 `UnikoConfig` at a glance

`UnikoConfig` (config.rs:616) is the root runtime config. Its sub-configs and their spec-mandated defaults (config.rs:826–895):

| Area | Field(s) | Default |
|---|---|---|
| Embedding | `embedding: EmbeddingConfig` | `bge_small_en_v15` — BGE-Small-EN-v1.5, **384-d**, query-side prefix only (`"Represent this sentence for searching relevant passages: "`; documents go in raw) |
| Reranker | `reranker: RerankerConfig` | **ENABLED**, `cross-encoder/ms-marco-MiniLM-L-6-v2`, `top_n=50`, `apply_sigmoid=true`, `style=cross-encoder` |
| NLP | `nlp: NlpConfig` | `kniv-deberta` xsmall + `onnx/cascade-int8.onnx`, `nlp_srl_enabled=true` |
| OCR | `ocr: OcrConfig` | disabled |
| Vector index | `vector_algorithm` / `metric` | `HnswSq{ m:16, ef_construction:100 }`, `Cosine` |
| Blob storage | `blob_storage: BlobStorage` | `Lance` (inline) |
| Consolidation | trigger threshold / interval | `20` observations / `900 s` |
| Chunking | `message_chunk_threshold`, `max/min_chunk_tokens` | `1024`, `256` / `32` |
| Recall | `recall_limit`, `recall_token_budget` | `15`, `8192` |
| Fusion | `vector_weight` / `bm25_weight`, `rrf_k` | `0.5` / `0.5`, `60` |
| Recall phases | `phase1_strategy` (α), phase-2 coverage gate | `boost` (α `0.6`), `0.65`. Phase 1's gate is the hardcoded `COVERAGE_GATE_PHASE1`, not config. |
| Entity admission | `entity_strict_admission`, `entity_other_min_confidence` | `true`, `0.9` |
| Session upkeep | `session_inactivity_secs` | `3600` |
| External overrides | `catalog_path`, `schema_path`, `observation_rules_path` | `None` |

`validate()` enforces cross-field constraints. Two that matter in practice: reranker `top_n` must be `>= recall_limit`; and enabling `recall_sparse_enabled` or setting `reranker.style=colbert` is rejected unless the embedding config actually has `sparse_dimensions` / `multivector_dimensions` set (i.e. a hybrid embedder is configured). `validate()` catches zero dimensions but **cannot** catch a model/dimension mismatch against an existing on-disk index.

> **Trust the code, not the prose.** A comment near config.rs:636 says the reranker is "disabled by default", which contradicts the actual `Default` impl (`enabled: true`). The reranker costs ~50–100 ms/query; if you don't want it, set `RerankerConfig{ enabled: false, .. }` explicitly.

> **Effective defaults come from `UnikoConfig::default()`.** The pipeline builds its `RecallConfig` via `RecallConfig::from_uniko_config(...)`. The standalone `RecallConfig::default()` and `ChunkConfig::default()` differ from what the running system uses — do not quote those as runtime defaults.

#### 10.1.2 Embedding presets

`EmbeddingConfig` (config.rs:22) carries `model_id`, `dimensions`, `batch_size`, prefixes, `provider`, plus the optional `sparse_dimensions` and `multivector_dimensions` that turn on hybrid retrieval. The presets:

| Preset | Dims | Notes |
|---|---|---|
| `bge_small_en_v15` | 384 | **default**, query prefix only (`"Represent this sentence for searching relevant passages: "`) |
| `minilm_l6_v2` | 384 | legacy `AllMiniLML6V2` |
| `nomic_v15` | 768 | requires `search_document:` / `search_query:` prefixes |
| `bge_large` | 1024 | larger dense |
| `bge_m3` | 1024 dense + sparse(250002) + ColBERT multivector | the only **hybrid** embedder today |
| `embeddinggemma` | — | alternate preset |

The hybrid (`bge-m3`) path needs **two xervo aliases** backed by the same model: `embed/default` (`ModelTask::Embed`, dense) and `embed/hybrid` (`ModelTask::EmbedHybrid`). The hybrid model implements only `HybridEmbeddingModel` and cannot back lone-dense columns, so the model loads twice (~2× VRAM). Sparse (`SparseVector`) and ColBERT (`List<Vector>`) columns exist on `Chunk`/`Observation` **only** when the hybrid dimensions are set; ColBERT vectors use `VectorAlgo::Flat` and feed only the MaxSim rerank, never first-stage retrieval.

> **Embedding dimension is frozen at KB creation** — it is part of the on-disk vector index. Switching embedders (e.g. BGE-Small 384-d → BGE-Large 1024-d) requires a **fresh KB**. Reopening a KB with a mismatched embedder/dimension corrupts the vector index.

#### 10.1.3 The model catalog (`config/catalog.json`)

The catalog is the xervo model manifest that maps aliases to concrete models/providers. The default `config/catalog.json` pins:

- `embed/default` → `BGESmallENV15`
- `nlp/default` → `kniv-deberta` xsmall + `cascade-int8.onnx`

`config/catalog_minilm.json` is an alternate that swaps the embed alias to `AllMiniLML6V2` (legacy 384-d). It also still carries the **pre-2026-06 `task: "raw"` NLP alias shape** (`options: {artifact, max_batch_size}`) rather than the current `ModelTask::Nlp` shape (`{onnx_path, tokenizer_path, label_maps_path, max_seq_len}`), so loading it via `catalog_path` yields a stale, incompatible NLP alias — it needs regenerating. Either catalog must match the dimensions the KB was created with.

Aliases used throughout the schema and search code are consts in `schema/mod.rs`: `EMBED_ALIAS`, `HYBRID_EMBED_ALIAS`, `NLP_ALIAS` (`"nlp/default"`), `RERANK_ALIAS`, `OCR_ALIAS`. LLM aliases (for `answer()`, triple refinement, topic naming, NL→Cypher) are registered separately through `LlmSpec` (`openai` / `openai_with_key_env` / `mistralrs`), which lowers to a uni-db `ModelAliasSpec` with `ModelTask::Generate` and a lazy warmup policy.

Two catalog subtleties from the benchmark harness generalize to any embedded use:

- A **shared `ModelRuntime`** (built via `KnowledgeBase::build_shared_runtime`, then passed to `open_with_runtime`) lets many KBs share one set of ONNX sessions / one VRAM arena. This is mandatory at concurrency ≥ 3 on an 8 GB GPU to avoid per-KB ONNX OOM.
- The prebuilt-runtime open path is the **only** way a `bge-m3` hybrid KB reopens — the catalog-open path rejects non-`Embed` vector-index aliases (a uni-db limitation, `#130`).

#### 10.1.4 The installed-schema snapshot (`config/schema.json`) and count drift

`config/schema.json` is a **persisted snapshot** of the installed schema — not the source of truth. Regenerate it with `cargo run --bin export-schema` (which prints the resulting label/edge counts) whenever the schema changes, and commit it on its own. It is only ever read when a caller sets `UnikoConfig::schema_path`; the default (`None`) registers the schema programmatically from `constants.rs`.

Always trust `constants.rs` (`labels::ALL` = 25, `edges::ALL` = 54). `register_schema` is idempotent and two-phase (labels, then edges, then a single `.apply()`); it installs everything from `constants.rs` regardless of what an older snapshot contains.

#### 10.1.5 NLP, OCR, and other model dependencies

The write-time NLP cascade default is `dragonscale-ai/kniv-deberta-nlp-base-en-xsmall` with the INT8 `onnx/cascade-int8.onnx` artifact — one shared DeBERTa encoder pass producing POS/NER/DEP/CLS, plus SRL (`nlp_srl_enabled=true` by default, which adds one extra ONNX forward per VERB per sentence). uniko-extract does not own the model; it resolves `NLP_ALIAS` through `kb.model_runtime()` and only reconciles representation gaps in `adapter::xervo_to_uniko`. If the runtime/alias is unavailable, `NlpPipeline::try_new` returns `None` and extraction falls back to rule-based NER.

Other model roles: rerankers default to `cross-encoder/ms-marco-MiniLM-L-6-v2` (with `BAAI/bge-reranker-base` and a generative `Qwen3-Reranker-0.6B` as alternatives, or in-process ColBERT MaxSim); OCR (tiered PDF, **off by default**) uses a PaddleOCR ONNX pipeline behind the `pdf-ocr` feature.

### 10.2 Environment perf knobs

Beyond `UnikoConfig`, a handful of environment variables tune the persistent-open path and measurement runs. These override `UniConfig` when opening a persistent KB (storage/mod.rs):

| Variable | Effect |
|---|---|
| `UNIKO_WAL_DISABLED` | disable the write-ahead log |
| `UNIKO_AUTOFLUSH_THRESHOLD` | override autoflush threshold |
| `UNIKO_AUTOFLUSH_INTERVAL_OFF` | disable interval-based autoflush |
| `UNIKO_BENCH_NO_MSG_EMBED=1` | pre-populate a zero message embedding to skip auto-embed — **measurement-only; invalidates recall** |
| `StripedLocks` stripe count | `UNIKO_RMW_STRIPES` (default 256) — `StripedLocks::from_env` supports env-driven sizing (default 256 stripes) |

The `RECALL_PROF` `eprintln!` lines in `recall/mod.rs` and `recall.rs` are unconditional stderr noise in the current hot path (they should be `tracing::debug!`) — expect them until removed.

### 10.3 Building uniko

#### 10.3.1 Toolchain and system prerequisites

- **Rust ≥ 1.91, edition 2024**, pinned via `rust-version` in the root `Cargo.toml` and `rust-toolchain.toml` (stable channel).
- A **C/C++ toolchain** (the ONNX Runtime and native crates need it).
- **`protoc`** (protobuf-compiler) on `PATH` — the stack statically links ONNX Runtime via `ort`, which requires `protoc` at build time.
- **`mold`** on Linux — `.cargo/config.toml` unconditionally sets `-C link-arg=-fuse-ld=mold`, so it is a hard build dependency, not an optimization.
- **`uv`** for the Python bindings and the docs site.

Dependencies come from crates.io with **no token, private repo, or credentials**: `uni-db 3.2.0` and `uni-xervo 0.17.0`. (Docs drift: `reasoning-with-locy.md` references `uni-db 2.4.1` Locy behavior; trust the manifest.)

> **The first build is slow.** `ort`, `tokenizers`, and `half` compile at `opt-level=3` even in the dev profile.

#### 10.3.2 Crate layout and the linear dependency stack

uniko is a Cargo workspace of layered crates over `uni-db` + `uni-xervo`:

```
uniko-py / uniko-cuda / uniko-metal   (PyO3 wheels)
        │
   uniko-api            (facade: logic-free re-exports)
        │
  uniko-memory (L4) ──► uniko-cortex (L5)   [deliberate reverse-altitude edge]
        │
  uniko-extract (L3)   uniko-pipes (L2)
        │
   uniko-store (L1)     ← the ONLY crate that touches uni-db
        │
     uni-db  +  uni-xervo
```

Layer numbers denote cognitive altitude, **not** build order — `uniko-memory` (L4) depends on `uniko-cortex` (L5) because cortex's P5/P6 sweeps subscribe to memory's P4 consolidation heartbeat.

**The uni-db seal** is the top build invariant, CI-enforced by a ripgrep gate: product crates (`uniko-memory`, `uniko-extract`, `uniko-cortex`, `uniko-pipes`) must never `use uni_db` or call `.db()`. They reach the graph only through `uniko-store`'s typed API. `.db()` remains a `pub` escape hatch for tests and the benchmark crate. Reviewed exceptions are tagged `// ALLOW:` on the same line; comment lines are exempt.

Six crates are publishable to crates.io: `uniko-store`, `uniko-pipes`, `uniko-extract`, `uniko-cortex`, `uniko-memory`, `uniko-api`. `uniko-bench` and `bindings/uniko-py` are `publish = false`.

#### 10.3.3 Feature flags

Feature flags forward straight down the stack (facade features on `uniko-api` re-forward to `uniko-memory` → `uniko-store`):

| Feature | Default | Crate(s) | Effect |
|---|---|---|---|
| `onnx` | off (facade) | uniko-extract, uniko-memory | ONNX NLP cascade + ONNX NER; declarative recall variant; abstractive answer path plumbing |
| `code-parse` | **on** (uniko-extract default) | uniko-extract | tree-sitter code AST entity extraction + code chunking |
| `pdf-ocr` | off | uniko-extract | tiered Native+OCR PDF doc-IR (`:Page`/`:Block`) |
| `llm` | off | uniko-memory, uniko-cortex | LLM abstractive summaries; LLM topic naming |
| `gpu-cuda` | off | uniko-store (+ down) | forward CUDA GPU to uni-db |
| `gpu-metal` | off | uniko-store (+ down) | forward Metal GPU to uni-db |
| `mistralrs` / `candle` | off | uniko-store | local-LLM providers via uni-xervo |
| `batch-record` | off | uniko-store | diagnostic bulk-batch capture for the bulk-vs-UNWIND bench |

Note that `uni-db`'s `provider-onnx` is always-on and statically linked regardless of the `onnx` facade feature; the `onnx` feature only adds uniko-extract's `ort`-backed adapters.

`uniko-bench` is CPU-by-default (`default = []`) so `cargo check/clippy/nextest --workspace` builds on CUDA-less CI; its `run.sh` supplies `--features gpu-cuda` plus the CUDA runtime environment when needed.

### 10.4 Testing and the check loop

**nextest is the runner of record.** Never `cargo test` — CI uses `cargo nextest run`. The local check loop mirrors CI exactly and should pass before any push:

```sh
# 1. uni-db seal — MUST print nothing
rg -n -e 'use uni_db' -e '\.db\(\)' \
   crates/uniko-memory/src crates/uniko-extract/src \
   crates/uniko-cortex/src crates/uniko-pipes/src \
 | grep -vE ':[0-9]+:[[:space:]]*//' | grep -v 'ALLOW:'

# 2. type-check the whole workspace
cargo check --workspace

# 3. lints as errors
cargo clippy --workspace -- -D warnings

# 4. formatting
cargo fmt --all --check

# 5. tests (nextest, not cargo test)
cargo nextest run --workspace

# 6. license + advisory gate
cargo deny check
```

Tooling installs: `cargo install cargo-nextest --locked`, `cargo install cargo-deny --locked`. CI runs on `ubuntu-latest` (`.github/workflows/ci.yml`) with a separate `cargo deny` job. Per the global testing policy, run tests in parallel; nextest parallelizes by default.

**The API surface is a tested contract.** `uniko-api` is logic-free by design: `lib.rs` is a wildcard `pub use uniko_cortex::*` and `tools.rs` is a curated explicit re-export of ~55 intent types from `uniko-memory`. The negative surface (what must stay hidden — `KnowledgeBase`, `PipelineSystem`, `IngestMessage`, etc.) is enforced by nine `compile_fail` doctests in `tools.rs`; the positive surface (~45 facade types must remain reachable) is enforced by `assert_exists::<T>()` in `tests/surface.rs`. These break loudly if the re-export list drifts — treat them as the API definition.

**uni-db is a separate project.** Never edit `../uni/`. On a suspected uni-db bug, build a minimal isolated repro (the pattern is `crates/uniko-store/tests/unidb_bytes_return_repro.rs`) and file it upstream against `rustic-ai/uni-db` rather than working around it in uniko.

The Python bindings dev loop:

```sh
cd bindings/uniko-py
uv run maturin develop
uv run pytest python/tests/ -n auto
```

### 10.5 Release process and versioning

One git tag ships **two artifact families** via **OIDC Trusted Publishing** (no API tokens in the repo): six Rust crates → crates.io, and Python wheels → PyPI.

**Version single-sourcing.** Crate versions use `version.workspace = true`; the wheel uses `dynamic = ["version"]`; the runtime `uniko.__version__` comes from `env!("CARGO_PKG_VERSION")`. A `check_version_sync` CI guard fails on drift. Cut a release with:

```sh
cargo set-version --workspace 0.1.0   # single-sources every version
# land on main, green CI
git tag v0.1.0 && git push origin v0.1.0
# then approve the `release` environment in the Actions UI
```

The workflow runs validation and builds **unconditionally**, then **pauses** the publish jobs on the `release` environment (required-reviewers gate). Jobs, in order: `guard` (tag == workspace version) → `validate-crates` → `build-wheels` (Linux x86_64 + aarch64, macOS arm64, Windows x64) → `build-sdist` → gated `publish-crates` / `publish-pypi` / `github-release`.

**Publish order is fixed by the dependency graph** and cannot be parallelized:

```
uniko-store → uniko-pipes → uniko-extract → uniko-cortex → uniko-memory → uniko-api
```

Release-runbook gotchas that will bite an unprepared release engineer:

- **Dry-run only verifies the leaf crate.** `cargo publish --dry-run` fully verifies only `uniko-store`; the other five use `cargo package --no-verify` because their packaged form needs `uniko-store 0.1.0` already on crates.io. Full-chain verification happens only at the real ordered publish.
- **Publish from a clean clone.** `git clone --local . /tmp/…` first — libgit2's status walk fails over the gitignored `data/`, `target/`, `.venv` (millions of files), and `--allow-dirty` does **not** help.
- **Brand-new crate names** need one manual `cargo publish` before Trusted Publishing can be configured.
- **PyPI publishing is DISABLED** until per-project 100 MB file-size increases land — all three wheels exceed the default limit (the base wheel bundles ~113 MiB of ONNX Runtime; GPU wheels add candle+mistralrs kernels). It is gated behind a repo **variable** `PYPI_PUBLISH_ENABLED` (must be a variable, not env). Until then, `github-release` attaches the wheels (2 GB/asset limit) as the interim distribution channel.
- **Pre-release rehearsal**: a `v0.1.0-rc.1` suffix is accepted; the guard compares the base version only.

Pre-release suffixes and `ort` binary constraints (see §10.7) mean aarch64 Linux builds natively on `ubuntu-24.04-arm` (no cross/QEMU), and macOS is aarch64-only.

### 10.6 Security posture

- **Private disclosure only.** No public issues/PRs for vulnerabilities. Preferred channel: GitHub private security advisories (`github.com/rustic-ai/uniko/security/advisories/new`); fallback `security@dragonscale.ai`. Acknowledgement target: a few business days.
- **Supported versions:** only the current `0.1.x` series.
- **Supply-chain gate:** `cargo deny check` enforces a license allow-list and security advisories per `deny.toml` in every check-loop run and in CI.
- **No stored secrets:** releases use OIDC Trusted Publishing, so there are no API tokens in the repository.

**Access-control posture inside the engine** is worth restating here because it is an operational footgun. Recall is **fail-open by default**: `ViewerScope::Unrestricted` returns policy-scoped `Fact`/`Observation` items *unfiltered* with only a `WARN`. Production callers serving a specific participant **must** set `RecallConfig.viewer` or build with `.scope_to_agent()`, or data leaks across visibility boundaries. Conversely, **unknown visibility schemes fail closed** — a typo like `secret:42` permanently admits no one. Visibility is enforced **only** for `Fact` and `Observation`; all other node types (Message, Chunk, Entity, Topic, Summary, Goal, Task, Episode…) carry no scope and pass the filter untouched. Recognized schemes: `null` / `""` / `public` (all), `private:{pid}`, `team:{tid}`, `org:{oid}`.

### 10.7 System dependencies and native linkage

The load-bearing native dependency is **ONNX Runtime**, linked statically through the `ort` crate. This drives most of the system-level requirements:

- **`protoc`** and a **C/C++ toolchain** are build-time requirements because of `ort`.
- `ort` links **only pyke prebuilt binaries per target triple**. Consequences: aarch64 Linux must build natively (`ubuntu-24.04-arm`), and macOS is **aarch64-only** — there is no `x86_64-apple-darwin` binary in the pinned `ort` release, and `onnx` is always-on.
- `tokenizers`, `ndarray`, and `half` are pulled by the ONNX stack and compile at `opt-level=3` even in dev.

The persistence and search layer is entirely in-process via **uni-db** (graph + vectors + BM25 full-text + Locy in one store) — nothing external to run or keep in sync. Blob bytes for `:ArtifactContent` default to **Lance** (inline), with local-FS content-addressed and S3-compatible (`object_store`) backends available. The chosen backend is persisted in `KnowledgeBaseStats`; **reopening a KB with a different `blob_storage.kind` is a hard error** (no implicit migration).

### 10.8 GPU builds and wheel variants

The Python SDK ships as **three interchangeable maturin wheels** that all install and `import` as `uniko` and are **mutually exclusive** (only one at a time):

| Wheel | Platform | ONNX Runtime EP | Local LLM | `uniko-api` features |
|---|---|---|---|---|
| `uniko` (base) | per-platform CPU | CPU (statically bundled, ~113 MiB) | none | `[onnx]` |
| `uniko-cuda` | Linux x86_64 | CUDA EP | mistralrs + candle CUDA kernels | `[onnx, gpu-cuda, mistralrs, candle]` |
| `uniko-metal` | macOS arm64 | CoreML | mistralrs + candle Metal kernels | `[onnx, gpu-metal, mistralrs, candle]` |

The GPU crates are **not separate source trees**: `bindings/uniko-cuda/Cargo.toml` and `bindings/uniko-metal/Cargo.toml` point `[lib] path` at `../uniko-py/src/lib.rs`, and their Python package is **copied** in from `bindings/uniko-py/python` by `scripts/bootstrap-wheel-variants.sh` (copy, not symlink — maturin does not follow symlinks). The only real difference between the three is the `uniko-api` feature set. All are `abi3` for CPython ≥ 3.10; the GPU wheels resolve host CUDA/cuDNN at runtime rather than bundling it.

Building the wheels:

```sh
# base CPU wheel
maturin build --profile dist -m bindings/uniko-py/Cargo.toml

# GPU variants: copy the python package in first, then build
scripts/bootstrap-wheel-variants.sh
CUDA_COMPUTE_CAP=80 maturin build --profile dist --auditwheel skip \
    -m bindings/uniko-cuda/Cargo.toml
maturin build --profile dist -m bindings/uniko-metal/Cargo.toml   # macOS/aarch64
```

`dist` = release + thin LTO + `codegen-units = 1`; see the profile comment in the
root `Cargo.toml` for the size/RSS trade-off and the
`CARGO_PROFILE_DIST_CODEGEN_UNITS` escape hatch for small runners.

GPU-build specifics:

- `uniko-cuda` requires the **CUDA toolkit (`nvcc`) at build time** because candle+mistralrs compile CUDA kernels. `CUDA_COMPUTE_CAP=80` targets Ampere PTX, which forward-JITs to newer archs (Ada/Hopper/Blackwell).
- `--auditwheel skip` is used so CUDA runtime libs are resolved at load time, not bundled.
- The `python/` directories in the GPU crates are gitignored and populated by `bootstrap-wheel-variants.sh`, which **must** run before `maturin build`.

A runtime trap: `LlmSpec.mistralrs()` constructs a spec in **all** wheels including base CPU, but the mistralrs provider only **compiles** in the CUDA/Metal variants. Building with a mistralrs spec on the base wheel fails at runtime, not at construction.

For running GPU benchmarks, `crates/uniko-bench/run.sh` is the launcher: `--build` compiles with `--features gpu-cuda` plus a linker shim, and the run path sets `LD_LIBRARY_PATH` to the ORT CUDA EP, cuDNN-13, and CUDA libs. A `NO_GPU=1` escape hatch forces the CPU path.

### 10.9 Two runtime workarounds you will see everywhere

Two constants appear in the Rust bindings and are load-bearing — do not "clean them up":

- **`#![recursion_limit = "512"]`** in `bindings/uniko-py/src/lib.rs` (and `256` in the extract/memory crates, `512` in some bench binaries). Proving `Send` for the deeply-nested lance/moka store futures exceeds the default recursion limit.
- **An 8 MB tokio worker stack** in the Python runtime. Recall/ingest build deeply-nested async state machines that overflow the default 2 MB stack in debug builds.

Both are required workarounds, not tuning. The Python module owns exactly one multi-threaded tokio runtime backing every awaitable and every `*_sync` `block_on`.

### 10.10 Operational shutdown and quiescence

`Uniko::shutdown()` **skips the pipeline drain** (with a `WARN`) if any `Agent`/`Session` clone still shares the `Arc<PipelineSystem>` — drop all derived handles first or in-flight work is not drained. The pipeline exposes a `quiesce()` barrier (polls the in-flight `AtomicUsize` to zero) for callers that need to wait for ingest to settle before shutdown. Graceful shutdown is phased in `ShutdownCoordinator::shutdown`: cancel ingest (5 s drain) → cancel consolidation (10 s drain) → cancel root → bounded join-or-abort within the total timeout (default `30 s`). Note the two drain sleeps are additive and independent of the join deadline, so a small `total_timeout` can be consumed entirely by the sleeps.

The Python `Uniko.shutdown()` additionally **poisons** its handle (`take()`s the inner `Option`), so any subsequent use raises `ConfigError("shut down")` — this models the consuming Rust `shutdown` that Python cannot otherwise express.

### 10.11 What is not yet shipped

For operational planning, these are explicitly **not** available in the current release and must not be depended on: an HTTP/MCP server, a CLI, production sparse/ColBERT late-interaction retrieval defaults (the machinery exists but is default-off and config-gated), multimodal ingest beyond the extension seam, rule induction, MCTS planning, and cross-agent sharing. The Python SDK is **alpha** — the API may change before 1.0. The largest known recall-quality gap is date-anchored/temporal questions.

---

## 11. Development, Contributing & Roadmap

This chapter is the working manual for anyone who builds *on* uniko or *inside* it: how the repository is laid out, the invariants CI enforces, the local check loop that mirrors CI, the sharp edges the source is littered with, the catalogue of uni-db bugs the codebase routes around, and where the project is going next. It is deliberately concrete — file paths, exact commands, and the reasons behind each rule — because most of the mistakes an engineer makes here are ones the code already anticipates and guards against.

### 11.1 Repository layout

uniko is a single Cargo workspace. The engine is six publishable crates in a strict linear dependency stack, plus one internal benchmark crate and the Python bindings.

```
uniko2/
├── crates/
│   ├── uniko-store/     L1  typed façade over uni-db: schema, search, Locy, model seam, blobs, locks
│   ├── uniko-pipes/     L2  content-free pipeline machinery: Step trait, circuit breaker, DLQ, cancel
│   ├── uniko-extract/   L3  NER, observations, chunking, atomic ingest, NLP cascade, embeddings, PDF
│   ├── uniko-cortex/    L5  P5 procedure promotion + P6 topic detection (altitude 5, sibling of extract)
│   ├── uniko-memory/    L4  the Uniko facade, 3-phase recall, P4 consolidation, workers, agent tools
│   ├── uniko-api/       —   thin re-export facade; the sanctioned public Rust surface
│   └── uniko-bench/     —   publish=false: LoCoMo/LME harnesses + write-path microbenches
├── bindings/
│   ├── uniko-py/        PyO3 async SDK (cdylib → uniko._uniko) + Pydantic overlay
│   ├── uniko-cuda/      GPU wheel variant: [lib] path → ../uniko-py/src/lib.rs, CUDA features
│   └── uniko-metal/     GPU wheel variant: same source, Metal features
├── config/             catalog.json, catalog_minilm.json, schema.json (external overrides)
├── website/            zensical docs site (uv-built)
├── scripts/            bootstrap-wheel-variants.sh and release helpers
└── AGENTS.md, CONTRIBUTING.md, DEV.md, RELEASING.md, SECURITY.md, CHANGELOG.md
```

The **dependency stack** is `uniko-store → {pipes, extract} → cortex → memory → api`, with the bindings and bench sitting on top of `uniko-api`/`uniko-memory`. Two facts about this graph trip up newcomers:

- **Layer numbers are cognitive altitude, not build order.** `uniko-memory` (L4) depends on `uniko-cortex` (L5). This is deliberate: cortex's P5/P6 sweeps are *subscribers* to memory's P4 consolidation heartbeat, so the trigger policy lives in the consolidation worker (`crates/uniko-memory/src/pipeline/consolidation_worker.rs`) and calls down into cortex. If cortex ever needed memory's runtime APIs it would become a true cycle; the documented fix is to invert via a sweep trait defined in memory and injected at the composition root (`crates/uniko-memory/src/lib.rs:15-18`), not a new dependency edge.
- **`uniko-cortex` has a path-only dev-dependency on `uniko-memory`** (for e2e tests that need `record_episode`). Path-only, on purpose, so it never becomes a published version cycle.

### 11.2 The AGENTS.md invariants

`AGENTS.md` is the contract every contributor (human or agent) works under. Four invariants are load-bearing.

#### Invariant 1 — the uni-db seal (CI-enforced)

uni-db is meant to be an implementation detail hidden behind `uniko-store`. Product crates (`uniko-memory`, `uniko-extract`, `uniko-cortex`, `uniko-pipes`) must reach the graph **only** through the typed `KnowledgeBase` API. They must never `use uni_db` and never call `kb.db()`.

This is checked by a ripgrep gate that must print nothing:

```sh
rg -n -e 'use uni_db' -e '\.db\(\)' \
   crates/uniko-memory/src crates/uniko-extract/src \
   crates/uniko-cortex/src crates/uniko-pipes/src \
 | grep -vE ':[0-9]+:[[:space:]]*//' | grep -v 'ALLOW:'
```

Rules of the gate:

- `.db()` remains a `pub` escape hatch — but only for `tests/` and the `uniko-bench` crate, which are out of scope for the seal.
- Comments are exempt (the `grep -vE` strips comment lines).
- A reviewed, intentional exception is tagged with a same-line `// ALLOW:` comment.
- The engine only re-exports the handful of uni-db types callers legitimately need: `Value`, `Transaction`, `RetryOptions`, `temporal::{Btic, TemporalValue}` (re-exported from `uni_db::common`), `xervo::{GenerationOptions, Message}`, `ModelAliasSpec`/`ModelTask`/`WarmupPolicy` (needed so a caller registering an extra model can name them — see `LlmSpec::into_alias_spec`), and `ModelRuntime` (from `uni_xervo::runtime`) (`crates/uniko-store/src/lib.rs:54`, `:65-87`).

**When you need a graph capability the façade doesn't expose, add it to `uniko-store` (a `repository/`, `operations/`, `search/`, or `*_in_tx` helper) — do not reach past the seal.** Note that `uni-xervo` NLP types (`NlpModel`, `NlpRequest`, `NlpResult`, `NlpTasks`) are a documented exception: uni-db doesn't re-export them, so `uniko-extract` depends on `uni-xervo` directly for the NLP cascade.

#### Invariant 2 — never edit `../uni/`

uni-db lives at `../uni/` as a **reference checkout only**. It is a separate crates.io project (`rustic-ai/uni-db`); the workspace pulls uni-db 3.2.0 and uni-xervo 0.17.0 from crates.io. When you hit a uni-db bug, the discipline is:

1. Build a **minimal, isolated reproduction test** in `uniko-store` (the canonical pattern is `crates/uniko-store/tests/unidb_bytes_return_repro.rs`).
2. File it upstream against `rustic-ai/uni-db`.
3. Add a *documented* workaround in `uniko-store` (see the workarounds catalogue in §11.6).

Never silently work around a uni-db bug without a repro that pins it there.

#### Invariant 3 — schema source of truth

`crates/uniko-store/src/schema/constants.rs` is authoritative for the graph model: `labels::ALL` = **25 node labels**, `edges::ALL` = **54 edge types**. The module doc in `schema/mod.rs` agrees. When in doubt, grep `constants.rs`.

`Pattern` and `CONTRADICTED_BY` are Locy-consumer additions (`episode_pattern_detector`, `contradiction_detector`) and are the two most commonly missed.

`config/schema.json` is a **snapshot**, not a second source of truth. It is generated by `cargo run --bin export-schema` and is only loaded when a caller explicitly sets `UnikoConfig::schema_path` (the default is `None`, i.e. programmatic `register_schema`). Because `schema_path` replaces the code schema wholesale, a stale snapshot silently yields a graph missing whatever was added since it was written — regenerate it in the same commit as any schema change, and check the printed label/edge counts match `constants.rs`.

#### Invariant 4 — config defaults come from `UnikoConfig::default()`

Effective runtime defaults are what the pipeline actually builds: `RecallConfig::from_uniko_config(...)`, seeded from `UnikoConfig::default()` (`crates/uniko-store/src/config.rs`, defaults around `config.rs:826-895`). The standalone `RecallConfig::default()` and `ChunkConfig::default()` **differ** and must not be quoted as runtime behavior. One trap in particular: `config.rs:636` has a stale comment saying the reranker is "disabled by default," but the `Default` impl sets `enabled: true` (ms-marco-MiniLM-L-6-v2, top_n=50). Trust the code.

Two more AGENTS.md house rules: **never `cargo test`** (nextest is the runner of record), and **never commit/push without explicit maintainer approval** — and never add AI-attribution trailers to any artifact.

### 11.3 Prerequisites and the three buildable surfaces

`DEV.md` is the consolidated build/test/run guide. There are three surfaces:

| Surface | Location | Toolchain |
|---|---|---|
| Rust workspace | `crates/*` | cargo, nextest |
| Python bindings | `bindings/uniko-py` | maturin + uv |
| Docs site | `website` | uv + zensical |

**Prerequisites:**

- Rust ≥ 1.91, edition 2024 (pinned via `rust-version` in the root `Cargo.toml` + `rust-toolchain.toml`, stable channel).
- A C/C++ toolchain and **`protobuf-compiler` (`protoc`) on `PATH`** — the stack statically links ONNX Runtime via `ort`.
- `uv` for Python and docs.
- No tokens, private repos, or credentials: uni-db and uni-xervo come from crates.io.

Expect a **slow first build** — `ort`, `tokenizers`, and `half` compile at `opt-level=3` even in the dev profile.

### 11.4 Contribution flow and the local check loop

The check loop mirrors CI (`.github/workflows/ci.yml`, ubuntu-latest) step for step. Run all six before pushing:

```sh
# 1. uni-db seal — MUST print nothing
rg -n -e 'use uni_db' -e '\.db\(\)' crates/uniko-memory/src crates/uniko-extract/src \
  crates/uniko-cortex/src crates/uniko-pipes/src | grep -vE ':[0-9]+:[[:space:]]*//' | grep -v 'ALLOW:'

# 2-6. build / lint / format / test / supply-chain
cargo check   --workspace
cargo clippy  --workspace -- -D warnings
cargo fmt     --all --check
cargo nextest run --workspace         # nextest, never `cargo test`
cargo deny    check                    # license allow-list + advisories, per deny.toml
```

Tool installs: `cargo install cargo-nextest --locked`, `cargo install cargo-deny --locked`. `cargo deny` runs as its own CI job.

Per the global project convention, run tests in parallel — `cargo nextest run` is already parallel by default.

**Python bindings dev loop:**

```sh
cd bindings/uniko-py
uv run maturin develop
uv run pytest python/tests/ -n auto
```

**Docs:** `cd website; uv sync; uv run zensical serve`.

**Branching / PR conventions** (`CONTRIBUTING.md`): standard feature-branch → PR → review flow, CI must be green, and commits/PRs carry no AI-attribution. When a commit resolves a GitHub issue, include a closing keyword (`Fixes #NNN`) on its own line.

### 11.5 The release process

One git tag ships two artifact families via **OIDC Trusted Publishing** (no API tokens in the repo): 6 Rust crates → crates.io and Python wheels → PyPI.

**Cutting a release** (`RELEASING.md`):

```sh
cargo set-version --workspace 0.1.0   # single-sources every version
git tag v0.1.0 && git push origin v0.1.0
# then approve the `release` environment in the GitHub Actions UI (required-reviewers gate)
```

Version is single-sourced: crate versions via `version.workspace = true`, the wheel via `dynamic = ["version"]`, and the runtime `uniko.__version__` via `env!("CARGO_PKG_VERSION")`. A `check_version_sync` CI guard fails on drift.

**Workflow jobs:** `guard` (tag == workspace version) → `validate-crates` → `build-wheels` (Linux x86_64 + aarch64, macOS arm64, Windows x64) → `build-sdist` → **gated** `publish-crates` / `publish-pypi` / `github-release`. Validation and builds run unconditionally; the publish jobs pause on the `release` environment until a reviewer approves.

**Publish order is fixed by the dependency graph:** `uniko-store → uniko-pipes → uniko-extract → uniko-cortex → uniko-memory → uniko-api`.

Release-time gotchas worth internalizing:

- **Publish from a clean clone** (`git clone --local . /tmp/rel`). libgit2's status walk fails over the gitignored `data/`, `target/`, and `.venv` millions-of-files dirs, and `--allow-dirty` does **not** help.
- **Dry-run only fully verifies the leaf crate** (`uniko-store`). The other five use `cargo package --no-verify` because their packaged form needs `uniko-store 0.1.0` already on crates.io. Full-chain verification only happens at the real, ordered publish.
- **Brand-new crate names need a one-time manual `cargo publish`** before trusted publishing can be configured.
- **PyPI publishing is currently DISABLED**, gated on the repo *variable* `PYPI_PUBLISH_ENABLED` (unset). All three wheels exceed PyPI's 100 MB default file-size limit (the base wheel bundles ~113 MiB of ONNX Runtime; GPU wheels add candle+mistralrs kernels), so per-project size increases must land first. Until then, `github-release` attaches the wheels (2 GB/asset limit) as the interim channel.
- **`ort` links only pyke prebuilt binaries per target triple.** aarch64 Linux builds *natively* on `ubuntu-24.04-arm` (don't cross-compile/QEMU); macOS is aarch64-only (no x86_64-apple-darwin binary in the ort rc, and `onnx` is always-on).

**The three wheel variants** (`uniko`, `uniko-cuda`, `uniko-metal`) are not separate source trees. `uniko-cuda` and `uniko-metal` point their `[lib] path` at `../uniko-py/src/lib.rs` and differ only in the Cargo feature set forwarded to `uniko-api`. Their Python packages are **copied** (not symlinked — maturin won't follow symlinks) from `bindings/uniko-py/python` by `scripts/bootstrap-wheel-variants.sh`, which must run before `maturin build`. All three set `module-name = uniko._uniko`, so they all `import uniko` and are **mutually exclusive** — only one can be installed at a time.

```sh
# base CPU wheel
maturin build --profile dist -m bindings/uniko-py/Cargo.toml
# GPU variants
scripts/bootstrap-wheel-variants.sh
CUDA_COMPUTE_CAP=80 maturin build --profile dist --auditwheel skip -m bindings/uniko-cuda/Cargo.toml
maturin build --profile dist -m bindings/uniko-metal/Cargo.toml   # macOS/aarch64
```

`uniko-cuda` requires the CUDA toolkit (`nvcc`) at build time; `CUDA_COMPUTE_CAP=80` (Ampere) forward-JITs to newer archs, and `--auditwheel skip` means CUDA runtime libs resolve at load time rather than being bundled.

**Security** (`SECURITY.md`): private disclosure only — GitHub private security advisories or `security@dragonscale.ai`, never public issues. Only the current `0.1.x` series is supported.

### 11.6 The uni-db workarounds catalogue

Because uniko sits on a live, evolving embedded engine, `uniko-store` carries a set of documented workarounds for uni-db bugs. Each has (or should have) an isolated repro (the repro tests live alongside the code in `crates/uniko-store`). Note: there is **no** `docs/bugs/UNI_DB_WORKAROUNDS.md` in the current tree — the working set below is documented inline at each workaround site. The table is the set an engineer will actually encounter.

| Bug / limitation | Symptom | Workaround (site) |
|---|---|---|
| **SSI insert-phantoms** | An empty `MATCH` registers no read-set, so two concurrent check-then-create callers on a non-unique index both read "absent" and both `CREATE` a duplicate. | `StripedLocks` (256 tokio mutexes) held across the existence re-read *and* the commit; canonical per-family lock keys (`locks.rs`, `storage/mod.rs`). |
| **Bytes via `RETURN`** | `DataType::Bytes` can't be read back through a Cypher `RETURN` (arrow LargeBinary type ambiguity). | Read blob bytes out-of-graph via `Data.artifact_bytes` / `fetch_blob`; repro at `tests/unidb_bytes_return_repro.rs`. |
| **BTIC lossy under `RETURN n`** | Node-wrapped projection stringifies Temporals lossily. | Always read BTIC columns with a **bare projection** (`RETURN f.valid_at`), never `RETURN n` (`facts.rs`, `deletion.rs`, `recall.rs`). |
| **#145 FOLD-alias returns 0.0** | A value-aggregate `FOLD` that is `YIELD`-renamed returns 0.0. | Compute `mean_importance` / `mean_episode_importance` in Rust; only trust `COUNT` in Locy FOLDs (`execution.rs`, `stdlib.rs`, cortex `procedures.rs`). |
| **RC12 — rule invocation** | Handing a bare rule name to `execute_rule` fails to parse; `$param` in a post-FOLD `HAVING` doesn't resolve. | Invoke registered rules by name via `query_rule` (builds `QUERY <name> RETURN ...`); push threshold filters into the Rust consumer (`upsert_procedure` applies `promote_threshold`, not the rule). |
| **`max(DateTime)` aggregate** | Serialized as LargeBinary, breaks readback. | `close_inactive_sessions` avoids the Cypher aggregate. |
| **#130 — catalog-open rejects non-Embed vector aliases** | A bge-m3 hybrid KB (sparse/ColBERT aliases) won't reopen via the catalog path. | Reopen only via the prebuilt-runtime path (`build_shared_runtime` + `open_with_runtime`). |
| **Bulk API doesn't return EIDs** | `bulk_insert_vertices/edges` skip the Cypher executor (~980× faster/row) but don't surface allocated EIDs or re-validate property names. | Callers validate keys up front; the `return_ids` arm falls back to UNWIND-Cypher (`batch.rs`, `edges.rs`). |
| **Multi-target edge label-filter leak** | Same-type edges to different labels can leak the wrong target under load. | Match by endpoint id, not label. |
| **`datetime_value` vs string coercion** | Writing DateTime as a string coerces poorly. | Prefer `datetime_value` for DateTime writes. |

Two general rules govern this catalogue: **don't work around a uni-db bug without a repro**, and if a workaround exists, cite it in the code so the next engineer knows it's load-bearing.

### 11.7 Gotchas every engineer must know

These are the sharp edges spread across the layers. They fall into a few families.

#### Concurrency and transactions

- **Hold the right `StripedLocks` key across the whole RMW.** `entity_id`, `content_id`, `session_id`, and `fact_id` are all **non-unique** in uni-db. Every row family has exactly one canonical lock-key builder; a divergent namespace silently breaks serialization (this is the class behind "issue #1"). Acquire external guards **outside** `transact_with_retry` so a single writer per key holds across all retry attempts.
- **`StripedLocks` are non-reentrant.** Acquiring the same stripe twice self-deadlocks. Use `lock_many` (dedups by *stripe index*, not key bytes) for multi-key acquisition — a hand-rolled per-key loop once deadlocked `batch_upsert_facts` past ~50 facts.
- **`transact_with_retry` re-runs the closure per attempt**, so any consumed input must survive re-use (capture by `Copy` or re-clone inside). In atomic ingest, `session_ctx` must **not** be mutated until *after* a successful commit, so each retry re-seeds `prepare_observations` from the same `sentence_ctx`; `entity_prep` is cloned per attempt.
- **Entity identity is `entity_id(name, canonical_type)`.** All NER writers must pass a type from the one shared lowercase vocabulary (`person`/`organization`/`location`/…). Mixing vocabularies duplicates entities. `text::normalize_canonical` is the single name-key normalizer.

#### Hybrid embedding and search

- **A hybrid embedder needs two aliases:** `embed/default` (dense, `ModelTask::Embed`) and `embed/hybrid` (`ModelTask::EmbedHybrid`), both backed by the same model. The hybrid model implements only `HybridEmbeddingModel` and can't back lone-dense columns, so it loads twice (~2× VRAM). Sparse/ColBERT columns exist only when `config.embedding.{sparse,multivector}_dimensions` is `Some` (bge-m3 today); `config.validate()` rejects `recall_sparse_enabled` or a `colbert` reranker style without them.
- **Embedding dimension is fixed at DB creation** (part of the on-disk vector index). Switching embedders (e.g. to a 1024-d model) requires a fresh KB; `validate()` catches zero dims but **not** a model/dimension mismatch. `config/catalog.json` and `catalog_minilm.json` differ only in the embed-alias model — a mismatched reopen corrupts the vector index.
- **Consolidation embed batching is chunked at 64** (`EMBED_BATCH_CHUNK_SIZE`) to avoid the ORT BFC arena OOM (~1.3 GB at ~6k inputs). `embed_batch_chunked` exists for the same reason. Don't remove it.

#### Recall behavior

- **Recall is fail-open by default.** `ViewerScope::Unrestricted` returns policy-scoped `Fact`/`Observation` items *unfiltered* with only a `WARN`. Production callers serving a specific participant **must** set `RecallConfig.viewer` or build with `.scope_to_agent()`, or data leaks across visibility. Unknown visibility schemes fail *closed* (a typo like `secret:42` locks a claim down permanently).
- **Visibility is enforced only for `Fact` and `Observation`.** All structural nodes (`Message`, `Chunk`, `Entity`, `Topic`, `Summary`, `Goal`, `Task`, `Episode`) carry no scope and pass `filter_bundle` untouched.
- **`reference_ts` is a per-query anchor, never a KB setting.** Recalling a historical corpus without setting it computes the temporal window around *now*, which never overlaps old data and silently disables the Phase-2 temporal channel.
- **Multi-query variant reformulation is opt-in** (empty `query_variants` = keywords-only). Enabling all four measured **−2.1pt** evidence and **3× latency** on LoCoMo. Don't turn it on expecting a win.
- **`answer_type_boost` defaults to 1.0 (no-op)** deliberately — the naive rule measured −0.149 R@5 / −0.186 NDCG@5.
- **Leftover debug prints ship in the hot path.** `recall/mod.rs` has three `eprintln!("RECALL_PROF ...")` statements (phase2 gate, sparse source, phase2 source) and `recall.rs:857` has a `RECALL_PROF` line. They write to stderr on every recall and should be `tracing::debug!` or removed.

#### Pipeline semantics

- **There is no retry module.** Despite the crate description, `uniko-pipes` has no `retry.rs`. The only automatic recovery is the circuit breaker's Open→HalfOpen probe. DLQ retry is data-only: `retry_count` is always written `0` and never read.
- **The Step runner lives downstream.** `run_step_chain` is in `uniko-memory`'s `ingest_worker.rs`, not in `uniko-pipes`. `uniko-pipes` defines only the vocabulary.
- **`Err` from `Step::execute` is coerced to `DeadLetter`** — a transport error can't declare its own policy, so it never `Abort`s. Only an explicit `StepOutcome::Failed{policy: Abort}` stops the chain. `Skip` and `DeadLetter` both *continue* the remaining steps on a partially-populated context.
- **Cancellation is cooperative and checked only between steps** — a long-running LLM call runs to completion after cancel unless it observes `ctx.cancel` itself.
- **`Uniko::shutdown` skips the pipeline drain** (with a `WARN`) if any `Agent`/`Session` clone still shares the `Arc<PipelineSystem>`. Drop all handles first.
- **Streamed turns (`submit`/`submit_source`) do not advance the session's cross-turn context and are not session-linked.** Only `observe()` preserves conversational fidelity. `await flush()` before a recall that must see streamed turns, and use one path per session, not both.

#### Consolidation, cortex, and rules

- **`valid_at` is one atomic `Btic`**, not a `valid_from`/`valid_until` pair. Recall uses `btic_overlaps`, not DateTime range joins.
- **`Entity.unstable` is windowed** (>4 invalidations in the last 30 days), gated on the window, not the lifetime `invalidation_count`.
- **`run_active_rules` must pass the union of all stdlib rules' params on every call** — uni-db evaluates all registered rules together, so an unresolved `$decay_rate` would fail every other rule.
- **Locy ≠ Cypher.** A second `MATCH` clause is a parse error (use one comma-joined `MATCH`); aggregate columns are `expr AS name` (no `VALUE` keyword); `$param` in a post-FOLD `HAVING` doesn't resolve (RC12). The `sequence_detector` rule string is authored around all three.
- **`stable_procedure_id` feeds the hasher `(agent, a, b)` with NUL separators** — order-sensitive, so `a→b` ≠ `b→a`. Don't "simplify" the ordering.
- **Topic ids are membership-derived.** If a community gains/loses one entity its `topic_id` changes and a *new* Topic node is created; old Topics aren't garbage-collected.
- **`record_procedure_use` never promotes candidates** — a `CANDIDATE` only reaches `ACTIVE` via the sequence-count path in `upsert_procedure`.

#### Python bindings

- **`Value::Bytes` can't round-trip through a Cypher `RETURN`** — binary payloads must go through `Data.artifact_bytes`.
- **`py_to_value` tests `bool` before `int`** (Python `bool` subclasses `int`); reordering silently coerces `True`/`False` to `1`/`0`.
- **`recursion_limit = 512` and an 8 MB tokio worker stack** are required workarounds for deeply-nested store futures (Send-proof recursion and debug-build stack overflow). Don't lower them.
- **`LlmSpec.mistralrs()` constructs a spec in all wheels** including base CPU, but the mistralrs provider only *compiles* in the cuda/metal variants — a mistralrs spec on the base wheel fails at *runtime*, not construction.
- **The negative-surface guards are the API contract.** `uniko-api` is logic-free; the 9 `compile_fail` doctests in `tools.rs` and the `assert_exists` test in `tests/surface.rs` break if the re-export list drifts. Never add code to `uniko-api`.

### 11.8 Known limitations and TODOs

Distilled from the code, `CHANGELOG.md`, and the module docs:

- **Not yet shipped** (Phases 4–6 — do not depend on these): HTTP/MCP server, CLI, rule induction, MCTS planning, cross-agent sharing. The Python SDK is **alpha** and the API may change before 1.0.
- **Temporal recall is the largest failure category.** Date-anchored questions dominate single-hop misses; the fix is `WHERE` filtering on `Session.started_at`/BTIC windows, pending as retrieval tuning continues.
- **`ensure_session_and_sender` still commits separately** from the atomic ingest tx — the module doc flags folding it into the single tx as a follow-up.
- **The CLS gate degrades under xervo.** xervo's `NlpModel` returns only the top act + confidence (no full softmax), so `cls_probs` is always empty and `cls_gate_admits` uses a top-1 reconstruction rather than the intended multi-label distribution.
- **Observation chunk ABOUT edges** were missing entity links for sessions 1–4 in one investigation — under investigation, depends on `session_observation_entity_ids` returning rows.
- **`bump_modality_presence` round-trips the whole map under a single lock** because uni-db Map props are atomic-per-column (no per-key `SET`).
- **`PdfInput` is mirrored, not re-exported,** in `uniko-pipes` (pipes is upstream of extract) — a duplicate-type-by-design that can drift.
- **Sparse/ColBERT hybrid retrieval is implemented but default-off** (§11.9), requiring a KB ingested with a hybrid embedder.

### 11.9 Roadmap themes

Three larger initiatives shape where uniko is headed.

#### Sparse + ColBERT hybrid retrieval

The plumbing is landed and config-gated default-off. With a bge-m3 hybrid embedder, `Chunk`/`Observation` nodes gain a `sparse_embedding` (`SparseVector`) column and a `colbert_embedding` (`List<Vector>`, per-token) column *in addition to* the dense vector — all three pointing at the same `embed/hybrid` alias and source property so uni-db fuses them into one `EmbedHybrid` forward pass. ColBERT vectors use `VectorAlgo::Flat` because per-token vectors feed only the MaxSim rerank, never first-stage retrieval. Recall wires a sparse channel (`uni.sparse.query`, filtered at the index level via `_vid IN (...)`) and an in-process ColBERT MaxSim rerank rescaled into the top-window score band. The next step is turning these on by default once the accuracy/latency trade validates on the benchmark arms (the bench already ships `arm0/A/B/C` sparse+colbert sweep configs).

#### LTM masterbook — the four-pillar proposal

A research program to deepen long-term memory, captured in the LTM masterbook plan. Its firm architectural constraint: **ingest stays LLM-free; any LLM work is offline-only.** The consolidation P4→cortex P5/P6 subscription is the seam it builds on, and Pillar 1 (M0) is the first deliverable. The design keeps the "compile once, query forever" discipline — write-time enrichment remains deterministic CPU work (regex + tree-sitter + the INT8 DeBERTa cascade).

#### PDF / document-VLM ingest

The tiered PDF path (`uniko-extract/src/ingest/pdf/`, feature `pdf-ocr`) already materializes a `:Page`/`:Block` document-IR graph with `HAS_PAGE`/`CONTAINS`/`NEXT_IN_READING_ORDER` edges and bbox/confidence/provenance, using uni-xervo-pdf with the VLM tier disabled and a `Ceiling(Ocr)` policy. Child `:Chunk` nodes are dual-attached to both `Block` and `Artifact` (so mean-pool embeddings and artifact recall both keep working). The roadmap direction, per the mid-2026 landscape assessment: specialized sub-2B doc-VLMs beat frontier LLMs at document extraction, but hallucination is the unsolved risk and a pure-Rust rasterizer is the blocker. uniko's structural edge — **document-IR as a graph** — is what differentiates it from flat OCR pipelines. The `pdf-extract` legacy path is wrapped in `catch_unwind` because it has ~50 known adversarial-input panics; PDF extraction failure never propagates as `Err` (the artifact + blob still persist and the failure lands in `PdfIngestResult.extraction_failure` so a caller can re-extract with a different backend later).

### 11.10 Where to measure — uniko-bench

All performance and accuracy claims trace to `crates/uniko-bench` (`publish = false`). It hosts two full harnesses (`uniko-bench` for LoCoMo, `longmemeval-bench` for LME), a Cypher console (`uniko-cypher`), an NLP compare/parity pair, and the write-path microbenches (`insert-microbench`, `update-microbench`, `mutation-set-microbench`, `bulk-vs-unwind`, `profile-writes`) used to isolate uni-db and NLP regressions as minimal upstream repros. Every model/device/recall/cost knob lives in one `--bench-config <path>.json`; the legacy flat-flag surface was retired and `reject_retired_flags` emits migration hints.

Benchmark numbers of record (LoCoMo10, gemini-3.1 judge, Mem0's verbatim judge prompt, 2026-05-26, on 22-core CPU + 8 GB GPU): LLM-judge **0.8117**, retrieval hit **0.8555**, F1 **0.321**, ingest 5,882 turns in **7.5 min at $0** (~76 ms/turn), mean Q&A latency **4.04 s**. Two caveats travel with these figures and must not be conflated: the 0.8117 judge score is the full **1,986-q** set, while the KTH cost/latency tables use the **1,540-q non-adversarial** subset; and these are self-measured internal-harness numbers, not a third-party leaderboard. Adversarial questions are excluded from the LLM judge and scored only by a negation-phrase whitelist.

Two bench-specific gotchas worth carrying into any perf investigation: **a shared `ModelRuntime` is mandatory for concurrency** (per-KB ONNX sessions OOM an 8 GB GPU at question-concurrency ≥ 3, and the prebuilt-runtime open path is the only way a bge-m3 hybrid KB reopens — uni-db #130); and **LoCoMo ingests turns sequentially while LME ingests sessions concurrently**, so per-turn milliseconds are not comparable across the two harnesses — always re-measure per config rather than quoting a single "ingest is N ms/turn" number.

---

## Appendix A: Glossary

**ABDUCE / abduction** — A Locy operation that decodes `CommandResult::Abduce` into an `AbductionResult` holding ranked `AbducedModification`s, each with a cost (RemoveEdge 1.0 / ChangeProperty 0.5 / AddEdge 1.5). Surfaced as `KnowledgeBase::abduce` and `Agent::abduce`.

**Action** — Episodic node recording a concrete agent operation (tool call). Written by `record_action` (F17–F20); output that overflows a token threshold spills to an Artifact via a PRODUCED edge.

**admission (entity)** — The `admit_entities` policy that filters raw NER candidates: drops Date/Measurement/Preference/QuotedString, gates `Other` by confidence, drops greeting-fragment Persons. Gated by `entity_strict_admission` (default true).

**Artifact / ArtifactContent** — Node for an ingested file/blob and its content-addressed bytes. ArtifactContent bytes live inline in Lance (default) or out-of-graph in Fs/S3 backends; deduped by SHA-256 hash.

**ASSUME** — Locy hypothetical reasoning: `ASSUME { } THEN { }` forks the graph, runs a query, and rolls back without mutating. Exposed via `AssumeBuilder` (`.then_query().param().run()`).

**BTIC (bitemporal interval)** — uni-db's native `DataType::Btic`: a half-open `[lo,hi)` valid-time interval with per-bound granularity and certainty. Facts carry it in `valid_at`; open interval uses `POS_INF` hi. Must be read via a bare column projection (`RETURN f.valid_at`), never `RETURN n`.

**Block** — Document-IR node inside a PDF Page (kind/text/reading_order/bbox/confidence), chained by NEXT_IN_READING_ORDER; produced by the tiered OCR path.

**catalog** — External `config/catalog.json` mapping model aliases (embed/default, nlp/default, rerank, etc.) to concrete xervo model specs; built into a runtime via `build_catalog_specs` / `embed_catalog`.

**Chunk** — Retrieval unit split from Message/Artifact/Session content; carries a dense embedding plus optional sparse/ColBERT columns when a hybrid embedder is used.

**circuit breaker** — Lock-free 3-state (Closed/Open/HalfOpen) breaker in uniko-pipes that wraps LLM calls; trips Closed→Open at a consecutive-failure threshold and probes Open→HalfOpen after a wall-clock recovery interval.

**ColBERT / MaxSim** — Late-interaction reranking over per-token `colbert_embedding` vectors; query multivectors are scored by MaxSim and rescaled into the top-window band. Requires a hybrid (BGE-M3) embedder.

**consolidation (P4)** — The heartbeat cycle that derives Facts from accumulated Observations: group by (subject,predicate), cosine-cluster object surface forms (threshold 0.88), mode-vote a canonical object, upsert bitemporal Facts, wire SUPPORTED_BY, run F38 contradiction + F39 drift, and record a ConsolidationCycle audit node.

**ConsolidationCycle** — Meta/audit node recording one P4 sweep, with PROCESSED/CREATED/REINFORCED/INVALIDATED edges; PROCESSED is the idempotency anchor.

**CONTRADICTED_BY** — Source-only edge (Fact→Episode) drawn by the contradiction_detector rule; present in `constants.rs` but omitted from the docs.

**contradiction (F38)** — Within a (subject,predicate) group, if the fraction of votes outside the prior Fact's cluster exceeds 0.40, close the prior open-BTIC Fact and wire INVALIDATES.

**coverage gate** — Per-phase recall exit test: coverage = 0.4·facet_coverage + 0.3·mean_score + 0.3·diversity; Phase 1 gate 0.75, Phase 2 gate 0.65, each also requiring ≥3 items.

**DLQ (DeadLetterQueue)** — uniko-pipes wrapper that persists an unrecoverably-failed step as a standalone `DeadLetter` node (step/error/node_ref/retry_count=0/max_retries). Retry/list/clear surfaces are intentionally not implemented — `retry_count` is written 0 and never read.

**drift (F39, entity)** — Cumulative INVALIDATES edges on an Entity within a rolling 30-day window; exceeding DRIFT_THRESHOLD (4) flips `Entity.unstable=true`, which recall reads to force Phase 2+.

**drift override (F58)** — Recall forces past cheap Phase 1 into Phase 2+ when a query entity resolves to an `unstable` Entity.

**Entity** — Semantic node for a named entity, identified by canonical `entity_id = 'ent_' + hex64(SHA256(lower(name) \0 canonical_type))`; carries frequency/confidence and the drift fields. `entity_id` is non-unique in uni-db, so writers must hold a StripedLocks key across check-then-create.

**Episode** — Episodic node recording subjective agent experience (action_type/outcome/importance); chained by FOLLOWED_BY within a 1-hour window and RECORDED_BY the agent Participant. Feeds procedure promotion.

**Fact** — Semantic (subject,predicate,object) triple consolidated from Observations; carries `confidence`, `observation_count`, bitemporal `valid_at`, and provenance edges (SUPPORTED_BY, DERIVED_BY/FROM, INVALIDATES). Never deleted — invalidated by closing its BTIC hi bound.

**Goal / Task** — Working-memory nodes; working memory is recomputed on demand by traversing Goal→Task→Session→Messages/Facts/Entities (no stored WorkingMemory node). Phases are derived from free-form status strings.

**hybrid search** — RRF fusion of vector + BM25 (+ optional sparse/ColBERT) channels, with per-`SearchTarget` Tier weighting.

**kniv-deberta** — The default NLP model (`dragonscale-ai/kniv-deberta-nlp-base-en-xsmall`, INT8): a small shared-encoder DeBERTa whose single forward pass yields POS/NER/DEP/SRL/CLS.

**Laplace confidence** — Fact confidence smoothing `(n+1)/(n+2)`; n=0→0.5, asymptotes to 1.0.

**Locy** — uni-db's embedded logic-programming layer (rules / ASSUME / ABDUCE / FOLD). Not Cypher: single comma-joined MATCH, `expr AS name` aggregates, no `$param` in post-FOLD HAVING (RC12).

**Message** — The atomic episodic unit; anchors provenance via SENT_BY/IN_SESSION/NEXT/MENTIONS edges. With Action, one of the only two directly-observed node types.

**Modality** — Coarse routing enum (Text/Code/Markup/Structured/Document/Pdf/Image/Audio/Video) shared by both ingest and recall; unknown binary routes to Text.

**MMR** — Maximal Marginal Relevance dedup in Phase 2 recall (Jaccard token overlap; lambda 0.7, hard-duplicate threshold 0.85).

**Observation** — Semantic node carrying a reconstructed, speaker-attributed triple (subject/predicate/object + temporal_anchor); linked OBSERVED_IN a Message and ABOUT its entities. The raw material P4 consolidates into Facts.

**ONNX cascade** — The write-time, LLM-free NLP pipeline: one shared-encoder ONNX forward pass (via xervo) producing all POS/NER/DEP/SRL/CLS heads, adapted into uniko types by `xervo_to_uniko`.

**Page** — Document-IR node for one PDF page (plain_markdown/block_count); HAS_PAGE from Artifact, CONTAINS its Blocks.

**Participant** — Node for a communication participant (user/assistant/system/tool); resolves team/org memberships for the Viewer policy.

**Pattern** — Node label added by the episode_pattern_detector Locy consumer; present in `constants.rs` but undocumented.

**Procedure** — Procedural node promoted from recurring successful (action_a→action_b) Episode pairs (P5); lifecycle candidate→active→deprecated via `promote_threshold` and effectiveness hysteresis (demote <0.4, repromote ≥0.6).

**PPR (Personalized PageRank)** — Rust-side weighted power-iteration spreading activation from entity seeds over the memory graph (damping 0.85, max 30 iters), used as a Phase 2 recall channel.

**recall cascade** — The 3-phase recall pipeline: Phase 1 Compact (consolidated Semantic/Procedural tier), Phase 2 Expand (episodic RRF+MMR + temporal/graph/cross-modal channels), Phase 3 Broaden (per-variant hybrid fan-out + rerank), each coverage-gated with early exit.

**RRF (Reciprocal Rank Fusion)** — Rank-fusion scoring `Σ weight_i/(k + rank_i)`, k=60, used to merge search channels and query variants.

**rule lifecycle** — Confidence-driven state machine for authored/induced Locy rules: candidate→active→demoted→pruned; confidence·=0.95 per missed cycle, +0.05 on match, demote <0.40 / repromote ≥0.60, prune after 90 days. Stdlib rules are exempt.

**sequence_detector** — The canonical stdlib Locy rule (defined once in uniko-cortex) that counts FOLLOWED_BY-adjacent success Episode pairs per agent; drives P5 procedure promotion. Threshold is applied in Rust, not in the rule.

**Session** — Node grouping Messages of one conversation; also the facade handle whose `observe()` durably ingests turns and advances cross-turn context.

**SSI (Serializable Snapshot Isolation)** — uni-db's transaction isolation. Catches lost updates (second committer aborts retriably → `UnikoError::Conflict`, retried by `transact_with_retry`) but NOT insert-phantoms — hence StripedLocks for check-then-create.

**Step trait** — The async composition point (`name`/`should_run`/`execute` → `StepOutcome`) declared in uniko-pipes and implemented by uniko-extract steps; the runner (`run_step_chain`) lives downstream in uniko-memory.

**StripedLocks** — 256 tokio async mutexes keyed by canonical byte-prefixed keys (entity/content/session/participant/fact/node) serializing RMW/check-then-create; `lock_many` dedups by stripe index for deadlock-free multi-key acquisition.

**Summary** — Semantic node holding a session/goal summary (extractive by default, optional LLM-abstractive behind the `llm` feature).

**Topic** — Thematic node clustering co-occurring Entities via weighted Label Propagation (P6); membership-derived `topic_id` changes when membership changes.

**Tier weighting** — Recall score scaling by node class: Semantic (Fact/Topic) 1.0, Procedural 0.9, Episodic 0.7, KnowledgeBase (Chunk/Obs) 0.5, Provenance (Message/Action) 0.4.

**uni-db** — The embedded multi-model engine (OpenCypher + Locy + vector/FTS/sparse indexes + BTIC + SSI + bulk insert) that stores the entire memory graph. uniko-store is its sole boundary; product crates may not `use uni_db` or call `.db()` (CI-enforced seal).

**uni-xervo** — The model runtime (`ModelRuntime`) providing ONNX embedding/NLP/rerank/OCR plus remote/mistralrs/candle LLM providers; reached through the `model.rs` seam.

**UnikoConfig** — The root runtime config (embedding/reranker/nlp/ocr + recall/consolidation/chunking/vector thresholds) with opinionated validated defaults (BGE-Small 384d, reranker on, SRL on).

**visibility** — Per-claim access-control scheme on Fact/Observation only: null/""/public, `private:{pid}`, `team:{tid}`, `org:{oid}`; unknown schemes fail closed. Enforced by `visibility_admits`/`filter_bundle` against a Viewer.

**wheel variants** — The three interchangeable maturin wheels of the Python SDK — `uniko` (CPU), `uniko-cuda` (NVIDIA), `uniko-metal` (Apple Silicon) — sharing one source tree and differing only in forwarded Cargo feature flags; all import as `uniko` and are mutually exclusive.
