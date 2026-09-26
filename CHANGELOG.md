# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking

- **`ModalityExtractor::extract` is replaced by `prepare` + `apply_in_tx`,**
  with a new defaulted `finish_post_commit`. Attachments now commit inside the
  same transaction as the turns carrying them, which requires uniko to own that
  transaction. A `tx` parameter on `extract` would not be enough: a unit must
  know every `content_id` it will merge *before* opening the transaction, so it
  can take the striped locks first, and a single call gives it no point at
  which to ask. The split also keeps decoding and model inference outside the
  transaction, where a retry does not re-pay them.

  No default body can soften this. One forwarding to the old `extract` would
  have to open its own transaction — exactly the non-atomicity this removes —
  and would silently degrade a host that believed it had migrated. Implementors
  get a hard error at the `impl` block instead.

  ```rust
  // before
  async fn extract(&self, kb: &KnowledgeBase, src: &IngestSource)
      -> Result<ArtifactIngestResult, UnikoError>
  {
      let caption = self.vlm.caption(src).await?;          // model
      let nid = kb.create_node("Artifact", &props).await?;  // commits
      Ok(ArtifactIngestResult { artifact_node_id: nid, .. })
  }

  // after
  async fn prepare(&self, kb: &KnowledgeBase, src: &IngestSource)
      -> Result<Box<dyn ModalityPrepared>, UnikoError>
  {
      let caption = self.vlm.caption(src).await?;   // model, no tx open
      let put = kb.put_blob(&hash, &bytes).await?;  // blob I/O
      Ok(Box::new(MyPrep { hash, size, caption, put }))
  }

  async fn apply_in_tx(
      &self, kb: &KnowledgeBase, tx: &Transaction,
      prep: &dyn ModalityPrepared,
      ctx: ArtifactContextNids, seen: &mut UnitArtifactSeen,
  ) -> Result<ArtifactIngestResult, UnikoError> {
      let prep = prep.as_any().downcast_ref::<MyPrep>().expect("own prep");
      let nid = kb.create_node_in_tx(tx, "Artifact", &props).await?;
      // ctx.message_nid is already resolved — do NOT look it up by id.
      Ok(ArtifactIngestResult { artifact_node_id: nid, .. })
  }
  ```

  Post-commit work — such as `Artifact.image_embedding`, which has no embedding
  config and is written through a self-committing update — moves to
  `finish_post_commit`. Python bindings are unaffected: Python-authored
  extractors were never exposed.

- **`Session::summarize` returns `SummarizeReport`** rather than
  `Option<NodeId>`; the Python binding returns `(summary_id, finalize_error)`.
  The chunk refresh it performs is still best-effort — failing summary
  generation over a transient conflict would regress existing callers — but its
  outcome is now reported instead of only logged, so a caller can tell that a
  summary was built from stale chunks.

### Added

- **Source revisions and retirement** (`rustic-ai/uniko#41`):

  ```rust
  // A newer revision of the same source replaces the previous one.
  session.ingest(
      IngestSource::text(page_v2).with_source("wiki-summit").with_revision("rev-b"),
  ).await?;

  agent.recall("summit elevation").await?;                          // rev-b only
  agent.recall_in("summit elevation",
      Scope::default().include_superseded()).await?;                // history too
  agent.retire_source("wiki-summit").await?;                        // neither
  ```

  `IngestSource::with_revision` / `Turn::revision` declare an immutable
  revision identity. Together with the existing content fingerprint it
  decides idempotency: the same revision with identical bytes is a no-op, and
  the same revision with changed bytes is rejected with `IdConflict`, because
  a revision id is a promise about the content.

  A new revision of a source stamps `superseded_at` on the previous one and
  records `SUPERSEDES` from new to old. `Agent::retire_source` stamps
  `Source.retired_at`. **Ordinary recall excludes both**, so replaced or
  retired evidence cannot ground a current answer, while
  `Scope::include_superseded` reaches it so an older result stays
  attributable to the revision that grounded it. Nothing is deleted.

  Landing this does not change anyone's results on its own: the exclusion set
  is resolved first, and when nothing has been retired or superseded it comes
  back empty and no predicate is emitted at all. Results shrink only once
  something has actually been retired or replaced.

  Retirement is recorded on the `:Source`, never on the content, so retiring
  one source cannot affect another that merely shares identical bytes — those
  share an `:ArtifactContent` row but not a `:Source`.

  `SUPERSEDES` was already declared `Rule → Rule`; uni-db permits one
  declaration per edge type, so both endpoint pairs are now declared
  together. Schema node and edge counts are unchanged.

- **Typed record provenance and pre-ranking recall filters**
  (`rustic-ai/uniko#39`):

  ```rust
  session.observe(
      Turn::new("agent", "the query returned 42 rows")
          .category("executed_result")
          .source("warehouse-prod"),
  ).await?;

  let scope = Scope::default().categories(["executed_result"]);
  let bundle = agent.recall_in("row counts", scope).await?;
  ```

  A caller can record a record class and a stable logical source without
  placing either in searchable prose. A new `:Source` node holds the logical
  identity, with `FROM_SOURCE` from the Message or Artifact; `category` and
  `source_id` are denormalised onto Message, Artifact, Chunk, Observation and
  Fact so the filter is a property predicate rather than a traversal.

  The filters resolve into the allow-set that candidate generation itself
  uses, so ranking, the result limit and the `coverage` score describe the
  permitted evidence — not items that ranked well and were discarded after
  Uniko had already scored them. A filter with no eligible matches returns
  empty rather than being padded with other categories. Every `RecallItem`
  reports its own `category` and `source_id`; `None` means "none recorded" or
  "no single traceable source", not "filtered out".

  `Episode` carries no provenance, so a provenance filter excludes it,
  matching the documented rule that a node which cannot anchor a dimension
  goes dark under a filter on it. `Artifact` and `Fact` become reachable by a
  filter for the first time: neither anchors session, participant or time, so
  the older dimensions had no arm for them.

  Schema is now **26 node types, 55 edge types**.

- **Atomic, idempotent multi-turn units** (`rustic-ai/uniko#40`):

  ```rust
  session.unit()
      .turn(Turn::new("alice", "what's the plan?").id("m-1"))
      .turn(Turn::new("bob", "ship it friday").id("m-2"))
      .commit().await?;
  ```

  Every message, its edges and chunks, the unit's merged entity upsert, each
  turn's mentions and observations, and every attachment are written in ONE
  transaction that commits once. A related pair is either wholly visible to
  later recall or wholly absent — the guarantee two separate `observe` calls
  cannot give, where an interruption leaves a question with no answer.

  Re-committing a unit whose ids are all present with identical content is a
  no-op. Reusing an id with different content, repeating an id within one unit,
  or submitting a unit that is only *partly* recorded is rejected: a correct
  caller cannot produce a partial unit, and both alternatives — ingesting only
  the absent turns, or treating the whole unit as a no-op — silently corrupt.

  `Session::observe` is now a one-turn unit, so the two entry points cannot
  drift. Two behaviour changes follow: attachments commit **with** their
  message rather than in separate transactions afterwards, and an idempotent
  replay advances the chain head so the following turn still gets its `NEXT`
  edge.

- `Agent::finalize_session` quiesces the streaming pipeline before reporting,
  so it cannot claim completion while ingest for that session is still in
  flight. The barrier is instance-wide rather than per-session: broader than
  needed, but it never waits too little.

### Fixed

- **Observation ordering was not a total order.** `session_observation_rows`
  ordered by the *message's* timestamp, which every observation from that
  message shares, so their relative order was arbitrary. The observation chunk
  surface concatenates those rows, so a reordering changed the chunk text and
  `finalize()` rebuilt an unchanged session — re-embedding every observation
  chunk, with nothing to explain why. Since `FinalizeReport::rebuilt` ORs the
  transcript and observation surfaces, this surfaced as a finalize claiming to
  have rewritten a session nobody had touched.

- **Session chunking no longer degrades failed reads into wrong data.**
  `session_chunk_rows` dropped a row whose id failed to decode, making an
  unreadable chunk surface look *absent*, and defaulted unreadable text to
  `""`, making a readable chunk look *changed*; `session_transcript_rows`
  substituted `"unknown"` for an unreadable speaker name, rewriting the
  transcript that the idempotency check compares. All three now propagate. A
  genuinely NULL speaker still reads as `"unknown"` — that case is legitimate.

- **Artifact ingest is atomic.** The node, its `HAS_CONTENT` edge, the
  provenance edges and the chunks were written in four-plus separate commits,
  so a failure between them left a committed Artifact with no chunks. Both the
  text and PDF paths now write in one transaction; for the tiered PDF path this
  also collapses what was on the order of `2*pages + 4*blocks + chunks`
  separate commits into a single one.

- **PDF provenance edges could vanish silently.** `link_pdf_context` skipped an
  unresolvable reference with no output at all — not even the debug line the
  artifact path emits — wrote `ATTACHED_TO` without `attached_at`, and had no
  `PRODUCED` leg.

### Changed

- **uni-db 3.4.0 → 4.1.1, uni-xervo 0.17.0 → 0.18.1.** No API changes were
  needed in uniko. The upgrade retires two workarounds:
    - The baseline snapshot published on every persistent open, which existed
      because a crash inside the first-flush window left a store that could
      never be reopened (`rustic-ai/uni-db#275`). 4.0.0 replays the WAL when
      there is no L1 data to collide with, so the flush is gone and the repro
      is now a live regression guard rather than an expected failure.
    - The per-channel narrow embed aliases (see Fixed), retired by uni-xervo
      0.18.1.

  4.0.0 alone was not adoptable: re-applying an unchanged schema to a populated
  label was rejected, so no persistent store could be reopened after its first
  write (`rustic-ai/uni-db#286`). Fixed in 4.1.0; the repro stays as a guard.

  4.1.0 was not adoptable either: full-text recall deadlocked. Lance's
  `ScanScheduler` binds its I/O loop to whichever runtime is current and the
  inverted index caches that loop in the shared session, while `similar_to`
  drove its FTS lookup on a per-call runtime it then dropped — so once a query
  loaded the index, every later query for an uncached term waited on a dead
  loop. A repeated term was served from cache, which is why it presented as
  "the second *different* query hangs" (`rustic-ai/uni-db#290`). Fixed in
  4.1.1, verified here against the LoCoMo bench, which previously hung three
  questions in and now completes.

### Fixed

- **The learned-sparse and ColBERT retrieval channels never ran.** Both
  produced their query vector through the `embed/hybrid` alias, but a sparse
  query needs a `SparseEmbeddingModel` and a ColBERT query a
  `MultiVectorEmbeddingModel` — an `EmbedHybrid` alias supplies neither, so
  every query failed. The write path tolerates the same mismatch (it falls back
  to the hybrid model's matching head), so ingest populated `sparse_embedding`
  and `colbert_embedding` normally and the failure was query-only. Failures were
  swallowed at `debug` (sparse) and `warn` (ColBERT), leaving an enabled channel
  indistinguishable from an unhelpful one while still costing a query per
  variant. Fixed upstream in uni-xervo 0.18.1, which adapts a hybrid handle to
  each narrow facade (`hybrid_adapter::HybridAsSparse` / `HybridAsMultiVector`),
  so one `embed/hybrid` alias serves the document pass and both query sides.
  The interim workaround — a dedicated `EmbedSparse` / `EmbedMultiVector` alias
  per channel, which cost the sparse head its place in the single forward pass —
  is removed with the 4.1.0 bump.
- **Sparse and ColBERT recall projections asked for a column the procedures do
  not yield.** `uni.sparse.query` and `uni.vector.query` yield `vid / score /
  rerank_score`, but both call sites wrote `YIELD node, score` and then
  `labels(node)`, which fails once the call actually executes. This was
  unreachable until the alias fix above, which is why it had never surfaced.
  Both now bind the vertex explicitly from the vid. Still required on uni-query
  4.1.0 — `fts_query_yields()` is unchanged.

- **A reused `message_id` silently kept the original turn.** `observe` treated
  "this id exists" as "this is a replay" without comparing content, so a second
  call under the same id returned the first record and discarded the new turn —
  with no error. Identical content stays idempotent; a genuine content
  disagreement now raises the new non-retriable `UnikoError::IdConflict`
  (`uniko.IdConflictError` in Python), which is deliberately not the retriable
  `Conflict` the ingest retry loop spins on.

- **Identical bytes ingested under two caller-chosen ids collapsed onto one
  artifact, and the second id resolved to nothing.** `ingest_artifact` matched
  on the content hash *before* the caller's `artifact_id`, returned the first
  artifact's node while echoing back the second id, and skipped
  `link_artifact_context` — so the second session got no `ATTACHED_TO` edge and
  recalled nothing from a document it had just ingested. Deduplication was also
  store-global, so this crossed agents, not just sessions.

  A caller-supplied id is now the artifact's identity: two ids over identical
  bytes are two artifacts, sharing one stored copy of the content on
  `:ArtifactContent` where the data model always said deduplication lived, and
  reusing one id for different bytes raises `UnikoError::IdConflict`. An
  **auto-generated** id expresses no identity, so identical bytes still dedup
  onto the existing artifact — re-running a corpus load does not duplicate
  every document. Either way a dedup hit now wires the *current* call's
  session/message provenance before returning, and reports the id the artifact
  actually lives under rather than a throwaway UUID that resolves to nothing.
  Both the text and PDF ingest paths were affected.

- **A session that only ingested documents could never recall them.**
  `Session::ingest` never created the `:Session` row — only `observe` did, via
  `ensure_session_and_sender` — so `link_artifact_context` found no session to
  attach to and silently skipped the `ATTACHED_TO` edge. Session-scoped
  recall's `Artifact -ATTACHED_TO-> Session` arm was correct all along; the
  edge was simply never written. Ingest now materializes the session first.

- **A crash before a new store's first flush left it permanently unopenable.**
  uni-db refuses to open a store with WAL segments and no snapshot manifest,
  and the manifest is written only by a flush — so every new store had a
  window (up to `auto_flush_interval`, 5s) in which a crash stranded
  committed, fsynced writes. Opening a persistent knowledge base now publishes
  a baseline snapshot before any caller write can reach it. Tracked upstream
  as `rustic-ai/uni-db#275`.

- **Phase 1 of the recall cascade contributed nothing on any facade-ingested
  knowledge base.** `phase1_strategy` defaults to `"boost"`, which scores
  session-level chunks reached via
  `Fact -SUPPORTED_BY-> Observation -OBSERVED_IN-> Message -IN_SESSION->
  Session -HAS_CHUNK-> Chunk`. That walk returned nothing, for two independent
  reasons:
    - `fact_session_chunk_ids` traversed `SUPPORTED_BY` **inbound** from the
      Fact, but the schema registers it `Fact → Observation`, so the pattern
      could never match.
    - No session-level chunks existed. `chunk_session` and
      `chunk_session_observations` were only ever called by `uniko-bench`; no
      facade path invoked them.
  Session-scoped recall and the observation → entity `ABOUT` bridge were dark
  for the same reason.
- **`Agent::delete_session` orphaned session-anchored chunks**, leaving deleted
  content live in the vector and full-text indexes and still recallable.
- **Consolidation never triggered itself, so `Fact`s, `Procedure`s and `Topic`s
  were never derived** for any library consumer. `ObservationsReady` — the
  signal that advances the consolidation worker's per-agent counter — was
  defined, re-exported and received, but nothing produced it, so neither the
  threshold nor the periodic timer (which only sweeps agents with a non-zero
  counter) ever fired. The ingest path now emits it, and `Agent::consolidate()`
  provides an explicit path that needs no streaming pipeline.

### Added

- **`Session::finalize`** — builds (or refreshes) a session's transcript and
  observation chunk surfaces, returning a `FinalizeReport`. Cheap and
  idempotent when the session has not grown: nothing is rewritten and nothing
  is re-embedded. Awaits the streaming pipeline first when streaming is on.
  `Session::summarize` now calls it best-effort, so existing callers get the
  fix without a code change.
- **`Agent::finalize_session`** and **`Agent::unfinalized_session_ids`** — the
  backfill path for knowledge bases ingested before the above fix. No automatic
  migration runs at `open()`; backfill is an explicit, resumable loop.
- **`ChunkMode`** on the session chunkers: `Once` keeps the previous
  build-once semantics, `Refresh` rebuilds a grown session by deleting the old
  generation and writing the replacement in a single transaction.
- **`Agent::consolidate`** — run one consolidation cycle on demand, returning
  `CycleStats`. Always available; no streaming pipeline required.
- `Session` is a **context manager** in Python: `async with agent.session(id)`
  (or a plain `with`) finalizes on exit, best-effort, without suppressing an
  in-flight exception.
- `FinalizeReport.ended_at` — `finalize` now stamps the Session's `ended_at`
  from its latest message. Note a Session counts as *open* while `ended_at` is
  null, so a finalized Session is skipped by the inactivity auto-close sweep.
- Python parity for all of the above (`Session.finalize`,
  `Agent.finalize_session`, `Agent.unfinalized_session_ids`,
  `Agent.consolidate`, plus `*_sync` twins, `FinalizeReport`, `CycleStats`,
  stubs, and the `uniko.models` mirror).

### Removed

- `UnikoConfig::phase1_coverage_threshold`. It was defaulted and range-validated
  but never read — recall uses the hardcoded `COVERAGE_GATE_PHASE1` constant —
  so setting it did nothing. Deserializing a config that still carries the key
  is unaffected (unknown fields are ignored).

### Changed

- `config/schema.json` regenerated from the live schema (25 labels, 54 edge
  types; the tracked snapshot had drifted to 22/48).
- `config/catalog_minilm.json` regenerated: its `nlp/default` alias still used
  the retired `task: "raw"` shape and would not load.
- A `Refresh` of the session chunks is now **incremental** — chunks are compared
  index by index and only the suffix from the first mismatch is rebuilt, so
  appending turns re-embeds the tail rather than the whole transcript.
- Documentation corrected against source across `docs/BLACKBOOK.md`, the
  website, `README.md`, and `bindings/uniko-py/README.md` — most consequentially
  the install story, which claimed uniko was unpublished and had no prebuilt
  wheels. Also fixed the wrong `query_variants` doc-comments in
  `uniko-store::config` and `uniko-memory::recall` (an empty vec selects the
  single `keywords` variant, not all four), which the website had copied
  verbatim.

## [0.2.0] - 2026-08-03

Dependency and packaging release. The headline is a fix: the agent-facing Locy
surface (`assume` / `abduce` / `run_rule`) did not work at all through the
`Uniko` facade — which is every Python SDK user.

### Fixed

- **`Agent::assume`, `Agent::abduce` and `Agent::run_rule` failed on every
  facade-built instance** with `LocyRuntimeError: Sub-plan error: Unresolved
  parameter: $agent_id`. uni-db resolves each *registered* rule as a sub-plan of
  any Locy program — including one that references no rules
  ([rustic-ai/uni-db#157](https://github.com/rustic-ai/uni-db/issues/157)) — and
  `Uniko` registers four parameterized stdlib rules at construction, so their
  parameters became mandatory for unrelated calls. The facade now binds them by
  default; caller-supplied parameters still take precedence. `run_active_rules`
  had always carried this union; the facade paths simply never did.

  Only the facade was affected. Code driving `KnowledgeBase` directly never
  registers the stdlib rules and was always fine.

### Changed

- **uni-db 2.5.0 → 3.2.0** (0.1.1 shipped on 2.5.0). Net **−24 dependency
  crates**: 3.2.0 drops the `lancedb` wrapper and with it the CJK tokenizer
  stack (lindera, jieba-rs, kanaria) and the rkyv zero-copy stack.
- **`TemporalValue::epoch_millis` was removed upstream** in a minor release.
  Replaced by `uniko_store::temporal_epoch_millis`, which reproduces the removed
  semantics exactly. Anything that called it through uniko is unaffected; direct
  callers of the uni-db accessor must migrate.
- **Wheels are built with a `dist` profile** — release plus whole-graph thin LTO
  at `codegen-units = 1`. The base `uniko` wheel is now **74.0 MiB** (197.4 MiB
  uncompressed), down from over PyPI's 100 MB per-file limit, which is what made
  PyPI publishing possible at all.
- **Linux builds now require `mold`.** `.cargo/config.toml` forces it as the
  link backend; install it before building from source.

### Added

- `uniko_store::temporal_epoch_millis` — epoch milliseconds from a
  datetime-shaped `TemporalValue`.
- `uniko_store::xervo` now re-exports `ModelAliasSpec`, `ModelTask` and
  `WarmupPolicy`, which callers registering an extra model already needed to
  name.
- **PyPI publishing**, as one job per project, so a rejection of one wheel
  cannot abort the others.
- **A PEP 503 package index** on the docs site, generated from the GitHub
  Release assets — a third install channel that does not depend on PyPI.
- `export-schema` binary for regenerating `config/schema.json`.

### Known limitations

Unchanged from 0.1.1, with one correction: the base `uniko` wheel now fits
under PyPI's default per-file limit, but **`uniko-cuda` and `uniko-metal` still
may not**. Those install from the GitHub Release assets or the package index.

### Installation

Rust:

```toml
[dependencies]
uniko-api = "0.2.0"
```

Python (CPU):

```sh
pip install uniko
```

GPU variants (install exactly one; all import as `uniko`):

```sh
pip install uniko-cuda    # NVIDIA CUDA, Linux x86_64
pip install uniko-metal   # Apple Silicon, macOS arm64
```

If a GPU wheel is not on PyPI, install it from the package index:

```sh
pip install uniko-cuda --extra-index-url https://rustic-ai.github.io/uniko/packages/
```

## [0.1.1] - 2026-07-08

**First public release.**

uniko is an embedded, Rust-native cognitive memory system for AI agents. It turns
conversations into structured, searchable, reasoned-over knowledge — with **no LLM in
the recall hot path**. It links into the host process like SQLite and requires no
external infrastructure (no Neo4j, no Qdrant, no PostgreSQL): a single in-process
graph + vector + full-text store with a Locy logic engine. Compile knowledge in once
at write time; query it forever.

Built on [`uni-db`](https://crates.io/crates/uni-db) (embedded multi-model graph
database) and `uni-xervo` (model runtime for embeddings, NLP, and reranking).

### Distribution & packaging

- **Rust crates on crates.io.** Six layered, independently-versioned crates, all
  sharing one workspace version:
  - `uniko-store` — Layer 1: graph storage, search, and the Locy runtime (the sole
    boundary to `uni-db`).
  - `uniko-pipes` — Layer 2: pipeline infrastructure (Step trait, circuit breaker,
    retry, dead-letter queue, metrics).
  - `uniko-extract` — Layer 3: content processing (NER, observations, chunking,
    ingest, embedding).
  - `uniko-memory` — Layer 4: memory management (pipelines, recall cascade, rules,
    consolidation).
  - `uniko-cortex` — Layer 5: higher reasoning (procedure promotion, topic detection).
  - `uniko-api` — the public facade: builders and re-exports, no logic.
- **Three prebuilt Python wheel variants**, all importing the same `uniko` package
  (one installable at a time) — this lifts 0.1.0's "build from source" requirement:
  - **`uniko`** — CPU, statically-bundled ONNX Runtime. Self-contained, zero runtime
    configuration.
  - **`uniko-cuda`** — NVIDIA CUDA: ONNX Runtime CUDA execution provider **plus**
    on-GPU local LLM inference via mistralrs + candle CUDA kernels (Ampere baseline,
    forward-compatible to Ada/Hopper/Blackwell). Linux x86_64.
  - **`uniko-metal`** — Apple Silicon: ONNX Runtime CoreML **plus** mistralrs + candle
    Metal kernels for on-GPU local LLM. macOS arm64.
  - Wheels are `abi3` for CPython ≥ 3.10 (one wheel per platform); GPU wheels resolve
    the host CUDA/cuDNN libraries at runtime rather than bundling them.
- **Single-sourced versioning.** Every crate, the Python distribution version, and the
  runtime `uniko.__version__` derive from one `[workspace.package].version`; CI guards
  the internal dependency pins and the wheel `pyproject.toml` files against drift.

### Added — cognitive memory engine

- **Embedded store on `uni-db`.** A single in-process graph database (graph + vector +
  full-text + Locy logic programming) backs the whole system; `uniko-store` is the only
  crate that touches it.
- **Typed knowledge-graph schema** — 24 node types and 53 edge types, organized around
  communication: messages between participants are the atomic unit, and entities,
  observations, facts, procedures, and topics derive from them with full provenance.
- **Atomic, LLM-free ingest.** Each message compiles into the graph in one
  all-or-nothing transaction, idempotent on `message_id`. CPU-side extraction runs
  first, with no LLM in the hot path:
  - local ONNX NLP cascade (`kniv-deberta`, INT8) producing POS / NER / SRL / DEP / CLS
    from a single encoder pass;
  - rule-based, code-AST, and ONNX named-entity recognition;
  - observation extraction (rules, a YAML rules engine, and SRL frames);
  - recursive text chunking;
  - embedding (BGE-small-en-v1.5, 384-d by default; pluggable) via `uni-db` auto-embed.
- **Entity-quality admission and canonicalization.** Strict admission (on by default)
  drops NER noise — temporal, measurement, and quoted-string spans already captured as
  observations, plus greeting/discourse fragments mis-tagged as people — and gates the
  low-confidence catch-all. Canonical text normalization collapses case and punctuation
  variants (`"Melanie."` / `"melanie"` → `melanie`) so a name keys identically across
  rule-based and ONNX NER, `ABOUT` edges, and consolidation, with a cross-source dedup
  cascade over overlapping mentions.
- **Asynchronous consolidation**, off the ingest hot path:
  - fact derivation from observation clusters (paraphrase collapsing), with
    `SUPPORTED_BY` edges weighted by Fact↔Observation cosine;
  - contradiction detection and entity-drift flagging over bitemporal (BTIC) intervals;
  - procedure promotion from repeated episode / action sequences;
  - topic detection via label-propagation communities (optional LLM naming, offline).
- **Three-phase recall cascade with coverage gating:** Phase 1 over compiled knowledge
  (Facts / Topics / Procedures) → Phase 2 hybrid **dense vector + BM25** over
  Episodes / Observations / Messages fused with Reciprocal-Rank Fusion (with optional
  temporal and graph / Personalized-PageRank channels) → Phase 3 full Chunk / Artifact
  fallback. Query understanding is local and rule-based; an optional **cross-encoder
  reranker** (MiniLM) rescores the fused results, and everything assembles into a
  token-budgeted context bundle.
- **Locy formal reasoning** — database-native rule execution for derived knowledge; the
  stdlib sequence detector is the live engine.
- **Agent tools and a query feedback loop.** Answered queries can be recorded as
  Episodes (question, answer, recall node IDs, coverage, usage), feeding procedure
  learning.
- **Visibility and access control** by participant / team / org (`public`,
  `private:{id}`, `team:{id}`, `org:{id}`), applied to recall through a cached viewer
  policy.
- **Python SDK (alpha).** `import uniko` exposes an async-first surface —
  `Uniko → agent(id) → Agent → session(id) → Session`, with verbs observe / recall /
  answer / query / ingest / forget, goals / tasks, and Locy logic — plus a blocking
  `*_sync` twin for every I/O call, a Pydantic IO layer, and a complete `py.typed`
  stub. Requires Python ≥ 3.10; runtime dependency `pydantic ≥ 2`.
- **Benchmark harness** (`uniko-bench`) with LoCoMo and LongMemEval runners,
  microbenchmarks, and performance telemetry.

#### Benchmarks

Self-measured on the full **LoCoMo10** set (10 conversations, 1,986 questions; 22-core
CPU + 8 GB consumer GPU). These are internal figures from the repo's own harness, not a
third-party leaderboard.

- LLM-judge score **0.8117** (81.2%, Gemini-3.1); retrieval hit **0.8555** (85.6%);
  F1 **0.321**; total LLM cost (answer + judge) **$3.55**.
- Ingest of 5,882 turns in **7.5 minutes at $0** (local ONNX only), ~76 ms/turn.
- Mean Q&A latency **4.04 s** (2.84 s recall + 1.20 s generation) — the fastest of six
  systems compared (vs Mem0, Graphiti, Cognee, RAG, and Full-Context), and 33–76×
  faster per-turn ingest than Graphiti / Cognee at $0 ingest cost.

### Known limitations

- **No HTTP or MCP API.** uniko ships as a Rust library (plus the Python SDK); there is
  no network surface or agent-facing tool server yet.
- **No CLI.**
- **Python SDK is alpha.** The full async surface ships with `*_sync` twins and typed
  stubs, but the API may still change before 1.0.
- **GPU wheels build the local-LLM stack from a pinned mistralrs/candle** (crates.io is
  frozen at mistralrs 0.8.1); building them requires a CUDA (13.x) or Metal toolchain.
  The CPU `uniko` wheel needs neither.
- **Some recall paths are still improving.** Date-anchored / temporal questions are the
  largest known failure category; retrieval tuning is ongoing.
- **Not yet shipped** (design / roadmap only): HTTP/MCP server, CLI, sparse / ColBERT
  late-interaction retrieval, multimodal ingest, rule induction, and cross-agent
  sharing.

### Installation

Rust:

```toml
[dependencies]
uniko-api = "0.1.1"
```

Python (CPU):

```sh
pip install uniko
```

GPU variants (install exactly one; all import as `uniko`):

```sh
pip install uniko-cuda    # NVIDIA CUDA, Linux x86_64
pip install uniko-metal   # Apple Silicon, macOS arm64
```

[0.2.0]: https://github.com/rustic-ai/uniko/releases/tag/v0.2.0
[0.1.1]: https://github.com/rustic-ai/uniko/releases/tag/v0.1.1
