//! Layer 1 storage engine wrapping uni-db.
//!
//! [`KnowledgeBase`] is the single entry point for all graph operations in
//! uniko.  Higher layers (Extract, Memory, Cortex) interact with the graph
//! exclusively through this struct.

pub mod batch;
pub mod batch_record;
pub mod blob;
pub mod deletion;
pub mod edges;
pub mod filter;
pub mod kb_stats;
pub mod migrations;
pub mod nodes;

use std::path::Path;
use std::sync::Arc;

use uni_db::{ModelAliasSpec, ModelTask, Uni, UniConfig, WarmupPolicy};

/// Diagnostic perf-knob override read from env vars. Set
/// `UNIKO_WAL_DISABLED=1` to disable WAL, or
/// `UNIKO_AUTOFLUSH_THRESHOLD=N` to override the L0 auto-flush
/// threshold. Returns Some(cfg) only if at least one knob is set, so
/// the default UniConfig isn't perturbed in normal use.
fn apply_perf_knobs_from_env() -> Option<UniConfig> {
    let wal = std::env::var("UNIKO_WAL_DISABLED").ok();
    let flush = std::env::var("UNIKO_AUTOFLUSH_THRESHOLD").ok();
    let flush_interval_off = std::env::var("UNIKO_AUTOFLUSH_INTERVAL_OFF").ok();
    if wal.is_none() && flush.is_none() && flush_interval_off.is_none() {
        return None;
    }
    let mut cfg = UniConfig::default();
    if matches!(wal.as_deref(), Some("1") | Some("true")) {
        cfg.wal_enabled = false;
        tracing::warn!("UNIKO_WAL_DISABLED=1 — running with WAL OFF (diagnostic only)");
    }
    if let Some(s) = flush
        && let Ok(n) = s.parse::<usize>()
    {
        cfg.auto_flush_threshold = n;
        tracing::warn!(threshold = n, "UNIKO_AUTOFLUSH_THRESHOLD set");
    }
    if matches!(flush_interval_off.as_deref(), Some("1") | Some("true")) {
        cfg.auto_flush_interval = None;
        tracing::warn!("UNIKO_AUTOFLUSH_INTERVAL_OFF=1 — disabling time-based flush");
    }
    Some(cfg)
}

use crate::config::UnikoConfig;
use crate::error::{Result, UnikoError};
use crate::schema::constants::{edges as edge_consts, labels};
use crate::schema::{
    EMBED_ALIAS, HYBRID_EMBED_ALIAS, NLP_ALIAS, OCR_ALIAS, RERANK_ALIAS, register_schema,
};
pub use edges::{Direction, EdgeRecord};
pub use filter::Filter;

/// Layer 1 storage engine wrapping a uni-db instance.
///
/// Provides typed CRUD operations, vector/fulltext/hybrid search, graph
/// traversal, and Locy runtime access.  All operations are async because
/// uni-db transactions are async.
///
/// # Examples
///
/// ```no_run
/// # async fn example() -> uniko_store::Result<()> {
/// use uniko_store::config::UnikoConfig;
/// use uniko_store::storage::KnowledgeBase;
///
/// let kb = KnowledgeBase::in_memory(UnikoConfig::default()).await?;
/// // ... use kb for CRUD, search, Locy ...
/// kb.shutdown().await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct KnowledgeBase {
    pub(crate) db: Arc<Uni>,
    pub(crate) config: UnikoConfig,
    /// Serializes the read-modify-write inside
    /// [`KnowledgeBase::bump_modality_presence`].
    ///
    /// uni-db's commit is last-writer-wins on a row, so without an
    /// in-process lock two concurrent bumps can each read the same
    /// pre-image and clobber one another. Bumps are rare (once per
    /// modality on first occurrence), so a single shared mutex is the
    /// simplest correct choice.
    pub(crate) kb_stats_lock: Arc<tokio::sync::Mutex<()>>,
    /// Per-key striped locks shared by every RMW site that reads a row,
    /// mutates it Rust-side, and writes it back.  See
    /// [`crate::locks`] for the design rationale.  Sites use distinct
    /// key namespaces (`fact:…`, `entity:…`, `node:…`) so collisions
    /// across sites are themselves harmless.
    pub(crate) rmw_locks: Arc<crate::locks::StripedLocks>,
    /// Per-key striped locks for the coarse session/sender *setup*
    /// critical section ([`KnowledgeBase::lock_session_setup`]).
    ///
    /// Deliberately a SEPARATE table from [`Self::rmw_locks`]: the setup
    /// guards are held across `merge_node`, which takes a `node:…` lock
    /// on `rmw_locks`. Sharing one table lets a `session:…`/`participant:…`
    /// key and the nested `node:…` key hash to the same stripe, and the
    /// task then blocks forever on the non-reentrant mutex it already
    /// holds (issue #36). Distinct tables make the nested acquisition
    /// structurally independent of the outer one.
    pub(crate) setup_locks: Arc<crate::locks::StripedLocks>,
}

impl KnowledgeBase {
    /// Create an in-memory knowledge base for testing or ephemeral use.
    ///
    /// Registers the full schema on creation and eagerly warms xervo
    /// models via [`prefetch_all`](uni_db::api::xervo::UniXervo::prefetch_all).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] if validation fails, or
    /// [`UnikoError::Storage`] if the database cannot be created.
    pub async fn in_memory(config: UnikoConfig) -> Result<Self> {
        Self::in_memory_with_xervo(config, Vec::new()).await
    }

    /// Create an in-memory knowledge base with extra xervo model aliases.
    ///
    /// Merges the catalog (from file or built-in) with `extra_catalog`.
    /// Use this to add LLM generation models (e.g., for benchmarks or
    /// answer synthesis).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] if validation fails, or
    /// [`UnikoError::Storage`] if the database cannot be created.
    pub async fn in_memory_with_xervo(
        config: UnikoConfig,
        extra_catalog: Vec<ModelAliasSpec>,
    ) -> Result<Self> {
        config.validate()?;
        let catalog = load_catalog(&config, &extra_catalog)?;
        let db = Uni::in_memory().xervo_catalog(catalog).build().await?;
        apply_schema(&db, &config).await?;
        prefetch_models(&db).await;
        Self {
            db: Arc::new(db),
            config,
            kb_stats_lock: Arc::new(tokio::sync::Mutex::new(())),
            rmw_locks: Arc::new(crate::locks::StripedLocks::from_env()),
            setup_locks: Arc::new(crate::locks::StripedLocks::from_env()),
        }
        .finalize_init()
        .await
    }

    /// Open or create a persistent knowledge base at `path`.
    ///
    /// Registers the full schema on open (idempotent) and eagerly
    /// warms xervo models.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] if validation fails, or
    /// [`UnikoError::Storage`] if the database cannot be opened.
    pub async fn open(path: impl AsRef<Path>, config: UnikoConfig) -> Result<Self> {
        Self::open_with_xervo(path, config, Vec::new()).await
    }

    /// Open a persistent knowledge base with extra xervo model aliases.
    ///
    /// Merges the catalog (from file or built-in) with `extra_catalog`.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] if validation fails, or
    /// [`UnikoError::Storage`] if the database cannot be opened.
    pub async fn open_with_xervo(
        path: impl AsRef<Path>,
        config: UnikoConfig,
        extra_catalog: Vec<ModelAliasSpec>,
    ) -> Result<Self> {
        Self::open_with_xervo_inner(path, config, extra_catalog, true).await
    }

    /// Open a KB without pre-warming any xervo models.
    ///
    /// Useful for tools that only inspect the graph (e.g. read-only
    /// Cypher) and never call `similar_to`/`generate`. Skipping prefetch
    /// saves the multi-minute model-download/load cost on each launch.
    /// Models still load lazily on first use if a query needs them.
    pub async fn open_with_xervo_no_prefetch(
        path: impl AsRef<Path>,
        config: UnikoConfig,
        extra_catalog: Vec<ModelAliasSpec>,
    ) -> Result<Self> {
        Self::open_with_xervo_inner(path, config, extra_catalog, false).await
    }

    async fn open_with_xervo_inner(
        path: impl AsRef<Path>,
        config: UnikoConfig,
        extra_catalog: Vec<ModelAliasSpec>,
        prefetch: bool,
    ) -> Result<Self> {
        config.validate()?;
        let catalog = load_catalog(&config, &extra_catalog)?;
        let mut builder = Uni::open(path.as_ref().to_string_lossy()).xervo_catalog(catalog);
        if let Some(uni_cfg) = apply_perf_knobs_from_env() {
            builder = builder.config(uni_cfg);
        }
        let db = builder.build().await?;
        apply_schema(&db, &config).await?;
        if prefetch {
            prefetch_models(&db).await;
        }
        Self {
            db: Arc::new(db),
            config,
            kb_stats_lock: Arc::new(tokio::sync::Mutex::new(())),
            rmw_locks: Arc::new(crate::locks::StripedLocks::from_env()),
            setup_locks: Arc::new(crate::locks::StripedLocks::from_env()),
        }
        .finalize_init()
        .await
    }

    /// Build a single ONNX/Xervo runtime that multiple KBs can share.
    ///
    /// At `--question-concurrency N` the bench opens N persistent KBs;
    /// without sharing, each opens its own `ModelRuntime`, which loads
    /// its own ONNX sessions (the per-session BFC arena dominates GPU
    /// VRAM). Sharing one runtime keeps weights and the activation
    /// arena resident exactly once.
    ///
    /// Implementation note: uni-db's `UniBuilder::xervo_runtime` takes
    /// a pre-built `Arc<ModelRuntime>`, but the runtime's provider
    /// registration is gated by `#[cfg(feature = "provider-*")]`
    /// inside uni-db — reproducing that gating outside the crate is
    /// fragile. Instead, we bootstrap by opening a tiny
    /// `Uni::in_memory()` with the catalog (which goes through all
    /// the provider-registration gates correctly), extract the
    /// resulting `Arc<ModelRuntime>`, and drop the bootstrap `Uni`.
    /// The model warmup runs once here, so callers get a hot runtime
    /// they can hand to many [`KnowledgeBase::open_with_runtime`]
    /// calls.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] if catalog validation fails,
    /// or [`UnikoError::Storage`] if the bootstrap `Uni` cannot be
    /// opened (the in-memory backend should not fail in practice).
    /// Returns [`UnikoError::Internal`] if the bootstrap `Uni`
    /// somehow finishes without registering an xervo runtime.
    pub async fn build_shared_runtime(
        config: &UnikoConfig,
        extra_catalog: &[ModelAliasSpec],
    ) -> Result<Arc<uni_xervo::runtime::ModelRuntime>> {
        config.validate()?;
        let catalog = load_catalog(config, extra_catalog)?;
        let bootstrap = Uni::in_memory().xervo_catalog(catalog).build().await?;
        let runtime = bootstrap.xervo().raw_runtime().cloned().ok_or_else(|| {
            UnikoError::Internal(
                "bootstrap Uni did not register an xervo runtime (was the catalog empty?)".into(),
            )
        })?;
        // Warm every alias up front so the first inference per KB
        // doesn't pay the cold-start latency.
        prefetch_models(&bootstrap).await;
        // Dropping the bootstrap Uni at the end of scope is fine —
        // the runtime is held by the returned `Arc`.
        drop(bootstrap);
        Ok(runtime)
    }

    /// Open a persistent KB that **shares** the supplied `ModelRuntime`.
    ///
    /// Use with [`KnowledgeBase::build_shared_runtime`] to run many
    /// concurrent KBs against one ONNX session, instead of paying the
    /// per-KB session cost. Each KB still has its own graph storage
    /// on disk; only the inference runtime is shared.
    ///
    /// Skips the per-KB model prefetch (the shared runtime is already
    /// warmed up at construction). Still applies the schema per-KB.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] if validation fails, or
    /// [`UnikoError::Storage`] if the database cannot be opened.
    pub async fn open_with_runtime(
        path: impl AsRef<Path>,
        config: UnikoConfig,
        runtime: Arc<uni_xervo::runtime::ModelRuntime>,
    ) -> Result<Self> {
        config.validate()?;
        let mut builder = Uni::open(path.as_ref().to_string_lossy()).xervo_runtime(runtime);
        // Apply the same env-driven storage perf knobs (WAL / autoflush) the
        // catalog-open path applies — otherwise opening via a shared runtime
        // silently ignores them.
        if let Some(uni_cfg) = apply_perf_knobs_from_env() {
            builder = builder.config(uni_cfg);
        }
        let db = builder.build().await?;
        apply_schema(&db, &config).await?;
        Self {
            db: Arc::new(db),
            config,
            kb_stats_lock: Arc::new(tokio::sync::Mutex::new(())),
            rmw_locks: Arc::new(crate::locks::StripedLocks::from_env()),
            setup_locks: Arc::new(crate::locks::StripedLocks::from_env()),
        }
        .finalize_init()
        .await
    }

    /// Run post-construction init steps. Currently:
    ///
    /// - [`init_kb_stats`](Self::init_kb_stats): writes the
    ///   `:KnowledgeBaseStats` singleton on first open, or verifies
    ///   `blob_storage` matches on reopen.
    ///
    /// All five public constructors funnel through this so the
    /// singleton row is always present after a successful `open` /
    /// `in_memory`.
    async fn finalize_init(self) -> Result<Self> {
        self.init_kb_stats().await?;
        Ok(self)
    }

    /// Direct access to the underlying uni-db instance.
    ///
    /// Escape hatch for advanced operations not covered by the typed
    /// [`KnowledgeBase`] API — intended for **tests** (graph assertions)
    /// and the **benchmark** crate (raw microbenchmarks). Product crates
    /// (`uniko-{memory,extract,cortex,pipes}`) must NOT use this: a CI gate
    /// forbids `.db()` and `use uni_db` in their `src/` (issue #2). Reach
    /// the graph through a typed method (`repository`/`operations`/`model`)
    /// or [`begin_tx`](Self::begin_tx) instead.
    pub fn db(&self) -> &Uni {
        &self.db
    }

    /// Open a fresh read-write [`Transaction`](uni_db::Transaction).
    ///
    /// The sanctioned way for higher crates to run a multi-statement write
    /// that [`transact_with_retry`](Self::transact_with_retry) can't wrap —
    /// e.g. one holding external [`StripedLocks`](crate::locks::StripedLocks)
    /// guards across the whole transaction (the entity-ingest path). The
    /// caller commits/rolls back the returned transaction and drives it
    /// only through validated `*_in_tx` helpers, never raw Cypher.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] (or [`UnikoError::Conflict`]) if the
    /// transaction cannot be opened.
    pub async fn begin_tx(&self) -> Result<uni_db::Transaction> {
        Ok(self.db.session().tx().await?)
    }

    /// Run `f` inside a transaction, automatically retrying transient
    /// storage conflicts (SSI aborts, constraint/transaction conflicts,
    /// commit/lock-timeout contention) with capped exponential backoff.
    ///
    /// This makes uni-db's SSI contention story reachable from uniko: a
    /// retriable failure — surfaced as [`UnikoError::Conflict`] by the
    /// `From<uni_db::UniError>` boundary — re-runs the closure from a fresh
    /// transaction instead of failing the caller. Non-retriable errors
    /// propagate immediately.
    ///
    /// The closure receives the [`uni_db::Transaction`] by value and returns it
    /// alongside a uniko [`Result`] (`(tx, result)`); the wrapper then commits
    /// or rolls back. Threading the transaction by value (rather than by
    /// borrow) keeps the bound free of higher-ranked lifetimes, so the closure
    /// can call the crate's validated `*_in_tx` write helpers AND remain `Send`
    /// when the whole operation is spawned (e.g. the ingest workers). Because
    /// the closure re-runs per attempt, any input it consumes must be re-usable
    /// across attempts (capture by `Copy` reference or re-clone inside).
    ///
    /// In-process [`crate::locks::StripedLocks`] guards must be acquired
    /// *outside* this call (lock first, retry the tx body inside) so the
    /// single-writer-per-key invariant holds across all attempts.
    ///
    /// `opts` reuses [`uni_db::RetryOptions`] for a single shared config type.
    /// (uni-db's own `Session::transact_with_retry` is not used directly
    /// because its closure must return `uni_db::Result`, which cannot carry the
    /// crate's validated-helper errors, and its backoff/metric internals are
    /// not part of uni-db's public surface.)
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Conflict`] if attempts are exhausted on a
    /// retriable conflict, or another [`UnikoError`] for non-retriable
    /// failures.
    pub async fn transact_with_retry<F, Fut, T>(
        &self,
        opts: uni_db::RetryOptions,
        mut f: F,
    ) -> Result<T>
    where
        F: FnMut(uni_db::Transaction) -> Fut,
        Fut: std::future::Future<Output = (uni_db::Transaction, Result<T>)> + Send,
    {
        let mut attempt: u32 = 1;
        loop {
            let tx = self.db.session().tx().await?;
            let (tx, result) = f(tx).await;
            match result {
                Ok(value) => match tx.commit().await {
                    Ok(_) => return Ok(value),
                    // `commit` consumed `tx`; nothing to roll back.
                    Err(e) => {
                        let err = UnikoError::from(e);
                        if err.is_retriable() && attempt < opts.max_attempts {
                            attempt += 1;
                            retry_backoff(&opts, attempt).await;
                        } else {
                            return Err(err);
                        }
                    }
                },
                Err(err) if err.is_retriable() && attempt < opts.max_attempts => {
                    tx.rollback();
                    attempt += 1;
                    retry_backoff(&opts, attempt).await;
                }
                Err(err) => {
                    tx.rollback();
                    return Err(err);
                }
            }
            tracing::debug!(attempt, "retrying transaction after retriable conflict");
        }
    }

    /// Acquire the per-entity RMW striped locks for `entity_ids`, in a
    /// deadlock-free order, returning the held guards.
    ///
    /// Entity dedup is a check-then-create across a transaction the caller
    /// owns. uni-db's `entity_id` index is non-unique, so two concurrent
    /// ingests that both read "absent" would both CREATE a duplicate row
    /// (an insert-phantom SSI does not catch — see the uni-db workarounds notes
    /// RC2). The caller must hold these guards across BOTH the existence
    /// re-read AND the commit so a second writer cannot interleave; the guards
    /// are dropped when the returned `Vec` goes out of scope (after commit).
    /// Ids are de-duplicated and sorted so concurrent callers acquire shared
    /// keys in the same order, preventing deadlock (mirrors
    /// [`KnowledgeBase::batch_upsert_facts`]).
    pub async fn lock_entity_ids(
        &self,
        entity_ids: &[String],
    ) -> Vec<tokio::sync::MutexGuard<'_, ()>> {
        let keys: Vec<Vec<u8>> = entity_ids
            .iter()
            .map(|id| crate::locks::entity_lock_key(id))
            .collect();
        // `lock_many` dedups by stripe index (not just by key bytes):
        // two distinct entity ids can hash to the same stripe, and
        // acquiring that non-reentrant stripe twice would self-deadlock.
        self.rmw_locks.lock_many(&keys).await
    }

    /// Acquire the per-session and per-participant RMW locks for a
    /// first-sight session/sender setup, in a deadlock-free order.
    ///
    /// The Session get-or-create, the Participant merge, and the
    /// `PARTICIPATED_IN` link are check-then-create operations on rows
    /// shared by every ingest of the same session. uni-db's `entity_id`
    /// (and equivalent) indexes are non-unique and SSI does not catch an
    /// insert-phantom, so concurrent ingests with independent
    /// `SessionContext`s would otherwise duplicate those rows or abort on
    /// a read-write antidependency. The caller holds these guards across
    /// the existence checks and their commits so a second writer cannot
    /// interleave; the guards drop when the returned `Vec` goes out of
    /// scope. Keys are de-duplicated and sorted so concurrent callers
    /// acquire shared keys in the same order, preventing deadlock (mirrors
    /// [`KnowledgeBase::lock_entity_ids`]).
    ///
    /// These guards come from [`Self::setup_locks`], a lock table separate
    /// from [`Self::rmw_locks`], because the caller holds them across
    /// [`merge_node`](Self::merge_node), which itself takes a `node:…`
    /// `rmw_locks` stripe. On a shared table an outer setup key and that
    /// nested node key can hash to the same stripe and self-deadlock on a
    /// non-reentrant `tokio::sync::Mutex` (issue #36). Callers must NOT
    /// take an outer `rmw_locks` guard (e.g. via
    /// [`lock_entity_ids`](Self::lock_entity_ids)) around these, so the
    /// two domains stay strictly ordered setup-then-RMW and no AB/BA
    /// cycle across domains is possible.
    pub async fn lock_session_setup(
        &self,
        session_id: &str,
        participant_id: &str,
    ) -> Vec<tokio::sync::MutexGuard<'_, ()>> {
        let keys = [
            crate::locks::session_lock_key(session_id),
            crate::locks::participant_lock_key(participant_id),
        ];
        // `lock_many` dedups by stripe index: the session and participant
        // keys can hash to the same stripe, and acquiring that
        // non-reentrant stripe twice would self-deadlock.
        self.setup_locks.lock_many(&keys).await
    }

    /// Rebuild both striped lock tables with `n` stripes each.
    ///
    /// Test-support only: `n == 1` forces every key in a domain onto one
    /// stripe, which turns the probabilistic collisions behind issue #36
    /// into a deterministic one so a regression test does not depend on
    /// hash luck. Call it before any concurrent work on the KB — replacing
    /// the tables drops any guards' backing mutexes for other clones.
    #[doc(hidden)]
    #[must_use]
    pub fn with_lock_stripes(mut self, n: usize) -> Self {
        self.rmw_locks = Arc::new(crate::locks::StripedLocks::new(n));
        self.setup_locks = Arc::new(crate::locks::StripedLocks::new(n));
        self
    }

    /// Runtime configuration.
    pub fn config(&self) -> &UnikoConfig {
        &self.config
    }

    /// Graceful shutdown, flushing pending writes.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Internal`] if other `Arc` references still
    /// exist, or [`UnikoError::Storage`] if shutdown fails.
    pub async fn shutdown(self) -> Result<()> {
        let db = Arc::try_unwrap(self.db).map_err(|_| {
            UnikoError::Internal("cannot shutdown: outstanding references exist".into())
        })?;
        db.shutdown().await?;
        Ok(())
    }
}

// ── Internal helpers ────────────────────────────────────────────────

/// Sleep before a retry attempt (`attempt == 2` is the first retry), using
/// capped exponential backoff: `base_backoff * 2^(attempt-2)` clamped to
/// `max_backoff`. Jitter is omitted deliberately — uniko runs against an
/// embedded single-process engine where the hot RMW paths are already
/// serialized by [`crate::locks::StripedLocks`], so retriable conflicts are
/// rare and there is no thundering herd to de-correlate.
async fn retry_backoff(opts: &uni_db::RetryOptions, attempt: u32) {
    let steps = attempt.saturating_sub(2).min(20);
    let delay = opts
        .base_backoff
        .saturating_mul(1u32 << steps)
        .min(opts.max_backoff);
    tokio::time::sleep(delay).await;
}

/// Load the xervo model catalog from a JSON file or build the default.
///
/// When `config.catalog_path` is set, reads from that file and appends
/// `extra`. Otherwise builds the default catalog from config + `extra`.
fn load_catalog(config: &UnikoConfig, extra: &[ModelAliasSpec]) -> Result<Vec<ModelAliasSpec>> {
    let mut catalog = if let Some(path) = &config.catalog_path {
        uni_db::xervo_catalog_from_file(path)
            .map_err(|e| UnikoError::Config(format!("catalog {}: {e}", path.display())))?
    } else {
        embed_catalog(config)
    };
    catalog.extend_from_slice(extra);
    Ok(catalog)
}

/// Eagerly download and warm every xervo model in the catalog.
///
/// `prefetch_all()` materializes each artifact into the repo snapshot
/// and loads the model into memory, so the first inference call hits a
/// pre-warmed runner.  Errors are surfaced at `warn` level (we keep
/// `required: false` so the KB still opens; failures are operationally
/// important to see).
async fn prefetch_models(db: &Uni) {
    let started = std::time::Instant::now();
    match db.xervo().prefetch_all().await {
        Ok(()) => {
            tracing::info!(
                elapsed_ms = started.elapsed().as_millis() as u64,
                "xervo prefetch_all complete"
            );
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                elapsed_ms = started.elapsed().as_millis() as u64,
                "xervo prefetch_all failed — models may load lazily on first use"
            );
        }
    }
}

/// Apply the schema from a JSON file or the builder-based registration.
async fn apply_schema(db: &Uni, config: &UnikoConfig) -> Result<()> {
    if let Some(path) = &config.schema_path {
        db.load_schema(path)
            .await
            .map_err(|e| UnikoError::Schema(e.to_string()))
    } else {
        register_schema(db, config).await
    }
}

/// Build the Xervo model catalog for embedding, reranking, and NLP inference.
///
/// Registers up to three models:
/// - `"embed/default"` — ONNX embedding model (Nomic 768d default).
/// - `"rerank/default"` — ONNX cross-encoder reranker (only when `config.reranker.enabled`).
/// - `"nlp/default"` — multi-task NER/POS/Dep/CLS via ONNX from HuggingFace.
///
/// All entries use `WarmupPolicy::Lazy` (loaded on first call) and
/// `required: false` (startup succeeds even if providers are unavailable).
///
/// Exposed publicly so tests and downstream crates that create raw
/// `Uni` instances can configure the same catalog.
pub fn embed_catalog(config: &UnikoConfig) -> Vec<ModelAliasSpec> {
    let embed_eps = resolve_eps(config.embedding.execution_providers.as_deref());
    let rerank_eps = resolve_eps(config.reranker.execution_providers.as_deref());
    // NLP defaults to the embedder's device list when the operator
    // hasn't explicitly overridden it — keeps all three local/onnx
    // aliases on the same device by default while still letting a
    // bench profile pin NLP to CPU when the embedder is on GPU.
    let nlp_eps = match config.nlp.execution_providers.as_deref() {
        Some(eps) => eps.to_vec(),
        None => embed_eps.clone(),
    };
    let ocr_eps = resolve_eps(config.ocr.execution_providers.as_deref());

    // The dense alias always serves every lone-dense auto-embed column
    // (Message, Summary, …), all computed embeddings (`kb.embed`), and query
    // embedding via `ModelTask::Embed`. A hybrid embedder additionally gets a
    // separate `EmbedHybrid` alias below — one model cannot back both, since
    // uni-db routes lone-dense columns through the `EmbeddingModel` trait,
    // which the hybrid model does not implement.
    let mut catalog = vec![
        ModelAliasSpec {
            alias: EMBED_ALIAS.to_string(),
            task: ModelTask::Embed,
            provider_id: config.embedding.provider.clone(),
            model_id: config.embedding.model_id.clone(),
            revision: None,
            warmup: WarmupPolicy::Lazy,
            required: false,
            timeout: None,
            load_timeout: None,
            retry: None,
            options: build_embed_options(&config.embedding, &embed_eps),
        },
        ModelAliasSpec {
            alias: NLP_ALIAS.to_string(),
            // Managed multi-task NLP: xervo owns tokenization + POS/NER/
            // DEP/SRL/CLS decode (uniko adapts the output). Was `Raw`
            // (uniko decoded the tensors in-crate) before the 2026-06
            // migration to `NlpModel`.
            task: ModelTask::Nlp,
            provider_id: "local/onnx".to_string(),
            model_id: config.nlp.model_id.clone(),
            revision: None,
            warmup: WarmupPolicy::Lazy,
            required: false,
            timeout: None,
            load_timeout: None,
            retry: None,
            // `OnnxNlpModel` reads `onnx_path` (not `artifact`) plus the
            // tokenizer / label-map asset names shipped in the model repo.
            options: serde_json::json!({
                "onnx_path": config.nlp.artifact,
                "tokenizer_path": "tokenizer.json",
                "label_maps_path": "label_maps.json",
                "max_seq_len": 128,
                "execution_providers": nlp_eps,
            }),
        },
    ];

    // Hybrid embedder: a second alias backed by the SAME model, registered
    // as `EmbedHybrid`, fills the dense + sparse + ColBERT group on Chunk /
    // Observation in one forward pass. It must be separate from EMBED_ALIAS
    // (see the comment there). The model loads twice (the runtime cache keys
    // on task), so expect ~2× the embedder's VRAM when hybrid is on.
    if config.embedding.sparse_dimensions.is_some()
        || config.embedding.multivector_dimensions.is_some()
    {
        catalog.push(ModelAliasSpec {
            alias: HYBRID_EMBED_ALIAS.to_string(),
            task: ModelTask::EmbedHybrid,
            provider_id: config.embedding.provider.clone(),
            model_id: config.embedding.model_id.clone(),
            revision: None,
            warmup: WarmupPolicy::Lazy,
            required: false,
            timeout: None,
            load_timeout: None,
            retry: None,
            options: build_embed_options(&config.embedding, &embed_eps),
        });
    }

    // The ColBERT reranker style needs no model alias: it re-scores via
    // the embed alias's multi-vector head (MaxSim) inside recall, so only
    // register a `rerank/default` model for the xervo reranker styles.
    if config.reranker.enabled && config.reranker.style != "colbert" {
        catalog.push(ModelAliasSpec {
            alias: RERANK_ALIAS.to_string(),
            task: ModelTask::Rerank,
            provider_id: "local/onnx".to_string(),
            model_id: config.reranker.model_id.clone(),
            revision: None,
            warmup: WarmupPolicy::Lazy,
            required: false,
            timeout: None,
            load_timeout: None,
            retry: None,
            options: serde_json::json!({
                "execution_providers": rerank_eps,
                "style": config.reranker.style,
            }),
        });
    }

    if config.ocr.enabled {
        // Two-stage pipeline OCR (DBNet detection + CRNN/CTC recognition) on
        // uni-db's `local/onnx` provider. Drives the `Ocr` tier of
        // `uni-xervo-pdf`. Option keys mirror uni-xervo's `local_onnx` OCR
        // loader (`onnx_path`/`char_dict_path`/`det_onnx_path`/…); defaults
        // target the English PP-OCRv5 export `monkt/paddleocr-onnx`.
        catalog.push(ModelAliasSpec {
            alias: OCR_ALIAS.to_string(),
            task: ModelTask::Ocr,
            provider_id: "local/onnx".to_string(),
            model_id: config.ocr.model_id.clone(),
            revision: None,
            warmup: WarmupPolicy::Lazy,
            required: false,
            timeout: None,
            load_timeout: None,
            retry: None,
            options: serde_json::json!({
                "onnx_path": config.ocr.rec_artifact,
                "char_dict_path": config.ocr.char_dict_path,
                "det_onnx_path": config.ocr.det_artifact,
                "image_height": config.ocr.image_height,
                "image_width": config.ocr.image_width,
                "normalization": config.ocr.normalization,
                "blank_class": 0,
                "execution_providers": ocr_eps,
            }),
        });
    }

    catalog
}

/// Resolve the ONNX execution-provider list for an alias.
///
/// Honours an explicit override from config when provided; otherwise
/// falls back to the build-time default — CUDA → CPU on `gpu-cuda`,
/// CoreML → CPU on `gpu-metal`, CPU otherwise. The returned `Vec`
/// goes into the alias's `options.execution_providers` JSON which
/// uni-xervo's `parse_execution_providers_option` consumes (see
/// `uni-xervo/src/provider/onnx_ep.rs`).
fn resolve_eps(override_eps: Option<&[String]>) -> Vec<String> {
    if let Some(eps) = override_eps {
        return eps.to_vec();
    }
    default_eps()
}

#[cfg(feature = "gpu-cuda")]
fn default_eps() -> Vec<String> {
    vec!["cuda".to_string(), "cpu".to_string()]
}

#[cfg(all(feature = "gpu-metal", not(feature = "gpu-cuda")))]
fn default_eps() -> Vec<String> {
    vec!["coreml".to_string(), "cpu".to_string()]
}

#[cfg(not(any(feature = "gpu-cuda", feature = "gpu-metal")))]
fn default_eps() -> Vec<String> {
    vec!["cpu".to_string()]
}

/// Compose the `options` JSON for the embed alias based on the
/// selected provider.
///
/// `local/onnx` accepts `execution_providers`; remote providers
/// (`remote/openai`, `remote/voyageai`, ...) take any
/// `provider_options` the user supplied verbatim and additionally
/// receive `embedding_dimensions` so xervo's OpenAI provider can
/// thread it into the `/embeddings` request.
fn build_embed_options(
    cfg: &crate::config::EmbeddingConfig,
    embed_eps: &[String],
) -> serde_json::Value {
    let mut opts = match cfg.provider_options.clone() {
        Some(serde_json::Value::Object(map)) => map,
        _ => serde_json::Map::new(),
    };
    if cfg.provider == "local/onnx" {
        // Caller-supplied provider_options pass through for models that
        // resolve via uni-xervo's HF-Hub fallback path (no preset).
        // Those models need `artifact`, `pooling`, `dimensions`,
        // `token_type_ids`, etc. on the alias options.  Plus the
        // execution-provider list xervo always wants for local/onnx.
        opts.insert(
            "execution_providers".to_string(),
            serde_json::json!(embed_eps),
        );
        return serde_json::Value::Object(opts);
    }
    opts.insert(
        "embedding_dimensions".to_string(),
        serde_json::json!(cfg.dimensions),
    );
    serde_json::Value::Object(opts)
}

/// Verify that `label` is a known node label.
pub(crate) fn validate_label(label: &str) -> Result<()> {
    if !labels::ALL.contains(&label) {
        return Err(UnikoError::Schema(format!("unknown node label: {label}")));
    }
    Ok(())
}

/// Verify that `edge_type` is a known edge type.
pub(crate) fn validate_edge_type(edge_type: &str) -> Result<()> {
    if !edge_consts::ALL.contains(&edge_type) {
        return Err(UnikoError::Schema(format!(
            "unknown edge type: {edge_type}"
        )));
    }
    Ok(())
}

/// Verify that a property name is safe for Cypher interpolation.
///
/// Accepts `[a-zA-Z_][a-zA-Z0-9_]*`.
pub(crate) fn validate_property_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(UnikoError::Schema("empty property name".into()));
    }
    let mut chars = name.chars();
    let first = chars.next().unwrap();
    if !first.is_ascii_alphabetic() && first != '_' {
        return Err(UnikoError::Schema(format!("invalid property name: {name}")));
    }
    for ch in chars {
        if !ch.is_ascii_alphanumeric() && ch != '_' {
            return Err(UnikoError::Schema(format!("invalid property name: {name}")));
        }
    }
    Ok(())
}

/// Cypher property fragments paired with their bound `(param_name, value)`
/// list — the output of [`build_kv_pairs`].
type KvPairs = (Vec<String>, Vec<(String, uni_db::Value)>);

/// Build property fragments + bound parameters.
///
/// `fmt(key, param_name)` formats each fragment — e.g. `format!("{key}:
/// ${param}")` for inline `CREATE` props or `format!("{var}.{key} =
/// ${param}")` for `SET` clauses.  Parameter names are `s{offset+i}`
/// to let callers interleave multiple prop sets in one query.
fn build_kv_pairs<F>(
    properties: &std::collections::HashMap<String, uni_db::Value>,
    offset: usize,
    fmt: F,
) -> Result<KvPairs>
where
    F: Fn(&str, &str) -> String,
{
    let mut fragments = Vec::with_capacity(properties.len());
    let mut params = Vec::with_capacity(properties.len());
    for (i, (key, val)) in properties.iter().enumerate() {
        validate_property_name(key)?;
        let param = format!("s{}", offset + i);
        fragments.push(fmt(key, &param));
        params.push((param, val.clone()));
    }
    Ok((fragments, params))
}

/// Build inline property syntax `prop1: $s0, prop2: $s1, ...` for CREATE.
///
/// Returns `(inline_fragment, params)`.
pub(crate) fn build_inline_props(
    properties: &std::collections::HashMap<String, uni_db::Value>,
    offset: usize,
) -> Result<(String, Vec<(String, uni_db::Value)>)> {
    if properties.is_empty() {
        return Ok((String::new(), Vec::new()));
    }
    let (fragments, params) =
        build_kv_pairs(properties, offset, |key, param| format!("{key}: ${param}"))?;
    Ok((fragments.join(", "), params))
}

/// Build a `SET` clause from a property map, returning `(cypher_fragment, params)`.
///
/// Generates `SET {var}.prop0 = $s{offset}, {var}.prop1 = $s{offset+1}, ...`
/// and a list of `(param_name, Value)` bindings.
pub(crate) fn build_set_clause(
    var: &str,
    properties: &std::collections::HashMap<String, uni_db::Value>,
    offset: usize,
) -> Result<(String, Vec<(String, uni_db::Value)>)> {
    if properties.is_empty() {
        return Ok((String::new(), Vec::new()));
    }
    let (fragments, params) = build_kv_pairs(properties, offset, |key, param| {
        format!("{var}.{key} = ${param}")
    })?;
    Ok((format!("SET {}", fragments.join(", ")), params))
}

impl std::fmt::Debug for KnowledgeBase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KnowledgeBase")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}
