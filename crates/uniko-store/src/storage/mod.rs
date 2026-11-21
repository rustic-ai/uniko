//! Layer 1 storage engine wrapping uni-db.
//!
//! [`KnowledgeBase`] is the single entry point for all graph operations in
//! uniko.  Higher layers (Extract, Memory, Cortex) interact with the graph
//! exclusively through this struct.

pub mod batch;
pub mod batch_record;
pub mod blob;
pub mod compatibility;
pub mod deletion;
pub mod edges;
pub mod filter;
pub mod kb_stats;
pub(crate) mod lease;
pub mod migrate_store;
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

use crate::config::{NlpConfig, NlpGeneration, UnikoConfig};
use crate::error::{Result, UnikoError};
use crate::schema::constants::{edges as edge_consts, labels};
use crate::schema::{
    EMBED_ALIAS, HYBRID_EMBED_ALIAS, NLP_ALIAS, OCR_ALIAS, RERANK_ALIAS, register_schema,
};
use crate::shared_runtime::{PrefetchPolicy, SharedRuntime};
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
    /// The model catalog this store handed to uni-db, kept because the
    /// runtime does not expose it (uni-xervo `ModelRuntime::catalog` is
    /// private). `None` when the store was opened over a runtime built
    /// elsewhere ([`KnowledgeBase::open_with_runtime`]), whose catalog
    /// this store never saw.
    pub(crate) catalog: Option<Arc<[ModelAliasSpec]>>,
    /// The store's own background prefetch, cancelled at shutdown.
    pub(crate) prefetch_task: Option<tokio::task::AbortHandle>,
    /// Marks the path open in this process while any clone of this store
    /// lives, so [`inspect`](Self::inspect) and [`migrate`](Self::migrate)
    /// refuse it. `None` for an in-memory store.
    pub(crate) open_path: Option<Arc<OpenPath>>,
}

/// Stores in use in this process, by canonical path. uni-db takes no lock
/// on a store directory, and two handles on one are two writers that don't
/// see each other's writes. So a path takes one claim at a time: one open
/// store, or one inspection or migration. Across processes the claim is
/// backed by an OS lock on [`LOCK_FILE`] in the store directory.
static OPEN_PATHS: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<std::path::PathBuf, Claim>>,
> = std::sync::LazyLock::new(Default::default);

/// The file in a store directory that an open store, inspection or
/// migration holds an exclusive OS lock on, so a second process can't open
/// the store under it. The lock goes with the handle, or with the process.
pub const LOCK_FILE: &str = ".uniko.lock";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Claim {
    /// An open store.
    Open,
    /// Being inspected or migrated.
    Exclusive,
}

/// A claim on a store path, released when dropped.
#[derive(Debug)]
pub(crate) struct OpenPath {
    key: std::path::PathBuf,
    exclusive: bool,
    /// The OS lock on the store's [`LOCK_FILE`], held for the claim's life.
    _lock: Option<std::fs::File>,
    /// For a store in an object store: the lease that stands in for the OS
    /// lock (see [`lease`]).
    lease: Option<lease::RemoteLease>,
}

impl OpenPath {
    /// Claim `path` for an open store, with a lease when it is an object
    /// store URI. Refused while the store is open, or being inspected or
    /// migrated, in this process or another.
    async fn claim_open(path: &Path) -> Result<Arc<Self>> {
        let mut claim = Self::claim(path)?;
        let uri = path.to_string_lossy();
        if uri.contains("://") && file_url_path(&uri).is_none() {
            // The registry entry is already in place, so a second open in
            // this process is refused while the lease is taken.
            claim.lease = Some(lease::RemoteLease::acquire_for_uri(&uri).await?);
        }
        Ok(Arc::new(claim))
    }

    /// Whether this claim's lease was taken over by another process.
    fn lease_lost(&self) -> bool {
        self.lease.as_ref().is_some_and(lease::RemoteLease::is_lost)
    }

    /// Release the lease, if any, so the next open needn't wait for it.
    async fn release_lease(&self) {
        if let Some(l) = &self.lease {
            l.release().await;
        }
    }

    /// Claim `path` for an open store: the in-process registry and, for a
    /// local store, the OS lock.
    fn claim(path: &Path) -> Result<Self> {
        let key = canonical(path);
        let mut open = OPEN_PATHS.lock().expect("open-path registry poisoned");
        match open.get(&key) {
            Some(Claim::Exclusive) => return Err(busy(path)),
            Some(Claim::Open) => return Err(already_open(path)),
            None => {}
        }
        let lock = os_lock(path, true)?;
        open.insert(key.clone(), Claim::Open);
        Ok(Self {
            key,
            exclusive: false,
            _lock: lock,
            lease: None,
        })
    }

    /// Claim `path` exclusively, to inspect or migrate it: refused while
    /// the store is open, inspected or migrated, in this process or another.
    /// A directory that isn't a store yet gets no lock file: inspecting it
    /// leaves it untouched.
    pub(crate) fn claim_exclusive(path: &Path) -> Result<Self> {
        let key = canonical(path);
        let mut open = OPEN_PATHS.lock().expect("open-path registry poisoned");
        match open.get(&key) {
            Some(Claim::Open) => {
                return Err(UnikoError::Config(format!(
                    "the store at {} is open in this process; shut it down first",
                    path.display()
                )));
            }
            Some(Claim::Exclusive) => return Err(busy(path)),
            None => {}
        }
        let lock = if is_store_dir(path) {
            os_lock(path, false)?
        } else {
            None
        };
        open.insert(key.clone(), Claim::Exclusive);
        Ok(Self {
            key,
            exclusive: true,
            _lock: lock,
            lease: None,
        })
    }
}

/// Whether `path` already holds a uni-db store.
fn is_store_dir(path: &Path) -> bool {
    path.join("catalog").join("schema.json").is_file() || path.join("schema.json").is_file()
}

/// Take the OS lock on `path`'s [`LOCK_FILE`], creating the directory when
/// `create` is set. `None` for a remote store URI, which has no local
/// directory to lock.
fn os_lock(path: &Path, create: bool) -> Result<Option<std::fs::File>> {
    let uri = path.to_string_lossy();
    // A `file://` URI is a local directory: lock it like a path. Any other
    // URI is an object store, which has no directory and holds a lease
    // instead (`OpenPath::claim_open`).
    let local;
    let path = if uri.contains("://") {
        match file_url_path(&uri) {
            Some(p) => {
                local = p;
                local.as_path()
            }
            None => return Ok(None),
        }
    } else {
        path
    };
    if !path.is_dir() {
        if !create || path.exists() {
            return Ok(None);
        }
        std::fs::create_dir_all(path)
            .map_err(|e| UnikoError::Storage(format!("create {}: {e}", path.display())))?;
    }
    let lock_path = path.join(LOCK_FILE);
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)
        .map_err(|e| UnikoError::Storage(format!("open {}: {e}", lock_path.display())))?;
    match file.try_lock() {
        Ok(()) => Ok(Some(file)),
        Err(std::fs::TryLockError::WouldBlock) => Err(UnikoError::Config(format!(
            "the store at {} is open in another process; a store directory takes one open \
             handle at a time",
            path.display()
        ))),
        Err(std::fs::TryLockError::Error(e)) => Err(UnikoError::Storage(format!(
            "lock {}: {e}",
            lock_path.display()
        ))),
    }
}

/// The local directory a `file://` URI names, or `None` for any other URI.
fn file_url_path(uri: &str) -> Option<std::path::PathBuf> {
    url::Url::parse(uri)
        .ok()
        .filter(|u| u.scheme() == "file")
        .and_then(|u| u.to_file_path().ok())
}

fn already_open(path: &Path) -> UnikoError {
    UnikoError::Config(format!(
        "the store at {} is already open in this process; use that instance (its clones \
         share it) or shut it down before opening it again",
        path.display()
    ))
}

fn busy(path: &Path) -> UnikoError {
    UnikoError::Config(format!(
        "the store at {} is being inspected or migrated in this process",
        path.display()
    ))
}

impl Drop for OpenPath {
    fn drop(&mut self) {
        let mut open = OPEN_PATHS.lock().expect("open-path registry poisoned");
        match open.get(&self.key) {
            Some(Claim::Open) if !self.exclusive => {
                open.remove(&self.key);
            }
            Some(Claim::Exclusive) if self.exclusive => {
                open.remove(&self.key);
            }
            _ => {}
        }
    }
}

/// The key a path is claimed under. A store that doesn't exist yet is keyed
/// by its canonical parent, so the key doesn't change once uni-db creates
/// the directory (on macOS a temp dir under `/var` canonicalizes to
/// `/private/var`).
fn canonical(path: &Path) -> std::path::PathBuf {
    if let Ok(p) = std::fs::canonicalize(path) {
        return p;
    }
    match (path.parent(), path.file_name()) {
        (Some(parent), Some(name)) => {
            let parent = if parent.as_os_str().is_empty() {
                Path::new(".")
            } else {
                parent
            };
            std::fs::canonicalize(parent)
                .map(|p| p.join(name))
                .unwrap_or_else(|_| path.to_path_buf())
        }
        _ => path.to_path_buf(),
    }
}

impl KnowledgeBase {
    /// Create an in-memory knowledge base for testing or ephemeral use.
    ///
    /// Registers the full schema on creation and eagerly warms each xervo
    /// model in the catalog; see [`model_status`](Self::model_status) for
    /// which ones loaded.
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
        Self::in_memory_with_prefetch(config, extra_catalog, PrefetchPolicy::Eager).await
    }

    /// [`in_memory_with_xervo`](Self::in_memory_with_xervo), loading models
    /// as `prefetch` says instead of always before returning.
    ///
    /// # Errors
    ///
    /// As [`in_memory_with_xervo`](Self::in_memory_with_xervo).
    pub async fn in_memory_with_prefetch(
        config: UnikoConfig,
        extra_catalog: Vec<ModelAliasSpec>,
        prefetch: PrefetchPolicy,
    ) -> Result<Self> {
        config.validate()?;
        let catalog = load_catalog(&config, &extra_catalog)?;
        let kept: Arc<[ModelAliasSpec]> = catalog.clone().into();
        let db = Uni::in_memory().xervo_catalog(catalog).build().await?;
        apply_schema(&db, &config).await?;
        let mut kb = Self {
            db: Arc::new(db),
            config,
            kb_stats_lock: Arc::new(tokio::sync::Mutex::new(())),
            rmw_locks: Arc::new(crate::locks::StripedLocks::from_env()),
            setup_locks: Arc::new(crate::locks::StripedLocks::from_env()),
            catalog: Some(kept),
            prefetch_task: None,
            open_path: None,
        };
        kb.prefetch_task = kb.prefetch_catalog(prefetch).await;
        kb.finalize_init().await
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
        Self::open_with_prefetch(path, config, extra_catalog, PrefetchPolicy::Eager).await
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
        Self::open_with_prefetch(path, config, extra_catalog, PrefetchPolicy::Lazy).await
    }

    /// [`open_with_xervo`](Self::open_with_xervo), loading models as
    /// `prefetch` says. Prefetch starts only after the store has been
    /// checked against the config, so a mismatched reopen fails first.
    ///
    /// # Errors
    ///
    /// As [`open_with_xervo`](Self::open_with_xervo).
    pub async fn open_with_prefetch(
        path: impl AsRef<Path>,
        config: UnikoConfig,
        extra_catalog: Vec<ModelAliasSpec>,
        prefetch: PrefetchPolicy,
    ) -> Result<Self> {
        config.validate()?;
        let catalog = load_catalog(&config, &extra_catalog)?;
        let kept: Arc<[ModelAliasSpec]> = catalog.clone().into();
        let open_path = OpenPath::claim_open(path.as_ref()).await?;
        let mut builder = Uni::open(path.as_ref().to_string_lossy()).xervo_catalog(catalog);
        if let Some(uni_cfg) = apply_perf_knobs_from_env() {
            builder = builder.config(uni_cfg);
        }
        let db = build_or_explain(builder, path.as_ref(), &config).await?;
        apply_schema(&db, &config).await?;
        // Check the store against the config (blob backend, embedding model)
        // before prefetching: a mismatched reopen should fail at once, not
        // after downloading a model it will never use.
        let kb = Self {
            db: Arc::new(db),
            config,
            kb_stats_lock: Arc::new(tokio::sync::Mutex::new(())),
            rmw_locks: Arc::new(crate::locks::StripedLocks::from_env()),
            setup_locks: Arc::new(crate::locks::StripedLocks::from_env()),
            catalog: Some(kept),
            prefetch_task: None,
            open_path: None,
        }
        .finalize_init()
        .await?;
        let mut kb = kb;
        kb.open_path = Some(open_path);
        kb.prefetch_task = kb.prefetch_catalog(prefetch).await;
        Ok(kb)
    }

    /// Create an in-memory store over a [`SharedRuntime`] instead of
    /// loading its own models.
    ///
    /// The runtime must serve every model `config` names, with the same
    /// provider, model, revision and options (execution providers aside);
    /// build it from the same config. It may hold more aliases than that.
    /// The store does not prefetch: the runtime's own
    /// [`PrefetchPolicy`] already decided that.
    ///
    /// # Errors
    ///
    /// [`UnikoError::Config`] when the config is invalid or the runtime
    /// does not serve its models; [`UnikoError::Storage`] if the database
    /// cannot be created.
    pub async fn in_memory_shared(config: UnikoConfig, runtime: &SharedRuntime) -> Result<Self> {
        config.validate()?;
        runtime.check_serves(&config)?;
        let db = Uni::in_memory()
            .xervo_runtime(runtime.runtime.clone())
            .build()
            .await?;
        Self::over_shared(db, config, runtime).await
    }

    /// Open or create a persistent store over a [`SharedRuntime`].
    ///
    /// The same requirements as [`in_memory_shared`](Self::in_memory_shared)
    /// apply. Each store keeps its own graph at `path`; only inference is
    /// shared, and shutting this store down leaves the runtime loaded.
    ///
    /// # Errors
    ///
    /// [`UnikoError::Config`] when the config is invalid or the runtime
    /// does not serve its models; [`UnikoError::Storage`] if the database
    /// cannot be opened.
    pub async fn open_shared(
        path: impl AsRef<Path>,
        config: UnikoConfig,
        runtime: &SharedRuntime,
    ) -> Result<Self> {
        config.validate()?;
        runtime.check_serves(&config)?;
        let open_path = OpenPath::claim_open(path.as_ref()).await?;
        let mut builder =
            Uni::open(path.as_ref().to_string_lossy()).xervo_runtime(runtime.runtime.clone());
        if let Some(uni_cfg) = apply_perf_knobs_from_env() {
            builder = builder.config(uni_cfg);
        }
        let db = build_or_explain(builder, path.as_ref(), &config).await?;
        let mut kb = Self::over_shared(db, config, runtime).await?;
        kb.open_path = Some(open_path);
        Ok(kb)
    }

    async fn over_shared(db: Uni, config: UnikoConfig, runtime: &SharedRuntime) -> Result<Self> {
        apply_schema(&db, &config).await?;
        Self {
            db: Arc::new(db),
            config,
            kb_stats_lock: Arc::new(tokio::sync::Mutex::new(())),
            rmw_locks: Arc::new(crate::locks::StripedLocks::from_env()),
            setup_locks: Arc::new(crate::locks::StripedLocks::from_env()),
            catalog: Some(runtime.catalog.clone()),
            prefetch_task: None,
            open_path: None,
        }
        .finalize_init()
        .await
    }

    /// Build a single model runtime that multiple KBs can share.
    ///
    /// Superseded by [`SharedRuntime::build`], whose handle also carries the
    /// catalog, so stores over it can be checked against their config.
    ///
    /// # Errors
    ///
    /// As [`SharedRuntime::build`].
    #[deprecated(note = "use SharedRuntime::build and KnowledgeBase::open_shared")]
    pub async fn build_shared_runtime(
        config: &UnikoConfig,
        extra_catalog: &[ModelAliasSpec],
    ) -> Result<Arc<uni_xervo::runtime::ModelRuntime>> {
        SharedRuntime::build(config, extra_catalog.to_vec(), PrefetchPolicy::Eager)
            .await
            .map(|rt| rt.model_runtime())
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
    #[deprecated(
        note = "use KnowledgeBase::open_shared, which checks the runtime serves the config"
    )]
    pub async fn open_with_runtime(
        path: impl AsRef<Path>,
        config: UnikoConfig,
        runtime: Arc<uni_xervo::runtime::ModelRuntime>,
    ) -> Result<Self> {
        config.validate()?;
        let open_path = OpenPath::claim_open(path.as_ref()).await?;
        let mut builder = Uni::open(path.as_ref().to_string_lossy()).xervo_runtime(runtime);
        // Apply the same env-driven storage perf knobs (WAL / autoflush) the
        // catalog-open path applies — otherwise opening via a shared runtime
        // silently ignores them.
        if let Some(uni_cfg) = apply_perf_knobs_from_env() {
            builder = builder.config(uni_cfg);
        }
        let db = build_or_explain(builder, path.as_ref(), &config).await?;
        apply_schema(&db, &config).await?;
        Self {
            db: Arc::new(db),
            config,
            kb_stats_lock: Arc::new(tokio::sync::Mutex::new(())),
            rmw_locks: Arc::new(crate::locks::StripedLocks::from_env()),
            setup_locks: Arc::new(crate::locks::StripedLocks::from_env()),
            catalog: None,
            prefetch_task: None,
            open_path: Some(open_path),
        }
        .finalize_init()
        .await
    }

    /// Load this store's catalog models as `policy` says. A model that
    /// fails is logged at `warn`, recorded as
    /// [`ModelState::Failed`](crate::ModelState::Failed) and skipped: the
    /// store still opens (catalog entries keep `required: false`) and the
    /// remaining aliases are still loaded.
    pub(crate) async fn prefetch_catalog(
        &self,
        policy: PrefetchPolicy,
    ) -> Option<tokio::task::AbortHandle> {
        let (Some(catalog), Some(runtime)) = (self.catalog.clone(), self.model_runtime()) else {
            return None;
        };
        policy.run(catalog, runtime).await
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

    /// Refuse a write once this store's lease (an object-store store's
    /// stand-in for a directory lock) was taken over by another process.
    /// Every transaction the store opens checks it.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] once the lease is lost.
    pub fn ensure_writable(&self) -> Result<()> {
        match &self.open_path {
            Some(claim) if claim.lease_lost() => Err(lease::RemoteLease::lost_error()),
            _ => Ok(()),
        }
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
    /// transaction cannot be opened, or [`UnikoError::Config`] once the
    /// store's lease is lost.
    pub async fn begin_tx(&self) -> Result<uni_db::Transaction> {
        self.ensure_writable()?;
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
            self.ensure_writable()?;
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
    ///
    /// A caller that also writes artifacts in the same transaction must use
    /// [`lock_ingest_unit`](Self::lock_ingest_unit) instead — see there for
    /// why the two key families cannot be acquired in separate calls.
    pub async fn lock_entity_ids(
        &self,
        entity_ids: &[String],
    ) -> Vec<tokio::sync::MutexGuard<'_, ()>> {
        self.lock_ingest_unit(entity_ids, &[]).await
    }

    /// Acquire the per-content RMW striped locks for `content_ids`
    /// (SHA-256 hex), in a deadlock-free order, returning the held guards.
    ///
    /// The `:ArtifactContent` merge is the same check-then-create hazard as
    /// entity dedup: without a guard two concurrent ingests of identical
    /// bytes both read "absent" and both CREATE a duplicate row. A caller
    /// driving the merge inside its own transaction (via
    /// [`merge_artifact_content_in_tx`](Self::merge_artifact_content_in_tx))
    /// must hold these across BOTH the existence read AND the commit.
    ///
    /// A caller that also upserts entities in the same transaction must use
    /// [`lock_ingest_unit`](Self::lock_ingest_unit) instead.
    pub async fn lock_content_ids(
        &self,
        content_ids: &[String],
    ) -> Vec<tokio::sync::MutexGuard<'_, ()>> {
        self.lock_ingest_unit(&[], content_ids).await
    }

    /// Acquire, in ONE deadlock-free pass, every `rmw_locks` guard an
    /// ingest unit needs: the per-entity locks for its merged entity batch
    /// plus the per-content locks for every artifact it will write.
    ///
    /// A unit opens one transaction and commits once, so it must take
    /// exactly one `rmw_locks` acquisition. [`StripedLocks::lock_many`]
    /// dedups by stripe index and acquires ascending, but only *within a
    /// single call* — so two sequential calls (entities, then content) can
    /// self-deadlock whenever an entity key and a content key land on the
    /// same non-reentrant stripe, and can deadlock AB/BA against a
    /// concurrent unit acquiring them in the other order. That is issue
    /// #36's failure mode with a new pair of key families. Passing the
    /// union here is safe however large it is.
    ///
    /// Callers must NOT hold these around
    /// [`lock_session_setup`](Self::lock_session_setup) — setup locks
    /// first and fully released, then RMW (see that method's contract).
    pub async fn lock_ingest_unit(
        &self,
        entity_ids: &[String],
        content_ids: &[String],
    ) -> Vec<tokio::sync::MutexGuard<'_, ()>> {
        let keys: Vec<Vec<u8>> = entity_ids
            .iter()
            .map(|id| crate::locks::entity_lock_key(id))
            .chain(
                content_ids
                    .iter()
                    .map(|id| crate::locks::content_lock_key(id)),
            )
            .collect();
        // `lock_many` dedups by stripe index (not just by key bytes): two
        // distinct keys — including one entity and one content key — can
        // hash to the same stripe, and acquiring that non-reentrant stripe
        // twice would self-deadlock.
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
        // Stop loading models nobody will use; a no-op once it finished.
        if let Some(task) = &self.prefetch_task {
            task.abort();
        }
        let db = Arc::try_unwrap(self.db).map_err(|_| {
            UnikoError::Internal("cannot shutdown: outstanding references exist".into())
        })?;
        db.shutdown().await?;
        // The store is closed: free its lease for the next opener.
        if let Some(claim) = &self.open_path {
            claim.release_lease().await;
        }
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
pub(crate) fn load_catalog(
    config: &UnikoConfig,
    extra: &[ModelAliasSpec],
) -> Result<Vec<ModelAliasSpec>> {
    let mut catalog = if let Some(path) = &config.catalog_path {
        uni_db::xervo_catalog_from_file(path)
            .map_err(|e| UnikoError::Config(format!("catalog {}: {e}", path.display())))?
    } else {
        embed_catalog(config)
    };
    catalog.extend_from_slice(extra);
    reject_duplicate_aliases(&catalog)?;
    Ok(catalog)
}

/// Fail on the first alias that appears twice, naming it.
///
/// uni-xervo rejects duplicates too, but its builder error does not say
/// which alias collided, and the usual cause is an extra spec reusing a
/// built-in name such as `embed/default`.
fn reject_duplicate_aliases(catalog: &[ModelAliasSpec]) -> Result<()> {
    let mut seen = std::collections::HashSet::with_capacity(catalog.len());
    for spec in catalog {
        if !seen.insert(spec.alias.as_str()) {
            return Err(UnikoError::Config(format!(
                "model alias '{}' is defined more than once in the catalog",
                spec.alias
            )));
        }
    }
    Ok(())
}

/// Open the store, turning uni-db's catalog check into a compatibility
/// report.
///
/// uni-db refuses at open a store whose vector indexes name a model alias
/// the catalog lacks: a hybrid store (`embed/hybrid`) opened with a
/// dense-only config. That would surface as a raw "missing alias" error;
/// inspecting the store explains it as [`UnikoError::IncompatibleStore`]
/// instead. Any other open error, or one inspection doesn't explain, is
/// returned as is.
async fn build_or_explain(
    builder: uni_db::UniBuilder,
    path: &Path,
    config: &UnikoConfig,
) -> Result<Uni> {
    match builder.build().await {
        Ok(db) => Ok(db),
        Err(e) => {
            let raw = UnikoError::from(e);
            let text = raw.to_string();
            if config.schema_path.is_none()
                && (text.contains("Missing Uni-Xervo alias")
                    || text.contains("catalog is required"))
                && let Ok((_, verdict)) = migrate_store::inspect_unclaimed(path, config).await
                && !verdict.is_compatible()
            {
                return Err(UnikoError::IncompatibleStore(Box::new(verdict)));
            }
            Err(raw)
        }
    }
}

/// Check the store against `config`, then apply the schema from a JSON file
/// or the builder-based registration.
///
/// The check runs first, on the schema already on disk: a config that the
/// store can't serve (another model or dimensions, a hybrid store opened
/// dense-only) or that needs a migration (new hybrid columns, a different
/// index) is refused with [`UnikoError::IncompatibleStore`] before uni-db
/// changes or rebuilds anything. A `schema_path` config skips the check:
/// its column types come from the file, not the config.
async fn apply_schema(db: &Uni, config: &UnikoConfig) -> Result<()> {
    if config.schema_path.is_none() {
        let stored = compatibility::inspect_db(db, false).await?;
        let verdict = compatibility::check(&stored, config);
        if !verdict.is_compatible() {
            return Err(UnikoError::IncompatibleStore(Box::new(verdict)));
        }
        // A cut-off hybrid backfill opens: the data is correct, some rows
        // just lack the new vectors until `enable_hybrid` is rerun.
        if stored.migration_in_progress == Some(compatibility::Migration::EnableHybrid)
            && !stored.migration_in_schema
        {
            tracing::warn!(
                "an enable_hybrid migration was cut off: rows it didn't reach take no part in \
                 sparse or ColBERT recall until it is rerun (store_info().hybrid_backfill_pending \
                 counts them)"
            );
        }
    }
    apply_schema_unchecked(db, config).await
}

/// Apply the schema without the compatibility check, for a migration that
/// has already checked it.
pub(crate) async fn apply_schema_unchecked(db: &Uni, config: &UnikoConfig) -> Result<()> {
    if let Some(path) = &config.schema_path {
        db.load_schema(path)
            .await
            .map_err(|e| UnikoError::Schema(e.to_string()))
    } else {
        register_schema(db, config).await
    }
}

/// The `local/onnx` loader options for the `nlp/default` alias.
///
/// `generation` is always pinned. uni-xervo >= 0.19 serves two kniv
/// generations and, absent this key, detects one from the repo. Before
/// 0.19.2 that probe read the v1 cascade repo as v7, because it also ships
/// `onnx/onnx_meta.json`; 0.19.2 reads the file's contents instead. Naming
/// the generation still outranks detection (uni-xervo `detect_generation`),
/// and keeps the loader options below matched to the model the config names.
///
/// The two generations take disjoint keys, and uni-xervo rejects a mix:
/// v1 names its one graph, tokenizer, label maps and 128-subword cap; v7
/// reads its graph set and window size from `onnx/onnx_meta.json` and takes
/// only a precision `variant`.
fn build_nlp_options(nlp: &NlpConfig, eps: &[String]) -> serde_json::Value {
    match nlp.generation {
        NlpGeneration::V1 => serde_json::json!({
            "generation": "v1",
            "onnx_path": nlp.artifact,
            "tokenizer_path": "tokenizer.json",
            "label_maps_path": "label_maps.json",
            "max_seq_len": 128,
            "execution_providers": eps,
        }),
        NlpGeneration::V7 => serde_json::json!({
            "generation": "v7",
            "variant": nlp.variant,
            "execution_providers": eps,
        }),
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
            options: build_nlp_options(&config.nlp, &nlp_eps),
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
        let mut ocr_options = serde_json::json!({
            "onnx_path": config.ocr.rec_artifact,
            "char_dict_path": config.ocr.char_dict_path,
            "det_onnx_path": config.ocr.det_artifact,
            "image_height": config.ocr.image_height,
            "image_width": config.ocr.image_width,
            "normalization": config.ocr.normalization,
            "blank_class": 0,
            "execution_providers": ocr_eps,
        });
        if let Some(max) = config.ocr.max_image_width {
            ocr_options["max_image_width"] = max.into();
        }
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
            options: ocr_options,
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

#[cfg(test)]
mod catalog_tests {
    use super::*;

    #[test]
    fn load_catalog_names_an_alias_defined_twice() {
        let config = UnikoConfig::default();
        let mut extra = embed_catalog(&config)
            .into_iter()
            .find(|s| s.alias == EMBED_ALIAS)
            .expect("default catalog has an embedding alias");
        extra.model_id = "some/other-model".into();

        let err = load_catalog(&config, &[extra]).unwrap_err();
        assert!(
            matches!(&err, UnikoError::Config(m) if m.contains(EMBED_ALIAS)),
            "{err}"
        );
    }

    #[test]
    fn load_catalog_keeps_distinct_extra_aliases() {
        let config = UnikoConfig::default();
        let mut extra = embed_catalog(&config).remove(0);
        extra.alias = "embed/extra".into();

        let catalog = load_catalog(&config, &[extra]).unwrap();
        assert_eq!(catalog.len(), embed_catalog(&config).len() + 1);
        assert_eq!(catalog.last().unwrap().alias, "embed/extra");
    }
}

#[cfg(test)]
mod open_path_tests {
    use super::*;
    use std::collections::HashMap;
    use uni_db::Value;

    #[test]
    fn a_path_takes_one_claim_at_a_time() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("kb");
        // Claimed before the directory exists, which the claim creates for
        // its lock file: same key either way.
        let open = OpenPath::claim(&path).unwrap();
        assert!(path.join(LOCK_FILE).exists());
        let err = OpenPath::claim_exclusive(&path).unwrap_err();
        assert!(err.to_string().contains("open in this process"), "{err}");
        let err = OpenPath::claim(&path).unwrap_err();
        assert!(err.to_string().contains("already open"), "{err}");
        drop(open);

        let exclusive = OpenPath::claim_exclusive(&path).unwrap();
        let err = OpenPath::claim(&path).unwrap_err();
        assert!(err.to_string().contains("inspected or migrated"), "{err}");
        assert!(OpenPath::claim_exclusive(&path).is_err());
        drop(exclusive);
        drop(OpenPath::claim(&path).unwrap());
        drop(OpenPath::claim_exclusive(&path).unwrap());
    }

    /// The OS lock refuses a second holder of the lock file, as another
    /// process would be: here, a second file handle taking it directly.
    #[test]
    fn the_os_lock_refuses_another_holder() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("kb");
        let open = OpenPath::claim(&path).unwrap();
        let other = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path.join(LOCK_FILE))
            .unwrap();
        assert!(matches!(
            other.try_lock(),
            Err(std::fs::TryLockError::WouldBlock)
        ));
        drop(open);
        other.try_lock().expect("released with the claim");
        drop(other);
        let err = os_lock_held_elsewhere(&path);
        assert!(err.contains("another process"), "{err}");
    }

    /// The error a claim gives while another handle holds the OS lock.
    fn os_lock_held_elsewhere(path: &Path) -> String {
        let holder = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path.join(LOCK_FILE))
            .unwrap();
        holder.try_lock().unwrap();
        let err = OpenPath::claim(path).unwrap_err().to_string();
        drop(holder);
        err
    }

    /// A store whose lease another process took over refuses every write:
    /// the lease is lost on its next renewal, and each transaction checks.
    #[tokio::test]
    async fn a_lost_lease_refuses_writes() {
        use object_store::ObjectStore;
        use object_store::memory::InMemory;
        let objects: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let prefix = object_store::path::Path::from("bucket/kb");
        let held = lease::RemoteLease::acquire(
            Arc::clone(&objects),
            &prefix,
            std::time::Duration::from_millis(50),
            std::time::Duration::from_millis(200),
        )
        .await
        .unwrap();
        let Ok(mut kb) = KnowledgeBase::in_memory(crate::config::UnikoConfig::default()).await
        else {
            return eprintln!("skipping: store unavailable");
        };
        kb.open_path = Some(Arc::new(OpenPath {
            key: std::path::PathBuf::from("lease-test"),
            exclusive: false,
            _lock: None,
            lease: Some(held),
        }));
        let props = HashMap::from([
            ("topic_id".to_string(), Value::String("t1".into())),
            ("name".to_string(), Value::String("t1".into())),
        ]);
        kb.create_node("Topic", &props)
            .await
            .expect("writes go through while the lease is held");

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let taker = lease::RemoteLease::acquire(
            Arc::clone(&objects),
            &prefix,
            lease::LEASE_TTL,
            lease::RENEW_EVERY,
        )
        .await
        .expect("an expired lease is taken over");
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        let props = HashMap::from([
            ("topic_id".to_string(), Value::String("t2".into())),
            ("name".to_string(), Value::String("t2".into())),
        ]);
        let err = kb.create_node("Topic", &props).await.unwrap_err();
        assert!(err.to_string().contains("lease was taken over"), "{err}");
        assert!(kb.begin_tx().await.is_err());
        taker.release().await;
    }

    #[test]
    fn a_file_url_locks_its_directory() {
        let dir = tempfile::tempdir().unwrap();
        let url = url::Url::from_directory_path(dir.path().join("kb")).unwrap();
        let lock = os_lock(Path::new(url.as_str()), true).unwrap();
        assert!(lock.is_some(), "a file:// URI is a local directory");
        assert!(dir.path().join("kb").join(LOCK_FILE).exists());
        assert!(
            os_lock(Path::new("s3://bucket/kb"), true)
                .unwrap()
                .is_none()
        );
    }

    /// Inspecting a directory that isn't a store leaves it untouched.
    #[test]
    fn an_exclusive_claim_on_a_non_store_dir_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        drop(OpenPath::claim_exclusive(dir.path()).unwrap());
        assert!(!dir.path().join(LOCK_FILE).exists());
        let missing = dir.path().join("nothing-here");
        drop(OpenPath::claim_exclusive(&missing).unwrap());
        assert!(!missing.exists());
    }
}
