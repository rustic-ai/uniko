//! Inspecting a store at a path, and the in-place migrations #66 supports.

use std::path::Path;
use std::time::{Duration, Instant};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uni_db::{ModelAliasSpec, Uni, Value};

use super::compatibility::{self, Compatibility, HYBRID_LABELS, Migration, StoreInfo};
use super::{
    KnowledgeBase, apply_perf_knobs_from_env, apply_schema_unchecked, embed_catalog, load_catalog,
};
use crate::config::UnikoConfig;
use crate::error::{Result, UnikoError};
use crate::schema::{EMBED_ALIAS, HYBRID_EMBED_ALIAS};
use crate::types::datetime_value;

/// Where a migration is, reported to its progress callback.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct MigrationProgress {
    /// The migration running.
    pub migration: Migration,
    /// `schema` (adding columns, rebuilding indexes), `backfill` (filling
    /// the new columns for existing rows) or `done`.
    pub phase: String,
    /// The label being backfilled, during `backfill`.
    pub label: Option<String>,
    /// Rows done so far in this label, or in total at `done`.
    pub done: u64,
    /// Rows to do in this label when the backfill started.
    pub total: u64,
}

/// What [`KnowledgeBase::migrate`] did.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct MigrationReport {
    /// The migration run.
    pub migration: Migration,
    /// Existing rows whose new columns were filled.
    pub rows_backfilled: u64,
    /// Rows the backfill re-embedded whose new columns still came back
    /// empty, for example a text the model returns no sparse terms for.
    /// They are passed over, so the run ends; they still count as pending,
    /// so a rerun tries them again.
    pub rows_unfilled: u64,
    /// The store needs nothing further for this config.
    pub completed: bool,
    /// A migration still marked unfinished on the store, when there is
    /// one: the reason `completed` is false after a run that did its own
    /// part (rerun that migration to finish it).
    pub migration_in_progress: Option<Migration>,
    /// How long it took.
    pub elapsed: Duration,
}

/// Texts re-embedded per backfill transaction when the config gives none.
const DEFAULT_BACKFILL_BATCH: usize = 64;

impl KnowledgeBase {
    /// What the store at `path` was built with, and how `config` compares.
    ///
    /// Reads the stored schema and the stats row. It applies no schema,
    /// changes no index or data, and loads no model; uni-db's own open and
    /// close may still write bookkeeping files (a manifest, its WAL). A
    /// local path that isn't a uni-db store (missing, empty, or some other
    /// directory) reports `exists: false` and no differences, and isn't
    /// touched at all.
    ///
    /// # Errors
    ///
    /// [`UnikoError::Config`] when a store at `path` is open, inspected or
    /// migrated in this process (inspect an open store through its handle's
    /// [`store_info`](Self::store_info) instead);
    /// [`UnikoError::Unsupported`] for a remote URI; or
    /// [`UnikoError::Storage`] when the store can't be read.
    pub async fn inspect(
        path: impl AsRef<Path>,
        config: &UnikoConfig,
    ) -> Result<(StoreInfo, Compatibility)> {
        let path = path.as_ref();
        reject_remote(path)?;
        if !is_store(path) {
            return Ok((StoreInfo::default(), Compatibility::default()));
        }
        let _claim = super::OpenPath::claim_exclusive(path)?;
        inspect_unclaimed(path, config).await
    }

    /// What this open store was built with.
    ///
    /// # Errors
    ///
    /// [`UnikoError::Storage`] when the schema or stats row can't be read.
    pub async fn store_info(&self) -> Result<StoreInfo> {
        compatibility::inspect_db(&self.db, true).await
    }

    /// Bring the store at `path` in line with `config` in place, with one
    /// of the supported migrations.
    ///
    /// - [`Migration::EnableHybrid`] adds the sparse and ColBERT columns to
    ///   a store built with the same model and dimensions, then re-embeds
    ///   existing chunks and observations in batches so the new columns are
    ///   filled. The embedding model loads for that.
    /// - [`Migration::RebuildVectorIndexes`] rebuilds the vector indexes for
    ///   a new algorithm, metric, query prefix or batch size. No vector is
    ///   rewritten and no model loads. (A new *document* prefix would change
    ///   what the stored vectors mean, so it needs a fresh store.)
    ///
    /// Nothing runs unless `migration` resolves every difference: a
    /// difference that needs a fresh store, or another migration, is
    /// refused before the store changes. Both migrations are resumable: a
    /// rerun after an interruption picks up the rows still to do, and a
    /// store that already matches reports `completed` at once. While an
    /// `EnableHybrid` backfill is unfinished the store still opens with the
    /// new config; [`StoreInfo::hybrid_backfill_pending`] says how much is
    /// left. `progress` is called after each phase and backfill batch.
    ///
    /// No other handle may have the store open, in this process (refused)
    /// or another one (not detectable: uni-db takes no lock).
    ///
    /// # Errors
    ///
    /// [`UnikoError::IncompatibleStore`] when `migration` can't resolve the
    /// differences, [`UnikoError::Config`] when there is no store at `path`,
    /// it is open in this process, or the config is invalid, and the
    /// storage or embedding error that stopped a batch; the rows already
    /// done stay done.
    pub async fn migrate(
        path: impl AsRef<Path>,
        config: UnikoConfig,
        migration: Migration,
        progress: &mut (dyn FnMut(&MigrationProgress) + Send),
    ) -> Result<MigrationReport> {
        let started = Instant::now();
        config.validate()?;
        let path = path.as_ref();
        reject_remote(path)?;
        if config.schema_path.is_some() {
            return Err(UnikoError::Config(
                "migrate can't run with schema_path set: the schema would come from that file, \
                 not from the config being migrated to"
                    .into(),
            ));
        }
        if !is_store(path) {
            return Err(UnikoError::Config(format!(
                "no store to migrate at {}",
                path.display()
            )));
        }
        let _claim = super::OpenPath::claim_exclusive(path)?;
        let catalog = inspection_catalog(&config, load_catalog(&config, &[])?);
        let db = open_raw(path, catalog).await?;
        let outcome = migrate_db(&db, &config, migration, progress).await;
        let closed = db.shutdown().await;
        let Backfill {
            filled: rows_backfilled,
            unfilled: rows_unfilled,
        } = outcome?;
        closed?;
        // The migration itself succeeded; if the store can't be re-read to
        // confirm it, report it as not known to be complete rather than
        // failing a run that did its work.
        let (completed, migration_in_progress) = match inspect_unclaimed(path, &config).await {
            Ok((info, verdict)) => (
                // Rows left unfilled were tried and can't be filled: they
                // don't keep the migration from being complete.
                verdict.is_compatible()
                    && info.hybrid_backfill_pending.unwrap_or(0) <= rows_unfilled
                    && info.migration_in_progress.is_none(),
                info.migration_in_progress,
            ),
            Err(e) => {
                tracing::warn!(error = %e, "could not re-read the store after migrating");
                (false, None)
            }
        };
        let rows = rows_backfilled + rows_unfilled;
        progress(&MigrationProgress {
            migration,
            phase: "done".into(),
            label: None,
            done: rows,
            total: rows,
        });
        Ok(MigrationReport {
            migration,
            rows_backfilled,
            rows_unfilled,
            completed,
            migration_in_progress,
            elapsed: started.elapsed(),
        })
    }
}

/// `catalog` plus whichever embed alias it lacks, so uni-db opens a store
/// whose indexes name either one: it checks at open that the catalog serves
/// every alias the stored indexes use, so a hybrid store couldn't otherwise
/// be opened to be inspected with a dense-only catalog.
///
/// A missing `embed/hybrid` is added as a `local/onnx` stub: providers such
/// as `remote/openai` reject the hybrid task when the runtime is built, and
/// the stub only has to exist, since every entry is lazy and inspection
/// embeds nothing.
pub(crate) fn inspection_catalog(
    config: &UnikoConfig,
    mut catalog: Vec<ModelAliasSpec>,
) -> Vec<ModelAliasSpec> {
    if catalog.is_empty() {
        catalog = embed_catalog(config)
            .into_iter()
            .filter(|s| s.alias == EMBED_ALIAS)
            .collect();
    }
    if !catalog.iter().any(|c| c.alias == HYBRID_EMBED_ALIAS) {
        let mut hybrid = config.clone();
        // A bare stub: the config's provider options belong to its own
        // provider (llama.cpp's `base_url`, say) and `local/onnx` rejects
        // options it doesn't know.
        hybrid.embedding.provider = "local/onnx".to_string();
        hybrid.embedding.provider_options = None;
        hybrid.embedding.execution_providers = None;
        hybrid
            .embedding
            .sparse_dimensions
            .get_or_insert(hybrid.embedding.dimensions);
        if let Some(stub) = embed_catalog(&hybrid)
            .into_iter()
            .find(|s| s.alias == HYBRID_EMBED_ALIAS)
        {
            catalog.push(stub);
        }
    }
    catalog
}

/// [`KnowledgeBase::inspect`] without the exclusive claim, for callers that
/// hold one already or are opening the store themselves.
pub(crate) async fn inspect_unclaimed(
    path: &Path,
    config: &UnikoConfig,
) -> Result<(StoreInfo, Compatibility)> {
    if !is_store(path) {
        return Ok((StoreInfo::default(), Compatibility::default()));
    }
    let db = open_raw(path, inspection_catalog(config, Vec::new())).await?;
    let info = compatibility::inspect_db(&db, true).await;
    db.shutdown().await?;
    let info = info?;
    let verdict = compatibility::check(&info, config);
    Ok((info, verdict))
}

/// Inspecting and migrating need a local store directory.
fn reject_remote(path: &Path) -> Result<()> {
    if path.to_string_lossy().contains("://") {
        return Err(UnikoError::Unsupported(format!(
            "inspecting or migrating a remote store ({}) isn't supported; use a local path",
            path.display()
        )));
    }
    Ok(())
}

/// Whether `path` holds a uni-db store: its schema catalog exists. Opening
/// anything else would create one.
fn is_store(path: &Path) -> bool {
    path.join("catalog").join("schema.json").is_file() || path.join("schema.json").is_file()
}

/// Open the store at `path` as uni-db alone: no schema applied, and models
/// only from `catalog`, which loads them lazily.
async fn open_raw(path: &Path, catalog: Vec<ModelAliasSpec>) -> Result<Uni> {
    let mut builder = Uni::open_existing(path.to_string_lossy());
    if !catalog.is_empty() {
        builder = builder.xervo_catalog(catalog);
    }
    if let Some(uni_cfg) = apply_perf_knobs_from_env() {
        builder = builder.config(uni_cfg);
    }
    Ok(builder.build().await?)
}

/// What a backfill did: rows whose new columns it filled, and rows it
/// re-embedded that still came back empty.
#[derive(Default)]
struct Backfill {
    filled: u64,
    unfilled: u64,
}

/// Run `migration` on an open raw store.
async fn migrate_db(
    db: &Uni,
    config: &UnikoConfig,
    migration: Migration,
    progress: &mut (dyn FnMut(&MigrationProgress) + Send),
) -> Result<Backfill> {
    let info = compatibility::inspect_db(db, true).await?;
    if !info.exists {
        return Err(UnikoError::Config(
            "no uniko store to migrate at this path".into(),
        ));
    }
    let verdict = compatibility::check(&info, config);
    if !verdict.resolved_by(migration) {
        return Err(UnikoError::IncompatibleStore(Box::new(verdict)));
    }
    let sparse = config.embedding.sparse_dimensions.is_some();
    let colbert = config.embedding.multivector_dimensions.is_some();
    let backfill = migration == Migration::EnableHybrid && (sparse || colbert);
    let mine = info.migration_in_progress == Some(migration);
    if verdict.is_compatible()
        && !mine
        && (!backfill || info.hybrid_backfill_pending.unwrap_or(0) == 0)
    {
        return Ok(Backfill::default());
    }

    progress(&MigrationProgress {
        migration,
        phase: "schema".into(),
        label: None,
        done: 0,
        total: 0,
    });
    // Mark the run, unless another migration's unfinished run is marked:
    // that marker stays until that migration completes. Mark before the
    // schema step (the long index rebuild) when the store already has the
    // marker column; a store from before it gets the column from that step.
    let marks = info.migration_in_progress.is_none() || mine;
    let has_marker_column = db
        .schema()
        .current()
        .properties
        .get(crate::schema::constants::labels::KNOWLEDGE_BASE_STATS)
        .is_some_and(|p| p.contains_key("migration"));
    if marks && has_marker_column {
        set_marker(db, Some(migration)).await?;
    }
    // uni-db adds the new columns and indexes, and rebuilds every vector
    // index whose config changed, before `apply` returns.
    apply_schema_unchecked(db, config).await?;
    if marks && !has_marker_column {
        set_marker(db, Some(migration)).await?;
    }

    let mut out = Backfill::default();
    if backfill {
        for (label, text) in HYBRID_LABELS {
            let b = backfill_label(
                db,
                config,
                migration,
                label,
                text,
                (sparse, colbert),
                progress,
            )
            .await?;
            out.filled += b.filled;
            out.unfilled += b.unfilled;
        }
    }
    // Record the new identity, then clear the marker: the migration is done.
    set_identity(db, config).await?;
    if marks {
        set_marker(db, None).await?;
    }
    Ok(out)
}

/// Re-embed the `label` rows whose hybrid columns aren't filled, a batch
/// per transaction, walking them once in node-id order. Rewriting the
/// embed source property makes uni-db recompute every auto-embed column of
/// the row, the new sparse and ColBERT ones included. A row that still
/// comes back empty is counted as unfilled and passed, so the walk always
/// ends.
async fn backfill_label(
    db: &Uni,
    config: &UnikoConfig,
    migration: Migration,
    label: &str,
    text: &str,
    (sparse, colbert): (bool, bool),
    progress: &mut (dyn FnMut(&MigrationProgress) + Send),
) -> Result<Backfill> {
    let Some(pending) = compatibility::pending_predicate("n", text, sparse, colbert) else {
        return Ok(Backfill::default());
    };
    let total = count(
        db,
        &format!("MATCH (n:{label}) WHERE {pending} RETURN count(n) AS c"),
    )
    .await?;
    let batch = if config.embedding.batch_size > 0 {
        config.embedding.batch_size
    } else {
        DEFAULT_BACKFILL_BATCH
    };
    // Node ids are unsigned, so the first page has no lower bound.
    let after_clause =
        |after: Option<i64>| after.map_or(String::new(), |a| format!("id(n) > {a} AND "));
    let mut out = Backfill::default();
    let mut after: Option<i64> = None;
    loop {
        let step = format!(
            "MATCH (n:{label}) WHERE {}{pending} \
             WITH n ORDER BY id(n) LIMIT {batch} \
             SET n.{text} = n.{text} \
             RETURN max(id(n)) AS last, count(n) AS c",
            after_clause(after)
        );
        let tx = db.session().tx().await?;
        let rows = tx.query(&step).await?;
        let row = rows.rows().first();
        let n = row
            .and_then(|r| r.get::<i64>("c").ok())
            .map_or(0, |n| u64::try_from(n).unwrap_or(0));
        let last = row.and_then(|r| r.get::<i64>("last").ok());
        tx.commit().await?;
        let Some(last) = last.filter(|_| n > 0) else {
            break;
        };
        // Of the rows just rewritten, the ones still pending couldn't be
        // filled; they are behind the cursor now and won't be retried.
        let still = count(
            db,
            &format!(
                "MATCH (n:{label}) WHERE {}id(n) <= {last} AND {pending} \
                 RETURN count(n) AS c",
                after_clause(after)
            ),
        )
        .await?;
        out.unfilled += still;
        out.filled += n - still.min(n);
        after = Some(last);
        progress(&MigrationProgress {
            migration,
            phase: "backfill".into(),
            label: Some(label.to_string()),
            done: out.filled + out.unfilled,
            total,
        });
    }
    Ok(out)
}

async fn count(db: &Uni, cypher: &str) -> Result<u64> {
    let rows = db.session().query(cypher).await?;
    Ok(rows
        .rows()
        .first()
        .and_then(|r| r.get::<i64>("c").ok())
        .map_or(0, |n| u64::try_from(n).unwrap_or(0)))
}

async fn set_marker(db: &Uni, migration: Option<Migration>) -> Result<()> {
    let value = match migration {
        Some(m) => Value::Map(
            [
                ("kind".to_string(), Value::String(m.as_str().to_string())),
                ("started_at".to_string(), datetime_value(Utc::now())),
            ]
            .into_iter()
            .collect(),
        ),
        None => Value::Null,
    };
    let tx = db.session().tx().await?;
    tx.query_with("MATCH (s:KnowledgeBaseStats) SET s.migration = $m, s.updated_at = $now")
        .param("m", value)
        .param("now", datetime_value(Utc::now()))
        .fetch_all()
        .await?;
    tx.commit().await?;
    Ok(())
}

async fn set_identity(db: &Uni, config: &UnikoConfig) -> Result<()> {
    let tx = db.session().tx().await?;
    tx.query_with("MATCH (s:KnowledgeBaseStats) SET s.embedding = $e, s.updated_at = $now")
        .param("e", compatibility::identity_map(config))
        .param("now", datetime_value(Utc::now()))
        .fetch_all()
        .await?;
    tx.commit().await?;
    Ok(())
}
