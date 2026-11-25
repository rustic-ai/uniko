//! Store compatibility inspection, the open-time preflight and in-place
//! migrations (#66). Model-free: stores open with a lazy catalog and hold no
//! rows to embed, so no model loads.

use std::path::Path;

use uniko_store::config::{EmbeddingConfig, UnikoConfig, VectorMetricChoice};
use uniko_store::{
    KnowledgeBase, Migration, MigrationProgress, PrefetchPolicy, UnikoError, Value, Verdict,
};

async fn open(path: &Path, config: UnikoConfig) -> uniko_store::Result<KnowledgeBase> {
    KnowledgeBase::open_with_prefetch(path, config, Vec::new(), PrefetchPolicy::Lazy).await
}

async fn create(path: &Path, config: UnikoConfig) {
    open(path, config)
        .await
        .expect("create")
        .shutdown()
        .await
        .unwrap();
}

fn bge_m3() -> UnikoConfig {
    UnikoConfig {
        embedding: EmbeddingConfig::bge_m3(),
        ..UnikoConfig::default()
    }
}

fn bge_m3_dense_only() -> UnikoConfig {
    let mut c = bge_m3();
    c.embedding.sparse_dimensions = None;
    c.embedding.multivector_dimensions = None;
    c
}

fn differences(err: &UnikoError) -> Vec<(String, Verdict)> {
    match err {
        UnikoError::IncompatibleStore(c) => c
            .differences
            .iter()
            .map(|d| (d.field.clone(), d.verdict))
            .collect(),
        other => panic!("expected IncompatibleStore, got {other}"),
    }
}

fn no_progress() -> impl FnMut(&MigrationProgress) + Send {
    |_| {}
}

#[tokio::test]
async fn inspect_reports_what_a_store_was_built_with() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("nothing-here");
    let (info, verdict) = KnowledgeBase::inspect(&missing, &UnikoConfig::default())
        .await
        .unwrap();
    assert!(!info.exists && verdict.is_compatible());
    assert!(!missing.exists(), "inspect must not create a store");

    let path = dir.path().join("kb");
    let config = UnikoConfig::default();
    create(&path, config.clone()).await;
    let (info, verdict) = KnowledgeBase::inspect(&path, &config).await.unwrap();
    assert!(info.exists && verdict.is_compatible(), "{verdict}");
    let e = &info.embedding;
    assert_eq!(
        e.model_id.as_deref(),
        Some(config.embedding.model_id.as_str())
    );
    assert_eq!(e.dimensions, Some(config.embedding.dimensions));
    assert_eq!(e.metric.as_deref(), Some("cosine"));
    assert_eq!(e.embed_alias.as_deref(), Some("embed/default"));
    assert!(!e.is_hybrid() && info.hybrid_backfill_pending.is_none());

    let (_, verdict) = KnowledgeBase::inspect(&path, &bge_m3()).await.unwrap();
    assert!(verdict.has_incompatible());
}

/// Another dimension is refused before uni-db sees the schema, with
/// uniko's error, and the store is left as it was.
#[tokio::test]
async fn a_dimension_change_is_refused_and_changes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kb");
    create(&path, UnikoConfig::default()).await;

    let err = open(&path, bge_m3()).await.err().expect("refused");
    let diffs = differences(&err);
    assert!(
        diffs.contains(&("dimensions".into(), Verdict::Incompatible)),
        "{diffs:?}"
    );
    // uniko's own error, naming the stored and requested sizes, not uni-db's
    // "property types are immutable".
    assert!(
        err.to_string()
            .starts_with("incompatible store: dimensions: store has 384, config asks for 1024"),
        "{err}"
    );

    let err = KnowledgeBase::migrate(&path, bge_m3(), Migration::EnableHybrid, &mut no_progress())
        .await
        .err()
        .expect("no migration can change dimensions");
    assert!(matches!(err, UnikoError::IncompatibleStore(_)), "{err}");

    let (info, _) = KnowledgeBase::inspect(&path, &UnikoConfig::default())
        .await
        .unwrap();
    assert_eq!(info.embedding.dimensions, Some(384));
    open(&path, UnikoConfig::default())
        .await
        .expect("the original config still opens")
        .shutdown()
        .await
        .unwrap();
}

/// Same model, dense-only to hybrid: a plain open is refused without
/// adding columns; the migration adds them and the store then opens.
#[tokio::test]
async fn enabling_hybrid_needs_the_migration() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kb");
    create(&path, bge_m3_dense_only()).await;

    let err = open(&path, bge_m3()).await.err().expect("refused");
    assert!(
        differences(&err)
            .iter()
            .all(|(_, v)| *v == Verdict::NeedsMigration(Migration::EnableHybrid)),
        "{err}"
    );
    let (info, _) = KnowledgeBase::inspect(&path, &bge_m3()).await.unwrap();
    assert!(!info.embedding.is_hybrid(), "a refused open added columns");

    let err = KnowledgeBase::migrate(
        &path,
        bge_m3(),
        Migration::RebuildVectorIndexes,
        &mut no_progress(),
    )
    .await
    .err()
    .expect("an index rebuild doesn't add columns");
    assert!(matches!(err, UnikoError::IncompatibleStore(_)));

    let mut phases = Vec::new();
    let report = KnowledgeBase::migrate(&path, bge_m3(), Migration::EnableHybrid, &mut |p| {
        phases.push(p.phase.clone());
    })
    .await
    .expect("migrate");
    assert!(report.completed, "{report:?}");
    assert_eq!(report.rows_backfilled, 0);
    assert_eq!(phases.first().map(String::as_str), Some("schema"));
    assert_eq!(phases.last().map(String::as_str), Some("done"));

    let (info, verdict) = KnowledgeBase::inspect(&path, &bge_m3()).await.unwrap();
    assert!(verdict.is_compatible(), "{verdict}");
    assert_eq!(info.embedding.sparse_dimensions, Some(250_002));
    assert_eq!(info.embedding.multivector_dimensions, Some(1024));
    assert_eq!(info.embedding.embed_alias.as_deref(), Some("embed/hybrid"));
    assert!(info.migration_in_progress.is_none());
    open(&path, bge_m3())
        .await
        .expect("opens as hybrid")
        .shutdown()
        .await
        .unwrap();

    // Running it again is a no-op.
    let again =
        KnowledgeBase::migrate(&path, bge_m3(), Migration::EnableHybrid, &mut no_progress())
            .await
            .unwrap();
    assert!(again.completed && again.rows_backfilled == 0);
}

#[tokio::test]
async fn a_hybrid_store_is_not_opened_dense_only() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kb");
    create(&path, bge_m3()).await;
    let err = open(&path, bge_m3_dense_only())
        .await
        .err()
        .expect("refused");
    let diffs = differences(&err);
    assert!(
        diffs.contains(&("sparse_dimensions".into(), Verdict::Incompatible)),
        "{diffs:?}"
    );
    let (info, _) = KnowledgeBase::inspect(&path, &bge_m3()).await.unwrap();
    assert!(
        info.embedding.is_hybrid(),
        "the hybrid columns are still there"
    );
}

/// The AC4 check against current uni-db: a hybrid store reopens with the
/// same config, and reapplying its schema is clean.
#[tokio::test]
async fn a_hybrid_store_reopens() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kb");
    create(&path, bge_m3()).await;
    for _ in 0..2 {
        let kb = open(&path, bge_m3()).await.expect("hybrid reopen");
        let info = kb.store_info().await.unwrap();
        assert!(info.embedding.is_hybrid());
        assert_eq!(info.hybrid_backfill_pending, Some(0));
        kb.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn an_index_change_needs_a_rebuild() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kb");
    create(&path, UnikoConfig::default()).await;
    let dot = UnikoConfig {
        vector_metric: VectorMetricChoice::Dot,
        ..UnikoConfig::default()
    };
    let err = open(&path, dot.clone()).await.err().expect("refused");
    assert_eq!(
        differences(&err),
        vec![(
            "vector_metric".to_string(),
            Verdict::NeedsMigration(Migration::RebuildVectorIndexes)
        )]
    );
    let report = KnowledgeBase::migrate(
        &path,
        dot.clone(),
        Migration::RebuildVectorIndexes,
        &mut no_progress(),
    )
    .await
    .unwrap();
    assert!(report.completed);
    let kb = open(&path, dot).await.expect("opens after the rebuild");
    assert_eq!(
        kb.store_info().await.unwrap().embedding.metric.as_deref(),
        Some("dot")
    );
    kb.shutdown().await.unwrap();
    let err = open(&path, UnikoConfig::default())
        .await
        .err()
        .expect("the old metric no longer fits");
    assert_eq!(
        differences(&err),
        vec![(
            "vector_metric".to_string(),
            Verdict::NeedsMigration(Migration::RebuildVectorIndexes)
        )]
    );
}

/// A stats row with only `{model_id, dimensions}` (written before the full
/// identity was kept) still checks out, and the record is completed.
#[tokio::test]
async fn an_older_identity_record_is_completed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kb");
    let config = UnikoConfig::default();
    {
        let kb = open(&path, config.clone()).await.unwrap();
        let tx = kb.db().session().tx().await.unwrap();
        tx.query_with("MATCH (s:KnowledgeBaseStats) SET s.embedding = $e")
            .param(
                "e",
                Value::Map(
                    [
                        (
                            "model_id".to_string(),
                            Value::String(config.embedding.model_id.clone()),
                        ),
                        ("dimensions".to_string(), Value::Int(384)),
                    ]
                    .into_iter()
                    .collect(),
                ),
            )
            .fetch_all()
            .await
            .unwrap();
        tx.commit().await.unwrap();
        kb.shutdown().await.unwrap();
    }
    let kb = open(&path, config).await.expect("reopens");
    let rows = kb
        .db()
        .session()
        .query("MATCH (s:KnowledgeBaseStats) RETURN s.embedding AS e")
        .await
        .unwrap();
    match rows.rows()[0].value("e") {
        Some(Value::Map(m)) => {
            assert!(
                m.contains_key("metric") && m.contains_key("algorithm"),
                "{m:?}"
            );
        }
        other => panic!("{other:?}"),
    }
    kb.shutdown().await.unwrap();
}

/// A remote embedder can't serve the hybrid task; inspecting or migrating
/// its store must still work (the stub hybrid alias is local and lazy).
#[tokio::test]
async fn a_remote_provider_store_can_be_inspected_and_migrated() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kb");
    let mut remote = UnikoConfig::default();
    remote.embedding.provider = "remote/openai".into();
    remote.embedding.model_id = "text-embedding-3-small".into();
    remote.embedding.provider_options = None;
    create(&path, remote.clone()).await;
    let (info, verdict) = KnowledgeBase::inspect(&path, &remote)
        .await
        .expect("inspect");
    assert!(info.exists && verdict.is_compatible(), "{verdict}");

    let mut dot = remote.clone();
    dot.vector_metric = VectorMetricChoice::Dot;
    let report = KnowledgeBase::migrate(
        &path,
        dot.clone(),
        Migration::RebuildVectorIndexes,
        &mut no_progress(),
    )
    .await
    .expect("migrate a remote-provider store");
    assert!(report.completed);
    open(&path, dot)
        .await
        .expect("opens")
        .shutdown()
        .await
        .unwrap();
}

/// uni-db resolves an automatic PQ `sub_vectors` (0) when it stores the
/// index; the check resolves it the same way, so the same config reopens.
#[tokio::test]
async fn auto_sub_vectors_reopens() {
    use uniko_store::config::VectorAlgorithm;
    for algo in [
        VectorAlgorithm::HnswPq {
            m: 16,
            ef_construction: 100,
            sub_vectors: 0,
        },
        VectorAlgorithm::IvfPq {
            partitions: 4,
            sub_vectors: 0,
        },
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("kb");
        let config = UnikoConfig {
            vector_algorithm: algo.clone(),
            ..UnikoConfig::default()
        };
        create(&path, config.clone()).await;
        open(&path, config.clone())
            .await
            .unwrap_or_else(|e| panic!("{algo:?} refused its own store: {e}"))
            .shutdown()
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn inspect_leaves_a_non_store_directory_alone() {
    let dir = tempfile::tempdir().unwrap();
    let other = dir.path().join("photos");
    std::fs::create_dir(&other).unwrap();
    std::fs::write(other.join("a.txt"), "not a store").unwrap();
    let (info, verdict) = KnowledgeBase::inspect(&other, &UnikoConfig::default())
        .await
        .unwrap();
    assert!(!info.exists && verdict.is_compatible());
    let entries: Vec<_> = std::fs::read_dir(&other).unwrap().collect();
    assert_eq!(
        entries.len(),
        1,
        "inspect created files in a non-store directory"
    );
    let err = KnowledgeBase::migrate(
        &other,
        UnikoConfig::default(),
        Migration::RebuildVectorIndexes,
        &mut no_progress(),
    )
    .await
    .unwrap_err();
    assert!(matches!(err, UnikoError::Config(_)), "{err}");
}

#[tokio::test]
async fn a_store_open_in_this_process_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kb");
    let kb = open(&path, UnikoConfig::default()).await.unwrap();
    let clone = kb.clone();
    for err in [
        KnowledgeBase::inspect(&path, &UnikoConfig::default())
            .await
            .unwrap_err(),
        KnowledgeBase::migrate(
            &path,
            UnikoConfig::default(),
            Migration::RebuildVectorIndexes,
            &mut no_progress(),
        )
        .await
        .unwrap_err(),
    ] {
        assert!(
            matches!(&err, UnikoError::Config(m) if m.contains("open in this process")),
            "{err}"
        );
    }
    // The open handle reports its own store instead.
    assert!(kb.store_info().await.unwrap().exists);
    drop(clone);
    kb.shutdown().await.unwrap();
    KnowledgeBase::inspect(&path, &UnikoConfig::default())
        .await
        .expect("inspectable once shut down");
}

/// A sparse-only store gaining the ColBERT column goes through the same
/// migration, and the store then opens with both.
#[tokio::test]
async fn a_sparse_only_store_gains_colbert() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kb");
    let mut sparse_only = bge_m3();
    sparse_only.embedding.multivector_dimensions = None;
    create(&path, sparse_only).await;
    let err = open(&path, bge_m3()).await.err().expect("refused");
    assert_eq!(
        differences(&err),
        vec![(
            "multivector_dimensions".to_string(),
            Verdict::NeedsMigration(Migration::EnableHybrid)
        )]
    );
    let report =
        KnowledgeBase::migrate(&path, bge_m3(), Migration::EnableHybrid, &mut no_progress())
            .await
            .unwrap();
    assert!(report.completed);
    let (info, verdict) = KnowledgeBase::inspect(&path, &bge_m3()).await.unwrap();
    assert!(verdict.is_compatible());
    assert_eq!(info.embedding.multivector_dimensions, Some(1024));
}

/// Leave the marker a run of `kind` cut off in `phase` (`schema` or
/// `backfill`) would.
async fn set_marker(path: &Path, config: &UnikoConfig, kind: &str, phase: &str) {
    let kb = open(path, config.clone()).await.unwrap();
    let tx = kb.db().session().tx().await.unwrap();
    tx.query_with("MATCH (s:KnowledgeBaseStats) SET s.migration = $m")
        .param(
            "m",
            Value::Map(
                [
                    ("kind".to_string(), Value::String(kind.into())),
                    ("phase".to_string(), Value::String(phase.into())),
                ]
                .into_iter()
                .collect(),
            ),
        )
        .fetch_all()
        .await
        .unwrap();
    tx.commit().await.unwrap();
    kb.shutdown().await.unwrap();
}

/// A migration interrupted after it set its marker shows as in progress;
/// a rerun finishes it, and another kind of migration leaves it alone.
#[tokio::test]
async fn an_interrupted_migration_is_reported_and_resumed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kb");
    create(&path, bge_m3()).await;
    set_marker(&path, &bge_m3(), "enable_hybrid", "backfill").await;
    let (info, verdict) = KnowledgeBase::inspect(&path, &bge_m3()).await.unwrap();
    assert!(verdict.is_compatible());
    assert_eq!(info.migration_in_progress, Some(Migration::EnableHybrid));

    // A rebuild for a metric change keeps the other migration's marker.
    let mut dot = bge_m3();
    dot.vector_metric = VectorMetricChoice::Dot;
    KnowledgeBase::migrate(
        &path,
        dot.clone(),
        Migration::RebuildVectorIndexes,
        &mut no_progress(),
    )
    .await
    .unwrap();
    let (info, _) = KnowledgeBase::inspect(&path, &dot).await.unwrap();
    assert_eq!(info.migration_in_progress, Some(Migration::EnableHybrid));

    let report = KnowledgeBase::migrate(
        &path,
        dot.clone(),
        Migration::EnableHybrid,
        &mut no_progress(),
    )
    .await
    .unwrap();
    assert!(report.completed, "{report:?}");
    let (info, _) = KnowledgeBase::inspect(&path, &dot).await.unwrap();
    assert_eq!(info.migration_in_progress, None);
}

/// A rebuild over another migration's unfinished backfill still marks
/// itself: cut off in its schema step, the next open is refused until it is
/// rerun, and the rerun puts the backfill's marker back.
#[tokio::test]
async fn a_rebuild_over_an_unfinished_backfill_marks_itself() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kb");
    create(&path, bge_m3()).await;
    set_marker(&path, &bge_m3(), "enable_hybrid", "backfill").await;
    let mut dot = bge_m3();
    dot.vector_metric = VectorMetricChoice::Dot;

    // Cut the run off at the start of its schema step, after it marked
    // itself: what a crash during the index build leaves.
    let (p, c) = (path.clone(), dot.clone());
    let crashed = tokio::spawn(async move {
        KnowledgeBase::migrate(
            &p,
            c,
            Migration::RebuildVectorIndexes,
            &mut |e: &MigrationProgress| assert_ne!(e.phase, "schema", "crash"),
        )
        .await
    })
    .await;
    assert!(crashed.is_err_and(|e| e.is_panic()));

    let (info, verdict) = KnowledgeBase::inspect(&path, &dot).await.unwrap();
    assert_eq!(
        info.migration_in_progress,
        Some(Migration::RebuildVectorIndexes)
    );
    assert!(!verdict.is_compatible());
    let err = open(&path, dot.clone()).await.expect_err("refused");
    assert!(matches!(err, UnikoError::IncompatibleStore(_)), "{err}");

    let report = KnowledgeBase::migrate(
        &path,
        dot.clone(),
        Migration::RebuildVectorIndexes,
        &mut no_progress(),
    )
    .await
    .unwrap();
    assert_eq!(
        report.migration_in_progress,
        Some(Migration::EnableHybrid),
        "{report:?}"
    );
    let (info, verdict) = KnowledgeBase::inspect(&path, &dot).await.unwrap();
    assert_eq!(info.migration_in_progress, Some(Migration::EnableHybrid));
    assert!(verdict.is_compatible(), "{verdict}");
    open(&path, dot)
        .await
        .expect("a cut-off backfill opens")
        .shutdown()
        .await
        .unwrap();
}

/// A cut-off index rebuild refuses the next open until it is rerun, since
/// the index may be only partly built; a cut-off hybrid backfill opens.
#[tokio::test]
async fn an_interrupted_rebuild_refuses_to_open_until_rerun() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kb");
    create(&path, bge_m3()).await;
    set_marker(&path, &bge_m3(), "rebuild_vector_indexes", "schema").await;

    let err = open(&path, bge_m3()).await.expect_err("refused");
    let UnikoError::IncompatibleStore(c) = err else {
        panic!("{err}");
    };
    assert_eq!(c.migrations(), vec![Migration::RebuildVectorIndexes], "{c}");
    assert!(
        c.differences
            .iter()
            .any(|d| d.field == "migration_in_progress"),
        "{c}"
    );
    let (info, verdict) = KnowledgeBase::inspect(&path, &bge_m3()).await.unwrap();
    assert_eq!(
        info.migration_in_progress,
        Some(Migration::RebuildVectorIndexes)
    );
    assert!(!verdict.is_compatible());

    // The index definition already matches, so only a forced rebuild
    // builds anything: the run reports a `rebuild` phase.
    let mut phases = Vec::new();
    let report = KnowledgeBase::migrate(
        &path,
        bge_m3(),
        Migration::RebuildVectorIndexes,
        &mut |p: &MigrationProgress| phases.push(p.phase.clone()),
    )
    .await
    .unwrap();
    assert!(report.completed, "{report:?}");
    assert!(phases.iter().any(|p| p == "rebuild"), "{phases:?}");
    open(&path, bge_m3())
        .await
        .expect("a finished rebuild opens")
        .shutdown()
        .await
        .unwrap();

    set_marker(&path, &bge_m3(), "enable_hybrid", "backfill").await;
    open(&path, bge_m3())
        .await
        .expect("a cut-off backfill still opens")
        .shutdown()
        .await
        .unwrap();
}

/// A cut-off rebuild on a store that now also needs hybrid columns: the one
/// migration to run is `enable_hybrid`, which covers the rebuild, rebuilds
/// the indexes and clears the marker.
#[tokio::test]
async fn enable_hybrid_finishes_a_cut_off_rebuild() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kb");
    create(&path, bge_m3_dense_only()).await;
    set_marker(
        &path,
        &bge_m3_dense_only(),
        "rebuild_vector_indexes",
        "schema",
    )
    .await;
    let (_, verdict) = KnowledgeBase::inspect(&path, &bge_m3()).await.unwrap();
    assert_eq!(
        verdict.migrations(),
        vec![Migration::EnableHybrid],
        "{verdict}"
    );
    let mut phases = Vec::new();
    let report = KnowledgeBase::migrate(
        &path,
        bge_m3(),
        Migration::EnableHybrid,
        &mut |p: &MigrationProgress| phases.push(p.phase.clone()),
    )
    .await
    .unwrap();
    assert!(report.completed, "{report:?}");
    assert!(phases.iter().any(|p| p == "rebuild"), "{phases:?}");
    let (info, _) = KnowledgeBase::inspect(&path, &bge_m3()).await.unwrap();
    assert_eq!(info.migration_in_progress, None);
    open(&path, bge_m3())
        .await
        .expect("opens once enable_hybrid finished")
        .shutdown()
        .await
        .unwrap();
}

/// An `enable_hybrid` cut off in its schema step (index builds included)
/// refuses to open like a cut-off rebuild, and its rerun rebuilds the
/// indexes outright; one cut off in its backfill opens.
#[tokio::test]
async fn an_enable_hybrid_cut_off_in_its_schema_step_refuses_to_open() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kb");
    create(&path, bge_m3()).await;
    set_marker(&path, &bge_m3(), "enable_hybrid", "schema").await;
    let err = open(&path, bge_m3()).await.expect_err("refused");
    let UnikoError::IncompatibleStore(c) = err else {
        panic!("{err}");
    };
    assert_eq!(c.migrations(), vec![Migration::EnableHybrid], "{c}");

    let mut phases = Vec::new();
    let report = KnowledgeBase::migrate(
        &path,
        bge_m3(),
        Migration::EnableHybrid,
        &mut |p: &MigrationProgress| phases.push(p.phase.clone()),
    )
    .await
    .unwrap();
    assert!(report.completed, "{report:?}");
    assert!(phases.iter().any(|p| p == "rebuild"), "{phases:?}");
    open(&path, bge_m3())
        .await
        .expect("opens once rerun")
        .shutdown()
        .await
        .unwrap();
}

/// A marker written before the phase was recorded counts as the schema
/// step: the index state is unknown, so the store refuses to open.
#[tokio::test]
async fn a_marker_with_no_phase_refuses_to_open() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kb");
    create(&path, bge_m3()).await;
    let kb = open(&path, bge_m3()).await.unwrap();
    let tx = kb.db().session().tx().await.unwrap();
    tx.query_with("MATCH (s:KnowledgeBaseStats) SET s.migration = $m")
        .param(
            "m",
            Value::Map(
                [("kind".to_string(), Value::String("enable_hybrid".into()))]
                    .into_iter()
                    .collect(),
            ),
        )
        .fetch_all()
        .await
        .unwrap();
    tx.commit().await.unwrap();
    kb.shutdown().await.unwrap();
    assert!(matches!(
        open(&path, bge_m3()).await,
        Err(UnikoError::IncompatibleStore(_))
    ));
}

const LOCK_CHILD_ENV: &str = "UNIKO_LOCK_CHILD_STORE";

/// The child half of `a_second_process_cannot_open_an_open_store`: opens the
/// store the parent holds, and must be refused. A no-op when run directly.
#[tokio::test]
async fn lock_child_opens_the_parents_store() {
    let Ok(store) = std::env::var(LOCK_CHILD_ENV) else {
        return;
    };
    let err = open(Path::new(&store), UnikoConfig::default())
        .await
        .expect_err("a second process is refused");
    assert!(
        matches!(&err, UnikoError::Config(m) if m.contains("open for writing elsewhere")),
        "{err}"
    );
    // Inspecting opens read-only, so the parent's writer lock doesn't
    // stop it.
    let started = std::time::Instant::now();
    let (info, verdict) = KnowledgeBase::inspect(Path::new(&store), &UnikoConfig::default())
        .await
        .expect("inspecting a store open elsewhere");
    assert!(info.exists);
    assert!(verdict.is_compatible(), "{verdict}");
    assert!(
        started.elapsed() < std::time::Duration::from_secs(5),
        "inspect waited for the writer: {:?}",
        started.elapsed()
    );
}

fn lock_child_test_name() -> String {
    let module = module_path!().split_once("::").map_or("", |(_, rest)| rest);
    if module.is_empty() {
        "lock_child_opens_the_parents_store".to_string()
    } else {
        format!("{module}::lock_child_opens_the_parents_store")
    }
}

/// A store open in one process can't be opened by another: uni-db's OS lock
/// on its `writer.lock` refuses the second, and the first's close frees it.
/// The other process can still inspect it.
#[tokio::test]
async fn a_second_process_cannot_open_an_open_store() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kb");
    let kb = open(&path, UnikoConfig::default()).await.unwrap();
    let out = std::process::Command::new(std::env::current_exe().expect("test binary"))
        .args(["--exact", &lock_child_test_name(), "--nocapture"])
        .env(LOCK_CHILD_ENV, &path)
        .output()
        .expect("spawn the child");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "the child was not refused: {stdout}");
    assert!(
        stdout.contains("1 passed"),
        "the child test didn't run: {stdout}"
    );
    kb.shutdown().await.unwrap();
    open(&path, UnikoConfig::default())
        .await
        .expect("free once closed")
        .shutdown()
        .await
        .unwrap();
}

/// The shared-runtime open path explains a hybrid store opened dense-only
/// the same way as the catalog path.
#[tokio::test]
async fn a_shared_runtime_open_is_explained_too() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kb");
    create(&path, bge_m3()).await;
    let dense = bge_m3_dense_only();
    let runtime = uniko_store::SharedRuntime::build(&dense, Vec::new(), PrefetchPolicy::Lazy)
        .await
        .expect("runtime");
    let err = KnowledgeBase::open_shared(&path, dense, &runtime)
        .await
        .err()
        .expect("refused");
    assert!(
        differences(&err).contains(&("sparse_dimensions".into(), Verdict::Incompatible)),
        "{err}"
    );
}

#[tokio::test]
async fn migrate_refuses_schema_path_and_unsupported_uris() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kb");
    create(&path, UnikoConfig::default()).await;
    let with_file = UnikoConfig {
        schema_path: Some(dir.path().join("schema.json")),
        ..UnikoConfig::default()
    };
    let err = KnowledgeBase::migrate(
        &path,
        with_file,
        Migration::RebuildVectorIndexes,
        &mut no_progress(),
    )
    .await
    .unwrap_err();
    assert!(
        matches!(&err, UnikoError::Config(m) if m.contains("schema_path")),
        "{err}"
    );
    for result in [
        KnowledgeBase::inspect(
            "abfss://container@acct.dfs.core.windows.net/kb",
            &UnikoConfig::default(),
        )
        .await
        .err(),
        KnowledgeBase::migrate(
            "abfss://container@acct.dfs.core.windows.net/kb",
            UnikoConfig::default(),
            Migration::RebuildVectorIndexes,
            &mut no_progress(),
        )
        .await
        .err(),
    ] {
        assert!(
            matches!(result, Some(UnikoError::Unsupported(_))),
            "{result:?}"
        );
    }
}

/// A llama.cpp store, like other remote embedders, can be inspected.
#[tokio::test]
async fn a_llamacpp_store_can_be_inspected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kb");
    let config = UnikoConfig {
        embedding: EmbeddingConfig::llamacpp("nomic-embed", "http://127.0.0.1:9/v1", 768, 2048),
        ..UnikoConfig::default()
    };
    create(&path, config.clone()).await;
    let (info, verdict) = KnowledgeBase::inspect(&path, &config)
        .await
        .expect("inspect");
    assert!(info.exists && verdict.is_compatible(), "{verdict}");
    assert_eq!(info.embedding.dimensions, Some(768));
}

/// A run that does its part while another migration stays marked says so,
/// and the final progress event counts every row the run touched.
#[tokio::test]
async fn the_report_names_a_migration_still_in_progress() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kb");
    create(&path, bge_m3()).await;
    set_marker(&path, &bge_m3(), "enable_hybrid", "backfill").await;
    let mut dot = bge_m3();
    dot.vector_metric = VectorMetricChoice::Dot;
    let mut last = None;
    let report = KnowledgeBase::migrate(&path, dot, Migration::RebuildVectorIndexes, &mut |p| {
        last = Some(p.clone());
    })
    .await
    .unwrap();
    assert!(!report.completed);
    assert_eq!(report.migration_in_progress, Some(Migration::EnableHybrid));
    let last = last.expect("progress events");
    assert_eq!(last.phase, "done");
    assert_eq!(last.done, report.rows_backfilled + report.rows_unfilled);
}
