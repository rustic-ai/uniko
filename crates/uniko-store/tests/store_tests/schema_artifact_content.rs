//! Dedicated integration tests for the `:ArtifactContent` label,
//! `HAS_CONTENT` edge, and `:KnowledgeBaseStats` singleton shipped in
//! Phase 1.
//!
//! These tests live separately from `schema_completeness.rs` because:
//! - they exercise property-level shape and singleton enforcement, not
//!   just label-count canaries; and
//! - the singleton-mismatch test needs an on-disk KB (the in-memory
//!   variant can't be reopened).
//
use uniko_store::KnowledgeBase;
use uniko_store::blob_store::BlobStorage;
use uniko_store::config::UnikoConfig;

#[tokio::test]
async fn artifact_content_label_has_expected_properties() {
    let kb = KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB");
    let info = kb
        .db()
        .get_label_info("ArtifactContent")
        .await
        .unwrap()
        .expect(":ArtifactContent must be registered");
    let names: Vec<&str> = info.properties.iter().map(|p| p.name.as_str()).collect();

    for required in [
        "content_id",
        "bytes",
        "uri",
        "mime",
        "size",
        "perceptual_hash",
        "audio_fingerprint",
        "created_at",
    ] {
        assert!(
            names.contains(&required),
            ":ArtifactContent missing required property {required}; have {names:?}"
        );
    }
    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn has_content_edge_registered() {
    let kb = KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB");
    assert!(
        kb.db().edge_type_exists("HAS_CONTENT").await.unwrap(),
        "HAS_CONTENT edge type must be registered"
    );
    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn knowledge_base_stats_singleton_exists_after_init() {
    let kb = KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB");
    let info = kb
        .db()
        .get_label_info("KnowledgeBaseStats")
        .await
        .unwrap()
        .expect(":KnowledgeBaseStats must be registered");
    let names: Vec<&str> = info.properties.iter().map(|p| p.name.as_str()).collect();
    for required in [
        "stats_id",
        "blob_storage",
        "modality_presence",
        "updated_at",
    ] {
        assert!(
            names.contains(&required),
            ":KnowledgeBaseStats missing {required}; have {names:?}"
        );
    }

    // `init_kb_stats` writes the singleton on first open. Reading the
    // modality_presence should succeed (default false) and reading the
    // singleton row count via Cypher should be exactly one.
    let presence = kb.read_modality_presence().await.unwrap();
    assert!(!presence.has_image_content);
    assert!(!presence.has_audio_content);

    let session = kb.db().session();
    let r = session
        .query("MATCH (s:KnowledgeBaseStats) RETURN count(s) AS n")
        .await
        .unwrap();
    let count: i64 = r.rows().first().unwrap().get("n").unwrap();
    assert_eq!(count, 1, "exactly one :KnowledgeBaseStats row expected");

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn reopen_with_mismatched_blob_storage_is_hard_error() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().to_path_buf();

    // First open: default (Lance) backend, write singleton.
    {
        let kb = KnowledgeBase::open(&path, UnikoConfig::default())
            .await
            .expect("first open Lance");
        kb.shutdown().await.unwrap();
    }

    // Reopen with Fs backend: must error from init_kb_stats.
    let blob_root = tmp.path().join("blobs");
    let cfg = UnikoConfig {
        blob_storage: BlobStorage::Fs {
            root: blob_root.clone(),
        },
        ..Default::default()
    };
    let err = KnowledgeBase::open(&path, cfg).await;
    assert!(
        err.is_err(),
        "reopening with a different blob_storage kind must fail"
    );
    let msg = format!("{}", err.unwrap_err());
    assert!(
        msg.contains("blob_storage mismatch"),
        "expected mismatch error, got: {msg}"
    );
}

/// Open without prefetching models, so these tests never download one.
async fn open_no_prefetch(
    path: &std::path::Path,
    config: UnikoConfig,
) -> uniko_store::Result<KnowledgeBase> {
    KnowledgeBase::open_with_xervo_no_prefetch(path, config, Vec::new()).await
}

#[tokio::test]
async fn reopen_with_a_different_embedding_model_is_hard_error() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().to_path_buf();
    let original = UnikoConfig::default();
    {
        let kb = open_no_prefetch(&path, original.clone())
            .await
            .expect("first open");
        kb.shutdown().await.unwrap();
    }

    // Same dimensions, different model: the case that would otherwise
    // mix incompatible vectors silently.
    let mut swapped = original.clone();
    swapped.embedding.model_id = format!("{}-other", original.embedding.model_id);
    let err = open_no_prefetch(&path, swapped)
        .await
        .err()
        .expect("reopening with a different model_id must fail");
    let msg = err.to_string();
    assert!(msg.contains("embedding mismatch"), "got: {msg}");
    assert!(
        msg.contains(&original.embedding.model_id),
        "error must name the stored model, got: {msg}"
    );

    // Different dimensions must fail too, whichever check catches it.
    let mut wider = original.clone();
    wider.embedding.dimensions = original.embedding.dimensions * 2;
    assert!(
        open_no_prefetch(&path, wider).await.is_err(),
        "reopening with different dimensions must fail"
    );

    // The original config still opens.
    let kb = open_no_prefetch(&path, original)
        .await
        .expect("reopen with the original embedding");
    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn reopen_records_embedding_on_a_store_that_predates_it() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().to_path_buf();
    let config = UnikoConfig::default();
    {
        let kb = open_no_prefetch(&path, config.clone())
            .await
            .expect("first open");
        // Simulate a store written before the identity was recorded.
        let session = kb.db().session();
        let tx = session.tx().await.unwrap();
        tx.query("MATCH (s:KnowledgeBaseStats) SET s.embedding = null")
            .await
            .unwrap();
        tx.commit().await.unwrap();
        kb.shutdown().await.unwrap();
    }

    {
        let kb = open_no_prefetch(&path, config.clone())
            .await
            .expect("an unrecorded store opens and records its embedding");
        kb.shutdown().await.unwrap();
    }

    let mut swapped = config.clone();
    swapped.embedding.model_id = format!("{}-other", config.embedding.model_id);
    let err = open_no_prefetch(&path, swapped)
        .await
        .err()
        .expect("once recorded, a different model must fail");
    assert!(err.to_string().contains("embedding mismatch"), "got: {err}");
}
