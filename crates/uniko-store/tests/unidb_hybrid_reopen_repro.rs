//! Repro: a persisted KB whose schema has a `Vector` auto-embed index bound to
//! an embedding alias whose catalog task is not `Embed` (here `EmbedHybrid`)
//! cannot be reopened via the `xervo_catalog` path.
//!
//! Create succeeds (the open-time validation only inspects vector-index aliases
//! that already exist in the persisted schema, and the schema is empty at
//! create time). Reopen fails:
//!
//! ```text
//! Internal error: Uni-Xervo alias 'embed/hybrid' must be an embed task
//! ```
//!
//! **FIXED upstream.** The test asserts the reopen SUCCEEDS. It failed against
//! uni-db 2.4.1 and ran as an `#[ignore]`d expected-failure; on 2026-08-02 it was
//! found green on both 3.0.1 and 3.2.0, so the fix landed in 3.0.0 or earlier.
//! The attribute is removed so it now guards against regression on every run.

use uni_db::{
    DataType, EmbeddingCfg, IndexType, ModelAliasSpec, ModelTask, Uni, VectorAlgo, VectorIndexCfg,
    VectorMetric, WarmupPolicy,
};

const ALIAS: &str = "embed/hybrid";

fn embed_hybrid_spec() -> ModelAliasSpec {
    ModelAliasSpec {
        alias: ALIAS.to_string(),
        task: ModelTask::EmbedHybrid,
        provider_id: "local/onnx".to_string(),
        model_id: "any/model".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({}),
    }
}

fn hybrid_vector_index() -> VectorIndexCfg {
    VectorIndexCfg {
        algorithm: VectorAlgo::Flat,
        metric: VectorMetric::Cosine,
        embedding: Some(EmbeddingCfg {
            alias: ALIAS.to_string(),
            source_properties: vec!["content".to_string()],
            batch_size: 8,
            document_prefix: None,
            query_prefix: None,
        }),
    }
}

#[tokio::test]
async fn vector_index_on_embedhybrid_alias_can_be_reopened() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_string_lossy().to_string();

    // 1) Create + apply a schema with a Vector auto-embed index on an EmbedHybrid alias.
    {
        let db = Uni::open(&path)
            .xervo_catalog(vec![embed_hybrid_spec()])
            .build()
            .await
            .expect("create");
        db.schema()
            .label("Doc")
            .property("content", DataType::String)
            .property_nullable("embedding", DataType::Vector { dimensions: 4 })
            .index("embedding", IndexType::Vector(hybrid_vector_index()))
            .apply()
            .await
            .expect("apply schema");
        db.shutdown().await.unwrap();
    }

    // 2) Reopen with the same catalog. The persisted schema now carries the
    //    Vector index, so the open-time alias validation runs.
    let reopened = Uni::open(&path)
        .xervo_catalog(vec![embed_hybrid_spec()])
        .build()
        .await;

    assert!(
        reopened.is_ok(),
        "reopen of a hybrid-vector-index KB failed: {:?}",
        reopened.err()
    );
}

/// Workaround verification: the SAME persisted KB reopens cleanly when opened
/// with a prebuilt `ModelRuntime` (`.xervo_runtime(...)`) instead of a catalog
/// (`.xervo_catalog(...)`) — uni-db skips the alias-task validation on the
/// prebuilt-runtime path. This is NOT `#[ignore]`d: it passes on 2.4.1 and is
/// the basis for uniko opening hybrid KBs via `KnowledgeBase::open_with_runtime`.
#[tokio::test]
async fn vector_index_on_embedhybrid_alias_reopens_via_prebuilt_runtime() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().to_string_lossy().to_string();

    // 1) Create + persist the hybrid-aliased Vector index (catalog path).
    {
        let db = Uni::open(&path)
            .xervo_catalog(vec![embed_hybrid_spec()])
            .build()
            .await
            .expect("create");
        db.schema()
            .label("Doc")
            .property("content", DataType::String)
            .property_nullable("embedding", DataType::Vector { dimensions: 4 })
            .index("embedding", IndexType::Vector(hybrid_vector_index()))
            .apply()
            .await
            .expect("apply schema");
        db.shutdown().await.unwrap();
    }

    // 2) Build a prebuilt runtime from the same catalog via a throwaway
    //    in-memory DB (empty schema → no validation; lazy → no model load).
    let bootstrap = Uni::in_memory()
        .xervo_catalog(vec![embed_hybrid_spec()])
        .build()
        .await
        .expect("bootstrap runtime");
    let runtime = bootstrap
        .xervo()
        .raw_runtime()
        .expect("runtime present")
        .clone();

    // 3) Reopen the persisted KB via the prebuilt runtime — validation skipped.
    let reopened = Uni::open(&path).xervo_runtime(runtime).build().await;
    assert!(
        reopened.is_ok(),
        "prebuilt-runtime reopen failed: {:?}",
        reopened.err()
    );
}
