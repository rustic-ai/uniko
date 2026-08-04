//! Repro / enhancement: a single multi-head model (e.g. BGE-M3) registered for
//! one task cannot serve a column that needs a different task. A model
//! registered as `EmbedHybrid` exposes a DENSE head, but a plain dense `Vector`
//! auto-embed column bound to that alias fails at CREATE:
//!
//! ```text
//! Capability mismatch: Model for alias 'embed/hybrid' does not implement EmbeddingModel
//! ```
//!
//! Consequence: to serve both a lone dense column and a hybrid (dense+multi)
//! group from the same model, callers must register it under two aliases (one
//! `Embed`, one `EmbedHybrid`); the runtime cache keys on `task`, so the model
//! loads twice.
//!
//! **FIXED upstream.** The test asserts the hybrid model CAN auto-embed a dense
//! column (one model, multiple tasks). It failed against uni-db 2.4.1 and ran as
//! an `#[ignore]`d expected-failure; on 2026-08-02 it was found green on both
//! 3.0.1 and 3.2.0, so the fix landed in 3.0.0 or earlier. The attribute is
//! removed so it now guards against regression on every run.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use uni_db::{
    DataType, EmbeddingCfg, IndexType, ModelAliasSpec, ModelTask, Uni, VectorAlgo, VectorIndexCfg,
    VectorMetric, WarmupPolicy,
};
use uni_xervo::runtime::ModelRuntime;
use uni_xervo::traits::hybrid::{HeadSet, HybridEmbedResult, HybridEmbeddingModel};
use uni_xervo::traits::{
    LoadedModelHandle, ModelInfo, ModelProvider, ProviderCapabilities, ProviderHealth,
};

const DIM: usize = 4;
const ALIAS: &str = "embed/hybrid";

// A multi-head model that can produce a dense head and a multi-vector head.
struct MockHybrid {
    calls: Arc<AtomicUsize>,
}

impl ModelInfo for MockHybrid {
    fn model_id(&self) -> &str {
        "mock-hybrid"
    }
}

#[async_trait]
impl HybridEmbeddingModel for MockHybrid {
    async fn embed(
        &self,
        texts: &[&str],
        heads: HeadSet,
    ) -> uni_xervo::error::Result<HybridEmbedResult> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let mut res = HybridEmbedResult::default();
        if heads.contains(HeadSet::DENSE) {
            res.dense = Some(texts.iter().map(|_| vec![1.0; DIM]).collect());
        }
        if heads.contains(HeadSet::MULTI_VECTOR) {
            res.multi_vector = Some(texts.iter().map(|_| vec![vec![1.0; DIM]]).collect());
        }
        Ok(res)
    }

    fn available_heads(&self) -> HeadSet {
        HeadSet::DENSE | HeadSet::MULTI_VECTOR
    }
}

struct HybridProvider {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl ModelProvider for HybridProvider {
    fn provider_id(&self) -> &'static str {
        "mock/hybrid"
    }
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supported_tasks: vec![ModelTask::EmbedHybrid],
        }
    }
    async fn load(&self, _spec: &ModelAliasSpec) -> uni_xervo::error::Result<LoadedModelHandle> {
        let handle: Arc<dyn HybridEmbeddingModel> = Arc::new(MockHybrid {
            calls: self.calls.clone(),
        });
        Ok(Arc::new(handle) as LoadedModelHandle)
    }
    async fn health(&self) -> ProviderHealth {
        ProviderHealth::Healthy
    }
}

fn spec() -> ModelAliasSpec {
    ModelAliasSpec {
        alias: ALIAS.to_string(),
        task: ModelTask::EmbedHybrid,
        provider_id: "mock/hybrid".to_string(),
        model_id: "mock-hybrid".to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({}),
    }
}

fn dense_index_on_hybrid_alias() -> VectorIndexCfg {
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
async fn hybrid_model_serves_a_dense_only_column() {
    let calls = Arc::new(AtomicUsize::new(0));
    let runtime = ModelRuntime::builder()
        .register_provider(HybridProvider {
            calls: calls.clone(),
        })
        .catalog(vec![spec()])
        .build()
        .await
        .expect("runtime");

    let db = Uni::in_memory()
        .xervo_runtime(runtime)
        .build()
        .await
        .expect("db");

    // A label whose ONLY embedding column is a plain dense `Vector`, auto-embedded
    // on the hybrid alias (a lone-dense group — heads_wanted == 1).
    db.schema()
        .label("Doc")
        .property("content", DataType::String)
        .property_nullable("embedding", DataType::Vector { dimensions: DIM })
        .index(
            "embedding",
            IndexType::Vector(dense_index_on_hybrid_alias()),
        )
        .apply()
        .await
        .expect("apply schema");

    let tx = db.session().tx().await.expect("tx");
    let res = tx.execute("CREATE (:Doc {content: 'hello world'})").await;

    assert!(
        res.is_ok(),
        "hybrid model could not auto-embed a dense column: {:?}",
        res.err()
    );
    tx.commit().await.expect("commit");
}
