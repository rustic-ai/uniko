//! End-to-end check that the ColBERT *query* embedding resolves.
//!
//! The MaxSim call takes its query vector as an explicit parameter, so a
//! test that supplies one by hand exercises the projection but never the
//! code that produces it. That gap hid a real failure: on the LoCoMo
//! matrix every question logged
//!
//! ```text
//! WARN colbert query embed failed, keeping RRF order
//!   error=Provider capability missing for alias 'embed/hybrid'
//!         (provider 'local/onnx'): MultiVectorEmbeddingModel
//! ```
//!
//! and the reranker silently kept RRF order — 0 of 105 questions differed
//! from the un-reranked arm. `colbert_rerank` logs at `warn` and returns,
//! so an inert ColBERT reranker is indistinguishable from one that simply
//! did not reorder anything.
//!
//! `embed_multivector_query` now resolves `MULTIVECTOR_EMBED_ALIAS`
//! (`EmbedMultiVector`) instead of the hybrid alias. Documents still get
//! `colbert_embedding` from the hybrid pass; only the query side moved.
//!
//! Ignored by default: loads bge-m3 (~2GB). Run with:
//!
//! ```sh
//! cargo nextest run -p uniko-extract --test colbert_query_embed \
//!   --run-ignored all --no-capture
//! ```

use uniko_store::config::{EmbeddingConfig, UnikoConfig};
use uniko_store::storage::KnowledgeBase;

#[ignore = "diagnostic: loads bge-m3 (~2GB)"]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn colbert_query_embedding_resolves() {
    // bge_m3() sets multivector_dimensions = Some(1024).
    let config = UnikoConfig {
        embedding: EmbeddingConfig::bge_m3(),
        ..Default::default()
    };

    let kb = match KnowledgeBase::in_memory(config).await {
        Ok(kb) => kb,
        Err(e) => {
            eprintln!("SKIP: KB unavailable: {e}");
            return;
        }
    };

    match uniko_extract::embedding::embed_multivector_query(&kb, "lattice tower in Paris").await {
        Ok(q) => {
            assert!(!q.is_empty(), "query multivector must not be empty");
            assert_eq!(
                q[0].len(),
                1024,
                "per-token vectors must match multivector_dimensions"
            );
            eprintln!(
                "colbert query embed OK — {} tokens x {} dims",
                q.len(),
                q[0].len()
            );
        }
        Err(e) => panic!("colbert query embed failed — {e}"),
    }
}
