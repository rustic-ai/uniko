//! Minimal repro for the learned-sparse recall channel failing at query
//! time while ingest writes sparse vectors fine.
//!
//! Observed on the LoCoMo bench (bge-m3, `recall.sparse_enabled = true`):
//! every `uni.sparse.query` returns
//!
//! ```text
//! Provider capability missing for alias 'embed/hybrid' (provider 'local/onnx'):
//! SparseEmbeddingModel
//! ```
//!
//! `recall::run_sparse_source` swallows that into an empty hit list at
//! `debug` level, so the channel silently contributes nothing while still
//! costing a query per variant (~+9% recall latency, 0 result change over
//! 105 questions).
//!
//! The two paths are asymmetric upstream:
//!
//! - WRITE (`uni-store/src/runtime/writer.rs:161`) calls `sparse_embedder`
//!   and, on a capability mismatch, falls back to
//!   `hybrid_single_head(.., HeadSet::SPARSE)` — explicitly for "issue #129".
//! - QUERY (`uni-query/src/query/df_graph/search_procedures.rs:844`) calls
//!   `sparse_embedder(&embedding_config.alias)` with no such fallback.
//!
//! `ModelRuntime::sparse_embedder` downcasts the cached handle to
//! `Arc<dyn SparseEmbeddingModel>`; an `EmbedHybrid` alias stores
//! `Arc<dyn HybridEmbeddingModel>`, so the downcast misses.
//!
//! This test pins which side is at fault. If the failure reproduces with
//! the SHIPPED `bge-m3` preset (all three heads, exactly what
//! `locomo-bgem3-norerank-retrieval.json` configures), the fault is not in
//! how the bench profile was written.
//!
//! Ignored by default: it downloads / loads bge-m3 (~2GB) and is a
//! diagnostic, not a gate. Run with:
//!
//! ```sh
//! cargo nextest run -p uniko-store --test sparse_query_alias_repro \
//!   --run-ignored all --no-capture
//! ```

use std::collections::HashMap;

use uniko_store::config::{EmbeddingConfig, UnikoConfig};
use uniko_store::storage::KnowledgeBase;
use uniko_store::{Value, id};

/// Build a KB whose Chunk rows carry a `sparse_embedding` column, seed two
/// chunks, then run the same sparse query recall issues.
async fn probe(embedding: EmbeddingConfig, label: &str) {
    let config = UnikoConfig {
        embedding,
        ..Default::default()
    };

    let kb = match KnowledgeBase::in_memory(config).await {
        Ok(kb) => kb,
        Err(e) => {
            eprintln!("[{label}] SKIP: KB unavailable: {e}");
            return;
        }
    };

    for text in [
        "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
        "Ada Lovelace wrote the first published algorithm.",
    ] {
        let mut props = HashMap::new();
        props.insert("chunk_id".to_string(), Value::String(id::new_id()));
        props.insert("text".to_string(), Value::String(text.to_string()));
        props.insert("index".to_string(), Value::Int(0));
        props.insert("chunk_type".to_string(), Value::String("text".into()));
        if let Err(e) = kb.create_node("Chunk", &props).await {
            eprintln!("[{label}] SKIP: chunk create failed: {e}");
            return;
        }
    }

    let p = std::collections::HashMap::new();

    // 1. Did ingest populate the sparse column at all?
    match kb
        .query_cypher(
            "MATCH (c:Chunk) RETURN count(c) AS total, \
             count(c.sparse_embedding) AS with_sparse",
            &p,
        )
        .await
    {
        Ok(rows) => eprintln!("[{label}] column check: {:?}", rows.first()),
        Err(e) => eprintln!("[{label}] column check FAILED — {e}"),
    }

    // 2. The raw procedure, yielding only the documented columns — isolates
    //    retrieval from uniko's `labels(node)` projection.
    match kb
        .query_cypher(
            "CALL uni.sparse.query('Chunk', 'sparse_embedding', 'lattice tower', 5, \
             null, null, {}) YIELD vid, score RETURN vid, score",
            &p,
        )
        .await
    {
        Ok(rows) => eprintln!("[{label}] raw CALL OK — {} rows: {:?}", rows.len(), rows),
        Err(e) => eprintln!("[{label}] raw CALL FAILED — {e}"),
    }

    // 2b. Candidate fix for the projection: bind the node via its vid
    //     instead of `YIELD node`.
    match kb
        .query_cypher(
            "CALL uni.sparse.query('Chunk', 'sparse_embedding', 'lattice tower', 5, \
             null, null, {}) YIELD vid, score \
             MATCH (n) WHERE id(n) = vid \
             RETURN id(n) AS nid, labels(n)[0] AS lbl, coalesce(n.text, '') AS content, score",
            &p,
        )
        .await
    {
        Ok(rows) => {
            eprintln!("[{label}] FIXED projection OK — {} rows", rows.len());
            for r in &rows {
                eprintln!(
                    "    lbl={:?} score={:?} content={:?}",
                    r.get("lbl"),
                    r.get("score"),
                    r.get("content")
                );
            }
        }
        Err(e) => eprintln!("[{label}] FIXED projection FAILED — {e}"),
    }

    // 3. Exactly what `recall::run_sparse_source` issues.
    match kb
        .recall_sparse_search(
            "Chunk",
            "sparse_embedding",
            "text",
            "lattice tower",
            5,
            None,
        )
        .await
    {
        Ok(rows) => eprintln!("[{label}] recall_sparse_search OK — {} rows", rows.len()),
        Err(e) => eprintln!("[{label}] recall_sparse_search FAILED — {e}"),
    }
}

/// The shipped preset: dense + sparse + ColBERT, all three heads. This is
/// what `{"preset": "bge-m3"}` in a bench profile resolves to.
#[ignore = "diagnostic: loads bge-m3 (~2GB)"]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sparse_query_with_shipped_bge_m3_preset() {
    probe(
        EmbeddingConfig::bge_m3(),
        "preset bge-m3 (dense+sparse+colbert)",
    )
    .await;
}

/// The variant the sparse A/B used: same model, sparse head declared but
/// ColBERT omitted. If this fails and the preset above succeeds, the bench
/// profile was misconfigured; if both fail identically, the profile is not
/// the variable.
#[ignore = "diagnostic: loads bge-m3 (~2GB)"]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sparse_query_with_sparse_only_hybrid() {
    let mut cfg = EmbeddingConfig::bge_m3();
    cfg.multivector_dimensions = None;
    probe(cfg, "inline bge-m3 (dense+sparse, no colbert)").await;
}

/// Validate the ColBERT MaxSim path, which shares the `YIELD node` defect
/// with the sparse channel.
///
/// Unlike sparse, ColBERT passes its query vector explicitly, so it never
/// needs a query-side embed alias — the projection is its only failure
/// mode, and it is reached as soon as the call executes.
#[ignore = "diagnostic: loads bge-m3 (~2GB)"]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn colbert_maxsim_projection() {
    let label = "bge-m3 preset (colbert enabled)";
    // bge_m3() sets multivector_dimensions = Some(1024).
    let config = UnikoConfig {
        embedding: EmbeddingConfig::bge_m3(),
        ..Default::default()
    };

    let kb = match KnowledgeBase::in_memory(config).await {
        Ok(kb) => kb,
        Err(e) => {
            eprintln!("[{label}] SKIP: KB unavailable: {e}");
            return;
        }
    };

    for text in [
        "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
        "Ada Lovelace wrote the first published algorithm.",
    ] {
        let mut props = HashMap::new();
        props.insert("chunk_id".to_string(), Value::String(id::new_id()));
        props.insert("text".to_string(), Value::String(text.to_string()));
        props.insert("index".to_string(), Value::Int(0));
        props.insert("chunk_type".to_string(), Value::String("text".into()));
        if let Err(e) = kb.create_node("Chunk", &props).await {
            eprintln!("[{label}] SKIP: chunk create failed: {e}");
            return;
        }
    }

    let p = HashMap::new();
    match kb
        .query_cypher(
            "MATCH (c:Chunk) RETURN count(c) AS total, \
             count(c.colbert_embedding) AS with_colbert",
            &p,
        )
        .await
    {
        Ok(rows) => eprintln!("[{label}] column check: {:?}", rows.first()),
        Err(e) => eprintln!("[{label}] column check FAILED — {e}"),
    }

    let ids: Vec<i64> = match kb
        .query_cypher("MATCH (c:Chunk) RETURN id(c) AS nid", &p)
        .await
    {
        Ok(rows) => rows
            .iter()
            .filter_map(|r| match r.get("nid") {
                Some(Value::Int(i)) => Some(*i),
                _ => None,
            })
            .collect(),
        Err(e) => {
            eprintln!("[{label}] id fetch FAILED — {e}");
            return;
        }
    };
    eprintln!("[{label}] candidate ids: {ids:?}");

    // A synthetic 2-token query multivector. The scores are not the point —
    // whether the procedure call and projection execute is.
    let qmulti: Vec<Vec<f32>> = vec![vec![0.02f32; 1024], vec![0.01f32; 1024]];

    match kb
        .recall_colbert_maxsim("Chunk", "colbert_embedding", "text", &qmulti, &ids)
        .await
    {
        Ok(rows) => {
            eprintln!("[{label}] colbert_maxsim OK — {} rows", rows.len());
            for r in &rows {
                eprintln!(
                    "    nid={} label={:?} score={:.4} content={:?}",
                    r.node_id,
                    r.label,
                    r.score,
                    r.content.chars().take(40).collect::<String>()
                );
            }
        }
        Err(e) => eprintln!("[{label}] colbert_maxsim FAILED — {e}"),
    }
}
