//! Perf comparison: a dense `similar_to` query over three schemas that differ
//! ONLY in which extra (unused-by-the-query) columns the rows carry —
//! dense / dense+sparse / dense+sparse+colbert.
//!
//! Motivation: bge-m3 hybrid recall measured ~65× slower than bge-small in the
//! identical entity-scoped `similar_to` fan-out. perf showed the time in
//! lance's read path (ZSTD decompress + msgpack deserialize), not the vector
//! math — i.e. read-amplification from materializing the heavy `sparse_embedding`
//! (250k-dim) + `colbert_embedding` (per-token multi-vector) columns even though
//! a dense score never reads them. This isolates that: identical data + identical
//! query, only the *presence* of the extra columns changes.
//!
//! The query has no vector index (brute-force scan), mirroring the real recall
//! shape (a `WHERE` filter forces a per-row `similar_to` scan), so every row is
//! read — which is where the column weight shows up.
//!
//! Run:
//!   cargo nextest run -p uniko-store --test integration hybrid_query_perf_repro:: \
//!       --run-ignored all --no-capture
//! Profile one config:
//!   perf record -g -- cargo nextest run ... ; perf report

use std::collections::HashMap;
use std::time::Instant;

use uni_db::{DataType, Uni, Value};

const N_OBS: usize = 100; // rows scanned per query
const DENSE_DIM: usize = 1024; // bge-m3 dense dim
const SPARSE_NNZ: usize = 256; // non-zeros in the learned-sparse vector
const SPARSE_DIM: usize = 250_002; // bge-m3 / XLM-R term-space
const COLBERT_TOKENS: usize = 48; // per-token vectors in the ColBERT multi-vector
const ITERS: usize = 5; // query repetitions (timing + perf-sampling)

/// Deterministic pseudo-random f32 stream (no rng dep).
fn rng(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed >> 11) as f32 / u32::MAX as f32) - 0.5
}

fn dense(seed: &mut u64) -> Vec<f32> {
    (0..DENSE_DIM).map(|_| rng(seed)).collect()
}

fn sparse(seed: &mut u64) -> Value {
    let mut idx: Vec<u32> = (0..SPARSE_NNZ)
        .map(|_| {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (*seed >> 33) as u32 % SPARSE_DIM as u32
        })
        .collect();
    idx.sort_unstable();
    idx.dedup();
    let values: Vec<f32> = (0..idx.len()).map(|_| rng(seed).abs()).collect();
    Value::SparseVector {
        indices: idx,
        values,
    }
}

fn colbert(seed: &mut u64) -> Value {
    Value::List(
        (0..COLBERT_TOKENS)
            .map(|_| Value::Vector(dense(seed)))
            .collect(),
    )
}

async fn build_and_time(with_sparse: bool, with_colbert: bool, tag: &str) -> u128 {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("hybrid_query_perf_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let db = Uni::open(dir.to_string_lossy().as_ref())
        .build()
        .await
        .unwrap();
    let mut sb = db.schema().label("Obs").property_nullable(
        "embedding",
        DataType::Vector {
            dimensions: DENSE_DIM,
        },
    );
    if with_sparse {
        sb = sb.property_nullable(
            "sparse_embedding",
            DataType::SparseVector {
                dimensions: SPARSE_DIM,
            },
        );
    }
    if with_colbert {
        sb = sb.property_nullable(
            "colbert_embedding",
            DataType::List(Box::new(DataType::Vector {
                dimensions: DENSE_DIM,
            })),
        );
    }
    sb.done().apply().await.unwrap();

    // Populate N_OBS rows (identical dense data across all three configs).
    let mut seed = 0x1234_5678u64;
    let rows: Vec<HashMap<String, Value>> = (0..N_OBS)
        .map(|_| {
            let mut m = HashMap::new();
            m.insert("embedding".into(), Value::Vector(dense(&mut seed)));
            if with_sparse {
                m.insert("sparse_embedding".into(), sparse(&mut seed));
            }
            if with_colbert {
                m.insert("colbert_embedding".into(), colbert(&mut seed));
            }
            m
        })
        .collect();
    let tx = db.session().tx().await.unwrap();
    tx.bulk_insert_vertices("Obs", rows).await.unwrap();
    tx.commit().await.unwrap();

    // Identical dense similar_to query, ITERS times. No vector index -> per-row scan.
    let mut qseed = 0xDEAD_BEEFu64;
    let qvec: Vec<f32> = dense(&mut qseed);
    let cypher = "MATCH (n:Obs) RETURN id(n) AS nid, similar_to([n.embedding], [$qvec]) AS score \
                  ORDER BY score DESC LIMIT 20";
    let sess = db.session();
    // warmup
    let _ = sess
        .query_with(cypher)
        .param("qvec", Value::Vector(qvec.clone()))
        .fetch_all()
        .await;

    let t = Instant::now();
    for _ in 0..ITERS {
        let _ = sess
            .query_with(cypher)
            .param("qvec", Value::Vector(qvec.clone()))
            .fetch_all()
            .await
            .unwrap();
    }
    let per_query_ms = t.elapsed().as_millis() / ITERS as u128;

    let size = std::fs::read_dir(dir.join("storage"))
        .map(|rd| {
            rd.flatten()
                .filter_map(|e| e.metadata().ok().map(|m| m.len()))
                .sum::<u64>()
        })
        .unwrap_or(0);
    println!(
        "[{tag}] dense{}{} | {N_OBS} rows | per_query={per_query_ms}ms | storage={}MB",
        if with_sparse { "+sparse" } else { "" },
        if with_colbert { "+colbert" } else { "" },
        size / 1_048_576,
    );
    let _ = std::fs::remove_dir_all(&dir);
    per_query_ms
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "perf repro; run with --run-ignored all --no-capture"]
async fn compare_dense_vs_sparse_vs_colbert() {
    let dense = build_and_time(false, false, "dense").await;
    let dense_sparse = build_and_time(true, false, "dense_sparse").await;
    let dense_sparse_colbert = build_and_time(true, true, "dense_sparse_colbert").await;
    println!("\n=== dense similar_to per-query, by columns PRESENT on the row ===");
    println!("  dense                 : {dense}ms  (baseline)");
    println!(
        "  dense+sparse          : {dense_sparse}ms  ({:.1}x)",
        dense_sparse as f64 / dense.max(1) as f64
    );
    println!(
        "  dense+sparse+colbert  : {dense_sparse_colbert}ms  ({:.1}x)",
        dense_sparse_colbert as f64 / dense.max(1) as f64
    );
}
