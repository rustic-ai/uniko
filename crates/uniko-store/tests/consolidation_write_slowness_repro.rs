//! Isolated repro — consolidation-scale **structural** writes are
//! pathologically slow on uni-db's on-disk backend.
//!
//! This strips away everything uniko-specific (embedder, NLP, recall, the
//! bench harness, even uniko-store's `KnowledgeBase`) and exercises only the
//! uni-db public write API — `bulk_insert_vertices` / `bulk_insert_edges` —
//! at the scale one LoCoMo conversation's P4 consolidation produces:
//! a few hundred Fact nodes and a few thousand edges, committed in a handful
//! of batches.
//!
//! What the bench logs showed during the slow phase: hundreds of Lance
//! dataset loads plus repeated `compact_all` / `optimize table <X>` passes
//! that iterate **every** edge-type/label in the schema (logging
//! `Table '<edge>_fwd' was not found` for the unpopulated ones). This repro
//! isolates whether that per-commit, whole-schema compaction is what makes
//! the write phase slow.
//!
//! Cases (all on-disk; identical data unless noted):
//!
//! - `control_few_edge_types_no_vector_index` — 2 edge types, no vectors.
//! - `many_edge_types_no_vector_index` — 53 edge types (uniko-sized), no vectors.
//! - `repro_with_vector_index` — 53 edge types + uniko-default HnswSq/Cosine index.
//! - `repro_incremental_commits_vector_index` — same vectors, ~300 small commits.
//! - `repro_write_on_ingested_kb` — opens a real KB (`UNIKO_REPRO_KB`), times a write.
//!
//! FINDING (2026-06-29): the first four are all sub-second (108ms / 117ms /
//! 256ms / 792ms), so raw bulk writes, schema edge-type count, the HnswSq
//! vector index, and 300-commit fragmentation are NOT the cause in isolation.
//! The fifth can't even raw-open a real KB — its auto-embed vector indexes
//! require the Uni-Xervo catalog — so the multi-minute consolidation hang is
//! NOT reproducible as a pure uni-db program; it needs the full embedder-
//! backed open path against the accumulated ingested state.
//!
//! On-disk path is `CARGO_TARGET_TMPDIR` (the repo filesystem) so the result
//! reflects the real storage backend, not tmpfs.
//!
//! Run:
//!   cargo nextest run -p uniko-store --test integration consolidation_write_slowness_repro:: \
//!       --run-ignored all --no-capture

use std::collections::HashMap;
use std::time::Instant;

use uni_db::{
    DataType, IndexType, ScalarType, Uni, Value, VectorAlgo, VectorIndexCfg, VectorMetric, Vid,
};

/// conv-26-scale: ~300 Facts, ~1200 Observations, ~8 edges/Fact per type.
const N_FACT: usize = 300;
const N_OBS: usize = 1200;
const EDGES_PER_FACT: usize = 8;
/// bge-small dimensionality; the uniko default vector index is `HnswSq` / Cosine.
const EMB_DIM: usize = 384;

/// Deterministic unit-ish embedding for node `i` (no rng dependency).
fn embedding_for(i: usize) -> Vec<f32> {
    let mut s = (i as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(1);
    (0..EMB_DIM)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f32 / u32::MAX as f32) - 0.5
        })
        .collect()
}

/// Open an on-disk uni-db and apply a schema with `n_edge_types` edge types
/// (`EDGE_0` and `EDGE_1` are the only ones populated below). When
/// `with_vec_index` is set, Fact + Observation also get an `embedding`
/// `HnswSq`/Cosine vector index — the uniko default.
async fn build_db(path: &std::path::Path, n_edge_types: usize, with_vec_index: bool) -> Uni {
    let db = Uni::open(path.to_string_lossy().as_ref())
        .build()
        .await
        .expect("open on-disk uni-db");

    fn vcfg() -> VectorIndexCfg {
        VectorIndexCfg {
            algorithm: VectorAlgo::HnswSq {
                m: 16,
                ef_construction: 100,
                partitions: None,
            },
            metric: VectorMetric::Cosine,
            embedding: None,
        }
    }

    let mut fact = db
        .schema()
        .label("Fact")
        .property("fact_id", DataType::String)
        .index("fact_id", IndexType::Scalar(ScalarType::Hash));
    if with_vec_index {
        fact = fact
            .property_nullable(
                "embedding",
                DataType::Vector {
                    dimensions: EMB_DIM,
                },
            )
            .index("embedding", IndexType::Vector(vcfg()));
    }
    let mut obs = fact
        .done()
        .label("Observation")
        .property("obs_id", DataType::String)
        .index("obs_id", IndexType::Scalar(ScalarType::Hash));
    if with_vec_index {
        obs = obs
            .property_nullable(
                "embedding",
                DataType::Vector {
                    dimensions: EMB_DIM,
                },
            )
            .index("embedding", IndexType::Vector(vcfg()));
    }
    let mut sb = obs.done();
    for i in 0..n_edge_types {
        sb = sb
            .edge_type(format!("EDGE_{i}").as_str(), &["Fact"], &["Observation"])
            .done();
    }
    sb.apply().await.expect("apply schema");
    db
}

fn node_props(id_key: &str, id_val: String, idx: usize, with_emb: bool) -> HashMap<String, Value> {
    let mut m = HashMap::new();
    m.insert(id_key.to_string(), Value::String(id_val));
    if with_emb {
        m.insert("embedding".to_string(), Value::Vector(embedding_for(idx)));
    }
    m
}

/// Insert the nodes (one commit) and the edges (two commits, one per
/// populated edge type — mirroring consolidation's batched edge writes).
/// Returns (node_insert_ms, edge_insert_ms).
async fn run_workload(db: &Uni, with_emb: bool) -> (u128, u128) {
    let sess = db.session();

    // ── nodes ──
    let t_nodes = Instant::now();
    let tx = sess.tx().await.unwrap();
    let fact_props: Vec<_> = (0..N_FACT)
        .map(|i| node_props("fact_id", format!("fact-{i:05}"), i, with_emb))
        .collect();
    let fact_vids = tx.bulk_insert_vertices("Fact", fact_props).await.unwrap();
    let obs_props: Vec<_> = (0..N_OBS)
        .map(|i| node_props("obs_id", format!("obs-{i:05}"), N_FACT + i, with_emb))
        .collect();
    let obs_vids = tx
        .bulk_insert_vertices("Observation", obs_props)
        .await
        .unwrap();
    tx.commit().await.unwrap();
    let node_ms = t_nodes.elapsed().as_millis();

    // build edge lists: each Fact → EDGES_PER_FACT Observations, two types.
    let mut e0: Vec<(Vid, Vid, HashMap<String, Value>)> = Vec::new();
    let mut e1: Vec<(Vid, Vid, HashMap<String, Value>)> = Vec::new();
    for (fi, &fv) in fact_vids.iter().enumerate() {
        for k in 0..EDGES_PER_FACT {
            let ov = obs_vids[(fi * EDGES_PER_FACT + k) % obs_vids.len()];
            e0.push((fv, ov, HashMap::new()));
            e1.push((fv, ov, HashMap::new()));
        }
    }

    // ── edges: one commit per type ──
    let t_edges = Instant::now();
    let tx = sess.tx().await.unwrap();
    tx.bulk_insert_edges("EDGE_0", e0).await.unwrap();
    tx.commit().await.unwrap();
    let tx = sess.tx().await.unwrap();
    tx.bulk_insert_edges("EDGE_1", e1).await.unwrap();
    tx.commit().await.unwrap();
    let edge_ms = t_edges.elapsed().as_millis();

    (node_ms, edge_ms)
}

/// Insert `N_OBS` observations (with embeddings) in many small commits of
/// `commit_size`, mimicking ingest's per-message commit pattern. Each commit
/// adds a few vectors to the `HnswSq` index, fragmenting it into one segment
/// per commit — the state the real KB is in when consolidation starts.
/// Returns (total_ms, slowest_single_commit_ms).
async fn run_incremental(db: &Uni, commit_size: usize) -> (u128, u128) {
    let sess = db.session();
    let t = Instant::now();
    let mut slowest = 0u128;
    let mut i = 0;
    while i < N_OBS {
        let end = (i + commit_size).min(N_OBS);
        let batch: Vec<_> = (i..end)
            .map(|j| node_props("obs_id", format!("obs-{j:05}"), j, true))
            .collect();
        let tc = Instant::now();
        let tx = sess.tx().await.unwrap();
        tx.bulk_insert_vertices("Observation", batch).await.unwrap();
        tx.commit().await.unwrap();
        slowest = slowest.max(tc.elapsed().as_millis());
        i = end;
    }
    (t.elapsed().as_millis(), slowest)
}

/// N single-Fact commits via **Cypher `CREATE`** — goes through uni-db's
/// trigger/CDC dispatch, which (with the plugin host active) enqueues a
/// deferral per write → `DeferralQueue::persist_locked` → fsync.
async fn run_cypher_commits(db: &Uni, n: usize) -> (u128, u128) {
    let sess = db.session();
    let t = Instant::now();
    let mut slowest = 0u128;
    for i in 0..n {
        let tc = Instant::now();
        let tx = sess.tx().await.unwrap();
        tx.execute(&format!("CREATE (:Fact {{fact_id: 'cyf-{i:05}'}})"))
            .await
            .unwrap();
        tx.commit().await.unwrap();
        slowest = slowest.max(tc.elapsed().as_millis());
    }
    (t.elapsed().as_millis(), slowest)
}

/// Same N single-Fact commits via **`bulk_insert_vertices`** — bypasses the
/// Cypher/trigger path entirely. Control for the Cypher case above.
async fn run_bulk_commits(db: &Uni, n: usize) -> (u128, u128) {
    let sess = db.session();
    let t = Instant::now();
    let mut slowest = 0u128;
    for i in 0..n {
        let tc = Instant::now();
        let tx = sess.tx().await.unwrap();
        tx.bulk_insert_vertices(
            "Fact",
            vec![node_props("fact_id", format!("blk-{i:05}"), i, false)],
        )
        .await
        .unwrap();
        tx.commit().await.unwrap();
        slowest = slowest.max(tc.elapsed().as_millis());
    }
    (t.elapsed().as_millis(), slowest)
}

async fn run_case(n_edge_types: usize, with_vec_index: bool, tag: &str) -> (u128, u128) {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("consolidation_repro_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let db = build_db(&dir, n_edge_types, with_vec_index).await;
    let (node_ms, edge_ms) = run_workload(&db, with_vec_index).await;
    println!(
        "[{tag}] edge_types={n_edge_types} vector_index={with_vec_index} nodes={} edges={} | node_insert={node_ms}ms edge_insert={edge_ms}ms total={}ms",
        N_FACT + N_OBS,
        2 * N_FACT * EDGES_PER_FACT,
        node_ms + edge_ms,
    );
    let _ = std::fs::remove_dir_all(&dir);
    (node_ms, edge_ms)
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "perf repro; run explicitly with --run-ignored all --no-capture"]
async fn control_few_edge_types_no_vector_index() {
    let (n, e) = run_case(2, false, "few").await;
    println!(
        "CONTROL few/no-vec: node={n}ms edge={e}ms total={}ms",
        n + e
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "perf repro; run explicitly with --run-ignored all --no-capture"]
async fn many_edge_types_no_vector_index() {
    let (n, e) = run_case(53, false, "many").await;
    println!(
        "MANY(53 edges)/no-vec: node={n}ms edge={e}ms total={}ms",
        n + e
    );
}

/// The suspected reproducer: same data, but Fact + Observation carry the
/// uniko-default `HnswSq`/Cosine `embedding` vector index. If this is orders
/// of magnitude slower than the no-vector-index cases, the cost is vector
/// index maintenance/optimization on write, not raw graph writes.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "perf repro; run explicitly with --run-ignored all --no-capture"]
async fn repro_with_vector_index() {
    let (n, e) = run_case(53, true, "vec").await;
    println!(
        "VEC(53 edges + HnswSq): node={n}ms edge={e}ms total={}ms",
        n + e
    );
}

/// Repro against a REAL ingested KB: set `UNIKO_REPRO_KB` to an ingested
/// `<dir>/conv-26` path. Opens it at the uni-db level (schema already
/// applied, no embedder), reads existing Observation ids, then times a
/// consolidation-shaped write: ~300 Fact nodes + ~2400 `SUPPORTED_BY` edges.
/// If THIS is slow (minutes) while the synthetic cases above are sub-second,
/// the cost is the accumulated ingested-KB state (delta/index fragmentation),
/// not the write API itself.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "set UNIKO_REPRO_KB=<dir>/conv-26 and run with --run-ignored all --no-capture"]
async fn repro_write_on_ingested_kb() {
    let Ok(kb_path) = std::env::var("UNIKO_REPRO_KB") else {
        println!("UNIKO_REPRO_KB not set — skipping");
        return;
    };
    let t_open = Instant::now();
    let db = match Uni::open(kb_path.as_str()).build().await {
        Ok(db) => db,
        Err(e) => {
            // Real uniko KBs declare auto-embed vector indexes, so a raw
            // `Uni::open` is refused ("Uni-Xervo catalog is required ...").
            // Opening one needs the embedder catalog — i.e. uniko-store's
            // `open_with_xervo`, which pulls in the full embedder stack.
            // That means the slowness is NOT reproducible as a pure uni-db
            // program against a real KB; it requires the catalog path.
            println!("cannot raw-open ingested KB (needs catalog): {e}");
            return;
        }
    };
    println!("opened {kb_path} in {}ms", t_open.elapsed().as_millis());

    // grab existing Observation VIDs to attach edges to.
    let t_read = Instant::now();
    let rows = db
        .session()
        .query_with("MATCH (o:Observation) RETURN id(o) AS id LIMIT 2400")
        .fetch_all()
        .await
        .expect("read observation ids");
    let obs_vids: Vec<Vid> = rows
        .rows()
        .iter()
        .filter_map(|r| r.get::<i64>("id").ok())
        .map(|i| Vid::new(i as u64))
        .collect();
    println!(
        "read {} observation ids in {}ms",
        obs_vids.len(),
        t_read.elapsed().as_millis()
    );
    assert!(!obs_vids.is_empty(), "KB has no Observations");

    // consolidation-shaped write: Fact nodes (no embedding) + SUPPORTED_BY edges.
    let sess = db.session();
    let t_facts = Instant::now();
    let tx = sess.tx().await.unwrap();
    let fact_props: Vec<_> = (0..N_FACT)
        .map(|i| {
            let mut m = HashMap::new();
            m.insert(
                "fact_id".to_string(),
                Value::String(format!("repro-fact-{i:05}")),
            );
            m
        })
        .collect();
    let fact_vids = tx.bulk_insert_vertices("Fact", fact_props).await.unwrap();
    tx.commit().await.unwrap();
    println!(
        "inserted {N_FACT} Facts in {}ms",
        t_facts.elapsed().as_millis()
    );

    let mut edges: Vec<(Vid, Vid, HashMap<String, Value>)> = Vec::new();
    for (fi, &fv) in fact_vids.iter().enumerate() {
        for k in 0..EDGES_PER_FACT {
            let ov = obs_vids[(fi * EDGES_PER_FACT + k) % obs_vids.len()];
            edges.push((fv, ov, HashMap::new()));
        }
    }
    let t_edges = Instant::now();
    let tx = sess.tx().await.unwrap();
    tx.bulk_insert_edges("SUPPORTED_BY", edges).await.unwrap();
    tx.commit().await.unwrap();
    println!(
        "INGESTED-KB: inserted {} SUPPORTED_BY edges in {}ms",
        N_FACT * EDGES_PER_FACT,
        t_edges.elapsed().as_millis()
    );
}

/// THE ISOLATED REPRO: 300 single-Fact commits via Cypher vs via bulk_insert,
/// same minimal schema, same on-disk backend. The gdb backtrace of the real
/// consolidation hang showed every fsync coming from
/// `uni_plugin_host::triggers::DeferralQueue::persist_locked`, which fires on
/// the Cypher/trigger write path but NOT on bulk_insert. If Cypher is
/// dramatically slower (per-commit fsync storm) while bulk is fast, that's the
/// hang reproduced in a self-contained uni-db program — no KB, no models.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "perf repro; run explicitly with --run-ignored all --no-capture"]
async fn repro_cypher_vs_bulk_commits() {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("consolidation_repro_cvb_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let db = build_db(&dir, 2, false).await;

    let n = 300;
    let (bulk_total, bulk_slow) = run_bulk_commits(&db, n).await;
    let (cy_total, cy_slow) = run_cypher_commits(&db, n).await;
    println!(
        "CYPHER-vs-BULK ({n} single-row commits): bulk total={bulk_total}ms (slowest={bulk_slow}ms) | cypher total={cy_total}ms (slowest={cy_slow}ms) | ratio={:.1}x",
        cy_total as f64 / bulk_total.max(1) as f64,
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The faithful reproducer: insert the same vectors but in ~300 small commits
/// (one per "message", like ingest) under the HnswSq index. If total time
/// blows up vs the single-shot VEC case while per-commit time climbs, the cost
/// is per-commit vector-index maintenance over accumulating segments.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "perf repro; run explicitly with --run-ignored all --no-capture"]
async fn repro_incremental_commits_vector_index() {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("consolidation_repro_incr_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let db = build_db(&dir, 53, true).await;
    let commit_size = 4;
    let (total_ms, slowest_ms) = run_incremental(&db, commit_size).await;
    println!(
        "INCREMENTAL: {} obs in {} commits of {commit_size} (HnswSq) | total={total_ms}ms slowest_commit={slowest_ms}ms",
        N_OBS,
        N_OBS / commit_size,
    );
    let _ = std::fs::remove_dir_all(&dir);
}
