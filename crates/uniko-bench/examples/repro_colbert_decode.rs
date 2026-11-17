//! Failing repro for uni-db #133: an `EmbedHybrid` auto-embedded multivector
//! (ColBERT) column reads back as `Null` with `unknown CypherValue tag: 97`.
//!
//! Opens a fresh hybrid (bge-m3) KB, inserts a few Observations whose `content`
//! is auto-embedded (`EmbedHybrid` -> dense + sparse + `colbert_embedding`),
//! then reads `colbert_embedding` back via Cypher. The read prints
//! `CypherValue decode error: unknown CypherValue tag: 97` and the column
//! comes back `Null` — while the dense `embedding` (Arrow FixedSizeList) and
//! sparse `sparse_embedding` (Arrow Struct) read back fine.
//!
//! Usage (via run.sh env so bge-m3 actually loads):
//!   cargo run --release --features gpu-cuda --example repro_colbert_decode -p uniko-bench -- \
//!       <kb_dir> <bge-m3 bench_config.json>

use std::collections::HashMap;
use std::path::PathBuf;

use uni_db::Value;
use uniko_bench::bench_config::BenchConfig;
use uniko_store::config::UnikoConfig;
use uniko_store::schema::constants::labels;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let kb_dir: PathBuf = std::env::args()
        .nth(1)
        .expect("usage: <kb_dir> <cfg>")
        .into();
    let bc: PathBuf = std::env::args()
        .nth(2)
        .expect("usage: <kb_dir> <cfg>")
        .into();

    let mut config = UnikoConfig::default();
    BenchConfig::load(&bc)
        .and_then(|b| b.apply_to_uniko_config(&mut config))
        .map_err(|e| anyhow::anyhow!("loading bench config {}: {e}", bc.display()))?;
    let _ = std::fs::remove_dir_all(&kb_dir);

    // Hybrid open path (build a shared runtime, then open via it).
    let runtime =
        uniko_store::SharedRuntime::build(&config, Vec::new(), uniko_store::PrefetchPolicy::Eager)
            .await?;
    let kb = uniko_bench::open_kb_with_runtime(&kb_dir, config, runtime).await?;

    // Insert Observations whose `content` triggers EmbedHybrid auto-embed
    // (fills dense `embedding`, `sparse_embedding`, and `colbert_embedding`).
    let items: Vec<HashMap<String, Value>> = (0..3)
        .map(|i| {
            let mut m = HashMap::new();
            m.insert("observation_id".into(), Value::String(format!("obs-{i}")));
            m.insert(
                "content".into(),
                Value::String(format!(
                    "sample observation {i} about cats and dogs at home"
                )),
            );
            m
        })
        .collect();
    let ids = kb.batch_create_nodes(labels::OBSERVATION, &items).await?;
    eprintln!(
        "inserted {} observations (auto-embedded via EmbedHybrid)",
        ids.len()
    );

    // Read each embedding column back. Dense + sparse decode fine; colbert fails.
    let res = kb
        .db()
        .session()
        .query_with(
            "MATCH (o:Observation) RETURN o.observation_id AS id, \
             o.embedding AS dense, o.sparse_embedding AS sparse, o.colbert_embedding AS colbert",
        )
        .fetch_all()
        .await?;

    let kind = |v: Option<&Value>| -> String {
        match v {
            None | Some(Value::Null) => "NULL".to_string(),
            Some(Value::Vector(x)) => format!("Vector(dims={})", x.len()),
            Some(Value::SparseVector { indices, .. }) => {
                format!("SparseVector(nnz={})", indices.len())
            }
            Some(Value::List(l)) => format!("List(len={})", l.len()),
            Some(other) => format!("{other:?}"),
        }
    };

    let mut colbert_null = 0usize;
    for row in res.rows() {
        let colbert = row.value("colbert");
        if matches!(colbert, None | Some(Value::Null)) {
            colbert_null += 1;
        }
        println!(
            "obs {:?}: dense={} sparse={} colbert={}",
            row.value("id").and_then(|v| if let Value::String(s) = v {
                Some(s.as_str())
            } else {
                None
            }),
            kind(row.value("dense")),
            kind(row.value("sparse")),
            kind(row.value("colbert")),
        );
    }

    if colbert_null == res.rows().len() && !res.rows().is_empty() {
        println!(
            "\nREPRODUCED: all {} colbert_embedding columns read back NULL \
             (see the 'unknown CypherValue tag: 97' lines above) while dense + sparse decode fine.",
            res.rows().len()
        );
    } else {
        println!("\nNOT reproduced: colbert columns were non-null.");
    }
    Ok(())
}
