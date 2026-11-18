//! LoCoMo benchmark harness for uniko cognitive memory.
//!
//! Ingests LoCoMo conversations, runs recall queries, and evaluates
//! answer quality using token-level F1 and LLM-as-judge metrics.
//!
//! # Usage
//!
//! ```bash
//! # Run with a bench-config profile
//! ./crates/uniko-bench/run.sh \
//!     --bench-config crates/uniko-bench/bench-configs/locomo-bge-openai.json \
//!     --data data/locomo10.json \
//!     --conversations conv-26 \
//!     --output data/results.json
//! ```
//!
//! Every model / device / recall / cost knob lives in the bench-config
//! JSON; CLI carries only what changes per invocation.

// mimalloc as global allocator — measured ~3x throughput on uni-db's
// concurrent_mutations benchmark (uni-db commit 65399a2b).
#[global_allocator]
static GLOBAL: uni_db::MiMalloc = uni_db::MiMalloc;

use uniko_bench::{
    IngestObserver, bench_config::BenchConfig, data, eval, events, ingest, pricing as pricing_mod,
    query, report,
};

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;

use data::{QuestionCategory, build_evidence_lookup, parse_sessions, resolve_evidence};
use query::QueryResult;

/// LoCoMo benchmark for uniko cognitive memory.
///
/// Most knobs (models, embedder, reranker, NLP cascade, recall
/// pipeline, pricing) now live in the bench-config JSON.  The CLI
/// carries only what varies per invocation.
#[derive(Parser)]
#[command(name = "uniko-bench", about = "Run LoCoMo benchmark against uniko")]
struct Cli {
    /// Path to the bench-config JSON profile.
    ///
    /// See [`uniko_bench::bench_config::BenchConfig`] doc comment for
    /// the schema, and `crates/uniko-bench/bench-configs/` for
    /// ready-to-use starter profiles.
    #[arg(long)]
    bench_config: PathBuf,

    /// Path to the LoCoMo dataset JSON (e.g. `data/locomo10.json`).
    #[arg(long)]
    data: PathBuf,

    /// Output JSON report file path.
    #[arg(long, default_value = "locomo_results.json")]
    output: PathBuf,

    /// Run only specific conversation IDs (comma-separated).
    #[arg(long)]
    conversations: Option<String>,

    /// Run only specific question categories (comma-separated, 1-5).
    #[arg(long)]
    categories: Option<String>,

    /// Directory for persistent KB storage.
    #[arg(long, default_value = "data/kb")]
    ingest_dir: PathBuf,

    /// Reuse existing KB from --ingest-dir (skip ingestion).
    #[arg(long)]
    reuse: bool,

    /// Path to a xervo model catalog JSON file (advanced — overrides
    /// the catalog built from the bench-config when present).
    #[arg(long)]
    catalog: Option<PathBuf>,

    /// Path to schema JSON file (advanced — overrides the default
    /// schema registration).
    #[arg(long)]
    schema: Option<PathBuf>,
}

/// Legacy flags that have been folded into the bench-config JSON.
/// Listed here so the pre-parse pass can emit migration-specific
/// errors when an operator still passes one of them.
const RETIRED_FLAGS: &[(&str, &str)] = &[
    ("--llm-alias", "models.gen.alias"),
    ("--llm-model-id", "models.gen.model_id"),
    ("--llm-provider", "models.gen.provider"),
    ("--llm-base-url", "models.gen.base_url"),
    (
        "--llm-use-default-options",
        "models.gen.use_default_options",
    ),
    ("--judge-alias", "models.judge.alias"),
    ("--judge-model-id", "models.judge.model_id"),
    ("--judge-provider", "models.judge.provider"),
    ("--judge-base-url", "models.judge.base_url"),
    ("--no-judge", "judge_enabled (set to false)"),
    ("--retrieval-only", "retrieval_only (set to true)"),
    (
        "--extract-triples-llm-alias",
        "models.extract_triples.alias (set models.extract_triples to an LlmAlias object)",
    ),
    (
        "--no-phase2-graph",
        "recall.phase2_graph_enabled (set false)",
    ),
    (
        "--no-phase2-temporal",
        "recall.phase2_temporal_enabled (set false)",
    ),
    ("--embedding", "models.embedder.preset"),
    (
        "--embedding-dim",
        "models.embedder.inline.dimensions (use inline embedder)",
    ),
    ("--no-reranker", "models.reranker.enabled (set false)"),
    ("--reranker-model", "models.reranker.model_id"),
    ("--reranker-style", "models.reranker.style"),
    ("--reranker-top-n", "models.reranker.top_n"),
    ("--recall-limit", "recall.limit"),
    ("--phase1-strategy", "recall.phase1_strategy"),
    ("--phase1-boost-alpha", "recall.phase1_boost_alpha"),
    ("--variants", "recall.variants (as JSON array)"),
    (
        "--no-consolidation-cluster",
        "recall.consolidation_cluster_objects (set false)",
    ),
    (
        "--no-date-augment-embedding",
        "recall.consolidation_date_augment_embedding (set false)",
    ),
    ("--pricing-csv", "cost.pricing_csv"),
    ("--events-jsonl", "cost.events_jsonl"),
    ("--no-events", "cost.no_events"),
];

/// Scan `argv` for retired flags before clap parses, so we can emit
/// a migration-pointer error instead of clap's generic "unexpected
/// argument" message.
///
/// # Errors
///
/// Returns an error with the migration hint when a retired flag is
/// detected.  The caller should propagate it to `main`.
fn reject_retired_flags(argv: &[String]) -> Result<()> {
    for arg in argv {
        let name = arg.split('=').next().unwrap_or(arg);
        if let Some((_, replacement)) = RETIRED_FLAGS.iter().find(|(flag, _)| *flag == name) {
            anyhow::bail!(
                "{name} is no longer accepted; move it under {replacement} in your \
                 --bench-config JSON.  See crates/uniko-bench/src/bench_config.rs for \
                 the full schema and crates/uniko-bench/bench-configs/ for examples."
            );
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                "uniko_bench=info,uniko_memory=warn,uniko_extract=warn,uniko_store=warn"
                    .parse()
                    .unwrap()
            }),
        )
        .init();

    let argv: Vec<String> = std::env::args().collect();
    reject_retired_flags(&argv)?;
    let cli = Cli::parse();
    let bench_cfg = BenchConfig::load(&cli.bench_config)
        .with_context(|| format!("loading bench config {}", cli.bench_config.display()))?;

    // Load dataset.
    tracing::info!(path = %cli.data.display(), "loading LoCoMo dataset");
    let samples = data::load_locomo(&cli.data)?;
    tracing::info!(conversations = samples.len(), "dataset loaded");

    // Filter conversations if specified.
    let samples: Vec<_> = if let Some(ref filter) = cli.conversations {
        let ids: Vec<&str> = filter.split(',').map(|s| s.trim()).collect();
        samples
            .into_iter()
            .filter(|s| ids.contains(&s.sample_id.as_str()))
            .collect()
    } else {
        samples
    };

    // Filter categories if specified.
    let category_filter: Option<Vec<u32>> = cli
        .categories
        .as_ref()
        .map(|c| c.split(',').filter_map(|s| s.trim().parse().ok()).collect());

    // Build UnikoConfig from defaults + bench-config overrides.
    let mut config = uniko_store::config::UnikoConfig {
        catalog_path: cli.catalog.clone(),
        schema_path: cli.schema.clone(),
        ..Default::default()
    };
    bench_cfg.apply_to_uniko_config(&mut config)?;

    // Build the LLM catalog (gen + judge + optional extract-triples).
    let extra_catalog = bench_cfg.build_catalog_specs();
    for spec in &extra_catalog {
        tracing::info!(
            alias = %spec.alias,
            provider = %spec.provider_id,
            model = %spec.model_id,
            options = %spec.options,
            "catalog entry"
        );
    }

    // One Xervo runtime shared across every KB opened in this run. Built from a
    // catalog (each model loads once) and opened over with
    // `open_shared`. (Before uni-db 5.2.1 this was also the only open path
    // that let a hybrid bge-m3 KB reopen, rustic-ai/uni-db#130; the
    // catalog-open path now checks embedding capability as well.)
    let shared_runtime = uniko_store::SharedRuntime::build(
        &config,
        extra_catalog.clone(),
        uniko_store::PrefetchPolicy::Eager,
    )
    .await
    .context("building shared Xervo runtime")?;

    let llm_alias: Option<&str> = if bench_cfg.retrieval_only {
        None
    } else {
        Some(bench_cfg.models.generator.alias.as_str())
    };
    let llm_use_default_options = bench_cfg.gen_use_default_options();
    let judge_alias: Option<&str> = if !bench_cfg.judge_enabled || bench_cfg.retrieval_only {
        None
    } else {
        Some(bench_cfg.models.judge.alias.as_str())
    };
    let gen_model_id = bench_cfg.models.generator.model_id.clone();
    let judge_model_id = bench_cfg.models.judge.model_id.clone();

    // Load pricing table and open the per-event JSONL writer.
    let pricing = match pricing_mod::Pricing::load(&bench_cfg.cost.pricing_csv) {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!(
                path = %bench_cfg.cost.pricing_csv.display(),
                error = %e,
                "could not load pricing CSV; cost columns will be zero",
            );
            pricing_mod::Pricing::empty()
        }
    };
    let events_writer: Option<events::EventWriter> = if bench_cfg.cost.no_events {
        None
    } else {
        let events_path = bench_cfg.cost.events_jsonl.clone().unwrap_or_else(|| {
            let stem = cli
                .output
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "locomo_results".to_string());
            cli.output
                .parent()
                .unwrap_or(std::path::Path::new("."))
                .join(format!("{stem}_events.jsonl"))
        });
        match events::EventWriter::open(&events_path) {
            Ok(w) => {
                tracing::info!(path = %events_path.display(), "events JSONL opened");
                Some(w)
            }
            Err(e) => {
                tracing::warn!(
                    path = %events_path.display(),
                    error = %e,
                    "could not open events JSONL; continuing without per-event capture",
                );
                None
            }
        }
    };
    let embedding_model_id = config.embedding.model_id.clone();

    // Run benchmark.
    let mut all_results: Vec<(QueryResult, f64, Option<f64>)> = Vec::new();
    let bench_start = Instant::now();

    for (conv_idx, sample) in samples.iter().enumerate() {
        tracing::info!(
            "[{}/{}] Processing {}",
            conv_idx + 1,
            samples.len(),
            sample.sample_id,
        );

        let sessions = parse_sessions(&sample.sample_id, &sample.conversation)
            .with_context(|| format!("parsing sessions for {}", sample.sample_id))?;

        tracing::info!(
            sessions = sessions.len(),
            turns = sessions.iter().map(|s| s.turns.len()).sum::<usize>(),
            questions = sample.qa.len(),
            "conversation structure",
        );

        let kb_dir = cli.ingest_dir.join(&sample.sample_id);
        // Decide reuse-vs-ingest before opening: `open_shared` creates the
        // dir, so `exists()` must be checked first. Both branches open via the
        // shared prebuilt runtime.
        let reuse_existing = cli.reuse && kb_dir.exists();
        let kb = uniko_bench::open_kb_with_runtime(&kb_dir, config.clone(), shared_runtime.clone())
            .await?;
        if reuse_existing {
            tracing::info!(path = %kb_dir.display(), "reusing existing KB");
        } else {
            let ingest_start = Instant::now();
            let observer = IngestObserver {
                events: events_writer.as_ref(),
                pricing: Some(&pricing),
                embedding_model: Some(embedding_model_id.clone()),
            };
            ingest::ingest_into_kb_with_observer(&kb, sample, &sessions, &observer).await?;
            tracing::info!(
                elapsed_ms = ingest_start.elapsed().as_millis(),
                "ingestion complete"
            );
        }

        // P4 + P5 + P6 sweep — identical to the legacy run flow.
        let triple_source = match bench_cfg.models.extract_triples.as_ref() {
            Some(triples) => uniko_memory::consolidation::TripleSource::Llm {
                alias: triples.alias.clone(),
            },
            None => uniko_memory::consolidation::TripleSource::SrlDep,
        };
        uniko_bench::run_post_ingest_sweep(&kb, &sample.sample_id, &triple_source).await;

        let evidence_lookup = build_evidence_lookup(&sessions);

        let bench_agent_id = uniko_bench::ensure_bench_agent(&kb, &sample.sample_id).await;

        let questions: Vec<_> = sample
            .qa
            .iter()
            .filter(|qa| {
                category_filter
                    .as_ref()
                    .is_none_or(|cats| cats.contains(&qa.category))
            })
            .collect();

        for (q_idx, qa) in questions.iter().enumerate() {
            if (q_idx + 1) % 50 == 0 {
                tracing::info!(
                    "[{}/{}] questions processed for {}",
                    q_idx + 1,
                    questions.len(),
                    sample.sample_id,
                );
            }

            let q_wall_start = Instant::now();
            let evidence_texts = resolve_evidence(qa, &evidence_lookup);
            let mut qr = query::run_query(
                &kb,
                &sample.sample_id,
                q_idx,
                qa,
                &evidence_texts,
                llm_alias,
                llm_use_default_options,
            )
            .await?;
            // `run_query` populates `answer_model` with the alias
            // (e.g. `llm/gen`); pricing.csv is keyed by HF model id,
            // so substitute it from the bench config here.
            if !qr.answer_model.is_empty() {
                qr.answer_model = gen_model_id.clone();
            }

            let f1 = eval::token_f1(&qr.predicted_answer, &qr.gold_answer, qr.category);

            let judge_score = if let Some(alias) = judge_alias
                && qr.category != QuestionCategory::Adversarial
            {
                match eval::llm_judge_with_usage(
                    &kb,
                    &qr.question,
                    &qr.gold_answer,
                    &qr.predicted_answer,
                    alias,
                )
                .await
                {
                    Ok(outcome) => {
                        qr.judge_latency_ms = Some(outcome.latency_ms);
                        qr.judge_input_tokens = outcome.prompt_tokens;
                        qr.judge_output_tokens = outcome.completion_tokens;
                        qr.judge_model = Some(judge_model_id.clone());
                        Some(outcome.score)
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "LLM judge failed");
                        None
                    }
                }
            } else {
                None
            };

            if let Some(writer) = events_writer.as_ref() {
                let answer_cost = qr
                    .answer_input_tokens
                    .and_then(|t| pricing.cost_input(&qr.answer_model, t))
                    .unwrap_or(0.0)
                    + qr.answer_output_tokens
                        .and_then(|t| pricing.cost_output(&qr.answer_model, t))
                        .unwrap_or(0.0);
                let judge_cost = qr
                    .judge_model
                    .as_deref()
                    .map(|m| {
                        qr.judge_input_tokens
                            .and_then(|t| pricing.cost_input(m, t))
                            .unwrap_or(0.0)
                            + qr.judge_output_tokens
                                .and_then(|t| pricing.cost_output(m, t))
                                .unwrap_or(0.0)
                    })
                    .unwrap_or(0.0);
                let event = events::BenchEvent::Query {
                    sample_id: qr.sample_id.clone(),
                    question_index: qr.question_index,
                    ts_unix_ms: events::now_unix_ms(),
                    wall_ms: q_wall_start.elapsed().as_millis() as u64,
                    recall_ms: qr.recall_latency_ms,
                    generation_ms: qr.generation_latency_ms,
                    judge_ms: qr.judge_latency_ms,
                    answer_input_tokens: qr.answer_input_tokens,
                    answer_output_tokens: qr.answer_output_tokens,
                    answer_model: qr.answer_model.clone(),
                    judge_input_tokens: qr.judge_input_tokens,
                    judge_output_tokens: qr.judge_output_tokens,
                    judge_model: qr.judge_model.clone(),
                    evidence_found: qr.evidence_found,
                    evidence_total: qr.evidence_total,
                    f1,
                    judge_score,
                    cost_usd: answer_cost + judge_cost,
                };
                if let Err(e) = writer.write(&event) {
                    tracing::warn!(error = %e, "failed to append Query event");
                }
            }

            let outcome = if f1 >= 0.5 { "success" } else { "failure" };
            // Bench-specific extras — sample_id / question_index /
            // category aren't part of the library's Episode schema, so
            // we attach them via `extra_state`.  Built-in fields
            // (question, answer, recall_*, answer_*) win on key clash,
            // so e.g. a stray "question" key here would be ignored.
            let extra_state = serde_json::json!({
                "category": format!("{:?}", qr.category),
                "sample_id": qr.sample_id.clone(),
                "question_index": qr.question_index,
            });
            let recall_ids: Vec<i64> = qr.recall_bundle.iter().map(|item| item.node_id).collect();
            let params = uniko_memory::RecordQueryEpisodeParams {
                question: &qr.question,
                answer: &qr.predicted_answer,
                recall_node_ids: &recall_ids,
                recall_coverage: qr.coverage,
                recall_tokens: qr.total_tokens,
                answer_input_tokens: qr.answer_input_tokens,
                answer_output_tokens: qr.answer_output_tokens,
                answer_model: Some(qr.answer_model.as_str()),
                importance: Some(f1.clamp(0.0, 1.0)),
                outcome: Some(outcome),
                action_type: Some("retrieve"),
                extra_state: Some(extra_state),
            };
            if let Err(e) = uniko_memory::record_query_episode(&kb, &bench_agent_id, params).await {
                tracing::debug!(error = %e, "episode recording failed");
            }

            all_results.push((qr, f1, judge_score));
        }

        uniko_bench::shutdown_kb(kb, &sample.sample_id).await;
        tracing::info!(
            sample_id = %sample.sample_id,
            "conversation complete",
        );

        let partial = report::aggregate_with_pricing(&all_results, conv_idx + 1, &pricing);
        if let Err(e) =
            report::write_json_with_pricing(&all_results, &partial, &cli.output, &pricing)
        {
            tracing::warn!(error = %e, "checkpoint write failed (continuing)");
        }
    }

    let bench_elapsed = bench_start.elapsed();
    tracing::info!(
        total_secs = bench_elapsed.as_secs(),
        total_questions = all_results.len(),
        "benchmark complete",
    );

    let report = report::aggregate_with_pricing(&all_results, samples.len(), &pricing);
    report::print_report(&report);
    report::write_json_with_pricing(&all_results, &report, &cli.output, &pricing)?;
    tracing::info!(path = %cli.output.display(), "results written");

    Ok(())
}
