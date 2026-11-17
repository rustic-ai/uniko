//! LongMemEval benchmark harness for uniko cognitive memory.
//!
//! Ingests LongMemEval items (each with its own chat history haystack),
//! runs recall queries, and evaluates retrieval quality and answer
//! correctness.
//!
//! Configuration goes through one `--bench-config <path>` JSON file
//! describing models / recall / LME knobs; CLI carries only what
//! changes per invocation (data, output, KB dir, reuse, question
//! filter, phase1 mode).
//!
//! # Usage
//!
//! ```bash
//! # Phase 1 gate (retrieval-only, SSU+SSA+MS categories)
//! cargo run -p uniko-bench --bin longmemeval-bench -- \
//!     --bench-config crates/uniko-bench/configs/lme_default.json \
//!     --data data/longmemeval_s_cleaned.json --phase1
//!
//! # Dev iteration (5 questions only)
//! cargo run -p uniko-bench --bin longmemeval-bench -- \
//!     --bench-config crates/uniko-bench/configs/lme_default.json \
//!     --data data/longmemeval_s_cleaned.json --max-questions 5
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use futures::stream::{self, StreamExt};

// mimalloc as global allocator — measured ~3x throughput on uni-db's
// concurrent_mutations benchmark (uni-db commit 65399a2b).
#[global_allocator]
static GLOBAL: uni_db::MiMalloc = uni_db::MiMalloc;

use uniko_bench::bench_config::BenchConfig;
use uniko_bench::longmemeval::data::{self, LmeQuestionType};
use uniko_bench::longmemeval::eval;
use uniko_bench::longmemeval::ingest;
use uniko_bench::longmemeval::query;
use uniko_bench::longmemeval::report;

#[derive(Parser)]
#[command(
    name = "longmemeval-bench",
    about = "Run LongMemEval benchmark against uniko"
)]
struct Cli {
    /// Path to bench config JSON (models, recall, LME knobs).
    #[arg(long)]
    bench_config: PathBuf,

    /// Path to LongMemEval JSON file (e.g., longmemeval_s_cleaned.json).
    #[arg(long)]
    data: PathBuf,

    /// Output JSON report file path.
    #[arg(long, default_value = "longmemeval_results.json")]
    output: PathBuf,

    /// Directory for persistent KB storage.
    #[arg(long, default_value = "data/lme_kb")]
    ingest_dir: PathBuf,

    /// Reuse existing KB from --ingest-dir (skip ingestion).
    #[arg(long)]
    reuse: bool,

    /// Phase 1 mode: filter to SSU+SSA+MS, retrieval-only.
    #[arg(long)]
    phase1: bool,

    /// Run only specific question IDs (comma-separated).
    #[arg(long)]
    questions: Option<String>,

    /// Max questions to process (for dev iteration).
    #[arg(long)]
    max_questions: Option<usize>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                "longmemeval_bench=info,uniko_bench=info,uniko_memory=warn,uniko_extract=info,uniko_store=warn"
                    .parse()
                    .unwrap()
            }),
        )
        .init();

    let cli = Cli::parse();

    let bench = BenchConfig::load(&cli.bench_config)
        .with_context(|| format!("loading bench config {}", cli.bench_config.display()))?;

    // Load dataset.
    tracing::info!(path = %cli.data.display(), "loading LongMemEval dataset");
    let mut items = data::load_longmemeval(&cli.data)?;
    tracing::info!(total_items = items.len(), "dataset loaded");

    // Apply question type filter.
    let type_filter: Option<Vec<LmeQuestionType>> = if cli.phase1 {
        Some(vec![
            LmeQuestionType::SingleSessionUser,
            LmeQuestionType::SingleSessionAssistant,
            LmeQuestionType::MultiSession,
        ])
    } else {
        bench.lme.question_types.as_ref().map(|types| {
            types
                .iter()
                .filter_map(|s| LmeQuestionType::from_shorthand(s.trim()))
                .collect()
        })
    };

    if let Some(ref filter) = type_filter {
        items.retain(|item| filter.contains(&item.question_type));
        tracing::info!(
            filtered = items.len(),
            types = ?filter.iter().map(|t| t.name()).collect::<Vec<_>>(),
            "filtered by question type",
        );
    }

    // Filter by specific question IDs.
    if let Some(ref ids_str) = cli.questions {
        let ids: Vec<&str> = ids_str.split(',').map(|s| s.trim()).collect();
        items.retain(|item| ids.contains(&item.question_id.as_str()));
        tracing::info!(filtered = items.len(), "filtered by question ID");
    }

    // Apply max questions limit.
    if let Some(max) = cli.max_questions {
        items.truncate(max);
        tracing::info!(truncated = items.len(), "limited to max questions");
    }

    // LLM mode: --phase1 implies retrieval-only.  Otherwise, both gen
    // and judge are driven from BenchConfig.models.
    let retrieval_only = cli.phase1 || bench.retrieval_only;
    let llm_alias_owned: Option<String> = if retrieval_only {
        None
    } else {
        Some(bench.models.generator.alias.clone())
    };
    let judge_alias_owned: Option<String> = if !bench.judge_enabled || retrieval_only {
        None
    } else {
        Some(bench.models.judge.alias.clone())
    };

    let mut config = uniko_store::config::UnikoConfig::default();
    bench
        .apply_to_uniko_config(&mut config)
        .context("applying bench config to UnikoConfig")?;

    let extra_catalog = bench.build_catalog_specs();
    for spec in &extra_catalog {
        tracing::info!(
            alias = %spec.alias,
            provider = %spec.provider_id,
            model = %spec.model_id,
            options = %spec.options,
            "catalog entry"
        );
    }

    // Items run concurrently up to `lme.question_concurrency`.
    type ResultEntry = (query::LmeQueryResult, Option<f64>);
    let all_results: Arc<tokio::sync::Mutex<Vec<ResultEntry>>> =
        Arc::new(tokio::sync::Mutex::new(Vec::with_capacity(items.len())));
    let bench_start = Instant::now();
    let total_items = items.len();
    let question_concurrency = bench.lme.question_concurrency.max(1);
    let session_concurrency = bench.lme.session_concurrency.max(1);
    let token_budget = bench.lme.token_budget;

    // Build the ONNX/Xervo runtime once, share across every concurrent KB.
    let shared_runtime = uniko_store::SharedRuntime::build(
        &config,
        extra_catalog.clone(),
        uniko_store::PrefetchPolicy::Eager,
    )
    .await
    .context("building shared Xervo runtime")?;
    tracing::info!(
        question_concurrency,
        "shared Xervo runtime built; all KBs in this process will reuse it"
    );

    let bench = Arc::new(bench);
    let cli = Arc::new(cli);
    let config = Arc::new(config);

    stream::iter(items.into_iter().enumerate())
        .for_each_concurrent(question_concurrency, |(item_idx, item)| {
            let bench = bench.clone();
            let cli = cli.clone();
            let config = config.clone();
            let runtime = shared_runtime.clone();
            let all_results = all_results.clone();
            let llm_alias_owned = llm_alias_owned.clone();
            let judge_alias_owned = judge_alias_owned.clone();
            async move {
                tracing::info!(
                    "[{}/{}] Processing {} (type={}, sessions={})",
                    item_idx + 1,
                    total_items,
                    item.question_id,
                    item.question_type.name(),
                    item.haystack_sessions.len(),
                );

                let kb_dir = cli.ingest_dir.join(&item.question_id);
                let (kb, evidence_map) = if cli.reuse && kb_dir.exists() {
                    tracing::info!(path = %kb_dir.display(), "reusing existing KB");
                    match uniko_bench::open_kb_with_runtime(
                        &kb_dir,
                        (*config).clone(),
                        runtime.clone(),
                    )
                    .await
                    {
                        Ok(kb) => {
                            let evidence_map = ingest::EvidenceMap {
                                answer_session_ids: item
                                    .answer_session_ids
                                    .iter()
                                    .cloned()
                                    .collect(),
                                answer_message_ids: Vec::new(),
                                session_to_messages: std::collections::HashMap::new(),
                            };
                            (kb, evidence_map)
                        }
                        Err(e) => {
                            tracing::warn!(question_id = %item.question_id, error = %e, "open_kb failed; skipping item");
                            return;
                        }
                    }
                } else {
                    let ingest_start = Instant::now();
                    match ingest::ingest_item(
                        &item,
                        &kb_dir,
                        (*config).clone(),
                        runtime.clone(),
                        session_concurrency,
                    )
                    .await
                    {
                        Ok(result) => {
                            tracing::info!(
                                question_id = %item.question_id,
                                elapsed_ms = ingest_start.elapsed().as_millis(),
                                "ingestion complete"
                            );
                            result
                        }
                        Err(e) => {
                            tracing::warn!(question_id = %item.question_id, error = %e, "ingest failed; skipping item");
                            return;
                        }
                    }
                };

                // P4 Consolidation + P5 procedure sweep + P6 topic sweep.
                let triple_source = match bench.models.extract_triples.as_ref() {
                    Some(alias) => uniko_memory::consolidation::TripleSource::Llm {
                        alias: alias.alias.clone(),
                    },
                    None => uniko_memory::consolidation::TripleSource::SrlDep,
                };
                uniko_bench::run_post_ingest_sweep(&kb, &item.question_id, &triple_source).await;

                let bench_agent_id =
                    uniko_bench::ensure_bench_agent(&kb, &item.question_id).await;

                let gold = data::gold_answer(&item);
                let qr = match query::run_lme_query(
                    &kb,
                    &item.question_id,
                    &item.question,
                    item.question_type,
                    gold,
                    &evidence_map,
                    token_budget,
                    llm_alias_owned.as_deref(),
                    Some(&item.question_date),
                )
                .await
                {
                    Ok(qr) => qr,
                    Err(e) => {
                        tracing::warn!(question_id = %item.question_id, error = %e, "query failed; skipping item");
                        return;
                    }
                };

                let judge_score = if let Some(alias) = judge_alias_owned.as_deref() {
                    let is_abstention = LmeQuestionType::is_abstention(&item.question_id);
                    if is_abstention {
                        Some(eval::abstention_score(&qr.predicted_answer))
                    } else {
                        match eval::lme_judge(
                            &kb,
                            &item.question,
                            gold,
                            &qr.predicted_answer,
                            item.question_type,
                            false,
                            alias,
                        )
                        .await
                        {
                            Ok(score) => Some(score),
                            Err(e) => {
                                tracing::warn!(error = %e, "LLM judge failed");
                                None
                            }
                        }
                    }
                } else {
                    None
                };

                tracing::info!(
                    question_id = %item.question_id,
                    contains_answer = qr.context_contains_answer,
                    recall_at_5 = format!("{:.3}", qr.recall_at_5),
                    recall_items = qr.recall_items,
                    recall_ms = qr.recall_latency_ms,
                    "query complete",
                );

                let outcome = match judge_score {
                    Some(s) if s >= 0.5 => "success",
                    Some(_) => "failure",
                    None => {
                        if qr.context_contains_answer {
                            "success"
                        } else {
                            "failure"
                        }
                    }
                };
                let state = serde_json::json!({
                    "topic": item.question.clone(),
                    "question": item.question.clone(),
                    "question_type": item.question_type.name(),
                    "question_id": item.question_id.clone(),
                });
                let params = uniko_memory::RecordEpisodeParams {
                    action_type: "retrieve".into(),
                    outcome: Some(outcome.into()),
                    state: Some(state),
                    importance: Some(judge_score.unwrap_or(0.5).clamp(0.0, 1.0)),
                    ..Default::default()
                };
                if let Err(e) =
                    uniko_memory::record_episode(&kb, &bench_agent_id, params).await
                {
                    tracing::debug!(error = %e, "episode recording failed");
                }

                let mut results = all_results.lock().await;
                results.push((qr, judge_score));
                let partial = report::aggregate_lme(&results, total_items);
                if let Err(e) = report::write_lme_json(&results, &partial, &cli.output) {
                    tracing::warn!(error = %e, "checkpoint write failed (continuing)");
                }
            }
        })
        .await;

    let all_results = Arc::try_unwrap(all_results)
        .map_err(|_| anyhow::anyhow!("all_results still has outstanding refs"))?
        .into_inner();

    let bench_elapsed = bench_start.elapsed();
    tracing::info!(
        total_secs = bench_elapsed.as_secs(),
        total_questions = all_results.len(),
        "benchmark complete",
    );

    let lme_report = report::aggregate_lme(&all_results, total_items);
    report::print_lme_report(&lme_report);
    report::write_lme_json(&all_results, &lme_report, &cli.output)?;
    tracing::info!(path = %cli.output.display(), "results written");

    if cli.phase1 {
        let pass = lme_report.overall_context_contains_rate >= 0.90;
        println!(
            "\n  Phase 1 Gate: context_contains_answer@R5 = {:.1}% (threshold: 90%)",
            lme_report.overall_context_contains_rate * 100.0
        );
        if pass {
            println!("  Result: PASS");
        } else {
            println!("  Result: FAIL — Fix P1-P3 or recall cascade before proceeding to Phase 2.");
        }
        println!();
    }

    Ok(())
}
