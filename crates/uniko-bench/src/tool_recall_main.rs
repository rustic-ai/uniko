//! `tool-recall`: score uniko recall on answers that live in tool output.
//!
//! ```text
//! cargo run --release -p uniko-bench --bin tool-recall -- \
//!     --mode turns --sessions 24 --seed 7 --output results.jsonl
//! ```
//!
//! Prints a per-category table to stderr and, with `--output`, appends one
//! JSON line per run.

use std::io::Write as _;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use serde_json::json;
use uniko_bench::tool_recall::run::{
    AGENT_ID, IngestMode, Summary, by_category, evaluate, evaluate_reader, ingest, summarize,
    summarize_reader,
};
use uniko_bench::tool_recall::synth::generate;
use uniko_memory::Uniko;

#[derive(Parser)]
#[command(about = "Score recall on answers that live in agent tool output")]
struct Args {
    /// How tool calls are recorded.
    #[arg(long, value_enum, default_value = "turns")]
    mode: IngestMode,
    /// Number of generated sessions.
    #[arg(long, default_value_t = 24)]
    sessions: usize,
    /// Generator seed.
    #[arg(long, default_value_t = 7)]
    seed: u64,
    /// Skip `Session::finalize`, so no session transcript chunks exist.
    #[arg(long)]
    no_finalize: bool,
    /// Free-form label stored with the result line.
    #[arg(long, default_value = "")]
    label: String,
    /// Append one JSON line with the results.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Print each question's score to stderr.
    #[arg(long)]
    verbose: bool,
}

fn row(name: &str, s: &Summary) -> String {
    format!(
        "{name:<13} {:>4} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>7.0} {:>6}",
        s.n,
        s.hit_at_1,
        s.hit_at_5,
        s.hit_at_k,
        s.hit_at_budget,
        s.mrr,
        s.coverage,
        s.context_complete,
        s.stale_above,
        s.mean_tokens,
        s.p50_latency_ms
    )
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .with_writer(std::io::stderr)
        .init();
    let args = Args::parse();
    let data = generate(args.seed, args.sessions);
    let calls: usize = data
        .sessions
        .iter()
        .flat_map(|s| &s.steps)
        .filter(|s| matches!(s, uniko_bench::tool_recall::synth::Step::Tool { .. }))
        .count();
    eprintln!(
        "{} sessions, {calls} tool calls, {} questions",
        data.sessions.len(),
        data.questions.len()
    );

    let dir = tempfile::tempdir()?;
    let t_build = std::time::Instant::now();
    let uniko = Uniko::builder().path(dir.path()).build().await?;
    let build_ms = t_build.elapsed().as_millis() as u64;
    let agent = uniko.agent(AGENT_ID);

    let ingest_ms = ingest(&agent, &data, args.mode, !args.no_finalize).await?;
    eprintln!("ingested in {:.1} s", ingest_ms as f64 / 1000.0);

    let scores = evaluate(&agent, &data).await?;
    if args.verbose {
        for s in &scores {
            eprintln!("{}", serde_json::to_string(s)?);
        }
    }
    let overall = summarize(&scores);
    let cats = by_category(&scores);

    eprintln!(
        "{:<13} {:>4} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>7} {:>6}",
        "category",
        "n",
        "hit@1",
        "hit@5",
        "hit@k",
        "hit@2k",
        "mrr",
        "cover",
        "ctx",
        "stale",
        "tokens",
        "p50ms"
    );
    for (c, s) in &cats {
        let name = serde_json::to_value(c)?
            .as_str()
            .unwrap_or_default()
            .to_string();
        eprintln!("{}", row(&name, s));
    }
    eprintln!("{}", row("all", &overall));

    let mut kinds = std::collections::BTreeMap::<String, usize>::new();
    for s in &scores {
        *kinds
            .entry(s.hit_kind.clone().unwrap_or_else(|| "miss".into()))
            .or_default() += 1;
    }
    eprintln!("hit item kinds: {kinds:?}");

    let reader = evaluate_reader(&agent, &data).await?;
    let reader_cats = summarize_reader(&reader);
    eprintln!(
        "read tools (one search, {} hits):",
        uniko_bench::tool_recall::run::READER_HITS
    );
    eprintln!(
        "{:<13} {:>4} {:>6} {:>6} {:>6} {:>7} {:>7}",
        "category", "n", "hit@1", "hit@5", "ctx", "abstain", "tokens"
    );
    for (c, s) in &reader_cats {
        let name = serde_json::to_value(c)?
            .as_str()
            .unwrap_or_default()
            .to_string();
        eprintln!(
            "{name:<13} {:>4} {:>6.3} {:>6.3} {:>6.3} {:>7.3} {:>7.0}",
            s.n, s.hit_at_1, s.hit_at_5, s.context_complete, s.abstained, s.mean_tokens
        );
    }

    if let Some(path) = &args.output {
        let line = json!({
            "label": args.label,
            "mode": args.mode,
            "seed": args.seed,
            "sessions": args.sessions,
            "finalize": !args.no_finalize,
            "tool_calls": calls,
            "questions": data.questions.len(),
            "build_ms": build_ms,
            "ingest_ms": ingest_ms,
            "overall": overall,
            "by_category": cats,
            "hit_kinds": kinds,
            "reader": reader_cats,
        });
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        writeln!(f, "{line}")?;
    }
    drop(agent);
    uniko.shutdown().await?;
    Ok(())
}
