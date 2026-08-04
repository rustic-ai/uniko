//! `export-schema` — regenerate `config/schema.json` from the code's schema.
//!
//! The persisted snapshot is produced by registering uniko's builder-based
//! schema into a throwaway in-memory KB and dumping the installed result. It
//! is an *input* to anything that sets [`UnikoConfig::schema_path`], so it has
//! to be regenerated deliberately whenever the schema in
//! `uniko_store::schema::constants` changes, and committed on its own.
//!
//! ```bash
//! # Overwrite the tracked snapshot (run from the workspace root)
//! cargo run --bin export-schema
//!
//! # Write somewhere else — e.g. to diff against the tracked copy
//! cargo run --bin export-schema -- --out /tmp/schema.json
//! ```
//!
//! This lived in `tests/diagnostics/` as `#[tokio::test] export_schema_json`
//! from 2026-04-20 until 2026-08-03. Its own docs called it a "one-time
//! utility" and the diagnostics module asserted "none execute in CI", but
//! nothing enforced that: it carried no `#[ignore]`, so every
//! `cargo nextest run --workspace` — CI included — silently rewrote the
//! tracked file. A generator whose product is a side effect does not belong
//! in the test harness at all, so it is a binary now: no test-selection flag
//! can reach it.

// mimalloc as global allocator — measured ~3x throughput on uni-db's
// concurrent_mutations benchmark (uni-db commit 65399a2b).
#[global_allocator]
static GLOBAL: uni_db::MiMalloc = uni_db::MiMalloc;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use uni_db::ModelAliasSpec;
use uniko_store::KnowledgeBase;
use uniko_store::config::UnikoConfig;

/// Export uniko's registered schema to a JSON file.
#[derive(Parser)]
#[command(
    name = "export-schema",
    about = "Regenerate config/schema.json from the code's schema registration"
)]
struct Cli {
    /// Destination path, relative to the current directory.
    #[arg(long, short = 'o', default_value = "config/schema.json")]
    out: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let kb =
        KnowledgeBase::in_memory_with_xervo(UnikoConfig::default(), Vec::<ModelAliasSpec>::new())
            .await
            .context("open in-memory KB")?;

    kb.db()
        .save_schema(&cli.out)
        .await
        .with_context(|| format!("save schema to {}", cli.out.display()))?;

    // Read back and parse: `save_schema` is the only writer, so this catches a
    // truncated or malformed dump before it is committed as an input others
    // load.
    let content = std::fs::read_to_string(&cli.out)
        .with_context(|| format!("read back {}", cli.out.display()))?;
    let parsed: serde_json::Value = serde_json::from_str(&content)
        .with_context(|| format!("{} is not valid JSON", cli.out.display()))?;

    let labels = parsed
        .get("labels")
        .and_then(serde_json::Value::as_object)
        .map_or(0, serde_json::Map::len);
    let edges = parsed
        .get("edge_types")
        .and_then(serde_json::Value::as_object)
        .map_or(0, serde_json::Map::len);

    println!(
        "wrote {} ({} bytes, {labels} labels, {edges} edge types)",
        cli.out.display(),
        content.len()
    );
    println!("commit this on its own, e.g. `schema: regenerate snapshot`");

    // Shut the store down before the runtime drops. Without this, Lance's
    // background tasks are cancelled mid-flight and `lance-datafusion` panics
    // on a `JoinError::Cancelled` in a worker thread during teardown — noise
    // after a successful export, but indistinguishable from a real failure to
    // anyone reading the output.
    kb.shutdown().await.context("shut down the store")?;
    Ok(())
}
