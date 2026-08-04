//! Bench diagnostic and one-off investigation tools.
//!
//! Every member here is `#[ignore]`d and is run manually against a specific
//! KB snapshot — none execute in CI. They are consolidated into this single
//! binary so the suite links once instead of once per file.
//!
//! That claim is now enforced rather than asserted. It previously read "or an
//! export utility", carving out `export_schema`, which carried no `#[ignore]`
//! and so rewrote the tracked `config/schema.json` on every
//! `cargo nextest run --workspace` — CI included. It is a binary now:
//! `cargo run --bin export-schema`. Nothing in this module writes to a tracked
//! file; a member that needs to belongs in `src/*_main.rs` instead.
//!
//! Run an individual tool, e.g.:
//! `cargo nextest run -p uniko-bench --test diagnostics graph_debug --run-ignored all --no-capture`

// Shared helpers live in `tests/common/mod.rs`; include them once for the
// whole binary so each member can reach them via `crate::common::*`.
#[path = "../common/mod.rs"]
mod common;

mod dump_obs_chunks;
mod graph_debug;
mod recall_debug;
mod review_observations;
mod single_hop_debug;
