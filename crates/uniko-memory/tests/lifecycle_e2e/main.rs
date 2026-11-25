//! Memory-lifecycle end-to-end tests.
//!
//! Consolidated binary covering consolidation (Observation → Fact, drift,
//! contradiction), rule lifecycle (creation, decay, re-promotion), and
//! policy/access-control filtering. Each former `tests/<name>.rs` file is
//! included as a module so the suite links once instead of once per file.
//!
//! Run: `cargo nextest run -p uniko-memory --test integration lifecycle_e2e::`

mod consolidation_e2e;
mod policy_e2e;
mod rule_execution_e2e;
mod rule_lifecycle_e2e;
