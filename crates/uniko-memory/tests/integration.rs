//! The uniko-memory integration tests, in one binary.
//!
//! Each suite or repro that used to be its own test target is a module here,
//! so the stack (uni-db, Lance, DataFusion, ONNX Runtime) is linked once
//! rather than once per file. Add a new suite as a module below, not as a new
//! file under `tests/`: `autotests = false` in Cargo.toml.

// Shared helpers, used by the suites as `crate::common`.
mod common;

#[path = "cloud_e2e.rs"]
mod cloud_e2e;

#[path = "finalize_idempotence_repro.rs"]
mod finalize_idempotence_repro;

#[path = "lifecycle_e2e/main.rs"]
mod lifecycle_e2e;

#[path = "nl_to_cypher_e2e.rs"]
mod nl_to_cypher_e2e;

#[path = "pipeline_e2e/main.rs"]
mod pipeline_e2e;

#[path = "recall_deadlock_bisect.rs"]
mod recall_deadlock_bisect;

#[path = "recall_e2e/main.rs"]
mod recall_e2e;

#[path = "recording_e2e/main.rs"]
mod recording_e2e;

#[path = "unit_crash_atomicity.rs"]
mod unit_crash_atomicity;
