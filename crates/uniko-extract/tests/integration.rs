//! The uniko-extract integration tests, in one binary.
//!
//! Each suite or repro that used to be its own test target is a module here,
//! so the stack (uni-db, Lance, DataFusion, ONNX Runtime) is linked once
//! rather than once per file. Add a new suite as a module below, not as a new
//! file under `tests/`: `autotests = false` in Cargo.toml.
// Hoisted from the suites that set it: it is a crate-level attribute.
#![recursion_limit = "256"]

#[path = "colbert_query_embed.rs"]
mod colbert_query_embed;

#[path = "extract_tests/main.rs"]
mod extract_tests;

#[path = "nlp_diagnostics/main.rs"]
mod nlp_diagnostics;

#[path = "nlp_observation_integration_test.rs"]
mod nlp_observation_integration_test;

#[path = "nlp_v7_integration_test.rs"]
mod nlp_v7_integration_test;
