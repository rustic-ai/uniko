//! The uniko-bench integration tests, in one binary.
//!
//! Each suite or repro that used to be its own test target is a module here,
//! so the stack (uni-db, Lance, DataFusion, ONNX Runtime) is linked once
//! rather than once per file. Add a new suite as a module below, not as a new
//! file under `tests/`: `autotests = false` in Cargo.toml.

// Shared helpers, used by the suites as `crate::common`.
mod common;

#[path = "diagnostics/main.rs"]
mod diagnostics;

#[path = "smoke_test.rs"]
mod smoke_test;

#[path = "xervo_nlp_argmax_repro.rs"]
mod xervo_nlp_argmax_repro;
