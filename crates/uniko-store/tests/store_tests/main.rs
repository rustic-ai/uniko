//! Core store integration test suites.
//!
//! Consolidated binary covering schema, storage, search, traversal, blob,
//! BTIC, Locy, migration, and UNWIND-batch behavior. Each former
//! `tests/<name>.rs` file is included as a module so the suite links once
//! instead of once per file.
//!
//! Run: `cargo nextest run -p uniko-store --test store_tests`

// Deep nested async future layout (lance dataset/IO cache via uni-db 2.0)
// exceeds rustc's default recursion limit of 128 when evaluating `Send`.
#![recursion_limit = "256"]

mod blob_store_roundtrip;
mod btic_tests;
mod deletion_tests;
mod hybrid_schema_tests;
mod locy_tests;
mod migration_artifact_content;
mod schema_artifact_content;
mod schema_completeness;
mod schema_tests;
mod scope_tests;
mod search_tests;
mod session_setup_lock_tests;
mod storage_tests;
mod traversal_tests;
mod unwind_batch_test;
