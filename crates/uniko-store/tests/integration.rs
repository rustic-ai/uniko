//! The uniko-store integration tests, in one binary.
//!
//! Each suite or repro that used to be its own test target is a module here,
//! so the stack (uni-db, Lance, DataFusion, ONNX Runtime) is linked once
//! rather than once per file. Add a new suite as a module below, not as a new
//! file under `tests/`: `autotests = false` in Cargo.toml.
// Hoisted from the suites that set it: it is a crate-level attribute.
#![recursion_limit = "256"]

#[path = "bug_repros/main.rs"]
mod bug_repros;

#[path = "consolidation_write_slowness_repro.rs"]
mod consolidation_write_slowness_repro;

#[path = "entity_upsert_bench.rs"]
mod entity_upsert_bench;

#[path = "hybrid_column_decode_repro.rs"]
mod hybrid_column_decode_repro;

#[path = "hybrid_query_perf_repro.rs"]
mod hybrid_query_perf_repro;

#[path = "phase2_deadlock_reduction.rs"]
mod phase2_deadlock_reduction;

#[path = "sparse_query_alias_repro.rs"]
mod sparse_query_alias_repro;

#[path = "stack_depth_repro.rs"]
mod stack_depth_repro;

#[path = "store_tests/main.rs"]
mod store_tests;

#[path = "unidb_bytes_residual_repro.rs"]
mod unidb_bytes_residual_repro;

#[path = "unidb_bytes_return_repro.rs"]
mod unidb_bytes_return_repro;

#[path = "unidb_filtered_search_repro.rs"]
mod unidb_filtered_search_repro;

#[path = "unidb_hybrid_reopen_repro.rs"]
mod unidb_hybrid_reopen_repro;

#[path = "unidb_in_tx_read_repro.rs"]
mod unidb_in_tx_read_repro;

#[path = "unidb_one_model_multi_task_repro.rs"]
mod unidb_one_model_multi_task_repro;

#[path = "unidb_schema_reapply_repro.rs"]
mod unidb_schema_reapply_repro;

#[path = "unidb_tx_snapshot_lag_repro.rs"]
mod unidb_tx_snapshot_lag_repro;

#[path = "unidb_azure_hybrid_repro.rs"]
mod unidb_azure_hybrid_repro;
#[path = "unidb_object_store_repro.rs"]
mod unidb_object_store_repro;

#[path = "unidb_two_handles_repro.rs"]
mod unidb_two_handles_repro;

#[path = "unidb_wal_no_manifest_repro.rs"]
mod unidb_wal_no_manifest_repro;

#[path = "unidb_windows_fsync_repro.rs"]
mod unidb_windows_fsync_repro;
