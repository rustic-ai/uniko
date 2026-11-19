//! Pipeline-orchestration end-to-end tests.
//!
//! Consolidated binary covering the async PipelineSystem lifecycle
//! (backpressure, health) and the cortex-sweep trigger gating after
//! consolidation cycles. Each former `tests/<name>.rs` file is included as a
//! module so the suite links once instead of once per file.
//!
//! Run: `cargo nextest run -p uniko-memory --test pipeline_e2e`

mod cortex_trigger_e2e;
mod dead_letter_recovery;
mod pipeline_integration;
