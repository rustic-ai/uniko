//! Recall-engine end-to-end tests.
//!
//! Consolidated binary covering the multi-phase recall cascade: phase-2
//! vector/fulltext fusion, temporal-interval channel, graph spreading
//! activation, cross-modal firing, and lazy modality dormancy. Each former
//! `tests/<name>.rs` file is included as a module so the suite links once
//! instead of once per file.
//!
//! Run: `cargo nextest run -p uniko-memory --test integration recall_e2e::`

mod recall_cross_modal_fire_e2e;
mod recall_graph_activation_e2e;
mod recall_modality_lazy_e2e;
mod recall_phase2_e2e;
mod recall_temporal_e2e;
