//! NLP pipeline diagnostic and audit tools.
//!
//! Every member is `#[ignore]`d and ONNX-model gated — they dump pipeline
//! output, audit CLS/NER quality, and inspect SRL frames against fixed
//! sentence sets. None run in CI. Consolidated into one binary so they link
//! once instead of once per file.
//!
//! Run an individual tool, e.g.:
//! `cargo nextest run -p uniko-extract --test integration nlp_diagnostics::cls_quality_audit --run-ignored all --no-capture`

mod cls_quality_audit;
mod cls_raw_dump;
mod dump_nlp_pipeline;
mod dump_phase_b_target_arcs;
mod dump_srl_frames;
mod ner_quality_test;
mod obs_for_misses_test;
