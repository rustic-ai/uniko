//! Core extract integration test suites.
//!
//! Consolidated binary covering observation extraction, SRL/rule parity, NER,
//! and the ingest pipeline (including a real-PDF end-to-end). Each former
//! `tests/<name>.rs` file is included as a module so the suite links once
//! instead of once per file.
//!
//! Run: `cargo nextest run -p uniko-extract --test integration extract_tests::`

mod ingest_atomic_tests;
mod ingest_pdf_e2e;
#[cfg(feature = "pdf-ocr")]
mod ingest_pdf_tiered_test;
mod ingest_tests;
mod media_document_tests;
mod ner_tests;
mod observation_extraction_test;
mod observation_rules_parity_test;
mod observation_tests;
mod srl_extraction_test;
