// Same fix as uniko-memory: trait-check overflow on `Send`/`Sync` for
// the boxed pipeline-step futures, which transitively contain uni-db's
// session/tx types → datafusion + sqlparser AST. 256 clears it; matches
// uni-db's own crate setting.
#![recursion_limit = "256"]

//! # uniko-extract — Layer 3: Content Processing
//!
//! Content intelligence steps: NER (entity extraction), observation extraction,
//! chunking (recursive, tree-sitter, DOM), ingest pipeline, and embedding computation.
//!
//! Depends on `uniko-pipes` and `uniko-store` (plus `uni-db` and `uni-xervo`
//! directly). Implements the `Step` trait for pipeline integration.

pub mod embedding;
pub mod ingest;
pub mod ner;
#[cfg(feature = "onnx")]
pub mod nlp;
pub mod observations;

/// Cargo features of this crate and whether this build compiled them in.
pub const COMPILED_FEATURES: &[(&str, bool)] = &[
    ("code-parse", cfg!(feature = "code-parse")),
    ("onnx", cfg!(feature = "onnx")),
    ("pdf-ocr", cfg!(feature = "pdf-ocr")),
];
