//! # uniko-store — Layer 1: Graph Storage
//!
//! Wraps uni-db to provide typed graph storage, search (vector, fulltext, hybrid),
//! and Locy runtime for the uniko cognitive memory system.
//!
//! This is the lowest layer. It depends only on uni-db and external utility crates.
//!
//! Boundary (issue #2): the product crates
//! (`uniko-{memory,extract,cortex,pipes}`) access the graph **only** through
//! this layer's typed API, so uni-db is an implementation detail. The Cypher
//! they used to issue inline now lives in [`repository`] (decoded reads) and
//! [`operations`] (writes); model-runtime access goes through [`model`]; the
//! uni-db value types they legitimately name are re-exported here ([`Value`],
//! [`Transaction`], [`temporal`], [`xervo`]). A CI grep gate forbids
//! `use uni_db` / [`KnowledgeBase::db`] in those crates' `src/`.
//!
//! [`KnowledgeBase::db`] remains a `pub` escape hatch for **tests** and the
//! **benchmark** crate only (both out of the gate's scope); product code must
//! not reintroduce it.

pub mod blob_store;
pub mod config;
pub mod error;
pub mod id;
pub mod locks;
pub mod locy;
pub mod model;
pub mod operations;
pub mod repository;
pub mod schema;
pub mod search;
pub mod storage;
pub mod text;
pub mod types;

pub use error::{Result, UnikoError};
#[doc(inline)]
pub use storage::KnowledgeBase;
#[doc(inline)]
pub use storage::deletion::DeletionReport;
// Diagnostic batch-recording surface, only present with the
// `batch-record` feature (used by the bulk-vs-UNWIND benchmark).
#[cfg(feature = "batch-record")]
#[doc(inline)]
pub use storage::batch_record::{RecordedBatch, enable_batch_recording, take_recorded_batches};
pub use types::*;

// Re-export `ModelRuntime` so callers can name the type without
// taking a direct dependency on `uni_xervo`. Needed for multi-KB
// workflows that share one ONNX session — see
// [`KnowledgeBase::build_shared_runtime`] and
// [`KnowledgeBase::open_with_runtime`]. uni-db wraps `ModelRuntime`
// internally but does not re-export it.
pub use uni_xervo::runtime::ModelRuntime;

// --- Sealed uni-db surface (issue #2) ---------------------------------
//
// These are the *only* uni-db types higher uniko crates legitimately
// need to name. Re-exporting them here means consumers write
// `use uniko_store::Value;` instead of `use uni_db::Value;`, so the
// CI gate can forbid `use uni_db` outside this crate while the graph
// stays reachable. Anything not re-exported here is an implementation
// detail of the storage layer and must be reached through a typed
// `KnowledgeBase` method, not the raw handle.
pub use uni_db::{RetryOptions, Transaction, Value};
/// Temporal / bitemporal value types used on `valid_at` BTIC columns.
///
/// [`Btic`](uni_db::common::uni_btic::Btic) is the in-memory bitemporal
/// interval that consolidation and recall manipulate directly; the
/// [`schema::btic`](crate::schema::btic) helpers construct/compare it.
pub mod temporal {
    pub use uni_db::common::TemporalValue;
    pub use uni_db::common::uni_btic::Btic;
}
/// Model-runtime value types used by the [`KnowledgeBase`] generation
/// and NLP seams (see `model` module). Re-exported so callers building
/// prompts never reach `uni_db::xervo` directly.
///
/// [`ModelAliasSpec`], [`ModelTask`] and [`WarmupPolicy`] describe a
/// catalog entry. They are part of this crate's own surface already —
/// [`KnowledgeBase::in_memory_with_xervo`] and its siblings take
/// `Vec<ModelAliasSpec>` — so a caller registering an extra model (the
/// facade's `LlmSpec`) has to be able to name them.
pub mod xervo {
    pub use uni_db::xervo::{GenerationOptions, Message};
    pub use uni_db::{ModelAliasSpec, ModelTask, WarmupPolicy};
}
