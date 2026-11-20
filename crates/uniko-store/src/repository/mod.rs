//! Repository layer (issue #2): decoded read/write methods on
//! [`KnowledgeBase`] that own the Cypher higher uniko crates used to
//! issue through the `kb.db()` escape hatch.
//!
//! Each method here mirrors the [`operations::facts`](crate::operations::facts)
//! house style: take a session/transaction internally, bind **every**
//! value as a parameter (never interpolate), reference labels/edges via
//! [`schema::constants`](crate::schema::constants) where practical, and
//! return decoded Rust structs — callers never see a `uni_db` `Record`,
//! `Value`, or `Session`.
//!
//! Grouped by domain (one submodule per consumer area). The submodules
//! are `impl KnowledgeBase` blocks plus their decoded return types.

pub mod consolidation;
pub mod entities;
pub mod episode;
pub mod ingest;
pub mod media;
pub mod policy;
pub mod procedures;
pub mod recall;
pub mod rules;
pub mod session;
pub mod sources;
pub mod topics;
pub mod working_memory;
