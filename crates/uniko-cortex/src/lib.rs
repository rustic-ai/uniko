//! # uniko-cortex — Layer 5: Higher Reasoning
//!
//! Procedure promotion, topic detection, MCTS planning, rule induction,
//! working memory traversal, and NL-to-Cypher translation.
//!
//! Depends on `uniko-store` only.  "Layer 5" is cognitive altitude, not
//! build order: in the dependency graph this crate is a sibling of
//! `uniko-extract`, and `uniko-memory` sits *above* it — memory's
//! consolidation worker (P4, the heartbeat) calls `promote_procedures_once`
//! and `detect_topics_once` after successful consolidation cycles. P5/P6
//! are subscribers to P4, which is why memory depends on cortex and not
//! the other way around. If cortex ever needs memory's runtime APIs
//! (e.g. MCTS planning calling recall), invert via a sweep trait defined
//! in memory rather than adding a dependency here.

pub mod procedures;
pub mod topics;

#[doc(inline)]
pub use procedures::{
    LifecycleConfig, MatchedProcedure, PromotionReport, match_procedures, promote_procedures_once,
    record_procedure_use,
};
#[doc(inline)]
pub use topics::{TopicConfig, TopicReport, detect_topics_once, detect_topics_once_with_llm};

/// Cargo features of this crate and whether this build compiled them in.
/// `llm` here is what turns on abstractive summaries, whichever crate's
/// feature enabled it.
pub const COMPILED_FEATURES: &[(&str, bool)] = &[("llm", cfg!(feature = "llm"))];
