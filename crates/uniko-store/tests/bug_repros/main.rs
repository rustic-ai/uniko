//! Store-level bug reproductions / regression guards.
//!
//! These reproduce store/uni-db defects (label disjunction, duplicate index
//! application, UNWIND edge handling) and act as regression guards against
//! future upgrades. Consolidated into one binary so they link once.
//!
//! Run: `cargo nextest run -p uniko-store --test bug_repros`

mod label_disjunction_repro;
mod label_disjunction_union_schema_panic;
mod locy_key_null_without_fold_repro;
mod registered_rule_param_leak_repro;
mod schema_apply_duplicate_index_repro;
mod unwind_edge_repro;
