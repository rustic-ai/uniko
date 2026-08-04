//! Rule management: stdlib Locy rules and lifecycle.

pub mod execution;
pub mod lifecycle;
pub mod stdlib;

pub(crate) use execution::stdlib_rule_params;
pub use execution::{RuleExecutionReport, consume_relevance_decay, run_active_rules};
pub use lifecycle::{
    AddRuleParams, DecayReport, RuleLifecycleConfig, add_rule, apply_decay_cycle, record_rule_match,
};
pub use stdlib::{is_stdlib_rule, register_stdlib_rules, relevance_decay_params};
