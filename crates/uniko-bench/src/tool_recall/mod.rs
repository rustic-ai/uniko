//! `tool-recall`: does recall find answers that live in agent tool output?
//!
//! A seeded generator ([`synth`]) writes agent sessions whose tool calls
//! return nested JSON, CSV query results, service logs and test output, and
//! asks questions whose gold evidence is one record in one call. [`run`]
//! ingests them through the public facade and scores recall with no LLM:
//! a question is a hit when every evidence token appears in one item.

pub mod run;
pub mod synth;
