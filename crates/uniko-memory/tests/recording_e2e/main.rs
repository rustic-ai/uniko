//! Agent-recording end-to-end tests.
//!
//! Consolidated binary covering the recording API surface: episode logging
//! and FOLLOWED_BY chaining, action recording with overflow artifacts, and
//! query-episode recording plus answer-query orchestration. Each former
//! `tests/<name>.rs` file is included as a module so the suite links once
//! instead of once per file.
//!
//! Run: `cargo nextest run -p uniko-memory --test integration recording_e2e::`

mod action_e2e;
mod episode_e2e;
mod query_episode_e2e;
mod tools_e2e;
