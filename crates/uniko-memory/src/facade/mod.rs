//! The high-level [`Uniko`] facade — the single end-user entry point.
//!
//! Lower layers (`uniko-store`, `uniko-pipes`, `uniko-extract`) expose
//! `KnowledgeBase`, the `Step` trait, worker queues, and model catalogs.
//! Driving them directly means hand-assembling a `KnowledgeBase`, a
//! `PipelineSystem`, and an [`Agent`](crate::Agent) before any memory work
//! can start. This module hides all of that behind one owning type that
//! opens with the validated best config and hands out agent-scoped views.
//!
//! End users touch only [`Uniko`], [`Agent`](crate::Agent), [`Session`],
//! [`Turn`], the `*Params` builders, and [`ContextBundle`](crate::recall::ContextBundle).
//! They never name `KnowledgeBase`, `PipelineSystem`, the `Step` trait, the
//! ingest task enums, or any uni-db type.
//!
//! ```no_run
//! # async fn demo() -> Result<(), uniko_store::UnikoError> {
//! use uniko_memory::{Turn, Uniko};
//!
//! let memory = Uniko::in_memory().await?;
//! let agent = memory.agent("assistant-1");
//! let mut session = agent.session("chat-1");
//! session.observe(Turn::new("alice", "I love hiking on weekends")).await?;
//! let context = agent.recall("hobbies").await?;
//! # let _ = context;
//! memory.shutdown().await?;
//! # Ok(())
//! # }
//! ```

mod data;
mod goals;
mod session;
#[cfg(test)]
mod tests;
mod uniko;

pub use data::{ArtifactView, Data, MessageView};
pub use goals::{GoalContext, GoalPhase, GoalView, Goals, TaskPhase, TaskView};
pub use session::{FinalizeReport, ObserveResult, Session, Turn};
// Shared by `Agent::finalize_session`; the module itself stays private.
pub(crate) use session::finalize_session;
pub use uniko::{LlmSpec, Uniko, UnikoBuilder};

use crate::policy::Viewer;

/// Access-control scope applied to an [`Agent`](crate::Agent)'s reads.
///
/// Resolved into a [`ViewerScope`](crate::recall::ViewerScope) at recall
/// time. [`AsAgent`](RecallScope::AsAgent) exists because building a
/// [`Viewer`] resolves team/org memberships from the graph (async, needs
/// the store), which a synchronous builder cannot do up front.
///
/// The default is [`Unrestricted`](RecallScope::Unrestricted) (fail-open),
/// matching the validated benchmark stack; opt into scoping with
/// [`UnikoBuilder::scope_to_agent`].
#[derive(Debug, Clone, Default)]
pub enum RecallScope {
    /// No visibility filtering — the agent sees every recalled item.
    #[default]
    Unrestricted,
    /// Filter reads to what the calling agent's own `participant_id`
    /// may see. The [`Viewer`] is resolved once per [`Agent`] and cached.
    AsAgent,
    /// Filter reads as an explicit, caller-supplied [`Viewer`].
    ///
    /// Use this to recall as a different participant than the agent
    /// identity, or to skip the membership round-trip with
    /// [`Viewer::from_parts`].
    As(Viewer),
}
