//! Session lifecycle management for the ingest pipeline.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use uniko_store::Value;

use uniko_store::schema::constants::edges;
use uniko_store::types::datetime_value;
use uniko_store::{KnowledgeBase, NodeId};

/// Get or create a Session node by `session_id`.
///
/// If a session with the given ID already exists, returns its internal
/// node ID.  Otherwise creates a new Session node with `started_at`
/// set to the provided timestamp.
///
/// # Errors
///
/// Returns a storage error if the graph operation fails.
pub async fn get_or_create_session(
    kb: &KnowledgeBase,
    session_id: &str,
    timestamp: &DateTime<Utc>,
) -> uniko_store::Result<NodeId> {
    // Fast path: session already exists.
    if let Some((nid, _)) = kb
        .get_node_by_ext_id("Session", "session_id", session_id)
        .await?
    {
        return Ok(nid);
    }

    // Create a new session.
    let mut props = HashMap::new();
    props.insert("session_id".into(), Value::String(session_id.to_string()));
    props.insert("started_at".into(), datetime_value(*timestamp));
    let start = std::time::Instant::now();
    let nid = kb.create_node("Session", &props).await?;
    tracing::info!(
        target: "tx_perf",
        tx_phase = "session_create",
        total_ms = start.elapsed().as_millis() as u64,
        commit_ms = start.elapsed().as_millis() as u64,
        "tx phase",
    );
    Ok(nid)
}

/// Ensure a Participant node exists for the given ID.
///
/// Creates the participant (kind `unknown`, named after its id) if absent.
/// On an existing one it updates `last_seen` only: a name, metadata or a
/// known kind (such as `agent` from registration) is never overwritten.
///
/// # Errors
///
/// Returns a storage error if the graph operation fails.
pub(crate) async fn ensure_participant(
    kb: &KnowledgeBase,
    participant_id: &str,
    timestamp: DateTime<Utc>,
) -> uniko_store::Result<NodeId> {
    let if_absent = HashMap::from([(
        "name".to_string(),
        Value::String(participant_id.to_string()),
    )]);
    let always = HashMap::from([("last_seen".to_string(), datetime_value(timestamp))]);
    let start = std::time::Instant::now();
    let (nid, _) = kb
        .ensure_participant(participant_id, "unknown", &if_absent, &always)
        .await?;
    let ms = start.elapsed().as_millis() as u64;
    tracing::info!(
        target: "tx_perf",
        tx_phase = "participant_ensure",
        total_ms = ms,
        commit_ms = ms,
        "tx phase",
    );
    Ok(nid)
}

/// Idempotently create a `PARTICIPATED_IN` edge from participant to
/// session.
///
/// Skips the create when the edge already exists, so concurrent ingests
/// of the same session (each holding the per-participant setup lock via
/// [`KnowledgeBase::lock_session_setup`]) don't accumulate duplicate
/// `PARTICIPATED_IN` edges. The existence check is authoritative only
/// under that lock; callers outside the locked first-sight path must not
/// rely on it for cross-process exclusion.
///
/// # Errors
///
/// Returns a storage error if the edge lookup or creation fails.
pub(crate) async fn link_participant_to_session(
    kb: &KnowledgeBase,
    participant_nid: NodeId,
    session_nid: NodeId,
) -> uniko_store::Result<()> {
    let start = std::time::Instant::now();
    let already_linked = kb
        .get_edges(
            participant_nid,
            edges::PARTICIPATED_IN,
            uniko_store::storage::edges::Direction::Outgoing,
        )
        .await?
        .iter()
        .any(|e| e.to == session_nid);
    if !already_linked {
        kb.create_edge(
            edges::PARTICIPATED_IN,
            participant_nid,
            session_nid,
            &HashMap::new(),
        )
        .await?;
    }
    let ms = start.elapsed().as_millis() as u64;
    tracing::info!(
        target: "tx_perf",
        tx_phase = "participant_link",
        total_ms = ms,
        commit_ms = ms,
        "tx phase",
    );
    Ok(())
}
