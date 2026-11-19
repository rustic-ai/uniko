//! Tool calls: an `:Action` per call, its result kept verbatim as an
//! `:Artifact` of kind `tool_output`.
//!
//! A tool's result is often the most precise thing an agent ever sees —
//! an order list, a query result, a log — and its meaning leans on context
//! outside any one line of it: the tool, its arguments, the field a value
//! sits under. So the result is not squeezed into an Action property or a
//! conversational turn. It goes through the normal artifact path, which
//! stores the bytes once, chunks them, and makes them searchable, and the
//! Action that produced it is linked `PRODUCED`.
//!
//! Calls are written by [`write_tool_calls_in_tx`] inside a caller's
//! transaction — the unit path, so a turn and the calls it made commit
//! together — or by [`ingest_tool_calls`], which opens one transaction for
//! a batch of calls recorded on their own.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde_json::Value as JsonValue;
use uniko_store::schema::{edges, labels};
use uniko_store::types::datetime_value;
use uniko_store::{KnowledgeBase, NodeId, UnikoError, Value};

use super::artifact::{ArtifactContextNids, UnitArtifactSeen};
use super::message::json_to_uni_value;
use super::source::{IngestOutcome, PreparedSource};

/// `action_type` of an Action recorded as a tool call.
pub const TOOL_CALL_ACTION_TYPE: &str = "tool_call";

/// Artifact kind of a tool call's verbatim result.
pub const TOOL_OUTPUT_KIND: &str = "tool_output";

/// A tool call ready to write: its result payload is already prepared
/// (blob stored, chunked) outside any transaction.
#[derive(Debug)]
pub struct PreparedToolCall {
    /// External id; the caller's call id when it has one.
    pub action_id: String,
    /// [`TOOL_CALL_ACTION_TYPE`] for a tool call, or the caller's own kind
    /// for an action recorded through `record_action`.
    pub action_type: String,
    /// The tool that was called.
    pub tool_name: Option<String>,
    /// The arguments the tool was called with.
    pub input: Option<JsonValue>,
    /// What to keep on `Action.output` itself: the result when it is small,
    /// or a stub pointing at the artifact.
    pub inline_output: Option<JsonValue>,
    /// The start of the result as text, for the Action's embedding when the
    /// inline output is only a stub.
    pub output_preview: Option<String>,
    /// Outcome: `"ok"`, `"error"`, …
    pub status: Option<String>,
    /// Error text for a failed call.
    pub error: Option<String>,
    /// When the call started.
    pub started_at: DateTime<Utc>,
    /// When the call ended.
    pub ended_at: DateTime<Utc>,
    /// The verbatim result, prepared as a `tool_output` artifact.
    pub payload: Option<PreparedSource>,
    /// SHA-256 hex of the result payload; the replay key with `action_id`.
    pub payload_hash: Option<String>,
    /// Message that triggered the call, by external id, when the call is
    /// written on its own. The unit path links the turn's own message.
    pub triggered_by_message_id: Option<String>,
    /// Action this one follows, by external id. When unset and
    /// `chain_in_session` is true, the session's latest earlier action.
    pub previous_action_id: Option<String>,
    /// Chain to the session's latest earlier action when no
    /// `previous_action_id` is given.
    pub chain_in_session: bool,
    /// Who may see the call's result (`private:{id}`, `team:{id}`, …).
    pub visibility: Option<String>,
    /// The participant that performed the call, when it is not the batch's
    /// default (the turn's sender, or the recording session's agent). It is
    /// created as kind `agent` if it doesn't exist yet.
    pub performer_id: Option<String>,
}

impl PreparedToolCall {
    /// Every `:ArtifactContent.content_id` this call will merge, for the
    /// caller's content locks.
    pub fn content_ids(&self) -> Vec<String> {
        self.payload
            .as_ref()
            .map(PreparedSource::content_ids)
            .unwrap_or_default()
    }
}

/// What writing one tool call produced.
#[derive(Debug)]
pub struct ToolCallOutcome {
    /// External id of the Action.
    pub action_id: String,
    /// Node id of the Action.
    pub action_node: NodeId,
    /// The result artifact, when the call carried one.
    pub payload: Option<IngestOutcome>,
    /// The call was already recorded with the same result; nothing written.
    pub was_replay: bool,
}

impl ToolCallOutcome {
    /// Node id of the result artifact, when there is one.
    #[must_use]
    pub fn artifact_node(&self) -> Option<NodeId> {
        self.payload.as_ref().map(|p| match p {
            IngestOutcome::Artifact(r) => r.artifact_node_id,
            IngestOutcome::Pdf(r) => r.artifact_node_id,
        })
    }

    /// External id of the result artifact, when there is one.
    #[must_use]
    pub fn artifact_id(&self) -> Option<&str> {
        self.payload.as_ref().map(|p| match p {
            IngestOutcome::Artifact(r) => r.artifact_id.as_str(),
            IngestOutcome::Pdf(r) => r.artifact_id.as_str(),
        })
    }
}

/// Graph context a batch of calls is written into.
#[derive(Debug, Clone, Copy)]
pub struct ToolWriteCtx {
    /// The session the calls belong to.
    pub session_nid: Option<NodeId>,
    /// The participant that performed them.
    pub performer_nid: NodeId,
    /// The message that triggered them, when already known in this
    /// transaction (the unit path).
    pub triggered_by_nid: Option<NodeId>,
}

/// SHA-256 hex of a result payload.
#[must_use]
pub fn payload_hash(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}

/// Write `calls` inside `tx`, in order.
///
/// Each call becomes an Action linked `PERFORMED_BY` the performer,
/// `IN_SESSION`, `TRIGGERED_BY` its message and `NEXT_ACTION` from the one
/// before it; its payload is applied as an artifact linked `PRODUCED`. A
/// call whose `action_id` already exists is a replay when its payload hash
/// matches and nothing is written; otherwise the call is refused.
///
/// # Errors
///
/// - [`UnikoError::IdConflict`] when an `action_id` exists with a
///   different result.
/// - any underlying read or write error.
pub async fn write_tool_calls_in_tx(
    kb: &KnowledgeBase,
    tx: &uniko_store::Transaction,
    calls: &[PreparedToolCall],
    ctx: ToolWriteCtx,
    performers: &HashMap<String, NodeId>,
    seen: &mut UnitArtifactSeen,
) -> uniko_store::Result<Vec<ToolCallOutcome>> {
    let mut out = Vec::with_capacity(calls.len());
    let mut prev_in_batch: Option<NodeId> = None;
    for call in calls {
        // Replay or conflict, decided before any write.
        if let Some((nid, props)) = kb
            .get_node_by_ext_id_in_tx(tx, labels::ACTION, "action_id", &call.action_id)
            .await?
        {
            let stored = match props.get("payload_hash") {
                Some(Value::String(h)) => Some(h.as_str()),
                _ => None,
            };
            if stored != call.payload_hash.as_deref() {
                return Err(UnikoError::id_conflict(
                    labels::ACTION,
                    "action_id",
                    &call.action_id,
                ));
            }
            prev_in_batch = Some(nid);
            out.push(ToolCallOutcome {
                action_id: call.action_id.clone(),
                action_node: nid,
                payload: None,
                was_replay: true,
            });
            continue;
        }

        let performer_nid = match &call.performer_id {
            Some(id) => *performers.get(id).ok_or_else(|| {
                UnikoError::Internal(format!(
                    "tool call '{}': performer '{id}' was not resolved before the write",
                    call.action_id
                ))
            })?,
            None => ctx.performer_nid,
        };
        let action_node = kb
            .create_node_in_tx(tx, labels::ACTION, &action_props(call))
            .await?;

        let mut edge_rows: Vec<(&str, NodeId, NodeId, HashMap<String, Value>)> = vec![(
            edges::PERFORMED_BY,
            action_node,
            performer_nid,
            HashMap::new(),
        )];
        if let Some(sid) = ctx.session_nid {
            edge_rows.push((edges::IN_SESSION, action_node, sid, HashMap::new()));
        }
        let trigger = match (
            ctx.triggered_by_nid,
            call.triggered_by_message_id.as_deref(),
        ) {
            (Some(nid), _) => Some(nid),
            (None, Some(mid)) => kb
                .get_node_by_ext_id_in_tx(tx, labels::MESSAGE, "message_id", mid)
                .await?
                .map(|(nid, _)| nid),
            (None, None) => None,
        };
        if let Some(mid) = trigger {
            edge_rows.push((edges::TRIGGERED_BY, action_node, mid, HashMap::new()));
        }
        let previous = match call.previous_action_id.as_deref() {
            Some(prev) => kb
                .get_node_by_ext_id_in_tx(tx, labels::ACTION, "action_id", prev)
                .await?
                .map(|(nid, _)| nid),
            None if call.chain_in_session => match prev_in_batch {
                Some(nid) => Some(nid),
                None => match ctx.session_nid {
                    Some(sid) => kb.latest_session_action_in_tx(tx, sid, action_node).await?,
                    None => None,
                },
            },
            None => None,
        };
        if let Some(prev) = previous {
            edge_rows.push((edges::NEXT_ACTION, prev, action_node, HashMap::new()));
        }
        kb.create_edges_in_tx(tx, &edge_rows).await?;

        let payload = match &call.payload {
            Some(prepared) => {
                let actx = ArtifactContextNids {
                    session_nid: ctx.session_nid,
                    message_nid: None,
                    action_nid: Some(action_node),
                };
                let outcome = prepared.apply_in_tx(kb, tx, actx, seen).await?;
                if let Some(vis) = call.visibility.as_deref()
                    && let Some(artifact_nid) = outcome.newly_created_artifact()
                {
                    kb.set_owned_chunks_visibility_in_tx(tx, artifact_nid, vis)
                        .await?;
                }
                Some(outcome)
            }
            None => None,
        };

        prev_in_batch = Some(action_node);
        out.push(ToolCallOutcome {
            action_id: call.action_id.clone(),
            action_node,
            payload,
            was_replay: false,
        });
    }
    Ok(out)
}

fn action_props(call: &PreparedToolCall) -> HashMap<String, Value> {
    let mut props: HashMap<String, Value> = HashMap::new();
    props.insert("action_id".into(), Value::String(call.action_id.clone()));
    props.insert(
        "action_type".into(),
        Value::String(call.action_type.clone()),
    );
    if let Some(name) = &call.tool_name {
        props.insert("tool_name".into(), Value::String(name.clone()));
    }
    if let Some(input) = &call.input {
        props.insert("input".into(), json_to_uni_value(input));
    }
    if let Some(output) = &call.inline_output {
        props.insert("output".into(), json_to_uni_value(output));
    }
    if let Some(status) = &call.status {
        props.insert("status".into(), Value::String(status.clone()));
    }
    if let Some(err) = &call.error {
        props.insert("error".into(), Value::String(err.clone()));
    }
    if let Some(h) = &call.payload_hash {
        props.insert("payload_hash".into(), Value::String(h.clone()));
    }
    props.insert("started_at".into(), datetime_value(call.started_at));
    props.insert("ended_at".into(), datetime_value(call.ended_at));
    props.insert(
        "duration_ms".into(),
        Value::Int(
            call.ended_at
                .signed_duration_since(call.started_at)
                .num_milliseconds()
                .max(0),
        ),
    );
    props
}

/// Make sure every explicit performer among `calls` has a Participant
/// (created as kind `agent`, otherwise left as it is) and return their node
/// ids by participant id.
///
/// Runs outside any transaction, before the write takes its locks, the
/// same way a turn's sender is set up: a participant with no actions yet is
/// a normal state, so it needs no rollback if the write then fails.
///
/// # Errors
///
/// Returns [`UnikoError::Storage`] on database failure.
pub async fn ensure_performers(
    kb: &KnowledgeBase,
    calls: &[PreparedToolCall],
) -> uniko_store::Result<HashMap<String, NodeId>> {
    let mut out = HashMap::new();
    for id in calls.iter().filter_map(|c| c.performer_id.as_deref()) {
        if out.contains_key(id) {
            continue;
        }
        let (nid, _) = kb
            .ensure_participant(id, "agent", &HashMap::new(), &HashMap::new())
            .await?;
        out.insert(id.to_string(), nid);
    }
    Ok(out)
}

/// Write `calls` in one transaction of their own.
///
/// For calls recorded apart from any turn: `Session::observe_tool` and
/// `record_action`. The default performer must already exist; explicit
/// performers are created (see [`ensure_performers`]). The session is
/// created when `create_session` is set and otherwise must exist, or the
/// call is recorded without its `IN_SESSION` link; a
/// `triggered_by_message_id` that names no message is skipped, as
/// `record_action` always did.
///
/// # Errors
///
/// As [`write_tool_calls_in_tx`], plus [`UnikoError::Storage`] when the
/// performer does not exist.
pub async fn ingest_tool_calls(
    kb: &KnowledgeBase,
    calls: &[PreparedToolCall],
    session_id: Option<&str>,
    create_session: bool,
    performer_id: &str,
) -> uniko_store::Result<Vec<ToolCallOutcome>> {
    let performer_nid = kb
        .get_node_by_ext_id(labels::PARTICIPANT, "participant_id", performer_id)
        .await?
        .map(|(nid, _)| nid)
        .ok_or_else(|| {
            UnikoError::Storage(format!(
                "tool call: Participant '{performer_id}' not found — create it before recording"
            ))
        })?;
    let session_nid = match session_id {
        Some(sid) if create_session => {
            let at = calls.first().map_or_else(Utc::now, |c| c.started_at);
            Some(super::session::get_or_create_session(kb, sid, &at).await?)
        }
        Some(sid) => kb
            .get_node_by_ext_id(labels::SESSION, "session_id", sid)
            .await?
            .map(|(nid, _)| nid),
        None => None,
    };
    let performers = ensure_performers(kb, calls).await?;
    let content_ids: Vec<String> = calls
        .iter()
        .flat_map(PreparedToolCall::content_ids)
        .collect();
    let _guards = kb.lock_content_ids(&content_ids).await;

    let retry_opts = uniko_store::RetryOptions::default();
    let mut attempts: u32 = 0;
    let outcomes = loop {
        attempts += 1;
        let mut seen = UnitArtifactSeen::new();
        let tx = kb.begin_tx().await?;
        let ctx = ToolWriteCtx {
            session_nid,
            performer_nid,
            triggered_by_nid: None,
        };
        match write_tool_calls_in_tx(kb, &tx, calls, ctx, &performers, &mut seen).await {
            Ok(outcomes) => match tx.commit().await {
                Ok(_) => break outcomes,
                Err(e) => {
                    let err = UnikoError::from(e);
                    if err.is_retriable() && attempts < retry_opts.max_attempts {
                        super::atomic::unit_retry_backoff(&retry_opts, attempts + 1).await;
                        continue;
                    }
                    return Err(err);
                }
            },
            Err(err) => {
                tx.rollback();
                if err.is_retriable() && attempts < retry_opts.max_attempts {
                    super::atomic::unit_retry_backoff(&retry_opts, attempts + 1).await;
                    continue;
                }
                return Err(err);
            }
        }
    };
    finish_tool_calls_post_commit(kb, calls, &outcomes).await;
    Ok(outcomes)
}

/// Best-effort work that must read committed rows: the payload's pooled
/// artifact embedding and the Action's own embedding.
pub async fn finish_tool_calls_post_commit(
    kb: &KnowledgeBase,
    calls: &[PreparedToolCall],
    outcomes: &[ToolCallOutcome],
) {
    for (call, outcome) in calls.iter().zip(outcomes) {
        if outcome.was_replay {
            continue;
        }
        if let (Some(prepared), Some(payload)) = (&call.payload, &outcome.payload) {
            prepared.finish_post_commit(kb, payload).await;
        }
        let text = action_embed_text(call);
        match crate::embedding::embed_document(kb, &text).await {
            Ok(vec) => {
                let props = HashMap::from([("embedding".to_string(), Value::Vector(vec))]);
                if let Err(e) = kb.update_node(outcome.action_node, &props).await {
                    tracing::debug!(error = %e, "tool call: action embedding write failed");
                }
            }
            Err(e) => tracing::debug!(error = %e, "tool call: action embedding failed"),
        }
    }
}

/// The text an Action is embedded from: its kind or tool, then summarised
/// arguments and inline output.
fn action_embed_text(call: &PreparedToolCall) -> String {
    const MAX_PAYLOAD_CHARS: usize = 400;
    let mut buf = call
        .tool_name
        .clone()
        .unwrap_or_else(|| call.action_type.clone());
    let render = |v: &JsonValue| match v {
        JsonValue::String(s) => s.clone(),
        other => other.to_string(),
    };
    if let Some(v) = &call.input {
        let text: String = render(v).chars().take(MAX_PAYLOAD_CHARS).collect();
        buf.push_str(&format!(" | input: {text}"));
    }
    let output = call
        .output_preview
        .clone()
        .or_else(|| call.inline_output.as_ref().map(render));
    if let Some(text) = output {
        let text: String = text.chars().take(MAX_PAYLOAD_CHARS).collect();
        buf.push_str(&format!(" | output: {text}"));
    }
    buf
}
