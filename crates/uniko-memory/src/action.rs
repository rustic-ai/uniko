//! Action recording — the `record_action` agent tool (F17–F20).
//!
//! Actions are the agent's explicit tool calls and operations: shell
//! commands, file edits, API requests, RPCs.  They differ from
//! Episodes (which are subjective learning experiences) in that they
//! carry concrete input/output payloads and link to the artifacts they
//! produce or the messages they were triggered by.
//!
//! When the output exceeds the configured token threshold
//! ([`UnikoConfig::action_output_artifact_threshold`], default 256
//! tokens — F20), the full payload is stored verbatim as an Artifact of
//! kind `tool_output` through the normal artifact path — content stored
//! once, chunked, searchable by recall — linked by `PRODUCED`, and the
//! Action keeps a short stub instead. The Action and that Artifact commit
//! in one transaction. For a tool call whose result should always be
//! searchable, prefer [`ToolCall`](crate::ToolCall).
//!
//! [`UnikoConfig::action_output_artifact_threshold`]:
//!     uniko_store::config::UnikoConfig::action_output_artifact_threshold

#[cfg(feature = "onnx")]
use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde_json::Value as JsonValue;

use uniko_extract::ingest::count_tokens;
use uniko_extract::ingest::modality::ModalityRegistry;
use uniko_extract::ingest::source::{IngestContext, prepare_source};
use uniko_extract::ingest::tool::{PreparedToolCall, ingest_tool_calls};
#[cfg(feature = "onnx")]
use uniko_extract::nlp::NlpPipeline;
#[cfg(feature = "onnx")]
use uniko_extract::nlp::types::NerEntityType;
use uniko_pipes::IngestSource;
#[cfg(feature = "onnx")]
use uniko_store::Value;
use uniko_store::id::new_id;
#[cfg(feature = "onnx")]
use uniko_store::schema::{edges, labels};
use uniko_store::{KnowledgeBase, NodeId, UnikoError};

/// Inputs for [`record_action`].
///
/// Most fields are optional; only `action_type` and the agent id (via
/// the function argument) are required.  Pre-set `action_id` to
/// integrate with external ID spaces (ADR-1).
#[derive(Debug, Clone, Default)]
pub struct RecordActionParams {
    /// Optional pre-assigned id.  When `None`, a fresh UUID v7 is used.
    pub action_id: Option<String>,
    /// Required: kind of action (e.g. `"shell"`, `"file_edit"`,
    /// `"http_request"`).
    pub action_type: String,
    /// Optional structured input payload — what was sent to the tool.
    pub input: Option<JsonValue>,
    /// Optional output payload — what came back.  Large outputs
    /// overflow to an Artifact (F20).
    pub output: Option<JsonValue>,
    /// Outcome status: `"success"`, `"failure"`, `"timeout"`, …
    pub status: Option<String>,
    /// When the action started.  Defaults to `now()`.
    pub started_at: Option<DateTime<Utc>>,
    /// When the action ended.  Defaults to `started_at`.
    pub ended_at: Option<DateTime<Utc>>,
    /// Error message when status is `"failure"`.
    pub error: Option<String>,
    /// Optional `message_id` (external key) of the Message that
    /// triggered this action.  When set, a `TRIGGERED_BY` edge is
    /// created (F19).
    pub triggered_by_message_id: Option<String>,
    /// Optional `session_id` (external key) of the Session this action
    /// belongs to.  When set, an `IN_SESSION` edge is created.
    pub session_id: Option<String>,
    /// Optional `action_id` of the previous action in this chain.
    /// When set, a `NEXT_ACTION` edge is created from the previous to
    /// the new action.
    pub previous_action_id: Option<String>,
}

/// Outcome of a successful [`record_action`] call.
///
/// Returns both the Action node and (when applicable) the Artifact
/// node created from output overflow.  Callers can chain further
/// metadata onto either.
#[derive(Debug, Clone)]
pub struct RecordActionResult {
    /// The new Action node id.
    pub action_node: NodeId,
    /// Externalised Artifact id when output overflowed; `None` otherwise.
    pub overflow_artifact: Option<NodeId>,
    /// What each best-effort stage did: NLP entity linking and, for an
    /// overflowed output, the artifact's ingest.
    pub diagnostics: Vec<uniko_store::Diagnostic>,
}

/// Record an Action node, wire its edges, and embed it.
///
/// Mandatory wiring: `PERFORMED_BY` from the Action to the agent's
/// Participant.  Optional wiring (when the params provide identifiers
/// that resolve to existing nodes): `TRIGGERED_BY` to the Message,
/// `IN_SESSION` to the Session, `NEXT_ACTION` from the previous
/// Action, `PRODUCED` to the output Artifact.  Each optional edge is
/// best-effort — when a referenced node is missing, the edge is
/// skipped rather than failing the call (so callers don't need to
/// pre-check existence under churn).  Everything is written in one
/// transaction.
///
/// Recording an `action_id` again with the same output is a no-op that
/// returns the existing Action; with a different output it is refused.
///
/// The Participant identified by `agent_id` must exist — callers
/// typically create it once at agent bootstrap (same contract as
/// [`crate::record_episode`]).  The Action's embedding is written after
/// the commit, best-effort.
///
/// # Errors
///
/// - [`UnikoError::Storage`] when the agent is missing or graph
///   operations fail.
/// - [`UnikoError::IdConflict`] when `action_id` was already recorded with
///   a different output.
pub async fn record_action(
    kb: &KnowledgeBase,
    agent_id: &str,
    params: RecordActionParams,
) -> Result<RecordActionResult, UnikoError> {
    let (result, diagnostics) =
        uniko_store::diagnostics::collect(record_action_collecting(kb, agent_id, params)).await;
    let mut result = result?;
    result.diagnostics = diagnostics;
    Ok(result)
}

/// [`record_action`]'s body, run inside the diagnostics scope.
async fn record_action_collecting(
    kb: &KnowledgeBase,
    agent_id: &str,
    params: RecordActionParams,
) -> Result<RecordActionResult, UnikoError> {
    let action_id = params.action_id.clone().unwrap_or_else(new_id);
    let started_at = params.started_at.unwrap_or_else(Utc::now);
    let ended_at = params.ended_at.unwrap_or(started_at);

    // ── Output overflow handling (F20) ──
    let threshold = kb.config().action_output_artifact_threshold;
    let (inline_output, overflow) = split_overflow(params.output.as_ref(), threshold);
    let output_bytes = params.output.as_ref().map(output_text);
    let payload_hash = output_bytes
        .as_deref()
        .map(|t| uniko_extract::ingest::tool::payload_hash(t.as_bytes()));
    let payload = match overflow {
        Some((text, structured)) => {
            let mut source = IngestSource::text(text)
                .with_id(format!("{action_id}:output"))
                .with_kind(uniko_extract::ingest::tool::TOOL_OUTPUT_KIND);
            if structured && let Ok(m) = uniko_pipes::content::Mime::parse("application/json") {
                source = source.with_mime(m);
            }
            source.metadata.insert(
                "tool_name".into(),
                JsonValue::String(params.action_type.clone()),
            );
            if let Some(input) = &params.input {
                source
                    .metadata
                    .insert("tool_arguments".into(), input.clone());
            }
            let context = IngestContext {
                session_id: params.session_id.clone(),
                triggered_by_message_id: None,
            };
            Some(prepare_source(kb, &ModalityRegistry::default(), source, &context).await?)
        }
        None => None,
    };

    let prepared = PreparedToolCall {
        action_id,
        action_type: params.action_type.clone(),
        tool_name: None,
        input: params.input.clone(),
        inline_output,
        output_preview: output_bytes.map(|t| t.chars().take(400).collect()),
        status: params.status.clone(),
        error: params.error.clone(),
        started_at,
        ended_at,
        payload,
        payload_hash,
        triggered_by_message_id: params.triggered_by_message_id.clone(),
        previous_action_id: params.previous_action_id.clone(),
        chain_in_session: false,
        visibility: None,
    };
    let outcomes = ingest_tool_calls(
        kb,
        std::slice::from_ref(&prepared),
        params.session_id.as_deref(),
        false,
        agent_id,
    )
    .await?;
    let outcome = outcomes
        .first()
        .ok_or_else(|| UnikoError::Internal("record_action: no outcome".into()))?;

    // ── NER via Kniv NLP cascade ──
    //
    // Wire MENTIONS edges from the Action to every Entity the NLP
    // model finds in input/output text.  This makes Actions
    // first-class participants in entity-anchored recall — searches
    // for "calls referencing API foo" can now traverse from Foo →
    // MENTIONS → Action, the same way they traverse from Foo →
    // MENTIONS → Message.  Best-effort: when the NLP runtime is
    // unavailable (no ONNX provider) we log and continue.
    #[cfg(feature = "onnx")]
    if !outcome.was_replay {
        let embed_text = build_embed_text(
            &params.action_type,
            params.input.as_ref(),
            params.output.as_ref(),
        );
        link_action_entities(kb, outcome.action_node, &embed_text).await;
    }

    Ok(RecordActionResult {
        action_node: outcome.action_node,
        overflow_artifact: outcome.artifact_node(),
        diagnostics: Vec::new(),
    })
}

/// Run Kniv NER on the Action's text representation and wire
/// `MENTIONS` edges to existing or freshly-merged Entity nodes.
#[cfg(feature = "onnx")]
///
/// We merge by entity name + type so an Action that mentions
/// "Caroline" gets the same Entity node as a Message that mentioned
/// her — that's the entire point of entity-anchored recall.  Edge
/// `count` defaults to 1; consolidation can roll counts up across
/// MENTIONS edges later.
async fn link_action_entities(kb: &KnowledgeBase, action_node: NodeId, text: &str) {
    use uniko_store::{Diagnostic, DiagnosticCode, Stage, StageStatus};
    let skipped = |code| {
        uniko_store::diagnostics::record(
            Diagnostic::new(Stage::Nlp, StageStatus::Failed, code).target("action_entities"),
        );
    };
    let Some(pipeline) = NlpPipeline::try_new(kb).await else {
        tracing::debug!("Kniv NLP unavailable; Action MENTIONS edges skipped");
        skipped(DiagnosticCode::NlpUnavailable);
        return;
    };
    let result = match pipeline.analyze(text).await {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!(error = %e, "Kniv analyze failed; Action MENTIONS skipped");
            skipped(DiagnosticCode::NlpFailed);
            return;
        }
    };
    uniko_store::diagnostics::record(Diagnostic::applied(Stage::Nlp).target("action_entities"));

    for span in result.entities {
        let entity_id = stable_entity_id(&span.text, span.entity_type);
        let mut props = HashMap::new();
        props.insert("name".into(), Value::String(span.text.clone()));
        props.insert(
            "entity_type".into(),
            Value::String(entity_type_str(span.entity_type).to_string()),
        );
        let entity_node = match kb
            .merge_node(labels::ENTITY, "entity_id", &entity_id, &props)
            .await
        {
            Ok(n) => n,
            Err(e) => {
                tracing::debug!(error = %e, entity = %span.text, "Entity merge failed");
                continue;
            }
        };
        let mut edge_props = HashMap::new();
        edge_props.insert("count".into(), Value::Int(1));
        if let Err(e) = kb
            .create_edge(edges::MENTIONS, action_node, entity_node, &edge_props)
            .await
        {
            tracing::debug!(error = %e, "MENTIONS edge failed");
        }
    }
}

/// Stable `entity_id` derived from `(name, entity_type)`.
///
/// Two Actions mentioning the same entity converge on the same
/// Entity node.  Matches the convention used by the P2 ingest path
/// when it deduplicates entities across messages. Wraps
/// [`uniko_store::id::stable_hex64`] with a NUL separator.
#[cfg(feature = "onnx")]
fn stable_entity_id(name: &str, entity_type: NerEntityType) -> String {
    // Canonical shared derivation (issue #1) so an entity mentioned by an
    // Action converges with the same entity mentioned in a message.
    uniko_store::id::entity_id(name, entity_type_str(entity_type))
}

/// Stable string representation of [`NerEntityType`] for storage on
/// the Entity node's `entity_type` property.
#[cfg(feature = "onnx")]
fn entity_type_str(t: NerEntityType) -> &'static str {
    // Canonical shared type vocabulary (issue #1): lowercase, aligned with
    // `uniko_extract`'s `EntityType::as_str()` so overlapping types
    // (person/organization/location/date) produce identical entity_ids
    // across the message-dedup and action-linking paths.
    match t {
        NerEntityType::Person => "person",
        NerEntityType::Organization => "organization",
        NerEntityType::Location => "location",
        NerEntityType::Date => "date",
        NerEntityType::Numeric => "numeric",
        NerEntityType::Event => "event",
        NerEntityType::Product => "product",
        NerEntityType::WorkOfArt => "work_of_art",
        NerEntityType::Group => "group",
        NerEntityType::Misc => "misc",
    }
}

/// Decide whether the output payload should be inlined on the Action
/// node or spilled into an Artifact.
///
/// Returns `(action_output, overflow)`:
/// - `action_output = Some(value)` when the payload (or a stub pointer)
///   should be stored on the Action.
/// - `overflow = Some((text, structured))` when an Artifact must be
///   created from `text`; `structured` marks a JSON payload.
///
/// A string is measured as is, a structured payload by its pretty-printed
/// JSON; either one over `threshold_tokens` overflows.
fn split_overflow(
    output: Option<&JsonValue>,
    threshold_tokens: usize,
) -> (Option<JsonValue>, Option<(String, bool)>) {
    let Some(out) = output else {
        return (None, None);
    };
    let structured = !out.is_string();
    let text = output_text(out);
    let tokens = count_tokens(&text);
    if tokens > threshold_tokens {
        // Replace the inlined output with a short stub so the Action still
        // records the kind of payload it produced.
        let stub = format!("<overflow: {tokens} tokens stored as a tool_output artifact>");
        (Some(JsonValue::String(stub)), Some((text, structured)))
    } else {
        (Some(out.clone()), None)
    }
}

/// The verbatim text of an output: a string as is, anything else as
/// pretty-printed JSON.
fn output_text(v: &JsonValue) -> String {
    match v {
        JsonValue::String(s) => s.clone(),
        other => serde_json::to_string_pretty(other).unwrap_or_else(|_| other.to_string()),
    }
}

#[cfg(any(feature = "onnx", test))]
/// Build the embedding source string for an Action.
///
/// Concatenates action type with summarised input and output text.
/// Long payloads are truncated to keep the embedding call's input
/// size predictable.
fn build_embed_text(
    action_type: &str,
    input: Option<&JsonValue>,
    output: Option<&JsonValue>,
) -> String {
    const MAX_PAYLOAD_CHARS: usize = 400;
    let mut buf = String::with_capacity(64);
    buf.push_str(action_type);
    if let Some(v) = input {
        buf.push_str(" | input: ");
        buf.push_str(&truncate_chars(&json_text(v), MAX_PAYLOAD_CHARS));
    }
    if let Some(v) = output {
        buf.push_str(" | output: ");
        buf.push_str(&truncate_chars(&json_text(v), MAX_PAYLOAD_CHARS));
    }
    buf
}

#[cfg(any(feature = "onnx", test))]
/// Render a JSON value as a short embedding-friendly string.
fn json_text(v: &JsonValue) -> String {
    match v {
        JsonValue::String(s) => s.clone(),
        other => other.to_string(),
    }
}

#[cfg(any(feature = "onnx", test))]
fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn split_overflow_inlines_short_string() {
        let (prop, overflow) = split_overflow(Some(&json!("short text")), 256);
        assert!(matches!(prop, Some(JsonValue::String(_))));
        assert!(overflow.is_none());
    }

    #[test]
    fn split_overflow_spills_long_string() {
        let big = "word ".repeat(2_000);
        let (prop, overflow) = split_overflow(Some(&JsonValue::String(big.clone())), 256);
        match prop {
            Some(JsonValue::String(stub)) => {
                assert!(stub.contains("overflow"), "stub was: {stub}");
            }
            other => panic!("expected stub String, got {other:?}"),
        }
        assert_eq!(overflow, Some((big, false)));
    }

    #[test]
    fn split_overflow_inlines_small_structured_payloads() {
        let (prop, overflow) = split_overflow(Some(&json!({"a": [1, 2, 3]})), 256);
        assert!(matches!(prop, Some(JsonValue::Object(_))));
        assert!(overflow.is_none());
    }

    #[test]
    fn split_overflow_spills_large_structured_payloads_as_json() {
        let items: Vec<_> = (0..200).map(|i| json!({"id": i, "status": "ok"})).collect();
        let (prop, overflow) = split_overflow(Some(&json!({"items": items})), 64);
        assert!(matches!(prop, Some(JsonValue::String(ref s)) if s.contains("overflow")));
        let (text, structured) = overflow.expect("overflowed");
        assert!(structured);
        assert!(
            text.contains("\"status\": \"ok\""),
            "pretty-printed JSON expected"
        );
    }

    #[test]
    fn truncate_chars_respects_limit() {
        assert_eq!(truncate_chars("abc", 10), "abc");
        let t = truncate_chars(&"x".repeat(20), 5);
        assert!(t.starts_with("xxxxx"));
        assert!(t.ends_with('…'));
    }

    #[test]
    fn build_embed_text_includes_action_type_and_payloads() {
        let txt = build_embed_text(
            "shell",
            Some(&json!("ls -la")),
            Some(&json!("file1\nfile2")),
        );
        assert!(txt.starts_with("shell"));
        assert!(txt.contains("ls -la"));
        assert!(txt.contains("file1"));
    }
}
