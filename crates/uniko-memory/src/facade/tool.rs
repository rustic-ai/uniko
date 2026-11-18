//! Tool calls recorded as first-class memory.
//!
//! A [`ToolCall`] is one invocation of one tool: its name, its arguments,
//! and what came back. Recorded with [`Turn::tool_call`](crate::Turn::tool_call)
//! it commits in the same transaction as the turn that made it; recorded
//! with [`Session::observe_tool`](crate::Session::observe_tool) it commits on
//! its own. Either way it becomes an `:Action` (`action_type =
//! "tool_call"`) linked to the turn and session, and its result is stored
//! verbatim as an `:Artifact` of kind `tool_output`, chunked and searchable
//! like a document and linked `Action -PRODUCED-> Artifact`.

use chrono::{DateTime, Utc};
use serde_json::Value as JsonValue;
use uniko_extract::ingest::count_tokens;
use uniko_extract::ingest::modality::ModalityRegistry;
use uniko_extract::ingest::source::{IngestContext, prepare_source};
use uniko_extract::ingest::tool::{
    PreparedToolCall, TOOL_CALL_ACTION_TYPE, TOOL_OUTPUT_KIND, ToolCallOutcome, payload_hash,
};
use uniko_pipes::IngestSource;
use uniko_pipes::content::Mime;
use uniko_store::{KnowledgeBase, NodeId, UnikoError};

use crate::policy::Visibility;

/// What a tool returned.
#[derive(Debug, Clone)]
pub enum ToolOutput {
    /// A JSON value, stored pretty-printed so its records chunk cleanly.
    Json(JsonValue),
    /// Text: a log, test output, a CSV, a diff.
    Text(String),
    /// Raw bytes with their MIME type (an image, a PDF).
    Bytes {
        /// The payload.
        bytes: Vec<u8>,
        /// Its MIME type, e.g. `"image/png"`.
        mime: String,
    },
}

/// One call of one tool, to record with
/// [`Turn::tool_call`](crate::Turn::tool_call) or
/// [`Session::observe_tool`](crate::Session::observe_tool).
///
/// # Examples
///
/// ```
/// use serde_json::json;
/// use uniko_memory::ToolCall;
///
/// let call = ToolCall::new("list_orders")
///     .call_id("call_01")
///     .arguments(json!({"customer_id": "C-1042", "page": 2}))
///     .json_output(json!({"meta": {"currency": "EUR"}, "data": {"items": []}}));
/// assert_eq!(call.tool_name(), "list_orders");
/// ```
#[derive(Debug, Clone)]
pub struct ToolCall {
    tool_name: String,
    call_id: Option<String>,
    arguments: Option<JsonValue>,
    output: Option<ToolOutput>,
    status: Option<String>,
    error: Option<String>,
    started_at: Option<DateTime<Utc>>,
    ended_at: Option<DateTime<Utc>>,
    triggered_by: Option<String>,
    previous_call: Option<String>,
    visibility: Option<Visibility>,
    category: Option<String>,
    source_id: Option<String>,
    revision_id: Option<String>,
}

impl ToolCall {
    /// A call of `tool_name`.
    #[must_use]
    pub fn new(tool_name: impl Into<String>) -> Self {
        Self {
            tool_name: tool_name.into(),
            call_id: None,
            arguments: None,
            output: None,
            status: None,
            error: None,
            started_at: None,
            ended_at: None,
            triggered_by: None,
            previous_call: None,
            visibility: None,
            category: None,
            source_id: None,
            revision_id: None,
        }
    }

    /// The tool's name.
    #[must_use]
    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }

    /// The host's own id for this call, which becomes the Action's
    /// `action_id`. Recording the same id again with the same result is a
    /// no-op; with a different result it is refused.
    ///
    /// Without one, the id is derived from the session, the triggering
    /// turn, the tool, its arguments and its result, so an identical retry
    /// is still recognised.
    #[must_use]
    pub fn call_id(mut self, call_id: impl Into<String>) -> Self {
        self.call_id = Some(call_id.into());
        self
    }

    /// The arguments the tool was called with.
    #[must_use]
    pub fn arguments(mut self, arguments: JsonValue) -> Self {
        self.arguments = Some(arguments);
        self
    }

    /// What the tool returned.
    #[must_use]
    pub fn output(mut self, output: ToolOutput) -> Self {
        self.output = Some(output);
        self
    }

    /// A JSON result.
    #[must_use]
    pub fn json_output(self, value: JsonValue) -> Self {
        self.output(ToolOutput::Json(value))
    }

    /// A text result.
    #[must_use]
    pub fn text_output(self, text: impl Into<String>) -> Self {
        self.output(ToolOutput::Text(text.into()))
    }

    /// Outcome: `"ok"`, `"error"`, `"timeout"`, …
    #[must_use]
    pub fn status(mut self, status: impl Into<String>) -> Self {
        self.status = Some(status.into());
        self
    }

    /// Error text for a failed call.
    #[must_use]
    pub fn error(mut self, error: impl Into<String>) -> Self {
        self.error = Some(error.into());
        self
    }

    /// When the call started (default: now).
    #[must_use]
    pub fn started_at(mut self, at: DateTime<Utc>) -> Self {
        self.started_at = Some(at);
        self
    }

    /// When the call ended (default: when it started).
    #[must_use]
    pub fn ended_at(mut self, at: DateTime<Utc>) -> Self {
        self.ended_at = Some(at);
        self
    }

    /// The message that made this call, by `message_id`. Only needed with
    /// [`Session::observe_tool`](crate::Session::observe_tool); a call added
    /// with [`Turn::tool_call`](crate::Turn::tool_call) is linked to its
    /// turn.
    #[must_use]
    pub fn triggered_by(mut self, message_id: impl Into<String>) -> Self {
        self.triggered_by = Some(message_id.into());
        self
    }

    /// The call this one follows, by call id. Without it a call follows the
    /// session's latest earlier call.
    #[must_use]
    pub fn after(mut self, call_id: impl Into<String>) -> Self {
        self.previous_call = Some(call_id.into());
        self
    }

    /// Who may see the result. A call added with
    /// [`Turn::tool_call`](crate::Turn::tool_call) takes the turn's
    /// visibility unless it sets its own.
    #[must_use]
    pub fn visibility(mut self, visibility: Visibility) -> Self {
        self.visibility = Some(visibility);
        self
    }

    /// The caller's record category for the result (issue #39).
    #[must_use]
    pub fn category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// The logical source the result came from (issue #39).
    #[must_use]
    pub fn source(mut self, source_id: impl Into<String>) -> Self {
        self.source_id = Some(source_id.into());
        self
    }

    /// The revision of that source (issue #41).
    #[must_use]
    pub fn revision(mut self, revision_id: impl Into<String>) -> Self {
        self.revision_id = Some(revision_id.into());
        self
    }

    pub(crate) fn visibility_or(&mut self, inherited: Option<&Visibility>) {
        if self.visibility.is_none() {
            self.visibility = inherited.cloned();
        }
    }
}

/// What recording one [`ToolCall`] produced.
#[derive(Debug, Clone)]
pub struct ToolCallResult {
    /// The call's id (its `action_id`).
    pub call_id: String,
    /// The Action's node id.
    pub action_node: NodeId,
    /// The result artifact's id, when the call had a result.
    pub artifact_id: Option<String>,
    /// The result artifact's node id, when the call had a result.
    pub artifact_node: Option<NodeId>,
    /// The call was already recorded with the same result: nothing written.
    pub was_replay: bool,
    /// What each best-effort ingest stage did for the result artifact.
    /// Filled by [`Session::observe_tool`](super::Session::observe_tool);
    /// empty for a call recorded as part of a turn.
    pub diagnostics: Vec<uniko_store::Diagnostic>,
}

impl From<&ToolCallOutcome> for ToolCallResult {
    fn from(o: &ToolCallOutcome) -> Self {
        Self {
            call_id: o.action_id.clone(),
            action_node: o.action_node,
            artifact_id: o.artifact_id().map(str::to_string),
            artifact_node: o.artifact_node(),
            was_replay: o.was_replay,
            diagnostics: Vec::new(),
        }
    }
}

/// The verbatim bytes of a result, the MIME they are stored under, and the
/// text the Action keeps inline when the result is small.
fn render_output(output: &ToolOutput) -> (Vec<u8>, Option<Mime>, Option<JsonValue>) {
    match output {
        ToolOutput::Json(v) => {
            let text = serde_json::to_string_pretty(v).unwrap_or_else(|_| v.to_string());
            (
                text.into_bytes(),
                Mime::parse("application/json").ok(),
                Some(v.clone()),
            )
        }
        ToolOutput::Text(t) => (
            t.clone().into_bytes(),
            None,
            Some(JsonValue::String(t.clone())),
        ),
        ToolOutput::Bytes { bytes, mime } => (bytes.clone(), Mime::parse(mime).ok(), None),
    }
}

/// Everything a recorded call needs that is not the call itself.
pub(crate) struct ToolPrepCtx<'a> {
    pub kb: &'a KnowledgeBase,
    pub extractors: &'a ModalityRegistry,
    pub session_id: &'a str,
    /// The turn that made the call, for the unit path.
    pub turn_message_id: Option<&'a str>,
    /// Tokens at or under which the result is also kept on the Action.
    pub inline_threshold: usize,
}

/// Prepare `call` for writing: store and chunk its result outside any
/// transaction, and decide its id and inline output.
pub(crate) async fn prepare_tool_call(
    ctx: &ToolPrepCtx<'_>,
    call: ToolCall,
) -> Result<PreparedToolCall, UnikoError> {
    let started_at = call.started_at.unwrap_or_else(Utc::now);
    let ended_at = call.ended_at.unwrap_or(started_at);
    let triggered_by = ctx
        .turn_message_id
        .map(str::to_string)
        .or_else(|| call.triggered_by.clone());

    let rendered = call.output.as_ref().map(render_output);
    let hash = rendered.as_ref().map(|(bytes, _, _)| payload_hash(bytes));
    let action_id = call.call_id.clone().unwrap_or_else(|| {
        uniko_store::id::stable_hex64("call", |h| {
            for part in [
                ctx.session_id,
                triggered_by.as_deref().unwrap_or(""),
                call.tool_name.as_str(),
                &call
                    .arguments
                    .as_ref()
                    .map(JsonValue::to_string)
                    .unwrap_or_default(),
                hash.as_deref().unwrap_or(""),
            ] {
                h.update(part.as_bytes());
                h.update(b"\x00");
            }
        })
    });

    let mut inline_output = None;
    let mut output_preview = None;
    let mut payload = None;
    if let Some((bytes, mime, inline)) = rendered {
        output_preview = std::str::from_utf8(&bytes)
            .ok()
            .map(|t| t.chars().take(400).collect());
        let artifact_id = format!("{action_id}:output");
        let tokens = std::str::from_utf8(&bytes).map_or(usize::MAX, count_tokens);
        inline_output = match inline {
            Some(v) if tokens <= ctx.inline_threshold => Some(v),
            _ => Some(JsonValue::String(format!(
                "<tool_output: {} bytes in artifact {artifact_id}>",
                bytes.len()
            ))),
        };
        let mut source = IngestSource::bytes(bytes)
            .with_id(artifact_id)
            .with_kind(TOOL_OUTPUT_KIND);
        if let Some(m) = mime {
            source = source.with_mime(m);
        }
        if let Some(c) = &call.category {
            source = source.with_category(c.clone());
        }
        if let Some(s) = &call.source_id {
            source = source.with_source(s.clone());
        }
        if let Some(r) = &call.revision_id {
            source = source.with_revision(r.clone());
        }
        source.metadata.insert(
            "tool_name".into(),
            JsonValue::String(call.tool_name.clone()),
        );
        if let Some(args) = &call.arguments {
            source
                .metadata
                .insert("tool_arguments".into(), args.clone());
        }
        let context = IngestContext {
            session_id: Some(ctx.session_id.to_string()),
            triggered_by_message_id: None,
        };
        payload = Some(prepare_source(ctx.kb, ctx.extractors, source, &context).await?);
    }

    Ok(PreparedToolCall {
        action_id,
        action_type: TOOL_CALL_ACTION_TYPE.to_string(),
        tool_name: Some(call.tool_name),
        input: call.arguments,
        inline_output,
        output_preview,
        status: call.status,
        error: call.error,
        started_at,
        ended_at,
        payload,
        payload_hash: hash,
        triggered_by_message_id: triggered_by,
        previous_action_id: call.previous_call,
        chain_in_session: true,
        visibility: call
            .visibility
            .filter(|v| *v != Visibility::Public)
            .map(|v| v.as_property()),
    })
}
