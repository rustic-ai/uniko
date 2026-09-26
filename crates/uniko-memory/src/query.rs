//! Query-time Episode recording — the path that lets P5 procedure
//! promotion learn from real production traffic.
//!
//! Two entry points:
//!
//! - [`record_query_episode`] — the primitive. Given a question, the
//!   answer the caller produced, and the [`ContextBundle`] that backed
//!   it, materialises an Episode under the supplied Participant. Use
//!   this when you already own the LLM call and just want recall +
//!   answer pairs to feed cortex.
//! - [`answer_query`] — convenience wrapper that runs
//!   [`crate::recall::recall`] + the caller's generator closure +
//!   (optional) [`record_query_episode`] in a single call. Bench and
//!   downstream API layers both target this so Episode recording
//!   stays a one-liner.
//!
//! Why this lives in `uniko-memory` rather than the API/bench layer:
//! Episodes are the input that cortex's P5 procedure promotion runs
//! over. Putting recording in the API layer alone would mean any
//! library-only caller silently starves P5; putting it in cortex
//! would invert the data flow (cortex is the consumer, not the
//! producer).
//!
//! Recording is **off by default** at the convenience wrapper — the
//! caller must supply [`QueryRecordOptions`] to opt in. The standalone
//! primitive is always explicit.
use serde_json::{Value as JsonValue, json};

use uniko_store::{KnowledgeBase, NodeId, UnikoError};

use crate::episode::{RecordEpisodeParams, record_episode};
use crate::recall::{ContextBundle, RecallConfig, RecallSource, recall};

/// Inputs for [`record_query_episode`].
///
/// All borrows live only for the duration of the call. The body
/// builds a `state` JSON object containing `question`, `answer`,
/// `recall_node_ids`, recall metadata, and (when supplied) the
/// answer LLM's token usage, then delegates to [`record_episode`].
///
/// `extra_state` lets benchmark / harness callers attach their own
/// fields (e.g. `sample_id`, `question_index`, `category`) without
/// the library having to know about them. It must be a JSON object;
/// non-object values are ignored. Fields built by this function take
/// precedence over `extra_state` keys with the same name.
#[derive(Debug, Clone)]
pub struct RecordQueryEpisodeParams<'a> {
    /// The question text — also used as the embedding topic.
    pub question: &'a str,
    /// The final answer the caller produced.
    pub answer: &'a str,
    /// Node IDs of the items that backed the answer. Typically
    /// `bundle.items.iter().map(|i| i.node_id).collect::<Vec<_>>()`.
    pub recall_node_ids: &'a [NodeId],
    /// Coverage score reported by the recall cascade (0.0–1.0).
    pub recall_coverage: f64,
    /// Token budget consumed by the recall bundle (per
    /// [`ContextBundle::total_tokens`]).
    pub recall_tokens: usize,
    /// Provider-reported prompt tokens for the answer LLM call.
    pub answer_input_tokens: Option<u64>,
    /// Provider-reported completion tokens for the answer LLM call.
    pub answer_output_tokens: Option<u64>,
    /// Model id that produced the answer (e.g. `gpt-4o-mini`).
    pub answer_model: Option<&'a str>,
    /// Subjective importance in `[0.0, 1.0]`. Forwarded to
    /// [`RecordEpisodeParams::importance`]; defaults to 0.5 when
    /// `None`. Bench passes the f1 score; production callers
    /// typically leave this `None`.
    pub importance: Option<f64>,
    /// "success" / "failure" / etc. Forwarded verbatim.
    pub outcome: Option<&'a str>,
    /// Defaults to `"retrieve"` when `None`.
    pub action_type: Option<&'a str>,
    /// Extra JSON object whose fields get merged into `state` under
    /// keys this function didn't already populate. Non-object values
    /// are silently ignored.
    pub extra_state: Option<JsonValue>,
}

/// Record an Episode capturing a recall + answer pair.
///
/// The resulting Episode's `state` JSON has shape:
///
/// ```json
/// {
///     "topic": "<question>",
///     "question": "<question>",
///     "answer": "<answer>",
///     "recall_node_ids": [...],
///     "recall_count": <usize>,
///     "recall_coverage": <f64>,
///     "recall_tokens": <usize>,
///     "answer_input_tokens": <u64?>,
///     "answer_output_tokens": <u64?>,
///     "answer_model": "<str?>",
///     ...extra_state fields
/// }
/// ```
///
/// `topic` is duplicated from `question` so the embedding path
/// (`uniko_extract::embedding::episode_topic_text`) picks it up
/// without needing to know about query-specific schema.
///
/// # Errors
///
/// Propagates [`UnikoError`] from [`record_episode`] — most commonly
/// when the supplied Participant doesn't exist.
pub async fn record_query_episode(
    kb: &KnowledgeBase,
    agent_id: &str,
    params: RecordQueryEpisodeParams<'_>,
) -> Result<NodeId, UnikoError> {
    // `JsonNumber::from_f64` rejects NaN/inf — clamp here so a NaN
    // coverage doesn't drop the field downstream.
    let coverage = finite_or_zero(params.recall_coverage);
    let mut state = json!({
        "topic": params.question,
        "question": params.question,
        "answer": params.answer,
        "recall_node_ids": params.recall_node_ids,
        "recall_count": params.recall_node_ids.len(),
        "recall_coverage": coverage,
        "recall_tokens": params.recall_tokens,
    });
    let obj = state.as_object_mut().expect("json! literal is an object");
    if let Some(t) = params.answer_input_tokens {
        obj.insert("answer_input_tokens".into(), json!(t));
    }
    if let Some(t) = params.answer_output_tokens {
        obj.insert("answer_output_tokens".into(), json!(t));
    }
    if let Some(m) = params.answer_model {
        obj.insert("answer_model".into(), json!(m));
    }

    // Merge extra_state — built-in keys win over caller-supplied ones.
    if let Some(JsonValue::Object(extra)) = params.extra_state {
        for (k, v) in extra {
            obj.entry(k).or_insert(v);
        }
    }

    let ep_params = RecordEpisodeParams {
        action_type: params.action_type.unwrap_or("retrieve").to_string(),
        outcome: params.outcome.map(str::to_string),
        state: Some(state),
        importance: params.importance,
        ..Default::default()
    };
    record_episode(kb, agent_id, ep_params).await
}

/// Replace `NaN` / `±inf` with `0.0` so values flowing into JSON or
/// other rejecters of non-finite floats stay representable.
fn finite_or_zero(x: f64) -> f64 {
    if x.is_finite() { x } else { 0.0 }
}

/// The answer surface a caller's generator closure returns.
#[derive(Debug, Clone)]
pub struct GeneratedAnswer {
    /// Final answer text shown to the user.
    pub text: String,
    /// Provider-reported prompt tokens, when available.
    pub input_tokens: Option<u64>,
    /// Provider-reported completion tokens, when available.
    pub output_tokens: Option<u64>,
    /// Model id the closure used. Persisted on the Episode so cost
    /// rollups can join per-model.
    pub model: Option<String>,
}

/// A generated answer plus the context it was grounded in.
///
/// Returned by [`Agent::answer`](crate::Agent::answer) and
/// [`answer_query`]. The `context` bundle's items carry their `kind` +
/// `sources`, so [`citations`](Answer::citations) can report what the
/// answer was grounded in.
#[derive(Debug)]
pub struct Answer {
    /// Final answer text.
    pub text: String,
    /// The recall bundle the answer was grounded in.
    pub context: ContextBundle,
    /// Model id the generator used, when reported.
    pub model: Option<String>,
    /// Provider-reported prompt tokens, when available.
    pub input_tokens: Option<u64>,
    /// Provider-reported completion tokens, when available.
    pub output_tokens: Option<u64>,
    /// `Some(episode_id)` when query recording was opted into *and*
    /// succeeded; `None` when opted out or when recording errored (logged
    /// at debug — the answer still flows through).
    pub recorded_episode: Option<NodeId>,
}

impl Answer {
    /// The deduped sources this answer was grounded in, for citations.
    ///
    /// This is grounding context — what the generator was shown — not
    /// model-attributed citations (the model isn't asked which items it
    /// used). Order follows the ranked bundle; duplicates are removed.
    #[must_use]
    pub fn citations(&self) -> Vec<&RecallSource> {
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for item in &self.context.items {
            for src in &item.sources {
                if seen.insert(src) {
                    out.push(src);
                }
            }
        }
        out
    }
}

/// Opt-in recording config for [`answer_query`].
///
/// Built deliberately as a distinct struct (rather than a `bool`)
/// so the participant id is unambiguous when recording is enabled —
/// the Participant must exist or [`record_query_episode`] will fail.
#[derive(Debug, Clone, Default)]
pub struct QueryRecordOptions {
    /// Participant id under which to record. Must already exist.
    pub participant_id: String,
    /// Defaults to `"retrieve"` when `None`.
    pub action_type: Option<String>,
    /// E.g. `"success"` / `"failure"`. Forwarded verbatim.
    pub outcome: Option<String>,
    /// Subjective importance in `[0.0, 1.0]`. Defaults to 0.5
    /// when `None`.
    pub importance: Option<f64>,
    /// Extra fields merged into `state`. See
    /// [`RecordQueryEpisodeParams::extra_state`].
    pub extra_state: Option<JsonValue>,
}

/// Run recall, hand the bundle to the caller's generator, and
/// (optionally) record an Episode capturing the pair.
///
/// The generator is a closure rather than a trait so callers can use
/// whatever LLM machinery they have — uniko-memory deliberately does
/// not own LLM selection or system prompts. A typical bench call
/// looks like:
///
/// ```ignore
/// let outcome = uniko_memory::answer_query(
///     &kb,
///     question,
///     &recall_config,
///     |bundle, q| async move {
///         let context = format_context(&bundle.items);
///         let text = kb.generate("llm/default", &messages, opts).await?;
///         Ok(uniko_memory::GeneratedAnswer {
///             text,
///             input_tokens: None,
///             output_tokens: None,
///             model: Some("gpt-4o-mini".into()),
///         })
///     },
///     Some(uniko_memory::QueryRecordOptions {
///         participant_id: "agent-1".into(),
///         outcome: Some("success".into()),
///         ..Default::default()
///     }),
/// ).await?;
/// ```
///
/// # Errors
///
/// - Recall failure propagates as [`UnikoError::Search`].
/// - Generator errors propagate verbatim.
/// - Recording failure is logged at `debug` and surfaces as
///   `episode_id = None` rather than erroring — failing to record
///   should never break a user-visible answer.
pub async fn answer_query<G, Fut>(
    kb: &KnowledgeBase,
    question: &str,
    recall_config: &RecallConfig,
    generator: G,
    record: Option<QueryRecordOptions>,
) -> Result<Answer, UnikoError>
where
    G: FnOnce(&ContextBundle, &str) -> Fut,
    Fut: std::future::Future<Output = Result<GeneratedAnswer, UnikoError>>,
{
    let bundle = recall(kb, question, recall_config).await?;
    let answer = generator(&bundle, question).await?;

    let episode_id = if let Some(opts) = record {
        let recall_ids: Vec<NodeId> = bundle.items.iter().map(|i| i.node_id).collect();
        let params = RecordQueryEpisodeParams {
            question,
            answer: &answer.text,
            recall_node_ids: &recall_ids,
            recall_coverage: bundle.coverage,
            recall_tokens: bundle.total_tokens,
            answer_input_tokens: answer.input_tokens,
            answer_output_tokens: answer.output_tokens,
            answer_model: answer.model.as_deref(),
            importance: opts.importance,
            outcome: opts.outcome.as_deref(),
            action_type: opts.action_type.as_deref(),
            extra_state: opts.extra_state,
        };
        match record_query_episode(kb, &opts.participant_id, params).await {
            Ok(id) => Some(id),
            Err(e) => {
                tracing::debug!(error = %e, "query episode recording failed");
                None
            }
        }
    } else {
        None
    };

    Ok(Answer {
        text: answer.text,
        context: bundle,
        model: answer.model,
        input_tokens: answer.input_tokens,
        output_tokens: answer.output_tokens,
        recorded_episode: episode_id,
    })
}

#[cfg(test)]
mod tests {
    use super::Answer;
    use crate::recall::{ContextBundle, RecallItem, RecallKind, RecallSource};

    /// `citations()` returns the bundle's sources in order, de-duplicated.
    #[test]
    fn citations_dedup_over_context_sources() {
        let from_msg = RecallSource::Message {
            message_id: "m1".into(),
            chunk_id: None,
        };
        let from_doc = RecallSource::Attachment {
            message_id: "m1".into(),
            artifact_id: "spec.pdf".into(),
            chunk_id: None,
        };
        let bundle = ContextBundle {
            items: vec![
                RecallItem {
                    node_id: 1,
                    kind: RecallKind::Fact,
                    score: 1.0,
                    content: "x".into(),
                    // duplicate `from_msg` across items collapses to one.
                    sources: vec![from_msg.clone(), from_doc.clone()],
                    category: None,
                    source_id: None,
                    revision_id: None,
                },
                RecallItem {
                    node_id: 2,
                    kind: RecallKind::Chunk,
                    score: 0.5,
                    content: "y".into(),
                    sources: vec![from_msg.clone()],
                    category: None,
                    source_id: None,
                    revision_id: None,
                },
            ],
            total_tokens: 0,
            phase1_only: false,
            phase2_only: false,
            coverage: 0.0,
        };
        let answer = Answer {
            text: "the deadline is Friday".into(),
            context: bundle,
            model: None,
            input_tokens: None,
            output_tokens: None,
            recorded_episode: None,
        };
        let cites = answer.citations();
        assert_eq!(cites.len(), 2, "two distinct sources, deduped");
        assert_eq!(cites[0], &from_msg, "order follows the ranked bundle");
        assert_eq!(cites[1], &from_doc);
    }
}
