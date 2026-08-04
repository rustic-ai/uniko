//! Recall and answer generation for benchmark questions.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::Result;
use uniko_memory::recall::{ContextBundle, RecallConfig, recall};
use uniko_store::KnowledgeBase;

use crate::data::{QaPair, QuestionCategory};

/// One retrieved item: node type, score, full content. Surfaced
/// per-question for offline diagnostics (which chunks/messages/obs
/// were actually pulled, in what order, with what scores).
#[derive(Debug, Clone, serde::Serialize)]
pub struct RecalledItem {
    pub node_id: i64,
    pub node_type: String,
    pub tier: String,
    pub score: f64,
    pub content: String,
}

/// Result of querying a single question.
#[derive(Debug)]
pub struct QueryResult {
    /// Conversation sample this question belongs to (LoCoMo `sample_id`).
    pub sample_id: String,
    /// Position of this question within `sample.qa` (post-category filter).
    /// Reproducible across runs because question order is stable.
    pub question_index: usize,
    /// Original question text.
    pub question: String,
    /// Gold answer from the dataset.
    pub gold_answer: String,
    /// Model-generated or retrieval-based answer.
    pub predicted_answer: String,
    /// Question category.
    pub category: QuestionCategory,
    /// Number of recall items retrieved.
    pub recall_items: usize,
    /// Full bundle dump (every item the recall layer surfaced, in
    /// score-descending order). Diagnostics only; large.
    pub recall_bundle: Vec<RecalledItem>,
    /// Evidence messages found / total evidence messages.
    pub evidence_found: usize,
    /// Total evidence messages for this question.
    pub evidence_total: usize,
    /// Recall phase latency in milliseconds.
    pub recall_latency_ms: u64,
    /// Answer generation latency in milliseconds (0 if retrieval-only).
    pub generation_latency_ms: u64,
    /// Whether the recall cascade exited early from Phase 1 (Compact).
    pub phase1_only: bool,
    /// Coverage score reported by the recall cascade (0.0–1.0).
    pub coverage: f64,
    /// Estimated total tokens in the recall bundle.
    pub total_tokens: usize,
    /// Provider-reported prompt tokens for the answer LLM call, when
    /// the provider returns a `usage` field (OpenAI/Anthropic/Google
    /// do; some local providers do not).  `None` in retrieval-only
    /// runs or when the provider omits usage.
    pub answer_input_tokens: Option<u64>,
    /// Provider-reported completion tokens for the answer LLM call.
    pub answer_output_tokens: Option<u64>,
    /// Model alias used for answer generation, empty in
    /// retrieval-only runs.
    pub answer_model: String,
    /// Wall-clock latency of the judge call when one ran.
    ///
    /// Populated by the caller after [`run_query`] returns — `run_query`
    /// itself never calls the judge.  Stored here so the reporter
    /// can sum judge latency / tokens / cost without threading a
    /// parallel collection.
    pub judge_latency_ms: Option<u64>,
    /// Provider-reported prompt tokens for the judge call.
    pub judge_input_tokens: Option<u64>,
    /// Provider-reported completion tokens for the judge call.
    pub judge_output_tokens: Option<u64>,
    /// Judge model alias when a judge ran.
    pub judge_model: Option<String>,
}

/// Run recall for a question and optionally generate an answer.
///
/// In retrieval-only mode, the "predicted answer" is the concatenated
/// content of the top recall items.
pub async fn run_query(
    kb: &KnowledgeBase,
    sample_id: &str,
    question_index: usize,
    qa: &QaPair,
    evidence_texts: &[String],
    llm_alias: Option<&str>,
    llm_use_default_options: bool,
) -> Result<QueryResult> {
    // Pull recall settings from the KB's UnikoConfig so the bench
    // honors `--reranker` and other recall overrides set on the
    // ingest config (rather than always using `RecallConfig::default()`).
    let recall_config = RecallConfig::from_uniko_config(kb.config());

    // Run recall.
    let recall_start = Instant::now();
    let bundle = recall(kb, &qa.question, &recall_config).await?;
    let recall_latency_ms = recall_start.elapsed().as_millis() as u64;

    // Compute evidence hit rate.
    let recalled_contents: Vec<&str> = bundle.items.iter().map(|i| i.content.as_str()).collect();
    let (evidence_found, evidence_total) =
        crate::eval::evidence_hit(&recalled_contents, evidence_texts, qa.gold_answer());

    // Generate answer.
    let gen_start = Instant::now();
    let (predicted_answer, answer_input_tokens, answer_output_tokens, answer_model) =
        if let Some(alias) = llm_alias {
            let answered = generate_answer_with_usage(
                kb,
                &bundle,
                &qa.question,
                alias,
                llm_use_default_options,
            )
            .await?;
            (
                answered.text,
                answered.prompt_tokens,
                answered.completion_tokens,
                alias.to_string(),
            )
        } else {
            (
                crate::retrieval_answer(&bundle, 5),
                None,
                None,
                String::new(),
            )
        };
    let generation_latency_ms = gen_start.elapsed().as_millis() as u64;

    let recall_bundle = bundle
        .items
        .iter()
        .map(|it| RecalledItem {
            node_id: it.node_id,
            node_type: format!("{:?}", it.kind),
            tier: format!("{:?}", it.kind.tier()),
            score: it.score,
            content: it.content.clone(),
        })
        .collect();

    Ok(QueryResult {
        sample_id: sample_id.to_string(),
        question_index,
        question: qa.question.clone(),
        gold_answer: qa.gold_answer().to_string(),
        predicted_answer,
        category: qa.question_category(),
        evidence_found,
        evidence_total,
        recall_items: bundle.items.len(),
        recall_bundle,
        recall_latency_ms,
        generation_latency_ms,
        phase1_only: bundle.phase1_only,
        coverage: bundle.coverage,
        total_tokens: bundle.total_tokens,
        answer_input_tokens,
        answer_output_tokens,
        answer_model,
        judge_latency_ms: None,
        judge_input_tokens: None,
        judge_output_tokens: None,
        judge_model: None,
    })
}

/// Generate an answer using the LLM from the recall context.
///
/// Calls the alias `llm_alias` via xervo with the bench's tailored
/// system prompt (teaches the model to read the `[N] (NodeType,
/// score=..., session_date=..., temporal=...): content` structure
/// produced by [`crate::format_context`]).  Public so a future
/// `uniko-api` HTTP surface can reuse the exact prompt + options the
/// bench validates against.
///
/// `use_default_options` skips the bench's per-request overrides
/// (`max_tokens=2048`, `temperature=0.1`) — required for OpenAI's
/// `gpt-5` / o-series, which rejects custom temperature.  Pass `true`
/// when targeting those models, `false` otherwise.
///
/// # Errors
///
/// Propagates any xervo `generate` error (alias not registered,
/// upstream API failure, etc.).
pub async fn generate_answer(
    kb: &KnowledgeBase,
    bundle: &ContextBundle,
    question: &str,
    llm_alias: &str,
    use_default_options: bool,
) -> Result<String> {
    let answered =
        generate_answer_with_usage(kb, bundle, question, llm_alias, use_default_options).await?;
    Ok(answered.text)
}

/// Answer + provider-reported token usage from one generate call.
///
/// `prompt_tokens` and `completion_tokens` are `None` when the
/// provider omits the `usage` field (some local backends do).
#[derive(Debug, Clone)]
pub struct AnsweredQuestion {
    /// Trimmed answer text returned by the LLM.
    pub text: String,
    /// Provider-reported input tokens.
    pub prompt_tokens: Option<u64>,
    /// Provider-reported output tokens.
    pub completion_tokens: Option<u64>,
}

/// Same as [`generate_answer`] but also surfaces token usage.
///
/// Splitting this out lets the bench account for cost without
/// breaking the existing `generate_answer` callers.
///
/// # Errors
///
/// Propagates any xervo `generate` error.
pub async fn generate_answer_with_usage(
    kb: &KnowledgeBase,
    bundle: &ContextBundle,
    question: &str,
    llm_alias: &str,
    use_default_options: bool,
) -> Result<AnsweredQuestion> {
    use uni_db::xervo::{GenerationOptions, Message};

    let node_ids: Vec<i64> = bundle.items.iter().map(|i| i.node_id).collect();
    let session_dates = fetch_session_dates(kb, &node_ids).await;
    let temporal_anchors = fetch_temporal_anchors(kb, &node_ids).await;
    let context = crate::format_context(bundle, &session_dates, &temporal_anchors);

    let system = "You are a helpful assistant answering questions about conversations. \
        Answer using the provided context. You may paraphrase or make direct \
        inferences from what the context says, including using the `session_date` \
        field to resolve relative dates like 'yesterday' or 'next month'. \
        When an item carries a `temporal` field, treat that date as the \
        resolved anchor for the claim — prefer it over relative phrases like \
        'last Friday' that appear inside the content. \
        For speculative questions phrased as 'Would X likely…?' or 'What would X \
        think about…?', reason from the speaker's stated preferences, beliefs, \
        and behaviors in the context to give a reasoned inference rather than \
        abstaining. Only say 'The information is not mentioned in the \
        conversation' when no relevant preferences or behaviors appear in the \
        context. \
        Answer concisely in one or two sentences.";

    let user = format!("Context:\n{context}\n\nQuestion: {question}\n\nAnswer:");

    let messages = vec![Message::system(system), Message::user(&user)];
    // Gemma 4 (and other reasoning models) emit chain-of-thought into a
    // separate `reasoning_content` field that consumes the token budget.
    // 2048 leaves headroom for ~1500 reasoning tokens + a 1-2 sentence
    // answer. For non-reasoning models this is just an upper bound;
    // they self-terminate at the EOS well before hitting it.
    // Reasoning models (gemma4 / phi4 via mistralrs) need a token budget
    // for chain-of-thought; non-reasoning local models tolerate the
    // custom temperature. OpenAI's gpt-5/o-series rejects both — it
    // requires `max_completion_tokens` and only accepts the default
    // temperature.  Caller passes `use_default_options=true` for those.
    let options = if use_default_options {
        GenerationOptions::default()
    } else {
        GenerationOptions {
            max_tokens: Some(2048),
            temperature: Some(0.1),
            ..Default::default()
        }
    };

    let result = kb
        .db()
        .xervo()
        .generate(llm_alias, &messages, options)
        .await?;
    let (prompt_tokens, completion_tokens) = result
        .usage
        .as_ref()
        .map(|u| {
            (
                Some(u.prompt_tokens as u64),
                Some(u.completion_tokens as u64),
            )
        })
        .unwrap_or((None, None));
    Ok(AnsweredQuestion {
        text: result.text.trim().to_string(),
        prompt_tokens,
        completion_tokens,
    })
}

/// Read a row column as a `YYYY-MM-DD` date string.
///
/// uni-db (>= 2.1.0) returns DateTime properties as `Value::Temporal` (uni
/// `4583ee870`), so we format the temporal epoch. The model only needs the
/// date for relative-date resolution.
fn row_date(row: &uni_db::Row, column: &str) -> Option<String> {
    use uni_db::Value;
    let idx = row.columns().iter().position(|c| c == column)?;
    match row.values().get(idx)? {
        Value::Temporal(t) => {
            let millis = uniko_store::temporal_epoch_millis(t)?;
            chrono::DateTime::<chrono::Utc>::from_timestamp_millis(millis)
                .map(|d| d.format("%Y-%m-%d").to_string())
        }
        _ => None,
    }
}

/// Resolve the originating Session.started_at for each recalled node.
///
/// Walks the same edge set as `longmemeval::query::extract_session_ids`
/// (Message →IN_SESSION, Session →HAS_CHUNK, Observation →OBSERVED_IN
/// →Message →IN_SESSION) but projects `started_at` instead of
/// `session_id`. Nodes with no Session ancestor (e.g. label types not
/// anchored in a session) are simply absent from the returned map.
///
/// One query per item — matches the per-item pattern already used for
/// session_id extraction. Cheap relative to the LLM call that follows.
pub async fn fetch_session_dates(kb: &KnowledgeBase, node_ids: &[i64]) -> HashMap<i64, String> {
    let mut out = HashMap::new();
    let session = kb.db().session();
    for &nid in node_ids {
        let q = format!(
            "MATCH (n) WHERE id(n) = {nid} \
             OPTIONAL MATCH (n)-[:IN_SESSION]->(s1:Session) \
             OPTIONAL MATCH (s2:Session)-[:HAS_CHUNK]->(n) \
             OPTIONAL MATCH (n)-[:OBSERVED_IN]->(:Message)-[:IN_SESSION]->(s3:Session) \
             RETURN coalesce(s1.started_at, s2.started_at, s3.started_at) AS ts \
             LIMIT 1"
        );
        if let Ok(res) = session.query_with(&q).fetch_all().await {
            for row in res.rows() {
                if let Some(date) = row_date(row, "ts") {
                    out.insert(nid, date);
                    break;
                }
            }
        }
    }
    out
}

/// Resolve the per-Observation `temporal_anchor` for each recalled node.
///
/// Phase A surfaces ARGM-TMP captures from SRL as a resolved absolute
/// date at ingest.  This helper pulls that field for any Observation in
/// the bundle so the LLM sees the resolved anchor rather than the raw
/// `"Last Fri"` phrase buried in chunk text.
///
/// Returns an empty entry for non-Observation node types or for
/// Observations whose `temporal_anchor` is null.
pub async fn fetch_temporal_anchors(kb: &KnowledgeBase, node_ids: &[i64]) -> HashMap<i64, String> {
    let mut out = HashMap::new();
    let session = kb.db().session();
    for &nid in node_ids {
        let q = format!(
            "MATCH (n:Observation) WHERE id(n) = {nid} \
             AND n.temporal_anchor IS NOT NULL \
             RETURN n.temporal_anchor AS ts \
             LIMIT 1"
        );
        if let Ok(res) = session.query_with(&q).fetch_all().await {
            for row in res.rows() {
                if let Some(date) = row_date(row, "ts") {
                    out.insert(nid, date);
                    break;
                }
            }
        }
    }
    out
}
