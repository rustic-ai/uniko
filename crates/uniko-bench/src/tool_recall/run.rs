//! Ingest a generated dataset through uniko's public facade, then score
//! recall against each question's gold evidence.

use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;
use uniko_extract::ingest::chunking::count_tokens;
use uniko_memory::recall::intent::extract_literals;
use uniko_memory::{
    Agent, ContextBundle, RecordActionParams, SearchOptions, ToolCall, ToolOutput, Turn,
};

use super::synth::{Category, Dataset, Evidence, PayloadFormat, Question, Step};

/// How tool calls reach uniko.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum IngestMode {
    /// The call is an assistant turn naming the tool and its arguments;
    /// the result is a turn with `content_type = "tool_result"`.
    Turns,
    /// The result goes through `Session::record_action`.
    Actions,
    /// The call is a `ToolCall` on the assistant turn that made it.
    Tools,
}

pub const AGENT_ID: &str = "bench-agent";

/// A context budget a host might actually spend on one recall.
pub const HIT_BUDGET_TOKENS: usize = 2048;

/// Ingest every session; returns wall time in milliseconds.
pub async fn ingest(
    agent: &Agent,
    data: &Dataset,
    mode: IngestMode,
    finalize: bool,
) -> Result<u64> {
    let t0 = Instant::now();
    for s in &data.sessions {
        let mut session = agent.session(&s.session_id);
        for step in &s.steps {
            match step {
                Step::Say {
                    message_id,
                    sender,
                    text,
                    at,
                } => {
                    session
                        .observe(Turn::new(sender, text).id(message_id).at(*at))
                        .await
                        .with_context(|| format!("observe {message_id}"))?;
                }
                Step::Tool {
                    message_id,
                    invocation: inv,
                } => {
                    let args = serde_json::to_string(&inv.arguments)?;
                    match mode {
                        IngestMode::Turns => {
                            session
                                .observe(
                                    Turn::new(AGENT_ID, format!("{}({args})", inv.tool_name))
                                        .id(message_id)
                                        .at(inv.at),
                                )
                                .await?;
                            session
                                .observe(
                                    Turn::new("tool", &inv.payload)
                                        .id(format!("{message_id}-result"))
                                        .content_type("tool_result")
                                        .at(inv.at),
                                )
                                .await
                                .with_context(|| format!("observe result {}", inv.call_id))?;
                        }
                        IngestMode::Tools => {
                            let output = match inv.format {
                                PayloadFormat::Json => {
                                    ToolOutput::Json(serde_json::from_str(&inv.payload)?)
                                }
                                _ => ToolOutput::Text(inv.payload.clone()),
                            };
                            let call = ToolCall::new(&inv.tool_name)
                                .call_id(&inv.call_id)
                                .arguments(inv.arguments.clone())
                                .output(output)
                                .status("ok")
                                .started_at(inv.at);
                            session
                                .observe(
                                    Turn::new(AGENT_ID, format!("Calling {}.", inv.tool_name))
                                        .id(message_id)
                                        .at(inv.at)
                                        .tool_call(call),
                                )
                                .await
                                .with_context(|| format!("tool call {}", inv.call_id))?;
                        }
                        IngestMode::Actions => {
                            session
                                .observe(
                                    Turn::new(AGENT_ID, format!("Calling {}.", inv.tool_name))
                                        .id(message_id)
                                        .at(inv.at),
                                )
                                .await?;
                            let output = match inv.format {
                                PayloadFormat::Json => serde_json::from_str(&inv.payload)?,
                                _ => Value::String(inv.payload.clone()),
                            };
                            session
                                .record_action(RecordActionParams {
                                    action_id: Some(inv.call_id.clone()),
                                    action_type: inv.tool_name.clone(),
                                    input: Some(inv.arguments.clone()),
                                    output: Some(output),
                                    status: Some("ok".into()),
                                    started_at: Some(inv.at),
                                    ended_at: Some(inv.at),
                                    triggered_by_message_id: Some(message_id.clone()),
                                    ..Default::default()
                                })
                                .await
                                .with_context(|| format!("record_action {}", inv.call_id))?;
                        }
                    }
                }
            }
        }
        if finalize {
            session.finalize().await?;
        }
    }
    Ok(t0.elapsed().as_millis() as u64)
}

fn squash(s: &str) -> String {
    s.split_whitespace().collect()
}

/// 1-based rank of the first item holding every token of `ev`.
fn rank_of(items: &[String], ev: &Evidence) -> Option<usize> {
    let tokens: Vec<String> = ev.tokens.iter().map(|t| squash(t)).collect();
    items
        .iter()
        .position(|it| tokens.iter().all(|t| it.contains(t.as_str())))
        .map(|i| i + 1)
}

#[derive(Debug, Clone, Serialize)]
pub struct QuestionScore {
    pub id: String,
    pub category: Category,
    /// Rank of the first evidence group, if found.
    pub rank: Option<usize>,
    /// Fraction of evidence groups found anywhere in the bundle.
    pub coverage: f64,
    /// The hit item also carries every inherited context value.
    pub context_complete: bool,
    /// `Latest` only: the stale record ranks above the current one.
    pub stale_above: bool,
    pub items: usize,
    pub tokens: usize,
    /// Tokens of every item up to and including the hit: what a reader
    /// must take in before it sees the evidence.
    pub tokens_to_hit: Option<usize>,
    pub latency_ms: u64,
    /// Kind of the item holding the first evidence group.
    pub hit_kind: Option<String>,
}

pub fn score(q: &Question, bundle: &ContextBundle, latency_ms: u64) -> QuestionScore {
    let items: Vec<String> = bundle.items.iter().map(|i| squash(&i.content)).collect();
    let item_tokens: Vec<usize> = bundle
        .items
        .iter()
        .map(|i| count_tokens(&i.content))
        .collect();
    let tokens = item_tokens.iter().sum();
    let ranks: Vec<Option<usize>> = q.evidence.iter().map(|e| rank_of(&items, e)).collect();
    let rank = ranks.first().copied().flatten();
    let coverage = if ranks.is_empty() {
        0.0
    } else {
        ranks.iter().filter(|r| r.is_some()).count() as f64 / ranks.len() as f64
    };
    // A caller receives an item's context header with its content, so the
    // inherited values count when either carries them.
    let with_context: Vec<String> = bundle
        .items
        .iter()
        .map(|i| {
            squash(&format!(
                "{} {}",
                i.context.as_deref().unwrap_or(""),
                i.content
            ))
        })
        .collect();
    let context_complete = rank.is_some_and(|r| {
        q.context
            .iter()
            .all(|c| with_context[r - 1].contains(squash(c).as_str()))
    });
    let stale_above = q
        .stale
        .as_ref()
        .is_some_and(|st| match (rank_of(&items, st), rank) {
            (Some(s), Some(r)) => s < r,
            (Some(_), None) => true,
            _ => false,
        });
    QuestionScore {
        id: q.id.clone(),
        category: q.category,
        rank,
        coverage,
        context_complete,
        stale_above,
        items: bundle.items.len(),
        tokens,
        tokens_to_hit: rank.map(|r| item_tokens[..r].iter().sum()),
        latency_ms,
        hit_kind: rank.map(|r| format!("{:?}", bundle.items[r - 1].kind)),
    }
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct Summary {
    pub n: usize,
    pub hit_at_1: f64,
    pub hit_at_5: f64,
    pub hit_at_k: f64,
    /// Hit within the first [`HIT_BUDGET_TOKENS`] tokens of the bundle.
    pub hit_at_budget: f64,
    pub mrr: f64,
    pub coverage: f64,
    pub context_complete: f64,
    pub stale_above: f64,
    pub mean_tokens: f64,
    pub p50_latency_ms: u64,
}

pub fn summarize<'a>(scores: impl IntoIterator<Item = &'a QuestionScore>) -> Summary {
    let scores: Vec<&QuestionScore> = scores.into_iter().collect();
    let n = scores.len();
    if n == 0 {
        return Summary::default();
    }
    let f = n as f64;
    let mean = |g: &dyn Fn(&QuestionScore) -> f64| scores.iter().map(|s| g(s)).sum::<f64>() / f;
    let mut lat: Vec<u64> = scores.iter().map(|s| s.latency_ms).collect();
    lat.sort_unstable();
    Summary {
        n,
        hit_at_1: mean(&|s| f64::from(u8::from(s.rank == Some(1)))),
        hit_at_5: mean(&|s| f64::from(u8::from(s.rank.is_some_and(|r| r <= 5)))),
        hit_at_k: mean(&|s| f64::from(u8::from(s.rank.is_some()))),
        hit_at_budget: mean(&|s| {
            f64::from(u8::from(
                s.tokens_to_hit.is_some_and(|t| t <= HIT_BUDGET_TOKENS),
            ))
        }),
        mrr: mean(&|s| s.rank.map_or(0.0, |r| 1.0 / r as f64)),
        coverage: mean(&|s| s.coverage),
        context_complete: mean(&|s| f64::from(u8::from(s.context_complete))),
        stale_above: mean(&|s| f64::from(u8::from(s.stale_above))),
        mean_tokens: mean(&|s| s.tokens as f64),
        p50_latency_ms: lat[n / 2],
    }
}

/// Recall every answerable question and score it.
pub async fn evaluate(agent: &Agent, data: &Dataset) -> Result<Vec<QuestionScore>> {
    let mut out = Vec::new();
    for q in data
        .questions
        .iter()
        .filter(|q| q.category != Category::Unanswerable)
    {
        let t = Instant::now();
        let bundle = agent.recall(&q.text).await?;
        out.push(score(q, &bundle, t.elapsed().as_millis() as u64));
    }
    Ok(out)
}

pub fn by_category(scores: &[QuestionScore]) -> BTreeMap<Category, Summary> {
    let mut groups: BTreeMap<Category, Vec<&QuestionScore>> = BTreeMap::new();
    for s in scores {
        groups.entry(s.category).or_default().push(s);
    }
    groups.into_iter().map(|(k, v)| (k, summarize(v))).collect()
}

/// How the read tools do on one question: a scripted reader that makes a
/// single `Sources::search` call with the question and reads only its hits.
#[derive(Debug, Clone, Serialize)]
pub struct ReaderScore {
    pub id: String,
    pub category: Category,
    /// Rank of the first hit holding the first evidence group.
    pub rank: Option<usize>,
    /// That hit also carries every inherited context value (in its text or
    /// its context header).
    pub context_complete: bool,
    /// For an unanswerable question: no hit names any literal the question
    /// asked about, so a reader correctly finds nothing.
    pub abstained: bool,
    /// Tokens of every hit's text and context.
    pub tokens: usize,
}

pub const READER_HITS: usize = 5;

pub async fn evaluate_reader(agent: &Agent, data: &Dataset) -> Result<Vec<ReaderScore>> {
    let mut out = Vec::new();
    for q in &data.questions {
        let hits = agent
            .sources()
            .search(&q.text, &SearchOptions::default().limit(READER_HITS))
            .await?;
        let texts: Vec<String> = hits.iter().map(|h| squash(&h.text)).collect();
        let full: Vec<String> = hits
            .iter()
            .map(|h| {
                squash(&format!(
                    "{} {}",
                    h.context.as_deref().unwrap_or(""),
                    h.text
                ))
            })
            .collect();
        let tokens = hits
            .iter()
            .map(|h| count_tokens(&h.text) + h.context.as_deref().map_or(0, count_tokens))
            .sum();
        let rank = q.evidence.first().and_then(|ev| rank_of(&texts, ev));
        let context_complete = rank.is_some_and(|r| {
            q.context
                .iter()
                .all(|c| full[r - 1].contains(squash(c).as_str()))
        });
        let abstained = q.category == Category::Unanswerable && {
            let lits: Vec<String> = extract_literals(&q.text)
                .iter()
                .map(|l| squash(l))
                .collect();
            !texts
                .iter()
                .any(|t| lits.iter().any(|l| t.contains(l.as_str())))
        };
        out.push(ReaderScore {
            id: q.id.clone(),
            category: q.category,
            rank,
            context_complete,
            abstained,
            tokens,
        });
    }
    Ok(out)
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct ReaderSummary {
    pub n: usize,
    pub hit_at_1: f64,
    pub hit_at_5: f64,
    pub context_complete: f64,
    /// Unanswerable questions only.
    pub abstained: f64,
    pub mean_tokens: f64,
}

pub fn summarize_reader(scores: &[ReaderScore]) -> BTreeMap<Category, ReaderSummary> {
    let mut groups: BTreeMap<Category, Vec<&ReaderScore>> = BTreeMap::new();
    for s in scores {
        groups.entry(s.category).or_default().push(s);
    }
    groups
        .into_iter()
        .map(|(k, v)| {
            let f = v.len() as f64;
            let mean = |g: &dyn Fn(&ReaderScore) -> f64| v.iter().map(|s| g(s)).sum::<f64>() / f;
            (
                k,
                ReaderSummary {
                    n: v.len(),
                    hit_at_1: mean(&|s| f64::from(u8::from(s.rank == Some(1)))),
                    hit_at_5: mean(&|s| f64::from(u8::from(s.rank.is_some()))),
                    context_complete: mean(&|s| f64::from(u8::from(s.context_complete))),
                    abstained: mean(&|s| f64::from(u8::from(s.abstained))),
                    mean_tokens: mean(&|s| s.tokens as f64),
                },
            )
        })
        .collect()
}
