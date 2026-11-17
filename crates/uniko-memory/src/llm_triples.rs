//! LLM-based `(subject, predicate, object)` extraction for P4.
//!
//! Replaces the SRL/DEP-derived triple on each unprocessed Observation
//! with one extracted by an LLM.  Higher precision than the rule path:
//! the LLM resolves pronouns, picks a single normalized predicate, and
//! truncates trailing junk like punctuation or filler clauses.
//!
//! Cost: one LLM call per Observation per cycle.  Run via concurrent
//! batches so I/O overlaps the model's per-request latency.  Failed
//! parses leave the Observation's existing triple untouched.

use futures::stream::{self, StreamExt};

use uniko_store::xervo::{GenerationOptions, Message};
use uniko_store::{KnowledgeBase, NodeId, UnikoError};

/// Concurrent in-flight LLM calls.
///
/// Tuned for a hosted gpt-4o-mini pool — small enough to stay under
/// the typical per-key rate limit, large enough to overlap network
/// I/O across most of the cycle's batch.
const CONCURRENCY: usize = 16;

/// One refined triple ready to overwrite an Observation's columns.
#[derive(Debug, Clone)]
pub struct LlmTriple {
    /// Internal node id of the source Observation.
    pub node_id: NodeId,
    /// Cleaned subject (e.g. proper name resolved from a pronoun).
    pub subject: String,
    /// Single normalized verb in snake_case.
    pub predicate: String,
    /// Object phrase, ≤ 6 words; `None` for intransitive observations.
    pub object: Option<String>,
}

/// Input batch row supplied by the caller.
///
/// Owns its strings to keep the `Send` bounds clean across the
/// `buffer_unordered` await — borrowed `&str` here would force the
/// caller's source data to outlive the entire concurrent batch.
#[derive(Debug, Clone)]
pub struct ObservationInput {
    /// Internal node id (round-trips to [`LlmTriple::node_id`]).
    pub node_id: NodeId,
    /// Free-text observation content (`Observation.content`).
    pub content: String,
    /// Speaker name, used to resolve first-person pronouns.
    pub speaker_hint: Option<String>,
}

/// Refine a batch of Observations' triples concurrently via LLM.
///
/// Returns one [`LlmTriple`] per successful extraction; rows where the
/// LLM declined or the response failed to parse are silently dropped
/// so the caller can keep the original SRL/DEP triple for those.
///
/// The Xervo runtime fans requests out across at most [`CONCURRENCY`]
/// in-flight calls.
///
/// # Errors
///
/// This function never returns an error: per-row LLM failures are
/// logged and the row is omitted from the result.  The signature uses
/// [`Result`] for symmetry with the rest of the consolidation API.
pub async fn refine_triples(
    kb: &KnowledgeBase,
    llm_alias: &str,
    inputs: &[ObservationInput],
) -> Result<Vec<LlmTriple>, UnikoError> {
    Ok(refine_triples_counted(kb, llm_alias, inputs).await.triples)
}

/// Outcome of [`refine_triples_counted`].
#[derive(Debug, Default)]
pub(crate) struct RefineOutcome {
    /// Rows the LLM refined.
    pub triples: Vec<LlmTriple>,
    /// Rows whose LLM call itself failed (transport, timeout, provider
    /// error), as opposed to a `SKIP` or an unparseable reply.
    pub failed: usize,
}

/// [`refine_triples`], also counting rows whose LLM call failed so the
/// caller can tell an unreachable endpoint from a model that declined.
pub(crate) async fn refine_triples_counted(
    kb: &KnowledgeBase,
    llm_alias: &str,
    inputs: &[ObservationInput],
) -> RefineOutcome {
    let results = stream::iter(inputs.iter().cloned())
        .map(|input| async move { extract_one(kb, llm_alias, &input).await })
        .buffer_unordered(CONCURRENCY)
        .collect::<Vec<_>>()
        .await;

    let mut outcome = RefineOutcome::default();
    for result in results {
        match result {
            Ok(Some(t)) => outcome.triples.push(t),
            Ok(None) => {}
            Err(()) => outcome.failed += 1,
        }
    }
    outcome
}

/// Refine a single observation's triple via LLM.
///
/// Returns `Ok(None)` when the model emits `SKIP` or the response is
/// malformed, and `Err(())` when the LLM call itself fails.  Transport
/// errors are logged at `WARN`; either way the batch continues with the
/// original SRL/DEP triple for that row.
async fn extract_one(
    kb: &KnowledgeBase,
    llm_alias: &str,
    input: &ObservationInput,
) -> Result<Option<LlmTriple>, ()> {
    let speaker = input.speaker_hint.as_deref().unwrap_or("the speaker");
    let user = format!(
        "Observation: \"{content}\"\nSpeaker: {speaker}\n\nTriple:",
        content = input.content,
    );

    let messages = vec![Message::system(SYSTEM_PROMPT), Message::user(&user)];
    let options = GenerationOptions {
        max_tokens: Some(64),
        temperature: Some(0.0),
        ..Default::default()
    };

    let text = match kb.generate(llm_alias, &messages, options).await {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!(
                target: "llm_triples",
                error = %e,
                node_id = input.node_id,
                "LLM triple extraction failed; keeping original triple"
            );
            return Err(());
        }
    };

    Ok(parse_triple(&text, input.node_id))
}

/// Parse the LLM response into a triple, or `None` for SKIP / malformed.
///
/// Accepts the canonical `SUBJECT|PREDICATE|OBJECT` form and a few
/// permissive fallbacks (extra surrounding whitespace, trailing
/// punctuation, the model occasionally wrapping the line in quotes).
fn parse_triple(raw: &str, node_id: NodeId) -> Option<LlmTriple> {
    // Trim whitespace + matched outer quotes.  The LLM sometimes wraps
    // its answer in `"..."` or backticks despite the format request.
    let line = raw
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())?
        .trim_matches(|c: char| c == '"' || c == '`' || c.is_whitespace())
        .trim_end_matches('.');

    if line.eq_ignore_ascii_case("skip") {
        return None;
    }

    // Per-part trimming also strips quote characters that survived
    // the outer trim — the model occasionally puts quotes around just
    // the object phrase (e.g. `Caroline|researches|"adoption agencies"`).
    let strip = |s: &str| -> String {
        s.trim()
            .trim_matches(|c: char| c == '"' || c == '`')
            .trim()
            .to_string()
    };
    let parts: Vec<String> = line.split('|').map(strip).collect();
    if parts.len() < 2 {
        return None;
    }
    let subject = parts.first().filter(|s| !s.is_empty())?.clone();
    let predicate_raw = parts.get(1).filter(|s| !s.is_empty())?;
    let predicate = predicate_raw
        .to_ascii_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join("_");
    if predicate.is_empty() {
        return None;
    }
    let object = parts.get(2).cloned().filter(|s| !s.is_empty());

    Some(LlmTriple {
        node_id,
        subject,
        predicate,
        object,
    })
}

const SYSTEM_PROMPT: &str = "\
You convert observations from a conversation into clean (subject, \
predicate, object) triples for a knowledge graph.

Output format: one line, exactly `SUBJECT|PREDICATE|OBJECT`.
- subject: the named entity. Resolve `I`/`me`/`my`/`we` to the \
provided speaker name.
- predicate: a single verb or short verb phrase in snake_case (e.g. \
`researches`, `is_pursuing`, `wants_to_visit`, `lives_in`). Drop \
auxiliaries (`is`, `was`, `has been`) unless they are the verb itself \
(use `is_a` for noun-attribution).
- object: the thing acted on or attributed, ≤ 6 words. May be empty \
for intransitive observations — write `SUBJECT|PREDICATE|` in that \
case.

If the observation is not a factual claim about the speaker (greeting, \
question, filler, partial fragment), output exactly:
SKIP

Examples:
Observation: \"Caroline researches adoption agencies\"
Triple: Caroline|researches|adoption agencies

Observation: \"I'm starting a dance studio\"
Speaker: Jon
Triple: Jon|is_starting|dance studio

Observation: \"hi how are you\"
Triple: SKIP

Output only the triple line; no commentary, no quotes.";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic_triple() {
        let t = parse_triple("Caroline|researches|adoption agencies", 42).unwrap();
        assert_eq!(t.node_id, 42);
        assert_eq!(t.subject, "Caroline");
        assert_eq!(t.predicate, "researches");
        assert_eq!(t.object.as_deref(), Some("adoption agencies"));
    }

    #[test]
    fn parse_intransitive_triple_with_empty_object() {
        let t = parse_triple("Jon|is_pursuing|", 1).unwrap();
        assert_eq!(t.subject, "Jon");
        assert_eq!(t.predicate, "is_pursuing");
        assert!(t.object.is_none());
    }

    #[test]
    fn parse_normalizes_predicate_to_snake_case() {
        let t = parse_triple("Sam | Wants To Visit | Paris", 1).unwrap();
        assert_eq!(t.predicate, "wants_to_visit");
    }

    #[test]
    fn parse_skip_returns_none() {
        assert!(parse_triple("SKIP", 1).is_none());
        assert!(parse_triple("skip", 1).is_none());
        assert!(parse_triple("  Skip\n", 1).is_none());
    }

    #[test]
    fn parse_strips_decorations() {
        let t = parse_triple("\"Caroline|researches|adoption agencies\".", 1).unwrap();
        assert_eq!(t.subject, "Caroline");
        assert_eq!(t.predicate, "researches");
        assert_eq!(t.object.as_deref(), Some("adoption agencies"));
    }

    #[test]
    fn parse_rejects_too_few_pipes() {
        assert!(parse_triple("not a triple", 1).is_none());
        assert!(parse_triple("only_one", 1).is_none());
    }

    #[test]
    fn parse_picks_first_nonempty_line() {
        // Some models emit blank lines or leading "Triple:" labels.
        let t = parse_triple("\n\nJon|is_starting|dance studio\n", 1).unwrap();
        assert_eq!(t.subject, "Jon");
    }
}
