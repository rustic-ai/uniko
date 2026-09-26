//! Pipeline 3 — Observation extraction.
//!
//! Extracts clean declarative observations from messages using the
//! NLP model's dependency tree, with CLS-based filtering and speaker
//! attribution for first-person pronouns.
//!
//! **Key principle**: observations are reconstructed from the DEP tree,
//! not raw text fragments. "I'm starting a dance studio" → "Jon is
//! starting a dance studio" (clean, declarative, speaker-attributed).
//!
//! Drive this via [`prepare_observations`] + [`apply_observations`] from
//! the atomic ingest path; the standalone Step adapter has been retired.

pub mod cleanup;
pub mod filter;
pub mod rules;
#[cfg(feature = "onnx")]
pub mod rules_engine;
pub mod temporal;
pub mod types;

pub use types::{ContradictionFlag, RawObservation};

use std::collections::HashMap;

use uniko_store::Value;

use uniko_store::schema::constants::{edges, labels};
use uniko_store::{NodeId, UnikoError};

/// Load the observation rule set, honouring `UnikoConfig.observation_rules_path`.
///
/// External paths are read once and cached in a process-global map keyed
/// by canonical path. The bundled `english.yml` is also cached (parsed
/// once) — reused across every message in a process.
#[cfg(feature = "onnx")]
fn load_observation_rules_from_path(
    cfg_path: Option<&std::path::Path>,
) -> &'static crate::observations::rules_engine::Rules {
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::{Mutex, OnceLock};

    static EXTERNAL: OnceLock<
        Mutex<HashMap<PathBuf, &'static crate::observations::rules_engine::Rules>>,
    > = OnceLock::new();

    if let Some(path) = cfg_path {
        let key = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let map = EXTERNAL.get_or_init(|| Mutex::new(HashMap::new()));
        // Recover from a poisoned lock: the cache only holds
        // path → &'static Rules pointers, so a panic during prior
        // insert can't have left it in an inconsistent shape.
        let mut guard = map.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(r) = guard.get(&key) {
            return r;
        }
        match crate::observations::rules_engine::load_rules_from_path(&key) {
            Ok(rules) => {
                let leaked: &'static _ = Box::leak(Box::new(rules));
                guard.insert(key, leaked);
                return leaked;
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "failed to load observation_rules_path; falling back to bundled rules"
                );
            }
        }
    }
    crate::observations::rules_engine::load_bundled_rules()
}

/// Multi-label CLS gate.
///
/// The CLS head emits a softmax over 8 raw labels. We accept a sentence
/// if **any** label whose prob clears `min_prob` belongs to the
/// configured informative set (default: `inform`, `status`, `request`,
/// `offer`). This handles the common case where the argmax is `social`
/// but the sentence carries propositional content (e.g. self-status
/// reports with affective language).
///
/// Falls back to the legacy `is_informative()` argmax check when
/// `cls_probs` is empty (e.g. data serialised before this field
/// existed).
#[cfg(feature = "onnx")]
fn cls_gate_admits(
    nlp_result: &crate::nlp::types::NlpResult,
    cls_labels: &[String],
    gate: &crate::observations::rules_engine::rules::ClsGate,
) -> bool {
    if nlp_result.cls_probs.is_empty() {
        // xervo's `NlpModel` reports only the top class + its confidence,
        // not the full softmax the probs path below consumes. Reconstruct
        // the old gate's intent as faithfully as the top-1 signal allows:
        // admit an informative top class outright; for a non-informative
        // top class with confidence `c`, the most any single informative
        // label could hold is `1 - c`, so only suppress when even that
        // upper bound cannot clear `min_prob` (`c > 1 - min_prob`). A
        // low-confidence "social"/"question" on an otherwise-declarative
        // sentence therefore still admits — matching the old
        // full-distribution gate — while a confident greeting is filtered.
        return nlp_result.sentence_class.is_informative()
            || nlp_result.cls_confidence <= 1.0 - gate.min_prob;
    }
    for (i, &p) in nlp_result.cls_probs.iter().enumerate() {
        if p < gate.min_prob {
            continue;
        }
        if let Some(name) = cls_labels.get(i)
            && gate.informative_labels.iter().any(|l| l == name)
        {
            return true;
        }
    }
    false
}

/// Substitute a resolved absolute date for a relative temporal phrase
/// in the observation `content` string, and return the resolved date
/// for the structured `temporal_anchor` slot.
///
/// The SRL rule engine renders observations with the surface form of a
/// temporal expression — `"Caroline went to the LGBTQ support group
/// yesterday"`.  Downstream consumers (LLM responders that don't share
/// the conversation date anchor, fact consolidation, downstream
/// embedders) shouldn't have to redo the relative-date math.  We
/// rewrite the content to embed the absolute date directly:
/// `"Caroline went to the LGBTQ support group on 2023-05-07"`.  The raw
/// surface phrase is preserved separately on
/// `RawObservation.temporal_phrase` for analytics.
///
/// Uses [`temporal::resolve_temporal_with_granularity`] (which returns
/// `None` on unparseable input) rather than [`temporal::resolve_temporal`]
/// (which silently falls back to `reference` on no-match) so that an
/// unrecognised phrase like `"next Whitsun"` doesn't get rewritten to
/// the message timestamp.
///
/// Returns the (possibly rewritten) content and the resolved anchor.
#[cfg_attr(not(feature = "onnx"), allow(dead_code))]
fn resolve_temporal_in_content(
    content: String,
    temporal_phrase: Option<&str>,
    reference: chrono::DateTime<chrono::Utc>,
) -> (String, Option<chrono::DateTime<chrono::Utc>>) {
    let resolved =
        temporal_phrase.and_then(|s| temporal::resolve_temporal_with_granularity(s, reference));
    let content = match (temporal_phrase, &resolved) {
        (Some(phrase), Some(r)) => {
            let date_str = r.point.format("%Y-%m-%d").to_string();
            content.replace(phrase, &date_str)
        }
        _ => content,
    };
    (content, resolved.map(|r| r.point))
}

// ── prep / apply split for the atomic ingest path ─────────────────

/// Inputs to [`prepare_observations`]. Bundles every read-only
/// dependency the prep phase needs. The atomic ingest orchestrator
/// builds one of these from typed inputs (no PipelineContext required).
#[derive(Debug)]
pub struct ObservationInputs<'a> {
    pub kb: &'a uniko_store::KnowledgeBase,
    /// VID of the Message that observations will reference.
    pub message_node_id: NodeId,
    pub content: &'a str,
    pub content_type: &'a str,
    /// Pre-resolved sender from the ingest step. When `None`, prep
    /// falls back to a SENT_BY lookup against `message_node_id`.
    pub sender: Option<(NodeId, String)>,
    /// `(NodeId, name)` pairs already created by the entity step.
    pub extracted_entities: &'a [(NodeId, String)],
    /// Per-sentence NLP results (POS / DEP / SRL / CLS). When `None`,
    /// the rule-based fallback runs.
    #[cfg(feature = "onnx")]
    pub nlp_results: Option<&'a [crate::nlp::types::NlpResult]>,
    /// Seed sentence context for pronoun resolution carried across
    /// messages within a session. `None` starts a fresh context.
    pub seed_sentence_ctx: Option<&'a crate::ingest::context::SentenceContext>,
    /// Effective message timestamp (from metadata or wall-clock).
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Optional path to a custom observation rules YAML; `None` uses
    /// bundled rules.
    pub observation_rules_path: Option<&'a std::path::Path>,
    /// The parent message's record category, inherited by every observation
    /// extracted from it (issue #39). Denormalised so the recall filter can
    /// reach derived items with a property predicate rather than walking
    /// OBSERVED_IN back to the message.
    pub category: Option<&'a str>,
    /// The parent message's logical source id, inherited likewise.
    pub source_id: Option<&'a str>,
    /// The parent's revision id (issue #41), inherited likewise.
    pub revision_id: Option<&'a str>,
}

/// Output of a successful [`prepare_observations`] call.
#[derive(Debug)]
pub struct ObservationPrep {
    pub all_obs: Vec<RawObservation>,
    pub used_model: bool,
    /// Resolved sender (either passed in or loaded via SENT_BY lookup).
    pub sender_ref: Option<(NodeId, String)>,
    /// Combined sender + entity refs used for ABOUT-edge construction.
    pub entity_refs: Vec<(NodeId, String)>,
    /// Updated sentence context for the session. `None` when the model
    /// path was not taken (or no NLP results were available).
    pub sentence_ctx_updated: Option<crate::ingest::context::SentenceContext>,
    pub sender_ms: u128,
    pub extract_ms: u128,
    /// Provenance inherited from the parent message.
    pub category: Option<String>,
    /// Provenance inherited from the parent message.
    pub source_id: Option<String>,
    /// Provenance inherited from the parent message.
    pub revision_id: Option<String>,
}

impl ObservationPrep {
    pub fn is_empty(&self) -> bool {
        self.all_obs.is_empty()
    }
}

/// Whether prep produced observations to write, or short-circuited.
#[derive(Debug)]
pub enum ObservationPrepOutcome {
    /// CLS gate / no-entities / empty extraction. Caller skips the write.
    Skip(String),
    /// Boxed: the prep is far larger than `Skip`, and an unboxed variant
    /// makes every `Skip` pay for it.
    Ready(Box<ObservationPrep>),
}

/// CPU + optional SENT_BY lookup. Does NOT open a transaction; does
/// NOT write anything. Returns `ObservationPrepOutcome::Skip` when
/// the CLS gate rejects the message or no observations were extracted.
///
/// # Errors
///
/// Returns [`UnikoError::Storage`] on a SENT_BY-lookup failure when
/// `sender` is `None` and the slow path is needed.
pub async fn prepare_observations(
    input: ObservationInputs<'_>,
) -> Result<ObservationPrepOutcome, UnikoError> {
    let step_start = std::time::Instant::now();

    // Slow-path sender resolution: only when caller didn't already
    // resolve it (e.g. legacy path through ObservationExtractionStep
    // without ctx.sender populated). The atomic orchestrator always
    // populates `input.sender` from MessageIngestResult, so this is
    // a no-op there.
    let sender_ref = if let Some(s) = input.sender.clone() {
        Some(s)
    } else {
        load_sender_ref_by_lookup(input.kb, input.message_node_id).await
    };
    let sender_ms = step_start.elapsed().as_millis();

    // `sender_name` is consumed only inside the `onnx` feature branch
    // below; suppress the unused warning on builds that exclude it.
    #[cfg_attr(not(feature = "onnx"), allow(unused_variables))]
    let sender_name = sender_ref.as_ref().map(|(_, name)| name.as_str());

    // 1. CLS gate (only when no per-sentence NLP results — those
    //    handle CLS per-sentence in the extraction loop below).
    #[cfg(feature = "onnx")]
    let has_nlp = input.nlp_results.is_some();
    #[cfg(not(feature = "onnx"))]
    let has_nlp = false;
    if !has_nlp && !filter::is_informative(input.content, Some(input.content_type)) {
        return Ok(ObservationPrepOutcome::Skip(
            "content not informative".into(),
        ));
    }

    // 2. Extract observations.
    let extract_start = std::time::Instant::now();
    let mut all_obs = Vec::new();
    #[allow(unused_mut)]
    let mut used_model = false;
    #[allow(unused_mut)]
    let mut sentence_ctx_updated: Option<crate::ingest::context::SentenceContext> = None;

    // 2a. Model-driven extraction from per-sentence DEP trees.
    #[cfg(feature = "onnx")]
    if let Some(nlp_results) = input.nlp_results {
        let labels = crate::nlp::assets::label_maps();
        let speaker = sender_name.unwrap_or("unknown");

        let mut sent_ctx = input
            .seed_sentence_ctx
            .cloned()
            .unwrap_or_else(|| crate::ingest::context::SentenceContext::new(speaker, Vec::new()));

        let rules = load_observation_rules_from_path(input.observation_rules_path);

        for nlp_result in nlp_results {
            if !cls_gate_admits(nlp_result, &labels.cls_labels, &rules.filters.cls_gate) {
                crate::nlp::decode::update_sentence_context(
                    &mut sent_ctx,
                    &nlp_result.words,
                    &nlp_result.pos_indices,
                    &nlp_result.dep_arcs,
                    &labels.pos_labels,
                );
                continue;
            }

            let dep_obs = crate::observations::rules_engine::extract_with_rules(
                rules,
                &nlp_result.words,
                &nlp_result.pos_indices,
                &nlp_result.dep_arcs,
                &labels.pos_labels,
                &nlp_result.srl_frames,
                speaker,
                &mut sent_ctx,
            );

            for obs in dep_obs {
                let temporal_phrase = obs.temporal.clone();
                let (content, temporal_anchor) = resolve_temporal_in_content(
                    obs.content,
                    temporal_phrase.as_deref(),
                    input.timestamp,
                );
                all_obs.push(RawObservation {
                    content,
                    subject: obs.subject,
                    predicate: obs.predicate,
                    object: obs.object,
                    temporal_phrase,
                    temporal_anchor,
                    observed_at: input.timestamp,
                    confidence: obs.confidence,
                });
            }
        }

        sentence_ctx_updated = Some(sent_ctx);
        used_model = true;
    }

    // 2b. Rule-based fallback (only when model unavailable).
    if !used_model {
        let entity_refs = combine_entity_refs(input.extracted_entities, &sender_ref);
        if !entity_refs.is_empty() {
            let rule_obs = rules::extract_observations_rule_based(
                input.content,
                &entity_refs,
                input.timestamp,
            );
            all_obs.extend(rule_obs);
        }
    }
    let extract_ms = extract_start.elapsed().as_millis();

    if all_obs.is_empty() {
        return Ok(ObservationPrepOutcome::Skip(
            "no observations extracted".into(),
        ));
    }

    let entity_refs = combine_entity_refs(input.extracted_entities, &sender_ref);

    Ok(ObservationPrepOutcome::Ready(Box::new(ObservationPrep {
        category: input.category.map(str::to_string),
        source_id: input.source_id.map(str::to_string),
        revision_id: input.revision_id.map(str::to_string),
        all_obs,
        used_model,
        sender_ref,
        entity_refs,
        sentence_ctx_updated,
        sender_ms,
        extract_ms,
    })))
}

/// Writes-only inside the caller's tx. Creates Observation nodes,
/// OBSERVED_IN edges (Obs → Message), and ABOUT edges (Obs → speaker
/// + matching entities).
///
/// Returns the new Observation node ids in input order. Caller owns
/// the commit.
///
/// # Errors
///
/// Returns [`UnikoError::Storage`] on any batched write failure.
pub async fn apply_observations(
    kb: &uniko_store::KnowledgeBase,
    tx: &uniko_store::Transaction,
    message_node_id: NodeId,
    prep: ObservationPrep,
) -> Result<Vec<NodeId>, UnikoError> {
    if prep.is_empty() {
        return Ok(Vec::new());
    }
    let ObservationPrep {
        all_obs,
        sender_ref,
        entity_refs,
        category,
        source_id,
        revision_id,
        ..
    } = prep;

    // 1. Batch create Observation nodes.
    let obs_props: Vec<HashMap<String, Value>> = all_obs
        .iter()
        .map(|raw| {
            let obs_id = uniko_store::id::new_id();
            let mut props = HashMap::new();
            props.insert("observation_id".into(), Value::String(obs_id));
            props.insert("content".into(), Value::String(raw.content.clone()));
            if let Some(ref category) = category {
                props.insert("category".into(), Value::String(category.clone()));
            }
            if let Some(ref source_id) = source_id {
                props.insert("source_id".into(), Value::String(source_id.clone()));
            }
            if let Some(ref revision_id) = revision_id {
                props.insert("revision_id".into(), Value::String(revision_id.clone()));
            }
            // Normalize the subject (the grouping/ABOUT key) so it keys
            // identically with Entity names and consolidation grouping. The
            // human-readable form lives in `content`, left untouched.
            props.insert(
                "subject".into(),
                Value::String(uniko_store::text::normalize_canonical(&raw.subject)),
            );
            if let Some(pred) = &raw.predicate {
                props.insert("predicate".into(), Value::String(pred.clone()));
            }
            if let Some(obj) = &raw.object {
                props.insert("object".into(), Value::String(obj.clone()));
            }
            if let Some(phrase) = &raw.temporal_phrase {
                props.insert("temporal_phrase".into(), Value::String(phrase.clone()));
            }
            if let Some(anchor) = raw.temporal_anchor {
                props.insert(
                    "temporal_anchor".into(),
                    uniko_store::types::datetime_value(anchor),
                );
            }
            props.insert(
                "observed_at".into(),
                uniko_store::types::datetime_value(raw.observed_at),
            );
            props.insert("confidence".into(), Value::Float(raw.confidence));
            props
        })
        .collect();
    let obs_create_start = std::time::Instant::now();
    let obs_count = obs_props.len();
    let obs_node_ids = kb
        .batch_create_nodes_in_tx(tx, labels::OBSERVATION, &obs_props)
        .await?;
    let obs_create_ms = obs_create_start.elapsed().as_millis();

    // 2. OBSERVED_IN edges (Observation → Message).
    let observed_start = std::time::Instant::now();
    let observed_edges: Vec<(NodeId, NodeId, HashMap<String, Value>)> = obs_node_ids
        .iter()
        .map(|&nid| (nid, message_node_id, HashMap::new()))
        .collect();
    kb.batch_create_edges_fast_in_tx(
        tx,
        edges::OBSERVED_IN,
        Some(labels::OBSERVATION),
        Some(labels::MESSAGE),
        &observed_edges,
    )
    .await?;
    let observed_ms = observed_start.elapsed().as_millis();

    // 3. ABOUT edges (Observation → speaker + matching entities).
    let about_build_start = std::time::Instant::now();
    let mut about_edges: Vec<(NodeId, NodeId, HashMap<String, Value>)> = Vec::new();
    for (i, raw) in all_obs.iter().enumerate() {
        let obs_nid = obs_node_ids[i];
        if let Some((sender_nid, _)) = &sender_ref {
            about_edges.push((obs_nid, *sender_nid, HashMap::new()));
        }
        for &(entity_nid, ref name) in &entity_refs {
            if sender_ref
                .as_ref()
                .is_some_and(|(nid, _)| *nid == entity_nid)
            {
                continue;
            }
            let subj = uniko_store::text::normalize_canonical(&raw.subject);
            let ename = uniko_store::text::normalize_canonical(name);
            if subj == ename || subj.contains(&ename) || ename.contains(&subj) {
                about_edges.push((obs_nid, entity_nid, HashMap::new()));
            }
        }
    }
    let about_build_ms = about_build_start.elapsed().as_millis();
    let about_count = about_edges.len();
    let about_write_start = std::time::Instant::now();
    if !about_edges.is_empty() {
        kb.batch_create_edges_fast_in_tx(
            tx,
            edges::ABOUT,
            Some(labels::OBSERVATION),
            None,
            &about_edges,
        )
        .await?;
    }
    let about_write_ms = about_write_start.elapsed().as_millis();

    tracing::info!(
        target: "apply_obs_breakdown",
        obs_create_ms = obs_create_ms as u64,
        observed_ms = observed_ms as u64,
        about_build_ms = about_build_ms as u64,
        about_write_ms = about_write_ms as u64,
        obs_count = obs_count as u64,
        about_count = about_count as u64,
        "apply_obs breakdown",
    );

    Ok(obs_node_ids)
}

/// Slow-path SENT_BY → Participant lookup. Only used when the caller
/// did not pre-resolve the sender (e.g. legacy ObservationExtractionStep
/// without ctx.sender).
async fn load_sender_ref_by_lookup(
    kb: &uniko_store::KnowledgeBase,
    message_node_id: NodeId,
) -> Option<(NodeId, String)> {
    use uniko_store::storage::edges::Direction;
    let edge_list = kb
        .get_edges(message_node_id, edges::SENT_BY, Direction::Outgoing)
        .await
        .ok()?;
    let edge = edge_list.first()?;
    let (_, props) = kb.get_node(edge.to).await.ok()??;
    let name = props.get("name").and_then(|v| v.as_str())?.to_string();
    Some((edge.to, name))
}

/// Combine sender ref (if any) with extracted entity refs, dedup by name.
fn combine_entity_refs(
    extracted_entities: &[(NodeId, String)],
    sender_ref: &Option<(NodeId, String)>,
) -> Vec<(NodeId, String)> {
    let mut refs: Vec<(NodeId, String)> =
        Vec::with_capacity(extracted_entities.len() + usize::from(sender_ref.is_some()));
    if let Some(sr) = sender_ref {
        refs.push(sr.clone());
    }
    for (nid, name) in extracted_entities {
        if !refs.iter().any(|(_, n)| n == name) {
            refs.push((*nid, name.clone()));
        }
    }
    refs
}

#[cfg(test)]
mod resolve_temporal_in_content_tests {
    use super::resolve_temporal_in_content;
    use chrono::{TimeZone, Utc};

    /// Conversation date 2023-05-08 (a Monday).  Used as the reference
    /// timestamp so "yesterday" resolves to 2023-05-07 and "last
    /// Friday" resolves to 2023-05-05.
    fn ref_date() -> chrono::DateTime<chrono::Utc> {
        Utc.with_ymd_and_hms(2023, 5, 8, 14, 0, 0).unwrap()
    }

    #[test]
    fn yesterday_substituted_with_iso_date() {
        let (content, anchor) = resolve_temporal_in_content(
            "Caroline went to the LGBTQ support group yesterday".into(),
            Some("yesterday"),
            ref_date(),
        );
        assert_eq!(
            content,
            "Caroline went to the LGBTQ support group 2023-05-07"
        );
        assert_eq!(
            anchor
                .expect("expected resolved anchor for 'yesterday'")
                .format("%Y-%m-%d")
                .to_string(),
            "2023-05-07"
        );
    }

    #[test]
    fn last_friday_substituted_with_iso_date() {
        // 2023-05-08 is Monday; "last Friday" → 2023-05-05.
        let (content, anchor) = resolve_temporal_in_content(
            "Melanie ran a charity race last Friday".into(),
            Some("last Friday"),
            ref_date(),
        );
        assert_eq!(content, "Melanie ran a charity race 2023-05-05");
        assert_eq!(anchor.unwrap().format("%Y-%m-%d").to_string(), "2023-05-05");
    }

    #[test]
    fn last_week_substituted_with_iso_date() {
        let (content, anchor) = resolve_temporal_in_content(
            "I went camping last week".into(),
            Some("last week"),
            ref_date(),
        );
        // resolve_temporal_with_granularity returns the start of "last
        // week" — the exact day depends on the implementation, but it
        // must NOT be the literal string "last week" and MUST be a
        // 10-character ISO date.
        assert!(
            !content.contains("last week"),
            "expected 'last week' to be substituted, got: {content:?}",
        );
        assert!(
            content.contains("I went camping "),
            "expected leading text preserved, got: {content:?}",
        );
        let date_str = anchor
            .expect("expected resolved anchor for 'last week'")
            .format("%Y-%m-%d")
            .to_string();
        assert!(
            content.ends_with(&date_str),
            "expected content to end with ISO date {date_str}, got: {content:?}",
        );
    }

    #[test]
    fn no_temporal_phrase_passes_content_through() {
        let (content, anchor) =
            resolve_temporal_in_content("Caroline bought a yellow dress".into(), None, ref_date());
        assert_eq!(content, "Caroline bought a yellow dress");
        assert!(anchor.is_none());
    }

    #[test]
    fn unparseable_phrase_leaves_content_untouched() {
        // "next Whitsun" — contains "sun" as a substring; pre-fix the
        // weekday regex matched it as a Sunday relative date.  After
        // the left-`\b` + hyphen-check tightening in
        // `parse_relative_weekday`, this phrase no longer resolves and
        // the content stays untouched.  Guards against the regex
        // regression returning.
        let (content, anchor) = resolve_temporal_in_content(
            "We celebrated next Whitsun".into(),
            Some("next Whitsun"),
            ref_date(),
        );
        assert_eq!(content, "We celebrated next Whitsun");
        assert!(
            anchor.is_none(),
            "unparseable phrase must not yield a resolved anchor"
        );
    }

    #[test]
    fn phrase_appearing_twice_both_substituted() {
        // Generous string replace — semantically what we want: any
        // occurrence of the temporal surface form refers to the same
        // resolved moment within a single observation.
        let (content, _) = resolve_temporal_in_content(
            "I went yesterday and yesterday it was sunny".into(),
            Some("yesterday"),
            ref_date(),
        );
        assert_eq!(content, "I went 2023-05-07 and 2023-05-07 it was sunny");
    }
}
