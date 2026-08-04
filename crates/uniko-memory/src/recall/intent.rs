//! Intent profile construction for recall queries.

use chrono::{DateTime, Utc};
use futures::future::join_all;
use uniko_extract::ner::rules::extract_entities_rule_based;
use uniko_extract::observations::temporal::resolve_temporal_with_granularity;
use uniko_store::{KnowledgeBase, UnikoError};

/// Content POS tags to keep for keyword extraction.
/// NOUN, VERB, PROPN, ADJ, NUM — drop function words.
#[cfg(feature = "onnx")]
const CONTENT_POS: &[&str] = &["NOUN", "VERB", "PROPN", "ADJ", "NUM"];

/// Variant labels recognised when building an [`IntentProfile`].
pub const VARIANT_KEYWORDS: &str = "keywords";
pub const VARIANT_ORIGINAL: &str = "original";
pub const VARIANT_DECLARATIVE: &str = "declarative";
pub const VARIANT_TYPE_ANCHORED: &str = "type_anchored";

/// One query reformulation. Each variant runs through the full recall
/// pipeline (hybrid + entity-scoped Cypher fan-out) on its own and is
/// fused with the others via reciprocal-rank fusion in the caller.
#[derive(Debug, Clone)]
pub struct QueryVariant {
    /// Stable label (`"original"`, `"keywords"`, `"declarative"`,
    /// `"type_anchored"`). Used for tracing and CLI selection.
    pub label: &'static str,
    /// Human-readable form. Doubles as the BM25 query string (`$qtxt`)
    /// inside the Cypher templates.
    pub text: String,
    /// Embedding of `text`. Empty when the embedder failed; downstream
    /// code falls back to BM25-only paths in that case.
    pub vector: Vec<f32>,
}

/// Structured representation of a recall query.
#[derive(Debug, Clone)]
pub struct IntentProfile {
    /// Per-variant `(text, embedding)` pairs. Always non-empty: at
    /// minimum the `keywords` variant is present (or `original` as a
    /// fallback when keyword stripping yields nothing).
    pub variants: Vec<QueryVariant>,
    /// Entity names extracted from the query, used for entity-scoped
    /// Cypher fan-out across all variants.
    pub entity_refs: Vec<String>,
    /// Number of facets: `max(entity_refs.len(), 1)`.
    pub facet_count: usize,
    /// Predicted entity_type of the answer, when the question's surface
    /// form gives a clear cue ("where" → location, "who" → person,
    /// "how many" → measurement, …). `None` when no rule fires —
    /// downstream code should treat absence as "no signal", not
    /// "anything goes". Populated by [`predict_answer_type`].
    pub expected_answer_type: Option<&'static str>,
    /// Half-open `[lo, hi)` temporal window parsed from the query (e.g.
    /// "last May" → `[2025-05-01, 2025-06-01)`).  `None` when no
    /// temporal phrase fires.  Drives the temporal-interval channel in
    /// Phase 2 recall.
    pub temporal_window: Option<(DateTime<Utc>, DateTime<Utc>)>,
    /// Modality of the *query* input (not the corpus). Always
    /// `QueryModalities::default()` for text-only callers; Track B's
    /// image/audio query entry points populate `image_vec` / `audio_vec`.
    /// Phase 3 lands the field plumbing so Track B drops in without
    /// touching IntentProfile call sites.
    pub query_modalities: QueryModalities,
}

/// Modality of a query input, as opposed to the corpus's
/// [`crate::recall::modality::ModalityPresence`] which describes the KB.
///
/// All fields default to "text-only" (no image/audio attached to the
/// query); a text-to-image search would flip `has_image_input` and
/// populate `image_vec`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct QueryModalities {
    /// User supplied an image alongside the text query.
    pub has_image_input: bool,
    /// User supplied audio alongside the text query.
    pub has_audio_input: bool,
    /// SigLIP-2 vision embedding of the query image, when present.
    pub image_vec: Option<Vec<f32>>,
    /// Joint-space audio embedding of the query audio, when present.
    pub audio_vec: Option<Vec<f32>>,
}

impl IntentProfile {
    /// First variant's vector. Kept for back-compat with callers that
    /// expect a single embedding (e.g. legacy tests). New code should
    /// iterate `variants`.
    pub fn intent_vec(&self) -> &[f32] {
        self.variants
            .first()
            .map(|v| v.vector.as_slice())
            .unwrap_or(&[])
    }

    /// First variant's text. Same back-compat reason as [`intent_vec`].
    pub fn keywords(&self) -> &str {
        self.variants.first().map(|v| v.text.as_str()).unwrap_or("")
    }

    /// Resolve [`Self::entity_refs`] to graph seed [`NodeId`]s for the
    /// spreading-activation channel.
    ///
    /// Each entity name is matched against `Entity.name` and
    /// `Participant.name` in a single Cypher round-trip — both labels
    /// are equally valid recall anchors (Caroline/Melanie show up as
    /// Participants in the bench corpus, named entities like locations
    /// or organisations as Entities).  Returns a deduped Vec of
    /// resolved NodeIds; empty when no entity_ref resolves or when
    /// `entity_refs` itself is empty.
    pub async fn resolve_seeds(&self, kb: &uniko_store::KnowledgeBase) -> Vec<uniko_store::NodeId> {
        kb.resolve_entity_seeds(&self.entity_refs).await
    }
}

/// Build an [`IntentProfile`] from a query string.
///
/// Runs the NLP cascade once on the question, then derives 1–4 query
/// variants whose embeddings are computed concurrently via `join_all`.
/// `enabled_variants` filters which variants get built. An empty slice is
/// the legacy single-variant default and builds only `keywords` — see the
/// rationale on the `want` closure below. Pass all four labels
/// (`keywords`, `original`, `declarative`, `type_anchored`) to opt into
/// full multi-query reformulation.
///
/// # Errors
///
/// Returns [`UnikoError::Embedding`] if the embedding runtime is
/// unavailable for *every* requested variant.
pub async fn build_intent(
    kb: &KnowledgeBase,
    query: &str,
    enabled_variants: &[String],
) -> Result<IntentProfile, UnikoError> {
    build_intent_at(kb, query, enabled_variants, None).await
}

/// Like [`build_intent`] but takes a `reference_ts` for temporal phrase
/// resolution.  Callers that know the conversation/session time should
/// pass it so phrases like "last May" resolve against the right
/// baseline rather than wall-clock now.
///
/// `reference_ts = None` falls back to `Utc::now()`.
///
/// # Errors
///
/// Returns [`UnikoError::Embedding`] if the embedding runtime is
/// unavailable for *every* requested variant.
pub async fn build_intent_at(
    kb: &KnowledgeBase,
    query: &str,
    enabled_variants: &[String],
    reference_ts: Option<DateTime<Utc>>,
) -> Result<IntentProfile, UnikoError> {
    let analysis = analyze_query(kb, query).await;
    let entity_refs = analysis.entity_refs.clone();
    let facet_count = entity_refs.len().max(1);
    let expected_answer_type = predict_answer_type(query);

    if let Some(t) = expected_answer_type {
        tracing::info!(expected_answer_type = %t, query = %query, "intent: answer type predicted");
    }

    // Build the variant text list. Order matters for back-compat:
    // `intent_vec()` and `keywords()` return the first variant.
    //
    // Empty `enabled_variants` is the *legacy single-variant* default
    // (just the POS-stripped `keywords` variant). The full 4-variant
    // multi-query reformulation must be opted in explicitly, e.g.
    // `--variants keywords,original,declarative,type_anchored`. This
    // is because measured A/B on full LoCoMo (757 questions, 4 of 10
    // conversations, MiniLM cross-encoder reranker on GPU, 2026-05-04)
    // showed multi-variant regressing overall evidence% by 2.1 points
    // and tripling per-query latency vs single-variant. The variant
    // infrastructure stays in place for future experimentation; the
    // *default* is the safe choice.
    let want = |name: &str| {
        if enabled_variants.is_empty() {
            name == VARIANT_KEYWORDS
        } else {
            enabled_variants.iter().any(|s| s == name)
        }
    };

    let mut variant_texts: Vec<(&'static str, String)> = Vec::new();

    if want(VARIANT_KEYWORDS) && !analysis.keywords.is_empty() {
        variant_texts.push((VARIANT_KEYWORDS, analysis.keywords.clone()));
    }
    if want(VARIANT_ORIGINAL) {
        variant_texts.push((VARIANT_ORIGINAL, query.to_string()));
    }
    if want(VARIANT_DECLARATIVE)
        && let Some(text) = build_declarative_text(&analysis)
    {
        variant_texts.push((VARIANT_DECLARATIVE, text));
    }
    if want(VARIANT_TYPE_ANCHORED)
        && let Some(text) = build_type_anchored_text(&entity_refs, expected_answer_type)
    {
        variant_texts.push((VARIANT_TYPE_ANCHORED, text));
    }

    // Always make sure we have at least one variant; the embedded text
    // is the safety net.
    if variant_texts.is_empty() {
        variant_texts.push((VARIANT_ORIGINAL, query.to_string()));
    }

    // Dedup on text — variants that happen to produce identical strings
    // (e.g. POS-stripping a 3-word question) waste an embedder call.
    variant_texts.dedup_by(|a, b| a.1 == b.1);

    // Embed every variant concurrently.  Failures degrade to an empty
    // vector so downstream BM25-only paths still fire; log at debug so
    // a silently misconfigured embedder still surfaces in traces.
    let embed_futures = variant_texts.iter().map(|(label, text)| {
        let text = text.clone();
        async move {
            match uniko_extract::embedding::embed_query(kb, &text).await {
                Ok(v) => v,
                Err(e) => {
                    tracing::debug!(variant = label, error = %e, "embed_query failed");
                    Vec::new()
                }
            }
        }
    });
    let vectors: Vec<Vec<f32>> = join_all(embed_futures).await;

    let variants: Vec<QueryVariant> = variant_texts
        .into_iter()
        .zip(vectors)
        .map(|((label, text), vector)| {
            tracing::info!(
                variant = label,
                text = %text,
                vec_len = vector.len(),
                "intent: variant built",
            );
            QueryVariant {
                label,
                text,
                vector,
            }
        })
        .collect();

    // Resolve any temporal phrase in the query into a half-open
    // [lo, hi) window for the temporal recall channel.  Reference time
    // is the caller-supplied conversation timestamp when available, else
    // wall-clock now (least informative — relative phrases like "last
    // week" become meaningless in this fallback).
    let reference = reference_ts.unwrap_or_else(Utc::now);
    let temporal_window = resolve_temporal_with_granularity(query, reference).map(|r| r.to_range());
    if let Some((lo, hi)) = temporal_window {
        tracing::info!(
            lo = %lo,
            hi = %hi,
            query = %query,
            "intent: temporal window resolved",
        );
    }

    Ok(IntentProfile {
        variants,
        entity_refs,
        facet_count,
        expected_answer_type,
        temporal_window,
        query_modalities: QueryModalities::default(),
    })
}

/// Bundle of derived facts about a question, used by variant builders.
#[derive(Debug, Clone, Default)]
struct QueryAnalysis {
    /// POS-filtered keywords (NOUN/VERB/PROPN/ADJ/NUM joined by spaces).
    /// Empty when the NLP pipeline is unavailable.
    keywords: String,
    /// Normalised entity names from NER.
    entity_refs: Vec<String>,
    /// SVO triples extracted from the question's DEP tree (only when
    /// the ONNX cascade ran). Empty otherwise.
    #[cfg(feature = "onnx")]
    svo_triples: Vec<uniko_extract::nlp::decode::SvoTriple>,
}

/// Build a declarative-form variant from the question's SVO structure.
///
/// For *"What pet does Caroline have?"* the DEP parse gives
/// `Caroline -[nsubj]-> have` and `pet -[obj]-> have`. The active form
/// is *"Caroline have pet"* — strips the WH-leader, keeps the slot
/// filler. Embeds closer to statement-form content in the index than
/// the question form does.
fn build_declarative_text(analysis: &QueryAnalysis) -> Option<String> {
    #[cfg(feature = "onnx")]
    {
        let triple = analysis.svo_triples.first()?;
        let mut parts: Vec<&str> = Vec::with_capacity(3);
        if !triple.subject.is_empty() {
            parts.push(triple.subject.as_str());
        }
        parts.push(triple.verb.as_str());
        if !triple.object.is_empty() {
            parts.push(triple.object.as_str());
        }
        let joined = parts.join(" ");
        if joined.split_whitespace().count() >= 2 {
            Some(joined)
        } else {
            None
        }
    }
    #[cfg(not(feature = "onnx"))]
    {
        let _ = analysis;
        None
    }
}

/// Build a type-anchored variant by joining entity names with a token
/// hinting at the expected answer's entity type.
///
/// For *"What pet does Caroline have?"* with `entity_refs=["Caroline"]`
/// and `expected_answer_type=Some("other")`, returns *"Caroline other"*.
/// The type word doubles as both a BM25 keyword bias and a vector-space
/// nudge toward observations mentioning that category.
fn build_type_anchored_text(
    entity_refs: &[String],
    expected_answer_type: Option<&'static str>,
) -> Option<String> {
    let t = expected_answer_type?;
    if entity_refs.is_empty() {
        return None;
    }
    let mut s = entity_refs.join(" ");
    s.push(' ');
    s.push_str(t);
    Some(s)
}

/// Predict the expected entity_type of a question's answer from its
/// surface form. Returns one of the strings stored in `Entity.entity_type`
/// (`"person"`, `"location"`, `"date"`, `"measurement"`,
/// `"organization"`, `"work_of_art"`), or `None` when no rule fires.
///
/// Pure regex over the question text; no model call. Tuned for high
/// precision over recall — emitting a wrong type would actively
/// mislead downstream reweighting, while emitting `None` just falls
/// back to baseline retrieval.
pub fn predict_answer_type(question: &str) -> Option<&'static str> {
    let q = question.trim().to_lowercase();
    // Order matters: more-specific patterns first.
    const RULES: &[(&str, &str)] = &[
        // Person
        (r"^(who|whose|by whom|with whom|to whom)\b", "person"),
        (
            r"\bwhich (person|friend|coworker|colleague|teacher|doctor|partner)\b",
            "person",
        ),
        (r"\bwhat is (his|her|their) name\b", "person"),
        // Location
        (r"^where\b|^from where\b|^to where\b", "location"),
        (
            r"\b(which|what) (city|country|place|state|park|address|venue|building|street|neighborhood|hotel|restaurant|cafe|store|airport)\b",
            "location",
        ),
        // Date/Time
        (r"^when\b", "date"),
        (r"\bwhat (time|day|date|year|month|hour)\b", "date"),
        (r"\bhow long ago\b", "date"),
        // Measurement / Numeric
        (
            r"^how (many|much|long|old|far|tall|big|fast|heavy|deep|wide)\b",
            "measurement",
        ),
        (
            r"\bhow many (days|years|months|weeks|hours|minutes|seconds|miles|km|kilometers)\b",
            "measurement",
        ),
        // Organization
        (
            r"\b(which|what) (company|organization|firm|university|college|school|team|brand|agency)\b",
            "organization",
        ),
        // Work of art (currently bucketed under `other` in our schema)
        (
            r"\b(which|what) (movie|book|song|album|game|show|film|novel|poem|painting)\b",
            "other",
        ),
    ];

    use regex::Regex;
    use std::sync::OnceLock;
    static COMPILED: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    let compiled = COMPILED.get_or_init(|| {
        RULES
            .iter()
            .map(|(pat, label)| (Regex::new(pat).expect("rule regex"), *label))
            .collect()
    });
    for (re, label) in compiled {
        if re.is_match(&q) {
            return Some(label);
        }
    }
    None
}

/// Analyze query with NLP pipeline to extract entities, content
/// keywords, and SVO triples that downstream variant builders consume.
///
/// Falls back to rule-based NER + raw query text when the ONNX cascade
/// is unavailable.
async fn analyze_query(kb: &KnowledgeBase, query: &str) -> QueryAnalysis {
    #[cfg(feature = "onnx")]
    if let Some(analysis) = analyze_query_onnx(kb, query).await {
        return analysis;
    }
    #[cfg(not(feature = "onnx"))]
    let _ = kb;

    // Fallback: rule-based NER, raw query as keywords.
    let raw = extract_entities_rule_based(query);
    let entity_refs: Vec<String> = raw
        .into_iter()
        .filter_map(|e| normalize_entity_text(&e.canonical_name))
        .collect();
    QueryAnalysis {
        keywords: query.to_string(),
        entity_refs,
        #[cfg(feature = "onnx")]
        svo_triples: Vec::new(),
    }
}

/// ONNX-cascade query analysis: NER + POS-filtered keywords + SVO
/// triples from the DEP tree.  Returns `None` when the NLP pipeline is
/// unavailable or `analyze()` fails — caller falls back to rule-based
/// NER in that case.
#[cfg(feature = "onnx")]
async fn analyze_query_onnx(kb: &KnowledgeBase, query: &str) -> Option<QueryAnalysis> {
    let pipeline = uniko_extract::nlp::NlpPipeline::try_new(kb)
        .await
        .or_else(|| {
            tracing::warn!("NLP pipeline unavailable for query analysis");
            None
        })?;
    let result = pipeline.analyze(query).await.ok()?;
    let labels = uniko_extract::nlp::assets::label_maps();

    // Entity names normalized for matching against node `name`
    // properties: strip trailing punctuation and possessive `'s`/`'`,
    // drop empty tokens.  Without this, `MATCH (:Participant
    // {name: "Caroline's"})` silently returns nothing.
    let entity_refs: Vec<String> = result
        .entities
        .iter()
        .filter_map(|e| normalize_entity_text(&e.text))
        .collect();

    let keywords: Vec<&str> = result
        .words
        .iter()
        .zip(result.pos_indices.iter())
        .filter_map(|(word, &pos_idx)| {
            let pos = labels
                .pos_labels
                .get(pos_idx)
                .map(String::as_str)
                .unwrap_or("");
            CONTENT_POS.contains(&pos).then_some(word.as_str())
        })
        .collect();

    // Walk the DEP tree to pull SVO triples for the declarative
    // variant.  Empty for short questions with no parsable verb — the
    // declarative variant builder will then skip itself.
    let svo_triples = uniko_extract::nlp::decode::extract_svo_triples(
        &result.words,
        &result.pos_indices,
        &result.dep_arcs,
        &labels.pos_labels,
    );

    let kw_str = if keywords.is_empty() {
        String::new()
    } else {
        keywords.join(" ")
    };

    tracing::info!(
        entities = ?entity_refs,
        keywords = %kw_str,
        svo_count = svo_triples.len(),
        "NLP query analysis",
    );

    Some(QueryAnalysis {
        keywords: kw_str,
        entity_refs,
        svo_triples,
    })
}

/// Normalize an entity span captured from a question for matching
/// against stored node `name` properties.
///
/// Strips trailing punctuation (`?`, `.`, `,`, `!`) and possessive
/// suffixes (`'s`, `\u{2019}s`, trailing `'`). Returns `None` for
/// empty / whitespace-only / single-char results (those are almost
/// always extraction noise, not real entities).
fn normalize_entity_text(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    let stripped = trimmed.trim_end_matches(['?', '.', ',', '!', ';', ':', '"']);
    let no_poss = stripped
        .strip_suffix("\u{2019}s")
        .or_else(|| stripped.strip_suffix("'s"))
        .or_else(|| stripped.strip_suffix('\u{2019}'))
        .or_else(|| stripped.strip_suffix('\''))
        .unwrap_or(stripped);
    let cleaned = no_poss.trim().to_string();
    if cleaned.chars().count() < 2 {
        None
    } else {
        Some(cleaned)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicts_basic_wh() {
        assert_eq!(
            predict_answer_type("Who attended the wedding?"),
            Some("person")
        );
        assert_eq!(
            predict_answer_type("Where did I attend my cousin's wedding?"),
            Some("location"),
        );
        assert_eq!(
            predict_answer_type("When did I book the Airbnb?"),
            Some("date"),
        );
        assert_eq!(
            predict_answer_type("How many items of clothing do I need?"),
            Some("measurement"),
        );
        assert_eq!(
            predict_answer_type("What time do I wake up on Tuesdays?"),
            Some("date"),
        );
        assert_eq!(
            predict_answer_type("What book am I currently reading?"),
            Some("other"),
        );
        assert_eq!(
            predict_answer_type("What university did I attend?"),
            Some("organization"),
        );
    }

    #[test]
    fn unknowns_return_none() {
        assert_eq!(predict_answer_type("What did I buy for my sister?"), None);
        assert_eq!(predict_answer_type("Why did I quit?"), None);
        assert_eq!(
            predict_answer_type("Tell me what was the rotation for Admon."),
            None,
        );
    }

    #[test]
    fn variant_type_anchored_with_known_type() {
        let entity_refs = vec!["Caroline".to_string()];
        let got = build_type_anchored_text(&entity_refs, Some("other"));
        assert_eq!(got.as_deref(), Some("Caroline other"));

        let multi = vec!["Caroline".to_string(), "Sweden".to_string()];
        let got = build_type_anchored_text(&multi, Some("date"));
        assert_eq!(got.as_deref(), Some("Caroline Sweden date"));
    }

    #[test]
    fn variant_type_anchored_skipped_when_inputs_missing() {
        // No expected type → no variant.
        let entity_refs = vec!["Caroline".to_string()];
        assert!(build_type_anchored_text(&entity_refs, None).is_none());
        // No entities → no variant.
        let empty: Vec<String> = vec![];
        assert!(build_type_anchored_text(&empty, Some("person")).is_none());
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn variant_declarative_from_svo_triple() {
        use uniko_extract::nlp::decode::SvoTriple;
        let analysis = QueryAnalysis {
            keywords: "pet Caroline have".to_string(),
            entity_refs: vec!["Caroline".to_string()],
            svo_triples: vec![SvoTriple {
                subject: "Caroline".to_string(),
                verb: "have".to_string(),
                object: "pet".to_string(),
            }],
        };
        let got = build_declarative_text(&analysis);
        assert_eq!(got.as_deref(), Some("Caroline have pet"));
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn variant_declarative_skipped_when_no_svo() {
        let analysis = QueryAnalysis {
            keywords: "pet".to_string(),
            entity_refs: vec![],
            svo_triples: vec![],
        };
        assert!(build_declarative_text(&analysis).is_none());
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn variant_declarative_skipped_when_too_short() {
        // Only a verb, no subject and no object → fewer than 2 tokens.
        use uniko_extract::nlp::decode::SvoTriple;
        let analysis = QueryAnalysis {
            keywords: String::new(),
            entity_refs: vec![],
            svo_triples: vec![SvoTriple {
                subject: String::new(),
                verb: "go".to_string(),
                object: String::new(),
            }],
        };
        assert!(build_declarative_text(&analysis).is_none());
    }
}
