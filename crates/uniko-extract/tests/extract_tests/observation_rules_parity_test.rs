//! Parity test: legacy `extract_dep_observations` vs new rule-driven matcher.
//!
//! Each fixture builds a synthetic DEP tree, runs both the legacy
//! Rust function and the new YAML-driven matcher with the bundled
//! `english.yml`, and asserts they emit the same observation set.
//! Gated on the `onnx` feature — both `nlp` and `observations::rules_engine`
//! are themselves ONNX-gated.
//!
//! Run: cargo nextest run -p uniko-extract --features onnx --test integration observation_rules_parity_test::

#![cfg(feature = "onnx")]

use uniko_extract::ingest::context::SentenceContext;
use uniko_extract::nlp::decode::extract_dep_observations;
use uniko_extract::nlp::types::DepArc;
use uniko_extract::observations::rules_engine::{extract_with_rules, load_bundled_rules};

fn arc(dependent: usize, head: usize, relation: &str) -> DepArc {
    DepArc {
        dependent,
        head,
        relation: relation.to_string(),
    }
}

fn pos_labels() -> Vec<String> {
    vec![
        "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN",
        "PUNCT", "SCONJ", "SYM", "VERB", "X",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn pos(tag: &str) -> usize {
    pos_labels().iter().position(|t| t == tag).unwrap_or(16)
}

fn assert_parity(
    name: &str,
    words: Vec<&str>,
    pos_indices: Vec<usize>,
    dep_arcs: Vec<DepArc>,
    speaker: &str,
) {
    let words: Vec<String> = words.into_iter().map(String::from).collect();

    let mut ctx_legacy = SentenceContext::new(speaker, vec!["Caroline".into()]);
    let legacy = extract_dep_observations(
        &words,
        &pos_indices,
        &dep_arcs,
        &pos_labels(),
        speaker,
        &mut ctx_legacy,
    );

    let mut ctx_new = SentenceContext::new(speaker, vec!["Caroline".into()]);
    let rules = load_bundled_rules();
    let new = extract_with_rules(
        rules,
        &words,
        &pos_indices,
        &dep_arcs,
        &pos_labels(),
        &[],
        speaker,
        &mut ctx_new,
    );

    let legacy_texts: Vec<String> = legacy.iter().map(|o| o.content.clone()).collect();
    let new_texts: Vec<String> = new.iter().map(|o| o.content.clone()).collect();
    // No-regression invariant: every observation the legacy extractor
    // emitted must still appear in the new matcher's output. The new
    // matcher may emit additional observations from Phase B patterns
    // (oblique_state_modifier, appositional_identity); we don't penalise
    // those here.
    for want in &legacy_texts {
        assert!(
            new_texts.contains(want),
            "[{name}] legacy observation lost in new matcher:\n  missing = {want:?}\n  legacy  = {legacy_texts:?}\n  new     = {new_texts:?}",
        );
    }

    assert_eq!(
        ctx_legacy.last_noun_subject, ctx_new.last_noun_subject,
        "[{name}] last_noun_subject diverged",
    );
    assert_eq!(
        ctx_legacy.last_noun_object, ctx_new.last_noun_object,
        "[{name}] last_noun_object diverged",
    );
}

#[test]
fn parity_simple_svo_with_speaker() {
    // "I lost my job"
    assert_parity(
        "simple_svo",
        vec!["I", "lost", "my", "job"],
        vec![pos("PRON"), pos("VERB"), pos("DET"), pos("NOUN")],
        vec![
            arc(0, 1, "nsubj"),
            arc(1, usize::MAX, "root"),
            arc(2, 3, "det"),
            arc(3, 1, "obj"),
        ],
        "Jon",
    );
}

#[test]
fn parity_third_person_keeps_name() {
    // "Caroline is happy" — copular adj
    assert_parity(
        "third_person_copular",
        vec!["Caroline", "is", "happy"],
        vec![pos("PROPN"), pos("AUX"), pos("ADJ")],
        vec![
            arc(0, 2, "nsubj"),
            arc(1, 2, "cop"),
            arc(2, usize::MAX, "root"),
        ],
        "Jon",
    );
}

#[test]
fn parity_verbal_with_oblique_modifier() {
    // "Jon got a temp job at the cafe"
    assert_parity(
        "verbal_with_obl",
        vec!["Jon", "got", "a", "temp", "job", "at", "the", "cafe"],
        vec![
            pos("PROPN"),
            pos("VERB"),
            pos("DET"),
            pos("NOUN"),
            pos("NOUN"),
            pos("ADP"),
            pos("DET"),
            pos("NOUN"),
        ],
        vec![
            arc(0, 1, "nsubj"),
            arc(1, usize::MAX, "root"),
            arc(2, 4, "det"),
            arc(3, 4, "compound"),
            arc(4, 1, "obj"),
            arc(5, 7, "case"),
            arc(6, 7, "det"),
            arc(7, 1, "obl"),
        ],
        "Jon",
    );
}

#[test]
fn parity_no_subject_no_object_skipped() {
    // "Running quickly" — VERB only, no nsubj/obj.
    assert_parity(
        "no_subject_no_obj",
        vec!["Running", "quickly"],
        vec![pos("VERB"), pos("ADV")],
        vec![arc(0, usize::MAX, "root"), arc(1, 0, "advmod")],
        "Jon",
    );
}

// ── Phase B: new patterns ─────────────────────────────────────────

fn run_new_matcher(
    words: Vec<&str>,
    pos_indices: Vec<usize>,
    dep_arcs: Vec<DepArc>,
    speaker: &str,
    other_speakers: Vec<String>,
) -> Vec<String> {
    let words: Vec<String> = words.into_iter().map(String::from).collect();
    let mut ctx = SentenceContext::new(speaker, other_speakers);
    let rules = load_bundled_rules();
    extract_with_rules(
        rules,
        &words,
        &pos_indices,
        &dep_arcs,
        &pos_labels(),
        &[],
        speaker,
        &mut ctx,
    )
    .into_iter()
    .map(|o| o.content)
    .collect()
}

#[test]
fn phase_b_oblique_state_modifier_single_parent() {
    // Approximates D2:14 "It'll be tough as a single parent" —
    // strip the unresolvable "it'll" and use "Caroline" subject:
    //   "Caroline is tough as a single parent"
    // After case-stripping: ADJ "tough" with cop "is", nsubj "Caroline",
    // obl "parent" (compound "single", det "a").
    let texts = run_new_matcher(
        vec!["Caroline", "is", "tough", "as", "a", "single", "parent"],
        vec![
            pos("PROPN"),
            pos("AUX"),
            pos("ADJ"),
            pos("ADP"),
            pos("DET"),
            pos("ADJ"),
            pos("NOUN"),
        ],
        vec![
            arc(0, 2, "nsubj"),
            arc(1, 2, "cop"),
            arc(2, usize::MAX, "root"),
            arc(3, 6, "case"),
            arc(4, 6, "det"),
            arc(5, 6, "amod"),
            arc(6, 2, "obl"),
        ],
        "Jon",
        vec!["Caroline".into()],
    );
    assert!(
        texts
            .iter()
            .any(|t| t.contains("Caroline") && t.contains("tough") && t.contains("single parent")),
        "expected an observation joining 'Caroline', 'tough', 'single parent', got: {texts:?}",
    );
}

#[test]
fn phase_b_oblique_state_modifier_breakup() {
    // Approximates D3:13 "Caroline is important after that tough breakup":
    // ADJ "important" anchors with cop, nsubj "Caroline", obl "breakup"
    // (det "that", amod "tough").
    let texts = run_new_matcher(
        vec![
            "Caroline",
            "is",
            "important",
            "after",
            "that",
            "tough",
            "breakup",
        ],
        vec![
            pos("PROPN"),
            pos("AUX"),
            pos("ADJ"),
            pos("ADP"),
            pos("DET"),
            pos("ADJ"),
            pos("NOUN"),
        ],
        vec![
            arc(0, 2, "nsubj"),
            arc(1, 2, "cop"),
            arc(2, usize::MAX, "root"),
            arc(3, 6, "case"),
            arc(4, 6, "det"),
            arc(5, 6, "amod"),
            arc(6, 2, "obl"),
        ],
        "Jon",
        vec!["Caroline".into()],
    );
    assert!(
        texts
            .iter()
            .any(|t| t.contains("Caroline") && t.contains("breakup")),
        "expected an observation joining 'Caroline' and 'breakup', got: {texts:?}",
    );
}

#[test]
fn phase_b_appositional_identity_sweden() {
    // Approximates D4:3 "my home country, Sweden":
    // NOUN "country" anchors with appos "Sweden". compound "home" +
    // nmod:poss "my" join the anchor's noun phrase.
    let texts = run_new_matcher(
        vec!["my", "home", "country", "Sweden"],
        vec![pos("PRON"), pos("NOUN"), pos("NOUN"), pos("PROPN")],
        vec![
            arc(0, 2, "nmod:poss"),
            arc(1, 2, "compound"),
            arc(2, usize::MAX, "root"),
            arc(3, 2, "appos"),
        ],
        "Jon",
        vec!["Caroline".into()],
    );
    assert!(
        texts
            .iter()
            .any(|t| t.contains("Sweden") && t.contains("country") && t.contains("Jon")),
        "expected an apposition observation linking speaker, 'country', and 'Sweden', got: {texts:?}",
    );
}

#[test]
fn parity_unresolvable_pronoun_dropped() {
    // "which is exciting" — relative pronoun anchor → drop.
    assert_parity(
        "unresolvable_pron",
        vec!["which", "is", "exciting"],
        vec![pos("PRON"), pos("AUX"), pos("ADJ")],
        vec![
            arc(0, 2, "nsubj"),
            arc(1, 2, "cop"),
            arc(2, usize::MAX, "root"),
        ],
        "Jon",
    );
}
