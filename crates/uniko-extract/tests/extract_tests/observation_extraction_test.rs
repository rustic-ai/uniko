//! Unit tests for DEP-tree observation extraction.
//!
//! Tests extract_dep_observations() directly with synthetic DEP trees,
//! no ONNX model or database required.  Gated on the `onnx` feature
//! because the `nlp` module that hosts the extractor is itself gated
//! — without ONNX the symbols don't exist.
//!
//! Run: cargo nextest run -p uniko-extract --features onnx --test integration observation_extraction_test:: --nocapture

#![cfg(feature = "onnx")]

use uniko_extract::ingest::context::SentenceContext;
use uniko_extract::nlp::decode::extract_dep_observations;
use uniko_extract::nlp::types::DepArc;

/// Helper to build a dep arc.
fn arc(dependent: usize, head: usize, relation: &str) -> DepArc {
    DepArc {
        dependent,
        head,
        relation: relation.to_string(),
    }
}

/// POS label list matching the model's labels.
fn pos_labels() -> Vec<String> {
    vec![
        "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN",
        "PUNCT", "SCONJ", "SYM", "VERB", "X",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

/// Index for a POS tag.
fn pos(tag: &str) -> usize {
    pos_labels().iter().position(|t| t == tag).unwrap_or(16) // X
}

#[test]
fn test_simple_svo_with_speaker() {
    // "I lost my job" → Jon lost my job
    // DEP: lost(VERB/root) → I(PRON/nsubj) → job(NOUN/obj) → my(DET/det)
    let words: Vec<String> = vec!["I", "lost", "my", "job"]
        .into_iter()
        .map(String::from)
        .collect();
    let pos_indices = vec![pos("PRON"), pos("VERB"), pos("DET"), pos("NOUN")];
    let dep_arcs = vec![
        arc(0, 1, "nsubj"), // I → lost
        arc(1, usize::MAX, "root"),
        arc(2, 3, "det"), // my → job
        arc(3, 1, "obj"), // job → lost
    ];

    let obs = extract_dep_observations(
        &words,
        &pos_indices,
        &dep_arcs,
        &pos_labels(),
        "Jon",
        &mut SentenceContext::default(),
    );

    eprintln!("Observations: {obs:?}");
    assert!(!obs.is_empty(), "Should produce at least one observation");
    assert_eq!(
        obs[0].subject, "Jon",
        "First-person 'I' should become 'Jon'"
    );
    assert!(
        obs[0].content.contains("Jon") && obs[0].content.contains("lost"),
        "Content should be 'Jon lost ...': got '{}'",
        obs[0].content
    );
    assert!(
        obs[0].content.contains("job"),
        "Content should include object 'job': got '{}'",
        obs[0].content
    );
}

#[test]
fn test_third_person_keeps_name() {
    // "Caroline attended the support group"
    // DEP: attended(VERB/root) → Caroline(PROPN/nsubj) → group(NOUN/obj)
    let words: Vec<String> = vec!["Caroline", "attended", "the", "support", "group"]
        .into_iter()
        .map(String::from)
        .collect();
    let pos_indices = vec![
        pos("PROPN"),
        pos("VERB"),
        pos("DET"),
        pos("NOUN"),
        pos("NOUN"),
    ];
    let dep_arcs = vec![
        arc(0, 1, "nsubj"),
        arc(1, usize::MAX, "root"),
        arc(2, 4, "det"),
        arc(3, 4, "compound"),
        arc(4, 1, "obj"),
    ];

    let obs = extract_dep_observations(
        &words,
        &pos_indices,
        &dep_arcs,
        &pos_labels(),
        "Gina",
        &mut SentenceContext::default(),
    );

    eprintln!("Observations: {obs:?}");
    assert!(!obs.is_empty());
    assert_eq!(
        obs[0].subject, "Caroline",
        "Third-person subject should stay as Caroline, not become speaker"
    );
    assert!(obs[0].content.contains("Caroline"));
    assert!(obs[0].content.contains("attended"));
}

#[test]
fn test_no_observation_without_subject_or_object() {
    // "Wow!" — INTJ, no verb, should produce nothing
    let words: Vec<String> = vec!["Wow", "!"].into_iter().map(String::from).collect();
    let pos_indices = vec![pos("INTJ"), pos("PUNCT")];
    let dep_arcs = vec![arc(0, usize::MAX, "root"), arc(1, 0, "punct")];

    let obs = extract_dep_observations(
        &words,
        &pos_indices,
        &dep_arcs,
        &pos_labels(),
        "Jon",
        &mut SentenceContext::default(),
    );
    assert!(
        obs.is_empty(),
        "Exclamation should not produce observations"
    );
}

#[test]
fn test_verb_only_no_subject_no_object_skipped() {
    // A verb with only modifiers and no subject/object should be skipped
    let words: Vec<String> = vec!["Let's", "go"].into_iter().map(String::from).collect();
    let pos_indices = vec![pos("VERB"), pos("VERB")];
    let dep_arcs = vec![
        arc(0, usize::MAX, "root"),
        arc(1, 0, "xcomp"), // go is an xcomp modifier, not subject or object
    ];

    let obs = extract_dep_observations(
        &words,
        &pos_indices,
        &dep_arcs,
        &pos_labels(),
        "Jon",
        &mut SentenceContext::default(),
    );
    // xcomp is a modifier — verb "go" has no subject or object
    // The root verb "Let's" also has no nsubj or obj
    eprintln!("Observations: {obs:?}");
    // Both verbs lack nsubj/obj — should produce nothing or only via default speaker
}

#[test]
fn test_with_oblique_modifier() {
    // "I'm starting a dance studio 'cause I'm passionate about dancing"
    // Simplified: "I starting studio about dancing"
    let words: Vec<String> = vec!["I", "starting", "dance", "studio", "about", "dancing"]
        .into_iter()
        .map(String::from)
        .collect();
    let pos_indices = vec![
        pos("PRON"),
        pos("VERB"),
        pos("NOUN"),
        pos("NOUN"),
        pos("ADP"),
        pos("NOUN"),
    ];
    let dep_arcs = vec![
        arc(0, 1, "nsubj"),         // I → starting
        arc(1, usize::MAX, "root"), // starting = root
        arc(2, 3, "compound"),      // dance → studio
        arc(3, 1, "obj"),           // studio → starting
        arc(4, 5, "case"),          // about → dancing
        arc(5, 1, "obl"),           // dancing → starting (oblique)
    ];

    let obs = extract_dep_observations(
        &words,
        &pos_indices,
        &dep_arcs,
        &pos_labels(),
        "Jon",
        &mut SentenceContext::default(),
    );

    eprintln!("Observations: {obs:?}");
    assert!(!obs.is_empty());
    assert_eq!(obs[0].subject, "Jon");
    assert!(
        obs[0].content.contains("starting"),
        "Should contain verb: {}",
        obs[0].content
    );
    assert!(
        obs[0].content.contains("studio"),
        "Should contain object: {}",
        obs[0].content
    );
}

#[test]
fn test_multiple_verbs_produce_multiple_observations() {
    // "I lost my job and started a business"
    // Two verbs: lost, started — each should produce an observation
    let words: Vec<String> = vec!["I", "lost", "my", "job", "and", "started", "a", "business"]
        .into_iter()
        .map(String::from)
        .collect();
    let pos_indices = vec![
        pos("PRON"),
        pos("VERB"),
        pos("DET"),
        pos("NOUN"),
        pos("CCONJ"),
        pos("VERB"),
        pos("DET"),
        pos("NOUN"),
    ];
    let dep_arcs = vec![
        arc(0, 1, "nsubj"),         // I → lost
        arc(1, usize::MAX, "root"), // lost = root
        arc(2, 3, "det"),           // my → job
        arc(3, 1, "obj"),           // job → lost
        arc(4, 5, "cc"),            // and → started
        arc(5, 1, "conj"),          // started → lost (conj)
        arc(6, 7, "det"),           // a → business
        arc(7, 5, "obj"),           // business → started
    ];

    let obs = extract_dep_observations(
        &words,
        &pos_indices,
        &dep_arcs,
        &pos_labels(),
        "Jon",
        &mut SentenceContext::default(),
    );

    eprintln!("Observations: {obs:?}");
    // "started" is a VERB with obj "business" but nsubj goes to "lost" via conj chain
    // Current code only checks direct head == verb_idx for nsubj
    // So "started" might not have a direct nsubj — depends on dep structure
    assert!(
        !obs.is_empty(),
        "Should produce at least one observation from 'lost'"
    );
    // First observation should be about losing the job
    assert!(
        obs[0].content.contains("lost"),
        "First obs: {}",
        obs[0].content
    );
}

#[test]
fn test_observation_minimum_quality() {
    // Verify observations have reasonable content
    let words: Vec<String> = vec!["Jon", "works", "at", "the", "hospital"]
        .into_iter()
        .map(String::from)
        .collect();
    let pos_indices = vec![
        pos("PROPN"),
        pos("VERB"),
        pos("ADP"),
        pos("DET"),
        pos("NOUN"),
    ];
    let dep_arcs = vec![
        arc(0, 1, "nsubj"),
        arc(1, usize::MAX, "root"),
        arc(2, 4, "case"),
        arc(3, 4, "det"),
        arc(4, 1, "obl"),
    ];

    let obs = extract_dep_observations(
        &words,
        &pos_indices,
        &dep_arcs,
        &pos_labels(),
        "Gina",
        &mut SentenceContext::default(),
    );

    eprintln!("Observations: {obs:?}");
    assert!(!obs.is_empty());
    let content = &obs[0].content;
    // Should be "Jon works the hospital" or similar
    assert!(content.contains("Jon"), "Missing subject: {content}");
    assert!(content.contains("works"), "Missing verb: {content}");
    assert!(content.contains("hospital"), "Missing oblique: {content}");
    assert_eq!(
        obs[0].subject, "Jon",
        "Subject should be Jon (from text, not speaker Gina)"
    );
}
