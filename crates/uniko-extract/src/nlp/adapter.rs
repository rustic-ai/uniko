//! Adapter: xervo `NlpModel` output → uniko's [`NlpResult`].
//!
//! xervo decodes POS/NER/DEP/SRL/CLS internally and returns a
//! token-oriented [`uni_xervo::traits::NlpResult`]. uniko's downstream
//! (`ner`, `observations`) consumes a word-oriented
//! [`crate::nlp::types::NlpResult`]. This module reconciles the three
//! representation differences the `nlp-parity` bench proved to be the
//! *only* divergences between the two decoders on identical weights:
//!
//! 1. **Tokenization** — xervo surfaces the SentencePiece metaspace: a
//!    token starting with a space (or `▁`) opens a new word, otherwise it
//!    continues the previous one. Grouping on that boundary recovers
//!    uniko's `token_to_word` word units (and folds trailing punctuation
//!    such as `.` back onto the number it follows).
//! 2. **Dependency heads** — xervo's `DepLink.head` (0.14+) is an
//!    `Option<usize>` 0-based *global* token index, `None` = sentence
//!    root; we map the head token to its word, treating `None`/self as
//!    root (the word-level [`DepArc::head`] sentinel is `usize::MAX`).
//! 3. **NER** — xervo returns raw per-token BIO tags (`B-PERSON`, …), so
//!    we BIO-merge them into spans via [`super::decode::build_span`] /
//!    [`super::decode::parse_ner_type`], exactly as the in-crate decoder
//!    did, preserving the same [`NerEntityType`](super::types::NerEntityType)
//!    mapping.
//!
//! kniv v7 differs from v1 in two of these, which [`Layout`] tells the
//! adapter about:
//!
//! - its tokens are already whole words, with no metaspace, so they map
//!   one to one instead of being regrouped (regrouping would glue every
//!   word of a message into one);
//! - its six dialogue acts are not v1's eight, so they are projected onto
//!   the v1 labels (see [`V7_ACTS`]) that the observation gate and
//!   [`SentenceClass::from_label`] are written against.

use uni_xervo::traits::NlpResult as XervoNlp;

use super::assets::LabelMaps;
use super::decode::{build_span, parse_ner_type};
use super::types::{DepArc, NerSpan, NlpResult, SentenceClass, SrlArg, SrlFrame};

/// How a model lays out its output, fixed per model.
#[derive(Debug, Clone, Copy, Default)]
pub struct Layout<'a> {
    /// Each token is one whole word (kniv v7), rather than a SentencePiece
    /// subword to regroup (kniv v1).
    pub word_tokens: bool,
    /// The model's own dialogue-act vocabulary, parallel to
    /// `SpeechAct::scores`, when it is not uniko's v1 label set (kniv v7).
    /// `None` reads acts as v1 labels.
    pub acts: Option<&'a [String]>,
}

/// kniv v7's dialogue acts and the v1 label each one stands in for.
///
/// v7 follows the ISO 24617-2 families; v1's labels split some of them
/// finer. `Directive` (asking someone to act) is v1's `request`, and
/// `Commissive` (committing oneself to act) is v1's `offer`, so both stay
/// informative as before. `Feedback` (acknowledging or answering) maps to
/// `confirm`. No v7 act maps to `status` or `reject`.
pub const V7_ACTS: &[(&str, &str)] = &[
    ("Inform", "inform"),
    ("Question", "question"),
    ("Directive", "request"),
    ("Commissive", "offer"),
    ("Feedback", "confirm"),
    ("Social", "social"),
];

/// The v1 label a kniv v7 dialogue act stands in for.
fn v1_act(v7: &str) -> Option<&'static str> {
    V7_ACTS
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(v7))
        .map(|(_, v1)| *v1)
}

/// Convert one xervo sentence result into uniko's [`NlpResult`].
///
/// `srl_enabled` mirrors `UnikoConfig.nlp_srl_enabled`; when `false`,
/// `srl_frames` is left empty regardless of what xervo returned.
pub fn xervo_to_uniko(
    x: &XervoNlp,
    labels: &LabelMaps,
    srl_enabled: bool,
    layout: Layout<'_>,
) -> NlpResult {
    let (words, tok2word) = if layout.word_tokens {
        word_tokens(x)
    } else {
        reconstruct_words(x)
    };
    let n = words.len();

    // First token index of each word (first-subword-wins, matching uniko).
    let mut first_tok: Vec<Option<usize>> = vec![None; n];
    for (i, &w) in tok2word.iter().enumerate() {
        first_tok[w].get_or_insert(i);
    }

    // POS: map each word's first-token tag string to its label index.
    let pos_unknown = labels.pos_labels.iter().position(|l| l == "X").unwrap_or(0);
    let pos_indices: Vec<usize> = (0..n)
        .map(|w| {
            first_tok[w]
                .and_then(|i| x.tokens[i].pos.as_deref())
                .and_then(|p| {
                    labels
                        .pos_labels
                        .iter()
                        .position(|l| l.eq_ignore_ascii_case(p))
                })
                .unwrap_or(pos_unknown)
        })
        .collect();

    // NER: per-word tag (first token), then label index + BIO-merge.
    let word_ner: Vec<&str> = (0..n)
        .map(|w| {
            first_tok[w]
                .and_then(|i| x.tokens[i].ner.as_deref())
                .unwrap_or("O")
        })
        .collect();
    let ner_indices: Vec<usize> = word_ner
        .iter()
        .map(|tag| {
            labels
                .ner_labels
                .iter()
                .position(|l| l.eq_ignore_ascii_case(tag))
                .unwrap_or(0)
        })
        .collect();
    let entities = merge_word_bio(&words, &word_ner);

    // DEP: one word-level arc per word, from its first token's link.
    let mut dep_arcs = Vec::new();
    for (w, slot) in first_tok.iter().enumerate() {
        let Some(i) = *slot else { continue };
        let Some(d) = x.tokens[i].dep.as_ref() else {
            continue;
        };
        // xervo head: 0-based global token index, `None` = sentence root.
        // Map the head token back to a word; a head landing in the same
        // word (or `None`) is the sentence root.
        let head = d
            .head
            .and_then(|ht| tok2word.get(ht).copied())
            .filter(|&hw| hw != w)
            .unwrap_or(usize::MAX);
        dep_arcs.push(DepArc {
            dependent: w,
            head,
            relation: d.relation.clone(),
        });
    }

    // SRL: token spans → word spans (xervo span is inclusive `[first,
    // last]`; uniko `end_word` is exclusive).
    let srl_frames = if srl_enabled {
        x.frames
            .iter()
            .map(|f| {
                let predicate_idx = tok2word.get(f.predicate_token).copied().unwrap_or(0);
                let args = f
                    .roles
                    .iter()
                    .filter_map(|r| {
                        let start = tok2word.get(r.span.0).copied()?;
                        let last = tok2word.get(r.span.1).copied()?;
                        let end = (last + 1).clamp(start + 1, n);
                        Some(SrlArg {
                            role: r.label.clone(),
                            text: words.get(start..end)?.join(" "),
                            start_word: start,
                            end_word: end,
                        })
                    })
                    .collect();
                SrlFrame {
                    predicate_idx,
                    predicate_word: words.get(predicate_idx).cloned().unwrap_or_default(),
                    args,
                }
            })
            .collect()
    } else {
        Vec::new()
    };

    let (cls_index, cls_confidence, cls_probs, sentence_class) = match layout.acts {
        Some(acts) => project_acts(x, acts, labels),
        None => v1_acts(x, labels),
    };

    NlpResult {
        words,
        ner_indices,
        pos_indices,
        cls_index,
        cls_confidence,
        cls_probs,
        entities,
        dep_arcs,
        sentence_class,
        srl_frames,
    }
}

/// CLS for a v1 model: the top speech act → class + index.
///
/// `cls_probs` stays empty, so the observation gate uses the top class and
/// its confidence (see `observations::cls_gate_admits`).
fn v1_acts(x: &XervoNlp, labels: &LabelMaps) -> (usize, f32, Vec<f32>, SentenceClass) {
    match x.speech_acts.first() {
        Some(act) => (
            labels
                .cls_labels
                .iter()
                .position(|l| l.eq_ignore_ascii_case(&act.label))
                .unwrap_or(0),
            act.confidence,
            Vec::new(),
            SentenceClass::from_label(&act.label),
        ),
        None => (0, 0.0, Vec::new(), SentenceClass::Statement),
    }
}

/// CLS for a model with its own act vocabulary (kniv v7): its per-act
/// scores, re-indexed onto the v1 labels through [`V7_ACTS`].
///
/// v7 scores each act with its own sigmoid, so a sentence can be both
/// `Inform` and `Social`. Filling `cls_probs` lets the observation gate
/// admit it on any informative act that clears its threshold, instead of
/// on the top one alone. Where two acts would land on one v1 label the
/// higher score wins.
fn project_acts(
    x: &XervoNlp,
    acts: &[String],
    labels: &LabelMaps,
) -> (usize, f32, Vec<f32>, SentenceClass) {
    let Some(act) = x.speech_acts.first() else {
        return (0, 0.0, Vec::new(), SentenceClass::Statement);
    };
    let index_of = |v1: &str| labels.cls_labels.iter().position(|l| l == v1);
    let mut probs = vec![0.0_f32; labels.cls_labels.len()];
    for (name, &score) in acts.iter().zip(&act.scores) {
        if let Some(i) = v1_act(name).and_then(index_of) {
            probs[i] = probs[i].max(score);
        }
    }
    // The top act by score; `act.label` can be "NONE" when no act clears
    // the model's own threshold.
    let top = act
        .scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .and_then(|(i, _)| acts.get(i))
        .map_or(act.label.as_str(), String::as_str);
    let v1 = v1_act(top).unwrap_or("inform");
    (
        index_of(v1).unwrap_or(0),
        act.confidence,
        probs,
        SentenceClass::from_label(v1),
    )
}

/// Word units for a model whose tokens are already words (kniv v7).
fn word_tokens(x: &XervoNlp) -> (Vec<String>, Vec<usize>) {
    let words = x.tokens.iter().map(|t| t.text.clone()).collect();
    (words, (0..x.tokens.len()).collect())
}

/// Reconstruct uniko word units from xervo tokens by SentencePiece
/// metaspace boundaries. Returns the word surfaces and a `token → word`
/// index map.
fn reconstruct_words(x: &XervoNlp) -> (Vec<String>, Vec<usize>) {
    let mut words: Vec<String> = Vec::new();
    let mut map: Vec<usize> = Vec::with_capacity(x.tokens.len());
    for tok in &x.tokens {
        let raw = tok.text.as_str();
        let starts_word = words.is_empty() || raw.starts_with([' ', '\u{2581}']);
        let piece = raw.trim_start_matches([' ', '\u{2581}']);
        if starts_word {
            words.push(piece.to_string());
        } else if let Some(last) = words.last_mut() {
            last.push_str(piece);
        } else {
            words.push(piece.to_string());
        }
        map.push(words.len() - 1);
    }
    (words, map)
}

/// BIO-merge per-word NER tags into [`NerSpan`]s (`end_word` exclusive).
///
/// `B-X` (or an orphan `I-X`) opens a span; a matching `I-X` extends it;
/// `O`/empty closes it. Mirrors `decode::merge_bio_spans`.
fn merge_word_bio(words: &[String], word_ner: &[&str]) -> Vec<NerSpan> {
    let mut spans = Vec::new();
    let mut run: Option<(String, usize)> = None; // (type suffix, start word)
    for (w, tag) in word_ner.iter().enumerate() {
        if *tag == "O" || tag.is_empty() {
            if let Some((suf, start)) = run.take() {
                spans.push(build_span(words, start, w, parse_ner_type(&suf)));
            }
            continue;
        }
        let (prefix, suffix) = tag.split_once('-').unwrap_or(("B", tag));
        let continues = matches!(&run, Some((cur, _)) if prefix == "I" && cur == suffix);
        if !continues {
            if let Some((suf, start)) = run.take() {
                spans.push(build_span(words, start, w, parse_ner_type(&suf)));
            }
            run = Some((suffix.to_string(), w));
        }
    }
    if let Some((suf, start)) = run.take() {
        spans.push(build_span(words, start, words.len(), parse_ner_type(&suf)));
    }
    spans
}
