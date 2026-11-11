//! A/B parity check: uniko's in-crate NLP decode vs xervo's `NlpModel`.
//!
//! Today uniko owns tokenization + label decoding for NLP: it pulls a
//! *raw* tensor runner from xervo (`raw_tensor_model("nlp/default")`)
//! and decodes NER / POS / DEP / SRL / CLS itself in
//! `uniko_extract::nlp`. xervo 0.13 ships a higher-level `NlpModel`
//! (`runtime.nlp_model(alias).analyze(..)`) that does the decode
//! internally. Before migrating uniko onto it, we need to know whether
//! the two produce the *same* structured output.
//!
//! This binary runs both pipelines over the same single-sentence inputs,
//! backed by the *same* ONNX artifact (so any divergence is decode /
//! label-map, not model weights), and reports per-axis agreement:
//!
//! * **tokenization** — do the word/token sequences line up at all?
//! * **POS** — per-token tag agreement (only where tokens align)
//! * **NER** — entity surface-span overlap (Jaccard), plus a label
//!   correspondence table (the two label schemes differ by design:
//!   uniko collapses OntoNotes into a 10-variant enum)
//! * **DEP** — `(dependent, head, relation)` arc overlap
//! * **CLS** — sentence-class label match (shared 8-act vocabulary)
//! * **SRL** — predicate overlap and per-predicate role overlap
//!
//! It does NOT change the production catalog or call sites: it registers
//! a throwaway `nlp/parity` alias (task = `nlp`) alongside the existing
//! `nlp/default` (task = `raw`) in a scratch KB.
//!
//! Build & run:
//!   cargo build -p uniko-bench --bin nlp-parity --release
//!   ./target/release/nlp-parity                       # built-in corpus
//!   ./target/release/nlp-parity --data lme.json --questions q1,q2

// mimalloc as global allocator — matches the other bench bins; measured
// ~3x throughput on uni-db's concurrent_mutations benchmark.
#[global_allocator]
static GLOBAL: uni_db::MiMalloc = uni_db::MiMalloc;

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use serde::Serialize;

use uni_db::ModelAliasSpec;
use uni_xervo::traits::{NlpRequest, NlpTasks};
use uniko_extract::nlp::NlpPipeline;
use uniko_extract::nlp::assets::label_maps;
use uniko_extract::nlp::types::NlpResult as UnikoNlp;
use uniko_store::KnowledgeBase;
use uniko_store::config::UnikoConfig;

/// Throwaway catalog alias for xervo's `NlpModel` path.
///
/// Distinct from the production `nlp/default` (task = `raw`) so both can
/// coexist in the scratch catalog and read the same model + artifact.
const PARITY_ALIAS: &str = "nlp/parity";

/// Built-in sentence corpus used when no `--data` is supplied.
///
/// Chosen to exercise every head: multi-entity, dates/quantities,
/// questions, greetings, and multi-verb clauses for SRL.
const SAMPLE_SENTENCES: &[&str] = &[
    "Barack Obama was born in Hawaii in 1961.",
    "Apple acquired Beats for three billion dollars in 2014.",
    "Could you send the report to Maria before Friday?",
    "Hey there, it's great to finally meet you!",
    "The cat sat on the mat while the dog watched.",
    "She bought a new laptop because her old one broke last week.",
    "Microsoft and Google compete fiercely in cloud computing.",
    "I think we should meet at the cafe near Central Park tomorrow morning.",
    "The Eiffel Tower attracts seven million visitors every year.",
    "John told Mary that he would finish the project by Monday.",
    "Quantum computers may eventually break modern encryption.",
    "After the storm passed, the volunteers cleaned up the beach.",
    // Passive voice + agent.
    "The proposal was rejected by the committee on Tuesday.",
    "The bridge was designed by a Spanish architect in 1998.",
    // Coordination + relative clauses.
    "The engineer who fixed the server also rewrote the deployment script.",
    "Sarah enjoys hiking, painting, and playing the violin on weekends.",
    "The book that won the prize was translated into twelve languages.",
    // Possessives + numbers.
    "Tesla's revenue grew by 47 percent in the third quarter.",
    "Maria's brother lent her his car for the trip to Denver.",
    "The recipe calls for two cups of flour and a teaspoon of salt.",
    // Money / percent / quantity / ordinal.
    "The startup raised forty million dollars in its second funding round.",
    "Unemployment fell to 3.5 percent last March.",
    "He finished third in the marathon despite an injured ankle.",
    // Multi-entity / orgs / places.
    "Amazon opened a new warehouse near Seattle to serve the Pacific Northwest.",
    "The United Nations met in Geneva to discuss the treaty.",
    "Leonardo da Vinci painted the Mona Lisa in the early sixteenth century.",
    // Negation + modality.
    "I don't believe the meeting was moved to Thursday.",
    "They could not agree on whether to expand the budget.",
    // Questions of several kinds.
    "When did the company first launch its mobile app?",
    "Why was the flight delayed for almost three hours?",
    "Who approved the final version of the contract?",
    // Imperatives / requests / offers.
    "Please forward the invoice to the accounting department.",
    "Let me know if you want me to book the conference room.",
    // Temporal expressions.
    "We launched the beta last spring and shipped the release in October.",
    "The museum is closed on Mondays and public holidays.",
    // Nested clauses / reported speech.
    "The analyst said that earnings would likely beat expectations.",
    "She promised to call her grandmother as soon as she landed.",
    // Comparatives / coordination of verbs.
    "The new model runs faster and uses less memory than the old one.",
    "He neither confirmed nor denied the rumor about the merger.",
    // Locations + movement.
    "The hikers crossed the river and camped beside the old mill.",
    "A package was delivered to the wrong address in Brooklyn.",
    // Mixed.
    "Dr. Chen presented her findings at the conference in Tokyo.",
    "The committee will review the application within two weeks.",
    "Children from the local school planted trees along the highway.",
    "The thunderstorm knocked out power across three counties.",
    "Our team celebrated the victory at a restaurant downtown.",
    "The senator voted against the bill despite pressure from donors.",
];

#[derive(Parser)]
#[command(
    name = "nlp-parity",
    about = "Diff uniko's NLP decode against xervo's NlpModel on identical inputs"
)]
struct Cli {
    /// Optional LongMemEval JSON file to source sentences from.
    ///
    /// When omitted, the built-in corpus is used.
    #[arg(long)]
    data: Option<PathBuf>,

    /// Comma-separated question IDs to pull messages from (with `--data`).
    #[arg(long)]
    questions: Option<String>,

    /// HuggingFace model id for the NLP cascade (both pipelines).
    #[arg(long, default_value = "dragonscale-ai/kniv-deberta-nlp-base-en-xsmall")]
    model_id: String,

    /// ONNX artifact under `model_id/`. Both pipelines load the *same*
    /// file so divergence isolates to decode, not weights.
    #[arg(long, default_value = "onnx/cascade-int8.onnx")]
    artifact: String,

    /// Cap the number of sentences analyzed.
    #[arg(long)]
    limit: Option<usize>,

    /// Scratch directory for the transient KB (deleted + recreated).
    #[arg(long, default_value = "/tmp/nlp_parity_scratch")]
    scratch_dir: PathBuf,

    /// Optional path to dump the full per-sentence comparison as JSON.
    #[arg(long)]
    json: Option<PathBuf>,
}

/// Multiset-overlap counts for one comparison axis.
#[derive(Debug, Default, Clone, Serialize)]
struct Overlap {
    /// Items present on both sides.
    matched: usize,
    /// Items only uniko produced.
    only_uniko: usize,
    /// Items only xervo produced.
    only_xervo: usize,
}

impl Overlap {
    /// Jaccard index over the union; `1.0` when both sides are empty.
    fn jaccard(&self) -> f64 {
        let union = self.matched + self.only_uniko + self.only_xervo;
        if union == 0 {
            1.0
        } else {
            self.matched as f64 / union as f64
        }
    }

    /// Fold another axis result into this running total.
    fn add(&mut self, other: &Overlap) {
        self.matched += other.matched;
        self.only_uniko += other.only_uniko;
        self.only_xervo += other.only_xervo;
    }
}

/// Per-sentence comparison record (also serialized to `--json`).
#[derive(Debug, Serialize)]
struct SentenceReport {
    sentence: String,
    uniko_tokens: Vec<String>,
    /// Raw xervo tokens (metaspace-prefixed, punctuation split out).
    xervo_tokens: Vec<String>,
    /// xervo tokens reconstructed into uniko-style word units (adapter).
    xervo_words: Vec<String>,
    tokens_aligned: bool,
    /// POS tags agreeing / total, when tokens align.
    pos_agree: Option<[usize; 2]>,
    ner: Overlap,
    /// `(uniko_type, xervo_label, surface)` for entity surfaces seen on
    /// both sides — lets a human eyeball the two label schemes.
    ner_label_pairs: Vec<(String, String, String)>,
    /// Full `(label, surface)` entity lists from each side (diagnostic).
    uniko_ner_spans: Vec<(String, String)>,
    xervo_ner_spans: Vec<(String, String)>,
    dep: Overlap,
    /// Full `dep|head|rel` arc-key lists from each side (diagnostic).
    uniko_dep_arcs: Vec<String>,
    xervo_dep_arcs: Vec<String>,
    srl_predicates: Overlap,
    srl_roles: Overlap,
    uniko_cls: String,
    xervo_cls: String,
    cls_match: bool,
}

/// Lowercase, collapse internal whitespace, trim surrounding punctuation.
///
/// Used so that surface comparisons ignore cosmetic tokenizer spacing.
fn normalize_surface(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_matches(|c: char| c.is_ascii_punctuation())
        .to_ascii_lowercase()
}

/// Multiset overlap of two key lists.
fn overlap(a: &[String], b: &[String]) -> Overlap {
    let mut counts: BTreeMap<&str, i64> = BTreeMap::new();
    for k in a {
        *counts.entry(k.as_str()).or_insert(0) += 1;
    }
    for k in b {
        *counts.entry(k.as_str()).or_insert(0) -= 1;
    }
    let mut o = Overlap::default();
    // matched = min(count_a, count_b) per key; the |delta| is the surplus
    // on whichever side has more.
    let mut a_total = a.len();
    let mut b_total = b.len();
    for delta in counts.values() {
        if *delta > 0 {
            o.only_uniko += *delta as usize;
        } else if *delta < 0 {
            o.only_xervo += (-*delta) as usize;
        }
    }
    a_total -= o.only_uniko;
    b_total -= o.only_xervo;
    debug_assert_eq!(a_total, b_total, "matched count must agree both ways");
    o.matched = a_total;
    o
}

/// uniko NER surfaces as normalized `(label, surface)` pairs.
fn uniko_ner(result: &UnikoNlp) -> Vec<(String, String)> {
    result
        .entities
        .iter()
        .map(|e| (format!("{:?}", e.entity_type), normalize_surface(&e.text)))
        .collect()
}

/// xervo NER surfaces, grouping contiguous same-typed tokens into spans.
///
/// `NlpToken::ner` is the bare entity type (`None` outside a span), so a
/// run of equal non-`None` labels is one entity. Surface text is sliced
/// from the original sentence by byte offsets for fidelity.
fn xervo_ner(result: &uni_xervo::traits::NlpResult, sentence: &str) -> Vec<(String, String)> {
    // xervo exposes RAW per-token BIO tags (`B-PERSON`, `I-PERSON`, `O`, …),
    // not merged spans, so merge them the way uniko's `merge_bio_spans`
    // does: a `B-` (or orphan `I-`) opens a span, a matching `I-` extends
    // it, and `O`/empty closes it. Surface sliced by byte offsets.
    let mut spans: Vec<(String, String)> = Vec::new();
    let mut run: Option<(String, usize, usize)> = None; // (type, start, end)
    for tok in &result.tokens {
        let tag = tok.ner.as_deref().unwrap_or("O");
        if tag == "O" || tag.is_empty() {
            if let Some((cur, s, e)) = run.take() {
                spans.push((cur, normalize_surface(&sentence[s..e])));
            }
            continue;
        }
        let (prefix, ty) = tag.split_once('-').unwrap_or(("B", tag));
        match &mut run {
            Some((cur, _s, end)) if prefix == "I" && *cur == ty => {
                *end = tok.end;
            }
            _ => {
                if let Some((cur, s, e)) = run.take() {
                    spans.push((cur, normalize_surface(&sentence[s..e])));
                }
                run = Some((ty.to_string(), tok.start, tok.end));
            }
        }
    }
    if let Some((cur, s, e)) = run.take() {
        spans.push((cur, normalize_surface(&sentence[s..e])));
    }
    spans
}

/// uniko dependency arcs as `dep|head|rel` keys (head `usize::MAX` → ROOT).
fn uniko_dep(result: &UnikoNlp) -> Vec<String> {
    result
        .dep_arcs
        .iter()
        .map(|a| {
            let dep = result.words.get(a.dependent).map_or("?", String::as_str);
            let head = if a.head == usize::MAX {
                "ROOT"
            } else {
                result.words.get(a.head).map_or("?", String::as_str)
            };
            format!(
                "{}|{}|{}",
                normalize_surface(dep),
                normalize_surface(head),
                a.relation.to_ascii_lowercase()
            )
        })
        .collect()
}

/// PROTOTYPE ADAPTER: reconstruct uniko-style word units from xervo tokens.
///
/// xervo surfaces the SentencePiece metaspace, so a token beginning with a
/// space (or `▁`) starts a new word and a token without one continues the
/// previous. Grouping on that boundary mirrors uniko's `token_to_word`
/// alignment: it strips the leading space and folds trailing punctuation
/// (`'1961'` + `'.'` → `'1961.'`) back onto the preceding word, so the two
/// pipelines compare on identical word boundaries.
///
/// Returns the word surfaces and a map `token_index -> word_index`.
fn reconstruct_xervo_words(result: &uni_xervo::traits::NlpResult) -> (Vec<String>, Vec<usize>) {
    let mut words: Vec<String> = Vec::new();
    let mut map: Vec<usize> = Vec::with_capacity(result.tokens.len());
    for tok in &result.tokens {
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

/// xervo dependency arcs lifted onto reconstructed uniko word units.
///
/// One arc per word, taken from the word's first token (first-subword-wins,
/// matching uniko). A head that lands in the same word is treated as ROOT.
fn xervo_dep_words(
    result: &uni_xervo::traits::NlpResult,
    words: &[String],
    tok2word: &[usize],
) -> Vec<String> {
    let surf = |w: usize| {
        words
            .get(w)
            .map_or_else(|| "?".to_string(), |s| normalize_surface(s))
    };
    let mut seen = vec![false; words.len()];
    let mut out = Vec::new();
    for (i, tok) in result.tokens.iter().enumerate() {
        let w = tok2word[i];
        if seen[w] {
            continue;
        }
        seen[w] = true;
        if let Some(d) = &tok.dep {
            // xervo heads are 0-based global token indices, `None` = root
            // (the token attaches to the sentence root). Map the head
            // token index to its word.
            let head = match d.head.and_then(|ht| tok2word.get(ht).copied()) {
                Some(hw) if hw != w => surf(hw),
                _ => "root".to_string(),
            };
            out.push(format!(
                "{}|{}|{}",
                surf(w),
                head,
                d.relation.to_ascii_lowercase()
            ));
        }
    }
    out
}

/// uniko SRL predicates and `pred::role::arg` role keys.
fn uniko_srl(result: &UnikoNlp) -> (Vec<String>, Vec<String>) {
    let mut preds = Vec::new();
    let mut roles = Vec::new();
    for f in &result.srl_frames {
        let pred = normalize_surface(&f.predicate_word);
        preds.push(pred.clone());
        for a in &f.args {
            roles.push(format!(
                "{}::{}::{}",
                pred,
                a.role.to_ascii_uppercase(),
                normalize_surface(&a.text)
            ));
        }
    }
    (preds, roles)
}

/// xervo SRL predicates and `pred::role::arg` role keys (spans → surface).
fn xervo_srl(result: &uni_xervo::traits::NlpResult) -> (Vec<String>, Vec<String>) {
    let surf = |i: usize| {
        result
            .tokens
            .get(i)
            .map_or(String::new(), |t| t.text.clone())
    };
    let mut preds = Vec::new();
    let mut roles = Vec::new();
    for f in &result.frames {
        let pred = normalize_surface(&surf(f.predicate_token));
        preds.push(pred.clone());
        for r in &f.roles {
            let (s, e) = r.span;
            let text: Vec<String> = (s..=e).map(surf).collect();
            roles.push(format!(
                "{}::{}::{}",
                pred,
                r.label.to_ascii_uppercase(),
                normalize_surface(&text.join(" "))
            ));
        }
    }
    (preds, roles)
}

/// Compare both pipelines on one sentence into a [`SentenceReport`].
fn compare(
    sentence: &str,
    uniko: &UnikoNlp,
    xervo: &uni_xervo::traits::NlpResult,
    labels: &uniko_extract::nlp::assets::LabelMaps,
) -> SentenceReport {
    let uniko_tokens: Vec<String> = uniko.words.clone();
    let xervo_tokens: Vec<String> = xervo.tokens.iter().map(|t| t.text.clone()).collect();
    // Adapter: reconcile xervo's segmentation to uniko's word boundaries
    // before any index-based comparison (POS/DEP), since the two only
    // differ in metaspace + punctuation grouping, not in subwords.
    let (xervo_words, tok2word) = reconstruct_xervo_words(xervo);
    let tokens_aligned = uniko_tokens.len() == xervo_words.len();

    // POS only meaningful when word sequences align 1:1. The first xervo
    // token of each word carries the word's tag (first-subword-wins).
    let pos_agree = if tokens_aligned {
        let mut first_tok: Vec<Option<usize>> = vec![None; xervo_words.len()];
        for (i, w) in tok2word.iter().enumerate() {
            first_tok[*w].get_or_insert(i);
        }
        let mut agree = 0usize;
        for (w, slot) in first_tok.iter().enumerate() {
            let u = uniko
                .pos_indices
                .get(w)
                .and_then(|&idx| labels.pos_labels.get(idx))
                .map(String::as_str);
            let x = slot.and_then(|i| xervo.tokens[i].pos.as_deref());
            if let (Some(u), Some(x)) = (u, x)
                && u.eq_ignore_ascii_case(x)
            {
                agree += 1;
            }
        }
        Some([agree, xervo_words.len()])
    } else {
        None
    };

    // NER: overlap on surface, plus label-correspondence on shared surfaces.
    let u_ner = uniko_ner(uniko);
    let x_ner = xervo_ner(xervo, sentence);
    let u_ner_surf: Vec<String> = u_ner.iter().map(|(_, s)| s.clone()).collect();
    let x_ner_surf: Vec<String> = x_ner.iter().map(|(_, s)| s.clone()).collect();
    let ner = overlap(&u_ner_surf, &x_ner_surf);
    let x_by_surf: BTreeMap<&str, &str> = x_ner
        .iter()
        .map(|(l, s)| (s.as_str(), l.as_str()))
        .collect();
    let ner_label_pairs: Vec<(String, String, String)> = u_ner
        .iter()
        .filter_map(|(ul, s)| {
            x_by_surf
                .get(s.as_str())
                .map(|xl| (ul.clone(), (*xl).to_string(), s.clone()))
        })
        .collect();

    let u_dep = uniko_dep(uniko);
    let x_dep = xervo_dep_words(xervo, &xervo_words, &tok2word);
    let dep = overlap(&u_dep, &x_dep);

    let (u_preds, u_roles) = uniko_srl(uniko);
    let (x_preds, x_roles) = xervo_srl(xervo);
    let srl_predicates = overlap(&u_preds, &x_preds);
    let srl_roles = overlap(&u_roles, &x_roles);

    let uniko_cls = labels
        .cls_labels
        .get(uniko.cls_index)
        .cloned()
        .unwrap_or_else(|| "?".to_string());
    let xervo_cls = xervo
        .speech_acts
        .first()
        .map_or_else(|| "?".to_string(), |a| a.label.clone());
    let cls_match = uniko_cls.eq_ignore_ascii_case(&xervo_cls);

    SentenceReport {
        sentence: sentence.to_string(),
        uniko_tokens,
        xervo_tokens,
        xervo_words,
        tokens_aligned,
        pos_agree,
        ner,
        ner_label_pairs,
        uniko_ner_spans: u_ner,
        xervo_ner_spans: x_ner,
        dep,
        uniko_dep_arcs: u_dep,
        xervo_dep_arcs: x_dep,
        srl_predicates,
        srl_roles,
        uniko_cls,
        xervo_cls,
        cls_match,
    }
}

/// Collect the sentences to analyze from `--data` or the built-in corpus.
fn collect_sentences(cli: &Cli) -> Result<Vec<String>> {
    let mut sentences: Vec<String> = match &cli.data {
        Some(path) => {
            let ids = cli
                .questions
                .as_deref()
                .context("--questions is required when --data is given")?;
            let want: Vec<String> = ids.split(',').map(|s| s.trim().to_string()).collect();
            let mut items = uniko_bench::longmemeval::data::load_longmemeval(path)?;
            items.retain(|i| want.contains(&i.question_id));
            if items.is_empty() {
                anyhow::bail!("no questions matched ids {want:?}");
            }
            let mut out = Vec::new();
            for item in &items {
                for session in &item.haystack_sessions {
                    for turn in session {
                        out.extend(split_sentences(&turn.content));
                    }
                }
            }
            out
        }
        None => SAMPLE_SENTENCES.iter().map(|s| (*s).to_string()).collect(),
    };
    if let Some(cap) = cli.limit {
        sentences.truncate(cap);
    }
    Ok(sentences)
}

/// Minimal sentence splitter shared by both pipelines for 1:1 units.
///
/// Splits on `.`/`!`/`?` and drops fragments under three words so the
/// comparison isn't dominated by trivial inputs.
fn split_sentences(text: &str) -> Vec<String> {
    text.split(['.', '!', '?'])
        .map(str::trim)
        .filter(|s| s.split_whitespace().count() >= 3)
        .map(|s| format!("{s}."))
        .collect()
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                "nlp_parity=info,uniko_extract=warn,uniko_store=warn,ort::ep=warn"
                    .parse()
                    .expect("static log filter parses")
            }),
        )
        .init();

    let cli = Cli::parse();
    let sentences = collect_sentences(&cli)?;
    tracing::info!(count = sentences.len(), "sentences to compare");

    // Fresh scratch KB. A full catalog file *replaces* the auto-built
    // one, so we list every alias we need: the raw `nlp/default` for
    // uniko's NlpPipeline and the `nlp/parity` (task=nlp) for xervo.
    if cli.scratch_dir.exists() {
        std::fs::remove_dir_all(&cli.scratch_dir).context("remove scratch")?;
    }
    std::fs::create_dir_all(&cli.scratch_dir).context("create scratch")?;
    let catalog_file = cli.scratch_dir.join("catalog.json");
    let catalog_json = serde_json::json!([
        {
            "alias": "embed/default",
            "task": "embed",
            "provider_id": "local/onnx",
            "model_id": "BGESmallENV15",
            "warmup": "lazy",
            "required": false
        },
        {
            "alias": "nlp/default",
            "task": "raw",
            "provider_id": "local/onnx",
            "model_id": cli.model_id,
            "warmup": "lazy",
            "required": false,
            "options": { "artifact": cli.artifact, "max_batch_size": 16 }
        },
        {
            "alias": PARITY_ALIAS,
            "task": "nlp",
            "provider_id": "local/onnx",
            "model_id": cli.model_id,
            "warmup": "lazy",
            "required": false,
            // OnnxNlpModel reads onnx_path (not `artifact`); point it at
            // the same file the raw alias uses for identical weights.
            // `generation` is pinned for the reason in embed_catalog.
            "options": {
                "generation": "v1",
                "onnx_path": cli.artifact,
                "tokenizer_path": "tokenizer.json",
                "label_maps_path": "label_maps.json",
                "max_seq_len": 128
            }
        }
    ]);
    std::fs::write(&catalog_file, serde_json::to_string_pretty(&catalog_json)?)
        .context("write catalog")?;

    let mut config = UnikoConfig::default();
    config.embedding = uniko_store::config::EmbeddingConfig::bge_small_en_v15();
    config.catalog_path = Some(catalog_file);
    config.nlp_srl_enabled = true; // ensure uniko populates SRL frames

    let kb = KnowledgeBase::open_with_xervo(&cli.scratch_dir, config, Vec::<ModelAliasSpec>::new())
        .await
        .context("open scratch KB")?;

    let pipeline = NlpPipeline::try_new(&kb)
        .await
        .context("NlpPipeline::try_new returned None — check catalog and ort/cuda setup")?;

    let runtime = kb
        .db()
        .xervo()
        .raw_runtime()
        .cloned()
        .context("xervo runtime not configured on the KB")?;
    let nlp = runtime
        .nlp_model(PARITY_ALIAS)
        .await
        .map_err(|e| anyhow::anyhow!("resolve {PARITY_ALIAS}: {e}"))?;

    let labels = label_maps();

    let mut reports = Vec::with_capacity(sentences.len());
    for sentence in &sentences {
        let uniko = pipeline
            .analyze(sentence)
            .await
            .with_context(|| format!("uniko analyze: {sentence}"))?;
        let mut xervo = nlp
            .analyze(vec![NlpRequest::new(sentence, NlpTasks::ALL)])
            .await
            .map_err(|e| anyhow::anyhow!("xervo analyze '{sentence}': {e}"))?;
        let xervo = xervo
            .pop()
            .context("xervo returned zero results for one request")?;
        reports.push(compare(sentence, &uniko, &xervo, labels));
    }

    print_summary(&reports);

    if let Some(path) = &cli.json {
        std::fs::write(path, serde_json::to_string_pretty(&reports)?)
            .with_context(|| format!("write json report: {}", path.display()))?;
        tracing::info!(path = %path.display(), "wrote full per-sentence report");
    }

    Ok(())
}

/// Print the aggregate + worst-divergence summary to stdout.
fn print_summary(reports: &[SentenceReport]) {
    let n = reports.len();
    let tokens_aligned = reports.iter().filter(|r| r.tokens_aligned).count();
    let cls_match = reports.iter().filter(|r| r.cls_match).count();

    let (mut pos_agree, mut pos_total) = (0usize, 0usize);
    let (mut ner, mut dep, mut preds, mut roles) = (
        Overlap::default(),
        Overlap::default(),
        Overlap::default(),
        Overlap::default(),
    );
    for r in reports {
        if let Some([a, t]) = r.pos_agree {
            pos_agree += a;
            pos_total += t;
        }
        ner.add(&r.ner);
        dep.add(&r.dep);
        preds.add(&r.srl_predicates);
        roles.add(&r.srl_roles);
    }

    let pct = |a: usize, b: usize| {
        if b == 0 {
            100.0
        } else {
            100.0 * a as f64 / b as f64
        }
    };

    println!("\n══════════════ NLP PARITY: uniko decode vs xervo NlpModel ══════════════");
    println!("sentences compared      : {n}");
    println!(
        "token sequences aligned : {tokens_aligned}/{n} ({:.1}%)",
        pct(tokens_aligned, n)
    );
    println!(
        "POS tag agreement       : {pos_agree}/{pos_total} ({:.1}%)  [aligned sentences only]",
        pct(pos_agree, pos_total)
    );
    println!(
        "NER surface overlap     : Jaccard {:.3}  (matched {}, only-uniko {}, only-xervo {})",
        ner.jaccard(),
        ner.matched,
        ner.only_uniko,
        ner.only_xervo
    );
    println!(
        "DEP arc overlap         : Jaccard {:.3}  (matched {}, only-uniko {}, only-xervo {})",
        dep.jaccard(),
        dep.matched,
        dep.only_uniko,
        dep.only_xervo
    );
    println!(
        "SRL predicate overlap   : Jaccard {:.3}  (matched {}, only-uniko {}, only-xervo {})",
        preds.jaccard(),
        preds.matched,
        preds.only_uniko,
        preds.only_xervo
    );
    println!(
        "SRL role overlap        : Jaccard {:.3}  (matched {}, only-uniko {}, only-xervo {})",
        roles.jaccard(),
        roles.matched,
        roles.only_uniko,
        roles.only_xervo
    );
    println!(
        "CLS label match         : {cls_match}/{n} ({:.1}%)",
        pct(cls_match, n)
    );

    // Surface the NER label-scheme correspondence (uniko enum ↔ xervo str).
    let mut pair_counts: BTreeMap<(String, String), usize> = BTreeMap::new();
    for r in reports {
        for (u, x, _surf) in &r.ner_label_pairs {
            *pair_counts.entry((u.clone(), x.clone())).or_insert(0) += 1;
        }
    }
    if !pair_counts.is_empty() {
        println!(
            "\nNER label correspondence (uniko enum → xervo label : count on shared surfaces):"
        );
        for ((u, x), c) in &pair_counts {
            println!("  {u:<14} → {x:<10} : {c}");
        }
    }

    // Show the sentences that diverged most, to make root-causing easy.
    let mut worst: Vec<&SentenceReport> = reports.iter().collect();
    worst.sort_by(|a, b| {
        let score = |r: &SentenceReport| {
            r.ner.only_uniko + r.ner.only_xervo + r.dep.only_uniko + r.dep.only_xervo
        };
        score(b).cmp(&score(a))
    });
    println!("\nMost-divergent sentences:");
    for r in worst.iter().take(5) {
        println!("  • {}", r.sentence);
        if !r.tokens_aligned {
            println!(
                "      token mismatch: uniko {} vs xervo {}",
                r.uniko_tokens.len(),
                r.xervo_tokens.len()
            );
        }
        println!(
            "      NER only-uniko {}, only-xervo {} | DEP only-uniko {}, only-xervo {} | CLS {} vs {}",
            r.ner.only_uniko,
            r.ner.only_xervo,
            r.dep.only_uniko,
            r.dep.only_xervo,
            r.uniko_cls,
            r.xervo_cls
        );
    }
    println!("════════════════════════════════════════════════════════════════════════\n");
}
