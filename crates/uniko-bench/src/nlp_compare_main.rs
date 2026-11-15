//! NLP model comparison: quality against gold labels, and speed.
//!
//! Runs one [`NlpConfig`] preset (`v1`, `v7-small`, …) through
//! [`NlpPipeline`] — the same path ingest takes — over the sentences
//! `scripts/nlp_eval_prep.py` samples, and scores what uniko consumes:
//!
//! * **NER**: span F1 on uniko's entity types, matched by character span
//!   with punctuation trimmed from both ends (v1 words keep theirs).
//! * **POS**: accuracy on the words whose character span matches a gold
//!   token exactly; `aligned_share` says how many did. Dependencies are not
//!   scored: the corpus heads follow spaCy's scheme (copula as head, `pobj`)
//!   and both generations emit Universal Dependencies.
//! * **Act**: the [`SentenceClass`] against the gold dialogue act, and the
//!   observation gate (would this sentence produce observations?) against
//!   whether the gold act carries content.
//! * **Message NER**: the same span F1 on four consecutive sentences joined
//!   into one message and run through `analyze_sentences`, as ingest does.
//!   Unlike the per-sentence score, this one sees what sentence splitting
//!   costs: a name cut at "F." or a fragment dropped for being short.
//! * **Speed**: load time, per-sentence `analyze`, and per-message
//!   `analyze_sentences` on the kniv prose messages.
//!
//! One preset per process, so a driver can read each one's peak memory.
//! Results are appended to `--output` as one JSON line.
//!
//!   uv run --no-project --with pyarrow python -I scripts/nlp_eval_prep.py eval.jsonl
//!   cargo run -p uniko-bench --release --bin nlp-compare -- \
//!       --data eval.jsonl --preset v7-small --output results.jsonl

#[global_allocator]
static GLOBAL: uni_db::MiMalloc = uni_db::MiMalloc;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde::{Deserialize, Serialize};

use uniko_extract::nlp::NlpPipeline;
use uniko_extract::nlp::assets::label_maps;
use uniko_extract::nlp::types::{NerEntityType, NlpResult, SentenceClass};
use uniko_extract::observations::rules_engine::rules::ClsGate;
use uniko_store::KnowledgeBase;
use uniko_store::config::{NlpConfig, UnikoConfig};

#[derive(Parser)]
#[command(
    name = "nlp-compare",
    about = "Score an NLP preset against gold labels"
)]
struct Cli {
    /// JSONL written by `scripts/nlp_eval_prep.py`.
    #[arg(long)]
    data: PathBuf,
    /// An `NlpConfig::PRESETS` name.
    #[arg(long)]
    preset: String,
    /// ONNX precision for v7 (`int8`, `fp16`, `fp32`).
    #[arg(long)]
    variant: Option<String>,
    /// JSONL file the result line is appended to.
    #[arg(long)]
    output: PathBuf,
    /// Print gold and predicted analyses for the first N sentences.
    #[arg(long, default_value = "0")]
    dump: usize,
}

#[derive(Deserialize)]
struct Gold {
    source: String,
    domain: String,
    text: String,
    tokens: Vec<String>,
    #[serde(default)]
    pos: Vec<String>,
    entities: Vec<GoldSpan>,
    #[serde(default)]
    act: Option<String>,
}

#[derive(Deserialize)]
struct GoldSpan {
    start: usize,
    end: usize,
    #[serde(rename = "type")]
    kind: String,
}

/// Character spans of `words` in `text`, found left to right. A word that
/// does not appear within a short window of the cursor gets `None`.
fn offsets<S: AsRef<str>>(text: &str, words: &[S]) -> Vec<Option<(usize, usize)>> {
    let mut cursor = 0;
    words
        .iter()
        .map(|w| {
            let w = w.as_ref().trim();
            if w.is_empty() {
                return None;
            }
            let hay = &text[cursor..];
            let at = hay.find(w)?;
            if hay[..at].chars().count() > 8 {
                return None;
            }
            let start = cursor + at;
            cursor = start + w.len();
            Some((start, cursor))
        })
        .collect()
}

/// Narrow `text[a..b]` past punctuation at either end, so `"goods,"` and
/// `"goods"` name the same span.
fn trim_punct(text: &str, a: usize, b: usize) -> Option<(usize, usize)> {
    let span = text.get(a..b)?;
    let is_p = |c: char| !c.is_alphanumeric() && c != '$' && c != '%';
    let lead = span.len() - span.trim_start_matches(is_p).len();
    let rest = span.trim_matches(is_p);
    (!rest.is_empty()).then(|| (a + lead, a + lead + rest.len()))
}

#[derive(Default, Serialize, Clone, Copy)]
struct Prf {
    tp: u64,
    fp: u64,
    fn_: u64,
}

impl Prf {
    fn add(
        &mut self,
        gold: &HashSet<(usize, usize, NerEntityType)>,
        pred: &HashSet<(usize, usize, NerEntityType)>,
    ) {
        let hit = gold.intersection(pred).count() as u64;
        self.tp += hit;
        self.fp += pred.len() as u64 - hit;
        self.fn_ += gold.len() as u64 - hit;
    }
    fn f1(self) -> f64 {
        let d = 2 * self.tp + self.fp + self.fn_;
        if d == 0 {
            0.0
        } else {
            2.0 * self.tp as f64 / d as f64
        }
    }
    fn precision(self) -> f64 {
        ratio(self.tp, self.tp + self.fp)
    }
    fn recall(self) -> f64 {
        ratio(self.tp, self.tp + self.fn_)
    }
}

fn ratio(a: u64, b: u64) -> f64 {
    if b == 0 { 0.0 } else { a as f64 / b as f64 }
}

fn gold_type(kind: &str) -> Option<NerEntityType> {
    Some(match kind {
        "PERSON" => NerEntityType::Person,
        "ORG" => NerEntityType::Organization,
        "GPE" | "LOC" | "FAC" => NerEntityType::Location,
        "DATE" | "TIME" => NerEntityType::Date,
        "MONEY" | "PERCENT" | "QUANTITY" | "CARDINAL" | "ORDINAL" => NerEntityType::Numeric,
        "EVENT" => NerEntityType::Event,
        "PRODUCT" => NerEntityType::Product,
        "WORK_OF_ART" => NerEntityType::WorkOfArt,
        "NORP" => NerEntityType::Group,
        "LAW" | "LANGUAGE" => NerEntityType::Misc,
        _ => return None,
    })
}

/// The class a gold act should land in. `filler` and `correction` have no
/// clean counterpart and are left out of class accuracy.
fn expected_class(act: &str) -> Option<SentenceClass> {
    Some(match act {
        "inform" => SentenceClass::Statement,
        "question" => SentenceClass::Question,
        "request" | "plan_commit" => SentenceClass::Command,
        "agreement" | "feedback" => SentenceClass::Acknowledgment,
        "social" => SentenceClass::Greeting,
        _ => return None,
    })
}

/// Whether a gold act carries content worth an observation.
fn gold_informative(act: &str) -> bool {
    matches!(act, "inform" | "request" | "plan_commit" | "correction")
}

/// The ingest gate (`observations::cls_gate_admits`), restated here because
/// it is private to the crate.
fn gate_admits(r: &NlpResult, labels: &[String], gate: &ClsGate) -> bool {
    if r.cls_probs.is_empty() {
        return r.sentence_class.is_informative() || r.cls_confidence <= 1.0 - gate.min_prob;
    }
    r.cls_probs.iter().enumerate().any(|(i, &p)| {
        p >= gate.min_prob
            && labels
                .get(i)
                .is_some_and(|name| gate.informative_labels.iter().any(|l| l == name))
    })
}

#[derive(Default)]
struct Scores {
    sentences: u64,
    ner: Prf,
    ner_untyped: Prf,
    ner_by_type: BTreeMap<String, Prf>,
    gold_tokens: u64,
    aligned: u64,
    pos_ok: u64,
    act_scored: u64,
    act_ok: u64,
    act_confusion: BTreeMap<String, BTreeMap<String, u64>>,
    gate_tp: u64,
    gate_fp: u64,
    gate_fn: u64,
    gate_tn: u64,
}

fn score(s: &mut Scores, g: &Gold, r: &NlpResult, gate: &ClsGate) {
    let maps = label_maps();
    s.sentences += 1;
    let gold_off = offsets(&g.text, &g.tokens);
    let pred_off = offsets(&g.text, &r.words);
    let span_of = |off: &[Option<(usize, usize)>], a: usize, b: usize| -> Option<(usize, usize)> {
        let first = off.get(a).copied().flatten()?;
        let last = off.get(b.checked_sub(1)?).copied().flatten()?;
        trim_punct(&g.text, first.0, last.1)
    };

    // NER. CoNLL04 only labels people, places and organisations, so other
    // predicted types are not counted against it.
    let scored = |t: NerEntityType| scored_type(&g.source, t);
    let gold: HashSet<_> = g
        .entities
        .iter()
        .filter_map(|e| {
            let t = gold_type(&e.kind)?;
            let (a, b) = span_of(&gold_off, e.start, e.end)?;
            Some((a, b, t))
        })
        .collect();
    let pred: HashSet<_> = r
        .entities
        .iter()
        .filter(|e| scored(e.entity_type))
        .filter_map(|e| {
            let (a, b) = span_of(&pred_off, e.start_word, e.end_word)?;
            Some((a, b, e.entity_type))
        })
        .collect();
    s.ner.add(&gold, &pred);
    let strip = |set: &HashSet<(usize, usize, NerEntityType)>| -> HashSet<_> {
        set.iter()
            .map(|&(a, b, _)| (a, b, NerEntityType::Misc))
            .collect()
    };
    s.ner_untyped.add(&strip(&gold), &strip(&pred));
    for t in [
        NerEntityType::Person,
        NerEntityType::Organization,
        NerEntityType::Location,
        NerEntityType::Date,
        NerEntityType::Numeric,
    ] {
        let only = |set: &HashSet<(usize, usize, NerEntityType)>| -> HashSet<_> {
            set.iter().filter(|x| x.2 == t).copied().collect()
        };
        s.ner_by_type
            .entry(format!("{t:?}"))
            .or_default()
            .add(&only(&gold), &only(&pred));
    }

    // POS and DEP on the words that line up with a gold token.
    if !g.pos.is_empty() {
        let gold_at: HashMap<(usize, usize), usize> = gold_off
            .iter()
            .enumerate()
            .filter_map(|(i, o)| Some(((*o)?, i)))
            .collect();
        let pred_to_gold: Vec<Option<usize>> = pred_off
            .iter()
            .map(|o| o.and_then(|o| gold_at.get(&o).copied()))
            .collect();
        s.gold_tokens += g.tokens.len() as u64;
        for (w, gi) in pred_to_gold.iter().enumerate() {
            let Some(gi) = *gi else { continue };
            s.aligned += 1;
            if maps
                .pos_labels
                .get(r.pos_indices.get(w).copied().unwrap_or(usize::MAX))
                == g.pos.get(gi)
            {
                s.pos_ok += 1;
            }
        }
    }

    // Dialogue act and the observation gate.
    if let Some(act) = &g.act {
        let got = format!("{:?}", r.sentence_class);
        *s.act_confusion
            .entry(act.clone())
            .or_default()
            .entry(got)
            .or_default() += 1;
        if let Some(want) = expected_class(act) {
            s.act_scored += 1;
            if want == r.sentence_class {
                s.act_ok += 1;
            }
        }
        let admit = gate_admits(r, &maps.cls_labels, gate);
        match (gold_informative(act), admit) {
            (true, true) => s.gate_tp += 1,
            (false, true) => s.gate_fp += 1,
            (true, false) => s.gate_fn += 1,
            (false, false) => s.gate_tn += 1,
        }
    }
}

/// Whether an entity type is scored on `source`. CoNLL04 only labels
/// people, places and organisations, so other predicted types do not count
/// against it.
fn scored_type(source: &str, t: NerEntityType) -> bool {
    source != "conll04"
        || matches!(
            t,
            NerEntityType::Person | NerEntityType::Location | NerEntityType::Organization
        )
}

/// NER span F1 for one message: `chunk`'s sentences joined by single
/// spaces into `text`, analysed as `results`.
fn score_message(prf: &mut Prf, chunk: &[&Gold], text: &str, results: &[NlpResult]) {
    let mut gold = HashSet::new();
    let mut base = 0;
    for g in chunk {
        let off = offsets(&g.text, &g.tokens);
        for e in &g.entities {
            let Some(t) = gold_type(&e.kind) else {
                continue;
            };
            let first = off.get(e.start).copied().flatten();
            let last = e
                .end
                .checked_sub(1)
                .and_then(|i| off.get(i).copied().flatten());
            if let (Some(a), Some(b)) = (first, last)
                && let Some((a, b)) = trim_punct(&g.text, a.0, b.1)
            {
                gold.insert((base + a, base + b, t));
            }
        }
        base += g.text.len() + 1;
    }
    // Each result's words are found from where the previous one ended; the
    // first word is searched for without a window, past any dropped fragment.
    let mut pred = HashSet::new();
    let mut cursor = 0;
    for r in results {
        let Some(first) = r.words.first() else {
            continue;
        };
        let Some(at) = text[cursor..].find(first.trim()) else {
            continue;
        };
        let start = cursor + at;
        let off: Vec<_> = offsets(&text[start..], &r.words)
            .into_iter()
            .map(|o| o.map(|(a, b)| (start + a, start + b)))
            .collect();
        if let Some(end) = off.iter().rev().find_map(|o| *o) {
            cursor = end.1;
        }
        for e in r
            .entities
            .iter()
            .filter(|e| scored_type(&chunk[0].source, e.entity_type))
        {
            let first = off.get(e.start_word).copied().flatten();
            let last = e
                .end_word
                .checked_sub(1)
                .and_then(|i| off.get(i).copied().flatten());
            if let (Some(a), Some(b)) = (first, last)
                && let Some((a, b)) = trim_punct(text, a.0, b.1)
            {
                pred.insert((a, b, e.entity_type));
            }
        }
    }
    prf.add(&gold, &pred);
}

fn report(s: &Scores) -> serde_json::Value {
    let prf = |p: Prf| serde_json::json!({"f1": p.f1(), "p": p.precision(), "r": p.recall(), "tp": p.tp, "fp": p.fp, "fn": p.fn_});
    serde_json::json!({
        "sentences": s.sentences,
        "ner": prf(s.ner),
        "ner_untyped": prf(s.ner_untyped),
        "ner_by_type": s.ner_by_type.iter().map(|(k, v)| (k.clone(), prf(*v))).collect::<BTreeMap<_, _>>(),
        "aligned_share": ratio(s.aligned, s.gold_tokens),
        "pos_acc": ratio(s.pos_ok, s.aligned),
        "act_acc": ratio(s.act_ok, s.act_scored),
        "act_scored": s.act_scored,
        "act_confusion": s.act_confusion,
        "gate": {
            "acc": ratio(s.gate_tp + s.gate_tn, s.gate_tp + s.gate_tn + s.gate_fp + s.gate_fn),
            "precision": ratio(s.gate_tp, s.gate_tp + s.gate_fp),
            "recall": ratio(s.gate_tp, s.gate_tp + s.gate_fn),
            "tp": s.gate_tp, "fp": s.gate_fp, "fn": s.gate_fn, "tn": s.gate_tn,
        },
    })
}

fn percentiles(mut ms: Vec<f64>) -> serde_json::Value {
    ms.sort_by(f64::total_cmp);
    let at = |p: f64| {
        ms.get(((ms.len() as f64 - 1.0) * p).round() as usize)
            .copied()
            .unwrap_or(0.0)
    };
    let mean = if ms.is_empty() {
        0.0
    } else {
        ms.iter().sum::<f64>() / ms.len() as f64
    };
    serde_json::json!({"n": ms.len(), "mean": mean, "p50": at(0.5), "p95": at(0.95), "p99": at(0.99)})
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn".parse().expect("filter")),
        )
        .init();
    let cli = Cli::parse();

    let gold: Vec<Gold> = std::fs::read_to_string(&cli.data)
        .context("read --data")?
        .lines()
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()
        .context("parse --data")?;

    let Some(mut nlp) = NlpConfig::preset(&cli.preset) else {
        bail!(
            "unknown preset {}; one of {:?}",
            cli.preset,
            NlpConfig::PRESETS
        );
    };
    if let Some(v) = &cli.variant {
        nlp.variant = v.clone();
    }
    let model_id = nlp.model_id.clone();
    let config = UnikoConfig {
        nlp,
        ..UnikoConfig::default()
    };

    let t0 = Instant::now();
    let kb = KnowledgeBase::in_memory(config).await.context("kb")?;
    kb.model_runtime()
        .context("xervo runtime")?
        .nlp_model("nlp/default")
        .await
        .map_err(|e| anyhow::anyhow!("{} did not load: {e}", cli.preset))?;
    let pipeline = NlpPipeline::try_new(&kb).await.context("pipeline")?;
    pipeline.analyze("The model is warming up now.").await?;
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;
    eprintln!("{}: loaded in {load_ms:.0} ms", cli.preset);

    let gate = uniko_extract::observations::rules_engine::rules::load_bundled_rules()
        .filters
        .cls_gate
        .clone();
    let mut by_source: BTreeMap<String, Scores> = BTreeMap::new();
    let mut by_domain: BTreeMap<String, Scores> = BTreeMap::new();
    let mut sentence_ms = Vec::with_capacity(gold.len());
    let mut words = 0u64;
    let t_all = Instant::now();
    for (i, g) in gold.iter().enumerate() {
        let t = Instant::now();
        let r = pipeline.analyze(&g.text).await?;
        if i < cli.dump {
            eprintln!(
                "--- {:?} {:.2} {}\n gold {:?}\n words {:?}\n gold ents {:?}\n pred ents {:?}",
                r.sentence_class,
                r.cls_confidence,
                g.text,
                g.tokens,
                r.words,
                g.entities
                    .iter()
                    .map(|e| (e.start, e.end, e.kind.as_str()))
                    .collect::<Vec<_>>(),
                r.entities
                    .iter()
                    .map(|e| (
                        e.start_word,
                        e.end_word,
                        e.entity_type,
                        e.text.as_str(),
                        r.words[e.start_word..e.end_word].join(" ")
                    ))
                    .collect::<Vec<_>>()
            );
        }
        sentence_ms.push(t.elapsed().as_secs_f64() * 1e3);
        words += g.tokens.len() as u64;
        score(by_source.entry(g.source.clone()).or_default(), g, &r, &gate);
        if g.source == "kniv-dev" {
            score(by_domain.entry(g.domain.clone()).or_default(), g, &r, &gate);
        }
        if i % 500 == 499 {
            eprintln!("  {}/{}", i + 1, gold.len());
        }
    }
    let sentence_secs = t_all.elapsed().as_secs_f64();

    // Four consecutive sentences of one source and domain make a message.
    let mut groups: BTreeMap<(&str, &str), Vec<&Gold>> = BTreeMap::new();
    for g in &gold {
        groups
            .entry((g.source.as_str(), g.domain.as_str()))
            .or_default()
            .push(g);
    }
    let mut message_ner: BTreeMap<String, Prf> = BTreeMap::new();
    let mut message_ms = Vec::new();
    let mut message_sentences = 0usize;
    for ((source, domain), pool) in &groups {
        for chunk in pool.chunks(4).filter(|c| c.len() == 4) {
            let text = chunk
                .iter()
                .map(|g| g.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            let t = Instant::now();
            let results = pipeline.analyze_sentences(&text).await?;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            if *source == "kniv-dev" && *domain != "conversation" {
                message_ms.push(ms);
                message_sentences += results.len();
            }
            score_message(
                message_ner.entry((*source).to_string()).or_default(),
                chunk,
                &text,
                &results,
            );
        }
    }

    let line = serde_json::json!({
        "preset": cli.preset,
        "model_id": model_id,
        "variant": cli.variant,
        "load_ms": load_ms,
        "sentence_ms": percentiles(sentence_ms),
        "words_per_sec": words as f64 / sentence_secs,
        "message_ms": percentiles(message_ms),
        "message_sentences": message_sentences,
        "message_ner": message_ner.iter().map(|(k, v)| (k.clone(), serde_json::json!({"f1": v.f1(), "p": v.precision(), "r": v.recall()}))).collect::<BTreeMap<_, _>>(),
        "by_source": by_source.iter().map(|(k, v)| (k.clone(), report(v))).collect::<BTreeMap<_, _>>(),
        "by_domain": by_domain.iter().map(|(k, v)| (k.clone(), report(v))).collect::<BTreeMap<_, _>>(),
    });
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&cli.output)?;
    writeln!(out, "{line}")?;
    let k = &line["by_source"]["kniv-dev"];
    eprintln!(
        "{}: ner {:.3} conll04 {:.3} | message ner {:.3} conll04 {:.3} | pos {:.3} act {:.3} gate {:.3} | {:.1} ms/sentence p50, {:.1} ms/message p50",
        cli.preset,
        k["ner"]["f1"].as_f64().unwrap_or(0.0),
        line["by_source"]["conll04"]["ner"]["f1"]
            .as_f64()
            .unwrap_or(0.0),
        message_ner.get("kniv-dev").map_or(0.0, |p| p.f1()),
        message_ner.get("conll04").map_or(0.0, |p| p.f1()),
        k["pos_acc"].as_f64().unwrap_or(0.0),
        k["act_acc"].as_f64().unwrap_or(0.0),
        k["gate"]["acc"].as_f64().unwrap_or(0.0),
        line["sentence_ms"]["p50"].as_f64().unwrap_or(0.0),
        line["message_ms"]["p50"].as_f64().unwrap_or(0.0),
    );
    Ok(())
}
