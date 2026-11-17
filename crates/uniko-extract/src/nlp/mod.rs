//! Multi-task NLP via xervo's managed `NlpModel`.
//!
//! [`NlpPipeline`] resolves the `nlp/default` alias to a xervo
//! [`NlpModel`](uni_xervo::traits::NlpModel) (registered as
//! `ModelTask::Nlp`) and turns raw text into uniko's structured
//! [`NlpResult`] via [`adapter::xervo_to_uniko`]. xervo owns
//! tokenization, ONNX inference, and POS/NER/DEP/SRL/CLS decoding; uniko
//! only reconciles output conventions in the adapter.
//!
//! If the runtime or alias is unavailable, [`NlpPipeline::try_new`]
//! returns `None` and callers fall back to rule-based extraction.
//!
//! Both kniv generations ([`NlpGeneration`]) go through the same adapter;
//! the pipeline tells it which layout the configured model uses. Under v7
//! [`NlpPipeline::analyze_sentences`] analyses a message in one call and
//! splits the result by sentence, since v7 reads whole windows of up to
//! 1,024 subwords and resolves each sentence with its neighbours in view.

pub mod adapter;
pub mod assets;
pub mod decode;
pub mod types;

use std::sync::Arc;

use uni_xervo::traits::{NlpModel, NlpRequest, NlpResult as XervoNlp, NlpTasks};

use uniko_store::KnowledgeBase;
use uniko_store::UnikoError;
use uniko_store::config::NlpGeneration;
use uniko_store::schema::NLP_ALIAS;

use types::NlpResult;

/// xervo-backed multi-task NLP pipeline.
///
/// A single `analyze` call runs xervo's `NlpModel` (tokenize → infer →
/// decode) and adapts the result into uniko's [`NlpResult`]. Only the
/// resolved model handle is per-instance; label maps are a global
/// `OnceLock`.
#[derive(Clone)]
pub struct NlpPipeline {
    model: Arc<dyn NlpModel>,
    /// Mirror of `UnikoConfig.nlp_srl_enabled`. When `false`, SRL is not
    /// requested and `NlpResult.srl_frames` stays empty.
    srl_enabled: bool,
    /// Which kniv generation the alias serves, from `UnikoConfig.nlp`.
    generation: NlpGeneration,
    /// The model's dialogue-act vocabulary under v7 (see
    /// [`adapter::Layout::acts`]); `None` under v1.
    acts: Option<Vec<String>>,
}

impl std::fmt::Debug for NlpPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NlpPipeline")
            .field("model", &"<NlpModel>")
            .field("srl_enabled", &self.srl_enabled)
            .field("generation", &self.generation)
            .finish_non_exhaustive()
    }
}

impl NlpPipeline {
    /// Try to create a pipeline from the KB's xervo runtime.
    ///
    /// Returns `None` if the runtime is absent or the `nlp/default` alias
    /// cannot be resolved as an NLP model (e.g. the model download
    /// fails) — callers fall back to rule-based extraction.
    pub async fn try_new(kb: &KnowledgeBase) -> Option<Self> {
        let runtime = kb.model_runtime()?;
        let model = match runtime.nlp_model(NLP_ALIAS).await {
            Ok(m) => {
                kb.note_model_loaded(NLP_ALIAS);
                m
            }
            Err(e) => {
                tracing::debug!(error = %e, alias = NLP_ALIAS, "xervo NlpModel unavailable");
                return None;
            }
        };
        let generation = kb.config().nlp.generation;
        let acts = match generation {
            NlpGeneration::V1 => None,
            NlpGeneration::V7 => Some(
                model
                    .label_maps()
                    .map(|maps| maps.cls.clone())
                    .unwrap_or_default(),
            ),
        };
        Some(Self {
            model,
            srl_enabled: kb.config().nlp_srl_enabled,
            generation,
            acts,
        })
    }

    /// The output layout of the configured model, for the adapter.
    fn layout(&self) -> adapter::Layout<'_> {
        adapter::Layout {
            word_tokens: self.generation == NlpGeneration::V7,
            acts: self.acts.as_deref(),
        }
    }

    fn adapt(&self, x: &XervoNlp) -> NlpResult {
        adapter::xervo_to_uniko(x, assets::label_maps(), self.srl_enabled, self.layout())
    }

    /// The NLP task set this pipeline requests (SRL gated by config).
    fn tasks(&self) -> NlpTasks {
        let mut tasks = NlpTasks::POS | NlpTasks::NER | NlpTasks::DEP | NlpTasks::CLS;
        if self.srl_enabled {
            tasks |= NlpTasks::SRL;
        }
        tasks
    }

    /// Run full NLP analysis on a single text string.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Pipeline`] if xervo inference fails or
    /// returns no result.
    pub async fn analyze(&self, text: &str) -> Result<NlpResult, UnikoError> {
        let mut out = self
            .model
            .analyze(vec![NlpRequest::new(text, self.tasks())])
            .await
            .map_err(|e| UnikoError::Pipeline(format!("xervo NLP analyze: {e}")))?;
        let x = out
            .pop()
            .ok_or_else(|| UnikoError::Pipeline("xervo NLP returned no result".into()))?;
        Ok(self.adapt(&x))
    }

    /// Analyze text by splitting into sentences first.
    ///
    /// Returns one [`NlpResult`] per sentence (sentences under 4 words
    /// are dropped by [`split_sentences`]). All sentences go through a
    /// single `analyze` call as separate requests, which keeps xervo's
    /// per-sentence token indexing (NER/DEP/SRL) local to each sentence.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Pipeline`] if xervo inference fails.
    pub async fn analyze_sentences(&self, text: &str) -> Result<Vec<NlpResult>, UnikoError> {
        if self.generation == NlpGeneration::V7 {
            let mut out = self
                .model
                .analyze(vec![NlpRequest::new(text, self.tasks())])
                .await
                .map_err(|e| UnikoError::Pipeline(format!("xervo NLP analyze: {e}")))?;
            let x = out
                .pop()
                .ok_or_else(|| UnikoError::Pipeline("xervo NLP returned no result".into()))?;
            return Ok(split_by_sentence(&x)
                .iter()
                .map(|s| self.adapt(s))
                .collect());
        }
        let sentences = split_sentences(text);
        if sentences.is_empty() {
            return Ok(Vec::new());
        }
        let tasks = self.tasks();
        let requests: Vec<NlpRequest<'_>> = sentences
            .iter()
            .map(|s| NlpRequest::new(s.as_str(), tasks))
            .collect();
        let results = self
            .model
            .analyze(requests)
            .await
            .map_err(|e| UnikoError::Pipeline(format!("xervo NLP analyze: {e}")))?;
        Ok(results.iter().map(|x| self.adapt(x)).collect())
    }
}

/// Fewest words a sentence needs to be analysed, as [`split_sentences`]
/// applies it.
const MIN_SENTENCE_WORDS: usize = 4;

/// Split one whole-message result into one result per sentence, each
/// indexed locally from 0, as a per-sentence request would have returned.
///
/// Sentences under [`MIN_SENTENCE_WORDS`] words are dropped, as
/// [`split_sentences`] does; when none is long enough the whole result is
/// kept as one, provided it is. A dependency head or SRL argument outside
/// its sentence is dropped (heads become the sentence root).
fn split_by_sentence(x: &XervoNlp) -> Vec<XervoNlp> {
    let ranges: Vec<(usize, usize)> = x
        .sentences
        .iter()
        .map(|s| s.token_range)
        .filter(|(first, last)| last + 1 - first >= MIN_SENTENCE_WORDS)
        .collect();
    if ranges.is_empty() {
        return if x.tokens.len() >= MIN_SENTENCE_WORDS {
            vec![x.clone()]
        } else {
            Vec::new()
        };
    }
    let index_of = |range: (usize, usize)| x.sentences.iter().position(|s| s.token_range == range);
    ranges
        .into_iter()
        .map(|(first, last)| {
            let local = |t: usize| (first..=last).contains(&t).then(|| t - first);
            let mut out = XervoNlp::default();
            out.tokens = x.tokens[first..=last]
                .iter()
                .map(|t| {
                    let mut t = t.clone();
                    t.word_index -= first;
                    if let Some(dep) = t.dep.as_mut() {
                        dep.head = dep.head.and_then(local);
                    }
                    t
                })
                .collect();
            if let Some(sentence) = index_of((first, last)).map(|i| &x.sentences[i]) {
                let mut sentence = sentence.clone();
                sentence.token_range = (0, last - first);
                out.sentences.push(sentence);
                let sentence_index = index_of((first, last));
                out.speech_acts = x
                    .speech_acts
                    .iter()
                    .filter(|a| Some(a.sentence_index) == sentence_index)
                    .map(|a| {
                        let mut a = a.clone();
                        a.sentence_index = 0;
                        a
                    })
                    .collect();
            }
            out.frames = x
                .frames
                .iter()
                .filter_map(|f| {
                    let mut f = f.clone();
                    f.predicate_token = local(f.predicate_token)?;
                    f.roles
                        .retain_mut(|r| match (local(r.span.0), local(r.span.1)) {
                            (Some(a), Some(b)) => {
                                r.span = (a, b);
                                true
                            }
                            _ => false,
                        });
                    Some(f)
                })
                .collect();
            out.entities = x
                .entities
                .iter()
                .filter_map(|e| {
                    let mut e = e.clone();
                    e.token_span = (local(e.token_span.0)?, local(e.token_span.1)?);
                    Some(e)
                })
                .collect();
            out
        })
        .collect()
}

/// Abbreviations a sentence does not end after: titles, months and a few
/// others. With single capital initials ("F.") and dotted acronyms
/// ("U.S."), this is kniv's reference rule (`kniv_v7/windows.py`,
/// `_NO_SPLIT`). Ones that often do end a sentence ("Inc.", "etc.") are
/// left out on purpose.
const NO_SPLIT_AFTER: &[&str] = &[
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Gen", "Col", "Lt", "Sgt", "Capt", "Cmdr", "Adm", "Maj",
    "Rep", "Sen", "Gov", "Pres", "Rev", "Hon", "St", "Mt", "Ft", "Sr", "Jr", "Jan", "Feb", "Mar",
    "Apr", "Jun", "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec", "No", "Nos", "vs", "Fig",
    "approx", "Vol", "pp", "Ave", "Blvd",
];

/// Whether `piece`, which ends in a period, ends in an abbreviation rather
/// than a sentence.
fn ends_in_abbreviation(piece: &str) -> bool {
    let piece = piece.trim_end_matches(['"', '\'', ')', ']']);
    let Some(word) = piece
        .strip_suffix('.')
        .and_then(|p| p.split_whitespace().next_back())
    else {
        return false;
    };
    let word = word.trim_start_matches(['"', '\'', '(', '[']);
    let initial = word.len() == 1 && word.chars().all(|c| c.is_ascii_uppercase());
    // "U.S" (its last period was stripped above): letters joined by periods.
    let acronym = word.len() > 1
        && word
            .split('.')
            .all(|p| p.len() == 1 && p.chars().all(|c| c.is_ascii_alphabetic()));
    initial || acronym || NO_SPLIT_AFTER.contains(&word)
}

/// Whether `c` can open a sentence: an uppercase letter, a digit, a quote
/// or an opening bracket.
fn opens_sentence(c: char) -> bool {
    c.is_uppercase() || c.is_ascii_digit() || matches!(c, '"' | '\'' | '(' | '[' | '“' | '‘')
}

/// Split text into sentences for per-sentence NLP analysis.
///
/// A sentence ends at `.`, `!` or `?`, after any closing quotes or
/// brackets, where whitespace follows and the next character can open a
/// sentence — but not after a title, a month, an initial or a dotted
/// acronym ("Sen. Robert F. Kennedy", "Dec. 13", "the U.S. Navy"). This is
/// the rule kniv v7's reference runtime uses, so v1 and v7 cut text alike.
///
/// Fragments under four words are dropped. If none survives, the whole
/// text is returned (provided it has at least four words).
fn split_sentences(text: &str) -> Vec<String> {
    let mut pieces: Vec<&str> = Vec::new();
    let mut start = 0;
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut i = 0;
    while i < chars.len() {
        if !matches!(chars[i].1, '.' | '!' | '?') {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        while j < chars.len() && matches!(chars[j].1, '"' | '\'' | ')' | ']' | '”' | '’') {
            j += 1;
        }
        let close = chars.get(j).map_or(text.len(), |c| c.0);
        let mut k = j;
        while k < chars.len() && chars[k].1.is_whitespace() {
            k += 1;
        }
        let boundary = k > j && chars.get(k).is_some_and(|c| opens_sentence(c.1));
        if boundary && !(chars[i].1 == '.' && ends_in_abbreviation(&text[start..close])) {
            pieces.push(&text[start..close]);
            start = chars[k].0;
        }
        i = k.max(i + 1);
    }
    pieces.push(&text[start..]);

    let mut sentences: Vec<String> = pieces
        .into_iter()
        .map(str::trim)
        .filter(|s| s.split_whitespace().count() >= MIN_SENTENCE_WORDS)
        .map(str::to_string)
        .collect();
    if sentences.is_empty() && text.split_whitespace().count() >= MIN_SENTENCE_WORDS {
        sentences.push(text.trim().to_string());
    }
    sentences
}

#[cfg(test)]
mod split_tests {
    use super::split_sentences;

    #[test]
    fn titles_initials_months_and_acronyms_do_not_end_a_sentence() {
        assert_eq!(
            split_sentences(
                "Sen. Robert F. Kennedy was shot in Los Angeles on Jun. 5 of that year. \
                 The U.S. Army sent Dr. Smith to look at the scene."
            ),
            [
                "Sen. Robert F. Kennedy was shot in Los Angeles on Jun. 5 of that year.",
                "The U.S. Army sent Dr. Smith to look at the scene.",
            ]
        );
    }

    #[test]
    fn a_sentence_ends_before_a_capital_digit_or_quote_only() {
        assert_eq!(
            split_sentences("We moved to Berlin in March. 2024 was a big year for us all."),
            [
                "We moved to Berlin in March.",
                "2024 was a big year for us all."
            ]
        );
        // A lowercase word after the period: not a boundary.
        assert_eq!(
            split_sentences("Pick one, e.g. the red one or the blue one."),
            ["Pick one, e.g. the red one or the blue one."]
        );
        assert_eq!(
            split_sentences("She said \"we are leaving now.\" Then they left the house."),
            [
                "She said \"we are leaving now.\"",
                "Then they left the house."
            ]
        );
    }

    #[test]
    fn names_titles_dates_and_acronyms_stay_in_one_sentence() {
        for text in [
            "Sen. Robert F. Kennedy spoke today.",
            "Dr. Smith met J. K. Rowling.",
            "He joined the U.S. Army in 1990.",
            "It was Dec. 13 when Lt. Col. Smith arrived.",
            "Mr. and Mrs. Lee met Prof. Chan at St. Mary hospital.",
            "Gov. Brown signed it on Jan. 5 in Washington, D.C. after a vote.",
            "The case was Roe vs. Wade.",
            "Michael D. Papagiannis wrote No. 7.",
        ] {
            assert_eq!(split_sentences(text), [text], "{text}");
        }
        assert_eq!(
            split_sentences("The vote was 5 to 4. Justice Kennedy wrote the opinion."),
            ["The vote was 5 to 4.", "Justice Kennedy wrote the opinion."]
        );
    }

    #[test]
    fn abbreviations_that_often_end_a_sentence_still_split() {
        assert_eq!(
            split_sentences("He works for Acme Widgets Inc. The company makes small parts."),
            [
                "He works for Acme Widgets Inc.",
                "The company makes small parts."
            ]
        );
    }

    #[test]
    fn short_fragments_drop_and_the_whole_text_is_the_fallback() {
        assert_eq!(
            split_sentences("Hi there! How are you doing today, my friend?"),
            ["How are you doing today, my friend?"]
        );
        assert_eq!(
            split_sentences("Yes. No. Maybe so."),
            ["Yes. No. Maybe so."]
        );
        assert!(split_sentences("Ok.").is_empty());
    }
}

#[cfg(test)]
mod v7_tests {
    use super::*;
    use types::SentenceClass;
    use uni_xervo::traits::{DepLink, NerEntity, NlpSentence, NlpToken, SpeechAct};

    /// v7's dialogue acts, in the order its `scores` are reported.
    fn acts() -> Vec<String> {
        [
            "Question",
            "Inform",
            "Directive",
            "Commissive",
            "Feedback",
            "Social",
        ]
        .map(String::from)
        .to_vec()
    }

    /// What uni-xervo's v7 path returns for "Ada met Bob in London. Thanks
    /// so much for that !": whole-word tokens with no metaspace, global
    /// dependency heads, and per-act scores.
    fn whole_message() -> XervoNlp {
        // (word, ner, head as a global token index, relation)
        let words: [(&str, Option<&str>, Option<usize>, &str); 12] = [
            ("Ada", Some("B-PERSON"), Some(1), "nsubj"),
            ("met", None, None, "root"),
            ("Bob", Some("B-PERSON"), Some(1), "obj"),
            ("in", None, Some(4), "case"),
            ("London", Some("B-GPE"), Some(1), "obl"),
            (".", None, Some(1), "punct"),
            ("Thanks", None, None, "root"),
            ("so", None, Some(8), "advmod"),
            ("much", None, Some(6), "advmod"),
            ("for", None, Some(10), "case"),
            // A head across the sentence boundary, which a split must drop.
            ("that", None, Some(2), "obl"),
            ("!", None, Some(6), "punct"),
        ];
        let mut x = XervoNlp::default();
        for (i, (text, ner, head, rel)) in words.into_iter().enumerate() {
            let mut t = NlpToken::default();
            t.text = text.to_string();
            t.word_index = i;
            t.ner = ner.map(String::from);
            let mut dep = DepLink::default();
            dep.head = head;
            dep.relation = rel.to_string();
            t.dep = Some(dep);
            x.tokens.push(t);
        }
        for range in [(0, 5), (6, 11)] {
            let mut s = NlpSentence::default();
            s.token_range = range;
            x.sentences.push(s);
        }
        for (i, scores) in [
            [0.05, 0.9, 0.1, 0.0, 0.0, 0.7],
            [0.0, 0.1, 0.0, 0.0, 0.2, 0.95],
        ]
        .into_iter()
        .enumerate()
        {
            let mut a = SpeechAct::default();
            a.sentence_index = i;
            a.label = if i == 0 { "Inform" } else { "Social" }.to_string();
            a.confidence = if i == 0 { 0.9 } else { 0.95 };
            a.scores = scores.to_vec();
            x.speech_acts.push(a);
        }
        let mut london = NerEntity::default();
        london.text = "London".into();
        london.label = "GPE".into();
        london.token_span = (4, 4);
        x.entities.push(london);
        x
    }

    fn adapt(x: &XervoNlp, acts: &[String]) -> types::NlpResult {
        let layout = adapter::Layout {
            word_tokens: true,
            acts: Some(acts),
        };
        adapter::xervo_to_uniko(x, assets::label_maps(), true, layout)
    }

    #[test]
    fn v7_words_map_one_to_one() {
        let acts = acts();
        let r = adapt(&whole_message(), &acts);
        assert_eq!(r.words.len(), 12);
        assert_eq!(r.words[4], "London");
        let persons: Vec<_> = r.entities.iter().map(|e| e.text.as_str()).collect();
        assert_eq!(persons, ["Ada", "Bob", "London"]);

        // The v1 layout regroups on a leading space, which v7 words lack:
        // the whole message would become one word.
        let v1 = adapter::xervo_to_uniko(
            &whole_message(),
            assets::label_maps(),
            true,
            adapter::Layout::default(),
        );
        assert_eq!(v1.words.len(), 1);
    }

    #[test]
    fn split_rebases_each_sentence() {
        let parts = split_by_sentence(&whole_message());
        assert_eq!(parts.len(), 2);
        let acts = acts();
        let first = adapt(&parts[0], &acts);
        let second = adapt(&parts[1], &acts);

        assert_eq!(first.words, ["Ada", "met", "Bob", "in", "London", "."]);
        assert_eq!(first.entities.len(), 3);
        let head = |r: &types::NlpResult, w: usize| {
            r.dep_arcs.iter().find(|a| a.dependent == w).unwrap().head
        };
        assert_eq!(head(&first, 0), 1, "Ada -> met");
        assert_eq!(head(&first, 1), usize::MAX, "met is the root");

        assert_eq!(second.words[0], "Thanks");
        assert_eq!(head(&second, 2), 0, "much -> Thanks, rebased");
        assert_eq!(
            head(&second, 4),
            usize::MAX,
            "a cross-sentence head is dropped"
        );
        assert_eq!(parts[1].speech_acts.len(), 1);
        assert_eq!(parts[1].speech_acts[0].sentence_index, 0);
        assert!(
            parts[1].entities.is_empty(),
            "London stays with sentence one"
        );
    }

    #[test]
    fn v7_acts_project_onto_v1_labels() {
        let acts = acts();
        let labels = assets::label_maps();
        let prob = |r: &types::NlpResult, v1: &str| {
            r.cls_probs[labels.cls_labels.iter().position(|l| l == v1).unwrap()]
        };
        let parts = split_by_sentence(&whole_message());

        let first = adapt(&parts[0], &acts);
        assert_eq!(first.sentence_class, SentenceClass::Statement);
        assert_eq!(first.cls_probs.len(), labels.cls_labels.len());
        assert!((prob(&first, "inform") - 0.9).abs() < 1e-6);
        assert!((prob(&first, "social") - 0.7).abs() < 1e-6);
        assert_eq!(labels.cls_labels[first.cls_index], "inform");

        let second = adapt(&parts[1], &acts);
        assert_eq!(second.sentence_class, SentenceClass::Greeting);
        assert!(
            (prob(&second, "confirm") - 0.2).abs() < 1e-6,
            "Feedback -> confirm"
        );
        assert_eq!(prob(&second, "status"), 0.0, "no v7 act maps to status");
    }

    #[test]
    fn split_keeps_the_four_word_floor() {
        let mut x = whole_message();
        x.tokens.truncate(3);
        x.sentences.truncate(1);
        x.sentences[0].token_range = (0, 2);
        assert!(split_by_sentence(&x).is_empty(), "three words are too few");
    }
}
