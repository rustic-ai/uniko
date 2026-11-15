//! A real kniv v7 model through `NlpPipeline`: the config preset reaches
//! the catalog, uni-xervo loads the v7 graph set, and the adapter turns its
//! word-level output into the entities and sentences uniko consumes.
//!
//! Ignored by default: it downloads the model on first run (the INT8 graphs
//! of `kniv-v7-small-en`, about 120 MB). `UNIKO_NLP_PRESET` picks another
//! size.
//!
//! Run:
//!   cargo nextest run -p uniko-extract --features onnx \
//!     --test nlp_v7_integration_test --run-ignored all --no-capture

#[cfg(feature = "onnx")]
mod onnx_tests {
    use uniko_extract::nlp::NlpPipeline;
    use uniko_extract::nlp::types::{NerEntityType, SentenceClass};
    use uniko_store::KnowledgeBase;
    use uniko_store::config::{NlpConfig, UnikoConfig};

    async fn pipeline() -> NlpPipeline {
        let preset = std::env::var("UNIKO_NLP_PRESET").unwrap_or_else(|_| "v7-small".into());
        let config = UnikoConfig {
            nlp: NlpConfig::preset(&preset).expect("known preset"),
            ..UnikoConfig::default()
        };
        let kb = KnowledgeBase::in_memory(config).await.expect("kb");
        // `try_new` folds every load failure into `None`; resolve the alias
        // first so a failure says why.
        if let Err(e) = kb
            .model_runtime()
            .expect("xervo runtime")
            .nlp_model("nlp/default")
            .await
        {
            panic!("{preset} did not load: {e}");
        }
        NlpPipeline::try_new(&kb).await.expect("pipeline")
    }

    #[tokio::test]
    #[ignore = "downloads a kniv v7 model"]
    async fn v7_extracts_entities_and_sentences() {
        let nlp = pipeline().await;
        let text = "Caroline moved to Berlin with her sister in March. \
                    She started a new job at Siemens last week. \
                    Thanks so much for asking!";

        let sentences = nlp.analyze_sentences(text).await.expect("analyze");
        for s in &sentences {
            eprintln!(
                "{:?} {:?} {:?}",
                s.sentence_class,
                s.words,
                s.entities
                    .iter()
                    .map(|e| (&e.text, e.entity_type))
                    .collect::<Vec<_>>()
            );
        }
        assert_eq!(sentences.len(), 3, "one result per sentence");
        assert_eq!(sentences[0].words[0], "Caroline", "words are not glued");
        assert!(
            sentences[0]
                .entities
                .iter()
                .any(|e| e.text == "Caroline" && e.entity_type == NerEntityType::Person),
            "Caroline is a person"
        );
        assert!(
            sentences[0].entities.iter().any(|e| e.text == "Berlin"),
            "Berlin is found"
        );
        assert!(
            sentences[1].entities.iter().any(|e| e.text == "Siemens"),
            "Siemens is found"
        );
        for s in &sentences {
            assert!(
                s.dep_arcs
                    .iter()
                    .all(|a| a.head == usize::MAX || a.head < s.words.len()),
                "heads stay inside their sentence"
            );
            assert!(!s.cls_probs.is_empty(), "v7 scores fill cls_probs");
        }
        assert_eq!(sentences[0].sentence_class, SentenceClass::Statement);

        let whole = nlp.analyze(text).await.expect("analyze");
        assert!(whole.words.len() > 20, "analyze keeps the whole text");
    }
}
