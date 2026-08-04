//! Check what observations the ONNX model produces for the specific
//! evidence messages that our single-hop retrieval misses.
//!
//! Run: cargo nextest run -p uniko-extract --features onnx \
//!   --test obs_for_misses_test --nocapture --run-ignored all

#[cfg(feature = "onnx")]
mod onnx_tests {
    use uni_db::ModelAliasSpec;
    use uniko_extract::ingest::context::SentenceContext;
    use uniko_extract::nlp::NlpPipeline;
    use uniko_extract::nlp::assets::label_maps;
    use uniko_extract::nlp::decode::extract_dep_observations;
    use uniko_store::KnowledgeBase;
    use uniko_store::config::UnikoConfig;

    async fn make_pipeline() -> NlpPipeline {
        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
        std::env::set_current_dir(workspace_root).expect("cd to workspace root");
        let config = UnikoConfig {
            catalog_path: Some(workspace_root.join("config/catalog.json")),
            schema_path: Some(workspace_root.join("config/schema.json")),
            ..Default::default()
        };
        let tmp = std::env::temp_dir().join(format!("uniko-obs-miss-test-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).ok();
        let kb = KnowledgeBase::open_with_xervo(&tmp, config, Vec::<ModelAliasSpec>::new())
            .await
            .expect("KB");
        NlpPipeline::try_new(&kb).await.expect("pipeline")
    }

    /// (dia_id, speaker, &mut SentenceContext::default(), text, question_it_answers)
    fn evidence_messages() -> Vec<(&'static str, &'static str, &'static str, &'static str)> {
        vec![
            (
                "D1:6",
                "Jon",
                "I've been into dancing since I was a kid and it's been my passion and escape. I wanna start a dance studio so I can teach others the joy that dancing brings me.",
                "How do Jon and Gina both like to destress?",
            ),
            (
                "D1:7",
                "Gina",
                "Wow Jon, same here! Dance is pretty much my go-to for stress relief. Got any fave styles?",
                "How do Jon and Gina both like to destress?",
            ),
            (
                "D1:2",
                "Jon",
                "Hey Gina! Good to see you too. Lost my job as a banker yesterday, so I'm gonna take a shot at starting my own business.",
                "Why did Jon decide to start his dance studio?",
            ),
            (
                "D1:4",
                "Jon",
                "Sorry to hear that! I'm starting a dance studio 'cause I'm passionate about dancing and it'd be great to share it with others.",
                "Why did Jon decide to start his dance studio?",
            ),
            (
                "D1:9",
                "Gina",
                "Yeah, me too! Contemporary dance is so expressive and graceful - it really speaks to me.",
                "What is Gina's favorite style of dance?",
            ),
            (
                "D1:8",
                "Jon",
                "Cool, Gina! I love all dances, but contemporary is my top pick. It's so expressive and powerful! What's your fave?",
                "What is Jon's favorite style of dance?",
            ),
            (
                "D4:2",
                "Gina",
                "Hey Jon! The store's doing great! It's a wild ride. How's the biz?",
                "How is Gina's store doing?",
            ),
            (
                "D7:5",
                "Jon",
                "Yeah, brand identity is key. Make sure yours stands out. Also be sure to build relationships with your customers - let them know you care. And don't forget to stay positive and motivate others. Your energy is contagious!",
                "What advice does Gina give to Jon?",
            ),
            (
                "D8:8",
                "Gina",
                "Thanks! I'm passionate about dance and fashion so combining them lets me show my creativity and share my love with others. Plus, I can add dance-inspired items to my store!",
                "Why did Gina combine her clothing business with dance?",
            ),
            (
                "D9:5",
                "Jon",
                "Yeah, Gina! It's been tough, but I'm living my true self. Dancing makes me so happy, and now I get to share that with other people. Seeing my students get better at it brings me such joy.",
                "What does Jon's dance make him?",
            ),
            (
                "D14:17",
                "Jon",
                "Sure thing, Gina! Your help means a lot to me. I'm not giving up.",
                "What does Jon tell Gina he won't do?",
            ),
            (
                "D15:7",
                "Jon",
                "Thanks, Gina! I'm excited! It's been a wild ride, but I'm feeling good and ready to give it my best.",
                "How does Jon feel about the opening night?",
            ),
        ]
    }

    #[tokio::test]
    #[ignore]
    async fn check_observations_for_missed_evidence() {
        let pipeline = make_pipeline().await;
        let labels = label_maps();

        eprintln!("\n=== Per-sentence observations for missed evidence messages ===\n");

        let mut total_obs = 0usize;
        let mut total_informative = 0usize;
        let mut total_sentences = 0usize;

        for (dia_id, speaker, text, question) in evidence_messages() {
            let results = pipeline.analyze_sentences(text).await.unwrap();

            eprintln!("[{dia_id}] {speaker}: \"{text}\"");
            eprintln!("  Answers: \"{question}\"");
            eprintln!("  Sentences: {}", results.len());

            let mut msg_obs = Vec::new();
            for (i, result) in results.iter().enumerate() {
                let informative = result.sentence_class.is_informative();
                total_sentences += 1;
                if informative {
                    total_informative += 1;
                }

                let sentence_text: String = result.words.join(" ");
                eprintln!(
                    "    S{i}: [{:?} {}] \"{}\"",
                    result.sentence_class,
                    if informative { "EXTRACT" } else { "skip" },
                    &sentence_text[..sentence_text.len().min(80)]
                );

                if informative {
                    let obs = extract_dep_observations(
                        &result.words,
                        &result.pos_indices,
                        &result.dep_arcs,
                        &labels.pos_labels,
                        speaker,
                        &mut SentenceContext::default(),
                    );
                    for o in &obs {
                        eprintln!("      → [{}] \"{}\"", o.subject, o.content);
                    }
                    msg_obs.extend(obs);
                }
            }

            if msg_obs.is_empty() {
                eprintln!("  Observations: NONE");
            } else {
                eprintln!("  Total observations: {}", msg_obs.len());
            }
            total_obs += msg_obs.len();
            eprintln!();
        }

        eprintln!("=== Summary ===");
        eprintln!("  Sentences: {total_sentences}");
        eprintln!("  Informative: {total_informative}");
        eprintln!("  Observations: {total_obs}");
    }
}
