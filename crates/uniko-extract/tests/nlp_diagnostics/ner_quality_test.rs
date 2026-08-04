//! NER quality evaluation on LoCoMo messages.
//!
//! Runs the ONNX NLP model on representative messages and prints
//! extracted entities to assess NER accuracy.
//!
//! Run: cargo nextest run -p uniko-extract --features onnx \
//!   --test ner_quality_test --nocapture --run-ignored all

#[cfg(feature = "onnx")]
mod onnx_tests {
    use uni_db::ModelAliasSpec;
    use uniko_extract::nlp::NlpPipeline;
    use uniko_store::KnowledgeBase;
    use uniko_store::config::UnikoConfig;

    async fn make_pipeline() -> NlpPipeline {
        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
        let catalog_path = workspace_root.join("config/catalog.json");
        let schema_path = workspace_root.join("config/schema.json");
        std::env::set_current_dir(workspace_root).expect("cd to workspace root");

        let config = UnikoConfig {
            catalog_path: Some(catalog_path),
            schema_path: Some(schema_path),
            ..Default::default()
        };
        let tmp_dir = std::env::temp_dir().join(format!("uniko-ner-test-{}", std::process::id()));
        std::fs::create_dir_all(&tmp_dir).ok();

        let kb = KnowledgeBase::open_with_xervo(&tmp_dir, config, Vec::<ModelAliasSpec>::new())
            .await
            .expect("KB with xervo");
        NlpPipeline::try_new(&kb).await.expect("NLP pipeline")
    }

    struct NerCase {
        dia_id: &'static str,
        speaker: &'static str,
        text: &'static str,
        /// Entities we expect the model to find.
        expected: &'static [&'static str],
    }

    fn locomo_messages() -> Vec<NerCase> {
        vec![
            NerCase {
                dia_id: "D1:1",
                speaker: "Gina",
                text: "Hey Jon! Good to see you. What's up? Anything new?",
                expected: &["Jon"],
            },
            NerCase {
                dia_id: "D1:2",
                speaker: "Jon",
                text: "Hey Gina! Good to see you too. Lost my job as a banker yesterday, so I'm gonna take a shot at starting my own business.",
                expected: &["Gina"],
            },
            NerCase {
                dia_id: "D1:3",
                speaker: "Gina",
                text: "Sorry about your job Jon, but starting your own business sounds awesome! Unfortunately, I also lost my job at Door Dash this month. What business are you thinking of starting?",
                expected: &["Jon", "Door Dash"],
            },
            NerCase {
                dia_id: "D1:8",
                speaker: "Jon",
                text: "Cool, Gina! I love all dances, but contemporary is my top pick. It's so expressive and powerful! What's your fave?",
                expected: &["Gina"],
            },
            NerCase {
                dia_id: "D1:9",
                speaker: "Gina",
                text: "Yeah, me too! Contemporary dance is so expressive and graceful - it really speaks to me.",
                expected: &[],
            },
            NerCase {
                dia_id: "D1:19",
                speaker: "Gina",
                text: "Thanks! We just did a contemporary piece called \"Finding Freedom.\" It was really emotional and powerful.",
                expected: &["Finding Freedom"],
            },
            NerCase {
                dia_id: "D2:4",
                speaker: "Jon",
                text: "Hey Gina! Thanks for asking. I'm on the hunt for the ideal spot for my dance studio and it's been quite a journey! I've been looking at different places in LA, even went to Paris last week to check out some studios there.",
                expected: &["Gina", "LA", "Paris"],
            },
            NerCase {
                dia_id: "D3:2",
                speaker: "Gina",
                text: "Hi Jon! So happy you're pushing forward with dancing! Inspiring. I emailed some wholesalers and one replied and said yes today! I'm over the moon because now I can expand my clothing store.",
                expected: &["Jon"],
            },
            NerCase {
                dia_id: "D4:2",
                speaker: "Gina",
                text: "Hey Jon! The store's doing great! It's a wild ride. How's the biz?",
                expected: &["Jon"],
            },
            NerCase {
                dia_id: "D6:8",
                speaker: "Gina",
                text: "Thanks! I'm passionate about fashion trends and finding unique pieces. Plus, I wanted to blend my love for dance and fashion, so it was a perfect match!",
                expected: &[],
            },
            NerCase {
                dia_id: "D7:5",
                speaker: "Jon",
                text: "Yeah, brand identity is key. Make sure yours stands out. Also be sure to build relationships with your customers - let them know you care. And don't forget to stay positive and motivate others.",
                expected: &[],
            },
            NerCase {
                dia_id: "D8:8",
                speaker: "Gina",
                text: "Thanks! I'm passionate about dance and fashion so combining them lets me show my creativity and share my love with others. Plus, I can add dance-inspired items to my store!",
                expected: &[],
            },
            NerCase {
                dia_id: "D9:5",
                speaker: "Jon",
                text: "Yeah, Gina! It's been tough, but I'm living my true self. Dancing makes me so happy, and now I get to share that with other people. Seeing my students get better at it brings me such joy.",
                expected: &["Gina"],
            },
            NerCase {
                dia_id: "D14:17",
                speaker: "Jon",
                text: "Sure thing, Gina! Your help means a lot to me. I'm not giving up.",
                expected: &["Gina"],
            },
            NerCase {
                dia_id: "D15:1",
                speaker: "Jon",
                text: "Hey Gina, hope you're doing great! Still working on my biz. Took a short trip last week to Rome to clear my mind a little.",
                expected: &["Gina", "Rome"],
            },
            NerCase {
                dia_id: "D15:7",
                speaker: "Jon",
                text: "Thanks, Gina! I'm excited! It's been a wild ride, but I'm feeling good and ready to give it my best.",
                expected: &["Gina"],
            },
        ]
    }

    #[tokio::test]
    #[ignore]
    async fn evaluate_ner_on_locomo() {
        let pipeline = make_pipeline().await;
        let cases = locomo_messages();

        let mut total_expected = 0usize;
        let mut total_found = 0usize;
        let mut total_extra = 0usize;

        for case in &cases {
            let result = pipeline.analyze(case.text).await.unwrap();

            let found: Vec<String> = result
                .entities
                .iter()
                .map(|e| format!("{} [{:?}]", e.text, e.entity_type))
                .collect();
            let found_texts: Vec<&str> = result.entities.iter().map(|e| e.text.as_str()).collect();

            // Check expected
            let mut hits = Vec::new();
            let mut misses = Vec::new();
            for &exp in case.expected {
                if found_texts
                    .iter()
                    .any(|f| f.to_lowercase().contains(&exp.to_lowercase()))
                {
                    hits.push(exp);
                    total_found += 1;
                } else {
                    misses.push(exp);
                }
                total_expected += 1;
            }

            // Extra entities (not in expected)
            let extras: Vec<&str> = found_texts
                .iter()
                .filter(|f| {
                    !case
                        .expected
                        .iter()
                        .any(|e| f.to_lowercase().contains(&e.to_lowercase()))
                })
                .copied()
                .collect();
            total_extra += extras.len();

            // Print
            let status = if misses.is_empty() && extras.is_empty() {
                "OK"
            } else if misses.is_empty() {
                "EXTRA"
            } else {
                "MISS"
            };

            eprintln!(
                "[{:5}] {} {}: \"{}\"",
                status,
                case.dia_id,
                case.speaker,
                &case.text[..case.text.len().min(80)]
            );
            if !found.is_empty() {
                eprintln!("         Found: {}", found.join(", "));
            }
            if !misses.is_empty() {
                eprintln!("         MISSED: {:?}", misses);
            }
            if !extras.is_empty() {
                eprintln!("         EXTRA:  {:?}", extras);
            }
        }

        let recall = if total_expected > 0 {
            total_found as f64 / total_expected as f64 * 100.0
        } else {
            100.0
        };
        eprintln!("\n=== NER Summary ===");
        eprintln!("  Expected entities: {total_expected}");
        eprintln!("  Found: {total_found}");
        eprintln!("  Recall: {recall:.1}%");
        eprintln!("  Extra (false positives): {total_extra}");
    }
}
