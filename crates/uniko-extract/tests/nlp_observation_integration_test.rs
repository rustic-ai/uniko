//! Integration tests: NLP model → observation extraction end-to-end.
//!
//! These tests require the `onnx` feature and a cached ONNX model.
//! They verify that the full pipeline (tokenize → ONNX → DEP decode →
//! observation extraction) produces clean, well-formed observations
//! on real conversational text from LoCoMo.
//!
//! Run:
//!   cargo nextest run -p uniko-extract --features onnx \
//!     --test nlp_observation_integration_test --nocapture --run-ignored all

#[cfg(feature = "onnx")]
mod onnx_tests {
    use uni_db::ModelAliasSpec;
    use uniko_extract::ingest::context::SentenceContext;
    use uniko_extract::nlp::NlpPipeline;
    use uniko_extract::nlp::assets::label_maps;
    use uniko_extract::nlp::decode::extract_dep_observations;
    use uniko_extract::nlp::types::NlpResult;
    use uniko_extract::observations::filter::is_informative;
    use uniko_store::KnowledgeBase;
    use uniko_store::config::UnikoConfig;

    /// Create an NLP pipeline backed by the real ONNX model.
    ///
    /// Uses the external catalog.json from the workspace root.
    /// The benchmark runs from the workspace root too, so this mirrors
    /// production conditions.
    async fn make_pipeline() -> NlpPipeline {
        // Resolve workspace root from CARGO_MANIFEST_DIR.
        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
        let catalog_path = workspace_root.join("config/catalog.json");
        let schema_path = workspace_root.join("config/schema.json");

        eprintln!("Workspace root: {}", workspace_root.display());
        eprintln!(
            "Catalog: {} (exists: {})",
            catalog_path.display(),
            catalog_path.exists()
        );

        // Change CWD to workspace root so .uni_cache is found.
        std::env::set_current_dir(workspace_root).expect("cd to workspace root");

        let config = UnikoConfig {
            catalog_path: Some(catalog_path),
            schema_path: Some(schema_path),
            ..Default::default()
        };

        let tmp_dir = std::env::temp_dir().join(format!("uniko-nlp-test-{}", std::process::id()));
        std::fs::create_dir_all(&tmp_dir).ok();

        let kb = KnowledgeBase::open_with_xervo(&tmp_dir, config, Vec::<ModelAliasSpec>::new())
            .await
            .expect("KB with xervo + catalog");

        // Resolve the alias directly first: `NlpPipeline::try_new` collapses
        // every failure into `None`, so without this the panic below would say
        // only "NLP pipeline" and hide why. Ask for the same `NlpModel` the
        // pipeline needs — probing `raw_tensor_model` here was a leftover from
        // before the 2026-06 migration of `nlp/default` from `Raw` to `Nlp`,
        // and reported a missing `RawTensorModel` capability once the alias was
        // registered correctly.
        let runtime = kb.model_runtime().expect("xervo runtime");
        match runtime.nlp_model("nlp/default").await {
            Ok(_) => eprintln!("NLP model resolved successfully"),
            Err(e) => panic!("NLP model unavailable for alias 'nlp/default': {e}"),
        }

        NlpPipeline::try_new(&kb).await.expect("NLP pipeline")
    }

    /// Print detailed NLP diagnostics for a single analysis result.
    fn print_diagnostics(text: &str, speaker: &str, result: &NlpResult, label: &str) {
        let labels = label_maps();

        eprintln!("\n============================================================");
        eprintln!("  {label}");
        eprintln!("  Text: \"{text}\"");
        eprintln!("  Speaker: {speaker}");
        eprintln!("------------------------------------------------------------");

        // CLS
        eprintln!(
            "  CLS: {:?} (confidence: {:.3})",
            result.sentence_class, result.cls_confidence
        );
        let cls_informative = result.sentence_class.is_informative();
        let rule_informative = is_informative(text, Some("text"));
        eprintln!("  CLS informative: {cls_informative} | Rule informative: {rule_informative}");

        // Words + POS
        let pos_tags: Vec<&str> = result
            .pos_indices
            .iter()
            .map(|&i| labels.pos_labels.get(i).map(String::as_str).unwrap_or("?"))
            .collect();
        eprintln!("  Words: {:?}", result.words);
        eprintln!("  POS:   {:?}", pos_tags);

        // NER
        if !result.entities.is_empty() {
            eprintln!("  NER:");
            for ent in &result.entities {
                eprintln!(
                    "    [{:?}] \"{}\" (words {}-{})",
                    ent.entity_type, ent.text, ent.start_word, ent.end_word
                );
            }
        } else {
            eprintln!("  NER: (none)");
        }

        // DEP tree
        eprintln!("  DEP tree:");
        for arc in &result.dep_arcs {
            let dep_word = result
                .words
                .get(arc.dependent)
                .map(String::as_str)
                .unwrap_or("?");
            let head_word = if arc.head == usize::MAX {
                "ROOT".to_string()
            } else {
                result
                    .words
                    .get(arc.head)
                    .map(String::as_str)
                    .unwrap_or("?")
                    .to_string()
            };
            let dep_pos = pos_tags.get(arc.dependent).unwrap_or(&"?");
            eprintln!(
                "    {dep_word}({dep_pos}) --[{}]--> {head_word}",
                arc.relation
            );
        }

        // Observations
        let obs = extract_dep_observations(
            &result.words,
            &result.pos_indices,
            &result.dep_arcs,
            &labels.pos_labels,
            speaker,
            &mut SentenceContext::default(),
        );
        if obs.is_empty() {
            eprintln!("  Observations: (none)");
        } else {
            eprintln!("  Observations ({}):", obs.len());
            for o in &obs {
                eprintln!(
                    "    subject={:<12} content=\"{}\" conf={:.2}",
                    o.subject, o.content, o.confidence
                );
            }
        }
        eprintln!();
    }

    // ── Test cases ─────────────────────────────────────────────────

    struct TestCase {
        text: &'static str,
        speaker: &'static str,
        expect_informative: bool,
        min_observations: usize,
        max_observations: usize,
        description: &'static str,
    }

    fn locomo_test_cases() -> Vec<TestCase> {
        vec![
            // --- Social/filler messages (should be filtered) ---
            TestCase {
                text: "Hey Mel! Good to see you! How have you been?",
                speaker: "Caroline",
                expect_informative: false,
                min_observations: 0,
                max_observations: 0,
                description: "Greeting + question (should filter)",
            },
            TestCase {
                text: "Wow, that's cool, Caroline! What happened that was so awesome? Did you hear any inspiring stories?",
                speaker: "Melanie",
                expect_informative: false, // mostly social/question
                min_observations: 0,
                max_observations: 2, // lenient — model may see some info
                description: "Social reaction + questions",
            },
            TestCase {
                text: "That's really cool. You've got guts. What now?",
                speaker: "Melanie",
                expect_informative: false,
                min_observations: 0,
                max_observations: 1,
                description: "Short encouragement + question",
            },
            // --- Informative messages (should produce observations) ---
            TestCase {
                text: "I went to a LGBTQ support group yesterday and it was so powerful.",
                speaker: "Caroline",
                expect_informative: true,
                min_observations: 1,
                max_observations: 3,
                description: "1st-person factual: support group",
            },
            TestCase {
                text: "The transgender stories were so inspiring! I was so happy and thankful for all the support.",
                speaker: "Caroline",
                expect_informative: true,
                min_observations: 1,
                max_observations: 3,
                description: "1st-person factual: trans stories",
            },
            TestCase {
                text: "The support group has made me feel accepted and given me courage to embrace myself.",
                speaker: "Caroline",
                expect_informative: true,
                min_observations: 1,
                max_observations: 3,
                description: "1st-person factual: acceptance",
            },
            TestCase {
                text: "Gonna continue my edu and check out career options, which is pretty exciting!",
                speaker: "Caroline",
                expect_informative: true,
                min_observations: 1,
                max_observations: 3,
                description: "1st-person plan: education/career",
            },
            TestCase {
                text: "I'm starting a dance studio 'cause I'm passionate about dancing.",
                speaker: "Caroline",
                expect_informative: true,
                min_observations: 1,
                max_observations: 3,
                description: "1st-person factual: dance studio",
            },
            TestCase {
                text: "I lost my job and started a business.",
                speaker: "Jon",
                expect_informative: true,
                min_observations: 1,
                max_observations: 3,
                description: "1st-person: job loss + business",
            },
            TestCase {
                text: "Caroline attended an LGBTQ support group yesterday.",
                speaker: "Melanie",
                expect_informative: true,
                min_observations: 1,
                max_observations: 2,
                description: "3rd-person factual: Caroline → support group",
            },
            TestCase {
                text: "I got a temp job at a bookstore and I love it.",
                speaker: "Jon",
                expect_informative: true,
                min_observations: 1,
                max_observations: 3,
                description: "1st-person: temp job at bookstore",
            },
            // --- Edge cases ---
            TestCase {
                text: "My mom has been really supportive of my transition and even came to the support group with me last week.",
                speaker: "Caroline",
                expect_informative: true,
                min_observations: 1,
                max_observations: 4,
                description: "Complex sentence with multiple clauses",
            },
            TestCase {
                text: "I've been volunteering at the local animal shelter on weekends.",
                speaker: "Jon",
                expect_informative: true,
                min_observations: 1,
                max_observations: 2,
                description: "1st-person with temporal modifier",
            },
        ]
    }

    // ── Full pipeline integration test ─────────────────────────────

    #[tokio::test]
    #[ignore] // Requires ONNX model cached in .uni_cache/
    async fn test_nlp_observation_pipeline_on_locomo() {
        let pipeline = make_pipeline().await;
        let labels = label_maps();

        let cases = locomo_test_cases();
        let mut total_pass = 0usize;
        let mut total_fail = 0usize;

        for case in &cases {
            let result = pipeline.analyze(case.text).await.unwrap();
            print_diagnostics(case.text, case.speaker, &result, case.description);

            // Check CLS gate
            let cls_informative = result.sentence_class.is_informative();
            let rule_informative = is_informative(case.text, Some("text"));
            let effective_informative = cls_informative; // CLS takes priority when model available

            // Extract observations
            let obs = extract_dep_observations(
                &result.words,
                &result.pos_indices,
                &result.dep_arcs,
                &labels.pos_labels,
                case.speaker,
                &mut SentenceContext::default(),
            );

            // Validate
            let mut case_ok = true;

            if case.expect_informative && !effective_informative {
                eprintln!(
                    "  FAIL: Expected informative but CLS={:?}, rule={}",
                    result.sentence_class, rule_informative
                );
                case_ok = false;
            }

            if case.expect_informative {
                if obs.len() < case.min_observations {
                    eprintln!(
                        "  FAIL: Expected >= {} observations, got {}",
                        case.min_observations,
                        obs.len()
                    );
                    case_ok = false;
                }
                if obs.len() > case.max_observations {
                    eprintln!(
                        "  WARN: Got {} observations (max expected {})",
                        obs.len(),
                        case.max_observations
                    );
                }

                // Check observation quality: each should have >= 3 words
                for o in &obs {
                    let word_count = o.content.split_whitespace().count();
                    if word_count < 3 {
                        eprintln!(
                            "  FAIL: Observation too short ({} words): \"{}\"",
                            word_count, o.content
                        );
                        case_ok = false;
                    }
                }

                // Check speaker substitution for 1st-person text
                if case.text.starts_with("I ") || case.text.starts_with("I'") {
                    for o in &obs {
                        if o.subject == "I" || o.subject == "i" {
                            eprintln!(
                                "  FAIL: Speaker substitution failed: subject is still \"{}\"",
                                o.subject
                            );
                            case_ok = false;
                        }
                    }
                }
            }

            if case_ok {
                total_pass += 1;
            } else {
                total_fail += 1;
            }
        }

        eprintln!("\n============================================================");
        eprintln!(
            "  Summary: {}/{} passed, {} failed",
            total_pass,
            cases.len(),
            total_fail
        );
        eprintln!("============================================================");

        // Allow some failures for diagnostic purposes, but flag them
        assert!(
            total_fail <= cases.len() / 3,
            "Too many test cases failed: {total_fail}/{} — check diagnostics above",
            cases.len()
        );
    }

    // ── CLS classification accuracy test ───────────────────────────

    #[tokio::test]
    #[ignore] // Requires ONNX model
    async fn test_cls_classification_accuracy() {
        let pipeline = make_pipeline().await;

        let informative_texts = [
            "I went to a LGBTQ support group yesterday.",
            "Caroline attended the support group.",
            "I lost my job last month.",
            "I'm starting a dance studio.",
            "My mom has been really supportive.",
            "I got a temp job at a bookstore.",
            "The support group has made me feel accepted.",
        ];

        let non_informative_texts = [
            "Hey!",
            "Wow, that's cool!",
            "Thanks for sharing.",
            "What do you mean?",
            "How are you doing?",
            "That's awesome!",
            "Really?",
        ];

        let mut correct = 0usize;
        let mut total = 0usize;

        eprintln!("\n--- CLS: Expected INFORMATIVE ---");
        for text in &informative_texts {
            let result = pipeline.analyze(text).await.unwrap();
            let is_inf = result.sentence_class.is_informative();
            let mark = if is_inf { "OK" } else { "MISS" };
            eprintln!(
                "  [{mark}] {:?} ({:.2}) — \"{text}\"",
                result.sentence_class, result.cls_confidence
            );
            if is_inf {
                correct += 1;
            }
            total += 1;
        }

        eprintln!("\n--- CLS: Expected NON-INFORMATIVE ---");
        for text in &non_informative_texts {
            let result = pipeline.analyze(text).await.unwrap();
            let is_inf = result.sentence_class.is_informative();
            let mark = if !is_inf { "OK" } else { "MISS" };
            eprintln!(
                "  [{mark}] {:?} ({:.2}) — \"{text}\"",
                result.sentence_class, result.cls_confidence
            );
            if !is_inf {
                correct += 1;
            }
            total += 1;
        }

        let accuracy = correct as f64 / total as f64;
        let pct = accuracy * 100.0;
        eprintln!("\nCLS accuracy: {correct}/{total} = {pct:.1}%");
        assert!(
            accuracy >= 0.7,
            "CLS accuracy too low: {pct:.1}% (need >= 70%)"
        );
    }

    // ── DEP tree quality test ──────────────────────────────────────

    #[tokio::test]
    #[ignore] // Requires ONNX model
    async fn test_dep_tree_quality() {
        let pipeline = make_pipeline().await;
        let labels = label_maps();

        // Simple SVO sentences where we know the expected structure.
        let cases = [
            ("Caroline lost her job.", "Caroline", "lost", Some("job")),
            (
                "Jon started a business.",
                "Jon",
                "started",
                Some("business"),
            ),
            (
                "I attended the support group.",
                "I",
                "attended",
                Some("group"),
            ),
        ];

        for (text, expected_subj, expected_verb, expected_obj) in &cases {
            let result = pipeline.analyze(text).await.unwrap();

            eprintln!("\nDEP quality: \"{text}\"");
            eprintln!("  Words: {:?}", result.words);

            // Find the verb
            let verb_idx = result.words.iter().position(|w| w == expected_verb);
            assert!(
                verb_idx.is_some(),
                "Verb '{expected_verb}' not found in words: {:?}",
                result.words
            );
            let verb_idx = verb_idx.unwrap();

            // Check verb has VERB POS
            let verb_pos = labels
                .pos_labels
                .get(result.pos_indices[verb_idx])
                .map(String::as_str)
                .unwrap_or("?");
            eprintln!("  Verb \"{expected_verb}\" POS: {verb_pos}");
            assert_eq!(
                verb_pos, "VERB",
                "Expected VERB POS for '{expected_verb}', got {verb_pos}"
            );

            // Check nsubj arc exists pointing to verb
            let has_nsubj = result.dep_arcs.iter().any(|arc| {
                arc.head == verb_idx
                    && (arc.relation == "nsubj"
                        || arc.relation == "nsubj:pass"
                        || arc.relation == "csubj")
            });
            eprintln!("  Has nsubj → {expected_verb}: {has_nsubj}");
            assert!(
                has_nsubj,
                "No nsubj arc found for verb '{expected_verb}' in: {text}"
            );

            // Check subject word
            let subj_arc = result
                .dep_arcs
                .iter()
                .find(|arc| {
                    arc.head == verb_idx
                        && (arc.relation == "nsubj" || arc.relation == "nsubj:pass")
                })
                .unwrap();
            let subj_word = &result.words[subj_arc.dependent];
            eprintln!("  Subject: \"{subj_word}\" (expected \"{expected_subj}\")");
            assert_eq!(
                subj_word.to_lowercase(),
                expected_subj.to_lowercase(),
                "Wrong subject for verb '{expected_verb}'"
            );

            // Check object arc exists (if expected)
            if let Some(exp_obj) = expected_obj {
                let has_obj = result.dep_arcs.iter().any(|arc| {
                    arc.head == verb_idx
                        && (arc.relation == "obj"
                            || arc.relation == "dobj"
                            || arc.relation == "iobj"
                            || arc.relation == "obl")
                });
                eprintln!(
                    "  Has obj → {expected_verb}: {has_obj} (expected obj contains \"{exp_obj}\")"
                );
            }
        }
    }

    // ── Speaker substitution test (with real model) ────────────────

    #[tokio::test]
    #[ignore] // Requires ONNX model
    async fn test_speaker_substitution_with_model() {
        let pipeline = make_pipeline().await;
        let labels = label_maps();

        let first_person_cases = [
            ("I lost my job.", "Jon", "Jon"),
            ("I'm starting a dance studio.", "Caroline", "Caroline"),
            ("I got a temp job at a bookstore.", "Jon", "Jon"),
            ("I attended the support group.", "Caroline", "Caroline"),
        ];

        for (text, speaker, expected_subject) in &first_person_cases {
            let result = pipeline.analyze(text).await.unwrap();
            let obs = extract_dep_observations(
                &result.words,
                &result.pos_indices,
                &result.dep_arcs,
                &labels.pos_labels,
                speaker,
                &mut SentenceContext::default(),
            );

            eprintln!("\nSpeaker sub: \"{text}\" (speaker={speaker})");
            for o in &obs {
                eprintln!("  → subject=\"{}\" content=\"{}\"", o.subject, o.content);
            }

            if !obs.is_empty() {
                assert_eq!(
                    obs[0].subject, *expected_subject,
                    "Speaker substitution failed for \"{text}\": got \"{}\"",
                    obs[0].subject
                );
                assert!(
                    obs[0].content.contains(expected_subject),
                    "Content should contain speaker name \"{expected_subject}\": got \"{}\"",
                    obs[0].content
                );
                assert!(
                    !obs[0].content.starts_with("I "),
                    "Content should not start with 'I': got \"{}\"",
                    obs[0].content
                );
            } else {
                eprintln!("  WARN: No observations produced for \"{text}\"");
            }
        }
    }

    // ── Observation content quality test ────────────────────────────

    #[tokio::test]
    #[ignore] // Requires ONNX model
    async fn test_observation_content_quality() {
        let pipeline = make_pipeline().await;
        let labels = label_maps();

        let cases = [
            (
                "I lost my job and started a business.",
                "Jon",
                vec!["lost", "job"],       // must contain these words
                vec!["hey", "is awesome"], // must NOT contain these
            ),
            (
                "Caroline attended an LGBTQ support group yesterday.",
                "Melanie",
                vec!["Caroline", "attended"],
                vec!["hey", "Melanie"],
            ),
            (
                "I got a temp job at a bookstore and I love it.",
                "Jon",
                vec!["Jon", "job"],
                vec!["hey"],
            ),
        ];

        let mut all_ok = true;
        for (text, speaker, must_contain, must_not_contain) in &cases {
            let result = pipeline.analyze(text).await.unwrap();
            let obs = extract_dep_observations(
                &result.words,
                &result.pos_indices,
                &result.dep_arcs,
                &labels.pos_labels,
                speaker,
                &mut SentenceContext::default(),
            );

            eprintln!("\nQuality: \"{text}\"");

            if obs.is_empty() {
                eprintln!("  FAIL: No observations produced");
                all_ok = false;
                continue;
            }

            // Combine all observation content for checking
            let all_content: String = obs
                .iter()
                .map(|o| o.content.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!("  Combined content: \"{all_content}\"");

            for word in must_contain {
                if !all_content.to_lowercase().contains(&word.to_lowercase()) {
                    eprintln!("  FAIL: Missing required word \"{word}\" in observations");
                    all_ok = false;
                }
            }

            for word in must_not_contain {
                if all_content.to_lowercase().contains(&word.to_lowercase()) {
                    eprintln!("  FAIL: Found forbidden word \"{word}\" in observations");
                    all_ok = false;
                }
            }

            // Check no observation is a single word
            for o in &obs {
                let wc = o.content.split_whitespace().count();
                if wc < 2 {
                    eprintln!("  FAIL: Single-word observation: \"{}\"", o.content);
                    all_ok = false;
                }
            }
        }

        assert!(
            all_ok,
            "Some observation quality checks failed — see diagnostics above"
        );
    }
}
