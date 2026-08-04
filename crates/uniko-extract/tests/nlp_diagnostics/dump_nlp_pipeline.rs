//! Dump full NLP pipeline output for all LoCoMo conv-30 messages.
//!
//! For each message: split into sentences, run ONNX model, log
//! sentence text, CLS classification, POS tags, DEP tree, whether
//! CLS filtered it, and derived observations.
//!
//! Run: cargo nextest run -p uniko-extract --features onnx \
//!   --test dump_nlp_pipeline --nocapture --run-ignored all

#[cfg(feature = "onnx")]
mod onnx_dump {
    use std::io::Write;

    use uni_db::ModelAliasSpec;
    use uniko_extract::ingest::context::SentenceContext;
    use uniko_extract::nlp::NlpPipeline;
    use uniko_extract::nlp::assets::label_maps;
    use uniko_extract::nlp::decode::extract_dep_observations;
    use uniko_store::KnowledgeBase;
    use uniko_store::config::UnikoConfig;

    async fn make_pipeline() -> NlpPipeline {
        let ws = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        std::env::set_current_dir(&ws).expect("cd");
        let config = UnikoConfig {
            catalog_path: Some(ws.join("config/catalog.json")),
            schema_path: Some(ws.join("config/schema.json")),
            ..Default::default()
        };
        let tmp = std::env::temp_dir().join(format!("uniko-nlp-dump-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).ok();
        let kb = KnowledgeBase::open_with_xervo(&tmp, config, Vec::<ModelAliasSpec>::new())
            .await
            .expect("KB");
        NlpPipeline::try_new(&kb).await.expect("pipeline")
    }

    fn load_messages() -> Vec<(String, String, String, String)> {
        // (dia_id, speaker, session_id, text)
        let ws = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let data: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(ws.join("data/locomo10.json")).unwrap())
                .unwrap();

        let conv = data
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["sample_id"] == "conv-30")
            .unwrap();

        let mut messages = Vec::new();
        let conversation = conv["conversation"].as_object().unwrap();
        let mut keys: Vec<&String> = conversation.keys().collect();
        keys.sort_by(|a, b| {
            let na: usize = a.replace("session_", "").parse().unwrap_or(999);
            let nb: usize = b.replace("session_", "").parse().unwrap_or(999);
            na.cmp(&nb)
        });

        for key in keys {
            if !key.starts_with("session") {
                continue;
            }
            let val = &conversation[key];
            if let Some(arr) = val.as_array() {
                for t in arr {
                    if let (Some(dia_id), Some(speaker), Some(text)) = (
                        t["dia_id"].as_str(),
                        t["speaker"].as_str(),
                        t["text"].as_str(),
                    ) {
                        messages.push((
                            dia_id.to_string(),
                            speaker.to_string(),
                            key.to_string(),
                            text.to_string(),
                        ));
                    }
                }
            }
        }
        messages
    }

    #[tokio::test]
    #[ignore]
    async fn dump_full_nlp_pipeline() {
        let pipeline = make_pipeline().await;
        let labels = label_maps();
        let messages = load_messages();

        let ws = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let out_path = ws.join("data/nlp_pipeline_dump.txt");
        let mut out = std::fs::File::create(&out_path).unwrap();

        eprintln!("Processing {} messages...", messages.len());

        let mut total_sentences = 0usize;
        let mut total_informative = 0usize;
        let mut total_filtered = 0usize;
        let mut total_observations = 0usize;
        let mut total_pronoun_resolved = 0usize;

        // Track SentenceContext per session for pronoun resolution.
        let mut current_session = String::new();
        let mut sent_ctx = SentenceContext::default();

        // Build speaker lists for "you" resolution.
        let all_speakers: Vec<String> = messages
            .iter()
            .map(|(_, s, _, _)| s.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        for (msg_idx, (dia_id, speaker, session_id, text)) in messages.iter().enumerate() {
            // Reset context at session boundary.
            if *session_id != current_session {
                current_session = session_id.clone();
                let others: Vec<String> = all_speakers
                    .iter()
                    .filter(|s| s.as_str() != speaker.as_str())
                    .cloned()
                    .collect();
                sent_ctx = SentenceContext::new(speaker, others);
            } else {
                // Update speaker within session.
                sent_ctx.speaker = speaker.clone();
                sent_ctx.other_speakers = all_speakers
                    .iter()
                    .filter(|s| s.as_str() != speaker.as_str())
                    .cloned()
                    .collect();
            }

            writeln!(
                out,
                "========================================================================"
            )
            .unwrap();
            writeln!(out, "[{dia_id}] {session_id} | {speaker}: \"{text}\"").unwrap();
            writeln!(
                out,
                "  context: subj={:?} obj={:?}",
                sent_ctx.last_noun_subject, sent_ctx.last_noun_object
            )
            .unwrap();
            writeln!(
                out,
                "------------------------------------------------------------------------"
            )
            .unwrap();

            let results = match pipeline.analyze_sentences(text).await {
                Ok(r) => r,
                Err(e) => {
                    writeln!(out, "  ERROR: {e}").unwrap();
                    continue;
                }
            };

            for (s_idx, result) in results.iter().enumerate() {
                total_sentences += 1;
                let sentence_text: String = result.words.join(" ");
                let informative = result.sentence_class.is_informative();

                if informative {
                    total_informative += 1;
                } else {
                    total_filtered += 1;
                }

                // POS tags
                let pos_tags: Vec<&str> = result
                    .pos_indices
                    .iter()
                    .map(|&i| labels.pos_labels.get(i).map(String::as_str).unwrap_or("?"))
                    .collect();

                // CLS
                writeln!(
                    out,
                    "  S{s_idx}: [{:?} {:.2}] [{}] \"{}\"",
                    result.sentence_class,
                    result.cls_confidence,
                    if informative { "EXTRACT" } else { "SKIP" },
                    sentence_text
                )
                .unwrap();

                // POS
                let pos_str: String = result
                    .words
                    .iter()
                    .zip(pos_tags.iter())
                    .map(|(w, p)| format!("{w}/{p}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                writeln!(out, "       POS: {pos_str}").unwrap();

                // DEP tree
                write!(out, "       DEP:").unwrap();
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
                    write!(out, " {dep_word}-[{}]->{head_word}", arc.relation).unwrap();
                }
                writeln!(out).unwrap();

                // NER
                if !result.entities.is_empty() {
                    let ner_str: String = result
                        .entities
                        .iter()
                        .map(|e| format!("[{:?}]{}", e.entity_type, e.text))
                        .collect::<Vec<_>>()
                        .join(", ");
                    writeln!(out, "       NER: {ner_str}").unwrap();
                }

                if informative {
                    // Extract observations with pronoun resolution.
                    let obs = extract_dep_observations(
                        &result.words,
                        &result.pos_indices,
                        &result.dep_arcs,
                        &labels.pos_labels,
                        speaker,
                        &mut sent_ctx,
                    );
                    if obs.is_empty() {
                        writeln!(out, "       OBS: (none)").unwrap();
                    } else {
                        for o in &obs {
                            total_observations += 1;
                            // Check if subject was resolved from a pronoun.
                            let resolved = o.subject != *speaker
                                && !result.words.iter().any(|w| w == &o.subject);
                            if resolved {
                                total_pronoun_resolved += 1;
                            }
                            let marker = if resolved { " [resolved]" } else { "" };
                            writeln!(
                                out,
                                "       OBS: [{}]{} \"{}\"",
                                o.subject, marker, o.content
                            )
                            .unwrap();
                        }
                    }
                } else {
                    // Still update context from skipped sentences.
                    uniko_extract::nlp::decode::update_sentence_context(
                        &mut sent_ctx,
                        &result.words,
                        &result.pos_indices,
                        &result.dep_arcs,
                        &labels.pos_labels,
                    );
                }
            }

            if (msg_idx + 1) % 50 == 0 {
                eprintln!("  {}/{} messages processed...", msg_idx + 1, messages.len());
            }
        }

        writeln!(
            out,
            "\n========================================================================"
        )
        .unwrap();
        writeln!(out, "SUMMARY").unwrap();
        writeln!(out, "  Messages: {}", messages.len()).unwrap();
        writeln!(out, "  Sentences: {total_sentences}").unwrap();
        writeln!(out, "  Informative (EXTRACT): {total_informative}").unwrap();
        writeln!(out, "  Filtered (SKIP): {total_filtered}").unwrap();
        writeln!(out, "  Observations: {total_observations}").unwrap();
        writeln!(out, "  Pronoun resolved: {total_pronoun_resolved}").unwrap();

        eprintln!("\nDone. Written to {}", out_path.display());
        eprintln!("  Messages: {}", messages.len());
        eprintln!("  Sentences: {total_sentences}");
        eprintln!("  Informative: {total_informative}");
        eprintln!("  Filtered: {total_filtered}");
        eprintln!("  Observations: {total_observations}");
        eprintln!("  Pronoun resolved: {total_pronoun_resolved}");
    }
}
