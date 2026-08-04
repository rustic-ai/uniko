//! Run the NLP pipeline on three specific conv-26 sentences that
//! Phase B's `oblique_state_modifier` and `appositional_identity` are
//! supposed to capture, and print the resulting DEP arcs + extracted
//! observations. Lets us see exactly what the parser emits and
//! whether the rule patterns can possibly fire.
//!
//! Run: cargo nextest run -p uniko-extract --features onnx \
//!     --test dump_phase_b_target_arcs --nocapture --run-ignored all

#[cfg(feature = "onnx")]
mod onnx_dump {
    use uni_db::ModelAliasSpec;
    use uniko_extract::ingest::context::SentenceContext;
    use uniko_extract::nlp::NlpPipeline;
    use uniko_extract::nlp::assets::label_maps;
    use uniko_extract::observations::rules_engine::{extract_with_rules, load_bundled_rules};
    use uniko_store::KnowledgeBase;
    use uniko_store::config::UnikoConfig;

    const TARGETS: &[&str] = &[
        // D2:14 — single parent
        "It'll be tough as a single parent, but I'm up for the challenge!",
        // D3:13 — tough breakup
        "Their love and help have been so important especially after that tough breakup.",
        // D4:3 — Sweden
        "This necklace is super special to me - a gift from my grandma in my home country, Sweden.",
    ];

    #[ignore]
    #[tokio::test]
    async fn dump_target_arcs() {
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
        let tmp = std::env::temp_dir().join(format!("uniko-phase-b-dump-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).ok();
        let kb = KnowledgeBase::open_with_xervo(&tmp, config, Vec::<ModelAliasSpec>::new())
            .await
            .expect("KB");
        let pipeline = NlpPipeline::try_new(&kb).await.expect("pipeline");
        let labels = label_maps();
        let rules = load_bundled_rules();

        for (i, text) in TARGETS.iter().enumerate() {
            println!("\n=== TARGET {i}: {text} ===");
            let results = pipeline.analyze_sentences(text).await.expect("analyze");
            for (s_idx, r) in results.iter().enumerate() {
                println!(
                    "  S{s_idx}: cls={:?}  words={:?}",
                    r.sentence_class, r.words
                );
                let probs: Vec<(String, f32)> = labels
                    .cls_labels
                    .iter()
                    .cloned()
                    .zip(r.cls_probs.iter().copied())
                    .collect();
                let mut sorted = probs.clone();
                sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                println!(
                    "       PROBS: {}",
                    sorted
                        .iter()
                        .map(|(n, p)| format!("{n}={p:.2}"))
                        .collect::<Vec<_>>()
                        .join(" ")
                );
                let pos: Vec<&str> = r
                    .pos_indices
                    .iter()
                    .map(|&i| labels.pos_labels.get(i).map(String::as_str).unwrap_or(""))
                    .collect();
                println!("       POS: {pos:?}");
                println!("       ARCS:");
                for a in &r.dep_arcs {
                    let dep = r.words.get(a.dependent).map(String::as_str).unwrap_or("?");
                    let head = if a.head == usize::MAX {
                        "ROOT".to_string()
                    } else {
                        r.words.get(a.head).cloned().unwrap_or_else(|| "?".into())
                    };
                    println!("         {dep} -[{}]-> {head}", a.relation);
                }
                let mut ctx = SentenceContext::new("Caroline", vec!["Melanie".into()]);
                let obs = extract_with_rules(
                    rules,
                    &r.words,
                    &r.pos_indices,
                    &r.dep_arcs,
                    &labels.pos_labels,
                    &r.srl_frames,
                    "Caroline",
                    &mut ctx,
                );
                println!(
                    "       OBS: {:?}",
                    obs.iter().map(|o| &o.content).collect::<Vec<_>>()
                );
            }
        }
    }
}
