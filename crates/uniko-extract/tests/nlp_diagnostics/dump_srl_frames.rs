//! Manual sanity check: run SRL on a handful of fixed sentences and
//! pretty-print every frame the cascade emits. Useful for verifying
//! the per-verb re-forward path before we plumb SRL frames into
//! observation extraction (Phase B).
//!
//! Also reports a coarse latency micro-benchmark — single-pass
//! (`analyze`) vs SRL-enabled (`analyze` with `nlp_srl_enabled=true`)
//! on the same sentences. Helps decide whether the per-verb cost is
//! acceptable.
//!
//! Run: cargo test -p uniko-extract --features onnx \
//!     --test dump_srl_frames -- --ignored --nocapture

#[cfg(feature = "onnx")]
mod onnx_dump {
    use std::time::Instant;

    use uni_db::ModelAliasSpec;
    use uniko_extract::nlp::NlpPipeline;
    use uniko_store::KnowledgeBase;
    use uniko_store::config::UnikoConfig;

    /// Mix of conv-26 turns that gave us trouble earlier (D2:14, D3:13,
    /// D4:3) plus two clearly-temporal/locative sentences for SRL
    /// sanity. Strings are the *single sentences* we care about, not
    /// whole multi-sentence messages, so the pipeline doesn't strip
    /// the relevant content via the CLS gate.
    const TARGETS: &[&str] = &[
        "It'll be tough as a single parent.",
        "Their love and help have been so important especially after that tough breakup.",
        "This necklace is a gift from my grandma in my home country, Sweden.",
        "I went to the Museum of Modern Art on January fourth.",
        "She gave me the necklace yesterday in her kitchen.",
    ];

    async fn make_pipeline(srl_enabled: bool) -> NlpPipeline {
        let ws = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        std::env::set_current_dir(&ws).expect("cd");
        let mut config = UnikoConfig {
            catalog_path: Some(ws.join("config/catalog.json")),
            schema_path: Some(ws.join("config/schema.json")),
            ..Default::default()
        };
        config.nlp_srl_enabled = srl_enabled;
        let tmp = std::env::temp_dir().join(format!(
            "uniko-srl-dump-{}-{}",
            if srl_enabled { "on" } else { "off" },
            std::process::id()
        ));
        std::fs::create_dir_all(&tmp).ok();
        let kb = KnowledgeBase::open_with_xervo(&tmp, config, Vec::<ModelAliasSpec>::new())
            .await
            .expect("KB");
        NlpPipeline::try_new(&kb).await.expect("pipeline")
    }

    #[ignore]
    #[tokio::test]
    async fn dump_srl_frames_for_target_sentences() {
        let pipeline = make_pipeline(true).await;
        for (i, sentence) in TARGETS.iter().enumerate() {
            println!("\n=== TARGET {i}: {sentence} ===");
            let result = pipeline.analyze(sentence).await.expect("analyze");
            println!("  words: {:?}", result.words);
            if result.srl_frames.is_empty() {
                println!("  (no frames — model emitted no recognised predicates)");
                continue;
            }
            for (f_idx, frame) in result.srl_frames.iter().enumerate() {
                println!(
                    "  frame[{f_idx}] predicate=`{}` (word #{})",
                    frame.predicate_word, frame.predicate_idx,
                );
                if frame.args.is_empty() {
                    println!("    (no args)");
                }
                for arg in &frame.args {
                    println!(
                        "    [{}..{}) {}: `{}`",
                        arg.start_word, arg.end_word, arg.role, arg.text,
                    );
                }
            }
        }
    }

    /// Latency micro-benchmark: average wall-clock time per sentence
    /// with SRL on vs off. Acceptance per the plan: ≤3× single-pass
    /// for ≤3 verbs/sentence.
    #[ignore]
    #[tokio::test]
    async fn srl_latency_micro_benchmark() {
        let off = make_pipeline(false).await;
        let on = make_pipeline(true).await;
        // Warm both pipelines with one untimed call each.
        let _ = off.analyze(TARGETS[0]).await.expect("warmup off");
        let _ = on.analyze(TARGETS[0]).await.expect("warmup on");

        let t_off = Instant::now();
        for s in TARGETS {
            let _ = off.analyze(s).await.expect("analyze off");
        }
        let off_ms = t_off.elapsed().as_millis() as f64 / TARGETS.len() as f64;

        let t_on = Instant::now();
        let mut total_frames = 0;
        for s in TARGETS {
            let r = on.analyze(s).await.expect("analyze on");
            total_frames += r.srl_frames.len();
        }
        let on_ms = t_on.elapsed().as_millis() as f64 / TARGETS.len() as f64;

        println!(
            "SRL latency: off={off_ms:.1}ms/sentence  on={on_ms:.1}ms/sentence \
             (×{:.2}, total frames across {} sentences = {})",
            on_ms / off_ms.max(1.0),
            TARGETS.len(),
            total_frames,
        );
    }
}
