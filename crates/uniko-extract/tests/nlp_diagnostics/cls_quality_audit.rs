//! CLS quality audit on the full conv-26 message stream.
//!
//! Runs the NLP pipeline over every sentence in conv-26 and reports:
//!   1. Class distribution (Statement / Question / Command / Greeting /
//!      Acknowledgment) and average confidence per class.
//!   2. Heuristic-vs-model confusion: sentences ending in `?` vs the
//!      CLS verdict; short phatic sentences vs Greeting; long declarative
//!      sentences vs Statement.
//!   3. Sample misclassifications — long declarative sentences classified
//!      as Greeting/Question/Acknowledgment, with confidence.
//!
//! Run: cargo nextest run -p uniko-extract --features onnx \
//!     --test cls_quality_audit --run-ignored all --nocapture

#[cfg(feature = "onnx")]
mod audit {
    use std::collections::HashMap;

    use uni_db::ModelAliasSpec;
    use uniko_extract::nlp::NlpPipeline;
    use uniko_extract::nlp::types::SentenceClass;
    use uniko_store::KnowledgeBase;
    use uniko_store::config::UnikoConfig;

    fn class_name(c: SentenceClass) -> &'static str {
        match c {
            SentenceClass::Statement => "Statement",
            SentenceClass::Question => "Question",
            SentenceClass::Command => "Command",
            SentenceClass::Greeting => "Greeting",
            SentenceClass::Acknowledgment => "Acknowledgment",
        }
    }

    /// Heuristic: does the sentence look like a Statement (long-ish, no
    /// `?`, no greeting tokens, has alpha words)?
    fn looks_declarative(text: &str) -> bool {
        let trimmed = text.trim();
        if trimmed.ends_with('?') {
            return false;
        }
        let words: Vec<&str> = trimmed.split_whitespace().collect();
        if words.len() < 6 {
            return false;
        }
        let lower = trimmed.to_lowercase();
        const PHATIC: &[&str] = &[
            "hi ",
            "hello",
            "hey ",
            "thanks",
            "thank you",
            "good morning",
            "good night",
            "bye",
            "see you",
        ];
        if PHATIC.iter().any(|p| lower.contains(p)) {
            return false;
        }
        true
    }

    fn looks_question(text: &str) -> bool {
        text.trim_end_matches(|c: char| c.is_whitespace())
            .ends_with('?')
    }

    fn looks_phatic(text: &str) -> bool {
        let lower = text.to_lowercase();
        let words = text.split_whitespace().count();
        if words > 8 {
            return false;
        }
        const PHATIC: &[&str] = &[
            "hi ",
            "hello",
            "hey ",
            "thanks",
            "thank you",
            "good morning",
            "good night",
            "bye!",
            "see you",
        ];
        PHATIC.iter().any(|p| lower.contains(p))
    }

    #[ignore]
    #[tokio::test]
    async fn audit_conv26_cls() {
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
        let tmp = std::env::temp_dir().join(format!("uniko-cls-audit-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).ok();
        let kb = KnowledgeBase::open_with_xervo(&tmp, config, Vec::<ModelAliasSpec>::new())
            .await
            .expect("KB");
        let pipeline = NlpPipeline::try_new(&kb).await.expect("pipeline");

        // Load conv-26 messages.
        let data: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(ws.join("data/locomo10.json")).unwrap())
                .unwrap();
        let conv = data
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["sample_id"] == "conv-26")
            .unwrap();
        let mut texts: Vec<String> = Vec::new();
        for (k, v) in conv["conversation"].as_object().unwrap() {
            if !k.starts_with("session") {
                continue;
            }
            if let Some(arr) = v.as_array() {
                for t in arr {
                    if let Some(text) = t["text"].as_str() {
                        texts.push(text.to_string());
                    }
                }
            }
        }
        println!("Loaded {} conv-26 messages", texts.len());

        // Tally.
        let mut counts: HashMap<&'static str, usize> = HashMap::new();
        let mut confs: HashMap<&'static str, Vec<f32>> = HashMap::new();
        let mut total_sents = 0usize;
        // Heuristic agreement.
        let mut q_total = 0usize;
        let mut q_match = 0usize; // ends with ? AND CLS=Question
        let mut decl_total = 0usize;
        let mut decl_as_statement = 0usize;
        let mut decl_misclass: Vec<(String, &'static str, f32)> = Vec::new();
        let mut phatic_total = 0usize;
        let mut phatic_as_greeting = 0usize;

        for text in &texts {
            let results = pipeline.analyze_sentences(text).await.expect("analyze");
            for r in &results {
                total_sents += 1;
                let name = class_name(r.sentence_class);
                *counts.entry(name).or_insert(0) += 1;
                confs.entry(name).or_default().push(r.cls_confidence);

                let sentence = r.words.join(" ");
                if looks_question(&sentence) {
                    q_total += 1;
                    if r.sentence_class == SentenceClass::Question {
                        q_match += 1;
                    }
                }
                if looks_declarative(&sentence) {
                    decl_total += 1;
                    if r.sentence_class == SentenceClass::Statement {
                        decl_as_statement += 1;
                    } else {
                        decl_misclass.push((sentence.clone(), name, r.cls_confidence));
                    }
                }
                if looks_phatic(&sentence) {
                    phatic_total += 1;
                    if r.sentence_class == SentenceClass::Greeting {
                        phatic_as_greeting += 1;
                    }
                }
            }
        }

        println!("\n=== Class distribution ({total_sents} sentences) ===");
        let mut classes: Vec<&&'static str> = counts.keys().collect();
        classes.sort();
        for name in classes {
            let n = counts[name];
            let v = &confs[name];
            let mean: f32 = if v.is_empty() {
                0.0
            } else {
                v.iter().sum::<f32>() / v.len() as f32
            };
            let pct = 100.0 * n as f64 / total_sents as f64;
            println!("  {name:<16} {n:>5} ({pct:>5.1}%)   mean conf={mean:.3}");
        }

        println!("\n=== Heuristic agreement ===");
        println!(
            "  ends-with-?         → CLS=Question: {q_match}/{q_total} ({:.1}%)",
            if q_total > 0 {
                100.0 * q_match as f64 / q_total as f64
            } else {
                0.0
            },
        );
        println!(
            "  long-declarative    → CLS=Statement: {decl_as_statement}/{decl_total} ({:.1}%)",
            if decl_total > 0 {
                100.0 * decl_as_statement as f64 / decl_total as f64
            } else {
                0.0
            },
        );
        println!(
            "  phatic short        → CLS=Greeting: {phatic_as_greeting}/{phatic_total} ({:.1}%)",
            if phatic_total > 0 {
                100.0 * phatic_as_greeting as f64 / phatic_total as f64
            } else {
                0.0
            },
        );

        // Group declarative misclassifications by predicted label.
        let mut by_class: HashMap<&'static str, Vec<(String, f32)>> = HashMap::new();
        for (s, c, conf) in decl_misclass {
            by_class.entry(c).or_default().push((s, conf));
        }
        println!("\n=== Long-declarative sentences misclassified (loss for extraction) ===",);
        let mut keys: Vec<&&'static str> = by_class.keys().collect();
        keys.sort();
        for k in keys {
            let v = &by_class[k];
            println!("\n  → predicted {k} ({} sentences):", v.len());
            // Show first 6 plus 3 random tail samples.
            for (s, conf) in v.iter().take(6) {
                let trimmed: String = s.chars().take(140).collect();
                println!("      [{conf:.2}] {trimmed}");
            }
            if v.len() > 6 {
                println!("      ... ({} more)", v.len() - 6);
            }
        }
    }
}
