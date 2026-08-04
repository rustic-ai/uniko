//! Raw CLS dump — for every conv-26 sentence, record the raw 8-label
//! head verdict (`inform` / `request` / `question` / `confirm` / `reject`
//! / `offer` / `social` / `status`) and confidence. No collapsing into
//! `SentenceClass` — this is what the model actually outputs.
//!
//! Writes a TSV to `data/cls_conv26.tsv` (one row per sentence) and a
//! summary table to stdout.
//!
//! Run: cargo test -p uniko-extract --features onnx \
//!     --test cls_raw_dump -- --ignored --nocapture

#[cfg(feature = "onnx")]
mod raw {
    use std::collections::HashMap;
    use std::io::Write;

    use uni_db::ModelAliasSpec;
    use uniko_extract::nlp::NlpPipeline;
    use uniko_extract::nlp::assets::label_maps;
    use uniko_store::KnowledgeBase;
    use uniko_store::config::UnikoConfig;

    #[ignore]
    #[tokio::test]
    async fn dump_raw_cls() {
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
        let tmp = std::env::temp_dir().join(format!("uniko-cls-raw-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).ok();
        let kb = KnowledgeBase::open_with_xervo(&tmp, config, Vec::<ModelAliasSpec>::new())
            .await
            .expect("KB");
        let pipeline = NlpPipeline::try_new(&kb).await.expect("pipeline");
        let labels = label_maps();

        let data: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(ws.join("data/locomo10.json")).unwrap())
                .unwrap();
        let conv = data
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["sample_id"] == "conv-26")
            .unwrap();
        let mut texts: Vec<(String, String, String)> = Vec::new(); // (dia_id, speaker, text)
        for (k, v) in conv["conversation"].as_object().unwrap() {
            if !k.starts_with("session") {
                continue;
            }
            if let Some(arr) = v.as_array() {
                for t in arr {
                    let dia = t["dia_id"].as_str().unwrap_or("").to_string();
                    let sp = t["speaker"].as_str().unwrap_or("").to_string();
                    let txt = t["text"].as_str().unwrap_or("").to_string();
                    texts.push((dia, sp, txt));
                }
            }
        }
        println!("Loaded {} conv-26 messages", texts.len());

        let out_path = ws.join("data/cls_conv26.tsv");
        let mut out = std::fs::File::create(&out_path).expect("create tsv");
        writeln!(out, "dia_id\tspeaker\tsent_idx\tcls_label\tconf\tsentence").unwrap();

        let mut counts: HashMap<&'static str, usize> = HashMap::new();
        let mut conf_sum: HashMap<&'static str, f32> = HashMap::new();
        let mut total = 0usize;
        // Confidence histogram bucketed per label.
        let mut buckets: HashMap<&'static str, [usize; 5]> = HashMap::new();

        for (dia, speaker, text) in &texts {
            let results = pipeline.analyze_sentences(text).await.expect("analyze");
            for (idx, r) in results.iter().enumerate() {
                let raw_label: &str = labels
                    .cls_labels
                    .get(r.cls_index)
                    .map(String::as_str)
                    .unwrap_or("?");
                // Lifetime-stable key for hashmap.
                let static_label: &'static str = match raw_label {
                    "inform" => "inform",
                    "request" => "request",
                    "question" => "question",
                    "confirm" => "confirm",
                    "reject" => "reject",
                    "offer" => "offer",
                    "social" => "social",
                    "status" => "status",
                    _ => "unknown",
                };
                total += 1;
                *counts.entry(static_label).or_insert(0) += 1;
                *conf_sum.entry(static_label).or_insert(0.0) += r.cls_confidence;
                let b = (r.cls_confidence * 5.0).floor() as usize;
                let b = b.min(4);
                buckets.entry(static_label).or_insert([0; 5])[b] += 1;

                let sentence = r.words.join(" ").replace('\t', " ");
                writeln!(
                    out,
                    "{dia}\t{speaker}\t{idx}\t{static_label}\t{:.3}\t{sentence}",
                    r.cls_confidence,
                )
                .unwrap();
            }
        }
        out.flush().ok();
        println!("Wrote per-sentence TSV to {}", out_path.display());

        println!("\n=== Raw CLS distribution ({total} sentences) ===");
        let mut labels_ord = [
            "inform", "request", "question", "confirm", "reject", "offer", "social", "status",
        ];
        labels_ord.sort();
        for name in labels_ord {
            let n = *counts.get(name).unwrap_or(&0);
            let s = *conf_sum.get(name).unwrap_or(&0.0);
            let mean = if n > 0 { s / n as f32 } else { 0.0 };
            let pct = 100.0 * n as f64 / total as f64;
            let b = buckets.get(name).copied().unwrap_or([0; 5]);
            println!(
                "  {name:<10} {n:>5} ({pct:>5.1}%)  mean conf={mean:.3}   conf hist [0-.2|-.4|-.6|-.8|-1]: {b:?}",
            );
        }
    }
}
