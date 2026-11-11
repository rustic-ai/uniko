//! Regression repro for the xervo `OnnxNlpModel::analyze` argmax bug.
//!
//! xervo `0.13.0` ships an `OnnxNlpModel` (`runtime.nlp_model(alias)
//! .analyze(..)`) that fails on every input with:
//!
//! ```text
//! ONNX invocation failure for alias 'nlp': argmax_axis only supports axis=1, got 2
//! ```
//!
//! Root cause: `uni-xervo/src/provider/local_onnx/nlp.rs:257` calls
//! `argmax_axis(&outputs.arc_scores, 2)` on a 2-D `[seq, seq]` matrix,
//! but `argmax_axis` only supports `axis = 1` (per-row argmax). The call
//! runs before the `NlpTasks::DEP` gate, so it breaks all task selections,
//! not just dependency parsing. Fix: change the axis from `2` to `1`.
//!
//! See `crates/uniko-bench/docs/xervo-nlp-argmax-bug.md` for the full
//! write-up.
//!
//! This test asserts the *correct, post-fix* behavior. It is `#[ignore]`d
//! because it needs the `kniv-deberta` model download plus an ONNX
//! runtime. Today it fails with the argmax error (reproducing the bug);
//! once xervo is fixed and the workspace dep is bumped, it passes and
//! becomes a regression guard.
//!
//! Run with:
//! ```bash
//! cargo test -p uniko-bench --no-default-features \
//!   --test xervo_nlp_argmax_repro -- --ignored --nocapture
//! ```

use uni_db::ModelAliasSpec;
use uni_xervo::traits::{NlpRequest, NlpTasks};
use uniko_store::KnowledgeBase;
use uniko_store::config::{EmbeddingConfig, UnikoConfig};

/// Alias registered as `task = nlp` to exercise the `NlpModel` path.
const PARITY_ALIAS: &str = "nlp/parity";

/// Resolve xervo's `NlpModel` for the kniv-deberta cascade and analyze
/// one sentence, asserting every head populates as designed.
#[tokio::test]
#[ignore = "needs kniv-deberta model download + ort runtime; run with --ignored"]
async fn xervo_nlp_model_analyze_populates_all_heads() {
    let scratch = std::env::temp_dir().join("uniko_xervo_nlp_repro");
    if scratch.exists() {
        std::fs::remove_dir_all(&scratch).expect("clear scratch");
    }
    std::fs::create_dir_all(&scratch).expect("create scratch");

    // A full catalog file *replaces* the auto-built one, so list every
    // alias we need. `nlp/parity` (task=nlp) drives `OnnxNlpModel`.
    let catalog_file = scratch.join("catalog.json");
    let catalog_json = serde_json::json!([
        {
            "alias": "embed/default",
            "task": "embed",
            "provider_id": "local/onnx",
            "model_id": "BGESmallENV15",
            "warmup": "lazy",
            "required": false
        },
        {
            "alias": PARITY_ALIAS,
            "task": "nlp",
            "provider_id": "local/onnx",
            "model_id": "dragonscale-ai/kniv-deberta-nlp-base-en-xsmall",
            "warmup": "lazy",
            "required": false,
            "options": {
                "generation": "v1",
                "onnx_path": "onnx/cascade-int8.onnx",
                "tokenizer_path": "tokenizer.json",
                "label_maps_path": "label_maps.json",
                "max_seq_len": 128
            }
        }
    ]);
    std::fs::write(
        &catalog_file,
        serde_json::to_string_pretty(&catalog_json).expect("serialize catalog"),
    )
    .expect("write catalog");

    let config = UnikoConfig {
        embedding: EmbeddingConfig::bge_small_en_v15(),
        catalog_path: Some(catalog_file),
        ..Default::default()
    };

    let kb = KnowledgeBase::open_with_xervo(&scratch, config, Vec::<ModelAliasSpec>::new())
        .await
        .expect("open scratch KB");

    let runtime = kb
        .db()
        .xervo()
        .raw_runtime()
        .cloned()
        .expect("xervo runtime configured");
    let nlp = runtime
        .nlp_model(PARITY_ALIAS)
        .await
        .expect("resolve nlp/parity NlpModel");

    // One sentence with a verb, proper nouns, and a clear dependency
    // structure — mirrors xervo's own (ignored) e2e test input.
    let req = NlpRequest::new("Alice bought a book in Paris.", NlpTasks::ALL);

    // PRE-FIX: this errors with "argmax_axis only supports axis=1, got 2".
    // POST-FIX: it returns one populated result.
    let results = nlp
        .analyze(vec![req])
        .await
        .expect("analyze must succeed once nlp.rs:257 uses axis=1");

    assert_eq!(results.len(), 1, "one result per request");
    let r = &results[0];

    assert!(!r.tokens.is_empty(), "tokens populated");

    let pos: Vec<&str> = r.tokens.iter().filter_map(|t| t.pos.as_deref()).collect();
    assert!(pos.contains(&"VERB"), "POS includes a VERB: {pos:?}");

    assert!(
        r.tokens.iter().any(|t| t.ner.is_some()),
        "at least one NER-tagged token (Alice / Paris)"
    );
    assert!(
        r.tokens.iter().any(|t| t.dep.is_some()),
        "DEP heads populated — the head decode at nlp.rs:257 ran cleanly"
    );
    assert!(!r.frames.is_empty(), "SRL frame for the predicate 'bought'");
}
