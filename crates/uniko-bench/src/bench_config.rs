//! Bench-profile JSON config.
//!
//! Replaces the legacy flat CLI flag set with a single JSON document
//! that describes every model, device, recall, and cost knob for a
//! LoCoMo bench run.  The bench binary loads one file via
//! `--bench-config <path>` and applies it on top of
//! [`UnikoConfig::default`]; CLI carries only what changes per
//! invocation (data file, output path, conversation / category
//! filters, KB reuse).
//!
//! # JSON shape
//!
//! ```json
//! {
//!   "models": {
//!     "gen": {
//!       "alias": "llm/gen",
//!       "model_id": "gpt-4o-mini",
//!       "provider": "remote/openai",
//!       "base_url": null,
//!       "use_default_options": null
//!     },
//!     "judge": {
//!       "alias": "llm/judge",
//!       "model_id": "gpt-4o-mini",
//!       "provider": "remote/openai",
//!       "base_url": null
//!     },
//!     "extract_triples": null,
//!     "embedder": { "preset": "bge-small" },
//!     "reranker": {
//!       "enabled": true,
//!       "model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
//!       "style": "cross-encoder",
//!       "top_n": 50,
//!       "execution_providers": null
//!     },
//!     "nlp": {
//!       "model_id": "dragonscale-ai/kniv-deberta-nlp-base-en-xsmall",
//!       "artifact": "onnx/cascade-int8.onnx",
//!       "max_batch_size": 16,
//!       "execution_providers": null
//!     }
//!   },
//!   "recall": {
//!     "limit": null,
//!     "phase1_strategy": "boost",
//!     "phase1_boost_alpha": 0.6,
//!     "phase2_graph_enabled": true,
//!     "phase2_temporal_enabled": true,
//!     "variants": [],
//!     "consolidation_cluster_objects": true,
//!     "consolidation_date_augment_embedding": true
//!   },
//!   "judge_enabled": true,
//!   "retrieval_only": false,
//!   "cost": {
//!     "pricing_csv": "crates/uniko-bench/pricing.csv",
//!     "events_jsonl": null,
//!     "no_events": false
//!   }
//! }
//! ```
//!
//! The `embedder` field accepts either `{"preset": "<name>"}` for one
//! of the hardcoded presets (`bge-small`, `bge-large`, `bge-m3`, `nomic`,
//! `minilm`, `embeddinggemma`) or
//! `{"inline": <EmbeddingConfig>}` for a fully specified config. The
//! `bge-m3` preset enables single-pass hybrid (dense + sparse + ColBERT);
//! pair it with `recall.sparse_enabled` and/or `reranker.style = "colbert"`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;
use uni_db::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uniko_store::config::{EmbeddingConfig, NlpConfig, RerankerConfig, UnikoConfig};

/// Root config document.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchConfig {
    /// Model selection (LLMs, embedder, reranker, NLP cascade).
    pub models: ModelsConfig,
    /// Recall-pipeline knobs (phase strategies, limits, variants).
    #[serde(default)]
    pub recall: RecallSettings,
    /// Whether the LLM judge runs.  When `false`, accuracy reports
    /// carry F1 only.
    #[serde(default = "default_true")]
    pub judge_enabled: bool,
    /// Skip answer-LLM generation entirely; "predicted answer" is the
    /// top-K recall items joined as text.
    #[serde(default)]
    pub retrieval_only: bool,
    /// Pricing CSV path + per-event JSONL output toggles.
    #[serde(default)]
    pub cost: CostSettings,
    /// LongMemEval-harness specific knobs (concurrency, token budget,
    /// question-type filter).  Other bench harnesses ignore this.
    #[serde(default)]
    pub lme: LmeSettings,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelsConfig {
    /// Generator LLM (answer model).  JSON key is `gen` for brevity;
    /// the Rust field is renamed because `gen` is reserved in edition 2024.
    #[serde(rename = "gen")]
    pub generator: LlmAlias,
    /// Judge LLM.  Required field; set [`BenchConfig::judge_enabled`]
    /// to `false` to disable judging without removing this entry.
    pub judge: LlmAlias,
    /// Optional LLM-refined triple extractor for P4 consolidation.
    /// `None` → SRL/DEP triples (no LLM call).
    #[serde(default)]
    pub extract_triples: Option<LlmAlias>,
    /// Embedder selection.
    pub embedder: EmbedderChoice,
    /// Cross-encoder reranker selection.
    pub reranker: RerankerConfig,
    /// NLP cascade selection.
    #[serde(default)]
    pub nlp: NlpConfig,
}

/// One generation alias (gen, judge, or extract-triples).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LlmAlias {
    /// xervo alias.  Must be `task/name` format per uni-db.
    pub alias: String,
    /// HuggingFace model id or provider model name.
    pub model_id: String,
    /// xervo provider id (e.g. `remote/openai`, `remote/vertexai`,
    /// `remote/llamacpp`, `local/candle`).
    pub provider: String,
    /// Optional OpenAI-compatible base URL override.
    #[serde(default)]
    pub base_url: Option<String>,
    /// Force `GenerationOptions::default()` (skip the bench's
    /// `max_tokens=2048, temperature=0.1` overrides).  Required for
    /// reasoning models (gpt-5, gemini-3-*).  `None` → auto: true
    /// when provider is `remote/openai`, false otherwise.
    #[serde(default)]
    pub use_default_options: Option<bool>,
}

/// Embedder selection — preset shorthand or fully inline config.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EmbedderChoice {
    /// `{"preset": "<name>"}` — resolves to a hardcoded preset.
    Preset { preset: String },
    /// `{"inline": <EmbeddingConfig>}` — full custom config. Boxed
    /// because `EmbeddingConfig` is much larger than the `Preset` variant.
    Inline { inline: Box<EmbeddingConfig> },
}

/// Recall-pipeline settings.  Defaults match the bench's historical
/// CLI defaults.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecallSettings {
    /// Override `UnikoConfig.recall_limit`.  `None` → keep default.
    #[serde(default)]
    pub limit: Option<usize>,
    /// Override the dense-vector weight in hybrid fusion. `None` → keep
    /// the `UnikoConfig` default (0.5).
    #[serde(default)]
    pub vector_weight: Option<f64>,
    /// Override the BM25 weight in hybrid fusion. `None` → keep default.
    #[serde(default)]
    pub bm25_weight: Option<f64>,
    /// Enable the learned-sparse retrieval channel (Chunk + Observation).
    /// Requires a hybrid embedder (e.g. the `bge-m3` preset). Default off.
    #[serde(default)]
    pub sparse_enabled: bool,
    #[serde(default = "default_phase1_strategy")]
    pub phase1_strategy: String,
    #[serde(default = "default_phase1_boost_alpha")]
    pub phase1_boost_alpha: f64,
    #[serde(default = "default_true")]
    pub phase2_graph_enabled: bool,
    #[serde(default = "default_true")]
    pub phase2_temporal_enabled: bool,
    /// Comma-list of query-reformulation variants.  Empty → legacy
    /// single-variant default (keywords only).
    #[serde(default)]
    pub variants: Vec<String>,
    #[serde(default = "default_true")]
    pub consolidation_cluster_objects: bool,
    #[serde(default = "default_true")]
    pub consolidation_date_augment_embedding: bool,
}

impl Default for RecallSettings {
    fn default() -> Self {
        Self {
            limit: None,
            vector_weight: None,
            bm25_weight: None,
            sparse_enabled: false,
            phase1_strategy: default_phase1_strategy(),
            phase1_boost_alpha: default_phase1_boost_alpha(),
            phase2_graph_enabled: true,
            phase2_temporal_enabled: true,
            variants: Vec::new(),
            consolidation_cluster_objects: true,
            consolidation_date_augment_embedding: true,
        }
    }
}

/// Cost-capture settings.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CostSettings {
    /// Pricing CSV path (resolved relative to the working directory
    /// the bench is invoked from).
    #[serde(default = "default_pricing_csv")]
    pub pricing_csv: PathBuf,
    /// Override the auto-derived `<output stem>_events.jsonl` path.
    #[serde(default)]
    pub events_jsonl: Option<PathBuf>,
    /// Disable the JSONL event writer entirely.
    #[serde(default)]
    pub no_events: bool,
}

impl Default for CostSettings {
    fn default() -> Self {
        Self {
            pricing_csv: default_pricing_csv(),
            events_jsonl: None,
            no_events: false,
        }
    }
}

/// LongMemEval-specific runtime knobs.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LmeSettings {
    /// Concurrent LME items in flight.  Each holds its own KB; tune
    /// against GPU memory + shared-runtime headroom.
    #[serde(default = "default_lme_question_concurrency")]
    pub question_concurrency: usize,
    /// Concurrent sessions ingested within one item.  Capped by uni-db
    /// write contention.
    #[serde(default = "default_lme_session_concurrency")]
    pub session_concurrency: usize,
    /// Token budget for the recall bundle handed to the LLM.
    #[serde(default = "default_lme_token_budget")]
    pub token_budget: usize,
    /// Optional filter to specific LME question types (shorthand:
    /// `ssu`, `ssa`, `ssp`, `ms`, `tr`, `ku`).  `None` → no filter.
    #[serde(default)]
    pub question_types: Option<Vec<String>>,
}

impl Default for LmeSettings {
    fn default() -> Self {
        Self {
            question_concurrency: default_lme_question_concurrency(),
            session_concurrency: default_lme_session_concurrency(),
            token_budget: default_lme_token_budget(),
            question_types: None,
        }
    }
}

fn default_lme_question_concurrency() -> usize {
    8
}

fn default_lme_session_concurrency() -> usize {
    8
}

fn default_lme_token_budget() -> usize {
    8192
}

fn default_true() -> bool {
    true
}

fn default_phase1_strategy() -> String {
    "boost".to_string()
}

fn default_phase1_boost_alpha() -> f64 {
    0.6
}

fn default_pricing_csv() -> PathBuf {
    PathBuf::from("crates/uniko-bench/pricing.csv")
}

impl BenchConfig {
    /// Load a bench config from disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or the JSON does
    /// not match the expected schema.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let body = std::fs::read_to_string(path)
            .with_context(|| format!("reading bench config at {}", path.display()))?;
        serde_json::from_str(&body)
            .with_context(|| format!("parsing bench config {}", path.display()))
    }

    /// Effective `use_default_options` for the generation LLM.
    ///
    /// When the JSON value is `None`, auto-detect: providers known to
    /// reject the bench's `max_tokens`/`temperature` overrides
    /// (`remote/openai`, `remote/vertexai` with reasoning models)
    /// switch on.  Today we conservatively only auto-enable for
    /// `remote/openai`; vertexai/gemini-3 callers should set the
    /// field explicitly in the JSON for clarity.
    #[must_use]
    pub fn gen_use_default_options(&self) -> bool {
        match self.models.generator.use_default_options {
            Some(v) => v,
            None => self.models.generator.provider == "remote/openai",
        }
    }

    /// Resolve the embedder choice into a concrete [`EmbeddingConfig`].
    ///
    /// # Errors
    ///
    /// Returns an error when [`EmbedderChoice::Preset`] names a
    /// preset that the bench doesn't recognise.
    pub fn resolve_embedder(&self) -> Result<EmbeddingConfig> {
        match &self.models.embedder {
            EmbedderChoice::Inline { inline } => Ok((**inline).clone()),
            EmbedderChoice::Preset { preset } => EmbeddingConfig::preset(preset).ok_or_else(|| {
                anyhow::anyhow!(
                    "unknown embedder preset {preset:?}; expected one of: nomic, minilm, \
                     bge-small, bge-large, bge-m3, embeddinggemma"
                )
            }),
        }
    }

    /// Apply the bench config's recall / model selections to a
    /// caller-supplied [`UnikoConfig`].
    ///
    /// # Errors
    ///
    /// Propagates [`Self::resolve_embedder`] errors.
    pub fn apply_to_uniko_config(&self, config: &mut UnikoConfig) -> Result<()> {
        config.embedding = self.resolve_embedder()?;
        config.reranker = self.models.reranker.clone();
        config.nlp = self.models.nlp.clone();

        if let Some(limit) = self.recall.limit {
            config.recall_limit = limit;
            // Reranker top_n must be >= recall_limit (UnikoConfig
            // validation rejects otherwise).
            if config.reranker.enabled && config.reranker.top_n < limit {
                config.reranker.top_n = limit;
            }
        }
        if let Some(w) = self.recall.vector_weight {
            config.recall_vector_weight = w;
        }
        if let Some(w) = self.recall.bm25_weight {
            config.recall_bm25_weight = w;
        }
        config.recall_sparse_enabled = self.recall.sparse_enabled;
        config.phase1_strategy = self.recall.phase1_strategy.clone();
        config.phase1_boost_alpha = self.recall.phase1_boost_alpha;
        config.phase2_graph_enabled = self.recall.phase2_graph_enabled;
        config.phase2_temporal_enabled = self.recall.phase2_temporal_enabled;
        config.consolidation_cluster_objects = self.recall.consolidation_cluster_objects;
        config.consolidation_date_augment_embedding =
            self.recall.consolidation_date_augment_embedding;
        if !self.recall.variants.is_empty() {
            config.query_variants = self.recall.variants.clone();
        }
        Ok(())
    }

    /// Build the xervo catalog entries the bench passes to the KB.
    ///
    /// Mirrors what `build_llm_catalog` did with the old CLI flag
    /// surface — emits one [`ModelAliasSpec`] for the generator,
    /// optionally one for the judge (when a distinct alias is
    /// requested), and one for `extract_triples` when configured.
    pub fn build_catalog_specs(&self) -> Vec<ModelAliasSpec> {
        let mut catalog = Vec::new();
        catalog.push(llm_alias_to_spec(&self.models.generator));
        if self.models.judge.alias != self.models.generator.alias {
            catalog.push(llm_alias_to_spec(&self.models.judge));
        }
        if let Some(ref triples) = self.models.extract_triples
            && triples.alias != self.models.generator.alias
            && triples.alias != self.models.judge.alias
        {
            catalog.push(llm_alias_to_spec(triples));
        }
        catalog
    }
}

/// Convert one [`LlmAlias`] into a xervo `ModelAliasSpec`.
fn llm_alias_to_spec(a: &LlmAlias) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: a.alias.clone(),
        task: ModelTask::Generate,
        provider_id: a.provider.clone(),
        model_id: a.model_id.clone(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: provider_options(&a.provider, a.base_url.as_deref()),
    }
}

/// Provider-specific catalog options used when materialising
/// [`ModelAliasSpec`] entries from a JSON-loaded [`LlmAlias`].
fn provider_options(provider: &str, base_url: Option<&str>) -> serde_json::Value {
    match provider {
        "remote/openai" => match base_url {
            Some(url) => serde_json::json!({"base_url": url}),
            None => serde_json::json!({}),
        },
        // A `llama-server` address is site-specific, so there is no
        // default to fall back to. Omitting `base_url` leaves the
        // options empty and uni-xervo rejects the alias by name at
        // catalog load, which is the right failure: silently inventing
        // `localhost:8080` would make a misconfigured arm look like a
        // model that answers badly.
        "remote/llamacpp" => match base_url {
            Some(url) => serde_json::json!({"base_url": url}),
            None => serde_json::json!({}),
        },
        "remote/vertexai" => {
            let mut opts = serde_json::Map::new();
            opts.insert("location".into(), serde_json::json!("global"));
            opts.insert(
                "api_token_env".into(),
                serde_json::json!("VERTEXAI_API_TOKEN"),
            );
            if let Ok(project) = std::env::var("VERTEXAI_PROJECT") {
                opts.insert("project_id".into(), serde_json::json!(project));
            }
            serde_json::Value::Object(opts)
        }
        _ => serde_json::json!({}),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_minimal_profile() {
        let json = r#"{
            "models": {
                "gen":   {"alias":"llm/gen","model_id":"gpt-4o-mini","provider":"remote/openai"},
                "judge": {"alias":"llm/judge","model_id":"gpt-4o-mini","provider":"remote/openai"},
                "embedder": {"preset": "bge-small"},
                "reranker": {
                    "enabled": true,
                    "model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    "top_n": 50,
                    "apply_sigmoid": false
                }
            }
        }"#;
        let cfg: BenchConfig = serde_json::from_str(json).expect("parse");
        assert_eq!(cfg.models.generator.model_id, "gpt-4o-mini");
        assert_eq!(cfg.models.generator.provider, "remote/openai");
        assert!(cfg.judge_enabled, "default true");
        assert!(!cfg.retrieval_only, "default false");
        assert!(cfg.gen_use_default_options(), "auto for remote/openai");
        let embed = cfg.resolve_embedder().expect("resolve");
        assert_eq!(embed.dimensions, 384, "bge-small is 384d");
    }

    #[test]
    fn auto_default_options_disabled_for_non_openai_provider() {
        let json = r#"{
            "models": {
                "gen":   {"alias":"llm/gen","model_id":"gemma-3-4b-it","provider":"remote/llamacpp","base_url":"http://127.0.0.1:8080/v1"},
                "judge": {"alias":"llm/judge","model_id":"gemma-3-4b-it","provider":"remote/llamacpp","base_url":"http://127.0.0.1:8080/v1"},
                "embedder": {"preset": "bge-small"},
                "reranker": {"enabled":true,"model_id":"x","top_n":50,"apply_sigmoid":false}
            }
        }"#;
        let cfg: BenchConfig = serde_json::from_str(json).expect("parse");
        assert!(
            !cfg.gen_use_default_options(),
            "off for providers other than remote/openai"
        );
    }

    #[test]
    fn rejects_unknown_preset() {
        let json = r#"{
            "models": {
                "gen":   {"alias":"llm/gen","model_id":"x","provider":"remote/openai"},
                "judge": {"alias":"llm/judge","model_id":"x","provider":"remote/openai"},
                "embedder": {"preset": "totally-fake"},
                "reranker": {"enabled":true,"model_id":"x","top_n":50,"apply_sigmoid":false}
            }
        }"#;
        let cfg: BenchConfig = serde_json::from_str(json).expect("parse");
        assert!(cfg.resolve_embedder().is_err());
    }

    #[test]
    fn deny_unknown_fields_at_root() {
        let json = r#"{
            "models": {
                "gen":   {"alias":"llm/gen","model_id":"x","provider":"remote/openai"},
                "judge": {"alias":"llm/judge","model_id":"x","provider":"remote/openai"},
                "embedder": {"preset": "bge-small"},
                "reranker": {"enabled":true,"model_id":"x","top_n":50,"apply_sigmoid":false}
            },
            "made_up_field": 42
        }"#;
        let r: Result<BenchConfig, _> = serde_json::from_str(json);
        assert!(r.is_err(), "deny_unknown_fields should reject extras");
    }
}
