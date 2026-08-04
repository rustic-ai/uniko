/// Runtime configuration for the uniko cognitive memory system.
///
/// All fields have spec-mandated defaults. Use `UnikoConfig::default()` and override
/// individual fields as needed. Call `validate()` before use to catch constraint violations.
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::blob_store::BlobStorage;
use crate::error::{Result, UnikoError};

// ── Embedding + Vector index config types ─────────────────────────

/// Embedding model selection.
///
/// Controls which ONNX embedding model is used for auto-embedding and
/// computed embeddings.  Dimensions must match the model's output.
///
/// Use [`EmbeddingConfig::nomic_v15`] (768d, recommended) or
/// [`EmbeddingConfig::minilm_l6_v2`] (384d, legacy) for presets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Embedding model identifier (e.g., `"NomicEmbedTextV15"`). Resolved
    /// by the `local/onnx` provider in uni-db's xervo catalog.
    pub model_id: String,
    /// Output vector dimensions (must match model).
    pub dimensions: usize,
    /// Batch size for auto-embed operations.
    pub batch_size: usize,
    /// Prefix prepended to documents before embedding.
    ///
    /// Nomic Embed Text v1.5 requires `"search_document: "` for optimal
    /// retrieval. Models without task prefixes (e.g., MiniLM) use `None`.
    pub document_prefix: Option<String>,
    /// Prefix prepended to queries before embedding.
    ///
    /// Nomic Embed Text v1.5 requires `"search_query: "` for optimal
    /// retrieval. Models without task prefixes (e.g., MiniLM) use `None`.
    pub query_prefix: Option<String>,
    /// Optional override for ONNX execution providers when registering
    /// the embedder with `local/onnx`. Values are xervo's EP string ids
    /// (e.g. `["cuda", "cpu"]`, `["coreml", "cpu"]`, `["cpu"]`). When
    /// `None`, [`embed_catalog`](crate::storage::embed_catalog) picks
    /// a feature-aware default (CUDA → CPU on `gpu-cuda`, CoreML → CPU
    /// on `gpu-metal`, CPU otherwise).
    ///
    /// Ignored when `provider != "local/onnx"`.
    #[serde(default)]
    pub execution_providers: Option<Vec<String>>,
    /// Xervo provider id used for the embed alias.  Defaults to
    /// `"local/onnx"` (the bundled ONNX provider).  Set to
    /// `"remote/openai"` to route embeddings through an
    /// OpenAI-compatible endpoint (production OpenAI, or a litellm
    /// proxy serving `/v1/embeddings`).  Other xervo embedding
    /// providers (`remote/voyageai`, `remote/cohere`,
    /// `remote/azure-openai`) work the same way; the catalog passes
    /// `provider_options` through verbatim.
    #[serde(default = "default_embed_provider")]
    pub provider: String,
    /// Provider-specific options merged into the alias's `options`
    /// JSON at catalog-build time.  For `remote/openai` this is where
    /// `base_url` lands so xervo points at the right endpoint
    /// (e.g. `{"base_url": "http://litellm-cloud:4000/v1"}`).
    #[serde(default)]
    pub provider_options: Option<serde_json::Value>,
    /// Learned-sparse head dimension (vocabulary size), or `None` for
    /// dense-only models.
    ///
    /// `Some(n)` opts the embed alias into single-pass hybrid mode: the
    /// `local/onnx` provider runs an `EmbedHybrid` forward pass that fills
    /// a `SPARSE_VECTOR(n)` column alongside the dense `embedding` (see
    /// [`embed_catalog`](crate::storage::embed_catalog)). Only BGE-M3
    /// (`aapot/bge-m3-onnx`, vocab 250002) currently exposes this head.
    #[serde(default)]
    pub sparse_dimensions: Option<usize>,
    /// ColBERT / late-interaction per-token dimension, or `None`.
    ///
    /// `Some(d)` adds a `List(Vector(d))` multi-vector column filled by the
    /// same `EmbedHybrid` pass, used for MaxSim reranking. BGE-M3 emits
    /// 1024-d per-token vectors.
    #[serde(default)]
    pub multivector_dimensions: Option<usize>,
}

fn default_embed_provider() -> String {
    "local/onnx".into()
}

impl EmbeddingConfig {
    /// Nomic Embed Text v1.5 — 768d, 8192 context, recommended.
    pub fn nomic_v15() -> Self {
        Self {
            model_id: "NomicEmbedTextV15".into(),
            sparse_dimensions: None,
            multivector_dimensions: None,
            dimensions: 768,
            batch_size: 32,
            document_prefix: Some("search_document: ".into()),
            query_prefix: Some("search_query: ".into()),
            execution_providers: None,
            provider: default_embed_provider(),
            provider_options: None,
        }
    }

    /// Nomic Embed Text v1.5 quantized — 768d, faster, lower memory.
    pub fn nomic_v15_quantized() -> Self {
        Self {
            model_id: "NomicEmbedTextV15Q".into(),
            sparse_dimensions: None,
            multivector_dimensions: None,
            dimensions: 768,
            batch_size: 32,
            document_prefix: Some("search_document: ".into()),
            query_prefix: Some("search_query: ".into()),
            execution_providers: None,
            provider: default_embed_provider(),
            provider_options: None,
        }
    }

    /// All-MiniLM-L6-v2 — 384d, legacy default for existing databases.
    pub fn minilm_l6_v2() -> Self {
        Self {
            model_id: "AllMiniLML6V2".into(),
            sparse_dimensions: None,
            multivector_dimensions: None,
            dimensions: 384,
            batch_size: 32,
            document_prefix: None,
            query_prefix: None,
            execution_providers: None,
            provider: default_embed_provider(),
            provider_options: None,
        }
    }

    /// BAAI/bge-small-en-v1.5 — 384d, BERT-based, MTEB-strong general
    /// retriever. Uses a query-side prefix only (documents go in raw).
    pub fn bge_small_en_v15() -> Self {
        Self {
            model_id: "BGESmallENV15".into(),
            sparse_dimensions: None,
            multivector_dimensions: None,
            dimensions: 384,
            batch_size: 32,
            document_prefix: None,
            query_prefix: Some("Represent this sentence for searching relevant passages: ".into()),
            execution_providers: None,
            provider: default_embed_provider(),
            provider_options: None,
        }
    }

    /// BAAI/bge-large-en-v1.5 — 1024d, BERT-large-based, top-tier MTEB
    /// quality. Same query-prefix convention as BGE-small. ~10× the
    /// parameter count of BGE-small; expect 2-3× slower per-query
    /// embedding.
    pub fn bge_large_en_v15() -> Self {
        Self {
            model_id: "BGELargeENV15".into(),
            sparse_dimensions: None,
            multivector_dimensions: None,
            dimensions: 1024,
            batch_size: 16,
            document_prefix: None,
            query_prefix: Some("Represent this sentence for searching relevant passages: ".into()),
            execution_providers: None,
            provider: default_embed_provider(),
            provider_options: None,
        }
    }

    /// `onnx-community/embeddinggemma-300m-ONNX` — 300M params, 768d,
    /// Google's Gemma-3-based retrieval embedder (Sentence-Transformers
    /// export by the ONNX community).
    ///
    /// Resolves through uni-xervo's `local/onnx` HF-Hub fallback path
    /// because the embedder isn't in the bundled preset table.  We
    /// therefore supply every ONNX option the resolver needs via
    /// `provider_options`:
    ///
    /// - `artifact: "onnx/model.onnx"` — standard onnx-community export path.
    /// - `pooling: "mean"` — Sentence-Transformers convention for the
    ///   official EmbeddingGemma checkpoint (the SBERT modules config
    ///   ships a mean-pool head).
    /// - `dimensions: 768` — full Matryoshka head; truncate via
    ///   per-call dimensions arg if you need 512/256/128.
    /// - `token_type_ids: false` — Gemma's tokenizer doesn't produce
    ///   them.
    /// - `max_seq_len: 2048` — Gemma supports 8K but most retrieval
    ///   chunks land under 2K; raise via override if needed.
    ///
    /// Document / query prefixes follow EmbeddingGemma's task-prompt
    /// template (the model was trained on these literal strings).
    pub fn embedding_gemma_300m() -> Self {
        Self {
            model_id: "onnx-community/embeddinggemma-300m-ONNX".into(),
            sparse_dimensions: None,
            multivector_dimensions: None,
            dimensions: 768,
            batch_size: 16,
            document_prefix: Some("title: none | text: ".into()),
            query_prefix: Some("task: search result | query: ".into()),
            execution_providers: None,
            provider: default_embed_provider(),
            provider_options: Some(serde_json::json!({
                "artifact": "onnx/model.onnx",
                "pooling": "mean",
                "dimensions": 768,
                "token_type_ids": false,
                "max_seq_len": 2048,
                "normalize": true,
            })),
        }
    }

    /// `google/embeddinggemma-300m` served via uni-xervo's
    /// `local/mistralrs` provider on GPU.
    ///
    /// Same model weights as [`Self::embedding_gemma_300m`], but
    /// loaded through mistralrs (candle-cuda) instead of ORT.
    /// mistralrs's `EmbeddingModelBuilder` handles pooling internally
    /// and probes the embedding dimension at load time, so we only
    /// need to pass dtype here.  The Sentence-Transformers task
    /// prompts (`document_prefix` / `query_prefix`) are applied at
    /// xervo's embed call site, so they work identically across
    /// runtimes.
    ///
    /// `dtype: "f32"` because mistralrs's candle backend on this stack
    /// dispatches embeddings on CPU under some build configurations
    /// (no candle-cuda compiled in, or no CUDA device visible) and
    /// F16 CPU inference produces NaN outputs.  F32 always works.
    /// Override `provider_options.dtype` if your build has working
    /// candle-cuda dispatch.
    pub fn embedding_gemma_300m_mistralrs() -> Self {
        Self {
            model_id: "google/embeddinggemma-300m".into(),
            sparse_dimensions: None,
            multivector_dimensions: None,
            dimensions: 768,
            batch_size: 16,
            document_prefix: Some("title: none | text: ".into()),
            query_prefix: Some("task: search result | query: ".into()),
            execution_providers: None,
            provider: "local/mistralrs".into(),
            provider_options: Some(serde_json::json!({
                "dtype": "f32",
            })),
        }
    }

    /// BAAI/bge-m3 (`aapot/bge-m3-onnx`) — single-pass hybrid: 1024-d
    /// dense + 250002-d learned-sparse + 1024-d ColBERT.
    ///
    /// Opts the embed alias into `EmbedHybrid`: one ONNX forward pass
    /// fills the dense `embedding`, the `sparse_embedding`
    /// (`SPARSE_VECTOR(250002)`), and the `colbert_embedding`
    /// (`List(Vector(1024))`) columns wherever a node declares them with
    /// the shared embed alias. Sparse adds a learned-lexical retrieval
    /// channel; ColBERT enables MaxSim reranking. BGE-M3 takes raw text
    /// (no task prefixes). The sparse vocabulary is XLM-RoBERTa's 250002
    /// tokens.
    #[must_use]
    pub fn bge_m3() -> Self {
        Self {
            model_id: "aapot/bge-m3-onnx".into(),
            // 250002 = XLM-RoBERTa vocabulary; BGE-M3's sparse head emits
            // one weight per vocab token. Must match the xervo preset.
            sparse_dimensions: Some(250002),
            // 1024-d per-token ColBERT vectors, same width as the dense head.
            multivector_dimensions: Some(1024),
            dimensions: 1024,
            batch_size: 16,
            document_prefix: None,
            query_prefix: None,
            execution_providers: None,
            provider: default_embed_provider(),
            provider_options: None,
        }
    }

    /// Look up a preset by short name.  Centralised here so bench /
    /// CLI tools don't each maintain their own `match` over the
    /// preset list.  Accepts a few aliases for the longer names.
    ///
    /// Returns `None` for unknown names — callers print their own
    /// "expected: …" error so users see the canonical list.
    #[must_use]
    pub fn preset(name: &str) -> Option<Self> {
        Some(match name {
            "nomic" => Self::nomic_v15(),
            "nomic-q" | "nomic-quantized" => Self::nomic_v15_quantized(),
            "minilm" => Self::minilm_l6_v2(),
            "bge-small" => Self::bge_small_en_v15(),
            "bge-large" => Self::bge_large_en_v15(),
            "bge-m3" => Self::bge_m3(),
            "embeddinggemma" | "gemma" => Self::embedding_gemma_300m(),
            "embeddinggemma-mistralrs" | "gemma-mistralrs" => {
                Self::embedding_gemma_300m_mistralrs()
            }
            _ => return None,
        })
    }
}

/// NLP cascade model selection.
///
/// The bench's NER + observation extraction steps invoke a Raw-task
/// xervo alias backed by uni-db's `local/onnx` provider.  Defaults
/// match the legacy hardcoded values that lived in
/// `storage/mod.rs::embed_catalog`; surfacing them here lets a bench
/// profile swap the cascade (e.g. `*-small` vs `*-xsmall`, FP32 vs
/// INT8) without a code change.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NlpConfig {
    /// HuggingFace model id for the NLP cascade.
    #[serde(default = "default_nlp_model_id")]
    pub model_id: String,
    /// Which ONNX artifact under `model_id/` to load.
    #[serde(default = "default_nlp_artifact")]
    pub artifact: String,
    /// Maximum batch size for the ONNX session.
    #[serde(default = "default_nlp_max_batch_size")]
    pub max_batch_size: usize,
    /// Optional execution-provider override.  `None` → feature-aware
    /// default (cuda+cpu on `gpu-cuda`, coreml+cpu on `gpu-metal`,
    /// cpu otherwise).
    #[serde(default)]
    pub execution_providers: Option<Vec<String>>,
}

impl Default for NlpConfig {
    fn default() -> Self {
        Self {
            model_id: default_nlp_model_id(),
            artifact: default_nlp_artifact(),
            max_batch_size: default_nlp_max_batch_size(),
            execution_providers: None,
        }
    }
}

fn default_nlp_model_id() -> String {
    "dragonscale-ai/kniv-deberta-nlp-base-en-xsmall".to_string()
}

fn default_nlp_artifact() -> String {
    "onnx/cascade-int8.onnx".to_string()
}

fn default_nlp_max_batch_size() -> usize {
    16
}

/// Cross-encoder reranker selection.
///
/// When `enabled`, recall fuses RRF candidates and then re-scores the
/// top `top_n` items with a cross-encoder via uni-db's `local/onnx`
/// provider.  Disabled by default to keep CPU-only CI cheap.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankerConfig {
    /// Whether to register and invoke the reranker.
    pub enabled: bool,
    /// HuggingFace model id for the cross-encoder ONNX export.
    pub model_id: String,
    /// Number of top RRF candidates to send to the reranker.
    ///
    /// Must be `>= recall_limit` when enabled; otherwise truncation
    /// would discard items the reranker hasn't seen.
    pub top_n: usize,
    /// Apply sigmoid to raw cross-encoder logits to map to `[0, 1]`.
    pub apply_sigmoid: bool,
    /// Optional override for ONNX execution providers when registering
    /// the reranker with `local/onnx`. Same semantics as the embedder
    /// equivalent on [`EmbeddingConfig`]. `None` → feature-aware default.
    #[serde(default)]
    pub execution_providers: Option<Vec<String>>,
    /// Reranker code path. `"cross-encoder"` (default) loads BERT-family
    /// encoders that emit a relevance logit per `(query, doc)` pair (e.g.
    /// `BAAI/bge-reranker-base`, `cross-encoder/ms-marco-MiniLM-L-6-v2`).
    /// `"generative"` loads a decoder-LM reranker that scores yes/no via
    /// next-token logits (e.g. `onnx-community/Qwen3-Reranker-0.6B-ONNX`).
    /// Both route through xervo's `local/onnx` reranker.
    ///
    /// `"colbert"` is handled entirely in uniko: instead of a reranker
    /// model it re-scores the top candidates by ColBERT MaxSim against
    /// the query's multi-vector embedding (requires a hybrid embedder
    /// with [`EmbeddingConfig::multivector_dimensions`] set). It registers
    /// no rerank alias. Any other value triggers a runtime error from xervo.
    #[serde(default = "default_reranker_style")]
    pub style: String,
}

fn default_reranker_style() -> String {
    "cross-encoder".to_string()
}

impl RerankerConfig {
    /// BAAI/bge-reranker-base — 278M params, accurate, CPU-feasible.
    pub fn bge_base() -> Self {
        Self {
            enabled: false,
            model_id: "BAAI/bge-reranker-base".into(),
            top_n: 50,
            apply_sigmoid: true,
            execution_providers: None,
            style: default_reranker_style(),
        }
    }

    /// MS-MARCO MiniLM-L6-v2 — 22M params, ~12× faster, lower accuracy.
    pub fn minilm_l6() -> Self {
        Self {
            enabled: false,
            model_id: "cross-encoder/ms-marco-MiniLM-L-6-v2".into(),
            top_n: 50,
            apply_sigmoid: true,
            execution_providers: None,
            style: default_reranker_style(),
        }
    }
}

impl Default for RerankerConfig {
    /// Default: `cross-encoder/ms-marco-MiniLM-L-6-v2`, **enabled**.
    ///
    /// MiniLM is 22M params and ~12× faster than BGE-reranker-base,
    /// so default-on costs ~50-100ms per query — small relative to
    /// the rest of recall and the rerank quality lift on multi-hop /
    /// open-domain.  Disable via `RerankerConfig { enabled: false, ..
    /// Default::default() }` or `--no-reranker` on the bench CLI.
    fn default() -> Self {
        Self {
            enabled: true,
            ..Self::minilm_l6()
        }
    }
}

/// Pipeline-OCR model selection for tiered PDF extraction.
///
/// When `enabled`, an `ocr/default` [`OcrModel`] alias is registered with
/// uni-db's `local/onnx` provider and used to drive the `Ocr` tier of
/// `uni-xervo-pdf` (DBNet detection + CRNN/CTC recognition). Disabled by
/// default to keep CPU-only CI cheap and avoid the model download.
///
/// Defaults target the English PP-OCRv5 export `monkt/paddleocr-onnx`
/// (Apache-2.0): server detection plus the English mobile recognizer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OcrConfig {
    /// Whether to register and use the OCR model.
    pub enabled: bool,
    /// HuggingFace model id hosting the detection + recognition ONNX exports.
    pub model_id: String,
    /// Recognizer ONNX path within `model_id/` (CRNN/CTC).
    pub rec_artifact: String,
    /// Character-dictionary path within `model_id/` (one token per line).
    pub char_dict_path: String,
    /// Detector ONNX path within `model_id/` (DBNet). Enables the two-stage
    /// detect → crop → recognize path with real per-block bounding boxes.
    pub det_artifact: String,
    /// Recognizer input height in pixels.
    pub image_height: u32,
    /// Recognizer input width in pixels.
    pub image_width: u32,
    /// Recognizer normalization: `"imagenet"` or `"siglip"`. Defaults to
    /// `"siglip"`, which is what the PP-OCRv5 export expects.
    pub normalization: String,
    /// Optional override for ONNX execution providers. `None` → feature-aware
    /// default (cpu unless a GPU feature is enabled), same as the embedder.
    #[serde(default)]
    pub execution_providers: Option<Vec<String>>,
}

impl Default for OcrConfig {
    /// English PP-OCRv5 (`monkt/paddleocr-onnx`), **disabled**.
    fn default() -> Self {
        Self {
            enabled: false,
            model_id: "monkt/paddleocr-onnx".to_string(),
            rec_artifact: "languages/english/rec.onnx".to_string(),
            char_dict_path: "languages/english/dict.txt".to_string(),
            det_artifact: "detection/v5/det.onnx".to_string(),
            image_height: 48,
            image_width: 320,
            normalization: "siglip".to_string(),
            execution_providers: None,
        }
    }
}

/// Vector index algorithm and quantization strategy.
///
/// Controls how embedding vectors are indexed for similarity search.
/// Choose based on dataset scale and recall/memory tradeoff.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VectorAlgorithm {
    /// HNSW with scalar quantization (default, best all-round).
    HnswSq {
        /// Max connections per node (higher = better recall, more memory).
        m: u32,
        /// Build-time search width (higher = better index quality, slower build).
        ef_construction: u32,
    },
    /// HNSW without quantization (maximum recall, ~4x more memory).
    HnswFlat {
        /// Max connections per node.
        m: u32,
        /// Build-time search width.
        ef_construction: u32,
    },
    /// HNSW with product quantization (large-scale, ~2-5% recall loss).
    HnswPq {
        /// Max connections per node.
        m: u32,
        /// Build-time search width.
        ef_construction: u32,
        /// Number of sub-vector segments for PQ (e.g., 48 for 768d → 16d each).
        sub_vectors: u32,
    },
    /// IVF with scalar quantization.
    IvfSq {
        /// Number of Voronoi partitions.
        partitions: u32,
    },
    /// IVF with product quantization.
    IvfPq {
        /// Number of Voronoi partitions.
        partitions: u32,
        /// Number of sub-vector segments.
        sub_vectors: u32,
    },
    /// IVF with residual quantization (best compression ratio).
    IvfRq {
        /// Number of Voronoi partitions.
        partitions: u32,
        /// Bits per dimension (default 8 if None).
        num_bits: Option<u8>,
    },
}

impl VectorAlgorithm {
    /// Convert to the uni-db `VectorAlgo` enum.
    pub(crate) fn to_uni_algo(&self) -> uni_db::VectorAlgo {
        match self {
            Self::HnswSq { m, ef_construction } => uni_db::VectorAlgo::HnswSq {
                m: *m,
                ef_construction: *ef_construction,
                partitions: None,
            },
            Self::HnswFlat { m, ef_construction } => uni_db::VectorAlgo::HnswFlat {
                m: *m,
                ef_construction: *ef_construction,
                partitions: None,
            },
            Self::HnswPq {
                m,
                ef_construction,
                sub_vectors,
            } => uni_db::VectorAlgo::HnswPq {
                m: *m,
                ef_construction: *ef_construction,
                sub_vectors: *sub_vectors,
                partitions: None,
            },
            Self::IvfSq { partitions } => uni_db::VectorAlgo::IvfSq {
                partitions: *partitions,
            },
            Self::IvfPq {
                partitions,
                sub_vectors,
            } => uni_db::VectorAlgo::IvfPq {
                partitions: *partitions,
                sub_vectors: *sub_vectors,
            },
            Self::IvfRq {
                partitions,
                num_bits,
            } => uni_db::VectorAlgo::IvfRq {
                partitions: *partitions,
                num_bits: *num_bits,
            },
        }
    }
}

/// Distance metric for vector similarity search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorMetricChoice {
    /// Cosine similarity (default, best for normalized embeddings).
    Cosine,
    /// Euclidean (L2) distance.
    L2,
    /// Dot product (for unnormalized embeddings).
    Dot,
}

impl VectorMetricChoice {
    /// Convert to the uni-db `VectorMetric` enum.
    pub(crate) fn to_uni_metric(self) -> uni_db::VectorMetric {
        match self {
            Self::Cosine => uni_db::VectorMetric::Cosine,
            Self::L2 => uni_db::VectorMetric::L2,
            Self::Dot => uni_db::VectorMetric::Dot,
        }
    }
}

// ── Main config ───────────────────────────────────────────────────

/// Configuration for all uniko runtime parameters.
///
/// Default values match the uniko specification v6.0 exactly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct UnikoConfig {
    // External configuration files
    /// Path to xervo model catalog JSON. When set, models are loaded from
    /// this file instead of the built-in catalog. Use `None` for defaults.
    #[serde(default)]
    pub catalog_path: Option<PathBuf>,
    /// Path to schema JSON. When set, schema is loaded from this file
    /// instead of the builder-based registration. Use `None` for defaults.
    #[serde(default)]
    pub schema_path: Option<PathBuf>,
    /// Path to observation extraction rules YAML. When set, the
    /// observation pipeline loads patterns from this file instead of
    /// the bundled `english.yml`. Lets us iterate on extraction
    /// patterns without recompiling.
    #[serde(default)]
    pub observation_rules_path: Option<PathBuf>,

    // Embedding + vector index
    /// Embedding model selection (controls dimensions and model ID).
    pub embedding: EmbeddingConfig,
    /// Cross-encoder reranker selection (disabled by default).
    #[serde(default)]
    pub reranker: RerankerConfig,
    /// NLP cascade selection (model id + ONNX artifact).
    #[serde(default)]
    pub nlp: NlpConfig,
    /// Pipeline-OCR selection for tiered PDF extraction (disabled by default).
    #[serde(default)]
    pub ocr: OcrConfig,
    /// Vector index algorithm and quantization strategy.
    pub vector_algorithm: VectorAlgorithm,
    /// Distance metric for similarity search.
    pub vector_metric: VectorMetricChoice,

    /// Blob backend for `:ArtifactContent` bytes. Persisted in
    /// `:KnowledgeBaseStats.blob_storage` on first open; reopening
    /// with a different variant is a hard error (no implicit
    /// migration). Default [`BlobStorage::Lance`].
    #[serde(default)]
    pub blob_storage: BlobStorage,

    // Pipeline capacities
    /// Bounded channel capacity for the ingest worker.
    pub ingest_queue_capacity: usize,
    /// Bounded channel capacity for the consolidation worker.
    pub consolidation_queue_capacity: usize,

    // Consolidation triggers
    /// Number of observations that trigger consolidation.
    pub consolidation_threshold: u32,
    /// Seconds between periodic consolidation runs.
    pub consolidation_interval_secs: u64,

    // Retry policy
    /// Maximum retry attempts for retryable operations.
    pub retry_max_attempts: u32,
    /// Initial delay in milliseconds before first retry (exponential backoff base).
    pub retry_initial_delay_ms: u64,
    /// Maximum delay in milliseconds between retries (backoff cap).
    pub retry_max_delay_ms: u64,

    // Circuit breaker
    /// Number of consecutive failures before the circuit breaker opens.
    pub circuit_failure_threshold: u32,
    /// Milliseconds the circuit breaker stays open before probing.
    pub circuit_recovery_ms: u64,

    // Chunking thresholds
    /// Token count above which messages are chunked.
    pub message_chunk_threshold: usize,
    /// Token count above which action outputs overflow to Artifact nodes.
    pub action_output_artifact_threshold: usize,

    // Chunk sizing
    /// Maximum tokens per chunk.
    pub max_chunk_tokens: usize,
    /// Minimum tokens per chunk (fragments below this are merged).
    pub min_chunk_tokens: usize,
    /// Overlap tokens between adjacent chunks (0 = auto: 10% of max, capped at 50).
    pub chunk_overlap_tokens: usize,

    // NLP cascade
    /// Whether to compute SRL frames (one extra ONNX forward per VERB
    /// per sentence). Phase A landed inert SRL plumbing — when `false`,
    /// `NlpResult.srl_frames` stays empty and downstream extraction
    /// behaves exactly as before. Default `true` so the model's SRL
    /// head is actually used; flip to `false` if profiling shows the
    /// per-verb re-forward cost is unacceptable for a given workload.
    pub nlp_srl_enabled: bool,

    // Entity admission (precision)
    /// Strict entity admission: drop NER outputs that aren't genuine named
    /// entities — `Date` (handled by observation temporal anchors),
    /// `Measurement`/`Preference`/`QuotedString` (captured in observation
    /// triples) — and greeting/discourse fragments wrongly grabbed as
    /// `Person`, and gate the `Other` catch-all by confidence. Default
    /// `true`. Set `false` to restore the legacy admit-everything behaviour
    /// (used for A/B comparison). Diagnosed: ~40% of entities were `date`
    /// noise; ~58% non-entities overall.
    pub entity_strict_admission: bool,
    /// Minimum confidence for an `EntityType::Other` span (ONNX
    /// Event/Product/WorkOfArt/Group/Misc) to be admitted as an Entity when
    /// `entity_strict_admission` is on.
    pub entity_other_min_confidence: f64,

    // Recall parameters
    /// Maximum items returned from recall.
    pub recall_limit: usize,
    /// Maximum total tokens in the context bundle.
    pub recall_token_budget: usize,
    /// Minimum fused score for result inclusion.
    pub recall_min_score: f64,
    /// Vector similarity weight in hybrid fusion \[0.0–1.0\].
    pub recall_vector_weight: f64,
    /// BM25 fulltext weight in hybrid fusion \[0.0–1.0\].
    pub recall_bm25_weight: f64,
    /// Add a learned-sparse retrieval channel for Chunk and Observation.
    ///
    /// When `true` (and the embedder is hybrid, i.e.
    /// [`EmbeddingConfig::sparse_dimensions`] is set), recall fans out an
    /// extra `uni.sparse.query` channel per variant whose results join the
    /// existing RRF fusion. Default `false` so the dense + BM25 baseline is
    /// unchanged. Requires a KB ingested with a hybrid embedder.
    #[serde(default)]
    pub recall_sparse_enabled: bool,
    /// Variant labels to enable for multi-query reformulation.
    ///
    /// Empty (the default) selects the **single** POS-stripped `keywords`
    /// variant. Opt into the full set explicitly with
    /// `vec!["keywords", "original", "declarative", "type_anchored"]`.
    /// Single-variant is the default because a measured A/B on full LoCoMo
    /// showed multi-variant regressing evidence% and tripling per-query
    /// latency; see `uniko_memory::recall::intent::build_intent_at`.
    #[serde(default)]
    pub query_variants: Vec<String>,
    /// `k` constant for reciprocal rank fusion across query variants.
    /// Higher values flatten the weight given to top ranks.
    pub rrf_k: f64,
    /// LIMIT applied to each per-variant Cypher query. `None` means
    /// "use `recall_limit`" (the recall layer falls back to it). Set
    /// to a smaller value when running 4 variants to keep the candidate
    /// union manageable.
    #[serde(default)]
    pub recall_per_variant_limit: Option<usize>,
    /// Phase 1 (Compact) contribution strategy: `merge` (default),
    /// `boost`, or `off`.  See `uniko_memory::recall::Phase1Strategy`.
    /// Stored as a string so deserialised configs stay compatible
    /// across feature flags.
    pub phase1_strategy: String,
    /// Multiplicative weight for Fact scores when computing the
    /// session-chunk boost under `phase1_strategy = "boost"`.
    pub phase1_boost_alpha: f64,

    // Memory decay
    /// Half-life in days for importance decay: `importance * exp(-ln(2) / half_life * age_days)`.
    pub half_life_days: f64,
    /// Importance threshold below which nodes are pruned.
    pub prune_below: f64,

    // Recall cascade thresholds
    /// Coverage threshold for Phase 2 (Expand) early exit.
    pub phase2_coverage_threshold: f64,
    /// MMR lambda (relevance vs diversity) for Phase 2 deduplication.
    /// `1.0` = pure relevance order; `0.0` = pure diversity.  Spec §IX
    /// default: 0.7.
    pub phase2_mmr_lambda: f64,
    /// Token-overlap threshold above which a Phase 2 candidate is
    /// dropped as a hard duplicate.  Spec §IX uses cosine > 0.85;
    /// without per-item embeddings we apply the same number to Jaccard
    /// overlap.
    pub phase2_mmr_duplicate_threshold: f64,

    /// Enable the temporal-interval channel in Phase 2 recall.  Fires
    /// only when the query has a parsed temporal phrase.
    pub phase2_temporal_enabled: bool,
    /// Enable the graph spreading-activation channel in Phase 2.  Fires
    /// only when the query has at least one resolvable entity seed.
    pub phase2_graph_enabled: bool,
    /// PPR damping factor for the graph channel.  Standard 0.85.
    pub phase2_graph_damping: f64,
    /// PPR power-iteration cap for the graph channel.
    pub phase2_graph_max_iter: usize,
    /// Per-edge-type weight multipliers for graph propagation.  Empty
    /// map means "use the built-in defaults" (see
    /// `uniko_memory::recall::default_phase2_graph_edge_weights`).
    #[serde(default)]
    pub phase2_graph_edge_weights: std::collections::HashMap<String, f64>,

    /// Enable cosine-similarity clustering of object surface forms in
    /// P4 Consolidation.  When `true` (default), near-duplicate object
    /// phrasings ("adoption agency" / "adoption agencies") collapse
    /// into one vote bucket before mode-voting picks the canonical.
    /// When `false`, fall back to legacy exact-string bucketing.
    pub consolidation_cluster_objects: bool,

    /// Prepend a human-readable month-year prefix (e.g. `"January 2024"`)
    /// to the text embedded for each Fact in P4 Consolidation.  Pulls
    /// the date from `first_observed` (the Fact's `valid_at_lo`) so
    /// temporally-near Facts co-locate in the embedding space.
    /// Asymmetric on purpose: queries are not augmented, since uniko's
    /// Phase-2 temporal channel already handles query-side temporal
    /// cues via BTIC overlap.
    pub consolidation_date_augment_embedding: bool,

    /// Inactivity window, in seconds, after which an open Session is
    /// auto-closed (F14).  A Session whose most recent message is older
    /// than this is stamped with `ended_at` and summarised (F59) on the
    /// next maintenance sweep.  Zero disables auto-close.
    pub session_inactivity_secs: u64,
}

impl Default for UnikoConfig {
    fn default() -> Self {
        Self {
            catalog_path: None,
            schema_path: None,
            observation_rules_path: None,
            embedding: EmbeddingConfig::bge_small_en_v15(),
            reranker: RerankerConfig::default(),
            nlp: NlpConfig::default(),
            ocr: OcrConfig::default(),
            vector_algorithm: VectorAlgorithm::HnswSq {
                m: 16,
                ef_construction: 100,
            },
            vector_metric: VectorMetricChoice::Cosine,
            blob_storage: BlobStorage::default(),
            ingest_queue_capacity: 200,
            consolidation_queue_capacity: 32,
            consolidation_threshold: 20,
            consolidation_interval_secs: 900,
            retry_max_attempts: 3,
            retry_initial_delay_ms: 500,
            retry_max_delay_ms: 30_000,
            circuit_failure_threshold: 5,
            circuit_recovery_ms: 60_000,
            message_chunk_threshold: 1024,
            action_output_artifact_threshold: 256,
            max_chunk_tokens: 256,
            min_chunk_tokens: 32,
            chunk_overlap_tokens: 0, // 0 = auto: 10% of max, capped at 50
            recall_limit: 15,
            recall_token_budget: 8192,
            recall_min_score: 0.001,
            recall_vector_weight: 0.5,
            recall_bm25_weight: 0.5,
            recall_sparse_enabled: false,
            nlp_srl_enabled: true,
            entity_strict_admission: true,
            entity_other_min_confidence: 0.9,
            query_variants: Vec::new(),
            rrf_k: 60.0,
            recall_per_variant_limit: None,
            // `boost` validated 2026-05-14 to beat `merge` by +0.5pts overall
            // and +0.9pts on Multi-hop on conv-26+conv-30 with reranker on,
            // `recall_limit=15`, `reranker_top_n=50`.  Boost only takes effect
            // when `recall_limit < reranker_top_n` so its session-level Fact
            // boosts can push chunks above the truncation cutoff.
            phase1_strategy: "boost".to_string(),
            // Validated 2026-05-14: α=0.6 with `recall_limit=15`,
            // `reranker_top_n=50`, `phase1_strategy=boost` ties or beats α=0.3
            // on every category in conv-26+conv-30, no regressions.
            phase1_boost_alpha: 0.6,
            half_life_days: 30.0,
            prune_below: 0.05,
            phase2_coverage_threshold: 0.65,
            phase2_mmr_lambda: 0.7,
            phase2_mmr_duplicate_threshold: 0.85,
            phase2_temporal_enabled: true,
            phase2_graph_enabled: true,
            phase2_graph_damping: 0.85,
            phase2_graph_max_iter: 30,
            phase2_graph_edge_weights: std::collections::HashMap::new(),
            consolidation_cluster_objects: true,
            consolidation_date_augment_embedding: true,
            // 1 hour matches the FOLLOWED_BY session-continuity heuristic
            // used by record_episode.
            session_inactivity_secs: 3_600,
        }
    }
}

impl UnikoConfig {
    /// Validate configuration constraints.
    ///
    /// Returns `Err(UnikoError::Config)` if any constraint is violated.
    pub fn validate(&self) -> Result<()> {
        if self.embedding.dimensions == 0 {
            return Err(UnikoError::Config(
                "embedding.dimensions must be positive".into(),
            ));
        }

        if self.embedding.batch_size == 0 {
            return Err(UnikoError::Config(
                "embedding.batch_size must be positive".into(),
            ));
        }

        if self.min_chunk_tokens >= self.max_chunk_tokens {
            return Err(UnikoError::Config(format!(
                "min_chunk_tokens ({}) must be less than max_chunk_tokens ({})",
                self.min_chunk_tokens, self.max_chunk_tokens,
            )));
        }

        if self.half_life_days <= 0.0 {
            return Err(UnikoError::Config(format!(
                "half_life_days ({}) must be positive",
                self.half_life_days,
            )));
        }

        if self.prune_below < 0.0 || self.prune_below >= 1.0 {
            return Err(UnikoError::Config(format!(
                "prune_below ({}) must be in [0.0, 1.0)",
                self.prune_below,
            )));
        }

        if self.phase2_coverage_threshold <= 0.0 || self.phase2_coverage_threshold > 1.0 {
            return Err(UnikoError::Config(format!(
                "phase2_coverage_threshold ({}) must be in (0.0, 1.0]",
                self.phase2_coverage_threshold,
            )));
        }

        if self.reranker.enabled && self.reranker.top_n < self.recall_limit {
            return Err(UnikoError::Config(format!(
                "reranker.top_n ({}) must be >= recall_limit ({}) when enabled",
                self.reranker.top_n, self.recall_limit,
            )));
        }

        if self.recall_sparse_enabled && self.embedding.sparse_dimensions.is_none() {
            return Err(UnikoError::Config(
                "recall_sparse_enabled requires a hybrid embedder with sparse_dimensions \
                 set (e.g. the bge-m3 preset)"
                    .into(),
            ));
        }

        if self.reranker.enabled
            && self.reranker.style == "colbert"
            && self.embedding.multivector_dimensions.is_none()
        {
            return Err(UnikoError::Config(
                "reranker.style = \"colbert\" requires a hybrid embedder with \
                 multivector_dimensions set (e.g. the bge-m3 preset)"
                    .into(),
            ));
        }

        if self.retry_initial_delay_ms > self.retry_max_delay_ms {
            return Err(UnikoError::Config(format!(
                "retry_initial_delay_ms ({}) must not exceed retry_max_delay_ms ({})",
                self.retry_initial_delay_ms, self.retry_max_delay_ms,
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_json_deserialises_to_default() {
        // The struct-level `#[serde(default)]` means an empty JSON
        // object must yield the same value as `Default::default()`.
        // Catches accidental removal of either the attribute or the
        // `impl Default` field that would otherwise default to
        // zero/empty under serde's per-field fallback.
        let parsed: UnikoConfig = serde_json::from_str("{}").expect("empty {} parses");
        let expected = UnikoConfig::default();
        assert_eq!(parsed.phase1_strategy, expected.phase1_strategy);
        assert_eq!(parsed.phase1_boost_alpha, expected.phase1_boost_alpha);
        assert_eq!(parsed.rrf_k, expected.rrf_k);
        assert_eq!(parsed.phase2_mmr_lambda, expected.phase2_mmr_lambda);
        assert_eq!(parsed.nlp_srl_enabled, expected.nlp_srl_enabled);
        assert_eq!(
            parsed.phase2_temporal_enabled,
            expected.phase2_temporal_enabled
        );
        assert_eq!(parsed.phase2_graph_enabled, expected.phase2_graph_enabled);
        assert_eq!(parsed.phase2_graph_damping, expected.phase2_graph_damping);
        assert_eq!(parsed.phase2_graph_max_iter, expected.phase2_graph_max_iter);
        assert_eq!(
            parsed.consolidation_cluster_objects,
            expected.consolidation_cluster_objects,
        );
        assert_eq!(
            parsed.consolidation_date_augment_embedding,
            expected.consolidation_date_augment_embedding,
        );
        assert_eq!(parsed.recall_limit, expected.recall_limit);
    }

    #[test]
    fn test_config_defaults() {
        let c = UnikoConfig::default();
        // Embedding defaults to BGE-small-en-v1.5 / 384d (the embedder we
        // actually run; see config/catalog.json + the canonical bench).
        assert_eq!(c.embedding.model_id, "BGESmallENV15");
        assert_eq!(c.embedding.dimensions, 384);
        assert_eq!(c.embedding.batch_size, 32);
        // Vector defaults to HnswSq / Cosine
        assert_eq!(
            c.vector_algorithm,
            VectorAlgorithm::HnswSq {
                m: 16,
                ef_construction: 100
            }
        );
        assert_eq!(c.vector_metric, VectorMetricChoice::Cosine);
        // Pipeline defaults
        assert_eq!(c.ingest_queue_capacity, 200);
        assert_eq!(c.consolidation_queue_capacity, 32);
        assert_eq!(c.consolidation_threshold, 20);
        assert_eq!(c.consolidation_interval_secs, 900);
        assert_eq!(c.retry_max_attempts, 3);
        assert_eq!(c.retry_initial_delay_ms, 500);
        assert_eq!(c.retry_max_delay_ms, 30_000);
        assert_eq!(c.circuit_failure_threshold, 5);
        assert_eq!(c.circuit_recovery_ms, 60_000);
        assert_eq!(c.message_chunk_threshold, 1024);
        assert_eq!(c.action_output_artifact_threshold, 256);
        assert_eq!(c.max_chunk_tokens, 256);
        assert_eq!(c.min_chunk_tokens, 32);
        assert_eq!(c.chunk_overlap_tokens, 0);
        assert_eq!(c.recall_limit, 15);
        assert_eq!(c.recall_token_budget, 8192);
        assert!((c.recall_min_score - 0.001).abs() < f64::EPSILON);
        assert!((c.recall_vector_weight - 0.5).abs() < f64::EPSILON);
        assert!((c.recall_bm25_weight - 0.5).abs() < f64::EPSILON);
        assert_eq!(c.half_life_days, 30.0);
        assert_eq!(c.prune_below, 0.05);
        assert_eq!(c.phase2_coverage_threshold, 0.65);
    }

    #[test]
    fn test_embedding_presets() {
        let nomic = EmbeddingConfig::nomic_v15();
        assert_eq!(nomic.dimensions, 768);

        let minilm = EmbeddingConfig::minilm_l6_v2();
        assert_eq!(minilm.dimensions, 384);

        let nomic_q = EmbeddingConfig::nomic_v15_quantized();
        assert_eq!(nomic_q.dimensions, 768);
        assert_eq!(nomic_q.model_id, "NomicEmbedTextV15Q");
    }

    #[test]
    fn test_config_validation_ok() {
        UnikoConfig::default()
            .validate()
            .expect("default config must be valid");
    }

    #[test]
    fn test_config_validation_fails() {
        // min >= max chunk tokens
        let c = UnikoConfig {
            min_chunk_tokens: 600,
            ..UnikoConfig::default()
        };
        assert!(c.validate().is_err());

        // half_life_days <= 0
        let c = UnikoConfig {
            half_life_days: 0.0,
            ..UnikoConfig::default()
        };
        assert!(c.validate().is_err());

        // prune_below out of range
        let c = UnikoConfig {
            prune_below: 1.0,
            ..UnikoConfig::default()
        };
        assert!(c.validate().is_err());

        // phase2 threshold out of range
        let c = UnikoConfig {
            phase2_coverage_threshold: 1.5,
            ..UnikoConfig::default()
        };
        assert!(c.validate().is_err());

        // retry initial > max
        let c = UnikoConfig {
            retry_initial_delay_ms: 50_000,
            ..UnikoConfig::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let original = UnikoConfig::default();
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: UnikoConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(original, restored);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn proptest_config_roundtrip(
            ingest_cap in 1usize..1000,
            consol_cap in 1usize..100,
            max_chunk in 100usize..2000,
            // Use integer-derived f64 to avoid JSON float precision drift
            half_life_tenths in 1u32..3650,
            dims in proptest::prop_oneof![Just(384usize), Just(768usize)],
        ) {
            let half_life = f64::from(half_life_tenths) / 10.0;
            let embedding = if dims == 384 {
                EmbeddingConfig::minilm_l6_v2()
            } else {
                EmbeddingConfig::nomic_v15()
            };
            let config = UnikoConfig {
                embedding,
                ingest_queue_capacity: ingest_cap,
                consolidation_queue_capacity: consol_cap,
                max_chunk_tokens: max_chunk,
                min_chunk_tokens: max_chunk / 2,
                half_life_days: half_life,
                ..UnikoConfig::default()
            };
            let json = serde_json::to_string(&config).unwrap();
            let restored: UnikoConfig = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(config, restored);
        }
    }
}
