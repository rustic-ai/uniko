//! Schema registration for the uniko cognitive memory system.
//!
//! Provides [`register_schema`] which registers all 25 node types, 54 edge
//! types, and all indexes with the database.  The call is idempotent: running
//! it multiple times on the same database has no visible effect beyond the
//! first.

pub mod btic;
pub mod constants;

mod actions;
mod artifact_content;
mod artifacts;
mod blocks;
mod chunks;
mod consolidation;
mod entities;
mod episodes;
mod facts;
mod goals;
mod kb_stats;
mod messages;
mod observations;
mod organization;
mod pages;
mod participants;
mod patterns;
mod procedures;
mod rules;
mod sessions;
mod sources;
mod summaries;
mod topics;

use uni_db::api::schema::EmbeddingCfg;
use uni_db::{IndexType, Uni, VectorAlgo, VectorIndexCfg};

use crate::config::UnikoConfig;
use crate::error::UnikoError;

pub use constants::{edges, labels};

/// Dense embedding model alias used by uni-db's Xervo runtime (`local/onnx`).
///
/// Serves every lone-dense auto-embed column (Message, Summary, …), all
/// computed embeddings (`kb.embed`), and query embedding. Registered with
/// `ModelTask::Embed`.
pub const EMBED_ALIAS: &str = "embed/default";

/// Single-pass hybrid alias (dense + sparse + ColBERT), registered with
/// `ModelTask::EmbedHybrid` only when the embedder exposes the heads.
///
/// Used exclusively by Chunk/Observation, whose dense, `sparse_embedding`,
/// and `colbert_embedding` columns share it so one `EmbedHybrid` pass fills
/// all three. A hybrid model implements only `HybridEmbeddingModel`, so it
/// CANNOT also back the lone-dense columns on [`EMBED_ALIAS`] — they must be
/// separate aliases (uni-db routes lone-dense groups through `EmbeddingModel`,
/// which the hybrid model does not implement).
pub const HYBRID_EMBED_ALIAS: &str = "embed/hybrid";

/// NLP ONNX model alias for multi-task inference (NER, POS, dep, CLS).
pub const NLP_ALIAS: &str = "nlp/default";

/// Cross-encoder reranker alias (ONNX `local/onnx`, registered only when enabled).
pub const RERANK_ALIAS: &str = "rerank/default";

/// Pipeline-OCR alias (ONNX `local/onnx`, registered only when enabled).
///
/// Drives the `Ocr` tier of tiered PDF extraction (`uni-xervo-pdf`).
pub const OCR_ALIAS: &str = "ocr/default";

/// Build a vector index from config (no auto-embed).
///
/// Used for node types whose embeddings are computed in application
/// code (Entity, Fact, Episode, etc.).
pub(crate) fn vector_index(config: &UnikoConfig) -> VectorIndexCfg {
    VectorIndexCfg {
        algorithm: config.vector_algorithm.to_uni_algo(),
        metric: config.vector_metric.to_uni_metric(),
        embedding: None,
    }
}

/// Build an `EmbeddingCfg` for `alias` over `source_property`.
fn embedding_cfg(alias: &str, source_property: &str, config: &UnikoConfig) -> EmbeddingCfg {
    EmbeddingCfg {
        alias: alias.to_string(),
        source_properties: vec![source_property.to_string()],
        batch_size: config.embedding.batch_size,
        document_prefix: config.embedding.document_prefix.clone(),
        query_prefix: config.embedding.query_prefix.clone(),
    }
}

/// The embed-alias config shared by the hybrid index group ([`HYBRID_EMBED_ALIAS`]).
///
/// The dense, sparse, and multi-vector indexes on Chunk/Observation must
/// point at the SAME alias and source property for uni-db to fuse them into
/// one single-pass `EmbedHybrid` inference; this builder keeps them identical.
fn hybrid_embedding_cfg(source_property: &str, config: &UnikoConfig) -> EmbeddingCfg {
    embedding_cfg(HYBRID_EMBED_ALIAS, source_property, config)
}

/// Build a dense auto-embed vector index on the plain [`EMBED_ALIAS`].
///
/// For lone-dense auto-embed nodes (Message, Summary, …). uni-db computes
/// and stores the embedding on create/update via the `Embed`-task model.
pub(crate) fn auto_embed_vector_index(
    source_property: &str,
    config: &UnikoConfig,
) -> VectorIndexCfg {
    VectorIndexCfg {
        algorithm: config.vector_algorithm.to_uni_algo(),
        metric: config.vector_metric.to_uni_metric(),
        embedding: Some(embedding_cfg(EMBED_ALIAS, source_property, config)),
    }
}

/// Build the dense auto-embed index for a hybrid node on [`HYBRID_EMBED_ALIAS`].
///
/// Same ANN config as [`auto_embed_vector_index`] but on the hybrid alias, so
/// the dense column joins the sparse + ColBERT columns in one `EmbedHybrid`
/// group. Used by Chunk/Observation when the embedder is hybrid.
pub(crate) fn auto_embed_hybrid_vector_index(
    source_property: &str,
    config: &UnikoConfig,
) -> VectorIndexCfg {
    VectorIndexCfg {
        algorithm: config.vector_algorithm.to_uni_algo(),
        metric: config.vector_metric.to_uni_metric(),
        embedding: Some(hybrid_embedding_cfg(source_property, config)),
    }
}

/// Build a learned-sparse auto-embed index sharing the embed alias.
///
/// Declared with the same alias and source as [`auto_embed_vector_index`]
/// so uni-db treats the dense and sparse columns as one single-pass
/// `EmbedHybrid` group. Sparse similarity is always dot product, so the
/// index carries no algorithm or metric.
///
/// # Panics
///
/// Panics if [`crate::config::EmbeddingConfig::sparse_dimensions`] is
/// `None`. Callers gate on it — the schema only adds the column for hybrid
/// embedders, so reaching this with a dense-only config is a bug.
pub(crate) fn auto_embed_sparse_index(source_property: &str, config: &UnikoConfig) -> IndexType {
    let dimensions = config
        .embedding
        .sparse_dimensions
        .expect("auto_embed_sparse_index requires embedding.sparse_dimensions");
    IndexType::sparse_with_embedding(dimensions, hybrid_embedding_cfg(source_property, config))
}

/// Build an exact multi-vector (ColBERT) auto-embed index for MaxSim rerank.
///
/// Uses `VectorAlgo::Flat` — the per-token vectors feed only the top-k
/// MaxSim rerank, never first-stage retrieval, so a MUVERA ANN index would
/// add build cost for no benefit. Shares the embed alias to join the hybrid
/// group.
pub(crate) fn auto_embed_multivector_index(
    source_property: &str,
    config: &UnikoConfig,
) -> VectorIndexCfg {
    VectorIndexCfg {
        algorithm: VectorAlgo::Flat,
        metric: config.vector_metric.to_uni_metric(),
        embedding: Some(hybrid_embedding_cfg(source_property, config)),
    }
}

/// Register the complete uniko schema with the database.
///
/// Idempotent: calling multiple times is safe.  Existing labels, properties,
/// edge types, and indexes are silently skipped by uni-db.
///
/// # Errors
///
/// Returns [`UnikoError::Schema`] if any registration step fails for a reason
/// other than "already exists".
pub async fn register_schema(db: &Uni, config: &UnikoConfig) -> crate::Result<()> {
    let builder = db.schema();

    // ── Phase 1: labels (properties + indexes) ──
    let builder = participants::register_labels(builder);
    let builder = goals::register_labels(builder, config);
    let builder = sessions::register_labels(builder, config);
    let builder = sources::register_labels(builder, config);
    let builder = messages::register_labels(builder, config);
    let builder = actions::register_labels(builder, config);
    let builder = episodes::register_labels(builder, config);
    let builder = artifacts::register_labels(builder, config);
    let builder = artifact_content::register_labels(builder);
    let builder = chunks::register_labels(builder, config);
    let builder = pages::register_labels(builder, config);
    let builder = blocks::register_labels(builder, config);
    let builder = entities::register_labels(builder, config);
    let builder = observations::register_labels(builder, config);
    let builder = facts::register_labels(builder, config);
    let builder = topics::register_labels(builder, config);
    let builder = summaries::register_labels(builder, config);
    let builder = procedures::register_labels(builder, config);
    let builder = patterns::register_labels(builder);
    let builder = rules::register_labels(builder);
    let builder = consolidation::register_labels(builder);
    let builder = organization::register_labels(builder);
    let builder = kb_stats::register_labels(builder);

    // ── Phase 2: edge types ──
    let builder = goals::register_edges(builder);
    let builder = sessions::register_edges(builder);
    let builder = sources::register_edges(builder);
    let builder = messages::register_edges(builder);
    let builder = actions::register_edges(builder);
    let builder = episodes::register_edges(builder);
    let builder = artifacts::register_edges(builder);
    let builder = artifact_content::register_edges(builder);
    let builder = chunks::register_edges(builder);
    let builder = pages::register_edges(builder);
    let builder = blocks::register_edges(builder);
    let builder = entities::register_edges(builder);
    let builder = observations::register_edges(builder);
    let builder = facts::register_edges(builder);
    let builder = topics::register_edges(builder);
    let builder = summaries::register_edges(builder);
    let builder = procedures::register_edges(builder);
    let builder = rules::register_edges(builder);
    let builder = consolidation::register_edges(builder);
    let builder = organization::register_edges(builder);

    builder
        .apply()
        .await
        .map_err(|e| UnikoError::Schema(e.to_string()))
}
