//! Layer 3: Chunk node type.

use uni_db::{DataType, IndexType, ScalarType, SchemaBuilder};

use super::constants::{edges, labels};
use crate::config::UnikoConfig;

pub(crate) fn register_labels<'a>(
    builder: SchemaBuilder<'a>,
    config: &UnikoConfig,
) -> SchemaBuilder<'a> {
    let mut b = builder
        .label(labels::CHUNK)
        // Typed provenance (issue #39), denormalised from the parent
        // Message/Artifact. Chunks are the primary content-bearing recall
        // unit, so a category or source filter that could not reach them
        // would gut recall rather than narrow it.
        .property_nullable("category", DataType::String)
        .property_nullable("source_id", DataType::String)
        // Which revision of that source this record came from (issue
        // #41). Denormalised beside `source_id` so the "is this evidence
        // still current" filter is a property predicate in candidate
        // generation rather than a traversal.
        .property_nullable("revision_id", DataType::String)
        .property("chunk_id", DataType::String)
        .property("text", DataType::String)
        .property_nullable("index", DataType::Int64)
        .property_nullable("start", DataType::Int64)
        .property_nullable("end", DataType::Int64)
        .property_nullable("token_count", DataType::Int64)
        .property_nullable("chunk_type", DataType::String)
        .property_nullable("language", DataType::String)
        .property_nullable("symbol_name", DataType::String)
        .property_nullable("speaker", DataType::String)
        .property_nullable("heading", DataType::String)
        .property_nullable("mime_type", DataType::String)
        // Modality + positioning. `modality` is nullable for migration —
        // existing rows are backfilled to `"text"` by the migration; new
        // ingests always set it explicitly. The image/audio/video
        // positioning fields (`bbox`, `time_start_ms`, etc.) land with
        // Track B together with the binary chunkers that populate them —
        // we defer declaring them until there's a producer. (The uni-db
        // `List<Float32>` → `List<Utf8>` Arrow-inference fallback that
        // once forced this is fixed in 2.1.0; deferring is now a choice.)
        .property_nullable("modality", DataType::String)
        // Tracks which derivation model produced this chunk. NULL for
        // non-derived chunks (e.g., direct text chunking).
        .property_nullable("source_model_version", DataType::String)
        // Soft-forget tombstone (mirrors `:Message.redacted`). `null` =
        // visible; `true` = the owning turn/document was forgotten and the
        // recall post-filter drops this chunk. NULL default → no backfill.
        .property_nullable("redacted", DataType::Bool)
        // Modality-specific scalars (page_number, time bounds, bbox, …)
        // ride in this JSON bag until a Cypher query needs to filter on
        // one — at which point that field gets promoted to its own
        // typed column. A deliberate flexible-schema choice: keeps the
        // common-case text chunk lean and avoids a schema migration per
        // new modality field. (The uni-db `List<T>` inference fallback
        // that originally motivated it is fixed in 2.1.0.)
        .property_nullable("metadata", DataType::CypherValue)
        .property_nullable(
            "embedding",
            DataType::Vector {
                dimensions: config.embedding.dimensions,
            },
        );
    // Hybrid embedders (e.g. bge-m3) add a learned-sparse and a ColBERT
    // column filled by the same single-pass `EmbedHybrid` inference as the
    // dense `embedding` (see `schema::auto_embed_sparse_index` /
    // `auto_embed_multivector_index`). Dense-only embedders skip them, so
    // existing KBs are unchanged.
    if let Some(sparse_dim) = config.embedding.sparse_dimensions {
        b = b.property_nullable(
            "sparse_embedding",
            DataType::SparseVector {
                dimensions: sparse_dim,
            },
        );
    }
    if let Some(mv_dim) = config.embedding.multivector_dimensions {
        b = b.property_nullable(
            "colbert_embedding",
            DataType::List(Box::new(DataType::Vector { dimensions: mv_dim })),
        );
    }
    // When hybrid, the dense `embedding` joins the sparse + ColBERT columns
    // on the EmbedHybrid alias (one pass); otherwise it stays on the plain
    // dense alias.
    let dense_idx = if config.embedding.sparse_dimensions.is_some()
        || config.embedding.multivector_dimensions.is_some()
    {
        super::auto_embed_hybrid_vector_index("text", config)
    } else {
        super::auto_embed_vector_index("text", config)
    };
    b = b
        .index("text", IndexType::FullText)
        .index("chunk_type", IndexType::Scalar(ScalarType::Hash))
        .index("language", IndexType::Scalar(ScalarType::Hash))
        .index("symbol_name", IndexType::Scalar(ScalarType::Hash))
        .index("speaker", IndexType::Scalar(ScalarType::Hash))
        .index("modality", IndexType::Scalar(ScalarType::Hash))
        .index("embedding", IndexType::Vector(dense_idx));
    if config.embedding.sparse_dimensions.is_some() {
        b = b.index(
            "sparse_embedding",
            super::auto_embed_sparse_index("text", config),
        );
    }
    if config.embedding.multivector_dimensions.is_some() {
        b = b.index(
            "colbert_embedding",
            IndexType::Vector(super::auto_embed_multivector_index("text", config)),
        );
    }
    b.done()
}

pub(crate) fn register_edges(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        // HAS_CHUNK: multi-source (Artifact, Message, Session, Block → Chunk).
        // Block is a source for tiered PDF extraction: each :Block owns a child
        // :Chunk (chunk_type = "block") that carries the embeddable text.
        .edge_type(
            edges::HAS_CHUNK,
            &[
                labels::ARTIFACT,
                labels::MESSAGE,
                labels::SESSION,
                labels::BLOCK,
            ],
            &[labels::CHUNK],
        )
        .property_nullable("index", DataType::Int64)
        .done()
}
