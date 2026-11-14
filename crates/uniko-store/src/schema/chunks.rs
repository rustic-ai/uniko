//! Layer 3: Chunk node type.

use uni_db::{DataType, IndexType, ScalarType, SchemaBuilder};

use super::constants::{edges, labels};
use crate::config::UnikoConfig;

/// What a chunk's embedding is computed from, in order.
const EMBED_SOURCES: &[&str] = &["context", "text"];

/// Analyzer for `Chunk.search_text`: whitespace tokens, lower-cased and
/// ASCII-folded, nothing stemmed or dropped, long tokens kept.
fn search_text_analyzer() -> uni_db::common::core::schema::AnalyzerConfig {
    use uni_db::common::core::schema::{AnalyzerConfig, BaseTokenizer};
    AnalyzerConfig {
        base: BaseTokenizer::Whitespace,
        stem: false,
        remove_stop_words: false,
        max_token_length: Some(128),
        ..AnalyzerConfig::default()
    }
}

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
        // Structure-aware chunks (tool output): the header of values the
        // slice inherits from outside itself, and where the slice sits
        // (`/data/items[3-9]`, `rows 10-19`). Embedded with `text`.
        .property_nullable("context", DataType::String)
        .property_nullable("address", DataType::String)
        // `context` + `text` normalised for exact lexical matching of
        // identifiers and figures; see `search_text_analyzer`.
        .property_nullable("search_text", DataType::String)
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
        // Inherited from the owning Message (mirrors `:Message.visibility`).
        .property_nullable("visibility", DataType::String)
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
    // A chunk is embedded from its context header and its text: a tool
    // output's record carries what it inherits (tool, address, enclosing
    // values) into its vector, and a chunk with no header embeds its text
    // alone, as before.
    //
    // When hybrid, the dense `embedding` joins the sparse + ColBERT columns
    // on the EmbedHybrid alias (one pass); otherwise it stays on the plain
    // dense alias.
    let dense_idx = if config.embedding.sparse_dimensions.is_some()
        || config.embedding.multivector_dimensions.is_some()
    {
        super::auto_embed_hybrid_vector_index(EMBED_SOURCES, config)
    } else {
        super::auto_embed_vector_index(EMBED_SOURCES, config)
    };
    b = b
        .index("text", IndexType::FullText)
        // Normalised search text (`uniko_extract::ingest::structure::
        // normalize`): already split into whole tokens, so the analyzer
        // only lower-cases and folds — no stemming or stop words, which
        // would mangle identifiers and figures.
        .index(
            "search_text",
            IndexType::FullTextWithAnalyzer(search_text_analyzer()),
        )
        .index("chunk_type", IndexType::Scalar(ScalarType::Hash))
        .index("language", IndexType::Scalar(ScalarType::Hash))
        .index("symbol_name", IndexType::Scalar(ScalarType::Hash))
        .index("speaker", IndexType::Scalar(ScalarType::Hash))
        .index("modality", IndexType::Scalar(ScalarType::Hash))
        .index("embedding", IndexType::Vector(dense_idx));
    if config.embedding.sparse_dimensions.is_some() {
        b = b.index(
            "sparse_embedding",
            super::auto_embed_sparse_index(EMBED_SOURCES, config),
        );
    }
    if config.embedding.multivector_dimensions.is_some() {
        b = b.index(
            "colbert_embedding",
            IndexType::Vector(super::auto_embed_multivector_index(EMBED_SOURCES, config)),
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
