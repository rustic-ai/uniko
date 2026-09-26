//! Layer 3: Artifact node type (5 multimodal embedding fields).

use uni_db::{DataType, IndexType, ScalarType, SchemaBuilder};

use super::constants::{edges, labels};
use crate::config::UnikoConfig;

pub(crate) fn register_labels<'a>(
    builder: SchemaBuilder<'a>,
    config: &UnikoConfig,
) -> SchemaBuilder<'a> {
    builder
        .label(labels::ARTIFACT)
        // Typed provenance (issue #39). `category` is the caller's own
        // record class — a user assertion, an executed result, a model
        // interpretation — kept OUT of searchable prose so a recall scope
        // can filter on it. `source_id` is denormalised from the
        // FROM_SOURCE edge so that filter is a property predicate rather
        // than a traversal, since it runs inside every candidate query.
        .property_nullable("category", DataType::String)
        .property_nullable("source_id", DataType::String)
        // Which revision of that source this record came from (issue
        // #41). Denormalised beside `source_id` so the "is this evidence
        // still current" filter is a property predicate in candidate
        // generation rather than a traversal.
        .property_nullable("revision_id", DataType::String)
        .property("artifact_id", DataType::String)
        .property("kind", DataType::String)
        .property_nullable("path", DataType::String)
        .property_nullable("content", DataType::String)
        .property_nullable("mime_type", DataType::String)
        .property_nullable("hash", DataType::String)
        .property_nullable("size", DataType::Int64)
        .property_nullable("language", DataType::String)
        .property_nullable("created_at", DataType::DateTime)
        .property_nullable("updated_at", DataType::DateTime)
        // Stamped when a newer revision of the same source arrives. An
        // artifact with this set is history: it still answers a historical
        // query but must not ground a current one (issue #41).
        .property_nullable("superseded_at", DataType::DateTime)
        // Typed nullable modality metadata. Indexed columns where common
        // queries deserve them (`duration_ms` for "audio > 5 min",
        // `page_count` for "PDFs > N pages"); long-tail fields land in
        // `modality_meta`.
        .property_nullable("width", DataType::Int32)
        .property_nullable("height", DataType::Int32)
        .property_nullable("duration_ms", DataType::Int64)
        .property_nullable("sample_rate", DataType::Int32)
        // `channels` is Int32 (not Int16) — uni-db's `DataType` does not
        // expose Int16; the extra bytes are immaterial.
        .property_nullable("channels", DataType::Int32)
        .property_nullable("fps", DataType::Float32)
        .property_nullable("frame_count", DataType::Int32)
        .property_nullable("page_count", DataType::Int32)
        .property_nullable("modality_meta", DataType::CypherValue)
        // Provenance for non-Action ingest (URL / filesystem / upload /
        // import). When NULL, the canonical provenance path is the
        // `CREATED_BY` edge to `Action`.
        .property_nullable("origin", DataType::CypherValue)
        // 5 multimodal embedding fields — all nullable
        .property_nullable(
            "text_embedding",
            DataType::Vector {
                dimensions: config.embedding.dimensions,
            },
        )
        .property_nullable(
            "image_embedding",
            DataType::Vector {
                dimensions: config.embedding.dimensions,
            },
        )
        .property_nullable(
            "audio_embedding",
            DataType::Vector {
                dimensions: config.embedding.dimensions,
            },
        )
        .property_nullable(
            "video_embedding",
            DataType::Vector {
                dimensions: config.embedding.dimensions,
            },
        )
        .property_nullable(
            "multimodal_embedding",
            DataType::Vector {
                dimensions: config.embedding.dimensions,
            },
        )
        .index("artifact_id", IndexType::Scalar(ScalarType::Hash))
        .index("path", IndexType::Scalar(ScalarType::BTree))
        .index("kind", IndexType::Scalar(ScalarType::Hash))
        // F29: content-hash dedup. Hash index keeps the
        // "have I already ingested these exact bytes?" probe O(1)
        // instead of a table scan during ingest.
        .index("hash", IndexType::Scalar(ScalarType::Hash))
        .index("content", IndexType::FullText)
        .index("language", IndexType::Scalar(ScalarType::Hash))
        .index("mime_type", IndexType::Scalar(ScalarType::Hash))
        .index("duration_ms", IndexType::Scalar(ScalarType::BTree))
        .index("page_count", IndexType::Scalar(ScalarType::BTree))
        .index(
            "text_embedding",
            IndexType::Vector(super::vector_index(config)),
        )
        .index(
            "image_embedding",
            IndexType::Vector(super::vector_index(config)),
        )
        .index(
            "audio_embedding",
            IndexType::Vector(super::vector_index(config)),
        )
        .index(
            "video_embedding",
            IndexType::Vector(super::vector_index(config)),
        )
        .index(
            "multimodal_embedding",
            IndexType::Vector(super::vector_index(config)),
        )
        .done()
}

pub(crate) fn register_edges(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        .edge_type(edges::CREATED_BY, &[labels::ARTIFACT], &[labels::ACTION])
        .done()
        .edge_type(edges::MODIFIED_BY, &[labels::ARTIFACT], &[labels::ACTION])
        .property_nullable("diff_summary", DataType::String)
        .done()
        // ATTACHED_TO: an Artifact shared mid-conversation links to the
        // Session and/or Message that introduced it (F22/F30), so a file
        // dropped into a chat is reachable from its context instead of
        // orphaning. Used when no producing Action exists (Actions use
        // PRODUCED instead).
        .edge_type(
            edges::ATTACHED_TO,
            &[labels::ARTIFACT],
            &[labels::SESSION, labels::MESSAGE],
        )
        .property_nullable("attached_at", DataType::DateTime)
        .done()
}
