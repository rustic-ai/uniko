//! Layer 2: Message node type.

use uni_db::{DataType, IndexType, ScalarType, SchemaBuilder};

use super::constants::{edges, labels};
use crate::config::UnikoConfig;

pub(crate) fn register_labels<'a>(
    builder: SchemaBuilder<'a>,
    config: &UnikoConfig,
) -> SchemaBuilder<'a> {
    builder
        .label(labels::MESSAGE)
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
        .property("message_id", DataType::String)
        .property("content", DataType::String)
        .property_nullable("content_type", DataType::String)
        .property("timestamp", DataType::DateTime)
        // Soft-forget tombstone. `null`/absent = visible; `true` = the
        // turn was forgotten and the recall post-filter drops it. NULL
        // default means no backfill migration is needed.
        .property_nullable("redacted", DataType::Bool)
        .property_nullable(
            "embedding",
            DataType::Vector {
                dimensions: config.embedding.dimensions,
            },
        )
        .index("message_id", IndexType::Scalar(ScalarType::Hash))
        .index("timestamp", IndexType::Scalar(ScalarType::BTree))
        .index("content", IndexType::FullText)
        .index(
            "embedding",
            IndexType::Vector(super::auto_embed_vector_index("content", config)),
        )
        .done()
}

pub(crate) fn register_edges(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        .edge_type(edges::SENT_BY, &[labels::MESSAGE], &[labels::PARTICIPANT])
        .property_nullable("role", DataType::String)
        .done()
        .edge_type(
            edges::ADDRESSED_TO,
            &[labels::MESSAGE],
            &[labels::PARTICIPANT],
        )
        .done()
        // IN_SESSION: multi-source (Message, Action, Episode → Session)
        .edge_type(
            edges::IN_SESSION,
            &[labels::MESSAGE, labels::ACTION, labels::EPISODE],
            &[labels::SESSION],
        )
        .done()
        .edge_type(edges::NEXT, &[labels::MESSAGE], &[labels::MESSAGE])
        .property_nullable("gap_ms", DataType::Int64)
        .done()
}
