//! Layer 2: Action node type.

use uni_db::{DataType, IndexType, ScalarType, SchemaBuilder};

use super::constants::{edges, labels};
use crate::config::UnikoConfig;

pub(crate) fn register_labels<'a>(
    builder: SchemaBuilder<'a>,
    config: &UnikoConfig,
) -> SchemaBuilder<'a> {
    builder
        .label(labels::ACTION)
        .property("action_id", DataType::String)
        .property("action_type", DataType::String)
        .property_nullable("input", DataType::CypherValue)
        .property_nullable("output", DataType::CypherValue)
        .property_nullable("status", DataType::String)
        .property_nullable("started_at", DataType::DateTime)
        .property_nullable("ended_at", DataType::DateTime)
        .property_nullable("duration_ms", DataType::Int64)
        .property_nullable("error", DataType::String)
        // The tool that was called, for a call recorded as a tool call
        // (`action_type = "tool_call"`); a plain action leaves it unset.
        .property_nullable("tool_name", DataType::String)
        // SHA-256 of the result payload, so a replayed call with the same
        // id is recognised and one with different output is refused.
        .property_nullable("payload_hash", DataType::String)
        .property_nullable(
            "embedding",
            DataType::Vector {
                dimensions: config.embedding.dimensions,
            },
        )
        .index("action_id", IndexType::Scalar(ScalarType::Hash))
        .index("action_type", IndexType::Scalar(ScalarType::Hash))
        .index("tool_name", IndexType::Scalar(ScalarType::Hash))
        .index("status", IndexType::Scalar(ScalarType::Hash))
        .index("started_at", IndexType::Scalar(ScalarType::BTree))
        .index("embedding", IndexType::Vector(super::vector_index(config)))
        .done()
}

pub(crate) fn register_edges(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        .edge_type(
            edges::PERFORMED_BY,
            &[labels::ACTION],
            &[labels::PARTICIPANT],
        )
        .done()
        // TRIGGERED_BY: multi-source (Action, Episode → Message)
        .edge_type(
            edges::TRIGGERED_BY,
            &[labels::ACTION, labels::EPISODE],
            &[labels::MESSAGE],
        )
        .done()
        .edge_type(edges::PRODUCED, &[labels::ACTION], &[labels::ARTIFACT])
        .done()
        .edge_type(edges::NEXT_ACTION, &[labels::ACTION], &[labels::ACTION])
        .done()
}
