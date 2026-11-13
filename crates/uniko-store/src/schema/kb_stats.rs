//! Layer 8: `:KnowledgeBaseStats` — singleton KB-level metadata node.
//
use uni_db::{DataType, IndexType, ScalarType, SchemaBuilder};

use super::constants::labels;

/// Register the `:KnowledgeBaseStats` singleton label.
///
/// Carries the modality-presence cache (which modalities have any
/// content in this KB), and the persisted `blob_storage` config and
/// embedding identity so mismatched reopens can hard-error. Treated as a singleton by
/// convention — there is at most one row per KB, keyed by a fixed
/// `stats_id` of `"singleton"`.
pub(crate) fn register_labels(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        .label(labels::KNOWLEDGE_BASE_STATS)
        .property("stats_id", DataType::String)
        .property_nullable("modality_presence", DataType::CypherValue)
        .property_nullable("blob_storage", DataType::CypherValue)
        .property_nullable("embedding", DataType::CypherValue)
        .property("updated_at", DataType::DateTime)
        .index("stats_id", IndexType::Scalar(ScalarType::Hash))
        .done()
}
