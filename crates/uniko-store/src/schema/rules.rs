//! Layer 5: Rule node type.

use uni_db::{DataType, IndexType, ScalarType, SchemaBuilder};

use super::constants::{edges, labels};

pub(crate) fn register_labels(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        .label(labels::RULE)
        .property("rule_id", DataType::String)
        .property("name", DataType::String)
        .property_nullable("source", DataType::String)
        .property_nullable("natural_language", DataType::String)
        .property_nullable("source_type", DataType::String)
        .property_nullable("status", DataType::String)
        .property_nullable("version", DataType::Int64)
        .property_nullable("confidence", DataType::Float64)
        .property_nullable("precision", DataType::Float64)
        .property_nullable("recall", DataType::Float64)
        .property_nullable("coverage", DataType::Int64)
        .property_nullable("created_at", DataType::DateTime)
        .property_nullable("validated_at", DataType::DateTime)
        .property_nullable("last_scored_at", DataType::DateTime)
        .index("rule_id", IndexType::Scalar(ScalarType::Hash))
        .index("name", IndexType::Scalar(ScalarType::Hash))
        .index("status", IndexType::Scalar(ScalarType::Hash))
        .index("source_type", IndexType::Scalar(ScalarType::Hash))
        .done()
}

pub(crate) fn register_edges(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        // SUPERSEDES is shared: a Rule replaced by a successor rule, and an
        // Artifact revision replaced by a newer revision of the same Source
        // (issue #41). uni-db permits one declaration per edge type, so both
        // endpoint pairs are declared together here rather than separately —
        // declaring it twice is rejected as a conflicting re-declaration.
        .edge_type(
            edges::SUPERSEDES,
            &[labels::RULE, labels::ARTIFACT],
            &[labels::RULE, labels::ARTIFACT],
        )
        .done()
        .edge_type(edges::COVERS, &[labels::RULE], &[labels::EPISODE])
        .property_nullable("correct", DataType::Int64)
        .done()
}
