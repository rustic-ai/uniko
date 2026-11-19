//! Layer 6: ConsolidationCycle and DeadLetter node types.

use uni_db::{DataType, IndexType, ScalarType, SchemaBuilder};

use super::constants::{edges, labels};

pub(crate) fn register_labels(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        // ── ConsolidationCycle ──
        .label(labels::CONSOLIDATION_CYCLE)
        .property("cycle_id", DataType::String)
        .property("agent_id", DataType::String)
        .property_nullable("started_at", DataType::DateTime)
        .property_nullable("completed_at", DataType::DateTime)
        .property_nullable("observations_processed", DataType::Int64)
        .property_nullable("episodes_involved", DataType::Int64)
        .property_nullable("facts_created", DataType::Int64)
        .property_nullable("facts_reinforced", DataType::Int64)
        .property_nullable("facts_invalidated", DataType::Int64)
        .property_nullable("procedures_promoted", DataType::Int64)
        .property_nullable("drift_alerts", DataType::Int64)
        .index("cycle_id", IndexType::Scalar(ScalarType::Hash))
        .index("agent_id", IndexType::Scalar(ScalarType::Hash))
        .index("started_at", IndexType::Scalar(ScalarType::BTree))
        // ── DeadLetter ──
        .label(labels::DEAD_LETTER)
        .property_nullable("step", DataType::String)
        .property_nullable("error", DataType::String)
        .property_nullable("node_ref", DataType::Int64)
        .property_nullable("retry_count", DataType::Int64)
        .property_nullable("max_retries", DataType::Int64)
        .property_nullable("next_retry_at", DataType::DateTime)
        .property_nullable("created_at", DataType::DateTime)
        // Added for retry and recovery (#65); NULL on rows written before.
        .property_nullable("dead_letter_id", DataType::String)
        .property_nullable("ingest_type", DataType::String)
        .property_nullable("payload", DataType::String)
        .property_nullable("status", DataType::String)
        .property_nullable("error_kind", DataType::String)
        .property_nullable("retriable", DataType::Bool)
        .property_nullable("agent_id", DataType::String)
        .property_nullable("updated_at", DataType::DateTime)
        .property_nullable("resolved_at", DataType::DateTime)
        .property_nullable("resolution", DataType::String)
        .index("dead_letter_id", IndexType::Scalar(ScalarType::Hash))
        .index("status", IndexType::Scalar(ScalarType::Hash))
        .done()
}

pub(crate) fn register_edges(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        .edge_type(
            edges::PROCESSED,
            &[labels::CONSOLIDATION_CYCLE],
            &[labels::OBSERVATION],
        )
        .done()
        .edge_type(
            edges::INVOLVED,
            &[labels::CONSOLIDATION_CYCLE],
            &[labels::EPISODE],
        )
        .done()
        .edge_type(
            edges::CREATED,
            &[labels::CONSOLIDATION_CYCLE],
            &[labels::FACT],
        )
        .done()
        // REINFORCED: facts this cycle re-observed (vs newly CREATED).
        // Distinct edge so provenance does not overload INVOLVED, which
        // targets Episode.
        .edge_type(
            edges::REINFORCED,
            &[labels::CONSOLIDATION_CYCLE],
            &[labels::FACT],
        )
        .done()
        .edge_type(
            edges::INVALIDATED,
            &[labels::CONSOLIDATION_CYCLE],
            &[labels::FACT],
        )
        .done()
        .edge_type(
            edges::PROMOTED,
            &[labels::CONSOLIDATION_CYCLE],
            &[labels::PROCEDURE],
        )
        .done()
        .edge_type(
            edges::APPLIED_RULE,
            &[labels::CONSOLIDATION_CYCLE],
            &[labels::RULE],
        )
        .done()
}
