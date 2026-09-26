//! Layer 4: Fact node type (with BTIC temporal validity).

use uni_db::{DataType, IndexType, ScalarType, SchemaBuilder};

use super::constants::{edges, labels};
use crate::config::UnikoConfig;

pub(crate) fn register_labels<'a>(
    builder: SchemaBuilder<'a>,
    config: &UnikoConfig,
) -> SchemaBuilder<'a> {
    builder
        .label(labels::FACT)
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
        .property("fact_id", DataType::String)
        .property("subject", DataType::String)
        .property("predicate", DataType::String)
        .property_nullable("object", DataType::String)
        .property_nullable("confidence", DataType::Float64)
        .property_nullable("observation_count", DataType::Int64)
        .property_nullable("valid_at", DataType::Btic)
        .property_nullable("source_rule", DataType::String)
        .property_nullable("visibility", DataType::String)
        // Node-level invalidation audit. Set when `valid_at` is closed —
        // by contradiction (F38), evidence removal, or subject erasure.
        // `null` = never invalidated. Unlike the `INVALIDATES` edge (which
        // exists only with a successor fact), these record a standalone
        // close. NULL default ⇒ no backfill migration.
        .property_nullable("invalidation_reason", DataType::String)
        .property_nullable("invalidated_at", DataType::DateTime)
        .property_nullable(
            "embedding",
            DataType::Vector {
                dimensions: config.embedding.dimensions,
            },
        )
        .index("fact_id", IndexType::Scalar(ScalarType::Hash))
        .index("subject", IndexType::Scalar(ScalarType::Hash))
        .index("subject", IndexType::FullText)
        .index("predicate", IndexType::Scalar(ScalarType::Hash))
        .index("confidence", IndexType::Scalar(ScalarType::BTree))
        .index("embedding", IndexType::Vector(super::vector_index(config)))
        .done()
}

pub(crate) fn register_edges(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        .edge_type(edges::SUPPORTED_BY, &[labels::FACT], &[labels::OBSERVATION])
        .property_nullable("weight", DataType::Float64)
        .done()
        // DERIVED_BY: Fact → Rule
        .edge_type(edges::DERIVED_BY, &[labels::FACT], &[labels::RULE])
        .done()
        // DERIVED_FROM: multi-source (Fact, Procedure → Episode, Action;
        // Artifact → Artifact for non-Action derivation chains like
        // pdf_page_render, video_keyframe, audio_segment, transcoded).
        .edge_type(
            edges::DERIVED_FROM,
            &[labels::FACT, labels::PROCEDURE, labels::ARTIFACT],
            &[labels::EPISODE, labels::ACTION, labels::ARTIFACT],
        )
        .property_nullable("derivation_kind", DataType::String)
        .property_nullable("derived_at", DataType::DateTime)
        .done()
        .edge_type(edges::INVALIDATES, &[labels::FACT], &[labels::FACT])
        .property_nullable("reason", DataType::String)
        // Timestamp of the invalidation, so F39 drift detection can count
        // invalidations within a rolling window (e.g. last 30 days)
        // instead of cumulatively.
        .property_nullable("invalidated_at", DataType::DateTime)
        .done()
        .edge_type(edges::SHARED_FROM, &[labels::FACT], &[labels::FACT])
        .property_nullable("shared_by", DataType::String)
        .property_nullable("shared_at", DataType::DateTime)
        .done()
        // CONTRADICTED_BY: Fact → Episode — provenance for a contradiction
        // surfaced by the `contradiction_detector` rule (the fact is also
        // BTIC-invalidated by the consumer).
        .edge_type(edges::CONTRADICTED_BY, &[labels::FACT], &[labels::EPISODE])
        .property_nullable("detected_at", DataType::DateTime)
        .done()
}
