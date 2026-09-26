//! Layer 1: Source node type — the stable logical origin of a record.
//!
//! A `:Source` is the thing a record *came from*: a web page, a dataset, an
//! uploaded file, an upstream system. It is deliberately separate from the
//! `:Artifact` that holds one fetch of it, because the logical source
//! outlives any single revision of its bytes (`rustic-ai/uniko#41`), and
//! because recall must be able to say "only evidence from these sources"
//! without that identity living in searchable prose (`rustic-ai/uniko#39`).
//!
//! Records point at it with `FROM_SOURCE`. Message, Artifact, Observation
//! and Fact also carry a denormalised `source_id`, so the recall
//! candidate-generation filter is a property predicate rather than a
//! multi-hop traversal — the same denormalisation `Artifact.hash` already
//! uses as a cache of `HAS_CONTENT.target.content_id`.

use uni_db::{DataType, IndexType, ScalarType, SchemaBuilder};

use super::constants::{edges, labels};
use crate::config::UnikoConfig;

pub(crate) fn register_labels<'a>(
    builder: SchemaBuilder<'a>,
    _config: &UnikoConfig,
) -> SchemaBuilder<'a> {
    builder
        .label(labels::SOURCE)
        // Caller-supplied stable identity. This is what a recall scope
        // filters on and what survives across revisions of the content.
        .property("source_id", DataType::String)
        // Human-readable label, for attribution in an answer.
        .property_nullable("name", DataType::String)
        // Where it came from, when there is a locator.
        .property_nullable("uri", DataType::String)
        .property("first_seen", DataType::DateTime)
        .property_nullable("last_seen", DataType::DateTime)
        // Retiring a source stops EVERY revision of it grounding a current
        // answer, without deleting anything — history stays attributable.
        // Retirement lives here, not on the content, so retiring one source
        // never affects another that merely shares identical bytes.
        .property_nullable("retired_at", DataType::DateTime)
        .index("source_id", IndexType::Scalar(ScalarType::Hash))
        .index("first_seen", IndexType::Scalar(ScalarType::BTree))
        .index("retired_at", IndexType::Scalar(ScalarType::BTree))
        .done()
}

pub(crate) fn register_edges(builder: SchemaBuilder<'_>) -> SchemaBuilder<'_> {
    builder
        // Message / Artifact → Source.
        .edge_type(
            edges::FROM_SOURCE,
            &[labels::MESSAGE, labels::ARTIFACT],
            &[labels::SOURCE],
        )
        .done()
}
