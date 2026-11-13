//! Fact assertion/invalidation — the `assert_fact` and `invalidate_fact`
//! agent tools (F33/F37).
//!
//! Most Facts are *derived* by P4 Consolidation from accumulated
//! Observations.  These two tools let an agent assert or retract a Fact
//! directly — for knowledge it holds that was never stated in a message
//! (e.g. a preference it inferred, or a correction it must apply now
//! rather than waiting for consolidation).
//!
//! Both reuse the bitemporal Fact machinery in `uniko-store`:
//! [`KnowledgeBase::upsert_fact_by_triple`] for assertion (idempotent on
//! the `(subject, predicate, object)` triple) and
//! [`KnowledgeBase::invalidate_fact`] for retraction (closes the Fact's
//! BTIC validity interval).

use std::collections::HashMap;

use chrono::{DateTime, Utc};

use uniko_extract::embedding::embed_document;
use uniko_store::operations::facts::FactUpsert;
use uniko_store::schema::{edges, labels};
use uniko_store::{KnowledgeBase, UnikoError};

/// Inputs for [`assert_fact`].
///
/// The triple `(subject, predicate, object)` and the visibility scope are
/// the Fact's identity; asserting the same triple again in the same scope
/// reinforces the existing Fact rather than creating a duplicate.
#[derive(Debug, Clone, Default)]
pub struct AssertFactParams {
    /// Triple subject (e.g. `"user"`).
    pub subject: String,
    /// Triple predicate (e.g. `"prefers"`).
    pub predicate: String,
    /// Triple object (e.g. `"dark mode"`).  `None` for unary facts.
    pub object: Option<String>,
    /// Number of observations this assertion represents.  Defaults to 1.
    pub observation_count: Option<i64>,
    /// When the fact was observed.  Defaults to `now()`.
    pub observed_at: Option<DateTime<Utc>>,
    /// `observation_id`s of Observations that support this Fact.  Each
    /// resolvable one gets a `SUPPORTED_BY` edge (best-effort).
    pub supporting_observation_ids: Vec<String>,
    /// Who may see the Fact. `None` (or [`Visibility::Public`]) asserts the
    /// public Fact of this triple; a scope asserts a separate Fact held
    /// under that scope, as consolidation does for scoped observations.
    ///
    /// [`Visibility::Public`]: crate::Visibility::Public
    pub visibility: Option<crate::Visibility>,
}

/// Assert a Fact from a triple, embedding it and wiring provenance.
///
/// Upserts the `(subject, predicate, object)` Fact (idempotent), embeds
/// the triple text, and creates `SUPPORTED_BY` → Observation edges for
/// any supplied `supporting_observation_ids` that resolve (missing ones
/// are skipped with a tracing debug).
///
/// Returns the [`FactUpsert`] so the caller can tell whether a new Fact
/// was created and read the resulting observation count.
///
/// # Errors
///
/// - [`UnikoError::Embedding`] when embedding the triple fails.
/// - [`UnikoError::Storage`] when the upsert or edge writes fail.
pub async fn assert_fact(
    kb: &KnowledgeBase,
    params: AssertFactParams,
) -> Result<FactUpsert, UnikoError> {
    let observed_at = params.observed_at.unwrap_or_else(Utc::now);
    let count = params.observation_count.unwrap_or(1);

    // Embed the triple in claim form so it co-locates with the
    // observations and questions it answers.
    let embed_text = match params.object.as_deref() {
        Some(obj) => format!("{} {} {obj}", params.subject, params.predicate),
        None => format!("{} {}", params.subject, params.predicate),
    };
    let embedding = embed_document(kb, &embed_text).await?;

    let visibility = params
        .visibility
        .as_ref()
        .map(crate::Visibility::as_property);
    let upsert = kb
        .upsert_scoped_fact_by_triple(
            &params.subject,
            &params.predicate,
            params.object.as_deref(),
            count,
            observed_at,
            Some(embedding),
            visibility.as_deref(),
        )
        .await?;

    // ── Optional SUPPORTED_BY (Fact → Observation) ──
    for obs_id in &params.supporting_observation_ids {
        match kb
            .get_node_by_ext_id(labels::OBSERVATION, "observation_id", obs_id)
            .await?
        {
            Some(obs) => {
                kb.create_edge(edges::SUPPORTED_BY, upsert.node_id, obs.0, &HashMap::new())
                    .await?;
            }
            None => {
                tracing::debug!(
                    observation_id = obs_id,
                    "assert_fact: supporting Observation not found — SUPPORTED_BY skipped"
                );
            }
        }
    }

    Ok(upsert)
}

/// Inputs for [`invalidate_fact`].
///
/// Identifies the stale Fact by its `fact_id` and, optionally, the
/// replacement Fact that supersedes it (wired via `INVALIDATES`).
#[derive(Debug, Clone, Default)]
pub struct InvalidateFactParams {
    /// `fact_id` of the Fact to retract.
    pub fact_id: String,
    /// `fact_id` of the Fact that replaces it, if any.  When set and
    /// resolvable, an `INVALIDATES` edge is created from the new Fact to
    /// the stale one.
    pub replacement_fact_id: Option<String>,
    /// Human-readable reason, stored on the `INVALIDATES` edge.
    pub reason: Option<String>,
    /// Invalidation time.  Defaults to `now()`.
    pub now: Option<DateTime<Utc>>,
}

/// Retract a Fact by closing its BTIC validity interval (F37).
///
/// Loads the stale Fact's current interval, closes it at `now`, and —
/// when `replacement_fact_id` resolves — records the supersession via an
/// `INVALIDATES` edge stamped with the time (so F39 drift detection can
/// window it).
///
/// # Errors
///
/// - [`UnikoError::Storage`] when the Fact is missing, has no decodable
///   `valid_at`, or the graph writes fail.
pub async fn invalidate_fact(
    kb: &KnowledgeBase,
    params: InvalidateFactParams,
) -> Result<(), UnikoError> {
    let now = params.now.unwrap_or_else(Utc::now);

    let stale = kb
        .get_node_by_ext_id(labels::FACT, "fact_id", &params.fact_id)
        .await?
        .ok_or_else(|| {
            UnikoError::Storage(format!(
                "invalidate_fact: Fact '{}' not found",
                params.fact_id
            ))
        })?;
    let stale_node = stale.0;

    let interval = kb.fact_valid_at(stale_node).await?.ok_or_else(|| {
        UnikoError::Storage(format!(
            "invalidate_fact: Fact '{}' has no valid_at interval to close",
            params.fact_id
        ))
    })?;

    // Resolve the replacement up-front; a supplied-but-missing
    // replacement is a caller error worth surfacing, not silently
    // dropping (unlike the best-effort enrichment edges elsewhere).
    let replacement = match params.replacement_fact_id.as_deref() {
        Some(rid) => {
            let node = kb
                .get_node_by_ext_id(labels::FACT, "fact_id", rid)
                .await?
                .ok_or_else(|| {
                    UnikoError::Storage(format!(
                        "invalidate_fact: replacement Fact '{rid}' not found"
                    ))
                })?;
            Some(node.0)
        }
        None => None,
    };

    kb.invalidate_fact(
        stale_node,
        &interval,
        now,
        replacement,
        params.reason.as_deref(),
    )
    .await
}
