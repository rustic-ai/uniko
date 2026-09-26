//! Observation recording — the `add_observation` agent tool (F34).
//!
//! Observations are atomic claims tied to the message they were drawn
//! from.  The ingest pipeline extracts them automatically; this tool
//! lets an agent add one explicitly — a claim it wants on record against
//! a specific message (e.g. an inference it made while reading) that the
//! NLP extractor would not surface.
//!
//! It reuses the pipeline's [`apply_observations`] writer so the
//! resulting node and its `OBSERVED_IN` / `ABOUT` edges are identical to
//! pipeline-extracted ones.  Because `OBSERVED_IN` anchors every
//! Observation to a Message, this tool requires a `message_id`; an
//! agent fact with no message context belongs in
//! [`crate::assert_fact`] instead.

use chrono::{DateTime, Utc};

use uniko_extract::observations::{ObservationPrep, RawObservation, apply_observations};
use uniko_store::schema::labels;
use uniko_store::{KnowledgeBase, NodeId, UnikoError, Value};

/// Inputs for [`add_observation`].
///
/// `message_id` identifies the Message the observation is anchored to
/// (`OBSERVED_IN`).  `subject` drives `ABOUT` entity matching downstream.
#[derive(Debug, Clone, Default)]
pub struct AddObservationParams {
    /// `message_id` of the Message this observation was drawn from.
    pub message_id: String,
    /// Full natural-language claim (e.g. `"Caroline plays clarinet"`).
    pub content: String,
    /// Triple subject — who/what the claim is about.
    pub subject: String,
    /// Optional triple predicate.
    pub predicate: Option<String>,
    /// Optional triple object.
    pub object: Option<String>,
    /// Optional surface temporal phrase ("last Friday").
    pub temporal_phrase: Option<String>,
    /// Optional resolved absolute time for the temporal phrase.
    pub temporal_anchor: Option<DateTime<Utc>>,
    /// When the observation was made.  Defaults to `now()`.
    pub observed_at: Option<DateTime<Utc>>,
    /// Confidence in `[0.0, 1.0]`.  Defaults to 1.0 (agent-asserted).
    pub confidence: Option<f64>,
}

/// Record an explicit Observation anchored to a Message.
///
/// Resolves the agent's Participant (as the observation's sender) and
/// the target Message, then writes the Observation plus its
/// `OBSERVED_IN` → Message and `ABOUT` → sender edges through the shared
/// [`apply_observations`] path, inside a single transaction.
///
/// Returns the new Observation's NodeId.
///
/// # Errors
///
/// - [`UnikoError::Storage`] when the agent or Message is missing, or
///   the transaction fails to commit.
pub async fn add_observation(
    kb: &KnowledgeBase,
    agent_id: &str,
    params: AddObservationParams,
) -> Result<NodeId, UnikoError> {
    // Resolve the agent Participant — it becomes the observation's
    // sender so an `ABOUT` → Participant edge anchors it.
    let participant = kb
        .get_node_by_ext_id(labels::PARTICIPANT, "participant_id", agent_id)
        .await?
        .ok_or_else(|| {
            UnikoError::Storage(format!(
                "add_observation: Participant '{agent_id}' not found — create it before recording"
            ))
        })?;
    let sender_name = match participant.1.get("name") {
        Some(Value::String(s)) => s.clone(),
        _ => agent_id.to_string(),
    };
    let sender_ref = (participant.0, sender_name);

    // Resolve the anchoring Message.
    let message = kb
        .get_node_by_ext_id(labels::MESSAGE, "message_id", &params.message_id)
        .await?
        .ok_or_else(|| {
            UnikoError::Storage(format!(
                "add_observation: Message '{}' not found",
                params.message_id
            ))
        })?;
    let message_node = message.0;

    let raw = RawObservation {
        content: params.content,
        subject: params.subject,
        predicate: params.predicate,
        object: params.object,
        temporal_phrase: params.temporal_phrase,
        temporal_anchor: params.temporal_anchor,
        observed_at: params.observed_at.unwrap_or_else(Utc::now),
        confidence: params.confidence.unwrap_or(1.0),
    };

    // Build the same prep the pipeline hands to `apply_observations`.
    // No model was used and there are no pre-resolved entity refs; the
    // sender ref is enough to anchor the ABOUT edge.
    let prep = ObservationPrep {
        all_obs: vec![raw],
        used_model: false,
        sender_ref: Some(sender_ref),
        entity_refs: Vec::new(),
        sentence_ctx_updated: None,
        sender_ms: 0,
        extract_ms: 0,
        category: None,
        source_id: None,
        revision_id: None,
    };

    let tx = kb.begin_tx().await?;
    let obs_ids = match apply_observations(kb, &tx, message_node, prep).await {
        Ok(ids) => ids,
        Err(e) => {
            tx.rollback();
            return Err(e);
        }
    };
    tx.commit().await?;

    obs_ids.into_iter().next().ok_or_else(|| {
        UnikoError::Storage("add_observation: apply_observations produced no node".into())
    })
}
