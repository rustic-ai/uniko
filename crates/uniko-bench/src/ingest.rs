//! Conversation ingestion into a fresh KnowledgeBase.
//!
//! For each conversation, creates an isolated in-memory KB, ingests all
//! turns with entity and observation extraction, then returns the KB
//! ready for querying.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};

use uni_db::ModelAliasSpec;
use uniko_extract::ingest::artifact::ingest_artifact;
use uniko_extract::ingest::atomic::ingest_message_atomic;
use uniko_pipes::types::{IngestArtifact, IngestMessage};
use uniko_store::KnowledgeBase;
use uniko_store::config::UnikoConfig;

use crate::data::{Conversation, LocomoSample, ParsedSession};
use crate::events::{BenchEvent, EventWriter, now_unix_ms};
use crate::pricing::Pricing;

/// Optional per-turn cost / event recording wired into the ingest loop.
///
/// All fields are independently optional — pass `None` for a field
/// to skip the corresponding accounting.  The bench binary populates
/// every field; library consumers that don't need cost telemetry
/// leave them all unset and get identical behaviour to the
/// pre-events code path.
#[derive(Default)]
pub struct IngestObserver<'a> {
    /// Sink for `BenchEvent::IngestTurn` rows.
    pub events: Option<&'a EventWriter>,
    /// Pricing table for embedding cost.
    pub pricing: Option<&'a Pricing>,
    /// Embedding model id (matches a row in `pricing.csv`).
    pub embedding_model: Option<String>,
}

/// Ingest a full LoCoMo conversation into a persistent KB.
///
/// Opens the KB at `ingest_dir`, then delegates to
/// [`ingest_into_kb`].  Use [`ingest_into_kb`] directly when the
/// caller already manages KB lifecycle (e.g. an HTTP service that
/// opens the KB once at warmup).
///
/// # Errors
///
/// Returns an error if the KB cannot be opened or any turn fails to
/// ingest.
pub async fn ingest_conversation(
    sample: &LocomoSample,
    sessions: &[ParsedSession],
    ingest_dir: &Path,
    config: UnikoConfig,
    extra_catalog: &[ModelAliasSpec],
) -> Result<Arc<KnowledgeBase>> {
    ingest_conversation_with_observer(
        sample,
        sessions,
        ingest_dir,
        config,
        extra_catalog,
        &IngestObserver::default(),
    )
    .await
}

/// Like [`ingest_conversation`] but threads an [`IngestObserver`]
/// through the per-turn loop so the caller can record bench events
/// and cost.
///
/// # Errors
///
/// Returns an error if the KB cannot be opened or any turn fails to
/// ingest.
pub async fn ingest_conversation_with_observer(
    sample: &LocomoSample,
    sessions: &[ParsedSession],
    ingest_dir: &Path,
    config: UnikoConfig,
    extra_catalog: &[ModelAliasSpec],
    observer: &IngestObserver<'_>,
) -> Result<Arc<KnowledgeBase>> {
    let kb = KnowledgeBase::open_with_xervo(ingest_dir, config, extra_catalog.to_vec())
        .await
        .context("creating KB")?;
    let kb = Arc::new(kb);
    ingest_into_kb_with_observer(&kb, sample, sessions, observer).await?;
    Ok(kb)
}

/// Ingest every session of a LoCoMo sample into an already-open KB.
///
/// The KB-lifecycle-free entry point: callers (the bench's
/// `ingest_conversation` wrapper, or a future HTTP `/memorize`
/// handler) can share the per-turn atomic ingest (Message + entities +
/// observations under one transaction) + session chunking without
/// re-opening the KB.
///
/// # Errors
///
/// Returns an error if any turn fails to ingest.
pub async fn ingest_into_kb(
    kb: &Arc<KnowledgeBase>,
    sample: &LocomoSample,
    sessions: &[ParsedSession],
) -> Result<()> {
    ingest_into_kb_with_observer(kb, sample, sessions, &IngestObserver::default()).await
}

/// Ingest every session of a LoCoMo sample with optional event /
/// cost recording.
///
/// Identical to [`ingest_into_kb`] when `observer` is the default
/// (every field `None`).  When `observer.events` is set, one
/// [`BenchEvent::IngestTurn`] is appended per turn.
///
/// # Errors
///
/// Returns an error if any turn fails to ingest.
pub async fn ingest_into_kb_with_observer(
    kb: &Arc<KnowledgeBase>,
    sample: &LocomoSample,
    sessions: &[ParsedSession],
    observer: &IngestObserver<'_>,
) -> Result<()> {
    let mut total_turns = 0u32;
    let mut total_entities = 0usize;
    let mut total_observations = 0usize;
    let mut total_session_chunks = 0usize;

    for session in sessions {
        let base_ts = parse_session_datetime(&session.date_time);

        // Create session-level context for pronoun resolution.
        // session_nid is resolved during the first ingest call.
        let mut session_ctx = uniko_extract::ingest::context::SessionContext::new(
            session.session_id.clone(),
            0, // will be set after first ingest
        );
        // Register both participants.
        session_ctx.register_participant(
            &sample.conversation.speaker_a,
            0, // nid resolved later
        );
        session_ctx.register_participant(&sample.conversation.speaker_b, 0);

        for (turn_idx, turn) in session.turns.iter().enumerate() {
            let timestamp = base_ts + Duration::seconds(turn_idx as i64 * 30);

            let other_speaker = other_speaker_name(&turn.speaker, &sample.conversation);

            // Update session context with current speaker.
            session_ctx.set_current_speaker(&turn.speaker);

            // Enrich message content with image caption/query if present.
            let mut content = turn.text.clone();
            if let Some(ref caption) = turn.blip_caption
                && !caption.is_empty()
            {
                if !content.is_empty() {
                    content.push(' ');
                }
                content.push_str(caption);
            }
            if let Some(ref query) = turn.query
                && !query.is_empty()
            {
                if !content.is_empty() {
                    content.push(' ');
                }
                content.push_str(query);
            }

            let msg = IngestMessage {
                message_id: format!("{}-{}", sample.sample_id, turn.dia_id),
                content: content.clone(),
                content_type: "text".to_string(),
                sender_id: turn.speaker.clone(),
                session_id: session.session_id.clone(),
                addressed_to: Some(vec![other_speaker]),
                timestamp,
                metadata: HashMap::new(),
                category: None,
                source_id: None,
                revision_id: None,
            };

            // Atomic ingest: Message + entities + observations in one tx.
            // NOTE: extraction now runs on the same enriched `content` as
            // the Message node (captions + query appended). Pre-migration
            // this path ran extraction on `turn.text` only — the
            // "captions confuse ONNX NER" workaround is no longer reachable
            // with a single-content atomic API.
            let result = ingest_message_atomic(kb, &msg, &mut session_ctx)
                .await
                .with_context(|| format!("ingesting {}", turn.dia_id))?;

            total_entities += result.extracted_entities.len();
            total_observations += result.extracted_observations.len();

            // Create Artifact node for image turns: caption+query as the
            // searchable text proxy. (uni-db #50 / DataType::Bytes landed in
            // 2.0 and ArtifactContent persists raw bytes; populating image
            // bytes here would require fetching img_url — out of scope for the
            // bench, which only has URLs, not the image files.)
            if turn.img_url.is_some() {
                let caption = turn.blip_caption.as_deref().unwrap_or("");
                let query_text = turn.query.as_deref().unwrap_or("");
                let artifact_content = format!("{caption} {query_text}").trim().to_string();

                if !artifact_content.is_empty() {
                    let img_path = turn.img_url.as_ref().and_then(|urls| urls.first().cloned());

                    let artifact = IngestArtifact {
                        artifact_id: format!("{}-{}-img", sample.sample_id, turn.dia_id),
                        // The id above is deliberate and per-turn, so it is
                        // the identity: two turns whose caption+query text
                        // happens to match must stay two artifacts.
                        caller_supplied_id: true,
                        content: artifact_content,
                        kind: "image".to_string(),
                        path: img_path,
                        metadata: HashMap::new(),
                        ..Default::default()
                    };

                    if let Err(e) = ingest_artifact(kb, &artifact).await {
                        tracing::warn!(
                            dia_id = %turn.dia_id,
                            error = %e,
                            "failed to ingest image artifact",
                        );
                    }
                }
            }

            // Derived sub-timings from the atomic timings — preserve the
            // legacy 3-bucket telemetry shape (ingest / ner / obs) so
            // downstream dashboards keep working.
            let timings = result.timings;
            let ingest_ms = (timings.setup_ms
                + timings.create_ms
                + timings.edges_ms
                + timings.chunk_ms
                + timings.commit_ms) as u128;
            let ner_ms = (timings.nlp_ms
                + timings.extract_ms
                + timings.prep_read_ms
                + timings.apply_entity_ms) as u128;
            let obs_ms = timings.apply_obs_ms;

            total_turns += 1;

            // Progress every 20 turns or on first turn.
            if total_turns == 1 || total_turns.is_multiple_of(20) {
                tracing::info!(
                    turn = total_turns,
                    dia_id = %turn.dia_id,
                    ingest_ms,
                    ner_ms,
                    obs_ms,
                    entities = result.extracted_entities.len(),
                    observations = result.extracted_observations.len(),
                    "turn processed",
                );
            }

            if let Some(writer) = observer.events {
                let wall_ms = timings.total_ms as u64;
                // Char-count proxy for embedding tokens (chars/4 ≈
                // tokens).  Real provider usage is unavailable because
                // uni-db's xervo.embed() does not surface the
                // openai/anthropic usage struct.  Local embedders
                // produce 0 cost regardless of chars.
                let embedding_chars = content.chars().count() as u64;
                let cost_usd = observer
                    .pricing
                    .zip(observer.embedding_model.as_deref())
                    .and_then(|(p, model)| p.cost_embedding(model, embedding_chars / 4))
                    .unwrap_or(0.0);
                let event = BenchEvent::IngestTurn {
                    sample_id: sample.sample_id.clone(),
                    turn_index: total_turns as usize - 1,
                    ts_unix_ms: now_unix_ms(),
                    wall_ms,
                    ingest_ms: ingest_ms as u64,
                    ner_ms: ner_ms as u64,
                    obs_ms: obs_ms as u64,
                    extraction_input_tokens: None,
                    extraction_output_tokens: None,
                    extraction_model: None,
                    embedding_chars,
                    embedding_model: observer.embedding_model.clone(),
                    cost_usd,
                };
                if let Err(e) = writer.write(&event) {
                    tracing::warn!(error = %e, "failed to append IngestTurn event");
                }
            }
        }

        // Chunk the session for retrieval (concatenates turns with speaker prefixes).
        let chunk_ids =
            uniko_extract::ingest::session_chunk::chunk_session(kb, &session.session_id)
                .await
                .with_context(|| format!("chunking session {}", session.session_id))?;
        total_session_chunks += chunk_ids.len();

        // Aggregate per-message observations into searchable session-level chunks.
        let obs_chunk_ids = uniko_extract::ingest::session_chunk::chunk_session_observations(
            kb,
            &session.session_id,
        )
        .await
        .with_context(|| format!("chunking session observations {}", session.session_id))?;
        total_session_chunks += obs_chunk_ids.len();
    }

    tracing::info!(
        sample_id = %sample.sample_id,
        turns = total_turns,
        entities = total_entities,
        observations = total_observations,
        session_chunks = total_session_chunks,
        "conversation ingested",
    );

    Ok(())
}

/// Parse session datetime string with flexible fallback.
///
/// LoCoMo uses the format `"4:04 pm on 20 January, 2023"`.
fn parse_session_datetime(dt: &str) -> DateTime<Utc> {
    // Try RFC 3339 first.
    if let Ok(parsed) = DateTime::parse_from_rfc3339(dt) {
        return parsed.with_timezone(&Utc);
    }

    // LoCoMo format: "4:04 pm on 20 January, 2023"
    // Strip "on " to get "4:04 pm 20 January, 2023"
    let normalized = dt.replace(" on ", " ");
    for fmt in &[
        "%l:%M %P %d %B, %Y", // "4:04 pm 20 January, 2023"
        "%l:%M %P %d %B %Y",  // "4:04 pm 20 January 2023"
        "%I:%M %P %d %B, %Y", // "04:04 pm 20 January, 2023"
    ] {
        if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(normalized.trim(), fmt) {
            return naive.and_utc();
        }
    }

    // Try standard formats.
    for fmt in &[
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%d %B %Y",
        "%B %d, %Y",
        "%Y-%m-%d",
    ] {
        if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(dt, fmt) {
            return naive.and_utc();
        }
        if let Ok(naive) = chrono::NaiveDate::parse_from_str(dt, fmt) {
            return naive.and_hms_opt(0, 0, 0).unwrap().and_utc();
        }
    }
    tracing::warn!(dt, "could not parse session datetime, using epoch");
    DateTime::UNIX_EPOCH
}

/// Get the other speaker's name in a 2-person conversation.
fn other_speaker_name(current_speaker: &str, conv: &Conversation) -> String {
    if current_speaker == conv.speaker_a {
        conv.speaker_b.clone()
    } else {
        conv.speaker_a.clone()
    }
}
