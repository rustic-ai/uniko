//! LongMemEval item ingestion into a fresh KnowledgeBase.
//!
//! Each LongMemEval item is self-contained with its own haystack
//! of chat sessions. We create one KB per item and ingest all
//! sessions with entity and observation extraction.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};
use futures::stream::{self, StreamExt};
use tokio::sync::Mutex;

use uniko_extract::ingest::atomic::ingest_message_atomic;
use uniko_pipes::types::IngestMessage;
use uniko_store::config::UnikoConfig;
use uniko_store::{KnowledgeBase, ModelRuntime};

use super::data::LongMemEvalItem;

/// Evidence mapping built during ingestion.
///
/// Tracks which sessions and messages contain evidence for the answer,
/// enabling retrieval metric computation (Recall@k, NDCG@k).
#[derive(Debug)]
pub struct EvidenceMap {
    /// Session IDs that contain evidence (from `answer_session_ids`).
    pub answer_session_ids: HashSet<String>,
    /// Message IDs where `has_answer == true`.
    pub answer_message_ids: Vec<String>,
    /// Maps session_id → list of message_ids in that session.
    pub session_to_messages: HashMap<String, Vec<String>>,
}

/// Ingest a single LongMemEval item into a fresh KB.
///
/// Creates a new KnowledgeBase, ingests all haystack sessions
/// concurrently (up to `session_concurrency` at a time), runs
/// entity/observation extraction, and returns the populated KB
/// along with an evidence map for retrieval evaluation.
///
/// `session_concurrency = 1` reproduces the original sequential
/// behavior.  Higher values overlap NLP work across sessions; all
/// workers write to the same `Arc<KnowledgeBase>`, so the actual win
/// is capped by uni-db's concurrent-write throughput.
pub async fn ingest_item(
    item: &LongMemEvalItem,
    ingest_dir: &Path,
    config: UnikoConfig,
    runtime: Arc<ModelRuntime>,
    session_concurrency: usize,
) -> Result<(Arc<KnowledgeBase>, EvidenceMap)> {
    // Shared runtime path: every KB across `--question-concurrency`
    // reuses one ONNX/Xervo runtime instead of loading its own. At
    // q≥3 the per-KB path OOMs an 8 GB GPU; sharing keeps VRAM
    // bounded by the single session's working budget.
    let kb = KnowledgeBase::open_with_runtime(ingest_dir, config, runtime)
        .await
        .context("creating KB")?;
    let kb = Arc::new(kb);

    // Shared mutable state.  evidence_map lives behind a Mutex because
    // sessions complete out of order; the merged map is consistent at
    // the end regardless.
    let evidence_map = Arc::new(Mutex::new(EvidenceMap {
        answer_session_ids: item.answer_session_ids.iter().cloned().collect(),
        answer_message_ids: Vec::new(),
        session_to_messages: HashMap::new(),
    }));

    // Cumulative counters / timings — atomics so workers can update
    // without blocking each other.  Sub-timings are u64 ms; top-level
    // accumulators are u64 ms (was u128, but timings fit comfortably).
    let total_turns = Arc::new(AtomicU64::new(0));
    let total_entities = Arc::new(AtomicUsize::new(0));
    let total_observations = Arc::new(AtomicUsize::new(0));
    let total_session_chunks = Arc::new(AtomicUsize::new(0));

    let t_ingest_msg_ms = Arc::new(AtomicU64::new(0));
    let t_entity_ms = Arc::new(AtomicU64::new(0));
    let t_obs_ms = Arc::new(AtomicU64::new(0));
    let t_session_chunk_ms = Arc::new(AtomicU64::new(0));
    let t_obs_chunk_ms = Arc::new(AtomicU64::new(0));

    // Sub-counters populated from AtomicTimings inside the worker loop.
    let t_ner_nlp_ms = Arc::new(AtomicU64::new(0));
    let t_ner_upsert_ms = Arc::new(AtomicU64::new(0));
    let t_obs_extract_ms = Arc::new(AtomicU64::new(0));
    let t_obs_nodes_ms = Arc::new(AtomicU64::new(0));

    // Build the per-session work items up front so the stream owns
    // them by value (cheap clones of the underlying Vecs).
    let session_jobs: Vec<(usize, Vec<super::data::LmeMessage>, String, String)> = item
        .haystack_sessions
        .iter()
        .zip(item.haystack_session_ids.iter())
        .zip(item.haystack_dates.iter())
        .enumerate()
        .map(|(idx, ((turns, sid), date))| (idx, turns.clone(), sid.clone(), date.clone()))
        .collect();

    let concurrency = session_concurrency.max(1);
    let question_id = item.question_id.clone();

    let results: Vec<Result<()>> = stream::iter(session_jobs)
        .map(|(session_idx, session_turns, session_id, date_str)| {
            // Clone Arcs for this worker.
            let kb = kb.clone();
            let evidence_map = evidence_map.clone();
            let total_turns = total_turns.clone();
            let total_entities = total_entities.clone();
            let total_observations = total_observations.clone();
            let total_session_chunks = total_session_chunks.clone();
            let t_ingest_msg_ms = t_ingest_msg_ms.clone();
            let t_entity_ms = t_entity_ms.clone();
            let t_obs_ms = t_obs_ms.clone();
            let t_session_chunk_ms = t_session_chunk_ms.clone();
            let t_obs_chunk_ms = t_obs_chunk_ms.clone();
            let t_ner_nlp_ms = t_ner_nlp_ms.clone();
            let t_ner_upsert_ms = t_ner_upsert_ms.clone();
            let t_obs_extract_ms = t_obs_extract_ms.clone();
            let t_obs_nodes_ms = t_obs_nodes_ms.clone();
            let question_id = question_id.clone();

            async move {
                let base_ts = parse_lme_datetime(&date_str);

                // Per-session state — pronoun resolution stays causal
                // within a single session (which is what the dataset
                // semantics require); concurrency is across sessions,
                // not turns within a session.
                let mut session_ctx =
                    uniko_extract::ingest::context::SessionContext::new(session_id.clone(), 0);
                session_ctx.register_participant("user", 0);
                session_ctx.register_participant("assistant", 0);

                let mut session_message_ids = Vec::with_capacity(session_turns.len());
                let mut local_evidence_msg_ids = Vec::new();

                for (turn_idx, turn) in session_turns.iter().enumerate() {
                    let timestamp = base_ts + Duration::seconds(turn_idx as i64 * 30);
                    let message_id = format!("{}-s{}-t{}", question_id, session_idx, turn_idx);

                    let other_role = if turn.role == "user" {
                        "assistant"
                    } else {
                        "user"
                    };

                    session_ctx.set_current_speaker(&turn.role);

                    let msg = IngestMessage {
                        message_id: message_id.clone(),
                        content: turn.content.clone(),
                        content_type: "text".to_string(),
                        sender_id: turn.role.clone(),
                        session_id: session_id.clone(),
                        addressed_to: Some(vec![other_role.to_string()]),
                        timestamp,
                        metadata: HashMap::new(),
                        category: None,
                        source_id: None,
                        revision_id: None,
                    };

                    // Atomic per-message ingest: one tx for Message +
                    // edges + chunks + Entities + MENTIONS + Observations
                    // + OBSERVED_IN + ABOUT. Replaces the legacy
                    // three-call sequence (ingest_message + entity_step
                    // + obs_step) with three independent commits.
                    let result = ingest_message_atomic(&kb, &msg, &mut session_ctx)
                        .await
                        .with_context(|| format!("ingesting {message_id}"))?;
                    total_entities.fetch_add(result.extracted_entities.len(), Ordering::Relaxed);
                    total_observations
                        .fetch_add(result.extracted_observations.len(), Ordering::Relaxed);

                    // Map atomic timings → bench counters.
                    let t = &result.timings;
                    let ingest_msg_chunk =
                        (t.setup_ms + t.create_ms + t.edges_ms + t.chunk_ms + t.commit_ms) as u64;
                    t_ingest_msg_ms.fetch_add(ingest_msg_chunk, Ordering::Relaxed);
                    t_entity_ms.fetch_add(
                        (t.prep_read_ms + t.apply_entity_ms) as u64,
                        Ordering::Relaxed,
                    );
                    t_obs_ms.fetch_add(t.apply_obs_ms as u64, Ordering::Relaxed);
                    // extract_ms covers all CPU NLP/NER/SRL work (incl.
                    // nlp_ms which is the per-sentence cascade). Surface
                    // it under the legacy nlp / extract counters so
                    // dashboards still see something non-zero.
                    t_ner_nlp_ms.fetch_add(t.nlp_ms as u64, Ordering::Relaxed);
                    t_ner_upsert_ms.fetch_add(t.apply_entity_ms as u64, Ordering::Relaxed);
                    t_obs_extract_ms.fetch_add(t.extract_ms as u64, Ordering::Relaxed);
                    t_obs_nodes_ms.fetch_add(t.apply_obs_ms as u64, Ordering::Relaxed);

                    if turn.has_answer {
                        local_evidence_msg_ids.push(message_id.clone());
                    }
                    session_message_ids.push(message_id);

                    let new_total = total_turns.fetch_add(1, Ordering::Relaxed) + 1;
                    if new_total == 1 || new_total.is_multiple_of(20) {
                        tracing::info!(
                            turn = new_total,
                            session = session_idx,
                            ingest_msg_ms = t_ingest_msg_ms.load(Ordering::Relaxed),
                            entity_ms = t_entity_ms.load(Ordering::Relaxed),
                            obs_ms = t_obs_ms.load(Ordering::Relaxed),
                            "turn processed (cumulative timings)",
                        );
                    }
                }

                // Chunk this session.
                let t0 = std::time::Instant::now();
                let chunk_ids =
                    uniko_extract::ingest::session_chunk::chunk_session(&kb, &session_id)
                        .await
                        .with_context(|| format!("chunking session {session_id}"))?;
                t_session_chunk_ms.fetch_add(t0.elapsed().as_millis() as u64, Ordering::Relaxed);
                total_session_chunks.fetch_add(chunk_ids.len(), Ordering::Relaxed);

                let t0 = std::time::Instant::now();
                let obs_chunk_ids =
                    uniko_extract::ingest::session_chunk::chunk_session_observations(
                        &kb,
                        &session_id,
                    )
                    .await
                    .with_context(|| format!("chunking observations {session_id}"))?;
                t_obs_chunk_ms.fetch_add(t0.elapsed().as_millis() as u64, Ordering::Relaxed);
                total_session_chunks.fetch_add(obs_chunk_ids.len(), Ordering::Relaxed);

                // Merge local results into the shared evidence_map.
                {
                    let mut em = evidence_map.lock().await;
                    em.answer_message_ids.extend(local_evidence_msg_ids);
                    em.session_to_messages
                        .insert(session_id, session_message_ids);
                }

                Ok::<(), anyhow::Error>(())
            }
        })
        .buffer_unordered(concurrency)
        .collect()
        .await;

    // Propagate the first error (if any) — matches sequential behavior
    // where a per-turn failure aborts the whole item ingest.
    for r in results {
        r?;
    }

    let evidence_map = Arc::try_unwrap(evidence_map)
        .map_err(|_| anyhow::anyhow!("evidence_map still has outstanding refs"))?
        .into_inner();

    let load = |a: &Arc<AtomicU64>| a.load(Ordering::Relaxed);
    let load_u = |a: &Arc<AtomicUsize>| a.load(Ordering::Relaxed);

    let total_ms = load(&t_ingest_msg_ms)
        + load(&t_entity_ms)
        + load(&t_obs_ms)
        + load(&t_session_chunk_ms)
        + load(&t_obs_chunk_ms);
    tracing::info!(
        question_id = %item.question_id,
        sessions = item.haystack_sessions.len(),
        turns = total_turns.load(Ordering::Relaxed),
        entities = load_u(&total_entities),
        observations = load_u(&total_observations),
        session_chunks = load_u(&total_session_chunks),
        evidence_messages = evidence_map.answer_message_ids.len(),
        session_concurrency = concurrency,
        "item ingested",
    );
    tracing::info!(
        question_id = %item.question_id,
        ingest_msg_ms = load(&t_ingest_msg_ms),
        entity_ms = load(&t_entity_ms),
        obs_ms = load(&t_obs_ms),
        session_chunk_ms = load(&t_session_chunk_ms),
        obs_chunk_ms = load(&t_obs_chunk_ms),
        total_ms = total_ms,
        "ingestion timing breakdown (cpu-time across workers)",
    );
    tracing::info!(
        question_id = %item.question_id,
        ner_nlp_ms = load(&t_ner_nlp_ms),
        ner_upsert_ms = load(&t_ner_upsert_ms),
        "entity extraction sub-timings",
    );
    tracing::info!(
        question_id = %item.question_id,
        obs_dep_extract_ms = load(&t_obs_extract_ms),
        obs_graph_nodes_ms = load(&t_obs_nodes_ms),
        "observation extraction sub-timings",
    );

    Ok((kb, evidence_map))
}

/// Parse LongMemEval datetime format: "YYYY/MM/DD (Day) HH:MM".
fn parse_lme_datetime(dt: &str) -> DateTime<Utc> {
    // Try the LongMemEval format first: "2023/04/10 (Mon) 23:07"
    for fmt in &["%Y/%m/%d (%a) %H:%M", "%Y/%m/%d %H:%M"] {
        if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(dt, fmt) {
            return naive.and_utc();
        }
    }

    // Try RFC 3339 fallback.
    if let Ok(parsed) = DateTime::parse_from_rfc3339(dt) {
        return parsed.with_timezone(&Utc);
    }

    // Try standard formats.
    for fmt in &["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"] {
        if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(dt, fmt) {
            return naive.and_utc();
        }
        if let Ok(naive) = chrono::NaiveDate::parse_from_str(dt, fmt) {
            return naive.and_hms_opt(0, 0, 0).unwrap().and_utc();
        }
    }

    tracing::warn!(dt, "could not parse LME datetime, using epoch");
    DateTime::UNIX_EPOCH
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Datelike;

    #[test]
    fn test_parse_lme_datetime() {
        let dt = parse_lme_datetime("2023/04/10 (Mon) 23:07");
        assert_eq!(dt.year(), 2023);
        assert_eq!(dt.month(), 4);
        assert_eq!(dt.day(), 10);
    }

    #[test]
    fn test_parse_lme_datetime_fallback() {
        let dt = parse_lme_datetime("2023-01-15T10:30:00+00:00");
        assert_eq!(dt.year(), 2023);

        let dt = parse_lme_datetime("garbage");
        assert_eq!(dt, DateTime::UNIX_EPOCH);
    }
}
