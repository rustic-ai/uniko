//! Atomic multi-turn ingest: a related set of turns, and their attachments,
//! written in ONE transaction that commits once.
//!
//! `ingest_message_atomic` gives per-message atomicity. That is not enough
//! for a conversational integration that records a user message and the
//! assistant's answer as one logical exchange: an interruption between the
//! two `observe` calls leaves half an answered turn permanently visible — a
//! question with no answer, indistinguishable from a real one during later
//! recall. This module is the all-or-nothing boundary for that set
//! (`rustic-ai/uniko#40`).

use std::collections::{HashMap, HashSet};

use uniko_pipes::types::IngestMessage;
use uniko_store::{KnowledgeBase, NodeId, UnikoError};

use super::atomic::{AtomicIngestResult, AtomicTimings};
use super::context::{SessionContext, advance_speaker};
use super::message::{MessageSetup, apply_message_writes_in_tx};
use super::source::{IngestOutcome, PreparedSource};
use crate::ner::dedup::{
    UnitEntityPrep, apply_entity_mentions_in_tx, apply_entity_upsert_nodes, merge_entity_preps,
};
use crate::observations::{
    ObservationInputs, ObservationPrepOutcome, apply_observations, prepare_observations,
};

/// One turn of a unit: the message plus the attachments that must land in
/// the same transaction as it.
#[derive(Debug)]
pub struct UnitTurn {
    /// The message to record.
    pub message: IngestMessage,
    /// Attachments, already prepared outside any transaction (blob PUT,
    /// hashing, chunking, model inference). Written inside the unit's
    /// transaction and linked `Artifact -ATTACHED_TO-> Message` for this
    /// turn.
    pub attachments: Vec<PreparedSource>,
}

/// What one [`ingest_turns_atomic`] call wrote.
#[derive(Debug)]
pub struct UnitIngestResult {
    /// One per input turn, in unit order. Same shape as a single-message
    /// ingest, so existing callers of [`AtomicIngestResult`] are unchanged.
    pub turns: Vec<AtomicIngestResult>,
    /// Attachment outcomes, grouped per turn and parallel to `turns`.
    pub attachments: Vec<Vec<IngestOutcome>>,
    /// True when every `message_id` was already present with identical
    /// content: nothing was written and no transaction was opened.
    pub was_replay: bool,
    /// Transaction attempts consumed (1 on the happy path).
    pub attempts: u32,
}

/// Fault-injection hook: fail the unit inside the transaction, after turn
/// `n`'s writes but before the commit, so a test can prove the rollback
/// path rather than only the pre-transaction rejection path. Those are
/// different guarantees.
///
/// Env vars are process-global, which is safe here only because nextest —
/// the runner of record — gives every test its own process.
#[doc(hidden)]
const FAIL_AFTER_TURN_ENV: &str = "UNIKO_TEST_FAIL_AFTER_TURN";

/// Fault-injection hook: `abort()` the PROCESS inside the transaction,
/// after turn `n`'s writes but before the commit.
///
/// Distinct from [`FAIL_AFTER_TURN_ENV`], which returns an error and lets
/// the transaction roll back cleanly. This kills the process outright — no
/// unwinding, no `Drop`, no rollback — which is the only way to test what
/// the issue actually asks for: that an interrupted unit leaves nothing
/// behind after the store is REOPENED. A clean rollback and a crash are
/// different guarantees, and only the second one depends on durability.
#[doc(hidden)]
const ABORT_AFTER_TURN_ENV: &str = "UNIKO_TEST_ABORT_AFTER_TURN";

/// Atomic multi-turn ingest.
///
/// Writes the WHOLE unit — every Message with its edges and chunks, the
/// unit's merged Entity upsert, every turn's MENTIONS and Observations, and
/// every attachment — inside one transaction that commits once. Any failure
/// drops the transaction, so a unit never lands half-written and
/// `session_ctx` is never advanced.
///
/// # Errors
///
/// - [`UnikoError::Config`] when the unit is empty, or a turn does not
///   belong to `session_ctx.session_id`.
/// - [`UnikoError::IdConflict`] when a `message_id` names existing content
///   that differs, when the unit repeats a `message_id` internally, or when
///   the unit is only *partly* already recorded.
/// - any underlying read, write, or extraction error.
pub async fn ingest_turns_atomic(
    kb: &KnowledgeBase,
    unit: &[UnitTurn],
    session_ctx: &mut SessionContext,
) -> uniko_store::Result<UnitIngestResult> {
    let started = std::time::Instant::now();

    // ── P0. Validate before touching the graph. ─────────────────────────
    if unit.is_empty() {
        return Err(UnikoError::Config(
            "ingest_turns_atomic: a unit must contain at least one turn".into(),
        ));
    }
    let mut seen_ids: HashSet<&str> = HashSet::with_capacity(unit.len());
    for turn in unit {
        if turn.message.session_id != session_ctx.session_id {
            return Err(UnikoError::Config(format!(
                "ingest_turns_atomic: turn '{}' targets session '{}' but the unit's \
                 context is session '{}' — a unit cannot span sessions",
                turn.message.message_id, turn.message.session_id, session_ctx.session_id
            )));
        }
        // Two turns sharing a message_id would mint two Message nodes whose
        // deterministic chunk ids collide, so reject rather than write
        // something that cannot be read back.
        if !seen_ids.insert(turn.message.message_id.as_str()) {
            return Err(UnikoError::id_conflict(
                "Message",
                "message_id",
                &turn.message.message_id,
            ));
        }
    }

    // ── P1. Idempotency probe for every turn, before anything is written.
    let mut present: Vec<String> = Vec::new();
    let mut absent: Vec<String> = Vec::new();
    let mut existing_nids: HashMap<&str, NodeId> = HashMap::new();
    for turn in unit {
        match kb
            .get_node_by_ext_id("Message", "message_id", &turn.message.message_id)
            .await?
        {
            Some((nid, props)) => {
                let stored = match props.get("content") {
                    Some(uniko_store::Value::String(s)) => s.as_str(),
                    _ => "",
                };
                // A content mismatch fails the WHOLE unit immediately —
                // non-retriable, so the retry loop below cannot spin on it.
                if stored != turn.message.content {
                    return Err(UnikoError::id_conflict(
                        "Message",
                        "message_id",
                        &turn.message.message_id,
                    ));
                }
                existing_nids.insert(turn.message.message_id.as_str(), nid);
                present.push(turn.message.message_id.clone());
            }
            None => absent.push(turn.message.message_id.clone()),
        }
    }

    if absent.is_empty() {
        // Whole-unit replay. Advance the chain head to the unit's LAST turn:
        // its NEXT edge already exists on disk, so a following unit chaining
        // off it is correct. (The single-message path leaves the head stale,
        // which costs the *next* turn its NEXT edge.)
        let last = &unit[unit.len() - 1].message;
        session_ctx.prev_message_nid = existing_nids.get(last.message_id.as_str()).copied();
        session_ctx.prev_message_ts = Some(last.timestamp);
        let turns = unit
            .iter()
            .map(|t| AtomicIngestResult {
                message_node_id: existing_nids
                    .get(t.message.message_id.as_str())
                    .copied()
                    .unwrap_or(0),
                chunk_node_ids: Vec::new(),
                session_node_id: session_ctx.session_nid,
                sender: None,
                extracted_entities: Vec::new(),
                extracted_observations: Vec::new(),
                timings: AtomicTimings::default(),
            })
            .collect();
        return Ok(UnitIngestResult {
            turns,
            attachments: unit.iter().map(|_| Vec::new()).collect(),
            was_replay: true,
            attempts: 0,
        });
    }
    if !present.is_empty() {
        return Err(UnikoError::partial_unit(&present, &absent));
    }

    // ── P2. Sessions and senders, once per distinct sender. ─────────────
    // These take `setup_locks` and commit on their own, so they must finish
    // — guards released — before any `rmw_locks` guard is taken (P7). A
    // `:Session` row with no messages is already a normal state, so their
    // being outside the unit's atomicity is correct, not a gap.
    let setup_start = std::time::Instant::now();
    let mut session_nid: NodeId = 0;
    let mut sender_nids: HashMap<String, NodeId> = HashMap::new();
    for turn in unit {
        if sender_nids.contains_key(&turn.message.sender_id) {
            continue;
        }
        let (snid, pnid) =
            super::message::ensure_session_and_sender(kb, &turn.message, session_ctx).await?;
        session_nid = snid;
        sender_nids.insert(turn.message.sender_id.clone(), pnid);
    }

    // ── P3. Recipients, per turn, over a PROGRESSIVE participant view. ──
    // `resolve_recipients`'s inference branch returns every cached
    // participant except the sender. With all of the unit's senders already
    // registered, turn 1 would be ADDRESSED_TO someone who has not spoken
    // yet — a divergence from the identical sequence of `observe` calls.
    // Feeding it only the senders seen at or before each turn keeps unit
    // ingest byte-identical to sequential ingest.
    let mut recipients: Vec<Vec<NodeId>> = Vec::with_capacity(unit.len());
    let mut progressive = session_ctx.clone();
    progressive.participants.clear();
    for turn in unit {
        let sender_nid = sender_nids[&turn.message.sender_id];
        progressive.register_participant(&turn.message.sender_id, sender_nid);
        recipients.push(
            super::message::resolve_recipients(kb, &turn.message, sender_nid, &progressive).await?,
        );
    }
    let setup_ms = setup_start.elapsed().as_millis();

    // ── P4. CPU extraction per turn. ────────────────────────────────────
    let extract_start = std::time::Instant::now();
    let mut per_turn_preps = Vec::with_capacity(unit.len());
    #[cfg(feature = "onnx")]
    let mut nlp_per_turn = Vec::with_capacity(unit.len());
    for turn in unit {
        let out = super::atomic::extract_for_unit(kb, &turn.message).await;
        per_turn_preps.push(crate::ner::dedup::prepare_entity_upsert(kb, out.deduped).await?);
        #[cfg(feature = "onnx")]
        nlp_per_turn.push(out.nlp_results);
    }
    let extract_ms = extract_start.elapsed().as_millis();

    // ── P5. One entity batch for the whole unit. ────────────────────────
    let unit_prep: UnitEntityPrep = merge_entity_preps(per_turn_preps);

    // ── P6/P7. One RMW acquisition over entities AND attachment content.
    // Two calls could self-deadlock on a shared stripe; see
    // `KnowledgeBase::lock_ingest_unit`.
    let content_ids: Vec<String> = unit
        .iter()
        .flat_map(|t| t.attachments.iter())
        .flat_map(PreparedSource::content_ids)
        .collect();
    let _guards = kb
        .lock_ingest_unit(&unit_prep.merged.entity_ids, &content_ids)
        .await;

    // Bases, hoisted so `session_ctx` is not borrowed inside the loop at
    // all. Every attempt re-seeds from these, which is what makes
    // "unmutated until commit" a property the compiler enforces rather than
    // a comment.
    let base_prev_nid = session_ctx.prev_message_nid;
    let base_prev_ts = session_ctx.prev_message_ts;
    let base_sentence_ctx = session_ctx.sentence_ctx.clone();
    let participants = session_ctx.participants.clone();
    let rules_path = kb.config().observation_rules_path.clone();

    let retry_opts = uniko_store::RetryOptions::default();
    let mut attempts: u32 = 0;

    let (
        turn_results,
        attachment_outcomes,
        final_prev_nid,
        final_prev_ts,
        final_sentence_ctx,
        commit_ms,
    ) = loop {
        attempts += 1;
        let mut prev_nid = base_prev_nid;
        let mut prev_ts = base_prev_ts;
        let mut sentence_ctx = base_sentence_ctx.clone();
        let merged_attempt = unit_prep.merged.clone();
        // Fresh per attempt: a rolled-back attempt's node ids are stale.
        let mut seen = super::artifact::UnitArtifactSeen::new();

        let tx = kb.begin_tx().await?;
        let body: uniko_store::Result<_> = async {
            // (a) Messages, in unit order. Chaining lives in locals.
            let mut message_nids: Vec<NodeId> = Vec::with_capacity(unit.len());
            let mut writes_per_turn = Vec::with_capacity(unit.len());
            for (i, turn) in unit.iter().enumerate() {
                let setup = MessageSetup {
                    session_nid,
                    participant_nid: sender_nids[&turn.message.sender_id],
                    recipient_nids: recipients[i].clone(),
                    prev_msg_nid: prev_nid,
                    prev_msg_ts: prev_ts,
                };
                let writes = apply_message_writes_in_tx(kb, &tx, &turn.message, &setup).await?;
                prev_nid = Some(writes.message_node_id);
                prev_ts = Some(turn.message.timestamp);
                message_nids.push(writes.message_node_id);
                writes_per_turn.push(writes);

                // Test-only: abort mid-unit so a rollback test can assert
                // that the turns written BEFORE this point leave no trace.
                if let Ok(raw) = std::env::var(ABORT_AFTER_TURN_ENV)
                    && raw.trim().parse::<usize>() == Ok(i)
                {
                    // Kill the process with the write in the open
                    // transaction and NOT committed.
                    std::process::abort();
                }
                if let Ok(raw) = std::env::var(FAIL_AFTER_TURN_ENV)
                    && raw.trim().parse::<usize>() == Ok(i)
                {
                    return Err(UnikoError::Internal(format!(
                        "{FAIL_AFTER_TURN_ENV}={raw}: injected failure after turn {i}"
                    )));
                }
            }

            // (b) ONE entity node upsert for the whole unit, so an entity
            //     named in two turns yields one row with a summed frequency.
            let matches = apply_entity_upsert_nodes(kb, &tx, merged_attempt).await?;

            // (c) ONE batched MENTIONS write, per-message sources — dedup
            //     must not collapse provenance.
            apply_entity_mentions_in_tx(kb, &tx, &unit_prep.mentions(&message_nids, &matches))
                .await?;

            // (d) Observations per turn, speaker and pronoun window threaded
            //     through locals.
            let mut obs_per_turn: Vec<Vec<NodeId>> = Vec::with_capacity(unit.len());
            let mut entities_per_turn: Vec<Vec<(NodeId, String)>> = Vec::with_capacity(unit.len());
            for (i, turn) in unit.iter().enumerate() {
                advance_speaker(&mut sentence_ctx, &participants, &turn.message.sender_id);
                let extracted = unit_prep.entities_for_turn(i, &matches);
                let sender_nid = sender_nids[&turn.message.sender_id];
                let inputs = ObservationInputs {
                    kb,
                    message_node_id: message_nids[i],
                    content: &turn.message.content,
                    content_type: &turn.message.content_type,
                    // ALWAYS Some: when this is None `prepare_observations`
                    // falls back to a SENT_BY lookup against `kb`, not `tx`,
                    // which reads outside this snapshot and silently misses
                    // the edge we just wrote.
                    sender: Some((sender_nid, turn.message.sender_id.clone())),
                    extracted_entities: &extracted,
                    #[cfg(feature = "onnx")]
                    nlp_results: nlp_per_turn[i].as_deref(),
                    seed_sentence_ctx: Some(&sentence_ctx),
                    timestamp: turn.message.timestamp,
                    observation_rules_path: rules_path.as_deref(),
                    category: turn.message.category.as_deref(),
                    source_id: turn.message.source_id.as_deref(),
                    revision_id: turn.message.revision_id.as_deref(),
                };
                let (obs_nids, sc_updated) = match prepare_observations(inputs).await? {
                    ObservationPrepOutcome::Skip(_) => (Vec::new(), None),
                    ObservationPrepOutcome::Ready(prep) => {
                        let sc = prep.sentence_ctx_updated.clone();
                        (
                            apply_observations(kb, &tx, message_nids[i], *prep).await?,
                            sc,
                        )
                    }
                };
                if let Some(sc) = sc_updated {
                    sentence_ctx = sc;
                }
                obs_per_turn.push(obs_nids);
                entities_per_turn.push(extracted);
            }

            // (e) Attachments, in order, in the SAME transaction. The
            //     Message nid comes from this transaction — resolving it by
            //     external id beforehand would always miss, and a miss is
            //     skipped silently.
            let mut attachment_outcomes: Vec<Vec<IngestOutcome>> = Vec::with_capacity(unit.len());
            for (i, turn) in unit.iter().enumerate() {
                let ctx = super::artifact::ArtifactContextNids {
                    session_nid: Some(session_nid),
                    message_nid: Some(message_nids[i]),
                    action_nid: None,
                };
                let mut per_turn = Vec::with_capacity(turn.attachments.len());
                for prepared in &turn.attachments {
                    per_turn.push(prepared.apply_in_tx(kb, &tx, ctx, &mut seen).await?);
                }
                attachment_outcomes.push(per_turn);
            }

            Ok((
                writes_per_turn,
                message_nids,
                entities_per_turn,
                obs_per_turn,
                attachment_outcomes,
                sentence_ctx,
                prev_nid,
                prev_ts,
            ))
        }
        .await;

        match body {
            Ok((
                writes_per_turn,
                message_nids,
                entities_per_turn,
                obs_per_turn,
                attachment_outcomes,
                sentence_ctx,
                prev_nid,
                prev_ts,
            )) => {
                let commit_start = std::time::Instant::now();
                match tx.commit().await {
                    Ok(_) => {
                        let results: Vec<AtomicIngestResult> = unit
                            .iter()
                            .enumerate()
                            .map(|(i, turn)| AtomicIngestResult {
                                message_node_id: message_nids[i],
                                chunk_node_ids: writes_per_turn[i].chunk_node_ids.clone(),
                                session_node_id: session_nid,
                                sender: Some((
                                    sender_nids[&turn.message.sender_id],
                                    turn.message.sender_id.clone(),
                                )),
                                extracted_entities: entities_per_turn[i].clone(),
                                extracted_observations: obs_per_turn[i].clone(),
                                timings: AtomicTimings {
                                    setup_ms,
                                    extract_ms,
                                    create_ms: writes_per_turn[i].create_ms,
                                    edges_ms: writes_per_turn[i].edges_ms,
                                    chunk_ms: writes_per_turn[i].chunk_ms,
                                    ..AtomicTimings::default()
                                },
                            })
                            .collect();
                        break (
                            results,
                            attachment_outcomes,
                            prev_nid,
                            prev_ts,
                            sentence_ctx,
                            commit_start.elapsed().as_millis(),
                        );
                    }
                    // `commit` consumed `tx`; nothing to roll back.
                    Err(e) => {
                        let err = UnikoError::from(e);
                        if err.is_retriable() && attempts < retry_opts.max_attempts {
                            super::atomic::unit_retry_backoff(&retry_opts, attempts + 1).await;
                            continue;
                        }
                        return Err(err);
                    }
                }
            }
            Err(err) => {
                tx.rollback();
                if err.is_retriable() && attempts < retry_opts.max_attempts {
                    super::atomic::unit_retry_backoff(&retry_opts, attempts + 1).await;
                    continue;
                }
                return Err(err);
            }
        }
    };

    // ── Post-commit, once. Order matters: assign the pronoun window first,
    //    then the speaker, so SessionContext's own fields end consistent.
    session_ctx.prev_message_nid = final_prev_nid;
    session_ctx.prev_message_ts = final_prev_ts;
    session_ctx.sentence_ctx = final_sentence_ctx;
    if let Some(last) = unit.last() {
        session_ctx.set_current_speaker(&last.message.sender_id);
    }

    let mut turns = turn_results;
    if let Some(first) = turns.first_mut() {
        first.timings.commit_ms = commit_ms;
        first.timings.total_ms = started.elapsed().as_millis();
    }

    // Post-commit, best-effort: mean-pooled artifact embeddings and any
    // host `finish_post_commit`. These read committed rows through a fresh
    // session, so they cannot run inside the transaction.
    for (i, turn) in unit.iter().enumerate() {
        if let Some(outcomes) = attachment_outcomes.get(i) {
            for (prepared, outcome) in turn.attachments.iter().zip(outcomes.iter()) {
                prepared.finish_post_commit(kb, outcome).await;
            }
        }
    }

    Ok(UnitIngestResult {
        turns,
        attachments: attachment_outcomes,
        was_replay: false,
        attempts,
    })
}
