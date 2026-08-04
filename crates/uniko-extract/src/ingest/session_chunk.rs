//! Session-level chunking for retrieval.
//!
//! Two chunking strategies:
//!
//! 1. **Transcript chunks** ([`chunk_session`]): concatenate all messages
//!    with speaker prefixes, then chunk into 400-512 token segments.
//!
//! 2. **Observation chunks** ([`chunk_session_observations`]): aggregate
//!    per-message Observation content into dense factual chunks. Individual
//!    Observation nodes are directly searchable in their own right (they
//!    carry a full-text index on `content` and a vector index on
//!    `embedding`); these chunks add a denser, session-granularity surface
//!    on top, wired `ABOUT` the entities and participants involved.
//!
//! Both surfaces are what session-scoped recall and the Phase 1 session
//! boost retrieve. [`ChunkMode`] decides what happens when they already
//! exist: [`ChunkMode::Once`] leaves them alone, [`ChunkMode::Refresh`]
//! rebuilds them if the session has grown.

use std::collections::{HashMap, HashSet};

use uniko_store::{KnowledgeBase, NodeId, Value};

use super::chunking::text::TextChunker;
use super::chunking::{ChunkConfig, ChunkData, Chunker};
use super::message::{create_chunks, create_chunks_in_tx};

/// How an existing session-level chunk surface is treated on a re-run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ChunkMode {
    /// Leave existing chunks untouched and return them. Build-once
    /// semantics: cheapest, but a session that grows afterwards keeps a
    /// permanently stale surface.
    #[default]
    Once,
    /// Rebuild when the session has changed since the chunks were built,
    /// no-op when it has not. The rebuild deletes the previous generation
    /// and writes the replacement in one transaction.
    Refresh,
}

/// The result of a session-level chunk build: the chunk ids, and whether
/// anything was actually written.
///
/// `rebuilt == false` means the surface was already current and no node was
/// created, deleted, or re-embedded.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionChunkOutcome {
    /// The session's chunk node ids for this surface, after the call.
    pub ids: Vec<NodeId>,
    /// Whether this call wrote to the graph.
    pub rebuilt: bool,
}

/// Delete `doomed` and write `chunks` in a single transaction.
///
/// Chunk writes go through `bulk_insert_vertices`, which is a plain insert,
/// and `chunk_id` carries no uniqueness constraint — so a rebuild that did
/// not delete first would silently duplicate every chunk and double the
/// recall candidates. Doing both in one transaction also means recall never
/// observes a session with no chunks at all.
async fn apply_plan(
    kb: &KnowledgeBase,
    plan: ChunkPlan,
    parent_ext_id: &str,
    parent_nid: NodeId,
    chunks: &[ChunkData],
) -> uniko_store::Result<Vec<NodeId>> {
    let ChunkPlan::Rebuild {
        keep,
        doomed,
        from_index,
    } = plan
    else {
        unreachable!("Reuse is short-circuited by the caller");
    };
    let to_create = &chunks[from_index.min(chunks.len())..];

    // Nothing existed: the plain create path already retries on conflict.
    if keep.is_empty() && doomed.is_empty() {
        return create_chunks(kb, parent_ext_id, parent_nid, to_create, "Session").await;
    }

    let start = std::time::Instant::now();
    let created = kb
        .transact_with_retry(uniko_store::RetryOptions::default(), |tx| async {
            let r = async {
                kb.detach_delete_nodes_in_tx(&tx, &doomed).await?;
                create_chunks_in_tx(kb, &tx, parent_ext_id, parent_nid, to_create, "Session").await
            }
            .await;
            (tx, r)
        })
        .await?;
    tracing::info!(
        target: "tx_perf",
        tx_phase = "session_chunks_replace",
        total_ms = start.elapsed().as_millis() as u64,
        node_count = created.len() as u64,
        deleted_count = doomed.len() as u64,
        reused_count = keep.len() as u64,
        "tx phase",
    );
    let mut out = keep;
    out.extend(created);
    Ok(out)
}

/// Drop a chunk surface that no longer has any source content.
///
/// Reachable when every message (or observation) behind a chunked session is
/// deleted afterwards: without this the chunks would linger in the vector and
/// full-text indexes and stay recallable, describing content that is gone.
async fn drop_stale_chunks(
    kb: &KnowledgeBase,
    mode: ChunkMode,
    existing: &[uniko_store::repository::ingest::SessionChunkRow],
) -> uniko_store::Result<SessionChunkOutcome> {
    if mode != ChunkMode::Refresh || existing.is_empty() {
        return Ok(SessionChunkOutcome::default());
    }
    let doomed: Vec<NodeId> = existing.iter().map(|r| r.node_id).collect();
    kb.transact_with_retry(uniko_store::RetryOptions::default(), |tx| async {
        let r = kb.detach_delete_nodes_in_tx(&tx, &doomed).await;
        (tx, r)
    })
    .await?;
    Ok(SessionChunkOutcome {
        ids: Vec::new(),
        rebuilt: true,
    })
}

/// What a refresh should do with the chunks already on the Session.
enum ChunkPlan {
    /// Nothing changed — reuse these ids, write nothing.
    Reuse(Vec<NodeId>),
    /// Keep the leading `keep` chunks, delete `doomed`, and create only
    /// `fresh[from_index..]`.
    Rebuild {
        keep: Vec<NodeId>,
        doomed: Vec<NodeId>,
        from_index: usize,
    },
}

/// Decide what to do about an existing chunk surface.
///
/// Chunking is deterministic, so an unchanged session re-chunks to
/// byte-identical text — that is the staleness test, and it costs no
/// embeddings. When the session *has* grown, the chunks are compared
/// index by index and only the suffix from the first mismatch is rebuilt.
/// Appending turns therefore re-embeds the tail, not the whole transcript,
/// which is what keeps `Refresh` affordable on a long-running session.
///
/// Reusing a prefix is safe unconditionally: a chunk is kept only when its
/// stored text exactly equals the freshly computed text at the same index.
fn resolve_existing(
    mode: ChunkMode,
    existing: &[uniko_store::repository::ingest::SessionChunkRow],
    fresh: &[ChunkData],
) -> ChunkPlan {
    let ids: Vec<NodeId> = existing.iter().map(|r| r.node_id).collect();
    if existing.is_empty() {
        return ChunkPlan::Rebuild {
            keep: Vec::new(),
            doomed: Vec::new(),
            from_index: 0,
        };
    }
    if mode == ChunkMode::Once {
        return ChunkPlan::Reuse(ids);
    }

    let common = existing
        .iter()
        .zip(fresh)
        .take_while(|(e, f)| e.text == f.text)
        .count();
    if common == existing.len() && common == fresh.len() {
        return ChunkPlan::Reuse(ids);
    }
    ChunkPlan::Rebuild {
        keep: ids[..common].to_vec(),
        doomed: ids[common..].to_vec(),
        from_index: common,
    }
}

/// Chunk all messages in a session into searchable Chunk nodes.
///
/// Build-once wrapper over [`chunk_session_with`]: skips entirely when the
/// session already has transcript chunks. See [`ChunkMode`] for the
/// rebuilding variant.
///
/// # Errors
///
/// Returns [`UnikoError::Storage`] on graph query or write failure.
pub async fn chunk_session(
    kb: &KnowledgeBase,
    session_id: &str,
) -> uniko_store::Result<Vec<NodeId>> {
    Ok(chunk_session_with(kb, session_id, ChunkMode::Once)
        .await?
        .ids)
}

/// Chunk all messages in a session into searchable Chunk nodes.
///
/// Queries messages in the session ordered by timestamp, concatenates
/// them with `"speaker: content\n"` prefixes, then runs the text
/// chunker. Creates Chunk nodes linked to the Session via HAS_CHUNK.
///
/// Under [`ChunkMode::Refresh`] an unchanged session is a no-op that
/// re-embeds nothing; a grown one is rebuilt atomically.
///
/// # Errors
///
/// Returns [`UnikoError::Storage`] on graph query or write failure.
pub async fn chunk_session_with(
    kb: &KnowledgeBase,
    session_id: &str,
    mode: ChunkMode,
) -> uniko_store::Result<SessionChunkOutcome> {
    // Look up the session node.
    let session_nid = match kb
        .get_node_by_ext_id("Session", "session_id", session_id)
        .await?
    {
        Some((nid, _)) => nid,
        None => {
            tracing::debug!(session_id, "session not found, skipping chunking");
            return Ok(SessionChunkOutcome::default());
        }
    };

    // Filter by chunk_type rather than walking every outgoing HAS_CHUNK:
    // the Session also owns the observation surface, and the two must not
    // mask each other's presence check.
    let existing = kb.session_chunk_rows(session_id, "session").await?;
    if mode == ChunkMode::Once && !existing.is_empty() {
        tracing::debug!(
            session_id,
            chunks = existing.len(),
            "session already chunked"
        );
        return Ok(SessionChunkOutcome {
            ids: existing.iter().map(|r| r.node_id).collect(),
            rebuilt: false,
        });
    }

    // Query all messages in this session with their sender names,
    // ordered by timestamp.
    let rows = kb.session_transcript_rows(session_id).await?;

    if rows.is_empty() {
        tracing::debug!(session_id, "no messages in session");
        return drop_stale_chunks(kb, mode, &existing).await;
    }

    // Concatenate messages with speaker prefixes.
    let mut transcript = String::new();
    let mut speakers = HashSet::new();

    for row in &rows {
        if !row.content.is_empty() {
            speakers.insert(row.speaker.clone());
            transcript.push_str(&row.speaker);
            transcript.push_str(": ");
            transcript.push_str(&row.content);
            transcript.push('\n');
        }
    }

    if transcript.is_empty() {
        return drop_stale_chunks(kb, mode, &existing).await;
    }

    // Chunk the transcript.
    let chunk_cfg = ChunkConfig::from_uniko_config(kb.config());
    let chunker = TextChunker;
    let mut chunks: Vec<ChunkData> = Chunker::chunk(&chunker, &transcript, &chunk_cfg);

    // Set metadata on each chunk.
    let speaker_list = {
        let mut sorted: Vec<&str> = speakers.iter().map(String::as_str).collect();
        sorted.sort_unstable();
        sorted.join(", ")
    };

    // Get session topic if available.
    let topic = kb.get_node(session_nid).await?.and_then(|(_, props)| {
        props
            .get("topic")
            .and_then(|v| v.as_str())
            .map(String::from)
    });

    for chunk in &mut chunks {
        chunk.chunk_type = "session".to_string();
        chunk.heading = topic.clone();
        // Store speakers in symbol_name field (reusing existing schema field).
        chunk.symbol_name = Some(speaker_list.clone());
    }

    let plan = match resolve_existing(mode, &existing, &chunks) {
        ChunkPlan::Reuse(ids) => {
            tracing::debug!(session_id, chunks = ids.len(), "session chunks up to date");
            return Ok(SessionChunkOutcome {
                ids,
                rebuilt: false,
            });
        }
        rebuild => rebuild,
    };
    let chunk_nids = apply_plan(kb, plan, session_id, session_nid, &chunks).await?;

    tracing::info!(
        session_id,
        messages = rows.len(),
        chunks = chunk_nids.len(),
        transcript_len = transcript.len(),
        "session chunked",
    );

    Ok(SessionChunkOutcome {
        ids: chunk_nids,
        rebuilt: true,
    })
}

/// Aggregate per-message Observations into searchable Chunk nodes.
///
/// Queries all Observation nodes linked to messages in this session,
/// deduplicates by normalized content, concatenates into a dense text
/// block, then creates Chunk node(s) with `chunk_type = "observation"`.
///
/// Also wires ABOUT edges from observation chunks to any entities
/// referenced by the underlying observations, preserving entity-scoped
/// search capability.
///
/// Build-once wrapper over [`chunk_session_observations_with`].
///
/// # Errors
///
/// Returns [`UnikoError::Storage`] on graph query or write failure.
pub async fn chunk_session_observations(
    kb: &KnowledgeBase,
    session_id: &str,
) -> uniko_store::Result<Vec<NodeId>> {
    Ok(
        chunk_session_observations_with(kb, session_id, ChunkMode::Once)
            .await?
            .ids,
    )
}

/// Aggregate per-message Observations into searchable Chunk nodes.
///
/// See [`chunk_session_observations`] for the surface this builds. Under
/// [`ChunkMode::Refresh`] an unchanged session is a no-op; a grown one is
/// rebuilt atomically, and the `ABOUT` fan-out is rebuilt with it (the
/// `DETACH DELETE` takes the previous generation's edges).
///
/// # Errors
///
/// Returns [`UnikoError::Storage`] on graph query or write failure.
pub async fn chunk_session_observations_with(
    kb: &KnowledgeBase,
    session_id: &str,
    mode: ChunkMode,
) -> uniko_store::Result<SessionChunkOutcome> {
    // Look up the session node.
    let session_nid = match kb
        .get_node_by_ext_id("Session", "session_id", session_id)
        .await?
    {
        Some((nid, _)) => nid,
        None => {
            tracing::debug!(session_id, "session not found, skipping obs chunking");
            return Ok(SessionChunkOutcome::default());
        }
    };

    // Check if observation chunks already exist (idempotent).
    let existing = kb.session_chunk_rows(session_id, "observation").await?;
    if mode == ChunkMode::Once && !existing.is_empty() {
        tracing::debug!(
            session_id,
            chunks = existing.len(),
            "observation chunks already exist"
        );
        return Ok(SessionChunkOutcome {
            ids: existing.iter().map(|r| r.node_id).collect(),
            rebuilt: false,
        });
    }

    // Query all observations in this session.
    let rows = kb.session_observation_rows(session_id).await?;

    if rows.is_empty() {
        tracing::debug!(session_id, "no observations in session");
        return drop_stale_chunks(kb, mode, &existing).await;
    }

    // Deduplicate and collect observations.
    let mut seen = HashSet::new();
    let mut text_block = String::new();
    let mut subjects = HashSet::new();

    for row in &rows {
        if row.content.is_empty() {
            continue;
        }

        // Deduplicate by normalized (lowercased, trimmed) content.
        let key = row.content.trim().to_lowercase();
        if !seen.insert(key) {
            continue;
        }

        if !row.subject.is_empty() {
            subjects.insert(row.subject.clone());
        }
        text_block.push_str(row.content.trim());
        text_block.push('\n');
    }

    if text_block.is_empty() {
        return drop_stale_chunks(kb, mode, &existing).await;
    }

    // Chunk the observation text.
    let chunk_cfg = ChunkConfig::from_uniko_config(kb.config());
    let chunker = TextChunker;
    let mut chunks: Vec<ChunkData> = Chunker::chunk(&chunker, &text_block, &chunk_cfg);

    // Set metadata on each chunk.
    let subject_list = {
        let mut sorted: Vec<&str> = subjects.iter().map(String::as_str).collect();
        sorted.sort_unstable();
        sorted.join(", ")
    };

    let topic = kb.get_node(session_nid).await?.and_then(|(_, props)| {
        props
            .get("topic")
            .and_then(|v| v.as_str())
            .map(String::from)
    });

    for chunk in &mut chunks {
        chunk.chunk_type = "observation".to_string();
        chunk.heading = topic.clone();
        chunk.symbol_name = Some(subject_list.clone());
    }

    let plan = match resolve_existing(mode, &existing, &chunks) {
        ChunkPlan::Reuse(ids) => {
            tracing::debug!(
                session_id,
                chunks = ids.len(),
                "observation chunks up to date"
            );
            return Ok(SessionChunkOutcome {
                ids,
                rebuilt: false,
            });
        }
        rebuild => rebuild,
    };
    let obs_ext_id = format!("{session_id}:obs");
    let chunk_nids = apply_plan(kb, plan, &obs_ext_id, session_nid, &chunks).await?;

    // Wire ABOUT edges from observation chunks to entities.
    let entity_nids = kb.session_observation_entity_ids(session_id).await?;

    if !entity_nids.is_empty() {
        let about_edges: Vec<(NodeId, NodeId, HashMap<String, Value>)> = chunk_nids
            .iter()
            .flat_map(|&chunk_nid| {
                entity_nids
                    .iter()
                    .map(move |&entity_nid| (chunk_nid, entity_nid, HashMap::new()))
            })
            .collect();
        // Observation chunks → entities. Source is :Chunk, target is :Entity.
        let about_start = std::time::Instant::now();
        kb.batch_create_edges_fast(
            "ABOUT",
            Some(uniko_store::schema::constants::labels::CHUNK),
            Some(uniko_store::schema::constants::labels::ENTITY),
            &about_edges,
        )
        .await?;
        let ms = about_start.elapsed().as_millis() as u64;
        tracing::info!(
            target: "tx_perf",
            tx_phase = "obs_chunk_about_entity",
            total_ms = ms,
            commit_ms = ms,
            edge_count = about_edges.len() as u64,
            "tx phase",
        );
    }

    // Also propagate Participant ABOUT edges so an obs Chunk is
    // reachable via the participant (e.g. Caroline) — needed for
    // entity-anchored multi-hop recall.
    let participant_nids = kb.session_observation_participant_ids(session_id).await?;
    if !participant_nids.is_empty() {
        let p_edges: Vec<(NodeId, NodeId, HashMap<String, Value>)> = chunk_nids
            .iter()
            .flat_map(|&chunk_nid| {
                participant_nids
                    .iter()
                    .map(move |&p_nid| (chunk_nid, p_nid, HashMap::new()))
            })
            .collect();
        let p_start = std::time::Instant::now();
        kb.batch_create_edges_fast(
            "ABOUT",
            Some(uniko_store::schema::constants::labels::CHUNK),
            Some(uniko_store::schema::constants::labels::PARTICIPANT),
            &p_edges,
        )
        .await?;
        let ms = p_start.elapsed().as_millis() as u64;
        tracing::info!(
            target: "tx_perf",
            tx_phase = "obs_chunk_about_participant",
            total_ms = ms,
            commit_ms = ms,
            edge_count = p_edges.len() as u64,
            "tx phase",
        );
    }

    tracing::info!(
        session_id,
        observations = seen.len(),
        chunks = chunk_nids.len(),
        entities = entity_nids.len(),
        participants = participant_nids.len(),
        "session observations chunked",
    );

    Ok(SessionChunkOutcome {
        ids: chunk_nids,
        rebuilt: true,
    })
}
