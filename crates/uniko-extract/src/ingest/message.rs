//! Message ingest: create the Message node and all associated edges.

use std::collections::HashMap;

use uniko_pipes::types::IngestMessage;
use uniko_store::Value;
use uniko_store::types::datetime_value;
use uniko_store::{KnowledgeBase, NodeId};

use super::chunking::{ChunkConfig, count_tokens, select_chunker};
use super::session::{ensure_participant, get_or_create_session, link_participant_to_session};

/// Inputs that the atomic ingest orchestrator computes in the pre-tx
/// phase and hands to [`apply_message_writes_in_tx`].
#[derive(Debug)]
pub struct MessageSetup {
    pub session_nid: NodeId,
    pub participant_nid: NodeId,
    pub recipient_nids: Vec<NodeId>,
    /// Snapshot of `session_ctx.prev_message_nid` BEFORE this message.
    /// Caller is responsible for updating `session_ctx` after a
    /// successful commit.
    pub prev_msg_nid: Option<NodeId>,
    /// Timestamp of the previous message in the chain, used to compute
    /// `NEXT.gap_ms`. `None` for the first message in a session.
    pub prev_msg_ts: Option<chrono::DateTime<chrono::Utc>>,
}

/// Per-phase write metrics; useful for callers that want to log a
/// granular breakdown.
#[derive(Debug)]
pub struct MessageWriteResult {
    pub message_node_id: NodeId,
    pub chunk_node_ids: Vec<NodeId>,
    pub create_ms: u128,
    pub edges_ms: u128,
    pub chunk_ms: u128,
    pub edge_count: usize,
}

/// Write Message + edges + chunks into the caller's tx.
///
/// All pre-tx setup (idempotency, session/participant ensure, recipient
/// resolution) is the caller's responsibility — supply
/// [`MessageSetup`] with the resolved ids. Does NOT commit.
///
/// # Errors
///
/// Returns [`uniko_store::UnikoError`] on any underlying write failure.
pub(crate) async fn apply_message_writes_in_tx(
    kb: &KnowledgeBase,
    tx: &uniko_store::Transaction,
    msg: &IngestMessage,
    setup: &MessageSetup,
) -> uniko_store::Result<MessageWriteResult> {
    let ts_value = datetime_value(msg.timestamp);

    // Create Message node (triggers auto-embed on content via uni-db's
    // process_embeddings_for_batch, which acquires the BGE-small ORT
    // Session Mutex).
    //
    // Measurement-only knob: UNIKO_BENCH_NO_MSG_EMBED=1 pre-populates
    // a zero embedding to skip auto-embed for A/B testing. NOT a
    // production code path — recall results from such runs are invalid.
    let create_start = std::time::Instant::now();
    let mut props = HashMap::new();
    props.insert("message_id".into(), Value::String(msg.message_id.clone()));
    props.insert("content".into(), Value::String(msg.content.clone()));
    props.insert(
        "content_type".into(),
        Value::String(msg.content_type.clone()),
    );
    props.insert("timestamp".into(), ts_value);
    if std::env::var("UNIKO_BENCH_NO_MSG_EMBED").is_ok() {
        // Placeholder embedding sized to the configured embedder's
        // dimension (was hardcoded 384, which mismatched any non-384 model).
        let dims = kb.config().embedding.dimensions;
        let zero_vec: Vec<Value> = (0..dims).map(|_| Value::Float(0.0)).collect();
        props.insert("embedding".into(), Value::List(zero_vec));
    }
    let message_nid = kb.create_node_in_tx(tx, "Message", &props).await?;
    let create_ms = create_start.elapsed().as_millis();

    // Create all per-message edges in ONE Cypher statement.
    let edges_start = std::time::Instant::now();
    let edge_count = 2 + setup.recipient_nids.len() + usize::from(setup.prev_msg_nid.is_some());
    // SENT_BY.role reflects the sender's kind so "messages from agents"
    // queries work. Callers signal it via `metadata["sender_role"]`
    // (e.g. "human" / "agent" / "service"); absent it, default "user".
    let sender_role = msg
        .metadata
        .get("sender_role")
        .and_then(|v| v.as_str())
        .unwrap_or("user");
    // NEXT.gap_ms = ms between this message and the previous one in the
    // chain. `None` for the first message; clamped to >= 0 to absorb
    // out-of-order timestamps.
    let gap_ms = setup
        .prev_msg_ts
        .map(|prev| (msg.timestamp - prev).num_milliseconds().max(0));
    kb.create_message_edges_in_tx(
        tx,
        message_nid,
        setup.participant_nid,
        sender_role,
        setup.session_nid,
        &setup.recipient_nids,
        setup.prev_msg_nid,
        gap_ms,
    )
    .await?;
    let edges_ms = edges_start.elapsed().as_millis();

    // Chunk long messages — in the same tx.
    let chunk_start = std::time::Instant::now();
    let chunk_threshold = kb.config().message_chunk_threshold;
    let chunk_node_ids = if count_tokens(&msg.content) > chunk_threshold {
        let chunk_cfg = ChunkConfig::from_uniko_config(kb.config());
        let chunker = select_chunker(&msg.content_type, None);
        let chunks = chunker.chunk(&msg.content, &chunk_cfg);
        create_chunks_in_tx(kb, tx, &msg.message_id, message_nid, &chunks, "Message").await?
    } else {
        Vec::new()
    };
    let chunk_ms = chunk_start.elapsed().as_millis();

    Ok(MessageWriteResult {
        message_node_id: message_nid,
        chunk_node_ids,
        create_ms,
        edges_ms,
        chunk_ms,
        edge_count,
    })
}

/// First-sight setup for Session + sender Participant (own commits today).
/// Uses `SessionContext` caches to skip on repeat invocations.
pub(crate) async fn ensure_session_and_sender(
    kb: &KnowledgeBase,
    msg: &IngestMessage,
    session_ctx: &mut super::context::SessionContext,
) -> uniko_store::Result<(NodeId, NodeId)> {
    let need_session = session_ctx.session_nid == 0;
    let need_participant = session_ctx
        .participant_nid(&msg.sender_id)
        .is_none_or(|nid| nid == 0);

    // Cold-path serialization: hold the per-session/-participant setup
    // locks across the check-then-create of these shared rows so
    // concurrent ingests of the same session (each with an independent
    // `SessionContext`, hence none cached) don't race into a duplicate
    // Session/Participant/`PARTICIPATED_IN` or an SSI read-write
    // antidependency abort. The warm path (both ids cached) skips the
    // lock and all DB work. Guards drop at function exit, after every
    // get-or-create commit below.
    let _setup_guards = if need_session || need_participant {
        Some(kb.lock_session_setup(&msg.session_id, &msg.sender_id).await)
    } else {
        None
    };

    let session_nid = if session_ctx.session_nid != 0 {
        session_ctx.session_nid
    } else {
        let nid = get_or_create_session(kb, &msg.session_id, &msg.timestamp).await?;
        session_ctx.session_nid = nid;
        nid
    };
    let participant_nid = if let Some(nid) = session_ctx.participant_nid(&msg.sender_id) {
        if nid != 0 {
            nid
        } else {
            let nid = ensure_participant(kb, &msg.sender_id, msg.timestamp).await?;
            link_participant_to_session(kb, nid, session_nid).await?;
            session_ctx.register_participant(&msg.sender_id, nid);
            nid
        }
    } else {
        let nid = ensure_participant(kb, &msg.sender_id, msg.timestamp).await?;
        link_participant_to_session(kb, nid, session_nid).await?;
        session_ctx.register_participant(&msg.sender_id, nid);
        nid
    };
    Ok((session_nid, participant_nid))
}

/// Resolve ADDRESSED_TO recipient node IDs.
///
/// If `msg.addressed_to` is provided, ensures each participant exists
/// and returns their node IDs. Otherwise infers recipients from the
/// in-memory [`SessionContext::participants`] cache — every participant
/// who has previously sent a message in this session was registered
/// there during their first message's ingest.
///
/// The cache and the on-disk PARTICIPATED_IN-edge set are populated by
/// the same code path (only on sender first-sight in atomic ingest),
/// so the cache is exactly equivalent to the previous
/// `get_edges(... PARTICIPATED_IN ...)` query — without the per-message
/// DB read that was costing ~30% of `edges_ms` at sess=24.
pub(crate) async fn resolve_recipients(
    kb: &KnowledgeBase,
    msg: &IngestMessage,
    sender_nid: NodeId,
    session_ctx: &super::context::SessionContext,
) -> uniko_store::Result<Vec<NodeId>> {
    if let Some(ref ids) = msg.addressed_to {
        let mut nids = Vec::with_capacity(ids.len());
        for pid in ids {
            // Hit the cache first; only `ensure_participant` (a
            // merge_node DB roundtrip) on miss.
            let nid = match session_ctx.participant_nid(pid) {
                Some(nid) if nid != 0 => nid,
                _ => super::session::ensure_participant(kb, pid, msg.timestamp).await?,
            };
            nids.push(nid);
        }
        Ok(nids)
    } else {
        Ok(session_ctx
            .participants
            .values()
            .copied()
            .filter(|&nid| nid != sender_nid && nid != 0)
            .collect())
    }
}

/// Create Chunk nodes + HAS_CHUNK edges for a parent node.
///
/// # Errors
///
/// Returns a storage error if either the node create or edge create
/// batch fails.
pub async fn create_chunks(
    kb: &KnowledgeBase,
    parent_ext_id: &str,
    parent_nid: NodeId,
    chunks: &[super::chunking::ChunkData],
    parent_label: &str,
) -> uniko_store::Result<Vec<NodeId>> {
    if chunks.is_empty() {
        return Ok(Vec::new());
    }
    let start = std::time::Instant::now();
    // Retry on retriable SSI conflicts (a conflict aborts the whole tx, so
    // a fresh attempt recreates the same deterministic chunk_ids — no
    // duplicates). The wrapper commits internally, so the separate
    // commit_ms is no longer measurable here; total_ms covers the retried
    // op (matching every other `transact_with_retry` site).
    let nids = kb
        .transact_with_retry(uniko_store::RetryOptions::default(), |tx| async {
            let r =
                create_chunks_in_tx(kb, &tx, parent_ext_id, parent_nid, chunks, parent_label).await;
            (tx, r)
        })
        .await?;
    let total_ms = start.elapsed().as_millis() as u64;
    tracing::info!(
        target: "tx_perf",
        tx_phase = match parent_label {
            "Session" => "session_chunks",
            "Message" => "message_chunks_standalone",
            _ => "chunks_standalone",
        },
        total_ms,
        chunk_count = chunks.len() as u64,
        "tx phase",
    );
    Ok(nids)
}

/// Same as [`create_chunks`] but defers commit to the caller's tx.
///
/// Used by [`apply_message_writes_in_tx`] to fold Chunk creation under
/// the same transaction as the Message node and its edges.
///
/// # Errors
///
/// Returns a storage error if either batched write fails.
pub async fn create_chunks_in_tx(
    kb: &KnowledgeBase,
    tx: &uniko_store::Transaction,
    parent_ext_id: &str,
    parent_nid: NodeId,
    chunks: &[super::chunking::ChunkData],
    parent_label: &str,
) -> uniko_store::Result<Vec<NodeId>> {
    if chunks.is_empty() {
        return Ok(Vec::new());
    }

    let chunk_props: Vec<HashMap<String, Value>> = chunks
        .iter()
        .map(|c| {
            let cid = uniko_store::id::chunk_id(parent_ext_id, c.index);
            let mut props = HashMap::new();
            props.insert("chunk_id".into(), Value::String(cid));
            props.insert("text".into(), Value::String(c.text.clone()));
            props.insert("index".into(), Value::Int(c.index as i64));
            props.insert("start".into(), Value::Int(c.start as i64));
            props.insert("end".into(), Value::Int(c.end as i64));
            props.insert("token_count".into(), Value::Int(c.token_count as i64));
            props.insert("chunk_type".into(), Value::String(c.chunk_type.clone()));
            if let Some(ref lang) = c.language {
                props.insert("language".into(), Value::String(lang.clone()));
            }
            if let Some(ref sym) = c.symbol_name {
                props.insert("symbol_name".into(), Value::String(sym.clone()));
            }
            if let Some(ref h) = c.heading {
                props.insert("heading".into(), Value::String(h.clone()));
            }
            if let Some(ref meta) = c.metadata {
                props.insert("metadata".into(), json_to_uni_value(meta));
            }
            props
        })
        .collect();

    let chunk_nids = kb
        .batch_create_nodes_in_tx(tx, "Chunk", &chunk_props)
        .await?;

    // Use the chunk's own index, not the position in this batch: a partial
    // rebuild creates only a suffix, and the edge must still record the
    // chunk's absolute position under the parent.
    let edges: Vec<(NodeId, NodeId, HashMap<String, Value>)> = chunk_nids
        .iter()
        .enumerate()
        .map(|(i, &cid)| {
            let mut props = HashMap::new();
            props.insert("index".into(), Value::Int(chunks[i].index as i64));
            (parent_nid, cid, props)
        })
        .collect();

    // Source label is dynamic (Message, Session, or Artifact); target is :Chunk.
    kb.batch_create_edges_fast_in_tx(
        tx,
        "HAS_CHUNK",
        Some(parent_label),
        Some(uniko_store::schema::constants::labels::CHUNK),
        &edges,
    )
    .await?;

    Ok(chunk_nids)
}

/// Convert a `serde_json::Value` into a `uniko_store::Value` suitable for
/// storage in a `DataType::CypherValue` column.
///
/// JSON numbers split across `Value::Int` (when they fit in `i64` and
/// have no fractional part) or `Value::Float` otherwise. Nulls become
/// `Value::Null`; arrays / objects recurse.
pub(crate) fn json_to_uni_value(v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float(f)
            } else {
                // u64 > i64::MAX — fall back to string to avoid lossy cast.
                Value::String(n.to_string())
            }
        }
        serde_json::Value::String(s) => Value::String(s.clone()),
        serde_json::Value::Array(arr) => Value::List(arr.iter().map(json_to_uni_value).collect()),
        serde_json::Value::Object(obj) => {
            let mut m: HashMap<String, Value> = HashMap::with_capacity(obj.len());
            for (k, v) in obj {
                m.insert(k.clone(), json_to_uni_value(v));
            }
            Value::Map(m)
        }
    }
}
