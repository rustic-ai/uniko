//! Artifact ingest: identity, content dedup, Artifact node, chunk, link.

use std::collections::HashMap;

use sha2::{Digest, Sha256};
use uniko_store::Value;

use uniko_pipes::types::IngestArtifact;
use uniko_store::storage::blob::MergeContent;
use uniko_store::{KnowledgeBase, NodeId};

use super::chunking::{ChunkConfig, select_chunker};
use super::message::create_chunks_in_tx;

/// Result of ingesting a single artifact.
#[derive(Debug)]
pub struct ArtifactIngestResult {
    /// Internal node ID of the Artifact.
    pub artifact_node_id: NodeId,
    /// External artifact id — pass to `agent.data().artifact(..)` to fetch
    /// it back.
    pub artifact_id: String,
    /// Node IDs of the created chunks.
    pub chunk_node_ids: Vec<NodeId>,
    /// Whether this artifact was a duplicate (skipped chunking).
    pub was_deduplicated: bool,
}

/// Node ids the caller has already resolved for an artifact's provenance
/// edges.
///
/// `Some` short-circuits the lookup; `None` falls back to a best-effort
/// in-transaction lookup by external id, preserving the "an unresolved
/// reference is logged and skipped" contract.
///
/// A multi-turn unit MUST populate `message_nid` from the Message it
/// created in the same transaction. Leaving it `None` still works — the
/// fallback is an in-tx read, which does see the unit's own writes — but
/// resolving it *before* the transaction would not: an uncommitted Message
/// is invisible to a committed read, the miss is silent by design, and the
/// `ATTACHED_TO` edge would vanish with no error anywhere.
#[derive(Debug, Default, Clone, Copy)]
pub struct ArtifactContextNids {
    /// `:Session` this artifact is attached to.
    pub session_nid: Option<NodeId>,
    /// `:Message` this artifact was attached to.
    pub message_nid: Option<NodeId>,
    /// `:Action` that produced this artifact.
    pub action_nid: Option<NodeId>,
}

/// How an artifact is identified within one unit.
///
/// Mirrors the identity split in [`ingest_artifact`]: a caller-supplied id
/// IS the identity (two ids over identical bytes are two artifacts sharing
/// one `:ArtifactContent`); an auto-generated id expresses no identity, so
/// identical bytes collapse. A hash-only memo would wrongly merge two
/// deliberately-distinct named artifacts.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ArtifactIdentity {
    /// Identified by a caller-supplied `artifact_id`.
    Named(String),
    /// Identified by content hash, because the id was auto-generated.
    Content(String),
}

/// Per-unit memo of artifacts already written in this transaction.
///
/// Two turns of one unit that attach identical bytes both reach the
/// idempotency probe. The in-tx probe does see the unit's earlier writes,
/// but the memo avoids re-reading and — more importantly — keeps the
/// `:ArtifactContent` merge to one call per content id. Threaded `&mut`
/// through the whole unit body and constructed **fresh per transaction
/// attempt**: a rolled-back attempt's node ids are stale.
#[derive(Debug, Default)]
pub struct UnitArtifactSeen {
    artifacts: HashMap<ArtifactIdentity, (NodeId, String)>,
    content: HashMap<String, NodeId>,
}

impl UnitArtifactSeen {
    /// A memo for a fresh transaction attempt.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The `(node_id, artifact_id)` this unit already wrote for `identity`,
    /// if any. The returned id is the one the artifact actually *lives*
    /// under, which for a content-deduped hit is the first writer's id, not
    /// the throwaway one a later call minted.
    #[must_use]
    pub fn get(&self, identity: &ArtifactIdentity) -> Option<(NodeId, String)> {
        self.artifacts.get(identity).cloned()
    }

    /// Record an artifact this unit wrote (or found) under `identity`.
    pub fn insert(&mut self, identity: ArtifactIdentity, nid: NodeId, artifact_id: String) {
        self.artifacts.insert(identity, (nid, artifact_id));
    }

    /// The `:ArtifactContent` node this unit already merged for `hash`.
    #[must_use]
    pub fn content_nid(&self, hash: &str) -> Option<NodeId> {
        self.content.get(hash).copied()
    }

    /// Record the `:ArtifactContent` node merged for `hash`.
    pub fn insert_content(&mut self, hash: String, nid: NodeId) {
        self.content.insert(hash, nid);
    }
}

/// Everything about one artifact that is computable before the unit's
/// transaction opens: the content hash, the blob PUT outcome, the detected
/// language and MIME, and the chunked text.
///
/// Built by [`prepare_artifact`] and consumed **by reference** by
/// [`ingest_artifact_in_tx`] — the blob PUT and the chunking are the
/// expensive parts and must never be redone on a transaction retry.
#[derive(Debug, Clone)]
pub struct ArtifactPrep {
    /// External id this artifact will be created under.
    pub artifact_id: String,
    /// Whether `artifact_id` came from the caller (identity) or was minted.
    pub caller_supplied_id: bool,
    /// SHA-256 of the content, hex-encoded. The `:ArtifactContent` key.
    pub hash: String,
    /// Content length in bytes.
    pub size: i64,
    /// Artifact kind, as supplied.
    pub kind: String,
    /// Source path, when known.
    pub path: Option<String>,
    /// Detected programming language, when the path implies one.
    pub language: Option<String>,
    /// Resolved MIME type.
    pub mime: String,
    /// Outcome of the blob backend PUT. Already persisted.
    pub put: uniko_store::blob_store::PutOutcome,
    /// Chunked content, ready for `create_chunks_in_tx`.
    pub chunks: Vec<super::chunking::ChunkData>,
    /// Optional provenance, resolved in-tx when the caller passes no nid.
    pub session_id: Option<String>,
    /// Optional provenance, resolved in-tx when the caller passes no nid.
    pub triggered_by_message_id: Option<String>,
    /// Optional provenance, resolved in-tx when the caller passes no nid.
    pub produced_by_action_id: Option<String>,
    /// Caller's record category (issue #39), written to `Artifact.category`
    /// and inherited by this artifact's chunks.
    pub category: Option<String>,
    /// Logical source id (issue #39), materialised as a `:Source` with a
    /// `FROM_SOURCE` edge and denormalised onto the artifact and its chunks.
    pub source_id: Option<String>,
    /// Revision identity for these bytes (issue #41).
    pub revision_id: Option<String>,
}

impl ArtifactPrep {
    /// How this artifact is identified within a unit.
    #[must_use]
    fn identity(&self) -> ArtifactIdentity {
        if self.caller_supplied_id {
            ArtifactIdentity::Named(self.artifact_id.clone())
        } else {
            ArtifactIdentity::Content(self.hash.clone())
        }
    }
}

/// Pre-transaction phase of artifact ingest: hash, blob PUT, chunk.
///
/// Performs no graph writes and holds no locks. [`ArtifactPrep::hash`] is
/// the key the caller must pass to
/// [`KnowledgeBase::lock_ingest_unit`](uniko_store::KnowledgeBase::lock_ingest_unit)
/// — or [`lock_content_ids`](uniko_store::KnowledgeBase::lock_content_ids)
/// for a lone artifact — before opening the transaction.
///
/// The blob PUT runs here deliberately. For the Lance backend it is a pure
/// passthrough whose bytes ride inside the graph write, so ingest is fully
/// transactional; for `Fs`/`S3` the object is persisted before this
/// returns, and running it inside the transaction would not make it
/// transactional, only lengthen the window. Its output is an input to the
/// `:ArtifactContent` create either way. An aborted unit therefore leaves a
/// content-addressed object that the next ingest of the same bytes reuses.
///
/// # Errors
///
/// Returns [`UnikoError::Storage`](uniko_store::UnikoError::Storage) if the
/// blob backend PUT fails.
pub async fn prepare_artifact(
    kb: &KnowledgeBase,
    artifact: &IngestArtifact,
) -> uniko_store::Result<ArtifactPrep> {
    let hash = hex::encode(Sha256::digest(artifact.content.as_bytes()));
    let language = detect_language(artifact.path.as_deref());

    let bytes = artifact.content.as_bytes();
    let size = bytes.len() as i64;
    let put = kb.put_blob(&hash, bytes).await?;

    // A caller-supplied `metadata["content_type"]` wins (e.g. a fetched URL
    // whose bytes are HTML), so URL / upload ingestion uses the right
    // chunker even without a file extension; otherwise infer from the
    // detected language.
    let hinted_content_type = artifact
        .metadata
        .get("content_type")
        .and_then(|v| v.as_str());
    let content_type =
        hinted_content_type.unwrap_or(if language.is_some() { "code" } else { "text" });
    let chunker = select_chunker(content_type, language.as_deref());
    let chunk_cfg = ChunkConfig::from_uniko_config(kb.config());
    let chunks = chunker.chunk(&artifact.content, &chunk_cfg);

    Ok(ArtifactPrep {
        artifact_id: artifact.artifact_id.clone(),
        caller_supplied_id: artifact.caller_supplied_id,
        hash,
        size,
        kind: artifact.kind.clone(),
        path: artifact.path.clone(),
        mime: uniko_pipes::content::mime_for_kind(&artifact.kind, language.as_deref()).to_string(),
        language,
        put,
        chunks,
        session_id: artifact.session_id.clone(),
        triggered_by_message_id: artifact.triggered_by_message_id.clone(),
        produced_by_action_id: artifact.produced_by_action_id.clone(),
        category: artifact.category.clone(),
        source_id: artifact.source_id.clone(),
        revision_id: artifact.revision_id.clone(),
    })
}

/// Ingest a prepared artifact into the caller's open transaction.
///
/// The in-transaction half of [`ingest_artifact`]: identity probe,
/// `:ArtifactContent` merge, `:Artifact` node, `HAS_CONTENT`, provenance
/// edges, and chunks — all deferred to `tx`'s commit.
///
/// # Preconditions
///
/// The caller holds the `content:<prep.hash>` RMW guard, acquired before
/// `tx` was opened and held until after `tx.commit()`.
///
/// # Post-commit
///
/// `Artifact.text_embedding` is **not** populated here — pooling reads
/// committed Chunk embeddings through a fresh session. Run
/// [`pool_artifact_embeddings_post_commit`] after the commit, or accept
/// NULL until the backfill migration runs.
///
/// # Errors
///
/// Returns [`UnikoError::IdConflict`](uniko_store::UnikoError::IdConflict)
/// when a caller-supplied id is reused for different content, or a storage
/// error on any graph write.
pub async fn ingest_artifact_in_tx(
    kb: &KnowledgeBase,
    tx: &uniko_store::Transaction,
    prep: &ArtifactPrep,
    ctx: ArtifactContextNids,
    seen: &mut UnitArtifactSeen,
) -> uniko_store::Result<ArtifactIngestResult> {
    // 1. Unit memo, before any DB read.
    if let Some((nid, under_id)) = seen.get(&prep.identity()) {
        if prep.caller_supplied_id && under_id != prep.hash {
            return Err(uniko_store::UnikoError::id_conflict(
                "Artifact",
                "artifact_id",
                &prep.artifact_id,
            ));
        }
        link_artifact_context_in_tx(kb, tx, nid, prep, ctx).await?;
        return Ok(ArtifactIngestResult {
            artifact_node_id: nid,
            artifact_id: if prep.caller_supplied_id {
                prep.artifact_id.clone()
            } else {
                under_id
            },
            chunk_node_ids: Vec::new(),
            was_deduplicated: true,
        });
    }

    // 2. Authoritative probe, INSIDE the transaction so it sees both
    //    committed state and anything this unit already wrote.
    if prep.caller_supplied_id {
        if let Some((existing_id, existing_props)) = kb
            .get_node_by_ext_id_in_tx(tx, "Artifact", "artifact_id", &prep.artifact_id)
            .await?
        {
            let stored = match existing_props.get("hash") {
                Some(Value::String(s)) => s.as_str(),
                _ => "",
            };
            if stored != prep.hash {
                return Err(uniko_store::UnikoError::id_conflict(
                    "Artifact",
                    "artifact_id",
                    &prep.artifact_id,
                ));
            }
            seen.artifacts
                .insert(prep.identity(), (existing_id, prep.artifact_id.clone()));
            link_artifact_context_in_tx(kb, tx, existing_id, prep, ctx).await?;
            return Ok(ArtifactIngestResult {
                artifact_node_id: existing_id,
                artifact_id: prep.artifact_id.clone(),
                chunk_node_ids: Vec::new(),
                was_deduplicated: true,
            });
        }
    } else if let Some((existing_id, existing_props)) = kb
        .get_node_by_ext_id_in_tx(tx, "Artifact", "hash", &prep.hash)
        .await?
    {
        // Report the id the artifact actually lives under, not the
        // throwaway UUID this call minted — echoing back an id that
        // resolves to nothing is what made this unfetchable before.
        let existing_ext_id = match existing_props.get("artifact_id") {
            Some(Value::String(s)) => s.clone(),
            _ => prep.artifact_id.clone(),
        };
        seen.insert(prep.identity(), existing_id, existing_ext_id.clone());
        link_artifact_context_in_tx(kb, tx, existing_id, prep, ctx).await?;
        return Ok(ArtifactIngestResult {
            artifact_node_id: existing_id,
            artifact_id: existing_ext_id,
            chunk_node_ids: Vec::new(),
            was_deduplicated: true,
        });
    }

    // 2b. Revision identity (issue #41). A revision id is a promise about
    //     the content, so the same revision with different bytes is a caller
    //     error rather than a silent update. Checked in-tx so it also sees a
    //     revision this same unit just wrote.
    if let Some(ref revision_id) = prep.revision_id
        && let Some((_, existing_props)) = kb
            .get_node_by_ext_id_in_tx(tx, "Artifact", "revision_id", revision_id)
            .await?
    {
        let stored = match existing_props.get("hash") {
            Some(Value::String(h)) => h.as_str(),
            _ => "",
        };
        if stored != prep.hash {
            return Err(uniko_store::UnikoError::id_conflict(
                "Artifact",
                "revision_id",
                revision_id,
            ));
        }
    }

    // 3. MERGE :ArtifactContent. Memoized per unit so two turns sharing
    //    bytes under DIFFERENT caller ids still converge on one content
    //    row — the case that keeps dedup on `:ArtifactContent`.
    let content_nid = match seen.content_nid(&prep.hash) {
        Some(nid) => nid,
        None => {
            let nid = kb
                .merge_artifact_content_in_tx(
                    tx,
                    MergeContent {
                        content_id: prep.hash.clone(),
                        bytes: prep.put.bytes_inline.clone(),
                        uri: prep.put.uri.clone(),
                        mime: prep.mime.clone(),
                        size: prep.size,
                        perceptual_hash: None,
                        audio_fingerprint: None,
                    },
                )
                .await?;
            seen.insert_content(prep.hash.clone(), nid);
            nid
        }
    };

    // 4. :Artifact metadata node (no `content` — that lives on
    //    :ArtifactContent). `hash` stays a denorm cache of
    //    `HAS_CONTENT.target.content_id`.
    let mut props = HashMap::new();
    props.insert(
        "artifact_id".into(),
        Value::String(prep.artifact_id.clone()),
    );
    props.insert("kind".into(), Value::String(prep.kind.clone()));
    if let Some(ref path) = prep.path {
        props.insert("path".into(), Value::String(path.clone()));
    }
    props.insert("hash".into(), Value::String(prep.hash.clone()));
    props.insert("size".into(), Value::Int(prep.size));
    if let Some(ref lang) = prep.language {
        props.insert("language".into(), Value::String(lang.clone()));
    }
    // Typed provenance (issue #39), denormalised onto the artifact so a
    // recall scope filters with a property predicate rather than a
    // traversal.
    if let Some(ref category) = prep.category {
        props.insert("category".into(), Value::String(category.clone()));
    }
    if let Some(ref source_id) = prep.source_id {
        props.insert("source_id".into(), Value::String(source_id.clone()));
    }
    if let Some(ref revision_id) = prep.revision_id {
        props.insert("revision_id".into(), Value::String(revision_id.clone()));
    }
    let artifact_nid = kb.create_node_in_tx(tx, "Artifact", &props).await?;
    seen.insert(prep.identity(), artifact_nid, prep.artifact_id.clone());

    // 5. HAS_CONTENT edge: Artifact → ArtifactContent, role="primary".
    let mut edge_props = HashMap::new();
    edge_props.insert("role".into(), Value::String("primary".into()));
    kb.create_edges_in_tx(
        tx,
        &[("HAS_CONTENT", artifact_nid, content_nid, edge_props)],
    )
    .await?;

    // 5b. Logical source: the :Source row is the normalised truth that
    //     outlives any one revision of these bytes (issue #41 builds on it);
    //     `Artifact.source_id` above is the denormalised filter path.
    if let Some(ref source_id) = prep.source_id {
        let source_nid = kb
            .merge_source_in_tx(tx, source_id, None, prep.path.as_deref())
            .await?;
        kb.create_edges_in_tx(
            tx,
            &[(
                uniko_store::schema::edges::FROM_SOURCE,
                artifact_nid,
                source_nid,
                HashMap::new(),
            )],
        )
        .await?;
    }

    // 5c. A newer revision replaces the previous current one for this
    //     source: stamp the old artifact and record the ordered history.
    //     Nothing is deleted, so the superseded revision stays attributable
    //     to the answers it grounded (issue #41).
    if let (Some(source_id), Some(revision_id)) = (&prep.source_id, &prep.revision_id) {
        kb.supersede_prior_revisions_in_tx(tx, source_id, revision_id, artifact_nid)
            .await?;
    }

    // 6. Contextual provenance (F18/F22/F30).
    link_artifact_context_in_tx(kb, tx, artifact_nid, prep, ctx).await?;

    // 7. Chunks.
    let chunk_nids = create_chunks_in_tx(
        kb,
        tx,
        &prep.artifact_id,
        artifact_nid,
        &prep.chunks,
        "Artifact",
        super::message::ChunkProvenance {
            category: prep.category.as_deref(),
            source_id: prep.source_id.as_deref(),
            revision_id: prep.revision_id.as_deref(),
        },
    )
    .await?;

    Ok(ArtifactIngestResult {
        artifact_node_id: artifact_nid,
        artifact_id: prep.artifact_id.clone(),
        chunk_node_ids: chunk_nids,
        was_deduplicated: false,
    })
}

/// Best-effort post-commit mean-pool of every artifact a unit created.
///
/// Pooling reads child `Chunk.embedding` values through a fresh session and
/// writes via a self-committing `update_node`, so it can only run after the
/// unit commits. Failures are logged at warn and left to the
/// `backfill_artifact_text_embedding` migration.
pub async fn pool_artifact_embeddings_post_commit(kb: &KnowledgeBase, artifact_nids: &[NodeId]) {
    for &nid in artifact_nids {
        if let Err(e) = kb.mean_pool_artifact_text_embedding(nid).await {
            tracing::warn!(
                target: "uniko_extract::ingest",
                error = %e,
                artifact_nid = nid,
                "mean_pool_artifact_text_embedding failed; leaving NULL for backfill"
            );
        }
    }
}

/// Ingest an artifact into the knowledge graph.
///
/// Identity depends on whether the caller named the artifact. With a
/// caller-supplied `artifact_id`, that id is the identity: identical bytes
/// under it are an idempotent replay, different bytes under it are an
/// [`UnikoError::IdConflict`](uniko_store::UnikoError::IdConflict), and a
/// second id over the same bytes is a second artifact. With an
/// auto-generated id, identical bytes dedup onto the existing artifact.
///
/// Bytes are stored once either way — the blob PUT and the
/// `:ArtifactContent` merge are keyed on the SHA-256, so deduplication
/// lives on `:ArtifactContent`. A dedup hit still wires this call's
/// session/message provenance before returning.
///
/// Every graph write lands in ONE transaction, so a failure part-way
/// through can no longer leave a committed Artifact with no chunks.
///
/// # Errors
///
/// Returns [`UnikoError::IdConflict`](uniko_store::UnikoError::IdConflict)
/// when a caller-supplied id is reused for different content, or a storage
/// error if any graph operation fails.
pub async fn ingest_artifact(
    kb: &KnowledgeBase,
    artifact: &IngestArtifact,
) -> uniko_store::Result<ArtifactIngestResult> {
    let prep = prepare_artifact(kb, artifact).await?;
    let _guards = kb.lock_content_ids(std::slice::from_ref(&prep.hash)).await;

    let result = kb
        .transact_with_retry(uniko_store::RetryOptions::default(), |tx| {
            let prep = &prep;
            async move {
                // Fresh per attempt: a rolled-back attempt's nids are stale.
                let mut seen = UnitArtifactSeen::new();
                let r =
                    ingest_artifact_in_tx(kb, &tx, prep, ArtifactContextNids::default(), &mut seen)
                        .await;
                (tx, r)
            }
        })
        .await?;

    if !result.chunk_node_ids.is_empty() {
        pool_artifact_embeddings_post_commit(kb, &[result.artifact_node_id]).await;
    }
    Ok(result)
}

/// Wire an artifact's optional conversational-context edges, inside `tx`.
///
/// `PRODUCED` (Action → Artifact), `ATTACHED_TO` (Artifact → Session) and
/// `ATTACHED_TO` (Artifact → Message). Each is best-effort: an unresolved
/// reference is logged at debug and skipped, so a stale id never aborts
/// ingestion.
///
/// A node id supplied through `ctx` short-circuits the lookup. Any fallback
/// lookup runs **in the transaction**, so it can see rows the same unit
/// created — resolving these before the transaction would silently drop the
/// edge, because a miss is not an error.
async fn link_artifact_context_in_tx(
    kb: &KnowledgeBase,
    tx: &uniko_store::Transaction,
    artifact_nid: NodeId,
    prep: &ArtifactPrep,
    ctx: ArtifactContextNids,
) -> uniko_store::Result<()> {
    use uniko_store::schema::{edges, labels};

    let attach_props = || {
        let mut p = HashMap::new();
        p.insert(
            "attached_at".into(),
            uniko_store::types::datetime_value(chrono::Utc::now()),
        );
        p
    };

    /// Resolve a provenance target: caller-supplied nid wins, else an
    /// in-tx lookup, else `None` with a debug line.
    async fn resolve(
        kb: &KnowledgeBase,
        tx: &uniko_store::Transaction,
        given: Option<NodeId>,
        label: &str,
        id_field: &str,
        ext_id: Option<&str>,
        what: &str,
    ) -> uniko_store::Result<Option<NodeId>> {
        if given.is_some() {
            return Ok(given);
        }
        let Some(ext_id) = ext_id else {
            return Ok(None);
        };
        match kb
            .get_node_by_ext_id_in_tx(tx, label, id_field, ext_id)
            .await?
        {
            Some((nid, _)) => Ok(Some(nid)),
            None => {
                tracing::debug!(ext_id, "ingest_artifact: {what} not found — edge skipped");
                Ok(None)
            }
        }
    }

    let mut pending: Vec<(&str, NodeId, NodeId, HashMap<String, Value>)> = Vec::new();

    // PRODUCED: Action → Artifact (note the reversed direction — the
    // producing Action is the edge source, matching record_action).
    if let Some(nid) = resolve(
        kb,
        tx,
        ctx.action_nid,
        labels::ACTION,
        "action_id",
        prep.produced_by_action_id.as_deref(),
        "producing Action",
    )
    .await?
    {
        pending.push((edges::PRODUCED, nid, artifact_nid, HashMap::new()));
    }

    // ATTACHED_TO: Artifact → Session.
    if let Some(nid) = resolve(
        kb,
        tx,
        ctx.session_nid,
        labels::SESSION,
        "session_id",
        prep.session_id.as_deref(),
        "Session",
    )
    .await?
    {
        pending.push((edges::ATTACHED_TO, artifact_nid, nid, attach_props()));
    }

    // ATTACHED_TO: Artifact → Message.
    if let Some(nid) = resolve(
        kb,
        tx,
        ctx.message_nid,
        labels::MESSAGE,
        "message_id",
        prep.triggered_by_message_id.as_deref(),
        "triggering Message",
    )
    .await?
    {
        pending.push((edges::ATTACHED_TO, artifact_nid, nid, attach_props()));
    }

    if !pending.is_empty() {
        kb.create_edges_in_tx(tx, &pending).await?;
    }
    Ok(())
}

/// Detect programming language from a file path extension.
fn detect_language(path: Option<&str>) -> Option<String> {
    let path = path?;
    let ext = path.rsplit('.').next()?;
    match ext {
        "py" => Some("python".into()),
        "rs" => Some("rust".into()),
        "js" | "mjs" | "cjs" => Some("javascript".into()),
        "ts" => Some("typescript".into()),
        "tsx" => Some("tsx".into()),
        "html" | "htm" => Some("html".into()),
        "css" => Some("css".into()),
        "json" => Some("json".into()),
        "csv" => Some("csv".into()),
        "md" | "markdown" => Some("markdown".into()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_language() {
        assert_eq!(detect_language(Some("main.py")), Some("python".into()));
        assert_eq!(detect_language(Some("lib.rs")), Some("rust".into()));
        assert_eq!(detect_language(Some("app.tsx")), Some("tsx".into()));
        assert_eq!(detect_language(Some("data.csv")), Some("csv".into()));
        assert_eq!(detect_language(Some("unknown.xyz")), None);
        assert_eq!(detect_language(None), None);
    }

    #[test]
    fn test_hash_consistency() {
        let hash1 = hex::encode(Sha256::digest(b"hello world"));
        let hash2 = hex::encode(Sha256::digest(b"hello world"));
        assert_eq!(hash1, hash2);
    }
}
