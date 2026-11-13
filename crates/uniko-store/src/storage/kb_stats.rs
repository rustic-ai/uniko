//! `:KnowledgeBaseStats` singleton host helpers.
//!
//! The KB-level metadata singleton carries:
//!
//! - `blob_storage` — the persisted blob-backend config, written on
//!   first open. Reopening a KB with a different backend variant is a
//!   hard error (no implicit migration).
//! - `embedding` — the embedding model id and dimensions the stored
//!   vectors were made with. Reopening with a different model is a hard
//!   error: same-dimension models would otherwise mix incompatible
//!   vectors silently. Stores created before this was recorded get it
//!   written on their next open.
//! - `modality_presence` — a flag map (`{"image": true, ...}`) updated
//!   incrementally by ingest paths; consumed by the recall cascade to
//!   skip cross-modal channels in text-only corpora. Phase 3 reads it;
//!   Track B writes flip the bits when binary ingest lands.
//
use std::collections::HashMap;

use chrono::Utc;
use uni_db::{Row, Transaction, Value};

use crate::blob_store::BlobStorage;
use crate::error::{Result, UnikoError};
use crate::storage::KnowledgeBase;
use crate::types::datetime_value;

/// Singleton primary key. Every KB has exactly one stats row.
const STATS_ID: &str = "singleton";

/// All modality keys persisted in the `modality_presence` map.
///
/// Centralised so writers (`bump_modality_presence`) and the initial
/// `empty_modality_presence` map agree on the key set.
const MODALITIES: &[&str] = &["image", "audio", "video", "multimodal"];

/// Stub view of the singleton row consumed by recall and ingest paths.
///
/// Carries only the bits Phase 1 + Phase 3 need; can be extended without
/// breaking callers (all fields are public for trivial construction in
/// tests).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModalityPresence {
    /// Any `:Artifact` carries non-NULL `image_embedding`.
    pub has_image_content: bool,
    /// Any `:Artifact` carries non-NULL `audio_embedding`.
    pub has_audio_content: bool,
    /// Any `:Artifact` carries non-NULL `video_embedding`.
    pub has_video_content: bool,
    /// Any `:Artifact` carries non-NULL `multimodal_embedding`.
    pub has_multimodal_indexed: bool,
}

impl KnowledgeBase {
    /// Ensure the `:KnowledgeBaseStats` singleton exists and matches
    /// `self.config.blob_storage` and `self.config.embedding`. Called
    /// from KB open/init.
    ///
    /// On first open: writes a fresh singleton with the configured
    /// backend serialized into the `blob_storage` map.
    ///
    /// On reopen: reads the persisted `blob_storage.kind` and embedding
    /// identity. If either disagrees with the current config, returns
    /// [`UnikoError::Config`] — there is no implicit migration. A row
    /// with no embedding identity (a store from before it was recorded)
    /// gets the current one written.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] on backend or embedding mismatch or
    /// [`UnikoError::Storage`] on database failure.
    pub async fn init_kb_stats(&self) -> Result<()> {
        let session = self.db.session();
        let result = session
            .query_with(
                "MATCH (s:KnowledgeBaseStats {stats_id: $sid}) \
                 RETURN s.blob_storage AS blob_storage, s.embedding AS embedding",
            )
            .param("sid", Value::String(STATS_ID.to_string()))
            .fetch_all()
            .await?;

        let row = result.rows().first();
        let persisted_kind: Option<String> = row.and_then(|r| match r.value("blob_storage") {
            Some(Value::Map(m)) => match m.get("kind") {
                Some(Value::String(s)) => Some(s.clone()),
                _ => None,
            },
            _ => None,
        });

        let persisted_embedding = row.and_then(|r| decode_embedding_identity(r.value("embedding")));

        let want_kind = self.config.blob_storage.kind();

        match persisted_kind {
            None => return self.write_kb_stats_first_open().await,
            Some(have) if have == want_kind => {}
            Some(have) => {
                return Err(UnikoError::Config(format!(
                    "blob_storage mismatch on KB reopen: persisted kind={have}, \
                     config kind={want_kind}. Implicit migration is not supported \
                     — open with the original backend, or migrate out-of-band."
                )));
            }
        }

        let want = EmbeddingIdentity::of(self);
        match persisted_embedding {
            None => self.write_embedding_identity().await,
            Some(have) if have == want => Ok(()),
            Some(have) => Err(UnikoError::Config(format!(
                "embedding mismatch on KB reopen: the store was built with \
                 model_id={} ({} dimensions), config has model_id={} ({} \
                 dimensions). Stored vectors are not comparable across models \
                 — open with the original embedding config, or re-ingest into \
                 a new store.",
                have.model_id, have.dimensions, want.model_id, want.dimensions,
            ))),
        }
    }

    /// Record the embedding identity on a row that predates it.
    async fn write_embedding_identity(&self) -> Result<()> {
        let session = self.db.session();
        let tx = session.tx().await?;
        tx.query_with(
            "MATCH (s:KnowledgeBaseStats {stats_id: $sid}) \
             SET s.embedding = $emb, s.updated_at = $now",
        )
        .param("sid", Value::String(STATS_ID.to_string()))
        .param("emb", EmbeddingIdentity::of(self).encode())
        .param("now", datetime_value(Utc::now()))
        .fetch_all()
        .await?;
        tx.commit().await?;
        Ok(())
    }

    /// Write the singleton row on first open. Called by `init_kb_stats`
    /// when no persisted row exists.
    async fn write_kb_stats_first_open(&self) -> Result<()> {
        let session = self.db.session();
        let tx = session.tx().await?;
        let cypher = "
            CREATE (s:KnowledgeBaseStats {
                stats_id: $sid,
                blob_storage: $bs,
                embedding: $emb,
                modality_presence: $mp,
                updated_at: $now
            })
        ";
        tx.query_with(cypher)
            .param("sid", Value::String(STATS_ID.to_string()))
            .param("bs", encode_blob_storage(&self.config.blob_storage))
            .param("emb", EmbeddingIdentity::of(self).encode())
            .param("mp", empty_modality_presence())
            .param("now", datetime_value(Utc::now()))
            .fetch_all()
            .await?;
        tx.commit().await?;
        Ok(())
    }

    /// Read the current `:KnowledgeBaseStats.modality_presence` flags.
    /// Returns `ModalityPresence::default()` (all false) if the
    /// singleton is missing or the map is empty.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn read_modality_presence(&self) -> Result<ModalityPresence> {
        let session = self.db.session();
        let result = session
            .query_with(
                "MATCH (s:KnowledgeBaseStats {stats_id: $sid}) \
                 RETURN s.modality_presence AS mp",
            )
            .param("sid", Value::String(STATS_ID.to_string()))
            .fetch_all()
            .await?;
        Ok(extract_modality_presence(result.rows().first()))
    }

    /// In-transaction read of `modality_presence`.
    ///
    /// Variant of [`KnowledgeBase::read_modality_presence`] that reuses
    /// an open transaction so a read-modify-write sequence (see
    /// [`KnowledgeBase::bump_modality_presence`]) observes a consistent
    /// snapshot rather than racing a concurrent writer.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    async fn read_modality_presence_in_tx(&self, tx: &Transaction) -> Result<ModalityPresence> {
        let result = tx
            .query_with(
                "MATCH (s:KnowledgeBaseStats {stats_id: $sid}) \
                 RETURN s.modality_presence AS mp",
            )
            .param("sid", Value::String(STATS_ID.to_string()))
            .fetch_all()
            .await?;
        Ok(extract_modality_presence(result.rows().first()))
    }

    /// Flip a single modality flag to `true` in the singleton's
    /// `modality_presence` map. Idempotent — re-flipping is a no-op.
    /// Used by Track B ingest paths; landed dormant in Phase 1 so the
    /// later wiring is a one-line call.
    ///
    /// `modality` accepts `"image"`, `"audio"`, `"video"`, or
    /// `"multimodal"` — any other value is silently accepted (forward
    /// compat).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on database failure.
    pub async fn bump_modality_presence(&self, modality: &str) -> Result<()> {
        // Serialize concurrent bumps. uni-db's transaction commit is
        // last-writer-wins on a row, so without this guard two parallel
        // bumps can each observe the same pre-image and clobber each
        // other's flip. Bumps are rare (first-occurrence per modality),
        // so the lock cost is negligible.
        let _guard = self.kb_stats_lock.lock().await;
        let session = self.db.session();
        let tx = session.tx().await?;

        // MERGE the singleton, then SET the flag. Two separate
        // statements share the same transaction so the row is created
        // if a Track B ingest fires before `init_kb_stats` has run.
        tx.query_with(
            "MERGE (s:KnowledgeBaseStats {stats_id: $sid}) \
             ON CREATE SET s.blob_storage = $bs, s.modality_presence = $mp, s.updated_at = $now",
        )
        .param("sid", Value::String(STATS_ID.to_string()))
        .param("bs", encode_blob_storage(&self.config.blob_storage))
        .param("mp", empty_modality_presence())
        .param("now", datetime_value(Utc::now()))
        .fetch_all()
        .await?;

        // Update the single flag by recomputing the map. uni-db's Map
        // properties are atomic — there's no per-key SET, so we round-
        // trip the full map. Cost is negligible (4-entry map).
        //
        // The read MUST share `tx` with the write below — otherwise
        // two concurrent `bump_modality_presence` calls can read the
        // same pre-image and produce a lost update.
        let cur = self.read_modality_presence_in_tx(&tx).await?;
        let mut m: HashMap<String, Value> = HashMap::new();
        for k in MODALITIES {
            m.insert((*k).to_string(), Value::Bool(false));
        }
        if cur.has_image_content {
            m.insert("image".into(), Value::Bool(true));
        }
        if cur.has_audio_content {
            m.insert("audio".into(), Value::Bool(true));
        }
        if cur.has_video_content {
            m.insert("video".into(), Value::Bool(true));
        }
        if cur.has_multimodal_indexed {
            m.insert("multimodal".into(), Value::Bool(true));
        }
        m.insert(modality.to_string(), Value::Bool(true));

        tx.query_with(
            "MATCH (s:KnowledgeBaseStats {stats_id: $sid}) \
             SET s.modality_presence = $mp, s.updated_at = $now",
        )
        .param("sid", Value::String(STATS_ID.to_string()))
        .param("mp", Value::Map(m))
        .param("now", datetime_value(Utc::now()))
        .fetch_all()
        .await?;

        tx.commit().await?;
        Ok(())
    }
}

/// Serialize a [`BlobStorage`] into the `:KnowledgeBaseStats.blob_storage`
/// map shape. Only `kind` is required for the mismatch check; the
/// remaining fields are persisted as informational metadata.
/// The embedding model a store's vectors were made with.
#[derive(Debug, PartialEq, Eq)]
struct EmbeddingIdentity {
    model_id: String,
    dimensions: i64,
}

impl EmbeddingIdentity {
    fn of(kb: &KnowledgeBase) -> Self {
        Self {
            model_id: kb.config.embedding.model_id.clone(),
            dimensions: i64::try_from(kb.config.embedding.dimensions).unwrap_or(i64::MAX),
        }
    }

    fn encode(&self) -> Value {
        let mut m: HashMap<String, Value> = HashMap::new();
        m.insert("model_id".into(), Value::String(self.model_id.clone()));
        m.insert("dimensions".into(), Value::Int(self.dimensions));
        Value::Map(m)
    }
}

fn decode_embedding_identity(value: Option<&Value>) -> Option<EmbeddingIdentity> {
    let Some(Value::Map(m)) = value else {
        return None;
    };
    let model_id = match m.get("model_id") {
        Some(Value::String(s)) => s.clone(),
        _ => return None,
    };
    let dimensions = match m.get("dimensions") {
        Some(Value::Int(n)) => *n,
        _ => return None,
    };
    Some(EmbeddingIdentity {
        model_id,
        dimensions,
    })
}

fn encode_blob_storage(bs: &BlobStorage) -> Value {
    let mut m: HashMap<String, Value> = HashMap::new();
    m.insert("kind".into(), Value::String(bs.kind().to_string()));
    match bs {
        BlobStorage::Lance => {}
        BlobStorage::Fs { root } => {
            m.insert("root".into(), Value::String(root.display().to_string()));
        }
        BlobStorage::S3 {
            bucket,
            prefix,
            endpoint,
            region,
        } => {
            m.insert("bucket".into(), Value::String(bucket.clone()));
            if let Some(p) = prefix {
                m.insert("prefix".into(), Value::String(p.clone()));
            }
            if let Some(e) = endpoint {
                m.insert("endpoint".into(), Value::String(e.clone()));
            }
            if let Some(r) = region {
                m.insert("region".into(), Value::String(r.clone()));
            }
        }
    }
    Value::Map(m)
}

fn empty_modality_presence() -> Value {
    let mut m: HashMap<String, Value> = HashMap::new();
    for k in MODALITIES {
        m.insert((*k).to_string(), Value::Bool(false));
    }
    Value::Map(m)
}

fn bool_flag(m: &HashMap<String, Value>, key: &str) -> bool {
    matches!(m.get(key), Some(Value::Bool(true)))
}

/// Convert an optional `:KnowledgeBaseStats` row into a
/// [`ModalityPresence`].
///
/// Returns [`ModalityPresence::default`] (all-false) when the row is
/// missing or its `mp` column is not a map.
fn extract_modality_presence(row: Option<&Row>) -> ModalityPresence {
    let mut p = ModalityPresence::default();
    if let Some(row) = row
        && let Some(Value::Map(m)) = row.value("mp")
    {
        p.has_image_content = bool_flag(m, "image");
        p.has_audio_content = bool_flag(m, "audio");
        p.has_video_content = bool_flag(m, "video");
        p.has_multimodal_indexed = bool_flag(m, "multimodal");
    }
    p
}
