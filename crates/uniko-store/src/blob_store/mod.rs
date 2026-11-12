//! Pluggable blob storage for `:ArtifactContent` bytes.
//!
//! Bytes for every unique blob live in one of three backends, chosen
//! once per KB at init via [`BlobStorage`]:
//!
//! - [`BlobStorage::Lance`] — bytes inline in `:ArtifactContent.bytes`.
//!   Single-tx with the graph write. Zero new deps. Default.
//! - [`BlobStorage::Fs`] — local content-addressed store at
//!   `<root>/<aa>/<bb>/<content_id>`. Two-level fanout from the first
//!   4 hex chars to bound directory size.
//! - [`BlobStorage::S3`] — S3-compatible object store (S3, R2, GCS,
//!   MinIO) via the `object_store` crate. **Not yet wired** — falls
//!   through to a runtime error until the `object_store` dep lands.
//!
//! The graph-side invariants (MERGE on `content_id`, N:1 fan-in via
//! `HAS_CONTENT`) are identical across backends. Only the path to
//! bytes differs.
//
use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;

pub mod fs;
pub mod lance;
pub mod s3;

/// Per-KB blob-storage backend selection.
///
/// Persisted in `:KnowledgeBaseStats.blob_storage` at first open;
/// reopening with a different variant is a hard error (no implicit
/// migration). See `crates/uniko-store/src/storage/mod.rs`.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum BlobStorage {
    /// Bytes stored in `:ArtifactContent.bytes`. Single-transaction
    /// with the graph write. Default.
    #[default]
    Lance,
    /// Local filesystem CAS rooted at `root`.
    Fs {
        /// Root directory. Created lazily on first PUT if absent.
        root: PathBuf,
    },
    /// S3-compatible object store. **Not yet implemented.**
    S3 {
        /// Bucket name (no `s3://` prefix).
        bucket: String,
        /// Optional key prefix (e.g., `"uniko/kb-prod/"`).
        prefix: Option<String>,
        /// Optional endpoint override for R2 / MinIO.
        endpoint: Option<String>,
        /// Optional region (defaults to provider-specific default).
        region: Option<String>,
    },
}

impl BlobStorage {
    /// Stable short discriminator used in error messages and the
    /// `:KnowledgeBaseStats.blob_storage` map.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Lance => "lance",
            Self::Fs { .. } => "fs",
            Self::S3 { .. } => "s3",
        }
    }
}

/// Result of a `put` call.
///
/// For [`BlobStorage::Lance`], `bytes_inline` carries the blob so the
/// caller can thread it into the same Cypher transaction. For `Fs`
/// and `S3`, the blob has already been written and `uri` carries the
/// backend-resolvable locator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PutOutcome {
    /// Bytes to write into `:ArtifactContent.bytes` (Lance backend only).
    pub bytes_inline: Option<Vec<u8>>,
    /// Backend-resolvable URI for external backends.
    pub uri: Option<String>,
}

/// Pluggable blob backend.
///
/// All methods are content-addressed by `content_id` (SHA-256 hex).
/// Implementations are idempotent on identical bytes: re-`put`-ing the
/// same hash is a no-op.
//
// `async_trait` tags each bodiless method's generated signature with a bare
// `#[must_use]`, and the boxed future it returns is already `#[must_use]`, so
// clippy >= 1.99 reports `double_must_use` on code this file never wrote. The
// allow has to sit on the trait: the attribute is emitted inside the macro.
#[allow(clippy::double_must_use)]
#[async_trait]
pub trait BlobStore: Send + Sync {
    /// Write bytes for `content_id`. Idempotent on identical bytes.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] if the backend write fails.
    async fn put(&self, content_id: &str, bytes: &[u8]) -> Result<PutOutcome>;

    /// Read bytes back. `uri` is the backend-resolvable locator stored
    /// on `:ArtifactContent.uri` (NULL for Lance — pass `None`).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] if the blob is missing or the
    /// backend read fails.
    async fn get(&self, content_id: &str, uri: Option<&str>) -> Result<Vec<u8>>;

    /// Best-effort existence check.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on transport / IO failure.
    async fn exists(&self, content_id: &str, uri: Option<&str>) -> Result<bool>;

    /// Delete a blob. Idempotent — deleting a missing blob is `Ok(())`.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`] on backend failure (other than
    /// "not found" which is silently ignored).
    async fn delete(&self, content_id: &str, uri: Option<&str>) -> Result<()>;
}

/// Construct the configured backend implementation.
///
/// # Errors
///
/// Returns [`UnikoError::Config`] if the backend cannot be initialized
/// from the given config (e.g., `Fs` root is unreachable, or `S3` is
/// requested but its support has not yet been compiled in).
pub fn build_backend(cfg: &BlobStorage) -> Result<Box<dyn BlobStore>> {
    match cfg {
        BlobStorage::Lance => Ok(Box::new(lance::LanceBlobStore::new()) as Box<dyn BlobStore>),
        BlobStorage::Fs { root } => {
            Ok(Box::new(fs::FsBlobStore::new(root.clone())?) as Box<dyn BlobStore>)
        }
        BlobStorage::S3 {
            bucket,
            prefix,
            endpoint,
            region,
        } => Ok(Box::new(s3::S3BlobStore::from_config(
            bucket,
            prefix.as_deref(),
            endpoint.as_deref(),
            region.as_deref(),
        )?) as Box<dyn BlobStore>),
    }
}

/// Helper: derive the canonical `Fs`-backend relative path
/// (`<aa>/<bb>/<content_id>`) for a given hex hash. Exposed so the
/// migration script can mirror layout without duplicating the rule.
///
/// Returns `None` when `content_id` is shorter than 4 characters —
/// the two-level fanout requires at least four hex chars. SHA-256 hex
/// is always 64 chars, so any caller passing a valid hash gets `Some`.
#[must_use]
pub fn fs_relative_path(content_id: &str) -> Option<PathBuf> {
    if content_id.len() < 4 {
        return None;
    }
    let (aa, rest) = content_id.split_at(2);
    let (bb, _) = rest.split_at(2);
    Some(PathBuf::from(aa).join(bb).join(content_id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fs_relative_path_fanout() {
        let p = fs_relative_path("abcdef1234567890").expect("valid 16-char hash");
        assert_eq!(p, PathBuf::from("ab/cd/abcdef1234567890"));
    }

    #[test]
    fn test_fs_relative_path_too_short() {
        assert!(fs_relative_path("abc").is_none());
        assert!(fs_relative_path("").is_none());
    }

    #[test]
    fn test_blob_storage_default() {
        assert!(matches!(BlobStorage::default(), BlobStorage::Lance));
    }

    #[test]
    fn test_blob_storage_kind() {
        assert_eq!(BlobStorage::Lance.kind(), "lance");
        assert_eq!(
            BlobStorage::Fs {
                root: PathBuf::from("/tmp/x")
            }
            .kind(),
            "fs"
        );
    }
}
