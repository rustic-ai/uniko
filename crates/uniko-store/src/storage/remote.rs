//! A store whose data lives in an object store ([`RemoteStorage`]).
//!
//! uni-db opens such a store hybrid (the graph tables, indexes and schema in
//! the bucket, the write-ahead log and id allocator in a local directory) or
//! remote-only (everything in the bucket). The settings are resolved once,
//! here, so uni-db and uniko's own checks reach the bucket with the same
//! endpoint and credentials.

use std::sync::Arc;

use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::path::Path as ObjPath;
use uni_db::common::CloudStorageConfig;

use crate::config::RemoteStorage;
use crate::error::{Result, UnikoError};

/// [`RemoteStorage`] with its environment fallbacks applied.
#[derive(Clone)]
pub(crate) struct ResolvedRemote {
    /// The store URL as configured, for uni-db and error messages.
    pub(crate) url: String,
    /// The key prefix under the bucket.
    pub(crate) prefix: ObjPath,
    /// What uni-db needs to reach the bucket.
    pub(crate) cloud: CloudStorageConfig,
}

impl std::fmt::Debug for ResolvedRemote {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolvedRemote")
            .field("url", &self.url)
            .finish_non_exhaustive()
    }
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

fn config_error(url: &str, why: impl std::fmt::Display) -> UnikoError {
    UnikoError::Config(format!("remote_storage.url {url:?}: {why}"))
}

/// Split `url` into its scheme, bucket and key prefix.
pub(crate) fn parse(url: &str) -> Result<(String, String, ObjPath)> {
    let parsed = url::Url::parse(url).map_err(|e| config_error(url, e))?;
    let scheme = parsed.scheme().to_string();
    if !matches!(scheme.as_str(), "s3" | "gs") {
        return Err(config_error(url, "expected an s3:// or gs:// URL"));
    }
    let bucket = parsed
        .host_str()
        .filter(|b| !b.is_empty())
        .ok_or_else(|| config_error(url, "no bucket"))?
        .to_string();
    let prefix = ObjPath::from(parsed.path().trim_matches('/'));
    Ok((scheme, bucket, prefix))
}

/// Whether `uri` names a store in an object store uniko supports.
pub(crate) fn is_object_store_uri(uri: &str) -> bool {
    parse(uri).is_ok()
}

/// Whether two store URLs name the same store: the same scheme, bucket and
/// prefix, whatever their slashes.
pub(crate) fn same_store(a: &str, b: &str) -> bool {
    matches!((parse(a), parse(b)), (Ok(a), Ok(b)) if a == b)
}

/// The object store `path` keeps its data in, if any: `config`'s
/// `remote_storage`, or a remote-only store's own URL with settings from the
/// environment.
pub(crate) fn for_path(
    path: &std::path::Path,
    config: &crate::config::UnikoConfig,
) -> Result<Option<ResolvedRemote>> {
    if let Some(remote) = &config.remote_storage {
        return ResolvedRemote::resolve(remote).map(Some);
    }
    let uri = path.to_string_lossy();
    if is_object_store_uri(&uri) {
        return ResolvedRemote::resolve(&RemoteStorage::s3(uri.into_owned())).map(Some);
    }
    Ok(None)
}

impl ResolvedRemote {
    /// Resolve `remote`, filling unset settings from the environment.
    pub(crate) fn resolve(remote: &RemoteStorage) -> Result<Self> {
        let (scheme, bucket, prefix) = parse(&remote.url)?;
        let pick = |v: &Option<String>, names: &[&str]| {
            v.clone().or_else(|| names.iter().find_map(|n| env(n)))
        };
        let cloud = match scheme.as_str() {
            "s3" => CloudStorageConfig::S3 {
                bucket,
                region: pick(&remote.region, &["AWS_REGION", "AWS_DEFAULT_REGION"]),
                endpoint: pick(&remote.endpoint, &["AWS_ENDPOINT_URL"]),
                access_key_id: pick(&remote.access_key_id, &["AWS_ACCESS_KEY_ID"]),
                secret_access_key: pick(&remote.secret_access_key, &["AWS_SECRET_ACCESS_KEY"]),
                session_token: pick(&remote.session_token, &["AWS_SESSION_TOKEN"]),
                virtual_hosted_style: remote.virtual_hosted_style,
            },
            _ => CloudStorageConfig::Gcs {
                bucket,
                service_account_path: pick(
                    &remote.service_account_path,
                    &["GOOGLE_APPLICATION_CREDENTIALS"],
                ),
                service_account_key: remote.service_account_key.clone(),
            },
        };
        Ok(Self {
            url: remote.url.clone(),
            prefix,
            cloud,
        })
    }

    /// A client for the bucket, built as uni-db builds its own from
    /// [`cloud`](Self::cloud).
    pub(crate) fn object_store(&self) -> Result<Arc<dyn ObjectStore>> {
        let err = |e: object_store::Error| {
            UnikoError::Config(format!("remote_storage {}: {e}", self.url))
        };
        match &self.cloud {
            CloudStorageConfig::S3 {
                bucket,
                region,
                endpoint,
                access_key_id,
                secret_access_key,
                session_token,
                virtual_hosted_style,
            } => {
                let mut b = AmazonS3Builder::new()
                    .with_bucket_name(bucket)
                    .with_virtual_hosted_style_request(*virtual_hosted_style);
                if let Some(v) = region {
                    b = b.with_region(v);
                }
                if let Some(v) = endpoint {
                    b = b.with_endpoint(v).with_allow_http(v.starts_with("http://"));
                }
                if let Some(v) = access_key_id {
                    b = b.with_access_key_id(v);
                }
                if let Some(v) = secret_access_key {
                    b = b.with_secret_access_key(v);
                }
                if let Some(v) = session_token {
                    b = b.with_token(v);
                }
                Ok(Arc::new(b.build().map_err(err)?))
            }
            CloudStorageConfig::Gcs {
                bucket,
                service_account_path,
                service_account_key,
            } => {
                let mut b = GoogleCloudStorageBuilder::new().with_bucket_name(bucket);
                if let Some(v) = service_account_path {
                    b = b.with_service_account_path(v);
                }
                if let Some(v) = service_account_key {
                    b = b.with_service_account_key(v);
                }
                Ok(Arc::new(b.build().map_err(err)?))
            }
            CloudStorageConfig::Azure { .. } => Err(UnikoError::Unsupported(format!(
                "remote_storage {}: Azure isn't supported",
                self.url
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_url_splits_into_bucket_and_prefix() {
        let (scheme, bucket, prefix) = parse("s3://kb-bucket/tenants/a/").unwrap();
        assert_eq!(
            (scheme.as_str(), bucket.as_str(), prefix.as_ref()),
            ("s3", "kb-bucket", "tenants/a")
        );
        let (_, _, root) = parse("gs://kb-bucket").unwrap();
        assert_eq!(root.as_ref(), "");
    }

    #[test]
    fn other_urls_are_refused() {
        for url in [
            "/local/path",
            "file:///tmp/kb",
            "az://acct/c",
            "s3://",
            "nonsense",
        ] {
            assert!(parse(url).is_err(), "{url}");
        }
    }

    #[test]
    fn explicit_settings_win_over_the_environment() {
        let remote = RemoteStorage::s3("s3://b/kb")
            .with_endpoint("http://localhost:4566")
            .with_region("eu-west-1")
            .with_credentials("id", "secret");
        let resolved = ResolvedRemote::resolve(&remote).unwrap();
        let CloudStorageConfig::S3 {
            bucket,
            region,
            endpoint,
            access_key_id,
            secret_access_key,
            ..
        } = &resolved.cloud
        else {
            panic!("an s3:// URL resolves to S3");
        };
        assert_eq!(bucket, "b");
        assert_eq!(region.as_deref(), Some("eu-west-1"));
        assert_eq!(endpoint.as_deref(), Some("http://localhost:4566"));
        assert_eq!(access_key_id.as_deref(), Some("id"));
        assert_eq!(secret_access_key.as_deref(), Some("secret"));
        assert_eq!(resolved.prefix.as_ref(), "kb");
        resolved.object_store().expect("builds a client");
    }

    #[test]
    fn store_urls_compare_by_bucket_and_prefix() {
        assert!(same_store("s3://b/kb", "s3://b/kb/"));
        assert!(same_store("s3://b", "s3://b/"));
        assert!(!same_store("s3://b/kb", "s3://b/other"));
        assert!(!same_store("s3://b/kb", "gs://b/kb"));
        assert!(!same_store("/local", "/local"));
    }

    #[test]
    fn debug_hides_secrets() {
        let remote = RemoteStorage::s3("s3://b").with_credentials("id", "top-secret");
        let shown = format!("{remote:?}");
        assert!(!shown.contains("top-secret"), "{shown}");
        assert!(shown.contains("<hidden>"), "{shown}");
    }
}
