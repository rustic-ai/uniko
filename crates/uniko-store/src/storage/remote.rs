//! A store whose data lives in an object store ([`RemoteStorage`]).
//!
//! uni-db opens such a store hybrid (the graph tables, indexes and schema in
//! the bucket, the write-ahead log and id allocator in a local directory) or
//! remote-only (everything in the bucket). The settings are resolved once,
//! here, so uni-db and uniko's own checks reach the bucket with the same
//! endpoint and credentials.

use std::sync::Arc;

use object_store::ObjectStore;
use object_store::path::Path as ObjPath;
use uni_db::common::CloudStorageConfig;
use uni_db::common::config::ObjectStoreConfig;
use uni_db::store::location::{LocationSpec, StoreLocation};

use crate::config::RemoteStorage;
use crate::error::{Result, UnikoError};

/// [`RemoteStorage`] parsed, in the form uni-db takes it.
#[derive(Clone)]
pub(crate) struct ResolvedRemote {
    /// The store URL as configured, for uni-db and error messages.
    pub(crate) url: String,
    /// The settings given explicitly, for uni-db to layer the environment
    /// under. `None` for an Azure store with no account named: uni-db's
    /// Azure settings need one, so then it reads all of them, the account
    /// included, from the environment.
    pub(crate) cloud: Option<CloudStorageConfig>,
}

impl std::fmt::Debug for ResolvedRemote {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolvedRemote")
            .field("url", &self.url)
            .finish_non_exhaustive()
    }
}

fn config_error(url: &str, why: impl std::fmt::Display) -> UnikoError {
    UnikoError::Config(format!("remote_storage.url {url:?}: {why}"))
}

/// Split `url` into its scheme, bucket and key prefix.
pub(crate) fn parse(url: &str) -> Result<(String, String, ObjPath)> {
    let parsed = url::Url::parse(url).map_err(|e| config_error(url, e))?;
    let scheme = parsed.scheme().to_string();
    if !matches!(scheme.as_str(), "s3" | "gs" | "az") {
        return Err(config_error(url, "expected an s3://, gs:// or az:// URL"));
    }
    let bucket = parsed
        .host_str()
        .filter(|b| !b.is_empty())
        .ok_or_else(|| config_error(url, "no bucket or container"))?
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

/// The Azure settings given explicitly. An account key or SAS token needs
/// the account it belongs to: `account`, else `AZURE_STORAGE_ACCOUNT` (or
/// `AZURE_STORAGE_ACCOUNT_NAME`), the names uni-db reads it by. With none of
/// them given, uni-db takes everything from the environment.
fn azure(remote: &RemoteStorage, container: String) -> Result<Option<CloudStorageConfig>> {
    let from_env = || {
        ["AZURE_STORAGE_ACCOUNT", "AZURE_STORAGE_ACCOUNT_NAME"]
            .iter()
            .find_map(|n| std::env::var(n).ok().filter(|v| !v.is_empty()))
    };
    let credentials = remote.access_key.is_some() || remote.sas_token.is_some();
    let account = match &remote.account {
        Some(a) => Some(a.clone()),
        None if credentials => Some(from_env().ok_or_else(|| {
            config_error(
                &remote.url,
                "an access_key or sas_token needs its storage account: set account, or \
                 AZURE_STORAGE_ACCOUNT",
            )
        })?),
        None => None,
    };
    Ok(account.map(|account| CloudStorageConfig::Azure {
        container,
        account,
        access_key: remote.access_key.clone(),
        sas_token: remote.sas_token.clone(),
    }))
}

/// Refuse a setting that belongs to another provider than the URL's: it
/// would otherwise be dropped without a word.
fn check_provider_settings(remote: &RemoteStorage, scheme: &str) -> Result<()> {
    let set = |v: &Option<String>| v.is_some();
    let fields: [(&str, &str, bool); 11] = [
        ("s3", "region", set(&remote.region)),
        ("s3", "endpoint", set(&remote.endpoint)),
        ("s3", "access_key_id", set(&remote.access_key_id)),
        ("s3", "secret_access_key", set(&remote.secret_access_key)),
        ("s3", "session_token", set(&remote.session_token)),
        ("s3", "virtual_hosted_style", remote.virtual_hosted_style),
        (
            "gs",
            "service_account_path",
            set(&remote.service_account_path),
        ),
        (
            "gs",
            "service_account_key",
            set(&remote.service_account_key),
        ),
        ("az", "account", set(&remote.account)),
        ("az", "access_key", set(&remote.access_key)),
        ("az", "sas_token", set(&remote.sas_token)),
    ];
    let wrong: Vec<&str> = fields
        .iter()
        .filter(|(for_scheme, _, is_set)| *is_set && *for_scheme != scheme)
        .map(|(_, name, _)| *name)
        .collect();
    if wrong.is_empty() {
        return Ok(());
    }
    Err(config_error(
        &remote.url,
        format!(
            "not a setting for a {scheme}:// store: {}",
            wrong.join(", ")
        ),
    ))
}

impl ResolvedRemote {
    /// Resolve `remote`: only the settings it gives explicitly go in
    /// [`cloud`](Self::cloud). uni-db layers the environment under them
    /// itself (`AWS_ENDPOINT_URL_S3` before `AWS_ENDPOINT_URL`,
    /// `AWS_VIRTUAL_HOSTED_STYLE_REQUEST`, `AWS_PROFILE`, Google's
    /// application-default credentials), so filling them from the
    /// environment here would only turn an inherited variable into an
    /// explicit setting that outranks the ones uni-db prefers.
    pub(crate) fn resolve(remote: &RemoteStorage) -> Result<Self> {
        let (scheme, bucket, _) = parse(&remote.url)?;
        check_provider_settings(remote, &scheme)?;
        let cloud = match scheme.as_str() {
            "s3" => Some(CloudStorageConfig::S3 {
                bucket,
                region: remote.region.clone(),
                endpoint: remote.endpoint.clone(),
                access_key_id: remote.access_key_id.clone(),
                secret_access_key: remote.secret_access_key.clone(),
                session_token: remote.session_token.clone(),
                virtual_hosted_style: remote.virtual_hosted_style,
            }),
            "gs" => Some(CloudStorageConfig::Gcs {
                bucket,
                service_account_path: remote.service_account_path.clone(),
                service_account_key: remote.service_account_key.clone(),
            }),
            _ => azure(remote, bucket)?,
        };
        Ok(Self {
            url: remote.url.clone(),
            cloud,
        })
    }

    /// A client rooted at the store's prefix, resolved by uni-db from
    /// [`cloud`](Self::cloud) and the environment exactly as it resolves
    /// the store it opens, so both reach the same bucket the same way.
    pub(crate) async fn store_root(&self) -> Result<Arc<dyn ObjectStore>> {
        let spec = LocationSpec {
            uri: self.url.clone(),
            cloud: self.cloud.clone(),
            options: Default::default(),
        };
        let location = StoreLocation::resolve_from_env(&spec, &ObjectStoreConfig::default())
            .await
            .map_err(|e| UnikoError::Config(format!("remote_storage {}: {e}", self.url)))?;
        Ok(location.meta_store())
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
        for url in ["/local/path", "file:///tmp/kb", "s3://", "nonsense"] {
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
        } = resolved.cloud.as_ref().expect("s3 settings")
        else {
            panic!("an s3:// URL resolves to S3");
        };
        assert_eq!(bucket, "b");
        assert_eq!(region.as_deref(), Some("eu-west-1"));
        assert_eq!(endpoint.as_deref(), Some("http://localhost:4566"));
        assert_eq!(access_key_id.as_deref(), Some("id"));
        assert_eq!(secret_access_key.as_deref(), Some("secret"));
    }

    /// Settings left unset stay unset, for uni-db to fill from the
    /// environment in its own order.
    #[test]
    fn unset_settings_are_left_to_uni_db() {
        let resolved = ResolvedRemote::resolve(&RemoteStorage::s3("s3://b/kb")).unwrap();
        let CloudStorageConfig::S3 {
            region,
            endpoint,
            access_key_id,
            virtual_hosted_style,
            ..
        } = resolved.cloud.as_ref().expect("s3 settings")
        else {
            panic!("an s3:// URL resolves to S3");
        };
        assert!(region.is_none() && endpoint.is_none() && access_key_id.is_none());
        assert!(!virtual_hosted_style);
        let resolved = ResolvedRemote::resolve(&RemoteStorage::s3("gs://b")).unwrap();
        assert!(matches!(
            &resolved.cloud,
            Some(CloudStorageConfig::Gcs {
                service_account_path: None,
                service_account_key: None,
                ..
            })
        ));
    }

    /// uniko's own client is built by uni-db's location resolution.
    #[tokio::test]
    async fn the_client_is_resolved_by_uni_db() {
        let remote = RemoteStorage::s3("s3://b/kb")
            .with_endpoint("http://localhost:4566")
            .with_region("us-east-1")
            .with_credentials("id", "secret");
        let resolved = ResolvedRemote::resolve(&remote).unwrap();
        resolved.store_root().await.expect("builds a client");
    }

    #[test]
    fn an_azure_url_names_a_container() {
        let (scheme, container, prefix) = parse("az://kb-container/tenants/a").unwrap();
        assert_eq!(
            (scheme.as_str(), container.as_str(), prefix.as_ref()),
            ("az", "kb-container", "tenants/a")
        );
        let resolved = ResolvedRemote::resolve(
            &RemoteStorage::azure("az://kb-container/kb", "acct").with_access_key("key"),
        )
        .unwrap();
        let Some(CloudStorageConfig::Azure {
            container,
            account,
            access_key,
            sas_token,
        }) = &resolved.cloud
        else {
            panic!("an az:// URL with an account resolves to Azure");
        };
        assert_eq!(
            (container.as_str(), account.as_str()),
            ("kb-container", "acct")
        );
        assert_eq!(access_key.as_deref(), Some("key"));
        assert!(sas_token.is_none());
    }

    /// With no account named, uni-db reads every Azure setting from the
    /// environment.
    #[test]
    fn an_azure_store_without_an_account_is_left_to_uni_db() {
        if std::env::var_os("AZURE_STORAGE_ACCOUNT").is_some()
            || std::env::var_os("AZURE_STORAGE_ACCOUNT_NAME").is_some()
        {
            return;
        }
        let resolved = ResolvedRemote::resolve(&RemoteStorage::new("az://c/kb")).unwrap();
        assert!(resolved.cloud.is_none());
        let err = ResolvedRemote::resolve(&RemoteStorage::new("az://c/kb").with_sas_token("sv=x"))
            .unwrap_err();
        assert!(err.to_string().contains("storage account"), "{err}");
    }

    #[test]
    fn a_setting_for_another_provider_is_refused() {
        for (remote, field) in [
            (
                RemoteStorage::new("s3://b").with_access_key("k"),
                "access_key",
            ),
            (
                RemoteStorage::azure("az://c", "acct").with_region("eu"),
                "region",
            ),
            (
                RemoteStorage::new("gs://b").with_credentials("id", "secret"),
                "access_key_id, secret_access_key",
            ),
        ] {
            let err = ResolvedRemote::resolve(&remote).unwrap_err();
            assert!(err.to_string().contains(field), "{err}");
        }
    }

    #[test]
    fn store_urls_compare_by_bucket_and_prefix() {
        assert!(same_store("s3://b/kb", "s3://b/kb/"));
        assert!(same_store("s3://b", "s3://b/"));
        assert!(!same_store("s3://b/kb", "s3://b/other"));
        assert!(!same_store("s3://b/kb", "gs://b/kb"));
        assert!(same_store("az://c/kb", "az://c/kb/"));
        assert!(!same_store("az://b/kb", "s3://b/kb"));
        assert!(!same_store("/local", "/local"));
    }

    #[test]
    fn debug_hides_secrets() {
        let remote = RemoteStorage::s3("s3://b").with_credentials("id", "top-secret");
        let shown = format!("{remote:?}");
        assert!(!shown.contains("top-secret"), "{shown}");
        let remote = RemoteStorage::azure("az://c", "acct")
            .with_access_key("azure-key-secret")
            .with_sas_token("sas-secret");
        let shown = format!("{remote:?}");
        assert!(!shown.contains("azure-key-secret"), "{shown}");
        assert!(!shown.contains("sas-secret"), "{shown}");
        assert!(shown.contains("acct"), "{shown}");
        assert!(shown.contains("<hidden>"), "{shown}");
    }
}
