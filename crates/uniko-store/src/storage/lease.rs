//! A lease on a store kept in an object store, so a second process can't
//! open it under a live one.
//!
//! uni-db takes no lock on a store, and a remote store has no directory to
//! take an OS lock on. So an open remote store holds a lease object,
//! `.uniko.lease` under the store's prefix:
//!
//! - **acquire**: a create-only put (`If-None-Match`). When a lease is there
//!   and unexpired, the open is refused, naming its holder. An expired one
//!   (its holder crashed) is taken over with a put conditional on its ETag,
//!   so two openers racing for it can't both win;
//! - **renew**: a background task rewrites the lease every
//!   [`RENEW_EVERY`], conditional on the ETag it last wrote, pushing its
//!   expiry [`LEASE_TTL`] ahead. A renewal that fails its precondition means
//!   another process took the lease: this holder marks it lost, and the
//!   store refuses every write from then on;
//! - **release**: on shutdown the lease is deleted, so the next open needn't
//!   wait for it to expire.
//!
//! A holder that stalls past its lease (a paused VM, a long GC) and wakes
//! with writes already under way can't be stopped from here: that needs
//! uni-db to fence its own commits on the lease. Clocks are compared across
//! hosts, so the TTL is also the margin for clock skew.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use object_store::path::Path as ObjPath;
use object_store::{ObjectStore, ObjectStoreExt, PutMode, PutOptions, PutPayload, UpdateVersion};
use serde::{Deserialize, Serialize};

use crate::error::{Result, UnikoError};

/// The lease object's name, under the store's prefix.
pub const LEASE_FILE: &str = ".uniko.lease";
/// How long a lease lasts without renewal.
pub const LEASE_TTL: Duration = Duration::from_secs(30);
/// How often the holder renews it.
pub const RENEW_EVERY: Duration = Duration::from_secs(10);

/// What a lease object holds.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LeaseBody {
    holder: String,
    expires_at_ms: i64,
}

fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

fn body(holder: &str, ttl: Duration) -> PutPayload {
    let b = LeaseBody {
        holder: holder.to_string(),
        expires_at_ms: now_ms() + i64::try_from(ttl.as_millis()).unwrap_or(i64::MAX),
    };
    PutPayload::from(serde_json::to_vec(&b).expect("a lease body serializes"))
}

fn storage(e: object_store::Error) -> UnikoError {
    UnikoError::Storage(format!("store lease: {e}"))
}

/// A held lease. Dropping it stops renewal; [`release`](Self::release)
/// also deletes it.
#[derive(Debug)]
pub(crate) struct RemoteLease {
    store: Arc<dyn ObjectStore>,
    path: ObjPath,
    holder: String,
    lost: Arc<AtomicBool>,
    renew: tokio::task::JoinHandle<()>,
}

impl RemoteLease {
    /// Acquire the lease on the store at `uri` (`s3://bucket/prefix`, ...),
    /// reaching the bucket the way uni-db does for a remote URI.
    pub(crate) async fn acquire_for_uri(uri: &str) -> Result<Self> {
        let url = url::Url::parse(uri)
            .map_err(|e| UnikoError::Config(format!("store URI {uri}: {e}")))?;
        let (store, prefix) = object_store::parse_url(&url).map_err(storage)?;
        Self::acquire(Arc::from(store), &prefix, LEASE_TTL, RENEW_EVERY).await
    }

    /// Acquire the lease under `prefix` in `store`.
    pub(crate) async fn acquire(
        store: Arc<dyn ObjectStore>,
        prefix: &ObjPath,
        ttl: Duration,
        renew_every: Duration,
    ) -> Result<Self> {
        let path = prefix.clone().join(LEASE_FILE);
        let holder = format!("{}-{}", std::process::id(), uuid::Uuid::now_v7());
        let created = store
            .put_opts(&path, body(&holder, ttl), PutOptions::from(PutMode::Create))
            .await;
        let result = match created {
            Ok(r) => r,
            Err(object_store::Error::AlreadyExists { .. }) => {
                Self::take_over_if_expired(&store, &path, &holder, ttl).await?
            }
            Err(e) => return Err(storage(e)),
        };
        let version = Arc::new(tokio::sync::Mutex::new(UpdateVersion {
            e_tag: result.e_tag,
            version: result.version,
        }));
        let lost = Arc::new(AtomicBool::new(false));
        let renew = tokio::spawn(renew_loop(
            Arc::clone(&store),
            path.clone(),
            holder.clone(),
            version,
            Arc::clone(&lost),
            ttl,
            renew_every,
        ));
        Ok(Self {
            store,
            path,
            holder,
            lost,
            renew,
        })
    }

    /// The lease exists: take it over if it has expired, else refuse.
    async fn take_over_if_expired(
        store: &Arc<dyn ObjectStore>,
        path: &ObjPath,
        holder: &str,
        ttl: Duration,
    ) -> Result<object_store::PutResult> {
        let got = store.get(path).await.map_err(storage)?;
        let seen = UpdateVersion {
            e_tag: got.meta.e_tag.clone(),
            version: got.meta.version.clone(),
        };
        let bytes = got.bytes().await.map_err(storage)?;
        let current: Option<LeaseBody> = serde_json::from_slice(&bytes).ok();
        if let Some(c) = &current
            && c.expires_at_ms > now_ms()
        {
            return Err(UnikoError::Config(format!(
                "the store is open in another process (lease holder {}, valid for {} more \
                 s); a store takes one open handle at a time",
                c.holder,
                (c.expires_at_ms - now_ms()) / 1000
            )));
        }
        // Expired (or unreadable): take it over, conditional on the version
        // seen, so of two openers racing for it only one wins.
        match store
            .put_opts(
                path,
                body(holder, ttl),
                PutOptions::from(PutMode::Update(seen)),
            )
            .await
        {
            Ok(r) => Ok(r),
            Err(object_store::Error::Precondition { .. }) => Err(UnikoError::Config(
                "the store is being opened by another process; a store takes one open \
                 handle at a time"
                    .into(),
            )),
            Err(e) => Err(storage(e)),
        }
    }

    /// Whether another process has taken the lease since this one acquired
    /// it. Once lost, it stays lost.
    pub(crate) fn is_lost(&self) -> bool {
        self.lost.load(Ordering::SeqCst)
    }

    /// The error a write gets once the lease is lost.
    pub(crate) fn lost_error() -> UnikoError {
        UnikoError::Config(
            "this store's lease was taken over by another process (it renewed too late); \
             writes are refused, so shut this instance down"
                .into(),
        )
    }

    /// Stop renewing and delete the lease, if this holder still has it.
    pub(crate) async fn release(&self) {
        self.renew.abort();
        if self.is_lost() {
            return;
        }
        // Delete only our own lease: re-read it, and leave it if another
        // holder's is there now.
        if let Ok(got) = self.store.get(&self.path).await
            && let Ok(bytes) = got.bytes().await
            && serde_json::from_slice::<LeaseBody>(&bytes).is_ok_and(|b| b.holder == self.holder)
            && let Err(e) = self.store.delete(&self.path).await
        {
            tracing::warn!(error = %e, "could not delete the store lease; it will expire");
        }
    }
}

impl Drop for RemoteLease {
    fn drop(&mut self) {
        self.renew.abort();
    }
}

/// Renew the lease every `every`, conditional on the version last written,
/// until it's lost or the task is aborted.
async fn renew_loop(
    store: Arc<dyn ObjectStore>,
    path: ObjPath,
    holder: String,
    version: Arc<tokio::sync::Mutex<UpdateVersion>>,
    lost: Arc<AtomicBool>,
    ttl: Duration,
    every: Duration,
) {
    loop {
        tokio::time::sleep(every).await;
        let mut v = version.lock().await;
        match store
            .put_opts(
                &path,
                body(&holder, ttl),
                PutOptions::from(PutMode::Update(v.clone())),
            )
            .await
        {
            Ok(r) => {
                *v = UpdateVersion {
                    e_tag: r.e_tag,
                    version: r.version,
                };
            }
            Err(
                object_store::Error::Precondition { .. } | object_store::Error::NotFound { .. },
            ) => {
                lost.store(true, Ordering::SeqCst);
                tracing::error!(
                    lease = %path,
                    "the store lease was taken over by another process; refusing writes"
                );
                return;
            }
            // A transient failure: keep the lease and try again next time.
            // If renewals keep failing past the TTL, another process can take
            // it over, and the next renewal finds that out.
            Err(e) => tracing::warn!(error = %e, lease = %path, "renewing the store lease failed"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;

    fn prefix() -> ObjPath {
        ObjPath::from("bucket/kb")
    }

    #[tokio::test]
    async fn a_live_lease_refuses_a_second_holder() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let a = RemoteLease::acquire(Arc::clone(&store), &prefix(), LEASE_TTL, RENEW_EVERY)
            .await
            .unwrap();
        let err = RemoteLease::acquire(Arc::clone(&store), &prefix(), LEASE_TTL, RENEW_EVERY)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("open in another process"), "{err}");
        a.release().await;
        RemoteLease::acquire(store, &prefix(), LEASE_TTL, RENEW_EVERY)
            .await
            .expect("free once released")
            .release()
            .await;
    }

    #[tokio::test]
    async fn an_expired_lease_is_taken_over_and_its_holder_learns_it_lost() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        // `a` renews too slowly for its TTL, as a stalled process would.
        let a = RemoteLease::acquire(
            Arc::clone(&store),
            &prefix(),
            Duration::from_millis(50),
            Duration::from_millis(300),
        )
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        let b = RemoteLease::acquire(Arc::clone(&store), &prefix(), LEASE_TTL, RENEW_EVERY)
            .await
            .expect("an expired lease is taken over");
        tokio::time::sleep(Duration::from_millis(400)).await;
        assert!(a.is_lost(), "a's next renewal finds b's lease");
        assert!(!b.is_lost());
        a.release().await;
        assert!(
            store.head(&prefix().join(LEASE_FILE)).await.is_ok(),
            "a lost holder leaves the new holder's lease alone"
        );
        b.release().await;
        assert!(store.head(&prefix().join(LEASE_FILE)).await.is_err());
    }

    #[tokio::test]
    async fn a_held_lease_is_renewed() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let a = RemoteLease::acquire(
            Arc::clone(&store),
            &prefix(),
            Duration::from_millis(200),
            Duration::from_millis(50),
        )
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert!(!a.is_lost());
        let err = RemoteLease::acquire(Arc::clone(&store), &prefix(), LEASE_TTL, RENEW_EVERY)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("open in another process"), "{err}");
        a.release().await;
    }
}

/// The same lease rules against a real S3 implementation: the AWS Emulator
/// (`scripts/aws-emulator/start.sh`), which honors `If-None-Match` and
/// `If-Match` as S3 does. Skipped unless `UNIKO_S3_EMULATOR_ENDPOINT`,
/// `UNIKO_S3_EMULATOR_CA` and `UNIKO_S3_EMULATOR_BUCKET` are set.
#[cfg(test)]
mod emulator_tests {
    use super::*;
    use object_store::ClientOptions;
    use object_store::aws::AmazonS3Builder;

    /// The emulator's bucket, or `None` (skip) when it isn't configured.
    fn emulator() -> Option<Arc<dyn ObjectStore>> {
        let endpoint = std::env::var("UNIKO_S3_EMULATOR_ENDPOINT").ok()?;
        let ca = std::env::var("UNIKO_S3_EMULATOR_CA").ok()?;
        let bucket = std::env::var("UNIKO_S3_EMULATOR_BUCKET").ok()?;
        let pem = std::fs::read(&ca).expect("read the emulator CA");
        let store = AmazonS3Builder::new()
            .with_endpoint(endpoint)
            .with_bucket_name(bucket)
            .with_region("us-east-1")
            .with_access_key_id("test")
            .with_secret_access_key("test")
            .with_virtual_hosted_style_request(false)
            .with_client_options(ClientOptions::new().with_root_certificate(
                object_store::Certificate::from_pem(&pem).expect("a PEM CA"),
            ))
            .build()
            .expect("an S3 client for the emulator");
        Some(Arc::new(store))
    }

    /// A prefix of its own per test, so tests run in parallel.
    fn prefix(test: &str) -> ObjPath {
        ObjPath::from(format!("{test}-{}", uuid::Uuid::now_v7()))
    }

    macro_rules! emulator_or_skip {
        () => {
            match emulator() {
                Some(s) => s,
                None => {
                    eprintln!("skipping: set UNIKO_S3_EMULATOR_* (scripts/aws-emulator/start.sh)");
                    return;
                }
            }
        };
    }

    #[tokio::test]
    async fn emulator_a_live_lease_refuses_a_second_holder_and_frees_on_release() {
        let store = emulator_or_skip!();
        let p = prefix("live");
        let a = RemoteLease::acquire(Arc::clone(&store), &p, LEASE_TTL, RENEW_EVERY)
            .await
            .expect("a create-only put on an empty prefix");
        let err = RemoteLease::acquire(Arc::clone(&store), &p, LEASE_TTL, RENEW_EVERY)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("open in another process"), "{err}");
        a.release().await;
        assert!(
            store.head(&p.clone().join(LEASE_FILE)).await.is_err(),
            "deleted on release"
        );
        RemoteLease::acquire(Arc::clone(&store), &p, LEASE_TTL, RENEW_EVERY)
            .await
            .expect("free once released")
            .release()
            .await;
    }

    #[tokio::test]
    async fn emulator_a_held_lease_is_renewed_with_if_match() {
        let store = emulator_or_skip!();
        let p = prefix("renew");
        let a = RemoteLease::acquire(
            Arc::clone(&store),
            &p,
            Duration::from_secs(2),
            Duration::from_millis(300),
        )
        .await
        .unwrap();
        // Several renewals past the TTL: each If-Match succeeded, or the
        // lease would be lost or another holder could take it.
        tokio::time::sleep(Duration::from_secs(3)).await;
        assert!(!a.is_lost());
        assert!(
            RemoteLease::acquire(Arc::clone(&store), &p, LEASE_TTL, RENEW_EVERY)
                .await
                .is_err(),
            "still held after its first TTL"
        );
        a.release().await;
    }

    #[tokio::test]
    async fn emulator_an_expired_lease_is_taken_over_and_the_old_holder_loses_it() {
        let store = emulator_or_skip!();
        let p = prefix("takeover");
        let a = RemoteLease::acquire(
            Arc::clone(&store),
            &p,
            Duration::from_millis(500),
            Duration::from_secs(2),
        )
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_secs(1)).await;
        let b = RemoteLease::acquire(Arc::clone(&store), &p, LEASE_TTL, RENEW_EVERY)
            .await
            .expect("an expired lease is taken over with If-Match");
        tokio::time::sleep(Duration::from_millis(2_500)).await;
        assert!(a.is_lost(), "a's If-Match renewal fails against b's lease");
        assert!(!b.is_lost());
        a.release().await;
        assert!(
            store.head(&p.clone().join(LEASE_FILE)).await.is_ok(),
            "b's lease stays"
        );
        b.release().await;
    }

    #[tokio::test]
    async fn emulator_two_openers_racing_for_an_expired_lease_get_one_winner() {
        let store = emulator_or_skip!();
        let p = prefix("race");
        let stale = RemoteLease::acquire(
            Arc::clone(&store),
            &p,
            Duration::from_millis(200),
            Duration::from_secs(60),
        )
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(500)).await;
        let attempts = futures::future::join_all(
            (0..6).map(|_| RemoteLease::acquire(Arc::clone(&store), &p, LEASE_TTL, RENEW_EVERY)),
        )
        .await;
        let (won, lost): (Vec<_>, Vec<_>) = attempts.into_iter().partition(Result::is_ok);
        assert_eq!(won.len(), 1, "exactly one opener takes the lease over");
        assert_eq!(lost.len(), 5);
        drop(stale);
        for l in won.into_iter().flatten() {
            l.release().await;
        }
    }
}
