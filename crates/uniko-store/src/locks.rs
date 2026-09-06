//! Striped async locks for serializing read-modify-write critical sections.
//!
//! uniko runs uni-db with SSI enabled (`UniConfig::ssi_enabled` defaults
//! to `true`), so two concurrent callers that read a row, mutate it
//! Rust-side, and write it back do not silently lose an update — the
//! second committer aborts with a retriable `SerializationConflict`
//! (surfaced as [`crate::UnikoError::Conflict`] and retried by
//! [`crate::KnowledgeBase::transact_with_retry`]).
//!
//! [`StripedLocks`] serialize same-key RMW *in-process* so those
//! conflicts are avoided up front rather than paid for as abort+retry
//! churn — and, critically, they also guard the check-then-create
//! pattern ("does this row exist? if not, CREATE it"), where two
//! concurrent transactions can each read "absent" and both insert a
//! duplicate row: an insert-phantom uni-db's read-set SSI does not
//! always catch (see the uni-db workarounds notes RC2). They provide
//! this without paying per-key allocation.
//!
//! # Design
//!
//! - Per-key locking, not global: callers compute key bytes (e.g.
//!   `"fact:<subject>\0<predicate>"`) and the table hashes them into a
//!   fixed-arity stripe array.  Collisions serialize unrelated keys
//!   harmlessly; the default 256 stripes give ~0.4% collision
//!   probability at 10 concurrent keys.
//! - Async-safe: holders may `.await` (uni-db tx commit is async), so
//!   stripes are [`tokio::sync::Mutex`] rather than [`std::sync::Mutex`].
//! - No reentrancy: stripes are non-reentrant `tokio::sync::Mutex`es, so
//!   a task that acquires a stripe it already holds blocks forever. This
//!   is not only about re-entering with the same *key*: two different
//!   keys collide whenever they hash to the same stripe, so any nested
//!   acquisition on one table is a latent self-deadlock (issue #36 —
//!   `session:…` and `node:Participant\0participant_id\0…` landed on
//!   stripe 149 of 256). A site that must hold a lock across a callee
//!   that locks again therefore uses a SEPARATE `StripedLocks` table
//!   (see [`crate::KnowledgeBase::lock_session_setup`] vs the
//!   `rmw_locks` table), never a larger stripe count — more stripes only
//!   lower the collision probability. Nested domains must be acquired in
//!   a fixed global order (setup before RMW) to keep them AB/BA-free.
//! - No timeout / no `try_lock`: RMW critical sections complete in
//!   milliseconds; a timeout would introduce partial-success states.
//!
//! # Out of scope
//!
//! - Cross-process locking — uniko is single-process today.
//! - Read locks / shared mode — RMW critical sections are always
//!   exclusive.
//! - CAS-style lock-free RMW — uni-db does not expose CAS server-side.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use tokio::sync::{Mutex, MutexGuard};

/// Environment override for stripe count.  Read once at construction.
const ENV_STRIPES: &str = "UNIKO_RMW_STRIPES";

/// Default stripe count.  Picked for ~0.4% collision probability at 10
/// concurrent keys; tunable via [`ENV_STRIPES`].
const DEFAULT_STRIPES: usize = 256;

/// Fixed-arity striped async lock keyed by arbitrary `&[u8]`.
///
/// See module docs for design rationale.
#[derive(Debug)]
pub struct StripedLocks {
    stripes: Box<[Mutex<()>]>,
}

impl StripedLocks {
    /// Build a [`StripedLocks`] with `n` stripes.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`.
    #[must_use]
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "StripedLocks must have at least 1 stripe");
        let stripes = (0..n).map(|_| Mutex::new(())).collect::<Vec<_>>();
        Self {
            stripes: stripes.into_boxed_slice(),
        }
    }

    /// Build a [`StripedLocks`] sized from the [`ENV_STRIPES`]
    /// environment variable, falling back to [`DEFAULT_STRIPES`].
    ///
    /// Non-numeric or zero values fall back to the default with a
    /// `tracing::warn!`.
    #[must_use]
    pub fn from_env() -> Self {
        let n = match std::env::var(ENV_STRIPES) {
            Ok(v) => match v.parse::<usize>() {
                Ok(n) if n > 0 => n,
                _ => {
                    tracing::warn!(
                        target: "uniko_store::locks",
                        value = %v,
                        default = DEFAULT_STRIPES,
                        "ignoring invalid {ENV_STRIPES}; using default",
                    );
                    DEFAULT_STRIPES
                }
            },
            Err(_) => DEFAULT_STRIPES,
        };
        Self::new(n)
    }

    /// Number of stripes in this table.
    #[must_use]
    pub fn stripe_count(&self) -> usize {
        self.stripes.len()
    }

    /// Acquire the stripe that hashes from `key`.
    ///
    /// Callers should hold the returned guard for the duration of the
    /// RMW critical section.
    pub async fn lock(&self, key: &[u8]) -> MutexGuard<'_, ()> {
        let idx = self.stripe_index(key);
        self.stripes[idx].lock().await
    }

    /// Acquire the stripes for a set of keys, **deduped by stripe index**
    /// and acquired in ascending stripe order.
    ///
    /// This is the safe way to hold several keys at once. Two *distinct*
    /// keys can hash to the same stripe; acquiring that stripe's
    /// non-reentrant [`Mutex`] twice in one call would self-deadlock, so
    /// keys are collapsed to their unique stripe set first. Sorting the
    /// stripe indices also gives a consistent global acquisition order,
    /// preventing AB/BA deadlocks between concurrent multi-key callers.
    ///
    /// Returns one guard per distinct stripe (may be fewer than `keys`).
    pub async fn lock_many(&self, keys: &[Vec<u8>]) -> Vec<MutexGuard<'_, ()>> {
        let mut idxs: Vec<usize> = keys.iter().map(|k| self.stripe_index(k)).collect();
        idxs.sort_unstable();
        idxs.dedup();
        let mut guards = Vec::with_capacity(idxs.len());
        for i in idxs {
            guards.push(self.stripes[i].lock().await);
        }
        guards
    }

    fn stripe_index(&self, key: &[u8]) -> usize {
        let mut h = DefaultHasher::new();
        key.hash(&mut h);
        // `stripes.len()` is non-zero (checked in `new`).
        (h.finish() as usize) % self.stripes.len()
    }
}

impl Default for StripedLocks {
    fn default() -> Self {
        Self::from_env()
    }
}

// ── Canonical lock-key builders ─────────────────────────────────────
//
// Every writer of a given logical row must hash the SAME key bytes for
// the stripe to actually serialize them. These helpers are the single
// source of truth for each row family's key, so a new write site cannot
// silently pick a divergent namespace (the defect behind issue #1).

/// Lock key for an `:Entity` row, keyed by its canonical `entity_id`.
///
/// Used by every Entity writer — dedup upsert, action linking, and
/// invalidation — so they all serialize on the same logical row.
#[must_use]
pub(crate) fn entity_lock_key(entity_id: &str) -> Vec<u8> {
    let mut k = Vec::with_capacity(7 + entity_id.len());
    k.extend_from_slice(b"entity:");
    k.extend_from_slice(entity_id.as_bytes());
    k
}

/// Lock key for an `:ArtifactContent` row, keyed by its `content_id`
/// (a content hash, globally unique).
#[must_use]
pub(crate) fn content_lock_key(content_id: &str) -> Vec<u8> {
    let mut k = Vec::with_capacity(8 + content_id.len());
    k.extend_from_slice(b"content:");
    k.extend_from_slice(content_id.as_bytes());
    k
}

/// Lock key for a `:Session` row, keyed by its `session_id`.
///
/// Serializes the first-sight get-or-create of the Session node so
/// concurrent ingests with independent `SessionContext`s don't race into
/// a duplicate or an SSI antidependency conflict.
///
/// Hashed in the dedicated *setup* lock table, not the general RMW one —
/// see [`crate::KnowledgeBase::lock_session_setup`].
#[must_use]
pub(crate) fn session_lock_key(session_id: &str) -> Vec<u8> {
    let mut k = Vec::with_capacity(8 + session_id.len());
    k.extend_from_slice(b"session:");
    k.extend_from_slice(session_id.as_bytes());
    k
}

/// Lock key for a `:Participant` row, keyed by its `participant_id`.
///
/// Serializes the first-sight ensure of the Participant node and its
/// `PARTICIPATED_IN` edge against the same antidependency race.
///
/// Hashed in the dedicated *setup* lock table, not the general RMW one —
/// see [`crate::KnowledgeBase::lock_session_setup`].
#[must_use]
pub(crate) fn participant_lock_key(participant_id: &str) -> Vec<u8> {
    let mut k = Vec::with_capacity(12 + participant_id.len());
    k.extend_from_slice(b"participant:");
    k.extend_from_slice(participant_id.as_bytes());
    k
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[tokio::test]
    async fn same_key_serializes() {
        let locks = Arc::new(StripedLocks::new(16));
        let counter = Arc::new(AtomicUsize::new(0));
        let observed_max = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..32 {
            let l = locks.clone();
            let c = counter.clone();
            let m = observed_max.clone();
            handles.push(tokio::spawn(async move {
                let _g = l.lock(b"shared-key").await;
                let v = c.fetch_add(1, Ordering::SeqCst) + 1;
                m.fetch_max(v, Ordering::SeqCst);
                tokio::task::yield_now().await;
                c.fetch_sub(1, Ordering::SeqCst);
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(
            observed_max.load(Ordering::SeqCst),
            1,
            "more than one holder of the same key was observed concurrently",
        );
    }

    #[tokio::test]
    async fn different_keys_can_overlap() {
        // With 256 stripes and only two distinct keys we expect them to
        // map to different stripes (collision probability ~0.4%); rerun
        // the same key pair if a flake ever shows up.
        let locks = Arc::new(StripedLocks::new(256));
        let g1 = locks.lock(b"key-one").await;
        // Should not block even though `g1` is held.
        let g2 = tokio::time::timeout(std::time::Duration::from_millis(50), locks.lock(b"key-two"))
            .await
            .expect("distinct keys must not block on the same stripe");
        drop(g2);
        drop(g1);
    }

    #[test]
    #[should_panic(expected = "at least 1 stripe")]
    fn zero_stripes_panics() {
        let _ = StripedLocks::new(0);
    }

    #[tokio::test]
    async fn lock_many_dedups_colliding_stripes_no_deadlock() {
        // 1 stripe forces every key onto the same stripe — the worst case
        // for the self-deadlock `lock_many` exists to prevent. Acquiring
        // two DISTINCT keys must collapse to a single stripe guard rather
        // than block on the same non-reentrant mutex twice.
        let locks = StripedLocks::new(1);
        let guards = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            locks.lock_many(&[b"session:x".to_vec(), b"participant:y".to_vec()]),
        )
        .await
        .expect("lock_many must not self-deadlock on colliding stripes");
        assert_eq!(
            guards.len(),
            1,
            "colliding keys collapse to one stripe guard"
        );
    }

    #[tokio::test]
    async fn lock_many_distinct_stripes_returns_all() {
        let locks = StripedLocks::new(256);
        let guards = locks
            .lock_many(&[
                b"entity:a".to_vec(),
                b"entity:b".to_vec(),
                b"entity:a".to_vec(),
            ])
            .await;
        // Duplicate key "entity:a" collapses; the two distinct keys
        // *usually* land on different stripes (256 stripes).
        assert!(
            (1..=2).contains(&guards.len()),
            "expected 1-2 stripe guards, got {}",
            guards.len()
        );
    }
}
