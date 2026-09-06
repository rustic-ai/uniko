//! Regression tests for the session-setup lock domain (issue #36).
//!
//! `lock_session_setup` guards are held across `merge_node`, which takes
//! its own `node:…` RMW stripe. When both came from one striped table, an
//! outer setup key and the nested node key that hashed to the same stripe
//! deadlocked the task on a non-reentrant mutex it already held. Forcing
//! one stripe per table makes that collision certain rather than ~0.4%
//! likely, so these fail deterministically on the old code.

use std::collections::HashMap;
use std::time::Duration;

use uni_db::Value;
use uniko_store::config::UnikoConfig;
use uniko_store::storage::KnowledgeBase;

async fn single_stripe_kb() -> KnowledgeBase {
    KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB")
        .with_lock_stripes(1)
}

#[tokio::test]
async fn merge_node_under_session_setup_guards_does_not_deadlock() {
    let kb = single_stripe_kb().await;

    let guards = kb.lock_session_setup("sess-1", "assistant").await;

    let mut props = HashMap::new();
    props.insert("name".into(), Value::String("assistant".into()));
    props.insert("kind".into(), Value::String("unknown".into()));

    let nid = tokio::time::timeout(
        Duration::from_secs(30),
        kb.merge_node("Participant", "participant_id", "assistant", &props),
    )
    .await
    .expect("merge_node must not block on a stripe the setup guards hold")
    .expect("merge_node must succeed");
    assert_ne!(nid, 0, "participant node must be created");

    // The update branch of the same merge, still under the setup guards.
    let again = tokio::time::timeout(
        Duration::from_secs(30),
        kb.merge_node("Participant", "participant_id", "assistant", &props),
    )
    .await
    .expect("repeat merge_node must not block either")
    .expect("repeat merge_node must succeed");
    assert_eq!(again, nid, "participant must not be duplicated");

    drop(guards);
    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn session_setup_still_serializes_concurrent_holders() {
    // Separating the domains must not weaken the mutual exclusion the
    // setup guards exist for: two holders of the same session/participant
    // keys still cannot overlap.
    let kb = single_stripe_kb().await;

    let first = kb.lock_session_setup("sess-1", "assistant").await;
    let blocked = tokio::time::timeout(
        Duration::from_millis(250),
        kb.lock_session_setup("sess-1", "assistant"),
    )
    .await;
    assert!(
        blocked.is_err(),
        "a second holder of the same setup keys must block until the first releases",
    );
    drop(first);

    let acquired = tokio::time::timeout(
        Duration::from_secs(5),
        kb.lock_session_setup("sess-1", "assistant"),
    )
    .await;
    assert!(
        acquired.is_ok(),
        "the setup keys must be acquirable once the first holder releases",
    );
    drop(blocked);
    drop(acquired);

    kb.shutdown().await.unwrap();
}
