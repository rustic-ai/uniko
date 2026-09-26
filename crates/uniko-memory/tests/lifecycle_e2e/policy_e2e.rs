//! Integration tests for the access-control policy (F66).

use std::collections::HashMap;

use uni_db::Value;

use uniko_memory::policy::{Viewer, filter_bundle, visibility_admits};
use uniko_memory::recall::{
    ContextBundle, RecallConfig, RecallItem, RecallKind, ViewerScope, recall,
};
use uniko_store::KnowledgeBase;
use uniko_store::config::UnikoConfig;
use uniko_store::schema::constants::{edges, labels};

async fn kb() -> KnowledgeBase {
    KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB")
}

async fn mk_participant(kb: &KnowledgeBase, pid: &str) -> i64 {
    let mut props = HashMap::new();
    props.insert("kind".into(), Value::String("agent".into()));
    kb.merge_node(labels::PARTICIPANT, "participant_id", pid, &props)
        .await
        .expect("participant")
}

async fn mk_team(kb: &KnowledgeBase, tid: &str) -> i64 {
    let mut props = HashMap::new();
    props.insert("name".into(), Value::String(tid.into()));
    kb.merge_node(labels::TEAM, "team_id", tid, &props)
        .await
        .expect("team")
}

async fn mk_org(kb: &KnowledgeBase, oid: &str) -> i64 {
    let mut props = HashMap::new();
    props.insert("name".into(), Value::String(oid.into()));
    kb.merge_node(labels::ORGANIZATION, "org_id", oid, &props)
        .await
        .expect("org")
}

async fn mk_fact(kb: &KnowledgeBase, fid: &str, visibility: Option<&str>) -> i64 {
    let mut props = HashMap::new();
    props.insert("subject".into(), Value::String("s".into()));
    props.insert("predicate".into(), Value::String("p".into()));
    props.insert("object".into(), Value::String("o".into()));
    if let Some(v) = visibility {
        props.insert("visibility".into(), Value::String(v.into()));
    }
    kb.merge_node(labels::FACT, "fact_id", fid, &props)
        .await
        .expect("fact")
}

fn bundle_with(items: Vec<RecallItem>) -> ContextBundle {
    ContextBundle {
        items,
        total_tokens: 0,
        phase1_only: false,
        phase2_only: false,
        coverage: 0.0,
    }
}

fn fact_item(node_id: i64) -> RecallItem {
    RecallItem {
        node_id,
        kind: RecallKind::Fact,
        score: 1.0,
        content: "x".into(),
        sources: Vec::new(),
        category: None,
        source_id: None,
        revision_id: None,
    }
}

#[test]
fn visibility_admits_smoke() {
    let v = Viewer::from_parts("alice", ["red".to_string()], ["acme".to_string()]);
    assert!(visibility_admits(None, &v));
    assert!(visibility_admits(Some("public"), &v));
    assert!(visibility_admits(Some("private:alice"), &v));
    assert!(!visibility_admits(Some("private:bob"), &v));
    assert!(visibility_admits(Some("team:red"), &v));
    assert!(!visibility_admits(Some("team:blue"), &v));
    assert!(visibility_admits(Some("org:acme"), &v));
    assert!(!visibility_admits(Some("org:other"), &v));
}

#[tokio::test]
async fn viewer_new_resolves_memberships() {
    let kb = kb().await;
    let alice = mk_participant(&kb, "alice").await;
    let team = mk_team(&kb, "red").await;
    let org = mk_org(&kb, "acme").await;
    kb.create_edge(edges::PART_OF_TEAM, alice, team, &HashMap::new())
        .await
        .expect("PART_OF_TEAM");
    kb.create_edge(edges::TEAM_IN_ORG, team, org, &HashMap::new())
        .await
        .expect("TEAM_IN_ORG");

    let v = Viewer::new(&kb, "alice").await.expect("viewer");
    assert!(v.teams.contains("red"));
    assert!(v.orgs.contains("acme"));
}

#[tokio::test]
async fn filter_bundle_excludes_private_to_other_participant() {
    let kb = kb().await;
    let _alice = mk_participant(&kb, "alice").await;
    let _bob = mk_participant(&kb, "bob").await;

    let public_fact = mk_fact(&kb, "f_pub", None).await;
    let private_to_alice = mk_fact(&kb, "f_alice", Some("private:alice")).await;
    let private_to_bob = mk_fact(&kb, "f_bob", Some("private:bob")).await;

    let bob_viewer = Viewer::new(&kb, "bob").await.expect("viewer");
    let mut bundle = bundle_with(vec![
        fact_item(public_fact),
        fact_item(private_to_alice),
        fact_item(private_to_bob),
    ]);
    filter_bundle(&kb, &mut bundle, &bob_viewer)
        .await
        .expect("filter");

    let visible_ids: Vec<i64> = bundle.items.iter().map(|i| i.node_id).collect();
    assert!(visible_ids.contains(&public_fact));
    assert!(!visible_ids.contains(&private_to_alice));
    assert!(visible_ids.contains(&private_to_bob));
}

#[tokio::test]
async fn filter_bundle_leaves_non_policy_items_alone() {
    let kb = kb().await;
    let _ = mk_participant(&kb, "alice").await;
    let alice = Viewer::new(&kb, "alice").await.expect("viewer");

    // Build a bundle with a non-policy item (Message) and a Fact with
    // a visibility the viewer can't see.  The Message must survive.
    let private_fact = mk_fact(&kb, "f_priv", Some("private:bob")).await;
    let mut bundle = bundle_with(vec![
        fact_item(private_fact),
        RecallItem {
            node_id: 999,
            kind: RecallKind::Message,
            score: 0.5,
            content: "x".into(),
            sources: Vec::new(),
            category: None,
            source_id: None,
            revision_id: None,
        },
    ]);
    filter_bundle(&kb, &mut bundle, &alice)
        .await
        .expect("filter");
    let kinds: Vec<RecallKind> = bundle.items.iter().map(|i| i.kind).collect();
    assert_eq!(kinds, vec![RecallKind::Message]);
}

/// Seed a Fact with distinctive `object` text plus a visibility scope.
async fn mk_fact_with_object(
    kb: &KnowledgeBase,
    fid: &str,
    object: &str,
    visibility: Option<&str>,
) -> i64 {
    let mut props = HashMap::new();
    props.insert("subject".into(), Value::String("project".into()));
    props.insert("predicate".into(), Value::String("status_is".into()));
    props.insert("object".into(), Value::String(object.into()));
    if let Some(v) = visibility {
        props.insert("visibility".into(), Value::String(v.into()));
    }
    kb.merge_node(labels::FACT, "fact_id", fid, &props)
        .await
        .expect("fact")
}

/// #5: `recall()` must apply access control when given a viewer. A
/// `private:alice` Fact that surfaces for an unrestricted recall must be
/// excluded when the same recall runs as viewer `bob`; a public Fact
/// stays. Proves recall() actually invokes `filter_bundle` (the fail-open
/// gap). Skips if the recall infra can't surface Facts in this env.
#[tokio::test]
async fn recall_applies_viewer_scope_filter() {
    let kb = kb().await;
    let _alice = mk_participant(&kb, "alice").await;
    let _bob = mk_participant(&kb, "bob").await;

    let query = "quarterly revenue outlook";
    mk_fact_with_object(&kb, "f-public", query, Some("public")).await;
    mk_fact_with_object(&kb, "f-private", query, Some("private:alice")).await;

    let unrestricted = RecallConfig {
        viewer: ViewerScope::Unrestricted,
        ..RecallConfig::default()
    };
    let res = recall(&kb, query, &unrestricted).await;
    if matches!(res, Err(uniko_store::UnikoError::Embedding(_))) {
        eprintln!("skipping: embedding unavailable");
        return;
    }
    let open_bundle = res.expect("unrestricted recall");
    let private_nid = fact_id(&kb, "f-private").await;
    let private_in_open = open_bundle.items.iter().any(|i| i.node_id == private_nid);
    if !private_in_open {
        eprintln!("skipping: recall did not surface Facts in this env (no embeddings/index)");
        return;
    }

    // As viewer bob, the private:alice fact must be filtered out; public stays.
    let public_nid = fact_id(&kb, "f-public").await;
    let scoped = RecallConfig {
        viewer: ViewerScope::As(Viewer::new(&kb, "bob").await.expect("viewer")),
        ..RecallConfig::default()
    };
    let scoped_bundle = recall(&kb, query, &scoped).await.expect("scoped recall");
    assert!(
        !scoped_bundle.items.iter().any(|i| i.node_id == private_nid),
        "private:alice Fact must be excluded for viewer bob"
    );
    assert!(
        scoped_bundle.items.iter().any(|i| i.node_id == public_nid),
        "public Fact must remain visible to viewer bob"
    );
}

/// Resolve a Fact's internal node id by its `fact_id`.
async fn fact_id(kb: &KnowledgeBase, fid: &str) -> i64 {
    kb.get_node_by_ext_id(labels::FACT, "fact_id", fid)
        .await
        .expect("lookup")
        .expect("fact exists")
        .0
}
