//! Programmatic verification that schema registration is complete.

use uni_db::Uni;
use uniko_store::config::UnikoConfig;
use uniko_store::schema::constants::{edges, labels};
use uniko_store::schema::register_schema;
use uniko_store::storage::embed_catalog;

async fn test_db() -> Uni {
    let config = UnikoConfig::default();
    let db = Uni::in_memory()
        .xervo_catalog(embed_catalog(&config))
        .build()
        .await
        .expect("in-memory db");
    register_schema(&db, &config)
        .await
        .expect("schema registration");
    db
}

// ── Counts ──

#[test]
fn test_label_count() {
    // Completeness tripwire: bump when a label is added to `labels::ALL`
    // (the array is the source of truth). Currently 26 node labels (added
    // SOURCE for typed record provenance, issue #39/#41).
    assert_eq!(labels::ALL.len(), 26, "expected 26 node labels");
}

#[test]
fn test_edge_count() {
    // Completeness tripwire: bump when an edge is added to `edges::ALL`
    // (the array is the source of truth). Currently 54 edge types — grown
    // from the earlier 48 via REINFORCED, ATTACHED_TO, CONTRADICTED_BY, and
    // later additions, and FROM_SOURCE for typed record provenance.
    assert_eq!(edges::ALL.len(), 55, "expected 55 edge types");
}

#[test]
fn test_no_duplicate_labels() {
    let mut seen = std::collections::HashSet::new();
    for label in labels::ALL {
        assert!(seen.insert(label), "duplicate label: {label}");
    }
}

#[test]
fn test_no_duplicate_edges() {
    let mut seen = std::collections::HashSet::new();
    for edge in edges::ALL {
        assert!(seen.insert(edge), "duplicate edge: {edge}");
    }
}

// ── DB verification ──

#[tokio::test]
async fn test_every_label_exists_in_db() {
    let db = test_db().await;
    for label in labels::ALL {
        assert!(
            db.label_exists(label).await.unwrap(),
            "label {label} not registered in DB"
        );
    }
    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_every_edge_exists_in_db() {
    let db = test_db().await;
    for edge in edges::ALL {
        assert!(
            db.edge_type_exists(edge).await.unwrap(),
            "edge type {edge} not registered in DB"
        );
    }
    db.shutdown().await.unwrap();
}

// ── Property verification for key node types ──

#[tokio::test]
async fn test_participant_properties() {
    let db = test_db().await;
    let info = db
        .get_label_info("Participant")
        .await
        .unwrap()
        .expect("Participant label must exist");
    let prop_names: Vec<&str> = info.properties.iter().map(|p| p.name.as_str()).collect();
    assert!(prop_names.contains(&"participant_id"));
    assert!(prop_names.contains(&"kind"));
    assert!(prop_names.contains(&"name"));
    assert!(prop_names.contains(&"capabilities"));
    assert!(prop_names.contains(&"metadata"));
    assert!(prop_names.contains(&"first_seen"));
    assert!(prop_names.contains(&"last_seen"));
    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_fact_has_btic_property() {
    let db = test_db().await;
    let info = db
        .get_label_info("Fact")
        .await
        .unwrap()
        .expect("Fact label must exist");
    let prop_names: Vec<&str> = info.properties.iter().map(|p| p.name.as_str()).collect();
    assert!(
        prop_names.contains(&"valid_at"),
        "Fact must have valid_at BTIC property"
    );
    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_artifact_has_five_embeddings() {
    let db = test_db().await;
    let info = db
        .get_label_info("Artifact")
        .await
        .unwrap()
        .expect("Artifact label must exist");
    let prop_names: Vec<&str> = info.properties.iter().map(|p| p.name.as_str()).collect();
    assert!(prop_names.contains(&"text_embedding"));
    assert!(prop_names.contains(&"image_embedding"));
    assert!(prop_names.contains(&"audio_embedding"));
    assert!(prop_names.contains(&"video_embedding"));
    assert!(prop_names.contains(&"multimodal_embedding"));
    db.shutdown().await.unwrap();
}

// ── Spec cross-reference: required labels ──

#[test]
fn test_spec_required_labels() {
    let required = [
        "Participant",
        "Goal",
        "Task",
        "Session",
        "Message",
        "Action",
        "Episode",
        "Artifact",
        "ArtifactContent",
        "Chunk",
        "Entity",
        "Observation",
        "Fact",
        "Topic",
        "Summary",
        "Procedure",
        "Rule",
        "ConsolidationCycle",
        "DeadLetter",
        "Organization",
        "Team",
        "KnowledgeBaseStats",
    ];
    for label in &required {
        assert!(
            labels::ALL.contains(label),
            "spec-required label {label} missing from constants::labels::ALL"
        );
    }
}

#[test]
fn test_spec_required_edges() {
    let required = [
        "OWNED_BY",
        "PARENT_GOAL",
        "PART_OF",
        "ASSIGNED_TO",
        "DEPENDS_ON",
        "SUBTASK_OF",
        "FOR_TASK",
        "FOR_GOAL",
        "PARTICIPATED_IN",
        "SENT_BY",
        "ADDRESSED_TO",
        "IN_SESSION",
        "NEXT",
        "PERFORMED_BY",
        "TRIGGERED_BY",
        "PRODUCED",
        "NEXT_ACTION",
        "RECORDED_BY",
        "INVOLVES",
        "MENTIONS",
        "FOLLOWED_BY",
        "HAS_CHUNK",
        "HAS_CONTENT",
        "CREATED_BY",
        "MODIFIED_BY",
        "OBSERVED_IN",
        "OBSERVED_DURING",
        "ABOUT",
        "SUPPORTED_BY",
        "DERIVED_BY",
        "DERIVED_FROM",
        "INVALIDATES",
        "SHARED_FROM",
        "BELONGS_TO",
        "SUMMARIZES",
        "OPERATES_ON",
        "USED_IN",
        "SUPERSEDES",
        "COVERS",
        "PROCESSED",
        "INVOLVED",
        "CREATED",
        "INVALIDATED",
        "PROMOTED",
        "APPLIED_RULE",
        "MEMBER_OF",
        "PART_OF_TEAM",
        "TEAM_IN_ORG",
    ];
    for edge in &required {
        assert!(
            edges::ALL.contains(edge),
            "spec-required edge {edge} missing from constants::edges::ALL"
        );
    }
}
