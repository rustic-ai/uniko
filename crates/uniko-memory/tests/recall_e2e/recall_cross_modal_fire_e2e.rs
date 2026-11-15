//! Proves the cross-modal image channel fires when (a) the corpus has
//! image content, (b) the RecallConfig toggle is on, and (c) the query
//! supplies an `image_vec` — and stays dormant when any one of those
//! preconditions fails.
//!
//! Companion to [`recall_modality_lazy_e2e.rs`] which proves the
//! dormant case in a text-only corpus.
//
use uni_db::Value;
use uniko_memory::recall::RecallConfig;
use uniko_memory::recall::intent::{IntentProfile, QueryModalities};
use uniko_memory::recall::modality::RecallCounters;
use uniko_memory::recall::phase2_expand;
use uniko_store::KnowledgeBase;
use uniko_store::config::UnikoConfig;

/// Embedding dimension of the default schema — Artifact's `*_embedding`
/// columns are sized to `UnikoConfig::default().embedding.dimensions`, so
/// derive it rather than hardcoding (which silently breaks if the default
/// embedder changes).
fn dim() -> usize {
    UnikoConfig::default().embedding.dimensions
}

/// Unit-norm vector with `1.0` at `slot` and `0.0` elsewhere.
fn one_hot(slot: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; dim()];
    v[slot] = 1.0;
    v
}

async fn seed_image_artifact(kb: &KnowledgeBase, aid: &str, vec: Vec<f32>) -> i64 {
    let session = kb.db().session();
    let tx = session.tx().await.unwrap();
    let r = tx
        .query_with(
            "CREATE (a:Artifact {artifact_id: $aid, kind: 'image', \
             created_at: datetime(), image_embedding: $vec}) RETURN id(a) AS vid",
        )
        .param("aid", aid)
        .param("vec", Value::Vector(vec))
        .fetch_all()
        .await
        .unwrap();
    tx.commit().await.unwrap();
    r.rows().first().unwrap().get("vid").unwrap()
}

#[tokio::test]
async fn image_channel_fires_when_all_preconditions_met() {
    let kb = KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("KB");

    let hit_id = seed_image_artifact(&kb, "img-hit", one_hot(0)).await;
    let _miss_id = seed_image_artifact(&kb, "img-miss", one_hot(1)).await;

    // Flip the corpus-side modality_presence flag — Track B will do this
    // from ingest; here we set it directly to model an indexed corpus.
    kb.bump_modality_presence("image").await.unwrap();

    // Intent: bag-of-text variants empty (no other phase 2 channels
    // would fire) but image_vec set → only the image channel can hit.
    let intent = IntentProfile {
        variants: Vec::new(),
        entity_refs: Vec::new(),
        facet_count: 1,
        expected_answer_type: None,
        temporal_window: None,
        query_modalities: QueryModalities {
            has_image_input: true,
            has_audio_input: false,
            image_vec: Some(one_hot(0)),
            audio_vec: None,
        },
        literals: Vec::new(),
    };

    let cfg = RecallConfig {
        enable_image_channel: true,
        // Keep min_score sane for cosine on one-hot vectors.
        min_score: 0.0,
        ..Default::default()
    };

    let counters = RecallCounters::new();
    let items = phase2_expand(&kb, &intent, &cfg, Some(counters.clone())).await;
    assert_eq!(
        counters.image(),
        1,
        "image channel should have fired exactly once"
    );
    assert_eq!(counters.audio(), 0);
    assert_eq!(counters.multimodal(), 0);
    assert!(
        !items.is_empty(),
        "phase2_expand should return at least the matching artifact"
    );
    // Top hit should be the artifact whose image_embedding aligns with
    // the query vec (slot 0).
    let top = &items[0];
    assert_eq!(
        top.node_id, hit_id,
        "expected slot-0 artifact ({hit_id}) at top, got {top:?}"
    );

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn image_channel_dormant_when_toggle_off() {
    let kb = KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("KB");
    seed_image_artifact(&kb, "img-a", one_hot(0)).await;
    kb.bump_modality_presence("image").await.unwrap();

    let intent = IntentProfile {
        variants: Vec::new(),
        entity_refs: Vec::new(),
        facet_count: 1,
        expected_answer_type: None,
        temporal_window: None,
        query_modalities: QueryModalities {
            has_image_input: true,
            has_audio_input: false,
            image_vec: Some(one_hot(0)),
            audio_vec: None,
        },
        literals: Vec::new(),
    };

    let cfg = RecallConfig::default(); // toggle off
    let counters = RecallCounters::new();
    let _ = phase2_expand(&kb, &intent, &cfg, Some(counters.clone())).await;
    assert_eq!(
        counters.total(),
        0,
        "no cross-modal channel should fire when toggle is off"
    );

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn image_channel_dormant_when_no_query_vec() {
    let kb = KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("KB");
    seed_image_artifact(&kb, "img-a", one_hot(0)).await;
    kb.bump_modality_presence("image").await.unwrap();

    let intent = IntentProfile {
        variants: Vec::new(),
        entity_refs: Vec::new(),
        facet_count: 1,
        expected_answer_type: None,
        temporal_window: None,
        // image_vec = None — the channel must NOT fire even with both
        // corpus-side and config-side gates open.
        query_modalities: QueryModalities::default(),
        literals: Vec::new(),
    };

    let cfg = RecallConfig {
        enable_image_channel: true,
        ..Default::default()
    };

    let counters = RecallCounters::new();
    let _ = phase2_expand(&kb, &intent, &cfg, Some(counters.clone())).await;
    assert_eq!(counters.total(), 0);

    kb.shutdown().await.unwrap();
}
