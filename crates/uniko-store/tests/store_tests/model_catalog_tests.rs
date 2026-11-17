//! The store keeps the model catalog it registers with uni-db.

use uniko_store::UnikoError;
use uniko_store::config::UnikoConfig;
use uniko_store::storage::KnowledgeBase;
use uniko_store::xervo::ModelTask;

#[tokio::test]
async fn in_memory_store_reports_its_catalog() {
    let kb = KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB");

    let entries = kb.model_entries().expect("catalog is known");
    let embed = entries
        .iter()
        .find(|e| e.alias == "embed/default")
        .expect("embed/default is registered");
    assert_eq!(embed.task, ModelTask::Embed);
    assert!(!embed.model_id.is_empty());
    assert!(entries.iter().any(|e| e.task == ModelTask::Nlp));

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn ensure_generator_rejects_a_non_generation_alias_by_task() {
    let kb = KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB");

    let err = kb.ensure_generator("embed/default").await.unwrap_err();
    assert!(
        matches!(&err, UnikoError::Config(m) if m.contains("not a generation model")),
        "{err}"
    );

    kb.shutdown().await.unwrap();
}
