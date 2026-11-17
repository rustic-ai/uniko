//! The store keeps the model catalog it registers with uni-db.

use uniko_store::GENERATION_PROVIDERS;
use uniko_store::UnikoError;
use uniko_store::config::UnikoConfig;
use uniko_store::storage::KnowledgeBase;
use uniko_store::xervo::{ModelAliasSpec, ModelTask, WarmupPolicy};

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

fn generate_spec(provider: &str, options: serde_json::Value) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: "gen/test".into(),
        task: ModelTask::Generate,
        provider_id: provider.into(),
        model_id: "some-model".into(),
        revision: None,
        warmup: WarmupPolicy::Eager,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options,
    }
}

/// Every provider in the table must be registered in this build: an
/// entry for a provider whose feature is off would let the facade accept
/// an alias the store then rejects.
#[tokio::test]
async fn every_listed_generation_provider_is_compiled_in() {
    for provider in GENERATION_PROVIDERS {
        let spec = generate_spec(provider, serde_json::Value::Null);
        // Options are provider-specific (vertexai and azure need their
        // own), so only an unknown-provider error counts as a failure.
        if let Err(e) = uniko_store::KnowledgeBase::validate_model_spec(&spec).await {
            let msg = e.to_string();
            assert!(
                !msg.contains("Unknown provider") && !msg.contains("format"),
                "{provider} is listed but not usable: {msg}"
            );
        }
    }
}

#[tokio::test]
async fn validate_model_spec_names_the_alias_on_bad_options() {
    let spec = generate_spec(
        "remote/anthropic",
        serde_json::json!({ "not_an_option": "x" }),
    );
    let err = uniko_store::KnowledgeBase::validate_model_spec(&spec)
        .await
        .unwrap_err();
    assert!(
        matches!(&err, UnikoError::Config(m) if m.contains("'gen/test'") && m.contains("not_an_option")),
        "{err}"
    );

    let unknown = generate_spec("remote/nope", serde_json::Value::Null);
    let err = uniko_store::KnowledgeBase::validate_model_spec(&unknown)
        .await
        .unwrap_err();
    assert!(
        matches!(&err, UnikoError::Config(m) if m.contains("remote/nope")),
        "{err}"
    );
}
