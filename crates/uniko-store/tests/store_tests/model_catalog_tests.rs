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

fn embed_spec(alias: &str, model_id: &str) -> ModelAliasSpec {
    ModelAliasSpec {
        alias: alias.into(),
        task: ModelTask::Embed,
        provider_id: "local/onnx".into(),
        model_id: model_id.into(),
        revision: None,
        warmup: WarmupPolicy::Eager,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::Value::Null,
    }
}

fn status_of<'a>(
    statuses: &'a [uniko_store::ModelStatus],
    alias: &str,
) -> &'a uniko_store::ModelStatus {
    statuses
        .iter()
        .find(|s| s.alias == alias)
        .unwrap_or_else(|| panic!("{alias} missing from model_status"))
}

#[tokio::test]
async fn prefetch_records_each_alias_and_keeps_going_past_a_failure() {
    use uniko_store::ModelState;

    // Extra aliases go after the built-in ones, so a second good alias
    // follows the bad one: a fail-fast prefetch would never load it.
    let mut config = UnikoConfig::default();
    config.reranker.enabled = false;
    let mut second = uniko_store::storage::embed_catalog(&config)
        .into_iter()
        .find(|s| s.alias == "embed/default")
        .unwrap();
    second.alias = "embed/second".into();
    let extra = vec![
        embed_spec("embed/missing", "uniko-test/no-such-model-repo"),
        second,
    ];
    let kb = KnowledgeBase::in_memory_with_xervo(config, extra)
        .await
        .expect("a failed optional model must not stop the store opening");

    let statuses = kb.model_status().await.expect("catalog is known");
    let entries = kb.model_entries().unwrap();
    assert_eq!(statuses.len(), entries.len());
    for (status, entry) in statuses.iter().zip(&entries) {
        assert_eq!(status.alias, entry.alias);
        assert_eq!(status.task, entry.task);
    }
    assert!(
        matches!(
            status_of(&statuses, "embed/missing").state,
            ModelState::Failed(_)
        ),
        "{:?}",
        status_of(&statuses, "embed/missing").state
    );
    assert_eq!(
        status_of(&statuses, "embed/default").state,
        ModelState::Loaded
    );
    assert_eq!(
        status_of(&statuses, "embed/second").state,
        ModelState::Loaded
    );

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_store_opened_without_prefetch_reports_models_not_loaded() {
    use uniko_store::ModelState;

    let dir = tempfile::tempdir().unwrap();
    let kb =
        KnowledgeBase::open_with_xervo_no_prefetch(dir.path(), UnikoConfig::default(), Vec::new())
            .await
            .unwrap();

    let statuses = kb.model_status().await.unwrap();
    assert!(!statuses.is_empty());
    assert!(
        statuses.iter().all(|s| s.state == ModelState::NotLoaded),
        "{statuses:?}"
    );

    // A successful call through the store marks the alias loaded.
    kb.embed("embed/default", &["hello"]).await.unwrap();
    let statuses = kb.model_status().await.unwrap();
    assert_eq!(
        status_of(&statuses, "embed/default").state,
        ModelState::Loaded
    );

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn model_status_redacts_endpoints_and_never_shows_options() {
    let spec = ModelAliasSpec {
        alias: "gen/remote".into(),
        warmup: WarmupPolicy::Lazy,
        ..generate_spec(
            "remote/openai",
            serde_json::json!({
                "api_key_env": "UNIKO_TEST_KEY_VAR",
                "base_url": "https://alice:hunter2@llm.example.com/v1?sig=zzz",
            }),
        )
    };
    let dir = tempfile::tempdir().unwrap();
    let kb =
        KnowledgeBase::open_with_xervo_no_prefetch(dir.path(), UnikoConfig::default(), vec![spec])
            .await
            .unwrap();

    let statuses = kb.model_status().await.unwrap();
    let remote = status_of(&statuses, "gen/remote");
    assert_eq!(remote.api_key_env.as_deref(), Some("UNIKO_TEST_KEY_VAR"));
    assert_eq!(
        remote.base_url.as_deref(),
        Some("https://llm.example.com/v1")
    );
    for rendered in [
        format!("{statuses:?}"),
        serde_json::to_string(&statuses).unwrap(),
    ] {
        assert!(!rendered.contains("hunter2"), "{rendered}");
        assert!(!rendered.contains("sig=zzz"), "{rendered}");
    }

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_store_over_a_shared_runtime_has_no_model_status() {
    let config = UnikoConfig::default();
    let runtime = KnowledgeBase::build_shared_runtime(&config, &[])
        .await
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let kb = KnowledgeBase::open_with_runtime(dir.path(), config, runtime)
        .await
        .unwrap();
    assert!(kb.model_status().await.is_none());
    kb.shutdown().await.unwrap();
}
