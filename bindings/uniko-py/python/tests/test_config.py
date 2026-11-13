# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Dragonscale Industries Inc.
"""Issue #45: configuring an instance from Python, and reading it back."""

from __future__ import annotations

import pytest

import uniko
from uniko.models import RerankerConfigModel, UnikoConfigModel
from uniko.models.typed import TypedUniko


def build(builder: uniko.UnikoBuilder) -> uniko.Uniko:
    return builder.in_memory().build_sync()


def test_partial_config_merges_onto_the_defaults() -> None:
    default = uniko.Uniko.in_memory_sync().config()
    cfg = build(
        uniko.Uniko.builder().config(
            {"recall_limit": 7, "embedding": {"batch_size": 8}}
        )
    ).config()
    assert cfg["recall_limit"] == 7
    assert cfg["embedding"]["batch_size"] == 8
    # Everything not named keeps its default, nested sections included.
    assert cfg["embedding"]["model_id"] == default["embedding"]["model_id"]
    assert cfg["recall_token_budget"] == default["recall_token_budget"]


def test_config_reads_back_in_the_shape_it_accepts() -> None:
    cfg = uniko.Uniko.in_memory_sync().config()
    assert {"embedding", "reranker", "nlp", "ocr", "recall_limit"} <= cfg.keys()
    # Feeding the read-back in again is accepted unchanged.
    again = build(uniko.Uniko.builder().config(cfg)).config()
    assert again == cfg


@pytest.mark.parametrize(
    ("config", "path"),
    [
        ({"recal_limit": 5}, "recal_limit"),
        ({"embedding": {"batchsize": 8}}, "embedding.batchsize"),
        ({"retry_max_attempts": 3}, "retry_max_attempts"),
    ],
)
def test_unknown_keys_raise_config_error(config: dict, path: str) -> None:
    with pytest.raises(uniko.ConfigError, match=f"'{path}'"):
        uniko.Uniko.builder().config(config)


def test_a_near_miss_key_gets_a_suggestion() -> None:
    with pytest.raises(uniko.ConfigError, match="did you mean 'recall_limit'"):
        uniko.Uniko.builder().config({"recal_limit": 5})


def test_a_section_given_a_scalar_raises() -> None:
    with pytest.raises(uniko.ConfigError, match="embedding: expected a mapping"):
        uniko.Uniko.builder().config({"embedding": "minilm"})


def test_invalid_values_raise_at_build_not_at_set() -> None:
    builder = uniko.Uniko.builder().config(
        {"recall_limit": 100, "reranker": {"enabled": True, "top_n": 10}}
    )
    with pytest.raises(uniko.ConfigError, match="top_n"):
        build(builder)


def test_wrong_value_types_raise_config_error() -> None:
    with pytest.raises(uniko.ConfigError, match="'recall_limit'"):
        uniko.Uniko.builder().config({"recall_limit": "many"})
    with pytest.raises(uniko.ConfigError, match="'reranker.top_n'"):
        uniko.Uniko.builder().config({"reranker": {"top_n": -1}})


def test_embedding_preset_and_dict() -> None:
    cfg = build(uniko.Uniko.builder().embedding("minilm")).config()
    assert cfg["embedding"]["dimensions"] == 384
    assert "MiniLM" in cfg["embedding"]["model_id"]

    cfg = build(uniko.Uniko.builder().embedding({"batch_size": 4})).config()
    assert cfg["embedding"]["batch_size"] == 4

    with pytest.raises(uniko.ConfigError, match="unknown embedding preset"):
        uniko.Uniko.builder().embedding("word2vec")


@pytest.mark.parametrize("reranker_first", [True, False])
def test_reranker_override_wins_in_either_order(reranker_first: bool) -> None:
    builder = uniko.Uniko.builder()
    base = {"reranker": {"enabled": True}}
    if reranker_first:
        builder = builder.reranker(None).config(base)
    else:
        builder = builder.config(base).reranker(None)
    assert build(builder).config()["reranker"]["enabled"] is False


def test_reranker_presets() -> None:
    cfg = build(uniko.Uniko.builder().reranker("minilm")).config()
    assert cfg["reranker"]["enabled"] is True
    with pytest.raises(uniko.ConfigError, match="unknown reranker preset"):
        uniko.Uniko.builder().reranker("colbert-xxl")


def test_provider_options_are_redacted_on_read_back() -> None:
    # A remote embedder, never contacted: models load lazily on first use.
    embedding = {
        "provider": "remote/llamacpp",
        "model_id": "remote-embedder",
        "provider_options": {
            "base_url": "http://127.0.0.1:9/secret-path/v1",
            "embedding_dimensions": 384,
            "max_input_tokens": 512,
        },
    }
    memory = uniko.Uniko.builder().in_memory().embedding(embedding).build_sync()
    cfg = memory.config()
    assert cfg["embedding"]["provider_options"] == "<redacted>"
    assert "secret-path" not in repr(cfg)
    assert (
        uniko.Uniko.in_memory_sync().config()["embedding"]["provider_options"] is None
    )


def test_config_model_partial_input() -> None:
    model = UnikoConfigModel(recall_limit=9, reranker=RerankerConfigModel(top_n=60))
    cfg = build(uniko.Uniko.builder().config(model)).config()
    assert cfg["recall_limit"] == 9
    assert cfg["reranker"]["top_n"] == 60
    # Fields left unset on the model keep the engine default.
    assert cfg["reranker"]["model_id"] == UnikoConfigModel().reranker.model_id


def test_config_model_rejects_unknown_keys() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        UnikoConfigModel.model_validate({"recal_limit": 5})


def test_typed_builder_config_and_read_back() -> None:
    uni = (
        TypedUniko.builder()
        .in_memory()
        .config(UnikoConfigModel(recall_limit=11))
        .reranker(None)
        .build_sync()
    )
    cfg = uni.config()
    assert isinstance(cfg, UnikoConfigModel)
    assert cfg.recall_limit == 11
    assert cfg.reranker.enabled is False


def test_streaming_pipeline_uses_the_configured_queue_capacities() -> None:
    memory = (
        uniko.Uniko.builder()
        .in_memory()
        .config({"ingest_queue_capacity": 13, "consolidation_queue_capacity": 3})
        .streaming(True)
        .build_sync()
    )
    health = memory.health()
    assert isinstance(health, uniko.PipelineHealth)
    assert health.ingest.queue_capacity == 13
    assert health.consolidation.queue_capacity == 3
    assert health.ingest.status in ("healthy", "degraded", "backpressured", "stalled")
    assert health.llm_circuit in ("closed", "open", "half_open")
    memory.shutdown_sync()


def test_health_is_none_without_streaming() -> None:
    assert uniko.Uniko.in_memory_sync().health() is None
    assert TypedUniko.in_memory_sync().health() is None


def test_zero_queue_capacity_is_rejected_at_build() -> None:
    builder = uniko.Uniko.builder().config({"ingest_queue_capacity": 0})
    with pytest.raises(uniko.ConfigError, match="ingest_queue_capacity"):
        build(builder)


def test_reopen_with_a_different_embedding_raises(tmp_path) -> None:
    store = str(tmp_path / "store")
    uniko.Uniko.builder().path(store).build_sync().shutdown_sync()
    builder = uniko.Uniko.builder().path(store).embedding({"model_id": "SomethingElse"})
    with pytest.raises(uniko.ConfigError, match="embedding mismatch"):
        builder.build_sync()
