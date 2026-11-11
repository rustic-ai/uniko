# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Dragonscale Industries Inc.
"""to_native() adapter tests — specs build the correct native builders, and the
native import stays lazy (specs/schemas work without touching the engine)."""

from __future__ import annotations

import pytest

from uniko.models import (
    IngestSourceSpec,
    LlmSpecAdapter,
    ScopeSpec,
    TurnSpec,
    llm_to_native,
)
from uniko.models import inputs as inputs_mod


def test_turn_spec_to_native_builds_turn():
    turn = TurnSpec(
        sender_id="alice",
        content="hi",
        message_id="m1",
        content_type="text",
        addressed_to=["bob"],
        metadata={"topic": "greeting"},
    ).to_native()
    assert repr(turn).startswith("Turn")


def test_turn_spec_to_native_with_attachments():
    turn = TurnSpec(
        sender_id="a",
        content="see attached",
        attachments=[IngestSourceSpec.from_text("doc body", mime="text/plain")],
    ).to_native()
    assert repr(turn).startswith("Turn")


def test_scope_spec_to_native_builds_scope():
    scope = ScopeSpec(sessions=["s1"], participants=["p1"]).to_native()
    assert repr(scope).startswith("Scope")


def test_ingest_source_to_native_each_variant():
    assert repr(IngestSourceSpec.from_text("hi").to_native()).startswith("IngestSource")
    assert repr(IngestSourceSpec.from_bytes(b"\x00\x01").to_native()).startswith(
        "IngestSource"
    )
    assert repr(IngestSourceSpec.from_path("/tmp/x").to_native()).startswith(
        "IngestSource"
    )


def test_ingest_source_to_native_modifiers_chain():
    src = IngestSourceSpec.from_text("hi", mime="text/plain", id="a1").to_native()
    assert repr(src).startswith("IngestSource")


def test_llm_to_native_each_provider():
    for payload in (
        {"provider": "openai", "alias": "a", "model_id": "gpt-4o-mini"},
        {
            "provider": "openai_with_key_env",
            "alias": "a",
            "model_id": "m",
            "key_env": "K",
        },
    ):
        spec = LlmSpecAdapter.validate_python(payload)
        assert repr(llm_to_native(spec)).startswith("LlmSpec")


def test_native_import_is_lazy(monkeypatch):
    """Specs and schemas must work without touching the native extension; only
    to_native() reaches for it."""

    def poisoned():
        raise RuntimeError("native extension touched")

    monkeypatch.setattr(inputs_mod, "_native", poisoned)

    # Construction + schema generation must NOT call _native.
    spec = TurnSpec(sender_id="a", content="hi")
    assert TurnSpec.model_json_schema()["properties"]["sender_id"]["type"] == "string"
    assert spec.content == "hi"

    # to_native() is the one place that reaches the engine.
    with pytest.raises(RuntimeError, match="native extension touched"):
        spec.to_native()
