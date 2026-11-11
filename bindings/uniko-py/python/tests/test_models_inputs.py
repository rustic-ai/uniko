# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Dragonscale Industries Inc.
"""Input spec tests — validation, discriminated unions, schema (pure Pydantic)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from uniko.models import (
    GoalSpec,
    IngestSourceSpec,
    LlmSpecAdapter,
    ScopeSpec,
    TaskSpec,
    TurnSpec,
    llm_to_native,
)


# ── GoalSpec / TaskSpec ────────────────────────────────────────────────────


def test_goal_spec_to_create_kwargs_drops_title_and_nones():
    spec = GoalSpec(title="Ship", status="planned", metrics={"qps": 10})
    kwargs = spec.to_create_kwargs()
    assert "title" not in kwargs
    assert kwargs == {"status": "planned", "metrics": {"qps": 10}}


def test_goal_spec_empty_title_rejected():
    with pytest.raises(ValidationError) as exc:
        GoalSpec(title="")
    assert exc.value.errors()[0]["type"] == "string_too_short"


def test_goal_spec_typo_kwarg_rejected():
    with pytest.raises(ValidationError) as exc:
        GoalSpec(title="x", descriptoin="oops")  # ty: ignore[unknown-argument]
    assert exc.value.errors()[0]["type"] == "extra_forbidden"


def test_goal_spec_status_is_freeform():
    # any string is legal — no enum constraint
    for status in ("achieved", "in-flight", "🚀"):
        assert GoalSpec(title="g", status=status).status == status


def test_task_spec_kwargs():
    spec = TaskSpec(title="do", priority=0.5, goal_id="g-1")
    assert spec.to_create_task_kwargs() == {"priority": 0.5, "goal_id": "g-1"}


def test_task_spec_empty_title_rejected():
    with pytest.raises(ValidationError):
        TaskSpec(title="")


# ── ScopeSpec ──────────────────────────────────────────────────────────────


def test_scope_spec_all_optional():
    assert ScopeSpec().model_dump(exclude_none=True) == {}
    s = ScopeSpec(sessions=["s1"], participants=["p1"])
    assert s.sessions == ["s1"] and s.participants == ["p1"]


# ── TurnSpec ───────────────────────────────────────────────────────────────


def test_turn_spec_minimal_and_full():
    assert TurnSpec(sender_id="a", content="hi").content == "hi"
    full = TurnSpec(
        sender_id="a",
        content="hi",
        message_id="m1",
        metadata={"k": 1},
        attachments=[IngestSourceSpec.from_text("doc")],
    )
    assert full.metadata == {"k": 1}
    assert full.attachments is not None
    assert full.attachments[0].origin.source == "text"


def test_turn_spec_empty_sender_rejected():
    with pytest.raises(ValidationError):
        TurnSpec(sender_id="", content="hi")


# ── IngestSourceSpec (discriminated union) ─────────────────────────────────


def test_ingest_source_each_variant():
    assert IngestSourceSpec.from_text("hi").origin.source == "text"
    assert IngestSourceSpec.from_bytes(b"\x00").origin.source == "bytes"
    assert IngestSourceSpec.from_path("/tmp/x").origin.source == "path"


def test_ingest_source_modifiers():
    s = IngestSourceSpec.from_text(
        "hi", mime="text/plain", id="a1", with_path="src://x"
    )
    assert s.mime == "text/plain" and s.id == "a1" and s.with_path == "src://x"


def test_ingest_source_bad_discriminator_rejected():
    with pytest.raises(ValidationError) as exc:
        # pydantic coerces the dict at runtime; the static type is the model union
        bad = {"source": "nope", "content": "x"}
        IngestSourceSpec(origin=bad)  # ty: ignore[invalid-argument-type]
    assert exc.value.errors()[0]["type"] == "union_tag_invalid"


def test_ingest_source_mixed_fields_rejected():
    # 'text' variant with a 'data' field set → extra_forbidden on the inner model
    with pytest.raises(ValidationError):
        mixed = {"source": "text", "content": "x", "data": b"y"}
        IngestSourceSpec(origin=mixed)  # ty: ignore[invalid-argument-type]


def test_ingest_source_schema_has_oneof_discriminator():
    schema = IngestSourceSpec.model_json_schema()
    origin = schema["properties"]["origin"]
    assert "oneOf" in origin
    assert origin.get("discriminator", {}).get("propertyName") == "source"


# ── LlmSpecModel (discriminated union via TypeAdapter) ─────────────────────


def test_llm_spec_each_provider():
    openai = LlmSpecAdapter.validate_python(
        {"provider": "openai", "alias": "a", "model_id": "gpt-4o-mini"}
    )
    assert openai.provider == "openai"
    keyenv = LlmSpecAdapter.validate_python(
        {
            "provider": "openai_with_key_env",
            "alias": "a",
            "model_id": "m",
            "key_env": "MY_KEY",
        }
    )
    assert keyenv.key_env == "MY_KEY"


def test_llm_spec_missing_required_rejected():
    with pytest.raises(ValidationError) as exc:
        LlmSpecAdapter.validate_python(
            {"provider": "openai_with_key_env", "alias": "a", "model_id": "m"}
        )
    assert exc.value.errors()[0]["type"] == "missing"


def test_llm_spec_unknown_provider_rejected():
    with pytest.raises(ValidationError) as exc:
        LlmSpecAdapter.validate_python({"provider": "anthropic", "alias": "a"})
    assert exc.value.errors()[0]["type"] == "union_tag_invalid"


def test_llm_spec_schema_has_discriminator():
    schema = LlmSpecAdapter.json_schema()
    assert "oneOf" in schema
    assert schema.get("discriminator", {}).get("propertyName") == "provider"


def test_llm_to_native_dispatches():
    spec = LlmSpecAdapter.validate_python(
        {"provider": "openai", "alias": "a", "model_id": "gpt-4o-mini"}
    )
    assert repr(llm_to_native(spec)).startswith("LlmSpec")
