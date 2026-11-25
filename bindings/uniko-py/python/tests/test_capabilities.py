"""The capability and model-status report (issue #56)."""

from __future__ import annotations

from typing import Any

import uniko
from uniko.models import Capabilities, TypedUniko


def _check(caps: dict[str, Any]) -> None:
    assert {"pdf-ocr", "onnx", "gpu-cuda", "candle"} <= set(caps["compiled_features"])
    assert caps["ocr_configured"] is False
    models = {m["alias"]: m for m in caps["models"]}
    assert models["embed/default"]["task"] == "embed"
    assert models["embed/default"]["state"] == "loaded"
    assert "options" not in models["embed/default"]


async def test_capabilities_async() -> None:
    uni = await (
        uniko.Uniko.builder()
        .in_memory()
        .llm(
            uniko.LlmSpec.llamacpp(
                "generate/a", "stub", "http://alice:pw@127.0.0.1:9/v1?k=s"
            )
        )
        .build()
    )
    caps = await uni.capabilities()
    _check(caps)
    assert caps["answer_alias"] == "generate/a"
    remote = next(m for m in caps["models"] if m["alias"] == "generate/a")
    assert remote["base_url"] == "http://127.0.0.1:9/v1"
    assert "pw" not in repr(caps)


def test_capabilities_sync() -> None:
    uni = uniko.Uniko.in_memory_sync()
    caps = uni.capabilities_sync()
    _check(caps)
    assert caps["answer_alias"] is None


async def test_typed_capabilities() -> None:
    uni = await TypedUniko.in_memory()
    caps = await uni.capabilities()
    assert isinstance(caps, Capabilities)
    _check(caps.model_dump())
    assert caps.models is not None
    assert all(
        m.state in {"not_loaded", "loading", "loaded", "failed"} for m in caps.models
    )


def test_a_failed_model_validates_with_its_error() -> None:
    caps = Capabilities.model_validate(
        {
            "compiled_features": {"pdf-ocr": False},
            "extractors": [],
            "models": [
                {
                    "alias": "embed/missing",
                    "task": "embed",
                    "provider_id": "local/onnx",
                    "model_id": "nope",
                    "state": "failed",
                    "error": "download failed",
                }
            ],
            "answer_alias": None,
            "ocr_compiled": False,
            "ocr_configured": False,
        }
    )
    assert caps.models is not None
    assert (caps.models[0].state, caps.models[0].error) == ("failed", "download failed")
