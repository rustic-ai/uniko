"""Startup readiness requirements (issue #60)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

import uniko
from uniko.models import ReadinessFailure, TypedUniko

R = uniko.Requirement
NO_RERANK = {"reranker": {"enabled": False}}


def _skip_if_embedder_unavailable(failures: list[dict[str, Any]]) -> None:
    """Skip when the default embedder cannot load here (no network, no cache)."""
    if any(
        f["code"] == "load_failed" and f["alias"] == "embed/default" for f in failures
    ):
        pytest.skip(f"default embedder unavailable: {failures}")


def test_static_requirements_fail_before_the_store_exists(tmp_path: Path) -> None:
    store = tmp_path / "store"
    with pytest.raises(uniko.ReadinessError) as info:
        (
            uniko.Uniko.builder()
            .path(str(store))
            .require([R.alias("embed/typo"), R.extractor("image"), R.embedder()])
            .build_sync()
        )
    err = info.value
    assert err.kind == "not_ready"
    assert [(f["requirement"], f["code"]) for f in err.failures] == [
        ("alias:embed/typo", "not_in_catalog"),
        ("extractor:image", "no_extractor"),
    ]
    assert isinstance(err, uniko.UnikoError)
    assert not store.exists()


def test_requirement_values() -> None:
    assert R.alias("gen/m").name == "alias:gen/m"
    assert R.accelerated("embed/default") == R.accelerated("embed/default")
    assert len({R.nlp(), R.nlp(), R.ocr()}) == 2
    assert repr(R.embedder()) == "Requirement(embedder)"
    with pytest.raises(uniko.ConfigError):
        R.extractor("smell")  # ty: ignore[invalid-argument-type]
    builder = uniko.Uniko.builder()
    for bad in (-1.0, math.nan):
        with pytest.raises(uniko.ConfigError):
            builder.readiness_timeout(bad)
    builder.readiness_timeout(math.inf).readiness_timeout(None)


def test_a_timed_out_build_releases_the_path(tmp_path: Path) -> None:
    def builder() -> uniko.UnikoBuilder:
        return (
            uniko.Uniko.builder().path(str(tmp_path)).config(NO_RERANK).prefetch("lazy")
        )

    # Cut the load off at once: the store must still be released.
    with pytest.raises(uniko.ReadinessError) as info:
        builder().require([R.embedder()]).readiness_timeout(0.0).build_sync()
    failures = info.value.failures
    _skip_if_embedder_unavailable(failures)
    assert [(f["requirement"], f["code"]) for f in failures] == [
        ("embedder", "timed_out")
    ]
    uni = builder().build_sync()
    uni.shutdown_sync()


async def test_a_lazy_store_loads_what_is_required() -> None:
    try:
        uni = await (
            uniko.Uniko.builder()
            .config(NO_RERANK)
            .prefetch("lazy")
            .require([R.embedder()])
            .readiness_timeout(600.0)
            .build()
        )
    except uniko.ReadinessError as e:
        _skip_if_embedder_unavailable(e.failures)
        raise
    models = {m["alias"]: m["state"] for m in (await uni.capabilities())["models"]}
    assert models["embed/default"] == "loaded"
    assert models["nlp/default"] == "not_loaded"
    assert await uni.check_readiness([R.embedder()], timeout=60.0) == []
    missing = await uni.check_readiness([R.alias("embed/typo")])
    assert missing == [
        {
            "requirement": "alias:embed/typo",
            "code": "not_in_catalog",
            "alias": "embed/typo",
            "error": None,
        }
    ]
    await uni.shutdown()


def test_typed_check_readiness() -> None:
    uni = TypedUniko.builder().config(NO_RERANK).prefetch("lazy").build_sync()
    failures = uni.check_readiness_sync([R.generator(), R.accelerated("nlp/default")])
    assert all(isinstance(f, ReadinessFailure) for f in failures)
    assert [(f.requirement, f.code) for f in failures] == [
        ("generator", "not_configured"),
        ("accelerated:nlp/default", "not_supported"),
    ]
    uni.shutdown_sync()
