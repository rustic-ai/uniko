"""Shared model runtimes and the prefetch policy (issue #57)."""

from __future__ import annotations

from pathlib import Path

import pytest

import uniko


def _texts(bundle: uniko.ContextBundle) -> str:
    return " | ".join(item.content for item in bundle.items)


async def test_two_instances_share_one_runtime(tmp_path: Path) -> None:
    runtime = await uniko.Uniko.builder().build_runtime()
    assert "embed/default" in runtime.aliases
    assert "embed/default" in repr(runtime)

    first = await uniko.Uniko.builder().runtime(runtime).build()
    second = await uniko.Uniko.builder().path(str(tmp_path)).runtime(runtime).build()
    await (
        first.agent("a")
        .session("s")
        .observe(uniko.Turn("alice", "I love hiking in the mountains every weekend"))
    )
    await (
        second.agent("a")
        .session("s")
        .observe(
            uniko.Turn("alice", "My favourite food is a spicy ramen with extra egg")
        )
    )
    assert "hiking" not in _texts(await second.agent("a").recall("hiking"))

    await first.shutdown()
    assert "ramen" in _texts(await second.agent("a").recall("ramen"))
    caps = await second.capabilities()
    assert caps["models"] is not None
    await second.shutdown()


def test_lazy_prefetch_sync() -> None:
    runtime = uniko.Uniko.builder().prefetch("lazy").build_runtime_sync()
    uni = uniko.Uniko.builder().runtime(runtime).build_sync()
    models = uni.capabilities_sync()["models"]
    assert all(m["state"] == "not_loaded" for m in models), models


def test_prefetch_rejects_an_unknown_policy() -> None:
    with pytest.raises(uniko.ConfigError) as info:
        uniko.Uniko.builder().prefetch("sometimes")  # ty: ignore[invalid-argument-type]
    assert "sometimes" in str(info.value)


def test_a_runtime_for_other_models_is_rejected() -> None:
    runtime = uniko.Uniko.builder().reranker(None).prefetch("lazy").build_runtime_sync()
    with pytest.raises(uniko.ConfigError) as info:
        uniko.Uniko.builder().runtime(runtime).build_sync()
    assert "rerank/default" in str(info.value)
