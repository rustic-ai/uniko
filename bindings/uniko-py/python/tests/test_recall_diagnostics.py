"""Recall stage diagnostics and strict recall (issue #59)."""

from __future__ import annotations

import pytest

import uniko
from uniko.models import ContextBundle, Diagnostic, TypedUniko

NO_RERANK = {"reranker": {"enabled": False}}
RUNBOOK = "# Runbook\n\nThe deploy key rotates every Friday at noon."


def _skip_if_embedder_failed(diagnostics: list) -> None:
    """Skip when the embedder could not load here (no network, no cache)."""
    if any(
        getattr(d, "stage", None) == "query_embedding" and d.status == "failed"
        for d in diagnostics
    ):
        pytest.skip("the embedder is unavailable here")


def _instance(config: dict) -> uniko.Uniko:
    try:
        return uniko.Uniko.builder().config(config).build_sync()
    except uniko.UnikoError as e:  # no network and no model cache
        pytest.skip(f"instance unavailable: {e}")


def test_a_recall_reports_its_stages() -> None:
    uni = _instance(NO_RERANK)
    agent = uni.agent("a")
    try:
        agent.session("s").ingest_sync(uniko.IngestSource.from_text(RUNBOOK))
    except uniko.UnikoError as e:
        pytest.skip(f"ingest unavailable: {e}")
    bundle = agent.recall_sync("when does the deploy key rotate")
    _skip_if_embedder_failed(bundle.diagnostics)
    stages = {(d.stage, d.status) for d in bundle.diagnostics}
    assert ("query_embedding", "applied") in stages, bundle.diagnostics
    assert all(d.status in {"applied", "skipped"} for d in bundle.diagnostics)
    first = bundle.diagnostics[0]
    assert isinstance(first, uniko.Diagnostic)
    assert repr(first).startswith("Diagnostic(")
    del agent
    uni.shutdown_sync()


def test_strict_recall_raises_when_a_stage_degrades() -> None:
    # The embedder cannot load, so recall falls back to keyword search.
    config = {
        **NO_RERANK,
        "embedding": {"model_id": "uniko-test/no-such-model-repo"},
    }
    try:
        uni = uniko.Uniko.builder().config(config).prefetch("lazy").build_sync()
    except uniko.UnikoError as e:
        pytest.skip(f"store did not open with a broken embedder: {e}")
    agent = uni.agent("a")
    bundle = agent.recall_sync("anything at all")
    failed = [d for d in bundle.diagnostics if d.stage == "query_embedding"]
    assert failed and all(
        (d.status, d.code) == ("failed", "embed_failed") for d in failed
    ), bundle.diagnostics

    with pytest.raises(uniko.DegradedError) as info:
        agent.recall_with_sync("anything at all", strict=True)
    err = info.value
    assert err.kind == "degraded"
    assert any(d["stage"] == "query_embedding" for d in err.diagnostics)
    assert isinstance(err, uniko.UnikoError)
    del agent
    uni.shutdown_sync()


async def test_typed_bundles_keep_diagnostics() -> None:
    try:
        uni = await TypedUniko.builder().config(NO_RERANK).build()
    except uniko.UnikoError as e:
        pytest.skip(f"instance unavailable: {e}")
    agent = uni.agent("a")
    bundle = await agent.recall_with("anything", strict=False)
    _skip_if_embedder_failed(bundle.diagnostics)
    assert isinstance(bundle, ContextBundle)
    assert all(isinstance(d, Diagnostic) for d in bundle.diagnostics)
    assert any(d.stage == "query_embedding" for d in bundle.diagnostics)
    assert not any(d.degraded for d in bundle.diagnostics)
    del agent
    await uni.shutdown()
