"""Ingest stage diagnostics (issue #59)."""

from __future__ import annotations

import pytest

import uniko
from uniko.models import ObserveResult, TypedUniko

NO_RERANK = {"reranker": {"enabled": False}}
CORRUPT_PDF = b"%PDF-1.7\nnot really a pdf"


def _instance(config: dict, *, streaming: bool = False) -> uniko.Uniko:
    try:
        return uniko.Uniko.builder().config(config).streaming(streaming).build_sync()
    except uniko.UnikoError as e:  # no network and no model cache
        pytest.skip(f"instance unavailable: {e}")


def test_observe_reports_the_nlp_stage() -> None:
    config = {**NO_RERANK, "nlp": {"model_id": "uniko-test/no-such-model-repo"}}
    uni = _instance(config)
    session = uni.agent("a").session("s")
    result = session.observe_sync(uniko.Turn("alice", "I moved to Lisbon last spring"))
    nlp = [d for d in result.diagnostics if d.stage == "nlp"]
    assert len(nlp) == 1, result.diagnostics
    # Rules stand in for the NLP model, or the build has no NLP at all.
    assert (nlp[0].status, nlp[0].code) in {
        ("fallback", "nlp_unavailable"),
        ("skipped", "not_compiled"),
    }
    assert all("Lisbon" not in repr(d) for d in result.diagnostics)


def test_a_corrupt_pdf_reports_failed_extraction() -> None:
    uni = _instance(NO_RERANK)
    session = uni.agent("a").session("s")
    outcome = session.ingest_sync(
        uniko.IngestSource.from_bytes(CORRUPT_PDF).with_mime("application/pdf")
    )
    pdf = [d for d in outcome.diagnostics if d.stage == "pdf_extraction"]
    assert [
        (d.status, d.code in {"extraction_failed", "tiered_failed"}) for d in pdf
    ] == [("failed", True)], outcome.diagnostics


def test_streaming_counts_degradations_in_health() -> None:
    uni = _instance(NO_RERANK, streaming=True)
    session = uni.agent("a").session("s")
    session.submit_source_sync(
        uniko.IngestSource.from_bytes(CORRUPT_PDF).with_mime("application/pdf")
    )
    session.flush_sync()
    health = uni.health()
    assert health is not None
    counted = sum(
        n
        for k, n in health.ingest.degradations.items()
        if k.startswith("pdf_extraction:")
    )
    assert counted == 1, health.ingest.degradations


async def test_typed_results_keep_diagnostics() -> None:
    try:
        uni = await TypedUniko.builder().config(NO_RERANK).build()
    except uniko.UnikoError as e:
        pytest.skip(f"instance unavailable: {e}")
    session = uni.agent("a").session("s")
    result = await session.observe(uniko.Turn("alice", "I adopted a beagle called Rex"))
    assert isinstance(result, ObserveResult)
    nlp = [d for d in result.diagnostics if d.stage == "nlp"]
    assert len(nlp) == 1, result.diagnostics
    if nlp[0].code == "nlp_unavailable":
        pytest.skip("the NLP model is unavailable here")
    assert not any(d.degraded for d in result.diagnostics), result.diagnostics
