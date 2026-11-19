"""Sparse and other vector values returned by `query` (issue #71)."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import pytest

import uniko

BGE_M3_VOCAB = 250002


def test_a_null_property_is_still_none() -> None:
    uni = uniko.Uniko.in_memory_sync()
    agent = uni.agent("a")
    agent.session("s").observe_sync(
        uniko.Turn("alice", "the deploy key rotates on Fridays")
    )
    rows = agent.query_sync("MATCH (m:Message) RETURN m.nonexistent_property AS v")
    assert rows
    assert all(r["v"] is None for r in rows)


def _check_sparse(value: Any) -> None:
    assert isinstance(value, dict), value
    assert set(value) == {"indices", "values"}
    indices: list[Any] = value["indices"]
    weights: list[Any] = value["values"]
    assert indices, "a populated sparse vector has entries"
    assert len(indices) == len(weights)
    assert all(isinstance(i, int) and 0 <= i < BGE_M3_VOCAB for i in indices)
    assert all(isinstance(w, float) and math.isfinite(w) for w in weights)


@pytest.mark.skipif(
    os.environ.get("EXPENSIVE_TESTS") != "1",
    reason="set EXPENSIVE_TESTS=1 to load BGE-M3",
)
def test_bge_m3_sparse_values_come_back(tmp_path: Path) -> None:
    def open_store() -> uniko.Uniko:
        return (
            uniko.Uniko.builder()
            .path(str(tmp_path / "store"))
            .embedding("bge-m3")
            .config({"reranker": {"enabled": False}})
            .build_sync()
        )

    query = (
        "MATCH (c:Chunk) WHERE c.sparse_embedding IS NOT NULL "
        "RETURN c.sparse_embedding AS sparse"
    )
    uni = open_store()
    agent = uni.agent("a")
    agent.session("s").ingest_sync(
        uniko.IngestSource.from_text(
            "The Eiffel Tower is a wrought-iron lattice tower in Paris."
        )
    )
    rows = agent.query_sync(query)
    assert rows
    for r in rows:
        _check_sparse(r["sparse"])
    del agent
    uni.shutdown_sync()

    uni = open_store()
    reopened = uni.agent("a").query_sync(query)
    assert [r["sparse"] for r in reopened] == [r["sparse"] for r in rows]
    for r in reopened:
        _check_sparse(r["sparse"])
