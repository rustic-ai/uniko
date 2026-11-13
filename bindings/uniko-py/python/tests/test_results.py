# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Dragonscale Industries Inc.
"""Result classes, per-call recall overrides, error names, and `_sync` stacks."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import warnings

import pytest

import uniko
from uniko.models import SummarizeReport, UnitResult
from uniko.models.typed import TypedUniko


def seeded() -> tuple[uniko.Uniko, uniko.Agent]:
    memory = uniko.Uniko.in_memory_sync()
    agent = memory.agent("assistant")
    session = agent.session("hikes")
    for text in (
        "I hike in the Alps every summer",
        "My favourite trail is the Haute Route",
        "I bought new hiking boots last week",
        "Hiking with friends is the best part of my weekend",
    ):
        session.observe_sync(uniko.Turn("alice", text))
    return memory, agent


# ── recall_with ────────────────────────────────────────────────────────────


def test_recall_with_overrides_apply_to_one_call() -> None:
    _, agent = seeded()
    assert len(agent.recall_with_sync("hiking", limit=1).items) <= 1
    assert agent.recall_with_sync("hiking", min_score=1e9).items == []
    # Nothing stuck: plain recall still returns items.
    assert agent.recall_sync("hiking").items


async def test_recall_with_async_and_scope() -> None:
    _, agent = seeded()
    scope = uniko.Scope().sessions(["no-such-session"])
    bundle = await agent.recall_with("hiking", scope, limit=3)
    assert bundle.items == [], "the scope filters out every session"


def test_recall_with_rejects_a_limit_above_reranker_top_n() -> None:
    memory, agent = seeded()
    top_n = memory.config()["reranker"]["top_n"]
    with pytest.raises(uniko.ConfigError, match="top_n"):
        agent.recall_with_sync("hiking", limit=top_n + 1)
    # With the reranker off for the call, the same limit is fine.
    agent.recall_with_sync("hiking", limit=top_n + 1, reranker=False)


def test_recall_with_cannot_enable_an_unregistered_reranker() -> None:
    memory = uniko.Uniko.builder().in_memory().reranker(None).build_sync()
    with pytest.raises(uniko.ConfigError, match="build time"):
        memory.agent("a").recall_with_sync("anything", reranker=True)


def test_typed_recall_with() -> None:
    uni = TypedUniko.in_memory_sync()
    agent = uni.agent("assistant")
    agent.session("s").observe_sync(uniko.Turn("alice", "sync hiking"))
    bundle = agent.recall_with_sync("hiking", limit=1)
    assert len(bundle.items) <= 1


# ── result classes ─────────────────────────────────────────────────────────


def test_summarize_returns_a_report() -> None:
    _, agent = seeded()
    report = agent.session("hikes").summarize_sync()
    assert isinstance(report, uniko.SummarizeReport)
    assert isinstance(report.summary, int), "a session with turns gets a summary"
    assert report.finalize_error is None
    assert isinstance(report.finalize, uniko.FinalizeReport)
    model = SummarizeReport.from_native(report)
    assert model.finalize is not None


def test_typed_session_commit_unit_and_summarize() -> None:
    uni = TypedUniko.in_memory_sync()
    session = uni.agent("a").session("typed-unit")
    unit = session.commit_unit_sync(
        [uniko.Turn("u", "first turn").id("t-1"), uniko.Turn("a", "second").id("t-2")]
    )
    assert isinstance(unit, UnitResult)
    assert len(unit) == 2 and not unit.was_replay
    assert isinstance(session.summarize_sync(), SummarizeReport)


def test_typed_agent_retire_and_restore_source() -> None:
    uni = TypedUniko.in_memory_sync()
    agent = uni.agent("a")
    agent.session("s").observe_sync(uniko.Turn("u", "versioned").source("doc-1"))
    assert agent.retire_source_sync("doc-1") is True
    assert agent.restore_source_sync("doc-1") is True


# ── names ──────────────────────────────────────────────────────────────────


def test_star_import_does_not_shadow_builtin_timeout_error() -> None:
    namespace: dict = {}
    exec("from uniko import *", namespace)
    assert "TimeoutError" not in namespace
    assert namespace["UnikoTimeoutError"] is uniko.UnikoTimeoutError
    assert issubclass(uniko.UnikoTimeoutError, uniko.UnikoError)


def test_old_timeout_error_name_still_resolves_with_a_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert uniko.TimeoutError is uniko.UnikoTimeoutError
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_all_lists_every_result_class() -> None:
    for name in (
        "FinalizeReport",
        "CycleStats",
        "UnitResult",
        "SummarizeReport",
        "PipelineHealth",
        "WorkerHealth",
    ):
        assert name in uniko.__all__, name


# ── _sync calls on a small thread stack ───────────────────────────────────


def test_sync_recall_survives_a_small_thread_stack() -> None:
    # Recall needs 1-2 MiB of stack. `_sync` work runs on the runtime's
    # 8 MiB workers, so a caller thread with far less must not matter. A
    # stack overflow aborts the process, hence the subprocess.
    script = textwrap.dedent(
        """
        import threading
        import uniko

        memory = uniko.Uniko.in_memory_sync()
        agent = memory.agent("a")
        agent.session("s").observe_sync(uniko.Turn("u", "hiking in the alps"))
        result = {}

        def work():
            result["items"] = len(agent.recall_sync("hiking").items)

        threading.stack_size(256 * 1024)
        thread = threading.Thread(target=work)
        thread.start()
        thread.join()
        assert result.get("items", 0) >= 1, result
        print("ok")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, (
        f"exit {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr[-4000:]}"
    )
    assert "ok" in proc.stdout
