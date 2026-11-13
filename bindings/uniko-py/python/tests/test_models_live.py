# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Dragonscale Industries Inc.
"""Live round-trips: native handles + from_native / adapters against in_memory().

Mirrors the existing phase tests: fixture-free, fresh in_memory() per test,
asyncio_mode=auto. Answer requires an LLM (tested via fake in
test_models_outputs.py), so it is not exercised live here.
"""

from __future__ import annotations

import json

import pytest

import uniko
from uniko.models import (
    AbductionResult,
    ArtifactView,
    ContextBundle,
    DeletionReport,
    GoalContext,
    GoalSpec,
    GoalView,
    IngestOutcome,
    IngestSourceSpec,
    MessageView,
    ObserveResult,
    ScopeSpec,
    TaskSpec,
    TaskView,
    TurnSpec,
    UnikoConfigModel,
)
from uniko.models import adapters


async def test_live_observe_returns_model() -> None:
    uni = await uniko.Uniko.in_memory()
    session = uni.agent("assistant").session("s1")
    result = await adapters.observe(
        session, TurnSpec(sender_id="alice", content="I love hiking.")
    )
    assert isinstance(result, ObserveResult)
    assert result.message_node_id > 0


async def test_live_recall_bundle_model() -> None:
    uni = await uniko.Uniko.in_memory()
    agent = uni.agent("assistant")
    await adapters.observe(
        agent.session("s1"),
        TurnSpec(sender_id="alice", content="I love hiking in the Dolomites."),
    )
    bundle = ContextBundle.from_native(await agent.recall("hiking"))
    assert isinstance(bundle, ContextBundle)
    # serialization round-trips
    json.loads(bundle.model_dump_json())


async def test_live_recall_in_with_scope_spec() -> None:
    uni = await uniko.Uniko.in_memory()
    agent = uni.agent("assistant")
    await adapters.observe(
        agent.session("s1"), TurnSpec(sender_id="alice", content="hiking trip")
    )
    await adapters.observe(
        agent.session("s2"), TurnSpec(sender_id="bob", content="cooking class")
    )
    bundle = await adapters.recall_in(agent, "activity", ScopeSpec(sessions=["s1"]))
    assert isinstance(bundle, ContextBundle)


async def test_live_ingest_outcome_model() -> None:
    uni = await uniko.Uniko.in_memory()
    session = uni.agent("assistant").session("s1")
    outcome = await adapters.ingest(
        session, IngestSourceSpec.from_text("# Spec\n\nrule one", id="doc1")
    )
    assert isinstance(outcome, IngestOutcome)
    assert outcome.artifact_id == "doc1"


async def test_live_data_views_models() -> None:
    uni = await uniko.Uniko.in_memory()
    agent = uni.agent("assistant")
    session = agent.session("s1")
    await adapters.observe(
        session, TurnSpec(sender_id="alice", content="hello", message_id="m1")
    )
    await adapters.ingest(session, IngestSourceSpec.from_text("body text", id="doc1"))

    msg = MessageView.from_native(await agent.data.message("m1"))
    assert msg.message_id == "m1" and msg.timestamp.tzinfo is not None
    art = ArtifactView.from_native(await agent.data.artifact("doc1"))
    assert "body text" in art.text


async def test_live_goals_models() -> None:
    uni = await uniko.Uniko.in_memory()
    agent = uni.agent("assistant")
    # Register the agent's participant (goals are owned by it).
    await adapters.observe(
        agent.session("setup"), TurnSpec(sender_id="assistant", content="kickoff")
    )
    goals = agent.goals

    await adapters.create_goal(
        goals, GoalSpec(title="Ship SDK", goal_id="g1", metrics={"phase": 3})
    )
    await adapters.create_task(
        goals, TaskSpec(title="write tests", goal_id="g1", task_id="t1", priority=0.8)
    )

    goal = GoalView.from_native(await goals.get("g1"))
    assert goal.goal_id == "g1" and goal.metrics == {"phase": 3}
    tasks = [TaskView.from_native(t) for t in await goals.tasks_of("g1")]
    assert any(t.task_id == "t1" and t.priority == 0.8 for t in tasks)
    ctx = GoalContext.from_native(await goals.context("g1"))
    assert ctx.goal.goal_id == "g1"


async def test_live_deletion_report_model() -> None:
    uni = await uniko.Uniko.in_memory()
    agent = uni.agent("assistant")
    await adapters.observe(
        agent.session("s1"), TurnSpec(sender_id="alice", content="bye", message_id="m1")
    )
    report = DeletionReport.from_native(await agent.delete_session("s1"))
    assert report.root_existed is True and report.nodes_deleted >= 1


async def test_live_abduction_result_model() -> None:
    uni = await uniko.Uniko.in_memory()
    agent = uni.agent("assistant")
    await adapters.observe(
        agent.session("s1"), TurnSpec(sender_id="alice", content="seed")
    )
    await agent.define_rule(
        "reachable", "CREATE RULE reachable AS MATCH (a:Episode) YIELD KEY a"
    )
    result = AbductionResult.from_native(
        await agent.abduce("ABDUCE reachable WHERE a.kind = 'query'")
    )
    assert isinstance(result, AbductionResult)
    for mod in result.modifications:
        assert isinstance(mod.validated, bool) and isinstance(mod.cost, float)


async def test_live_config_model() -> None:
    uni = await uniko.Uniko.in_memory()
    cfg = UnikoConfigModel.from_native(uni)
    assert isinstance(cfg.embedding.dimensions, int)
    assert isinstance(cfg.embedding.model_id, str)
    assert isinstance(cfg.recall_min_score, float)


async def test_answer_requires_llm() -> None:
    # Answer can't be exercised live without an LLM; document the contract.
    uni = await uniko.Uniko.in_memory()
    agent = uni.agent("assistant")
    with pytest.raises(uniko.ConfigError):
        await agent.answer("anything?")


# ── sync twins (mirror test_phase4_sync.py) ────────────────────────────────


def test_live_sync_observe_recall_config() -> None:
    uni = uniko.Uniko.in_memory_sync()
    agent = uni.agent("assistant")
    session = agent.session("s1")
    result = adapters.observe_sync(
        session, TurnSpec(sender_id="alice", content="sync hiking")
    )
    assert isinstance(result, ObserveResult)
    bundle = ContextBundle.from_native(agent.recall_sync("hiking"))
    assert isinstance(bundle, ContextBundle)
    cfg = UnikoConfigModel.from_native(uni)
    assert isinstance(cfg.recall_limit, int)
