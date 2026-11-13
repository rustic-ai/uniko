# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Dragonscale Industries Inc.
"""Typed wrapper handle tests — specs in, models out, passthroughs raw."""

from __future__ import annotations

import uniko
from uniko.models import (
    ContextBundle,
    GoalSpec,
    GoalView,
    IngestOutcome,
    IngestSourceSpec,
    MessageView,
    ObserveResult,
    ScopeSpec,
    TaskSpec,
    TurnSpec,
    UnikoConfigModel,
)
from uniko.models import TypedUniko


async def test_typed_full_journey() -> None:
    uni = await TypedUniko.in_memory()
    assert isinstance(uni, TypedUniko)

    agent = uni.agent("assistant")
    session = agent.session("s1")

    # observe takes a spec → ObserveResult model
    result = await session.observe(
        TurnSpec(
            sender_id="alice",
            content="I love hiking in the Dolomites.",
            message_id="m1",
        )
    )
    assert isinstance(result, ObserveResult)

    # recall → ContextBundle model
    bundle = await agent.recall("hiking")
    assert isinstance(bundle, ContextBundle)

    # data view → model (or None)
    msg = await agent.data.message("m1")
    assert isinstance(msg, MessageView) and msg.message_id == "m1"
    assert await agent.data.message("missing") is None


async def test_typed_observe_accepts_native_turn() -> None:
    uni = await TypedUniko.in_memory()
    session = uni.agent("assistant").session("s1")
    # A native Turn also works (coercion passes it through unchanged).
    result = await session.observe(uniko.Turn("alice", "native turn works"))
    assert isinstance(result, ObserveResult)


async def test_typed_ingest_and_scope() -> None:
    uni = await TypedUniko.in_memory()
    agent = uni.agent("assistant")
    session = agent.session("s1")
    outcome = await session.ingest(IngestSourceSpec.from_text("doc body", id="doc1"))
    assert isinstance(outcome, IngestOutcome) and outcome.artifact_id == "doc1"

    await session.observe(TurnSpec(sender_id="alice", content="alpine hiking"))
    scoped = await agent.recall_in("hiking", ScopeSpec(sessions=["s1"]))
    assert isinstance(scoped, ContextBundle)


async def test_typed_goals_via_specs() -> None:
    uni = await TypedUniko.in_memory()
    agent = uni.agent("assistant")
    # register the agent participant
    await agent.session("setup").observe(
        TurnSpec(sender_id="assistant", content="kickoff")
    )
    goals = agent.goals

    gid = await goals.create(
        GoalSpec(title="Ship SDK", goal_id="g1", metrics={"phase": 3})
    )
    assert isinstance(gid, int)
    await goals.create_task(
        TaskSpec(title="tests", goal_id="g1", task_id="t1", priority=0.8)
    )

    goal = await goals.get("g1")
    assert isinstance(goal, GoalView) and goal.metrics == {"phase": 3}
    assert await goals.start("g1") is True  # transition stays a bool
    assert any(g.goal_id == "g1" for g in await goals.active())


async def test_typed_query_stays_passthrough() -> None:
    uni = await TypedUniko.in_memory()
    agent = uni.agent("assistant")
    await agent.session("s1").observe(TurnSpec(sender_id="alice", content="hi"))
    rows = await agent.query("MATCH (n) RETURN n LIMIT 1")
    assert isinstance(rows, list)
    assert all(isinstance(r, dict) for r in rows)


async def test_typed_config_model() -> None:
    uni = await TypedUniko.in_memory()
    cfg = uni.config()
    assert isinstance(cfg, UnikoConfigModel)
    assert isinstance(cfg.embedding.dimensions, int)
    assert cfg.embedding.model_id


def test_typed_sync_journey() -> None:
    uni = TypedUniko.in_memory_sync()
    agent = uni.agent("assistant")
    session = agent.session("s1")
    result = session.observe_sync(TurnSpec(sender_id="alice", content="sync hiking"))
    assert isinstance(result, ObserveResult)
    bundle = agent.recall_sync("hiking")
    assert isinstance(bundle, ContextBundle)
    assert isinstance(uni.config(), UnikoConfigModel)


def test_typed_builder() -> None:
    uni = TypedUniko.builder().in_memory().streaming(False).build_sync()
    assert isinstance(uni, TypedUniko)
    assert isinstance(uni.config(), UnikoConfigModel)
