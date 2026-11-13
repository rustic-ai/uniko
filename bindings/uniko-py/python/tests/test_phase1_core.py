"""Phase 1 core-loop tests: Uniko/Agent/Session/Turn + recall/answer/query.

Exercised against an in-memory instance. The observe/recall test drives the
real embedding + NLP pipeline (models load on first use).
"""

from __future__ import annotations

import datetime

import pytest

import uniko


async def test_open_agent_and_query_empty_graph() -> None:
    uni = await uniko.Uniko.in_memory()
    agent = uni.agent("assistant")
    assert agent.agent_id == "assistant"
    # Exercises the query path + Value/Record converter on an empty graph.
    rows = await agent.query("MATCH (n) RETURN n LIMIT 5")
    assert isinstance(rows, list)


async def test_answer_without_llm_raises_config_error() -> None:
    uni = await uniko.Uniko.in_memory()
    agent = uni.agent("assistant")
    with pytest.raises(uniko.ConfigError) as info:
        await agent.answer("what do you know?")
    assert info.value.kind == "config"


async def test_session_and_turn_builder_chaining() -> None:
    uni = await uniko.Uniko.in_memory()
    agent = uni.agent("assistant")
    session = agent.session("conversation-1")
    assert session.session_id == "conversation-1"
    turn = (
        uniko.Turn("alice", "here is the plan")
        .id("m1")
        .content_type("text")
        .addressed_to(["bob"])
        .metadata("topic", "planning")
        .at(datetime.datetime.now(datetime.timezone.utc))
    )
    assert repr(turn).startswith("Turn")


async def test_observe_then_recall_roundtrip() -> None:
    uni = await uniko.Uniko.in_memory()
    agent = uni.agent("assistant")
    session = agent.session("conversation-1")

    result = await session.observe(
        uniko.Turn("alice", "I love hiking in the mountains.")
    )
    assert result.message_node_id > 0
    assert result.session_node_id > 0
    assert repr(result).startswith("ObserveResult")

    bundle = await agent.recall("hiking")
    assert isinstance(bundle.items, list)
    assert bundle.coverage >= 0.0
    # If anything was recalled, the typed getters must work end to end.
    for item in bundle.items:
        assert isinstance(item.content, str)
        assert isinstance(item.kind, str)
        assert isinstance(item.sources, list)

    # A node row exercises the Value::Node -> dict converter.
    rows = await agent.query("MATCH (n) RETURN n LIMIT 1")
    if rows:
        node = rows[0]["n"]
        assert "id" in node and "labels" in node and "properties" in node


async def test_builder_in_memory_builds() -> None:
    uni = await uniko.Uniko.builder().in_memory().streaming(False).build()
    agent = uni.agent("assistant")
    assert agent.agent_id == "assistant"


async def test_shutdown_poisons_handle() -> None:
    uni = await uniko.Uniko.in_memory()
    await uni.shutdown()
    with pytest.raises(uniko.UnikoError):
        uni.agent("assistant")
