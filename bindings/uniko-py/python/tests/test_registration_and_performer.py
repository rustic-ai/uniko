# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Dragonscale Industries Inc.
"""Agent registration (#67) and explicit tool-call performers (#68)."""

from __future__ import annotations

from pathlib import Path

import pytest

import uniko
from uniko.models import Registration, ToolCallSpec, TurnSpec
from uniko.models.typed import TypedUniko

PERFORMER = (
    "MATCH (:Action {action_id: 'call'})-[:PERFORMED_BY]->(p:Participant) "
    "RETURN p.participant_id AS who"
)
SENDER = (
    "MATCH (:Message {message_id: 'question'})-[:SENT_BY]->(p:Participant) "
    "RETURN p.participant_id AS who"
)
KIND = (
    "MATCH (p:Participant {participant_id: 'reporter'}) "
    "RETURN p.kind AS kind, p.name AS name"
)


def test_register_is_idempotent_and_records_nothing_else() -> None:
    agent = uniko.Uniko.in_memory_sync().agent("reporter")
    reg = agent.register_sync(name="Reporter")
    assert (reg.participant_id, reg.outcome) == ("reporter", "created")
    assert agent.register_sync().outcome == "existing"
    assert agent.query_sync(KIND) == [{"kind": "agent", "name": "Reporter"}]
    assert agent.query_sync("MATCH (n:Session) RETURN count(n) AS c") == [{"c": 0}]


async def test_register_async_upgrades_an_org_member() -> None:
    memory = await uniko.Uniko.in_memory()
    assert await memory.add_org_member("reporter", "acme")
    agent = memory.agent("reporter")
    assert (await agent.register()).outcome == "upgraded"
    assert (await agent.register()).outcome == "existing"
    rows = await agent.query(KIND)
    assert rows[0]["kind"] == "agent"


def test_goals_on_a_fresh_store_need_no_prior_turn() -> None:
    """The issue's reproduction: this used to raise "Participant not found"."""
    reporter = uniko.Uniko.builder().in_memory().build_sync().agent("reporter")
    reporter.goals.create_sync("Goal", goal_id="g")
    reporter.goals.create_task_sync("Task", goal_id="g")
    assert reporter.query_sync(KIND)[0]["kind"] == "agent"


def test_registration_survives_a_reopen(tmp_path: Path) -> None:
    def open_store() -> uniko.Uniko:
        return uniko.Uniko.builder().path(str(tmp_path / "store")).build_sync()

    uni = open_store()
    assert uni.agent("reporter").register_sync().outcome == "created"
    uni.shutdown_sync()

    uni = open_store()
    agent = uni.agent("reporter")
    assert agent.register_sync().outcome == "existing"
    assert agent.query_sync(KIND)[0]["kind"] == "agent"
    del agent
    uni.shutdown_sync()


def _unit() -> list[uniko.Turn]:
    call = uniko.ToolCall("search").call_id("call").text_output("Evidence")
    return [
        uniko.Turn("user", "Question")
        .id("question")
        .tool_call(call.performer("reporter"))
    ]


def test_an_embedded_call_can_name_its_performer(tmp_path: Path) -> None:
    """The issue's reproduction, now attributed to the reporting agent."""

    def open_store() -> uniko.Uniko:
        return uniko.Uniko.builder().path(str(tmp_path / "store")).build_sync()

    uni = open_store()
    session = uni.agent("reporter").session("research")
    assert not session.commit_unit_sync(_unit()).was_replay
    agent = uni.agent("reporter")
    assert agent.query_sync(PERFORMER) == [{"who": "reporter"}]
    assert agent.query_sync(SENDER) == [{"who": "user"}]
    del agent, session
    uni.shutdown_sync()

    uni = open_store()
    session = uni.agent("reporter").session("research")
    assert session.commit_unit_sync(_unit()).was_replay
    assert uni.agent("reporter").query_sync(PERFORMER) == [{"who": "reporter"}]
    del session
    uni.shutdown_sync()


def test_tool_call_reads_back_its_name_and_performer() -> None:
    call = uniko.ToolCall("search")
    assert (call.tool_name, call.performer_id) == ("search", None)
    assert call.performer("reporter").performer_id == "reporter"


def test_a_different_performer_for_the_same_call_is_refused() -> None:
    session = uniko.Uniko.in_memory_sync().agent("reporter").session("research")

    def call(performer: str) -> uniko.ToolCall:
        return (
            uniko.ToolCall("search")
            .call_id("call")
            .text_output("Evidence")
            .performer(performer)
        )

    assert not session.observe_tool_sync(call("auditor")).was_replay
    assert session.observe_tool_sync(call("auditor")).was_replay
    with pytest.raises(uniko.IdConflictError):
        session.observe_tool_sync(call("someone-else"))


def test_typed_spec_carries_the_performer() -> None:
    typed = TypedUniko(uniko.Uniko.in_memory_sync())
    session = typed.agent("reporter").session("research")
    spec = ToolCallSpec(
        tool_name="search", call_id="call", text_output="Evidence", performer="reporter"
    )
    turn = TurnSpec(
        sender_id="user", content="Question", message_id="question", tool_calls=[spec]
    )
    session.commit_unit_sync([turn])
    agent = typed.agent("reporter")
    reg = agent.register_sync()
    assert isinstance(reg, Registration)
    # Naming a performer creates it as `unknown`; registering makes it an
    # agent.
    assert (reg.participant_id, reg.outcome) == ("reporter", "upgraded")
    assert agent.query_sync(PERFORMER) == [{"who": "reporter"}]
