# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Dragonscale Industries Inc.
"""Agent registration (#67)."""

from __future__ import annotations

from pathlib import Path

import uniko

KIND = (
    "MATCH (p:Participant {participant_id: 'reporter'}) "
    "RETURN p.kind AS kind, p.name AS name"
)


def test_register_is_idempotent_and_records_nothing_else() -> None:
    agent = uniko.Uniko.in_memory_sync().agent("reporter")
    assert agent.register_sync(name="Reporter") == "created"
    assert agent.register_sync() == "existing"
    assert agent.query_sync(KIND) == [{"kind": "agent", "name": "Reporter"}]
    assert agent.query_sync("MATCH (n:Session) RETURN count(n) AS c") == [{"c": 0}]


async def test_register_async_upgrades_an_org_member() -> None:
    memory = await uniko.Uniko.in_memory()
    assert await memory.add_org_member("reporter", "acme")
    agent = memory.agent("reporter")
    assert await agent.register() == "upgraded"
    assert await agent.register() == "existing"
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
    assert uni.agent("reporter").register_sync() == "created"
    uni.shutdown_sync()

    uni = open_store()
    agent = uni.agent("reporter")
    assert agent.register_sync() == "existing"
    assert agent.query_sync(KIND)[0]["kind"] == "agent"
    del agent
    uni.shutdown_sync()
