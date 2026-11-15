# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Dragonscale Industries Inc.
"""Tool calls: recorded with a turn or on their own, stored verbatim,
idempotent on their call id, and found by recall."""

from __future__ import annotations

import json

import pytest

import uniko
from uniko.models import ToolCallResult, ToolCallSpec, TurnSpec
from uniko.models.typed import TypedUniko

ORDERS = {
    "meta": {"currency": "EUR", "amount_unit": "cents"},
    "data": {
        "items": [
            {"order_id": f"ORD-{60_000 + i}", "status": "shipped", "amount": 900 + i}
            for i in range(25)
        ]
    },
}


def test_tool_call_commits_with_its_turn() -> None:
    agent = uniko.Uniko.in_memory_sync().agent("assistant")
    session = agent.session("py-tools-1")
    call = (
        uniko.ToolCall("list_orders")
        .call_id("py-call-1")
        .arguments({"customer_id": "C-7"})
        .json_output(ORDERS)
        .status("ok")
    )
    result = session.observe_sync(
        uniko.Turn("assistant", "Checking the orders.").id("py-m-1").tool_call(call)
    )
    (tc,) = result.tool_calls
    assert isinstance(tc, uniko.ToolCallResult)
    assert tc.call_id == "py-call-1" and not tc.was_replay
    assert tc.artifact_id is not None

    raw = agent.data.artifact_bytes_sync(tc.artifact_id)
    assert raw is not None and json.loads(raw) == ORDERS


def test_observe_tool_replays_and_refuses_a_changed_result() -> None:
    session = uniko.Uniko.in_memory_sync().agent("assistant").session("py-tools-2")

    def call(output: str) -> uniko.ToolCall:
        return uniko.ToolCall("run_tests").call_id("py-call-2").text_output(output)

    first = session.observe_tool_sync(call("1 failed, 39 passed"))
    assert not first.was_replay
    again = session.observe_tool_sync(call("1 failed, 39 passed"))
    assert again.was_replay and again.action_node == first.action_node
    with pytest.raises(uniko.IdConflictError):
        session.observe_tool_sync(call("40 passed"))


async def test_observe_tool_async() -> None:
    memory = await uniko.Uniko.in_memory()
    session = memory.agent("assistant").session("py-tools-3")
    out = await session.observe_tool(
        uniko.ToolCall("fetch_logs").text_output(
            "12:00:01 ERROR payments error=timeout"
        )
    )
    assert out.artifact_id is not None


def test_tool_call_spec_and_typed_session() -> None:
    with pytest.raises(ValueError):
        ToolCallSpec(tool_name="t", json_output={"a": 1}, text_output="b")

    memory = TypedUniko(uniko.Uniko.in_memory_sync())
    session = memory.agent("assistant").session("py-tools-4")
    spec = TurnSpec(
        sender_id="assistant",
        content="Looking that up.",
        message_id="py-m-4",
        tool_calls=[
            ToolCallSpec(tool_name="lookup", call_id="py-call-4", json_output={"v": 1})
        ],
    )
    observed = session.observe_sync(spec)
    assert [t.call_id for t in observed.tool_calls] == ["py-call-4"]
    standalone = session.observe_tool_sync(
        ToolCallSpec(tool_name="lookup", call_id="py-call-5", text_output="ok")
    )
    assert isinstance(standalone, ToolCallResult)


def test_tool_call_result_is_recalled() -> None:
    agent = uniko.Uniko.in_memory_sync().agent("assistant")
    agent.session("py-tools-6").observe_sync(
        uniko.Turn("assistant", "Pulling orders.")
        .id("py-m-6")
        .tool_call(
            uniko.ToolCall("list_orders").call_id("py-call-6").json_output(ORDERS)
        )
    )
    bundle = agent.recall_sync("What is the status of order ORD-60017?")
    assert any("ORD-60017" in item.content for item in bundle.items)


def test_sources_search_outline_read_and_tool_calls() -> None:
    agent = uniko.Uniko.in_memory_sync().agent("assistant")
    agent.session("py-src-1").observe_sync(
        uniko.Turn("assistant", "Pulling orders.")
        .id("py-m-src")
        .tool_call(
            uniko.ToolCall("list_orders")
            .call_id("py-call-src")
            .arguments({"customer_id": "C-7"})
            .json_output(ORDERS)
        )
    )
    src = agent.sources
    hits = src.search_sync("status of ORD-60011", limit=5)
    hit = next(h for h in hits if "ORD-60011" in h.text)
    assert hit.source_kind == "artifact" and hit.call_id == "py-call-src"
    assert hit.context is not None and "meta.currency=EUR" in hit.context

    outline = src.outline_sync(hit.source_kind, hit.source_id)
    assert outline is not None and outline.format == "json"
    assert any(e.address == "/data/items" for e in outline.entries)

    read = src.read_sync(hit.source_kind, hit.source_id, address="/data/items/11")
    assert read is not None and json.loads(read.text)["order_id"] == "ORD-60011"
    with pytest.raises(ValueError):
        src.read_sync(hit.source_kind, hit.source_id, address="/x", lines=(1, 2))

    view = src.tool_call_sync("py-call-src")
    assert view is not None and view.arguments == {"customer_id": "C-7"}
    assert [v.call_id for v in src.tool_calls_sync(session_id="py-src-1")] == [
        "py-call-src"
    ]

    typed = TypedUniko(uniko.Uniko.in_memory_sync()).agent("a").sources
    assert typed.search_sync("ORD-1") == []
