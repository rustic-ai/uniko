"""Phase 2 tests: Data, Goals, ingest, streaming."""

from __future__ import annotations

import uniko


async def test_goals_lifecycle() -> None:
    uni = await uniko.Uniko.in_memory()
    agent = uni.agent("assistant")
    # Goals are OWNED_BY the agent's Participant, which must exist first — it is
    # created when the agent sends a turn.
    await agent.session("setup").observe(uniko.Turn("assistant", "kicking off"))
    goals = agent.goals

    gid = await goals.create(
        "Ship the Python SDK",
        goal_id="g1",
        description="PyO3 bindings",
        metrics={"target_phase": 3},
    )
    assert isinstance(gid, int)

    goal = await goals.get("g1")
    assert goal is not None
    assert goal.goal_id == "g1"
    assert goal.title == "Ship the Python SDK"
    # Phase is derived from the facade's status mapping; assert it's a valid
    # phase string rather than pinning the default.
    assert goal.phase in {"planned", "active", "completed", "abandoned"}
    assert goal.metrics == {"target_phase": 3}

    assert isinstance(await goals.start("g1"), bool)
    started = await goals.get("g1")
    assert started is not None
    assert started.phase == "active"
    assert any(g.goal_id == "g1" for g in await goals.active())

    tid = await goals.create_task(
        "write tests", goal_id="g1", task_id="t1", priority=0.8
    )
    assert isinstance(tid, int)
    tasks = await goals.tasks_of("g1")
    assert any(t.task_id == "t1" and t.priority == 0.8 for t in tasks)

    ctx = await goals.context("g1")
    assert ctx is not None
    assert ctx.goal.goal_id == "g1"

    assert await goals.complete("g1", {"done": True}) is True
    completed = await goals.get("g1")
    assert completed is not None
    assert completed.phase == "completed"


async def test_data_message_and_artifact_lookup() -> None:
    uni = await uniko.Uniko.in_memory()
    agent = uni.agent("assistant")
    session = agent.session("conversation-1")

    await session.observe(uniko.Turn("alice", "hello world").id("m1"))
    msg = await agent.data.message("m1")
    assert msg is not None
    assert msg.message_id == "m1"
    assert msg.sender_id == "alice"
    assert msg.timestamp.tzinfo is not None

    outcome = await session.ingest(
        uniko.IngestSource.from_text("# Spec\n\nrequirement one").with_id("doc1")
    )
    assert outcome.artifact_id == "doc1"
    assert outcome.kind == "artifact"

    artifact = await agent.data.artifact("doc1")
    assert artifact is not None
    assert artifact.artifact_id == "doc1"
    assert "requirement one" in artifact.text

    # Unknown ids resolve to None, never raise.
    assert await agent.data.message("nope") is None
    assert await agent.data.artifact("nope") is None


async def test_turn_attachment_ingested() -> None:
    uni = await uniko.Uniko.in_memory()
    agent = uni.agent("assistant")
    session = agent.session("conversation-1")

    turn = uniko.Turn("alice", "see the attached spec").attach(
        uniko.IngestSource.from_text("attached document body")
    )
    result = await session.observe(turn)
    assert len(result.attachments) == 1
    outcome = result.attachments[0]
    assert isinstance(outcome, uniko.IngestOutcome)
    assert outcome.kind == "artifact"
    assert outcome.chunk_node_ids, "the attachment was chunked"
    assert outcome.page_node_ids == [] and outcome.extraction_error is None


async def test_delete_session_cascade() -> None:
    uni = await uniko.Uniko.in_memory()
    agent = uni.agent("assistant")
    session = agent.session("conversation-1")
    await session.observe(uniko.Turn("alice", "to be deleted").id("m1"))

    report = await agent.delete_session("conversation-1")
    assert report.root_existed is True
    assert report.nodes_deleted >= 1
    assert await agent.data.message("m1") is None


async def test_streaming_submit_and_flush() -> None:
    uni = await uniko.Uniko.builder().in_memory().streaming(True).build()
    agent = uni.agent("assistant")
    session = agent.session("conversation-1")

    await session.submit(uniko.Turn("alice", "streamed turn").id("m1"))
    await session.flush()

    msg = await agent.data.message("m1")
    assert msg is not None
    assert msg.message_id == "m1"
