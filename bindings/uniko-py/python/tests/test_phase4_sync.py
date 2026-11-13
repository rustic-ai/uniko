"""Phase 4 tests: the blocking ``*_sync`` skins.

These are plain (non-``async``) tests: they drive the whole facade with **no
event loop**, exercising the sync skins that ``block_on`` the shared runtime.
"""

from __future__ import annotations

import threading
import time

import uniko


def test_sync_core_loop() -> None:
    uni = uniko.Uniko.in_memory_sync()
    agent = uni.agent("assistant")
    session = agent.session("s1")

    result = session.observe_sync(uniko.Turn("alice", "rockets reach orbit at dawn"))
    assert isinstance(result, uniko.ObserveResult)
    assert result.message_node_id > 0

    bundle = agent.recall_sync("rockets")
    assert isinstance(bundle, uniko.ContextBundle)
    assert isinstance(bundle.items, list)

    rows = agent.query_sync("MATCH (n) RETURN n LIMIT 3")
    assert isinstance(rows, list)

    report = uni.purge_sync()
    assert isinstance(report, uniko.DeletionReport)

    # `shutdown` consumes the instance: all derived handles must be dropped
    # first (Python has no move semantics), or it raises ConfigError.
    del session, agent
    uni.shutdown_sync()


def test_sync_builder() -> None:
    uni = uniko.Uniko.builder().in_memory().streaming(False).build_sync()
    assert isinstance(uni, uniko.Uniko)
    uni.shutdown_sync()


def test_sync_data_roundtrip() -> None:
    uni = uniko.Uniko.in_memory_sync()
    agent = uni.agent("assistant")
    session = agent.session("s1")

    result = session.observe_sync(uniko.Turn("alice", "hello there").id("m-100"))
    assert result.message_node_id > 0

    view = agent.data.message_sync("m-100")
    assert view is not None
    assert view.message_id == "m-100"
    assert view.content == "hello there"

    # Unknown id resolves to None rather than raising.
    assert agent.data.message_sync("nope") is None


def test_sync_goals() -> None:
    uni = uniko.Uniko.in_memory_sync()
    agent = uni.agent("assistant")
    # Seed the agent's participant so goal ownership resolves.
    agent.session("s1").observe_sync(uniko.Turn("assistant", "planning session"))

    goal_id = agent.goals.create_sync("ship the sync API", description="phase 4")
    assert isinstance(goal_id, int)
    assert goal_id > 0

    assert agent.goals.start_sync("ship the sync API") in (True, False)

    active = agent.goals.active_sync()
    assert isinstance(active, list)
    assert all(isinstance(g, uniko.GoalView) for g in active)

    task_id = agent.goals.create_task_sync("write tests", goal_id="ship the sync API")
    assert isinstance(task_id, int)
    tasks = agent.goals.tasks_sync()
    assert isinstance(tasks, list)


def test_sync_assume_rolls_back() -> None:
    uni = uniko.Uniko.in_memory_sync()
    agent = uni.agent("assistant")

    rows = (
        agent.assume(
            "ASSUME { CREATE (:Fact {fact_id: 'srv-port', subject: 'srv', "
            "predicate: 'port', object: '9090'}) }"
        )
        .then_query("MATCH (f:Fact {subject: 'srv'}) RETURN f")
        .run_sync()
    )
    assert isinstance(rows, list)
    assert len(rows) == 1

    # The hypothetical mutation must not survive into the real graph.
    after = agent.query_sync("MATCH (f:Fact {subject: 'srv'}) RETURN f")
    assert after == []


def test_sync_abduce() -> None:
    uni = uniko.Uniko.in_memory_sync()
    agent = uni.agent("assistant")
    agent.define_rule_sync(
        "reachable", "CREATE RULE reachable AS MATCH (a:Episode) YIELD KEY a"
    )
    result = agent.abduce_sync("ABDUCE reachable WHERE a.kind = 'query'")
    assert isinstance(result, uniko.AbductionResult)
    assert isinstance(result.modifications, list)
    for mod in result.modifications:
        assert "modification" in mod
        assert isinstance(mod["validated"], bool)
        assert isinstance(mod["cost"], float)


def test_sync_call_releases_the_gil() -> None:
    """A `*_sync` call must release the GIL so other Python threads progress.

    A background thread spins a pure-Python counter. If the sync skin held the
    GIL across `block_on`, the counter could not advance during the call; the
    skin's `Python::detach` releases it, so the counter ticks up.
    """
    uni = uniko.Uniko.in_memory_sync()
    agent = uni.agent("assistant")
    session = agent.session("s1")

    stop = threading.Event()
    counter = {"n": 0}

    def spin() -> None:
        while not stop.is_set():
            counter["n"] += 1

    worker = threading.Thread(target=spin)
    worker.start()
    try:
        time.sleep(0.01)  # let the worker reach its loop
        before = counter["n"]
        # A real ingest: tens of ms of GIL-free Rust work.
        session.observe_sync(uniko.Turn("alice", "the counter should keep climbing"))
        advanced = counter["n"] - before
    finally:
        stop.set()
        worker.join()

    assert advanced > 0, "background thread made no progress — GIL was not released"
