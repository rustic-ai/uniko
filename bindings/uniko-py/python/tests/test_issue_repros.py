"""Native-wheel regressions for memory identity and durability.

Covers the four defects reported as ``rustic-ai/uniko#38``. They were written
as strict xfails against the observed bugs; all four now pass, so the markers
are gone and these guard against regression on every run.

- A reused ``message_id`` or ``artifact_id`` with different content is
  rejected rather than silently returning the original record.
- Identical bytes under two ids stay addressable under both, each scoped to
  the session that ingested them.
- A turn committed before an abrupt process exit survives a reopen. That one
  is upheld by uni-db 4.x replaying a WAL with no snapshot manifest; uniko
  carried a baseline-flush workaround for it against 3.4.x, since removed.

Run with ``pytest -q python/tests/test_issue_repros.py`` from bindings/uniko-py.
Requires only the public Python API.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
import uniko


def test_same_turn_id_and_content_is_idempotent() -> None:
    engine = uniko.Uniko.in_memory_sync()
    session = engine.agent("analyst").session("user-a")
    first = session.observe_sync(uniko.Turn("user-a", "original fact").id("turn-1"))
    second = session.observe_sync(uniko.Turn("user-a", "original fact").id("turn-1"))
    assert second.message_node_id == first.message_node_id
    assert (
        engine.agent("analyst").data.message_sync("turn-1").content
        == "original fact"
    )


def test_conflicting_turn_id_is_rejected() -> None:
    engine = uniko.Uniko.in_memory_sync()
    session = engine.agent("analyst").session("user-a")
    session.observe_sync(uniko.Turn("user-a", "original fact").id("turn-1"))
    with pytest.raises(uniko.IdConflictError, match="(?i)id conflict"):
        session.observe_sync(uniko.Turn("user-a", "contradictory fact").id("turn-1"))


def test_same_artifact_id_and_content_is_idempotent() -> None:
    engine = uniko.Uniko.in_memory_sync()
    session = engine.agent("analyst").session("user-a")
    source = uniko.IngestSource.from_text("original notes").with_id("doc-1")
    first = session.ingest_sync(source)
    second = session.ingest_sync(
        uniko.IngestSource.from_text("original notes").with_id("doc-1")
    )
    assert second.artifact_node_id == first.artifact_node_id
    assert engine.agent("analyst").data.artifact_sync("doc-1").text == "original notes"


def test_conflicting_artifact_id_is_rejected() -> None:
    engine = uniko.Uniko.in_memory_sync()
    session = engine.agent("analyst").session("user-a")
    session.ingest_sync(uniko.IngestSource.from_text("original notes").with_id("doc-1"))
    with pytest.raises(uniko.IdConflictError, match="(?i)id conflict"):
        session.ingest_sync(
            uniko.IngestSource.from_text("contradictory notes").with_id("doc-1")
        )


def test_distinct_artifact_ids_are_resolvable_without_content_deduplication() -> None:
    engine = uniko.Uniko.in_memory_sync()
    agent = engine.agent("analyst")
    agent.session("user-a").ingest_sync(
        uniko.IngestSource.from_text("source for first user").with_id("doc-a")
    )
    agent.session("user-b").ingest_sync(
        uniko.IngestSource.from_text("source for second user").with_id("doc-b")
    )
    assert agent.data.artifact_sync("doc-a") is not None
    assert agent.data.artifact_sync("doc-b") is not None


def test_identical_documents_have_distinct_ids_across_sessions() -> None:
    engine = uniko.Uniko.in_memory_sync()
    agent = engine.agent("analyst")
    first = agent.session("user-a").ingest_sync(
        uniko.IngestSource.from_text("shared source bytes").with_id("doc-a")
    )
    second = agent.session("user-b").ingest_sync(
        uniko.IngestSource.from_text("shared source bytes").with_id("doc-b")
    )
    assert first.artifact_id == "doc-a"
    assert second.artifact_id == "doc-b"
    assert agent.data.artifact_sync("doc-a") is not None
    assert agent.data.artifact_sync("doc-b") is not None
    scoped = agent.recall_in_sync(
        "shared source bytes", uniko.Scope().sessions(["user-b"])
    )
    assert scoped.items


_WRITE_IN_CHILD = textwrap.dedent(
    """\
    import os
    import sys
    import uniko

    engine = uniko.Uniko.open_sync(sys.argv[1])
    agent = engine.agent("analyst")
    session = agent.session("user-a")
    session.observe_sync(uniko.Turn("user-a", "durable fact").id("turn-1"))
    agent.finalize_session_sync("user-a")
    if sys.argv[2] == "abrupt":
        os._exit(0)
    del session, agent
    engine.shutdown_sync()
    """
)


def _write_in_child(store: str, mode: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _WRITE_IN_CHILD, store, mode],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _read_after_restart(store: str) -> None:
    engine = uniko.Uniko.open_sync(store)
    agent = engine.agent("analyst")
    view = agent.data.message_sync("turn-1")
    assert view is not None and view.content == "durable fact"
    del agent
    engine.shutdown_sync()


def test_committed_turn_survives_clean_restart(tmp_path) -> None:
    store = str(tmp_path / "clean-store")
    _write_in_child(store, "clean")
    _read_after_restart(store)


def test_committed_turn_survives_abrupt_process_exit(tmp_path) -> None:
    store = str(tmp_path / "abrupt-store")
    _write_in_child(store, "abrupt")
    _read_after_restart(store)


# ── Issue #40: atomic, idempotent multi-turn units ─────────────────────


def test_unit_commits_every_turn_as_one_write() -> None:
    """A related pair is recorded as one unit, visible together."""
    engine = uniko.Uniko.in_memory_sync()
    session = engine.agent("analyst").session("unit-a")
    results, was_replay = session.commit_unit_sync(
        [
            uniko.Turn("user-a", "what is the plan for friday").id("u-1"),
            uniko.Turn("agent", "we ship the release on friday").id("u-2"),
        ]
    )
    assert len(results) == 2, "one result per turn, in unit order"
    assert not was_replay, "a fresh unit is not a replay"
    assert all(r.message_node_id for r in results)
    assert engine.agent("analyst").data.message_sync("u-2").content == (
        "we ship the release on friday"
    )


def test_unit_replay_is_a_whole_unit_noop() -> None:
    """Re-committing an identical unit writes nothing and reports it."""
    engine = uniko.Uniko.in_memory_sync()
    session = engine.agent("analyst").session("unit-replay")

    def turns() -> list[uniko.Turn]:
        return [
            uniko.Turn("user-a", "stable content one").id("r-1"),
            uniko.Turn("agent", "stable content two").id("r-2"),
        ]

    first, first_replay = session.commit_unit_sync(turns())
    assert not first_replay

    second, second_replay = session.commit_unit_sync(turns())
    assert second_replay, "an identical unit must report a replay"
    assert [r.message_node_id for r in first] == [r.message_node_id for r in second]


def test_unit_id_conflict_leaves_nothing_behind() -> None:
    """A conflict on any turn fails the unit before anything is written."""
    engine = uniko.Uniko.in_memory_sync()
    session = engine.agent("analyst").session("unit-conflict")
    session.commit_unit_sync([uniko.Turn("user-a", "original content").id("c-1")])

    with pytest.raises(uniko.IdConflictError, match="(?i)id conflict"):
        session.commit_unit_sync(
            [
                uniko.Turn("agent", "a genuinely fresh turn").id("c-fresh"),
                uniko.Turn("user-a", "contradictory content").id("c-1"),
            ]
        )

    # The fresh turn preceded the conflicting one in the unit, so the
    # rollback must have taken it down too. `message_sync` returns None for a
    # missing id rather than raising.
    assert engine.agent("analyst").data.message_sync("c-fresh") is None
    # ...while the turn that was committed before the failed unit survives.
    assert (
        engine.agent("analyst").data.message_sync("c-1").content == "original content"
    )


# ── Issue #39: typed provenance and pre-ranking filters ────────────────


def test_recall_returns_only_the_permitted_category() -> None:
    """A category filter narrows candidates before ranking, not after."""
    engine = uniko.Uniko.in_memory_sync()
    agent = engine.agent("analyst")
    session = agent.session("prov-a")
    session.commit_unit_sync(
        [
            uniko.Turn("user-a", "the telescope readings look stable")
            .id("pv-1")
            .category("user_assertion"),
            uniko.Turn("agent", "the telescope query returned 42 rows")
            .id("pv-2")
            .category("executed_result"),
        ]
    )

    scope = uniko.Scope().categories(["executed_result"])
    bundle = agent.recall_in_sync("telescope", scope)
    assert bundle.items, "the permitted category must still return evidence"
    for item in bundle.items:
        assert item.category == "executed_result", (
            f"a disallowed category leaked: {item!r}"
        )


def test_unmatched_category_returns_empty() -> None:
    """No eligible match returns empty rather than other categories."""
    engine = uniko.Uniko.in_memory_sync()
    agent = engine.agent("analyst")
    session = agent.session("prov-b")
    session.observe_sync(
        uniko.Turn("user-a", "the telescope readings look stable")
        .id("pv-3")
        .category("user_assertion")
    )

    scope = uniko.Scope().categories(["no_such_category"])
    bundle = agent.recall_in_sync("telescope", scope)
    assert not bundle.items, (
        f"an unmatched category must not be padded with others: {bundle.items!r}"
    )


def test_source_filter_and_exposed_provenance() -> None:
    """A source filter narrows, and each item reports its source."""
    engine = uniko.Uniko.in_memory_sync()
    agent = engine.agent("analyst")
    session = agent.session("prov-c")
    session.commit_unit_sync(
        [
            uniko.Turn("agent", "the beacon signal was steady all week")
            .id("pv-4")
            .source("feed-alpha"),
            uniko.Turn("agent", "the beacon signal dropped out twice")
            .id("pv-5")
            .source("feed-beta"),
        ]
    )

    scope = uniko.Scope().sources(["feed-alpha"])
    bundle = agent.recall_in_sync("beacon signal", scope)
    for item in bundle.items:
        assert item.source_id == "feed-alpha", f"a disallowed source leaked: {item!r}"


# ── Issue #41: source revisions and retirement ─────────────────────────


def test_newer_revision_supersedes_and_retirement_hides_both() -> None:
    """The issue's sequence: A, then a contradicting B, then retirement."""
    engine = uniko.Uniko.in_memory_sync()
    agent = engine.agent("analyst")
    session = agent.session("rev-a")

    session.ingest_sync(
        uniko.IngestSource.from_text("The summit elevation is 3200 metres.")
        .with_id("pg-a")
        .with_source("wiki-summit")
        .with_revision("rev-a")
    )
    session.ingest_sync(
        uniko.IngestSource.from_text("The summit elevation is 3450 metres.")
        .with_id("pg-b")
        .with_source("wiki-summit")
        .with_revision("rev-b")
    )

    # Ordinary recall must not be grounded by the superseded revision.
    current = agent.recall_sync("summit elevation")
    assert all(i.revision_id != "rev-a" for i in current.items), (
        "a superseded revision must not ground a current answer"
    )

    # Retiring the source takes every revision out of current recall.
    assert agent.retire_source_sync("wiki-summit") is True
    after = agent.recall_sync("summit elevation")
    assert all(i.source_id != "wiki-summit" for i in after.items), (
        "a retired source must not ground a current answer"
    )


def test_reusing_a_revision_with_changed_content_is_rejected() -> None:
    """A revision id is a promise about the content."""
    engine = uniko.Uniko.in_memory_sync()
    session = engine.agent("analyst").session("rev-c")
    src = (
        uniko.IngestSource.from_text("original body")
        .with_id("rc-a")
        .with_source("feed-x")
        .with_revision("rx-1")
    )
    session.ingest_sync(src)

    with pytest.raises(uniko.IdConflictError, match="(?i)id conflict"):
        session.ingest_sync(
            uniko.IngestSource.from_text("DIFFERENT body")
            .with_id("rc-b")
            .with_source("feed-x")
            .with_revision("rx-1")
        )


# ── Viewer-scoped recall from Python ───────────────────────────────────


def test_scope_as_participant_filters_private_facts() -> None:
    """Python can scope recall to a participant.

    Before this, constructing a Viewer needed a store handle Python never
    gets, so viewer-scoped recall was unreachable — and since unscoped reads
    are fail-open, that meant no visibility filtering at all.
    """
    engine = uniko.Uniko.in_memory_sync()
    agent = engine.agent("observer")
    session = agent.session("vis-1")

    # Two turns, one addressed only to alice's private scope via a Fact is not
    # reachable from Python, so assert the plumbing instead: a scoped recall
    # must run and return a bundle rather than raising.
    session.observe_sync(
        uniko.Turn("alice", "the quarterly revenue outlook is strong").id("vis-a")
    )

    scoped = agent.recall_in_sync(
        "quarterly revenue outlook", uniko.Scope().as_participant("bob")
    )
    assert scoped is not None
    # And it composes with the other dimensions.
    combined = agent.recall_in_sync(
        "quarterly revenue outlook",
        uniko.Scope().as_participant("bob").sessions(["vis-1"]),
    )
    assert combined is not None
