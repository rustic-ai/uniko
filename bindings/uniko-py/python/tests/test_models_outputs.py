# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Dragonscale Industries Inc.
"""Output model tests — field fidelity, recursion, serialization, schema.

These use fake duck-typed objects (same attribute surface as the native frozen
snapshots) so they exercise the Pydantic layer without the engine.
"""

from __future__ import annotations

import datetime
import json

from uniko.models import (
    AbductionResult,
    Answer,
    ArtifactView,
    ContextBundle,
    DeletionReport,
    GoalContext,
    GoalView,
    IngestOutcome,
    MessageView,
    ObserveResult,
    RecallItem,
    RecallSource,
    TaskView,
)


# ── fakes (duck-types of the native pyclasses) ─────────────────────────────


class FakeSource:
    kind = "message"
    message_id = "m-1"
    artifact_id = None
    chunk_id = None


class FakeItem:
    node_id = 7
    kind = "observation"
    score = 0.91
    content = "User prefers dark mode."
    sources = [FakeSource()]


class FakeBundle:
    items = [FakeItem()]
    total_tokens = 12
    phase1_only = True
    phase2_only = False
    coverage = 0.5


class FakeAnswer:
    text = "Because you said so."
    model = "gpt-4o-mini"
    input_tokens = 10
    output_tokens = 4
    recorded_episode = None
    context = FakeBundle()

    def citations(self):
        return [FakeSource()]


# ── RecallSource / RecallItem / ContextBundle ──────────────────────────────


def test_recall_source_field_fidelity():
    s = RecallSource.from_native(FakeSource())
    assert s.kind == "message"
    assert s.message_id == "m-1"
    assert s.artifact_id is None and s.chunk_id is None


def test_recall_item_nested_recursion():
    item = RecallItem.from_native(FakeItem())
    assert item.node_id == 7 and item.score == 0.91
    assert len(item.sources) == 1
    assert isinstance(item.sources[0], RecallSource)
    assert item.sources[0].message_id == "m-1"


def test_context_bundle_recursion_and_len():
    bundle = ContextBundle.from_native(FakeBundle())
    assert len(bundle) == 1
    assert isinstance(bundle.items[0], RecallItem)
    assert bundle.coverage == 0.5


def test_context_bundle_model_dump_json_roundtrip():
    bundle = ContextBundle.from_native(FakeBundle())
    data = json.loads(bundle.model_dump_json())
    assert data["items"][0]["sources"][0]["kind"] == "message"
    assert data["total_tokens"] == 12


def test_context_bundle_json_schema_has_defs():
    schema = ContextBundle.model_json_schema()
    assert {"RecallItem", "RecallSource"} <= set(schema.get("$defs", {}))


# ── Answer (custom from_native via citations() method) ──────────────────────


def test_answer_from_native_calls_citations_method():
    answer = Answer.from_native(FakeAnswer())
    assert answer.text == "Because you said so."
    assert answer.model == "gpt-4o-mini"
    assert isinstance(answer.context, ContextBundle)
    assert len(answer.citations) == 1
    assert isinstance(answer.citations[0], RecallSource)


def test_answer_citations_in_schema_and_dump():
    answer = Answer.from_native(FakeAnswer())
    assert "citations" in Answer.model_json_schema()["properties"]
    assert answer.model_dump()["citations"][0]["kind"] == "message"


# ── AbductionResult (list of dicts) ────────────────────────────────────────


class FakeAbduction:
    modifications = [
        {"modification": {"add_edge": "A->B"}, "validated": True, "cost": 0.5},
        {"modification": "raw-string-mod", "validated": False, "cost": 1.25},
    ]


def test_abduction_result_from_native_dicts():
    result = AbductionResult.from_native(FakeAbduction())
    assert len(result) == 2
    assert result.modifications[0].validated is True
    assert result.modifications[0].modification == {"add_edge": "A->B"}
    # free-form: a str modification is accepted unchanged
    assert result.modifications[1].modification == "raw-string-mod"
    assert result.modifications[1].cost == 1.25


# ── remaining output models: field fidelity per model ──────────────────────


def test_observe_result_tuple_and_optionals():
    class FakeObserve:
        message_node_id = 1
        chunk_node_ids = [2, 3]
        session_node_id = 4
        sender_node_id = None
        sender_id = None
        extracted_entities = [(5, "Alice"), (6, "Dolomites")]
        extracted_observations = [7]
        attachments: list = []

    r = ObserveResult.from_native(FakeObserve())
    assert r.extracted_entities == [(5, "Alice"), (6, "Dolomites")]
    assert isinstance(r.extracted_entities[0], tuple)
    assert r.sender_node_id is None


def test_deletion_report():
    class FakeReport:
        nodes_deleted = 1
        edges_deleted = 2
        facts_invalidated = 3
        chains_repaired = 4
        nodes_redacted = 5
        root_existed = True

    r = DeletionReport.from_native(FakeReport())
    assert r.nodes_deleted == 1 and r.root_existed is True


def test_message_view_datetime_passthrough():
    ts = datetime.datetime(2026, 6, 20, 12, 0, tzinfo=datetime.timezone.utc)

    class FakeMessage:
        message_id = "m-1"
        sender_id = "alice"
        content = "hi"
        timestamp = ts
        session_id = "s-1"
        addressed_to = ["bob"]
        attachments = []

    m = MessageView.from_native(FakeMessage())
    assert m.timestamp == ts
    assert m.addressed_to == ["bob"]


def test_artifact_view_optionals():
    class FakeArtifact:
        artifact_id = "a-1"
        kind = "document"
        mime = None
        path = None
        text = "body"
        attached_to_message = None

    a = ArtifactView.from_native(FakeArtifact())
    assert a.text == "body" and a.mime is None


def test_goal_view_metrics_freeform():
    class FakeGoal:
        goal_id = "g-1"
        title = "Ship it"
        description = None
        status = "achieved"
        phase = "completed"
        created_at = None
        deadline = None
        completed_at = None
        metrics = None

    # metrics accepts dict, str, list, and None without ValidationError
    for metrics in ({"x": 1}, "done", [1, 2], None):
        FakeGoal.metrics = metrics
        g = GoalView.from_native(FakeGoal())
        assert g.metrics == metrics
        assert g.phase == "completed"


def test_task_view():
    class FakeTask:
        task_id = "t-1"
        title = "do"
        description = None
        status = "todo"
        phase = "planned"
        priority = 0.5
        goal_id = "g-1"
        created_at = None
        completed_at = None

    t = TaskView.from_native(FakeTask())
    assert t.priority == 0.5 and t.phase == "planned"


def test_goal_context_double_recursion():
    class FakeGoal:
        goal_id = "g-1"
        title = "Ship"
        description = None
        status = "active"
        phase = "active"
        created_at = None
        deadline = None
        completed_at = None
        metrics = None

    class FakeTask:
        task_id = "t-1"
        title = "do"
        description = None
        status = "todo"
        phase = "planned"
        priority = None
        goal_id = "g-1"
        created_at = None
        completed_at = None

    class FakeContext:
        goal = FakeGoal()
        tasks = [FakeTask()]
        sessions = ["s-1"]
        recent_messages = ["m-1"]
        facts = ["f-1"]
        entities = ["Alice"]

    ctx = GoalContext.from_native(FakeContext())
    assert isinstance(ctx.goal, GoalView)
    assert isinstance(ctx.tasks[0], TaskView)
    assert ctx.entities == ["Alice"]


def test_ingest_outcome():
    class FakeOutcome:
        kind = "artifact"
        artifact_id = "a-1"
        artifact_node_id = 9
        chunk_node_ids = [10, 11]
        was_deduplicated = False
        page_count = None
        extraction_failed = None

    o = IngestOutcome.from_native(FakeOutcome())
    assert o.chunk_node_ids == [10, 11] and o.was_deduplicated is False
