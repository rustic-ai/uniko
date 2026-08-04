# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Dragonscale Industries Inc.
"""Pydantic output models — typed, serializable mirrors of the native snapshots.

Every model mirrors a frozen ``#[pyclass]`` returned by ``uniko._uniko`` and adds
``.model_dump()`` / ``.model_dump_json()`` / ``.model_json_schema()`` for FastAPI
and other Pydantic consumers. They are built from a native object with
``Model.from_native(native)``.

``ConfigDict(from_attributes=True)`` is the workhorse: ``model_validate`` reads
attributes straight off the frozen pyclass and nested lists recurse automatically
(each child model carries its own ``from_attributes`` config). Two models need a
custom ``from_native`` because the native shape isn't plain attributes:

* :class:`Answer` — ``citations`` is a *method* on the native object, not a field.
* :class:`AbductionResult` — ``modifications`` is a list of plain dicts.
"""

from __future__ import annotations

import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

# Frozen + from_attributes on every output model. Frozen because these are
# read-only snapshots; mutating them would not write back to the graph.
_OUT = ConfigDict(from_attributes=True, frozen=True)


class RecallSource(BaseModel):
    """Provenance pointer for a recalled item (native ``RecallSource``)."""

    model_config = _OUT

    kind: str
    message_id: Optional[str] = None
    artifact_id: Optional[str] = None
    chunk_id: Optional[str] = None

    @classmethod
    def from_native(cls, native: Any) -> "RecallSource":
        return cls.model_validate(native)


class RecallItem(BaseModel):
    """A recalled node with score and lineage (native ``RecallItem``)."""

    model_config = _OUT

    node_id: int
    kind: str
    score: float
    content: str
    sources: list[RecallSource] = Field(default_factory=list)

    @classmethod
    def from_native(cls, native: Any) -> "RecallItem":
        return cls.model_validate(native)


class ContextBundle(BaseModel):
    """A recall result: scored items plus coverage stats (native ``ContextBundle``)."""

    model_config = _OUT

    items: list[RecallItem] = Field(default_factory=list)
    total_tokens: int
    phase1_only: bool
    phase2_only: bool
    coverage: float

    @classmethod
    def from_native(cls, native: Any) -> "ContextBundle":
        return cls.model_validate(native)

    def __len__(self) -> int:
        return len(self.items)


class Answer(BaseModel):
    """A generated answer plus its grounding context (native ``Answer``).

    ``citations`` is materialized as a real field (the native side exposes it as a
    method) so it appears in ``model_dump()`` and the JSON schema.
    """

    model_config = _OUT

    text: str
    model: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    recorded_episode: Optional[int] = None
    context: ContextBundle
    citations: list[RecallSource] = Field(default_factory=list)

    @classmethod
    def from_native(cls, native: Any) -> "Answer":
        # citations() is a method and context is a nested native pyclass, so we
        # build explicitly rather than relying on model_validate(from_attributes).
        return cls(
            text=native.text,
            model=native.model,
            input_tokens=native.input_tokens,
            output_tokens=native.output_tokens,
            recorded_episode=native.recorded_episode,
            context=ContextBundle.from_native(native.context),
            citations=[RecallSource.model_validate(c) for c in native.citations()],
        )


class ObserveResult(BaseModel):
    """What an ``observe`` produced (native ``ObserveResult``)."""

    model_config = _OUT

    message_node_id: int
    chunk_node_ids: list[int]
    session_node_id: int
    sender_node_id: Optional[int] = None
    sender_id: Optional[str] = None
    # (node_id, entity_name) pairs.
    extracted_entities: list[tuple[int, str]] = Field(default_factory=list)
    extracted_observations: list[int] = Field(default_factory=list)
    attachment_count: int

    @classmethod
    def from_native(cls, native: Any) -> "ObserveResult":
        return cls.model_validate(native)


class DeletionReport(BaseModel):
    """Counts from a forget/delete/purge operation (native ``DeletionReport``)."""

    model_config = _OUT

    nodes_deleted: int
    edges_deleted: int
    facts_invalidated: int
    chains_repaired: int
    nodes_redacted: int
    root_existed: bool

    @classmethod
    def from_native(cls, native: Any) -> "DeletionReport":
        return cls.model_validate(native)


class FinalizeReport(BaseModel):
    """What ``Session.finalize()`` built (native ``FinalizeReport``)."""

    model_config = _OUT

    transcript_chunks: list[int]
    observation_chunks: list[int]
    rebuilt: bool
    ended_at: Optional[datetime.datetime] = None

    @classmethod
    def from_native(cls, native: Any) -> "FinalizeReport":
        return cls.model_validate(native)


class CycleStats(BaseModel):
    """What one consolidation cycle derived (native ``CycleStats``)."""

    model_config = _OUT

    observations_processed: int
    facts_created: int
    facts_reinforced: int
    facts_invalidated: int
    drift_alerts: int

    @classmethod
    def from_native(cls, native: Any) -> "CycleStats":
        return cls.model_validate(native)


class MessageView(BaseModel):
    """A stored message (native ``MessageView``)."""

    model_config = _OUT

    message_id: str
    sender_id: str
    content: str
    timestamp: datetime.datetime
    session_id: str
    addressed_to: list[str] = Field(default_factory=list)
    attachments: list[str] = Field(default_factory=list)

    @classmethod
    def from_native(cls, native: Any) -> "MessageView":
        return cls.model_validate(native)


class ArtifactView(BaseModel):
    """A stored artifact / document (native ``ArtifactView``)."""

    model_config = _OUT

    artifact_id: str
    kind: str
    mime: Optional[str] = None
    path: Optional[str] = None
    text: str
    attached_to_message: Optional[str] = None

    @classmethod
    def from_native(cls, native: Any) -> "ArtifactView":
        return cls.model_validate(native)


class GoalView(BaseModel):
    """A goal (native ``GoalView``). ``metrics`` is free-form JSON."""

    model_config = _OUT

    goal_id: str
    title: str
    description: Optional[str] = None
    status: str
    phase: str
    created_at: Optional[datetime.datetime] = None
    deadline: Optional[datetime.datetime] = None
    completed_at: Optional[datetime.datetime] = None
    metrics: Optional[Any] = None

    @classmethod
    def from_native(cls, native: Any) -> "GoalView":
        return cls.model_validate(native)


class TaskView(BaseModel):
    """A task (native ``TaskView``)."""

    model_config = _OUT

    task_id: str
    title: str
    description: Optional[str] = None
    status: str
    phase: str
    priority: Optional[float] = None
    goal_id: Optional[str] = None
    created_at: Optional[datetime.datetime] = None
    completed_at: Optional[datetime.datetime] = None

    @classmethod
    def from_native(cls, native: Any) -> "TaskView":
        return cls.model_validate(native)


class GoalContext(BaseModel):
    """A goal plus its surrounding context (native ``GoalContext``)."""

    model_config = _OUT

    goal: GoalView
    tasks: list[TaskView] = Field(default_factory=list)
    sessions: list[str] = Field(default_factory=list)
    recent_messages: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)

    @classmethod
    def from_native(cls, native: Any) -> "GoalContext":
        return cls.model_validate(native)


class IngestOutcome(BaseModel):
    """Result of ingesting an artifact (native ``IngestOutcome``)."""

    model_config = _OUT

    kind: str
    artifact_id: str
    artifact_node_id: int
    chunk_node_ids: list[int] = Field(default_factory=list)
    was_deduplicated: bool
    page_count: Optional[int] = None
    extraction_failed: Optional[bool] = None

    @classmethod
    def from_native(cls, native: Any) -> "IngestOutcome":
        return cls.model_validate(native)


class AbducedModification(BaseModel):
    """One proposed graph modification from an abductive query.

    ``modification`` is free-form JSON (a dict or a str on the native side), so it
    is typed ``Any`` to avoid rejecting valid payloads.
    """

    model_config = ConfigDict(frozen=True)

    modification: Any
    validated: bool
    cost: float


class AbductionResult(BaseModel):
    """Ranked, validated modifications from an abductive query (native ``AbductionResult``)."""

    model_config = ConfigDict(frozen=True)

    modifications: list[AbducedModification] = Field(default_factory=list)

    @classmethod
    def from_native(cls, native: Any) -> "AbductionResult":
        # native.modifications is a list of plain dicts {modification, validated, cost}.
        return cls(
            modifications=[
                AbducedModification.model_validate(m) for m in native.modifications
            ]
        )

    def __len__(self) -> int:
        return len(self.modifications)


__all__ = [
    "RecallSource",
    "RecallItem",
    "ContextBundle",
    "Answer",
    "ObserveResult",
    "DeletionReport",
    "MessageView",
    "ArtifactView",
    "GoalView",
    "TaskView",
    "GoalContext",
    "IngestOutcome",
    "AbducedModification",
    "AbductionResult",
]
