# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Dragonscale Industries Inc.
"""String literal types for the closed sets the native API passes as ``str``.

The extension hands these values over as plain strings; these aliases let a
type checker catch a misspelt phase or kind before it fails at runtime::

    from uniko.types import GoalPhase

    def phase_label(phase: GoalPhase) -> str: ...
"""

from __future__ import annotations

from typing import Literal, TypedDict

#: A goal's lifecycle phase (``GoalView.phase``, ``Goals.in_phase``).
GoalPhase = Literal["planned", "active", "completed", "abandoned"]

#: A task's lifecycle phase (``TaskView.phase``, ``Goals.tasks_in``).
TaskPhase = Literal["planned", "active", "completed", "blocked"]

#: What a recalled item is (``RecallItem.kind``).
RecallKind = Literal[
    "chunk", "observation", "fact", "procedure", "topic", "episode", "message", "other"
]

#: Where a recalled item's content came from (``RecallSource.kind``).
RecallSourceKind = Literal["message", "attachment", "document", "session", "tool"]

#: What ``Session.ingest`` produced (``IngestOutcome.kind``).
IngestKind = Literal["artifact", "pdf"]

#: A streaming-pipeline worker's state (``WorkerHealth.status``).
WorkerStatus = Literal["healthy", "degraded", "backpressured", "stalled"]

#: The LLM circuit breaker's state (``PipelineHealth.llm_circuit``).
CircuitState = Literal["closed", "open", "half_open"]

#: What ``Agent.register`` did (``Registration.outcome``).
RegistrationOutcome = Literal["created", "upgraded", "existing"]

#: An in-place store migration (``Uniko.migrate_store``).
MigrationName = Literal["enable_hybrid", "rebuild_vector_indexes"]

#: How a store differs from a config (``StoreDifference.verdict``):
#: ``"unknown"`` is a verdict this build doesn't know.
VerdictName = Literal["incompatible", "needs_migration", "unknown"]


class MigrationProgressEvent(TypedDict):
    """What a ``Uniko.migrate_store`` ``progress`` callback is called with."""

    migration: MigrationName
    phase: Literal["schema", "rebuild", "backfill", "done"]
    label: str | None
    done: int
    total: int


#: Where a dead letter is in its lifecycle (``DeadLetter.status``).
DeadLetterStatus = Literal["pending", "retrying", "exhausted", "resolved", "legacy"]

__all__ = [
    "CircuitState",
    "DeadLetterStatus",
    "RegistrationOutcome",
    "GoalPhase",
    "MigrationName",
    "MigrationProgressEvent",
    "IngestKind",
    "RecallKind",
    "RecallSourceKind",
    "TaskPhase",
    "VerdictName",
    "WorkerStatus",
]
