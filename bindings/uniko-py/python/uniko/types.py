# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Dragonscale Industries Inc.
"""String literal types for the closed sets the native API passes as ``str``.

The extension hands these values over as plain strings; these aliases let a
type checker catch a misspelt phase or kind before it fails at runtime::

    from uniko.types import GoalPhase

    def phase_label(phase: GoalPhase) -> str: ...
"""

from __future__ import annotations

from typing import Literal

#: A goal's lifecycle phase (``GoalView.phase``, ``Goals.in_phase``).
GoalPhase = Literal["planned", "active", "completed", "abandoned"]

#: A task's lifecycle phase (``TaskView.phase``, ``Goals.tasks_in``).
TaskPhase = Literal["planned", "active", "completed", "blocked"]

#: What a recalled item is (``RecallItem.kind``).
RecallKind = Literal[
    "chunk", "observation", "fact", "procedure", "topic", "episode", "message", "other"
]

#: Where a recalled item's content came from (``RecallSource.kind``).
RecallSourceKind = Literal["message", "attachment", "document"]

#: What ``Session.ingest`` produced (``IngestOutcome.kind``).
IngestKind = Literal["artifact", "pdf"]

#: A streaming-pipeline worker's state (``WorkerHealth.status``).
WorkerStatus = Literal["healthy", "degraded", "backpressured", "stalled"]

#: The LLM circuit breaker's state (``PipelineHealth.llm_circuit``).
CircuitState = Literal["closed", "open", "half_open"]

__all__ = [
    "CircuitState",
    "GoalPhase",
    "IngestKind",
    "RecallKind",
    "RecallSourceKind",
    "TaskPhase",
    "WorkerStatus",
]
