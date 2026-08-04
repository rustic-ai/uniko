# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Dragonscale Industries Inc.
"""Pydantic IO layer for the uniko SDK.

A pure-Python overlay on the native ``uniko._uniko`` extension that adds
validation, JSON-schema/serialization (FastAPI-ready), and Pydantic ergonomics:

* **Output models** (:mod:`~uniko.models.outputs`) — typed mirrors of the native
  snapshots; build with ``Model.from_native(native)``.
* **Input specs** (:mod:`~uniko.models.inputs`) — validated payloads that build
  native builders via ``spec.to_native()``.
* **Config** (:mod:`~uniko.models.config`) — ``UnikoConfigModel``.
* **Adapters** (:mod:`~uniko.models.adapters`) — spec-in/model-out helpers.
* **Typed handles** (:mod:`~uniko.models.typed`) — ``TypedUniko`` & friends that
  auto-accept specs and auto-return models, so callers never touch
  ``from_native``/``to_native``.

These live in the ``uniko.models`` namespace and never clobber the native
top-level names (``uniko.ContextBundle`` remains the native class).
"""

from __future__ import annotations

from .adapters import (
    answer_in,
    answer_in_sync,
    create_goal,
    create_goal_sync,
    create_task,
    create_task_sync,
    ingest,
    ingest_sync,
    observe,
    observe_sync,
    recall_in,
    recall_in_sync,
)
from .config import UnikoConfigModel
from .inputs import (
    GoalSpec,
    IngestSourceSpec,
    LlmSpecAdapter,
    LlmSpecModel,
    ScopeSpec,
    TaskSpec,
    TurnSpec,
    llm_to_native,
)
from .outputs import (
    AbducedModification,
    AbductionResult,
    Answer,
    ArtifactView,
    ContextBundle,
    DeletionReport,
    FinalizeReport,
    CycleStats,
    GoalContext,
    GoalView,
    IngestOutcome,
    MessageView,
    ObserveResult,
    RecallItem,
    RecallSource,
    TaskView,
)
from .typed import (
    TypedAgent,
    TypedData,
    TypedGoals,
    TypedSession,
    TypedUniko,
    TypedUnikoBuilder,
)

# Back-compat aliases for the original spike names (deprecated; kept one release).
ContextBundleModel = ContextBundle
RecallItemModel = RecallItem
RecallSourceModel = RecallSource

__all__ = [
    # Output models
    "RecallSource",
    "RecallItem",
    "ContextBundle",
    "Answer",
    "ObserveResult",
    "DeletionReport",
    "FinalizeReport",
    "CycleStats",
    "MessageView",
    "ArtifactView",
    "GoalView",
    "TaskView",
    "GoalContext",
    "IngestOutcome",
    "AbducedModification",
    "AbductionResult",
    # Config
    "UnikoConfigModel",
    # Input specs
    "GoalSpec",
    "TaskSpec",
    "ScopeSpec",
    "TurnSpec",
    "IngestSourceSpec",
    "LlmSpecModel",
    "LlmSpecAdapter",
    "llm_to_native",
    # Adapters
    "create_goal",
    "create_goal_sync",
    "create_task",
    "create_task_sync",
    "observe",
    "observe_sync",
    "ingest",
    "ingest_sync",
    "recall_in",
    "recall_in_sync",
    "answer_in",
    "answer_in_sync",
    # Typed handles
    "TypedUniko",
    "TypedUnikoBuilder",
    "TypedAgent",
    "TypedSession",
    "TypedData",
    "TypedGoals",
    # Back-compat aliases
    "ContextBundleModel",
    "RecallItemModel",
    "RecallSourceModel",
]
