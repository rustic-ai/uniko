# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Dragonscale Industries Inc.
"""Typed wrapper handles over the native ``uniko._uniko`` surface.

Each ``Typed*`` class wraps a native handle (``self._inner``) and applies one
mechanical rule: native methods that return a snapshot return the corresponding
Pydantic model instead; methods that accept a builder accept a *spec* (or an
already-native object); and untyped surfaces (Cypher rows, rule params, raw
bytes) pass straight through.

Start typed from the top::

    uni = await TypedUniko.in_memory()
    agent = uni.agent("assistant")
    bundle = await agent.recall("hiking")        # -> ContextBundle model
    await agent.session("s1").observe(TurnSpec(sender_id="a", content="hi"))

Maintenance note: this layer mirrors an evolving native surface — it must track
changes to ``python/uniko/_uniko.pyi``.
"""

from __future__ import annotations

from typing import Any, Optional

from .config import UnikoConfigModel
from .outputs import (
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
    TaskView,
)


def _native_arg(x: Any) -> Any:
    """Coerce a spec (``*Spec`` / ``LlmSpecModel``) to its native builder; pass
    an already-native object through unchanged."""
    return x.to_native() if hasattr(x, "to_native") else x


# ── Data ───────────────────────────────────────────────────────────────────


class TypedData:
    """Typed wrapper over the native ``Data`` handle."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    async def message(self, message_id: str) -> Optional[MessageView]:
        m = await self._inner.message(message_id)
        return MessageView.from_native(m) if m is not None else None

    def message_sync(self, message_id: str) -> Optional[MessageView]:
        m = self._inner.message_sync(message_id)
        return MessageView.from_native(m) if m is not None else None

    async def artifact(self, artifact_id: str) -> Optional[ArtifactView]:
        a = await self._inner.artifact(artifact_id)
        return ArtifactView.from_native(a) if a is not None else None

    def artifact_sync(self, artifact_id: str) -> Optional[ArtifactView]:
        a = self._inner.artifact_sync(artifact_id)
        return ArtifactView.from_native(a) if a is not None else None

    async def artifact_bytes(self, artifact_id: str) -> Optional[bytes]:
        return await self._inner.artifact_bytes(artifact_id)

    def artifact_bytes_sync(self, artifact_id: str) -> Optional[bytes]:
        return self._inner.artifact_bytes_sync(artifact_id)


# ── Goals ──────────────────────────────────────────────────────────────────


class TypedGoals:
    """Typed wrapper over the native ``Goals`` handle."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    # Reads → models.
    async def all(self) -> list[GoalView]:
        return [GoalView.from_native(g) for g in await self._inner.all()]

    def all_sync(self) -> list[GoalView]:
        return [GoalView.from_native(g) for g in self._inner.all_sync()]

    async def in_phase(self, phase: str) -> list[GoalView]:
        return [GoalView.from_native(g) for g in await self._inner.in_phase(phase)]

    def in_phase_sync(self, phase: str) -> list[GoalView]:
        return [GoalView.from_native(g) for g in self._inner.in_phase_sync(phase)]

    async def active(self) -> list[GoalView]:
        return [GoalView.from_native(g) for g in await self._inner.active()]

    def active_sync(self) -> list[GoalView]:
        return [GoalView.from_native(g) for g in self._inner.active_sync()]

    async def planned(self) -> list[GoalView]:
        return [GoalView.from_native(g) for g in await self._inner.planned()]

    def planned_sync(self) -> list[GoalView]:
        return [GoalView.from_native(g) for g in self._inner.planned_sync()]

    async def completed(self) -> list[GoalView]:
        return [GoalView.from_native(g) for g in await self._inner.completed()]

    def completed_sync(self) -> list[GoalView]:
        return [GoalView.from_native(g) for g in self._inner.completed_sync()]

    async def get(self, goal_id: str) -> Optional[GoalView]:
        g = await self._inner.get(goal_id)
        return GoalView.from_native(g) if g is not None else None

    def get_sync(self, goal_id: str) -> Optional[GoalView]:
        g = self._inner.get_sync(goal_id)
        return GoalView.from_native(g) if g is not None else None

    async def tasks(self) -> list[TaskView]:
        return [TaskView.from_native(t) for t in await self._inner.tasks()]

    def tasks_sync(self) -> list[TaskView]:
        return [TaskView.from_native(t) for t in self._inner.tasks_sync()]

    async def tasks_in(self, phase: str) -> list[TaskView]:
        return [TaskView.from_native(t) for t in await self._inner.tasks_in(phase)]

    def tasks_in_sync(self, phase: str) -> list[TaskView]:
        return [TaskView.from_native(t) for t in self._inner.tasks_in_sync(phase)]

    async def tasks_of(self, goal_id: str) -> list[TaskView]:
        return [TaskView.from_native(t) for t in await self._inner.tasks_of(goal_id)]

    def tasks_of_sync(self, goal_id: str) -> list[TaskView]:
        return [TaskView.from_native(t) for t in self._inner.tasks_of_sync(goal_id)]

    async def context(self, goal_id: str) -> Optional[GoalContext]:
        c = await self._inner.context(goal_id)
        return GoalContext.from_native(c) if c is not None else None

    def context_sync(self, goal_id: str) -> Optional[GoalContext]:
        c = self._inner.context_sync(goal_id)
        return GoalContext.from_native(c) if c is not None else None

    # Transitions → bool passthrough.
    async def start(self, goal_id: str) -> bool:
        return await self._inner.start(goal_id)

    def start_sync(self, goal_id: str) -> bool:
        return self._inner.start_sync(goal_id)

    async def abandon(self, goal_id: str) -> bool:
        return await self._inner.abandon(goal_id)

    def abandon_sync(self, goal_id: str) -> bool:
        return self._inner.abandon_sync(goal_id)

    async def set_status(self, goal_id: str, status: str) -> bool:
        return await self._inner.set_status(goal_id, status)

    def set_status_sync(self, goal_id: str, status: str) -> bool:
        return self._inner.set_status_sync(goal_id, status)

    async def complete(self, goal_id: str, result: Any = None) -> bool:
        return await self._inner.complete(goal_id, result)

    def complete_sync(self, goal_id: str, result: Any = None) -> bool:
        return self._inner.complete_sync(goal_id, result)

    async def start_task(self, task_id: str) -> bool:
        return await self._inner.start_task(task_id)

    def start_task_sync(self, task_id: str) -> bool:
        return self._inner.start_task_sync(task_id)

    async def block_task(self, task_id: str) -> bool:
        return await self._inner.block_task(task_id)

    def block_task_sync(self, task_id: str) -> bool:
        return self._inner.block_task_sync(task_id)

    async def complete_task(self, task_id: str) -> bool:
        return await self._inner.complete_task(task_id)

    def complete_task_sync(self, task_id: str) -> bool:
        return self._inner.complete_task_sync(task_id)

    async def set_task_status(self, task_id: str, status: str) -> bool:
        return await self._inner.set_task_status(task_id, status)

    def set_task_status_sync(self, task_id: str, status: str) -> bool:
        return self._inner.set_task_status_sync(task_id, status)

    # Creation ← specs.
    async def create(self, spec: Any) -> int:
        return await self._inner.create(spec.title, **spec.to_create_kwargs())

    def create_sync(self, spec: Any) -> int:
        return self._inner.create_sync(spec.title, **spec.to_create_kwargs())

    async def create_task(self, spec: Any) -> int:
        return await self._inner.create_task(spec.title, **spec.to_create_task_kwargs())

    def create_task_sync(self, spec: Any) -> int:
        return self._inner.create_task_sync(spec.title, **spec.to_create_task_kwargs())


# ── Session ────────────────────────────────────────────────────────────────


class TypedSession:
    """Typed wrapper over the native ``Session`` handle."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    @property
    def session_id(self) -> str:
        return self._inner.session_id

    async def observe(self, turn: Any) -> ObserveResult:
        return ObserveResult.from_native(await self._inner.observe(_native_arg(turn)))

    def observe_sync(self, turn: Any) -> ObserveResult:
        return ObserveResult.from_native(self._inner.observe_sync(_native_arg(turn)))

    async def ingest(self, source: Any) -> IngestOutcome:
        return IngestOutcome.from_native(await self._inner.ingest(_native_arg(source)))

    def ingest_sync(self, source: Any) -> IngestOutcome:
        return IngestOutcome.from_native(self._inner.ingest_sync(_native_arg(source)))

    async def submit(self, turn: Any) -> None:
        await self._inner.submit(_native_arg(turn))

    def submit_sync(self, turn: Any) -> None:
        self._inner.submit_sync(_native_arg(turn))

    async def submit_source(self, source: Any) -> None:
        await self._inner.submit_source(_native_arg(source))

    def submit_source_sync(self, source: Any) -> None:
        self._inner.submit_source_sync(_native_arg(source))

    async def flush(self) -> None:
        await self._inner.flush()

    def flush_sync(self) -> None:
        self._inner.flush_sync()

    async def __aenter__(self) -> "TypedSession":
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return await self._inner.__aexit__(exc_type, exc, tb)

    def __enter__(self) -> "TypedSession":
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return self._inner.__exit__(exc_type, exc, tb)

    async def finalize(self) -> FinalizeReport:
        return FinalizeReport.from_native(await self._inner.finalize())

    def finalize_sync(self) -> FinalizeReport:
        return FinalizeReport.from_native(self._inner.finalize_sync())

    async def summarize(self) -> Optional[int]:
        return await self._inner.summarize()

    def summarize_sync(self) -> Optional[int]:
        return self._inner.summarize_sync()

    async def forget_turn(self, message_id: str) -> DeletionReport:
        return DeletionReport.from_native(await self._inner.forget_turn(message_id))

    def forget_turn_sync(self, message_id: str) -> DeletionReport:
        return DeletionReport.from_native(self._inner.forget_turn_sync(message_id))

    async def delete_turn(self, message_id: str) -> DeletionReport:
        return DeletionReport.from_native(await self._inner.delete_turn(message_id))

    def delete_turn_sync(self, message_id: str) -> DeletionReport:
        return DeletionReport.from_native(self._inner.delete_turn_sync(message_id))

    async def delete_document(self, artifact_id: str) -> DeletionReport:
        return DeletionReport.from_native(
            await self._inner.delete_document(artifact_id)
        )

    def delete_document_sync(self, artifact_id: str) -> DeletionReport:
        return DeletionReport.from_native(self._inner.delete_document_sync(artifact_id))


# ── Agent ──────────────────────────────────────────────────────────────────


class TypedAgent:
    """Typed wrapper over the native ``Agent`` handle."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    @property
    def agent_id(self) -> str:
        return self._inner.agent_id

    @property
    def data(self) -> TypedData:
        return TypedData(self._inner.data)

    @property
    def goals(self) -> TypedGoals:
        return TypedGoals(self._inner.goals)

    def session(self, session_id: str) -> TypedSession:
        return TypedSession(self._inner.session(session_id))

    async def recall(self, query: str) -> ContextBundle:
        return ContextBundle.from_native(await self._inner.recall(query))

    def recall_sync(self, query: str) -> ContextBundle:
        return ContextBundle.from_native(self._inner.recall_sync(query))

    async def answer(self, question: str) -> Answer:
        return Answer.from_native(await self._inner.answer(question))

    def answer_sync(self, question: str) -> Answer:
        return Answer.from_native(self._inner.answer_sync(question))

    async def recall_in(self, query: str, scope: Any) -> ContextBundle:
        return ContextBundle.from_native(
            await self._inner.recall_in(query, _native_arg(scope))
        )

    def recall_in_sync(self, query: str, scope: Any) -> ContextBundle:
        return ContextBundle.from_native(
            self._inner.recall_in_sync(query, _native_arg(scope))
        )

    async def answer_in(self, question: str, scope: Any) -> Answer:
        return Answer.from_native(
            await self._inner.answer_in(question, _native_arg(scope))
        )

    def answer_in_sync(self, question: str, scope: Any) -> Answer:
        return Answer.from_native(
            self._inner.answer_in_sync(question, _native_arg(scope))
        )

    async def abduce(self, program: str, params: Any = None) -> AbductionResult:
        return AbductionResult.from_native(await self._inner.abduce(program, params))

    def abduce_sync(self, program: str, params: Any = None) -> AbductionResult:
        return AbductionResult.from_native(self._inner.abduce_sync(program, params))

    async def delete_session(self, session_id: str) -> DeletionReport:
        return DeletionReport.from_native(await self._inner.delete_session(session_id))

    def delete_session_sync(self, session_id: str) -> DeletionReport:
        return DeletionReport.from_native(self._inner.delete_session_sync(session_id))

    async def consolidate(self) -> CycleStats:
        return CycleStats.from_native(await self._inner.consolidate())

    def consolidate_sync(self) -> CycleStats:
        return CycleStats.from_native(self._inner.consolidate_sync())

    async def finalize_session(self, session_id: str) -> FinalizeReport:
        return FinalizeReport.from_native(
            await self._inner.finalize_session(session_id)
        )

    def finalize_session_sync(self, session_id: str) -> FinalizeReport:
        return FinalizeReport.from_native(self._inner.finalize_session_sync(session_id))

    async def unfinalized_session_ids(self) -> list[str]:
        return await self._inner.unfinalized_session_ids()

    def unfinalized_session_ids_sync(self) -> list[str]:
        return self._inner.unfinalized_session_ids_sync()

    async def forget_participant(self, participant_id: str) -> DeletionReport:
        return DeletionReport.from_native(
            await self._inner.forget_participant(participant_id)
        )

    def forget_participant_sync(self, participant_id: str) -> DeletionReport:
        return DeletionReport.from_native(
            self._inner.forget_participant_sync(participant_id)
        )

    # Untyped passthroughs (dynamic Cypher rows / Locy results / native builders).
    async def query(self, cypher: str) -> list[dict]:
        return await self._inner.query(cypher)

    def query_sync(self, cypher: str) -> list[dict]:
        return self._inner.query_sync(cypher)

    async def query_in(self, cypher: str, scope: Any) -> list[dict]:
        return await self._inner.query_in(cypher, _native_arg(scope))

    def query_in_sync(self, cypher: str, scope: Any) -> list[dict]:
        return self._inner.query_in_sync(cypher, _native_arg(scope))

    async def define_rule(self, name: str, source: str) -> int:
        return await self._inner.define_rule(name, source)

    def define_rule_sync(self, name: str, source: str) -> int:
        return self._inner.define_rule_sync(name, source)

    async def run_rule(
        self, name: str, return_cols: list[str], params: Any = None
    ) -> list[dict]:
        return await self._inner.run_rule(name, return_cols, params)

    def run_rule_sync(
        self, name: str, return_cols: list[str], params: Any = None
    ) -> list[dict]:
        return self._inner.run_rule_sync(name, return_cols, params)

    def assume(self, assume_block: str) -> Any:
        # Returns the native AssumeBuilder (.then_query(...).run() yields dicts).
        return self._inner.assume(assume_block)


# ── Uniko + builder ────────────────────────────────────────────────────────


class TypedUnikoBuilder:
    """Typed wrapper over the native ``UnikoBuilder``."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def path(self, path: str) -> "TypedUnikoBuilder":
        self._inner = self._inner.path(path)
        return self

    def in_memory(self) -> "TypedUnikoBuilder":
        self._inner = self._inner.in_memory()
        return self

    def llm(self, spec: Any) -> "TypedUnikoBuilder":
        self._inner = self._inner.llm(_native_arg(spec))
        return self

    def streaming(self, on: bool) -> "TypedUnikoBuilder":
        self._inner = self._inner.streaming(on)
        return self

    def scope_to_agent(self) -> "TypedUnikoBuilder":
        self._inner = self._inner.scope_to_agent()
        return self

    async def build(self) -> "TypedUniko":
        return TypedUniko(await self._inner.build())

    def build_sync(self) -> "TypedUniko":
        return TypedUniko(self._inner.build_sync())


class TypedUniko:
    """Typed wrapper over the native ``Uniko`` entry point."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    # Entry points — start typed from the top.
    @classmethod
    async def open(cls, path: str) -> "TypedUniko":
        from uniko import _uniko

        return cls(await _uniko.Uniko.open(path))

    @classmethod
    def open_sync(cls, path: str) -> "TypedUniko":
        from uniko import _uniko

        return cls(_uniko.Uniko.open_sync(path))

    @classmethod
    async def in_memory(cls) -> "TypedUniko":
        from uniko import _uniko

        return cls(await _uniko.Uniko.in_memory())

    @classmethod
    def in_memory_sync(cls) -> "TypedUniko":
        from uniko import _uniko

        return cls(_uniko.Uniko.in_memory_sync())

    @classmethod
    def builder(cls) -> TypedUnikoBuilder:
        from uniko import _uniko

        return TypedUnikoBuilder(_uniko.Uniko.builder())

    def agent(self, agent_id: str) -> TypedAgent:
        return TypedAgent(self._inner.agent(agent_id))

    def config(self) -> UnikoConfigModel:
        return UnikoConfigModel.model_validate(self._inner.config())

    async def purge(self) -> DeletionReport:
        return DeletionReport.from_native(await self._inner.purge())

    def purge_sync(self) -> DeletionReport:
        return DeletionReport.from_native(self._inner.purge_sync())

    async def shutdown(self) -> None:
        await self._inner.shutdown()

    def shutdown_sync(self) -> None:
        self._inner.shutdown_sync()


__all__ = [
    "TypedUniko",
    "TypedUnikoBuilder",
    "TypedAgent",
    "TypedSession",
    "TypedData",
    "TypedGoals",
]
