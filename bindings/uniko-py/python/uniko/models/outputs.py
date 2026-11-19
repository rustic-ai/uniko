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

from pydantic import BaseModel, ConfigDict, Field, computed_field

from ..types import (
    CircuitState,
    DeadLetterStatus,
    GoalPhase,
    MigrationName,
    IngestKind,
    RecallKind,
    RecallSourceKind,
    RegistrationOutcome,
    TaskPhase,
    VerdictName,
    WorkerStatus,
)

# Frozen + from_attributes on every output model. Frozen because these are
# read-only snapshots; mutating them would not write back to the graph.
_OUT = ConfigDict(from_attributes=True, frozen=True)


class RecallSource(BaseModel):
    """Provenance pointer for a recalled item (native ``RecallSource``)."""

    model_config = _OUT

    kind: RecallSourceKind
    message_id: Optional[str] = None
    session_id: Optional[str] = None
    call_id: Optional[str] = None
    tool_name: Optional[str] = None
    artifact_id: Optional[str] = None
    chunk_id: Optional[str] = None

    @classmethod
    def from_native(cls, native: Any) -> "RecallSource":
        return cls.model_validate(native)


class RecallItem(BaseModel):
    """A recalled node with score and lineage (native ``RecallItem``)."""

    model_config = _OUT

    node_id: int
    kind: RecallKind
    score: float
    content: str
    sources: list[RecallSource] = Field(default_factory=list)
    category: Optional[str] = None
    source_id: Optional[str] = None
    revision_id: Optional[str] = None
    token_count: int = 0
    context: Optional[str] = None
    address: Optional[str] = None

    @classmethod
    def from_native(cls, native: Any) -> "RecallItem":
        return cls.model_validate(native)


class Diagnostic(BaseModel):
    """One stage outcome of a recall or ingest (native ``Diagnostic``)."""

    model_config = _OUT

    stage: str = Field(description="e.g. 'query_embedding', 'rerank', 'sparse'.")
    status: str = Field(description="'applied', 'skipped', 'fallback' or 'failed'.")
    code: Optional[str] = Field(
        default=None, description="Why, when the status is not 'applied'."
    )
    target: Optional[str] = Field(
        default=None, description="The node label, model alias or source."
    )
    count: Optional[int] = None

    @property
    def degraded(self) -> bool:
        """Whether the stage fell back or failed."""
        return self.status in ("fallback", "failed")


class ContextBundle(BaseModel):
    """A recall result: scored items plus coverage stats (native ``ContextBundle``)."""

    model_config = _OUT

    items: list[RecallItem] = Field(default_factory=list)
    total_tokens: int
    phase1_only: bool
    phase2_only: bool
    coverage: float
    diagnostics: list[Diagnostic] = Field(default_factory=list)

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
    attachments: list["IngestOutcome"] = Field(default_factory=list)
    tool_calls: list["ToolCallResult"] = Field(default_factory=list)
    diagnostics: list[Diagnostic] = Field(default_factory=list)

    @classmethod
    def from_native(cls, native: Any) -> "ObserveResult":
        return cls.model_validate(native)


class SearchHit(BaseModel):
    """A lexical search hit (native ``SearchHit``)."""

    model_config = _OUT

    source_kind: str
    source_id: str
    chunk_id: str
    chunk_index: int
    address: Optional[str] = None
    start: int
    end: int
    text: str
    context: Optional[str] = None
    score: float
    session_id: Optional[str] = None
    call_id: Optional[str] = None
    tool_name: Optional[str] = None

    @classmethod
    def from_native(cls, native: Any) -> "SearchHit":
        return cls.model_validate(native)


class OutlineEntry(BaseModel):
    """One entry of an outline (native ``OutlineEntry``)."""

    model_config = _OUT

    address: str
    kind: str
    summary: str
    start: int
    end: int
    depth: int = 0


class Outline(BaseModel):
    """The shape of a source (native ``Outline``)."""

    model_config = _OUT

    source_kind: str
    source_id: str
    format: str
    total_bytes: int
    total_lines: int
    entries: list[OutlineEntry] = Field(default_factory=list)
    truncated: bool = False

    @classmethod
    def from_native(cls, native: Any) -> "Outline":
        return cls.model_validate(native)


class ReadResult(BaseModel):
    """Verbatim text read from a source (native ``ReadResult``)."""

    model_config = _OUT

    source_kind: str
    source_id: str
    text: str
    start: int
    end: int
    first_line: int
    last_line: int
    truncated: bool = False
    next_bytes: Optional[tuple[int, int]] = None

    @classmethod
    def from_native(cls, native: Any) -> "ReadResult":
        return cls.model_validate(native)


class ToolCallView(BaseModel):
    """A recorded tool call (native ``ToolCallView``)."""

    model_config = _OUT

    call_id: str
    tool_name: Optional[str] = None
    action_type: str
    arguments: Any = None
    status: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[datetime.datetime] = None
    ended_at: Optional[datetime.datetime] = None
    triggered_by: Optional[str] = None
    session_id: Optional[str] = None
    artifact_id: Optional[str] = None

    @classmethod
    def from_native(cls, native: Any) -> "ToolCallView":
        return cls.model_validate(native)


class ToolCallResult(BaseModel):
    """What recording one tool call produced (native ``ToolCallResult``)."""

    model_config = _OUT

    call_id: str
    action_node: int
    artifact_id: Optional[str] = None
    artifact_node: Optional[int] = None
    was_replay: bool = False
    diagnostics: list[Diagnostic] = Field(default_factory=list)

    @classmethod
    def from_native(cls, native: Any) -> "ToolCallResult":
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


class ModelStatus(BaseModel):
    """One model alias and its load state (an entry of :class:`Capabilities`).

    Provider options are never included. ``api_key_env`` is the variable
    name only, and ``base_url`` has userinfo and query removed.
    """

    model_config = _OUT

    alias: str
    task: str
    provider_id: str
    model_id: str
    revision: Optional[str] = None
    state: str = Field(description="'not_loaded', 'loaded' or 'failed'.")
    error: Optional[str] = Field(
        default=None, description="Set when state is 'failed'."
    )
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None
    requested_execution_providers: Optional[list[str]] = None
    active_execution_providers: Optional[list[str]] = None


class Capabilities(BaseModel):
    """What a build and instance can do (``Uniko.capabilities()``)."""

    model_config = _OUT

    compiled_features: dict[str, bool]
    extractors: list[str]
    models: Optional[list[ModelStatus]] = None
    answer_alias: Optional[str] = None
    ocr_compiled: bool
    ocr_configured: bool

    @classmethod
    def from_native(cls, native: Any) -> "Capabilities":
        return cls.model_validate(native)


class ReadinessFailure(BaseModel):
    """One unmet requirement, as ``check_readiness()`` returns it.

    ``ReadinessError.failures`` carries the same fields as plain dicts;
    validate one with ``ReadinessFailure.model_validate(row)``.
    """

    model_config = _OUT

    requirement: str = Field(
        description="As written, e.g. 'embedder' or 'alias:gen/m'."
    )
    code: str = Field(
        description="'not_in_catalog', 'not_configured', 'not_compiled', "
        "'no_extractor', 'load_failed', 'timed_out', 'nlp_unavailable' "
        "or 'not_accelerated'."
    )
    alias: Optional[str] = None
    error: Optional[str] = Field(
        default=None, description="The redacted load error, for 'load_failed'."
    )

    @classmethod
    def from_native(cls, native: Any) -> "ReadinessFailure":
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
    #: Observations sent for LLM triple refinement (0 on the default path).
    triples_requested: int = 0
    #: Observations whose triple was replaced by a parsed LLM reply; the
    #: rest kept the rule-based triple.
    triples_refined: int = 0
    #: Observations whose LLM call failed outright (endpoint down, timeout).
    triples_failed: int = 0

    @classmethod
    def from_native(cls, native: Any) -> "CycleStats":
        return cls.model_validate(native)


class MessageView(BaseModel):
    """A stored message (native ``MessageView``)."""

    model_config = _OUT

    message_id: str
    sender_id: str
    content: str
    content_type: str = "text"
    category: Optional[str] = None
    visibility: Optional[str] = None
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
    phase: GoalPhase
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
    phase: TaskPhase
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

    kind: IngestKind
    artifact_id: str
    artifact_node_id: int
    chunk_node_ids: list[int] = Field(default_factory=list)
    was_deduplicated: bool
    page_count: Optional[int] = None
    extraction_failed: Optional[bool] = None
    extraction_error: Optional[str] = None
    page_node_ids: list[int] = Field(default_factory=list)
    block_node_ids: list[int] = Field(default_factory=list)
    diagnostics: list[Diagnostic] = Field(default_factory=list)

    @classmethod
    def from_native(cls, native: Any) -> "IngestOutcome":
        return cls.model_validate(native)


ObserveResult.model_rebuild()


class UnitResult(BaseModel):
    """What ``Session.commit_unit()`` committed (native ``UnitResult``)."""

    model_config = _OUT

    turns: list[ObserveResult] = Field(default_factory=list)
    was_replay: bool
    diagnostics: list[Diagnostic] = Field(default_factory=list)

    @classmethod
    def from_native(cls, native: Any) -> "UnitResult":
        return cls.model_validate(native)

    def message_node_ids(self) -> list[int]:
        return [t.message_node_id for t in self.turns]

    def __len__(self) -> int:
        return len(self.turns)


class SummarizeReport(BaseModel):
    """What ``Session.summarize()`` produced (native ``SummarizeReport``)."""

    model_config = _OUT

    summary: Optional[int] = None
    finalize: Optional[FinalizeReport] = None
    finalize_error: Optional[str] = None

    @classmethod
    def from_native(cls, native: Any) -> "SummarizeReport":
        return cls.model_validate(native)


class WorkerHealth(BaseModel):
    """One streaming-pipeline worker's health (native ``WorkerHealth``)."""

    model_config = _OUT

    status: WorkerStatus
    queue_depth: int
    queue_capacity: int
    items_processed: int
    items_failed: int
    avg_latency_ms: float
    degradations: dict[str, int] = Field(default_factory=dict)
    dead_lettered: int = 0
    retries: int = 0
    retries_succeeded: int = 0

    @classmethod
    def from_native(cls, native: Any) -> "WorkerHealth":
        return cls.model_validate(native)


class DeadLetterCounts(BaseModel):
    """Dead letters per status (native ``DeadLetterCounts``)."""

    model_config = _OUT

    pending: int
    retrying: int
    exhausted: int
    resolved: int
    legacy: int

    @classmethod
    def from_native(cls, native: Any) -> "DeadLetterCounts":
        return cls.model_validate(native)


class RetryRefusal(BaseModel):
    """A dead letter ``retry_pending`` could not resubmit."""

    model_config = _OUT

    id: str
    reason: str


class RetryReport(BaseModel):
    """What ``DeadLetters.retry_pending`` did (native ``RetryReport``)."""

    model_config = _OUT

    retried: list[str] = Field(default_factory=list)
    refused: list[RetryRefusal] = Field(default_factory=list)

    @classmethod
    def from_native(cls, native: Any) -> "RetryReport":
        return cls.model_validate(native)


class StoredEmbedding(BaseModel):
    """The embedding setup a store was built with (native ``StoredEmbedding``)."""

    model_config = _OUT

    model_id: Optional[str] = None
    dimensions: Optional[int] = None
    sparse_dimensions: Optional[int] = None
    multivector_dimensions: Optional[int] = None
    algorithm: Optional[str] = None
    metric: Optional[str] = None
    embed_alias: Optional[str] = None
    document_prefix: Optional[str] = None
    query_prefix: Optional[str] = None
    batch_size: Optional[int] = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_hybrid(self) -> bool:
        """The store has the sparse or ColBERT column."""
        return (
            self.sparse_dimensions is not None
            or self.multivector_dimensions is not None
        )


class StoreInfo(BaseModel):
    """What a store was built with (native ``StoreInfo``)."""

    model_config = _OUT

    exists: bool
    embedding: StoredEmbedding
    hybrid_backfill_pending: Optional[int] = None
    migration_in_progress: Optional[MigrationName] = None

    @classmethod
    def from_native(cls, native: Any) -> "StoreInfo":
        return cls.model_validate(native)


class StoreDifference(BaseModel):
    """One way a store differs from a config (native ``StoreDifference``)."""

    model_config = _OUT

    field: str
    stored: str
    requested: str
    verdict: VerdictName
    migration: Optional[MigrationName] = None


class StoreReport(BaseModel):
    """What ``Uniko.inspect_store`` found (native ``StoreReport``)."""

    model_config = _OUT

    info: StoreInfo
    differences: list[StoreDifference] = Field(default_factory=list)
    compatible: bool
    migrations: list[MigrationName] = Field(default_factory=list)

    @classmethod
    def from_native(cls, native: Any) -> "StoreReport":
        return cls.model_validate(native)


class MigrationReport(BaseModel):
    """What ``Uniko.migrate_store`` did (native ``MigrationReport``)."""

    model_config = _OUT

    migration: MigrationName
    rows_backfilled: int
    rows_unfilled: int
    completed: bool
    elapsed_secs: float
    migration_in_progress: Optional[MigrationName] = None

    @classmethod
    def from_native(cls, native: Any) -> "MigrationReport":
        return cls.model_validate(native)


class Registration(BaseModel):
    """What ``Agent.register`` did (native ``Registration``)."""

    model_config = _OUT

    participant_id: str
    outcome: RegistrationOutcome

    @classmethod
    def from_native(cls, native: Any) -> "Registration":
        return cls.model_validate(native)


class DeadLetter(BaseModel):
    """One dead-lettered streaming item (native ``DeadLetter``)."""

    model_config = _OUT

    id: str
    step: str
    error: str
    error_kind: Optional[str] = None
    retriable: bool
    status: DeadLetterStatus
    retry_count: int
    max_retries: int
    ingest_type: Optional[str] = None
    agent_id: Optional[str] = None
    replayable: bool
    created_at: Optional[datetime.datetime] = None
    updated_at: Optional[datetime.datetime] = None
    next_retry_at: Optional[datetime.datetime] = None
    resolved_at: Optional[datetime.datetime] = None
    resolution: Optional[str] = None

    @classmethod
    def from_native(cls, native: Any) -> "DeadLetter":
        return cls.model_validate(native)


class PipelineHealth(BaseModel):
    """A streaming-pipeline health snapshot (native ``PipelineHealth``)."""

    model_config = _OUT

    ingest: WorkerHealth
    consolidation: WorkerHealth
    llm_circuit: CircuitState
    uptime_secs: int

    @classmethod
    def from_native(cls, native: Any) -> "PipelineHealth":
        return cls.model_validate(native)


class PromotionReport(BaseModel):
    """What procedure promotion did (native ``PromotionReport``)."""

    model_config = _OUT

    created: int
    reinforced: int
    promoted: int

    @classmethod
    def from_native(cls, native: Any) -> "PromotionReport":
        return cls.model_validate(native)


class TopicReport(BaseModel):
    """What topic detection did (native ``TopicReport``)."""

    model_config = _OUT

    created: int
    updated: int
    entities_assigned: int

    @classmethod
    def from_native(cls, native: Any) -> "TopicReport":
        return cls.model_validate(native)


class ConsolidationReport(BaseModel):
    """What ``Agent.consolidate()`` did (native ``ConsolidationReport``)."""

    model_config = _OUT

    cycle: CycleStats
    procedures: PromotionReport
    topics: TopicReport

    @classmethod
    def from_native(cls, native: Any) -> "ConsolidationReport":
        return cls.model_validate(native)


class FactUpsert(BaseModel):
    """What ``Agent.assert_fact()`` wrote (native ``FactUpsert``)."""

    model_config = _OUT

    node_id: int
    fact_id: str
    was_created: bool
    observation_count: int

    @classmethod
    def from_native(cls, native: Any) -> "FactUpsert":
        return cls.model_validate(native)


class ActionResult(BaseModel):
    """What ``record_action()`` wrote (native ``ActionResult``)."""

    model_config = _OUT

    action_node: int
    overflow_artifact: Optional[int] = None
    diagnostics: list[Diagnostic] = Field(default_factory=list)

    @classmethod
    def from_native(cls, native: Any) -> "ActionResult":
        return cls.model_validate(native)


class RuleView(BaseModel):
    """A rule's state and, from ``explain_rule``, its source (native ``RuleView``)."""

    model_config = _OUT

    name: str
    source_type: str
    status: str
    confidence: float
    source: Optional[str] = None
    natural_language: Optional[str] = None

    @classmethod
    def from_native(cls, native: Any) -> "RuleView":
        return cls.model_validate(native)


class MatchedProcedure(BaseModel):
    """A procedure ``match_procedures()`` found applicable (native ``MatchedProcedure``)."""

    model_config = _OUT

    procedure_id: str
    name: str
    effectiveness: float

    @classmethod
    def from_native(cls, native: Any) -> "MatchedProcedure":
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
    "Diagnostic",
    "Answer",
    "ObserveResult",
    "ToolCallResult",
    "SearchHit",
    "Outline",
    "OutlineEntry",
    "ReadResult",
    "ToolCallView",
    "DeletionReport",
    "FinalizeReport",
    "CycleStats",
    "UnitResult",
    "SummarizeReport",
    "WorkerHealth",
    "PipelineHealth",
    "DeadLetter",
    "Registration",
    "StoredEmbedding",
    "StoreInfo",
    "StoreDifference",
    "StoreReport",
    "MigrationReport",
    "DeadLetterCounts",
    "RetryReport",
    "RetryRefusal",
    "PromotionReport",
    "TopicReport",
    "ConsolidationReport",
    "FactUpsert",
    "ActionResult",
    "RuleView",
    "MatchedProcedure",
    "MessageView",
    "ArtifactView",
    "GoalView",
    "TaskView",
    "GoalContext",
    "IngestOutcome",
    "AbducedModification",
    "AbductionResult",
]
