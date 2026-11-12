# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Dragonscale Industries Inc.
"""Pydantic input specs — validated payloads that build native builders.

Each spec validates its inputs *before* anything crosses the FFI boundary
(``extra="forbid"`` turns a mistyped keyword into a named error), then produces
the corresponding native builder via ``to_native()``.

The native builder imports are **lazy** — done inside ``to_native()`` — so this
module imports with no compiled extension present (e.g. a FastAPI service that
only needs the request/response schemas, not the engine).
"""

from __future__ import annotations

import datetime
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

# Reject unknown keys on every input spec — catches typo'd kwargs at the edge.
_IN = ConfigDict(extra="forbid")


def _native():
    """Lazily import the compiled extension (keeps module import engine-free)."""
    from uniko import _uniko

    return _uniko


# ── Goal / Task creation (clean kwarg payloads) ────────────────────────────


class GoalSpec(BaseModel):
    """Validated payload for ``Goals.create`` / ``Goals.create_sync``.

    ``status`` is deliberately free-form — the facade stores it raw and derives
    the goal *phase* from it, so any string is legal.
    """

    model_config = _IN

    title: str = Field(min_length=1, description="Human-readable goal title.")
    goal_id: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = Field(
        default=None,
        description="Free-form status string; the facade derives phase from it.",
    )
    metrics: Optional[Any] = Field(
        default=None, description="Free-form success metrics merged into the goal."
    )
    guardrails: Optional[Any] = None
    deadline: Optional[datetime.datetime] = None
    parent_goal_id: Optional[str] = None

    def to_create_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``Goals.create`` (``title`` is positional)."""
        return self.model_dump(exclude_none=True, exclude={"title"})


class TaskSpec(BaseModel):
    """Validated payload for ``Goals.create_task`` / ``Goals.create_task_sync``."""

    model_config = _IN

    title: str = Field(min_length=1, description="Human-readable task title.")
    task_id: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[float] = None
    goal_id: Optional[str] = None
    depends_on_task_id: Optional[str] = None
    subtask_of_task_id: Optional[str] = None

    def to_create_task_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``Goals.create_task`` (``title`` is positional)."""
        return self.model_dump(exclude_none=True, exclude={"title"})


# ── Scope (builder chain) ──────────────────────────────────────────────────


class ScopeSpec(BaseModel):
    """Validated payload for a recall/query ``Scope``."""

    model_config = _IN

    sessions: Optional[list[str]] = None
    participants: Optional[list[str]] = None
    since: Optional[datetime.datetime] = None
    until: Optional[datetime.datetime] = None

    def to_native(self) -> Any:
        scope = _native().Scope()
        if self.sessions is not None:
            scope = scope.sessions(self.sessions)
        if self.participants is not None:
            scope = scope.participants(self.participants)
        if self.since is not None:
            scope = scope.since(self.since)
        if self.until is not None:
            scope = scope.until(self.until)
        return scope


# ── IngestSource (mutually-exclusive one-of, discriminated union) ───────────


class _TextSource(BaseModel):
    model_config = _IN
    source: Literal["text"] = "text"
    content: str


class _BytesSource(BaseModel):
    model_config = _IN
    source: Literal["bytes"] = "bytes"
    data: bytes


class _PathSource(BaseModel):
    model_config = _IN
    source: Literal["path"] = "path"
    path: str


class IngestSourceSpec(BaseModel):
    """Validated payload for an ``IngestSource``.

    The content origin is a discriminated union on ``source`` — exactly one of
    text / bytes / path — which both enforces the one-of at validation time and
    produces a clean ``oneOf`` JSON schema for OpenAPI.
    """

    model_config = _IN

    origin: Union[_TextSource, _BytesSource, _PathSource] = Field(
        discriminator="source"
    )
    mime: Optional[str] = None
    id: Optional[str] = None
    with_path: Optional[str] = None

    # Ergonomic constructors (named to match the native ``IngestSource.from_*``)
    # so callers don't hand-build the nested ``origin``.
    @classmethod
    def from_text(cls, content: str, **kwargs: Any) -> "IngestSourceSpec":
        return cls(origin=_TextSource(content=content), **kwargs)

    @classmethod
    def from_bytes(cls, data: bytes, **kwargs: Any) -> "IngestSourceSpec":
        return cls(origin=_BytesSource(data=data), **kwargs)

    @classmethod
    def from_path(cls, path: str, **kwargs: Any) -> "IngestSourceSpec":
        return cls(origin=_PathSource(path=path), **kwargs)

    def to_native(self) -> Any:
        ns = _native()
        o = self.origin
        if o.source == "text":
            src = ns.IngestSource.from_text(o.content)
        elif o.source == "bytes":
            src = ns.IngestSource.from_bytes(o.data)
        else:
            src = ns.IngestSource.from_path(o.path)
        if self.mime is not None:
            src = src.with_mime(self.mime)
        if self.id is not None:
            src = src.with_id(self.id)
        if self.with_path is not None:
            src = src.with_path(self.with_path)
        return src


# ── Turn (builder chain + nested attachments) ──────────────────────────────


class TurnSpec(BaseModel):
    """Validated payload for a conversational ``Turn``."""

    model_config = _IN

    sender_id: str = Field(min_length=1)
    content: str
    message_id: Optional[str] = None
    content_type: Optional[str] = None
    at: Optional[datetime.datetime] = None
    addressed_to: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None
    attachments: Optional[list[IngestSourceSpec]] = None

    def to_native(self) -> Any:
        turn = _native().Turn(self.sender_id, self.content)
        if self.message_id is not None:
            turn = turn.id(self.message_id)
        if self.content_type is not None:
            turn = turn.content_type(self.content_type)
        if self.at is not None:
            turn = turn.at(self.at)
        if self.addressed_to is not None:
            turn = turn.addressed_to(self.addressed_to)
        if self.metadata:
            for key, value in self.metadata.items():  # native takes (key, value)
                turn = turn.metadata(key, value)
        if self.attachments:
            turn = turn.attachments([a.to_native() for a in self.attachments])
        return turn


# ── LlmSpec (mutually-exclusive one-of, discriminated union) ───────────────


class _OpenAi(BaseModel):
    model_config = _IN
    provider: Literal["openai"] = "openai"
    alias: str
    model_id: str
    base_url: Optional[str] = None

    def to_native(self) -> Any:
        return _native().LlmSpec.openai(self.alias, self.model_id, self.base_url)


class _OpenAiKeyEnv(BaseModel):
    model_config = _IN
    provider: Literal["openai_with_key_env"] = "openai_with_key_env"
    alias: str
    model_id: str
    key_env: str
    base_url: Optional[str] = None

    def to_native(self) -> Any:
        return _native().LlmSpec.openai_with_key_env(
            self.alias, self.model_id, self.key_env, self.base_url
        )


class _LlamaCpp(BaseModel):
    model_config = _IN
    provider: Literal["llamacpp"] = "llamacpp"
    alias: str
    model_id: str
    base_url: str
    key_env: Optional[str] = None

    def to_native(self) -> Any:
        return _native().LlmSpec.llamacpp(
            self.alias, self.model_id, self.base_url, self.key_env
        )


# Discriminated-union alias; validate raw dicts with ``LlmSpecAdapter``.
LlmSpecModel = Annotated[
    Union[_OpenAi, _OpenAiKeyEnv, _LlamaCpp],
    Field(discriminator="provider"),
]

LlmSpecAdapter: TypeAdapter = TypeAdapter(LlmSpecModel)


def llm_to_native(spec: Any) -> Any:
    """Build a native ``LlmSpec`` from any ``LlmSpecModel`` variant."""
    return spec.to_native()


__all__ = [
    "GoalSpec",
    "TaskSpec",
    "ScopeSpec",
    "IngestSourceSpec",
    "TurnSpec",
    "LlmSpecModel",
    "LlmSpecAdapter",
    "llm_to_native",
]
