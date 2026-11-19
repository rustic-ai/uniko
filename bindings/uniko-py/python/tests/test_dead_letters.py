# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Dragonscale Industries Inc.
"""Dead letters (#65): inspect, retry and resolve streaming-ingest failures."""

from __future__ import annotations

import pytest

import uniko
from uniko.models import DeadLetter, DeadLetterCounts, RetryReport
from uniko.models.typed import TypedUniko

PNG = bytes([0x42] * 16)


class Flaky:
    """An image extractor that fails while ``failing`` is set."""

    def __init__(self) -> None:
        self.failing = True

    def __call__(self, request: object) -> str:
        if self.failing:
            raise RuntimeError("captioner offline")
        return "a red bicycle leaning on a wall"


def _streaming(extractor: Flaky) -> uniko.Uniko:
    return (
        uniko.Uniko.builder()
        .in_memory()
        .streaming(True)
        .extractor("image", extractor)
        .build_sync()
    )


def _fail_one(uni: uniko.Uniko, source_id: str) -> tuple[uniko.Session, str]:
    session = uni.agent("viewer").session("dl")
    session.submit_source_sync(
        uniko.IngestSource.from_bytes(PNG).with_mime("image/png").with_id(source_id)
    )
    session.flush_sync()
    letters = uni.dead_letters.list_sync()
    assert len(letters) == 1, letters
    return session, letters[0].id


def test_a_failure_is_listed_retried_and_resolved() -> None:
    extractor = Flaky()
    uni = _streaming(extractor)
    session, letter_id = _fail_one(uni, "img-1")

    letter = uni.dead_letters.get_sync(letter_id)
    assert letter is not None
    assert letter.status == "pending"
    assert letter.retriable and letter.replayable
    assert letter.ingest_type == "source"
    assert (letter.retry_count, letter.max_retries) == (0, 3)
    assert letter.next_retry_at is not None
    assert uni.dead_letters.counts_sync().pending == 1

    extractor.failing = False
    uni.dead_letters.retry_sync(letter_id)
    session.flush_sync()

    letter = uni.dead_letters.get_sync(letter_id)
    assert letter is not None
    assert (letter.status, letter.resolution, letter.retry_count) == (
        "resolved",
        "retried",
        1,
    )
    health = uni.health()
    assert health is not None
    ingest = health.ingest
    assert (ingest.dead_lettered, ingest.retries, ingest.retries_succeeded) == (1, 1, 1)
    with pytest.raises(uniko.ConfigError):
        uni.dead_letters.retry_sync(letter_id)


async def test_async_resolve_and_retry_pending() -> None:
    uni = _streaming(Flaky())
    session = uni.agent("viewer").session("dl")
    await session.submit_source(
        uniko.IngestSource.from_bytes(PNG).with_mime("image/png").with_id("img-2")
    )
    await session.flush()
    dl = uni.dead_letters
    [letter] = await dl.list(status="pending")
    report = await dl.retry_pending()
    assert report.retried == [letter.id] and report.refused == []
    await session.flush()
    letter = await dl.get(letter.id)
    assert letter is not None and letter.status == "pending"
    assert letter.retry_count == 1
    assert await dl.resolve(letter.id, "not worth keeping")
    assert not await dl.resolve(letter.id)
    assert (await dl.counts()).resolved == 1


def test_retry_needs_streaming_and_bad_status_is_rejected() -> None:
    uni = uniko.Uniko.in_memory_sync()
    assert uni.dead_letters.list_sync() == []
    with pytest.raises(uniko.ConfigError):
        uni.dead_letters.retry_sync("missing")
    with pytest.raises(ValueError):
        uni.dead_letters.list_sync(status="stuck")  # ty: ignore[invalid-argument-type]


def test_typed_dead_letters() -> None:
    extractor = Flaky()
    typed = TypedUniko(_streaming(extractor))
    session = typed.agent("viewer").session("dl")
    session.submit_source_sync(
        uniko.IngestSource.from_bytes(PNG).with_mime("image/png").with_id("img-3")
    )
    session.flush_sync()
    [letter] = typed.dead_letters.list_sync()
    assert isinstance(letter, DeadLetter)
    assert letter.status == "pending"
    counts = typed.dead_letters.counts_sync()
    assert isinstance(counts, DeadLetterCounts) and counts.pending == 1
    report = typed.dead_letters.retry_pending_sync()
    assert isinstance(report, RetryReport) and report.retried == [letter.id]
    health = typed.health()
    assert health is not None and health.ingest.dead_lettered == 1


def test_retries_stop_at_the_limit_and_force_goes_past_it() -> None:
    uni = (
        uniko.Uniko.builder()
        .in_memory()
        .streaming(True)
        .config({"dead_letter_max_retries": 1})
        .extractor("image", Flaky())
        .build_sync()
    )
    session, letter_id = _fail_one(uni, "img-4")
    dl = uni.dead_letters
    dl.retry_sync(letter_id)
    session.flush_sync()
    letter = dl.get_sync(letter_id)
    assert letter is not None
    assert (letter.status, letter.retry_count) == ("exhausted", 1)
    with pytest.raises(uniko.ConfigError):
        dl.retry_sync(letter_id)
    dl.retry_sync(letter_id, force=True)
    session.flush_sync()
    letter = dl.get_sync(letter_id)
    assert letter is not None
    assert (letter.status, letter.retry_count) == ("exhausted", 2)
    assert dl.resolve_sync(letter_id, "gave up")
    letter = dl.get_sync(letter_id)
    assert letter is not None
    assert (letter.status, letter.resolution, letter.replayable) == (
        "resolved",
        "gave up",
        False,
    )
