# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Dragonscale Industries Inc.
"""Store compatibility and in-place migrations (#66). Model-free: the
stores hold no rows to embed, and opens use lazy prefetch."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

import uniko
from uniko.models import MigrationReport, StoreInfo, StoreReport
from uniko.models.typed import TypedUniko
from uniko.types import MigrationProgressEvent


def _create(path: Path, config: dict[str, Any] | None = None) -> None:
    builder = uniko.Uniko.builder().path(str(path)).prefetch("lazy")
    if config is not None:
        builder = builder.config(config)
    builder.build_sync().shutdown_sync()


def test_inspect_reports_a_store_and_a_missing_path(tmp_path: Path) -> None:
    missing = uniko.Uniko.inspect_store_sync(str(tmp_path / "none"))
    assert not missing.info.exists and missing.compatible
    assert not (tmp_path / "none").exists()

    store = tmp_path / "kb"
    _create(store)
    report = uniko.Uniko.inspect_store_sync(str(store))
    assert report.compatible and report.differences == []
    emb = report.info.embedding
    assert (emb.dimensions, emb.metric, emb.is_hybrid) == (384, "cosine", False)

    other = uniko.Uniko.inspect_store_sync(str(store), embedding="bge-m3")
    assert not other.compatible
    fields = {d.field: d.verdict for d in other.differences}
    assert fields["dimensions"] == "incompatible"


def test_opening_with_another_model_raises_before_changing_anything(
    tmp_path: Path,
) -> None:
    store = tmp_path / "kb"
    _create(store)
    with pytest.raises(uniko.IncompatibleStoreError) as caught:
        uniko.Uniko.builder().path(str(store)).embedding("bge-m3").build_sync()
    err = caught.value
    assert getattr(err, "kind") == "incompatible_store"
    differences = getattr(err, "differences")
    assert any(d["field"] == "dimensions" for d in differences)
    assert uniko.Uniko.inspect_store_sync(str(store)).compatible, "store left as it was"


def test_an_index_change_is_migrated_with_progress(tmp_path: Path) -> None:
    store = tmp_path / "kb"
    _create(store)
    dot = {"vector_metric": "Dot"}
    report = uniko.Uniko.inspect_store_sync(str(store), dot)
    assert report.migrations == ["rebuild_vector_indexes"]
    assert report.differences[0].verdict == "needs_migration"

    events: list[MigrationProgressEvent] = []
    done = uniko.Uniko.migrate_store_sync(
        str(store), "rebuild_vector_indexes", dot, progress=events.append
    )
    assert done.completed and done.migration == "rebuild_vector_indexes"
    assert done.migration_in_progress is None
    assert [e["phase"] for e in events][0] == "schema"
    last = events[-1]
    assert last["phase"] == "done"
    rows = done.rows_backfilled + done.rows_unfilled
    assert last["done"] == last["total"] == rows

    uni = (
        uniko.Uniko.builder().path(str(store)).prefetch("lazy").config(dot).build_sync()
    )
    info = uni.store_info_sync()
    assert info.embedding.metric == "dot"
    uni.shutdown_sync()

    with pytest.raises(ValueError):
        uniko.Uniko.migrate_store_sync(str(store), "reindex")  # ty: ignore[invalid-argument-type]


async def test_typed_inspect_and_migrate(tmp_path: Path) -> None:
    store = tmp_path / "kb"
    _create(store)
    report = await TypedUniko.inspect_store(str(store), {"vector_metric": "L2"})
    assert isinstance(report, StoreReport) and report.migrations == [
        "rebuild_vector_indexes"
    ]
    done = await TypedUniko.migrate_store(
        str(store), "rebuild_vector_indexes", {"vector_metric": "L2"}
    )
    assert isinstance(done, MigrationReport) and done.completed
    assert done.migration_in_progress is None
    uni = (
        await TypedUniko.builder()
        .path(str(store))
        .prefetch("lazy")
        .config({"vector_metric": "L2"})
        .build()
    )
    info = await uni.store_info()
    assert isinstance(info, StoreInfo) and info.embedding.metric == "l2"
    await uni.shutdown()


def _bge_m3_dense() -> dict[str, Any]:
    """bge-m3 with only its dense head: the same model as a hybrid store, so
    ``enable_hybrid`` is what tells the two apart. Taken from the preset (the
    Python API exposes presets only through a built instance's config) with
    the sparse and ColBERT heads dropped."""
    uni = (
        uniko.Uniko.builder()
        .in_memory()
        .prefetch("lazy")
        .embedding("bge-m3")
        .build_sync()
    )
    try:
        embedding = dict(uni.config()["embedding"])
    finally:
        uni.shutdown_sync()
    assert embedding["sparse_dimensions"] and embedding["multivector_dimensions"]
    embedding.update(sparse_dimensions=None, multivector_dimensions=None)
    return embedding


def test_incompatible_store_error_has_the_report_shape(tmp_path: Path) -> None:
    store = tmp_path / "kb"
    _create(store)
    with pytest.raises(uniko.IncompatibleStoreError) as caught:
        uniko.Uniko.builder().path(str(store)).embedding("bge-m3").build_sync()
    err = caught.value
    assert err.compatible is False
    assert err.migrations == []
    assert err.differences, "at least one difference"
    for d in err.differences:
        assert set(d) == {"field", "stored", "requested", "verdict", "migration"}
        assert d["verdict"] in {"incompatible", "needs_migration"}
        assert isinstance(d["field"], str) and isinstance(d["stored"], str)
    dims = next(d for d in err.differences if d["field"] == "dimensions")
    assert (dims["verdict"], dims["migration"]) == ("incompatible", None)

    # The same differences, flat, as inspect_store reports them.
    report = uniko.Uniko.inspect_store_sync(str(store), embedding="bge-m3")
    assert err.differences == [
        {
            "field": d.field,
            "stored": d.stored,
            "requested": d.requested,
            "verdict": d.verdict,
            "migration": d.migration,
        }
        for d in report.differences
    ]


def test_an_incompatible_difference_offers_no_migrations(tmp_path: Path) -> None:
    store = tmp_path / "kb"
    _create(store)
    report = uniko.Uniko.inspect_store_sync(str(store), embedding="bge-m3")
    assert not report.compatible
    assert report.migrations == []


def test_inspect_leaves_a_non_store_directory_alone(tmp_path: Path) -> None:
    other = tmp_path / "notes"
    other.mkdir()
    (other / "readme.txt").write_text("not a store")
    before = sorted(p.relative_to(other) for p in other.rglob("*"))

    report = uniko.Uniko.inspect_store_sync(str(other))
    assert report.info.exists is False
    assert sorted(p.relative_to(other) for p in other.rglob("*")) == before
    assert (other / "readme.txt").read_text() == "not a store"


def test_a_store_open_in_this_process_is_refused(tmp_path: Path) -> None:
    store = tmp_path / "kb"
    uni = uniko.Uniko.builder().path(str(store)).prefetch("lazy").build_sync()
    try:
        with pytest.raises(uniko.ConfigError, match="open in this process"):
            uniko.Uniko.inspect_store_sync(str(store))
        with pytest.raises(uniko.ConfigError, match="open in this process"):
            uniko.Uniko.migrate_store_sync(
                str(store), "rebuild_vector_indexes", {"vector_metric": "Dot"}
            )
    finally:
        uni.shutdown_sync()
    assert uniko.Uniko.inspect_store_sync(str(store)).info.exists


def test_enable_hybrid_on_an_empty_store(tmp_path: Path) -> None:
    store = tmp_path / "kb"
    _create(store, {"embedding": _bge_m3_dense()})
    before = uniko.Uniko.inspect_store_sync(str(store), embedding="bge-m3")
    assert before.migrations == ["enable_hybrid"]
    assert before.info.embedding.is_hybrid is False

    done = uniko.Uniko.migrate_store_sync(
        str(store), "enable_hybrid", embedding="bge-m3"
    )
    assert done.completed and done.migration == "enable_hybrid"
    assert (done.rows_backfilled, done.rows_unfilled) == (0, 0)

    after = uniko.Uniko.inspect_store_sync(str(store), embedding="bge-m3")
    assert after.compatible and after.info.embedding.is_hybrid


def test_a_raising_progress_callback_does_not_stop_a_migration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = tmp_path / "kb"
    _create(store)
    calls: list[str] = []

    def boom(event: MigrationProgressEvent) -> None:
        calls.append(event["phase"])
        raise RuntimeError("progress callback failed")

    done = uniko.Uniko.migrate_store_sync(
        str(store), "rebuild_vector_indexes", {"vector_metric": "Dot"}, progress=boom
    )
    assert isinstance(done, uniko.MigrationReport)
    assert done.completed and done.rows_unfilled == 0
    assert calls and calls[-1] == "done", "called for every event despite raising"
    assert "progress callback failed" in capsys.readouterr().err, "and reported"
    assert uniko.Uniko.inspect_store_sync(
        str(store), {"vector_metric": "Dot"}
    ).compatible


@pytest.mark.parametrize("raised", [KeyboardInterrupt, SystemExit])
def test_a_progress_callback_cant_interrupt_or_exit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], raised: type[BaseException]
) -> None:
    store = tmp_path / "kb"
    _create(store)
    calls: list[str] = []

    def stop(event: MigrationProgressEvent) -> None:
        calls.append(event["phase"])
        raise raised("stop from progress")

    done = uniko.Uniko.migrate_store_sync(
        str(store), "rebuild_vector_indexes", {"vector_metric": "Dot"}, progress=stop
    )
    assert done.completed and calls[-1] == "done"
    err = capsys.readouterr().err
    assert raised.__name__ in err and "stop from progress" in err


def test_a_panicking_progress_callback_does_not_stop_a_migration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Calling a ``_sync`` method from the callback panics inside the
    extension (a runtime started from within the runtime); the panic is
    printed and ignored rather than aborting the migration."""
    store = tmp_path / "kb"
    _create(store)
    calls: list[str] = []

    def nested(event: MigrationProgressEvent) -> None:
        calls.append(event["phase"])
        uniko.Uniko.inspect_store_sync(str(store))

    done = uniko.Uniko.migrate_store_sync(
        str(store), "rebuild_vector_indexes", {"vector_metric": "Dot"}, progress=nested
    )
    assert isinstance(done, uniko.MigrationReport)
    assert done.completed and done.migration_in_progress is None
    assert calls[0] == "schema" and calls[-1] == "done", "called for every event"
    assert "PanicException" in capsys.readouterr().err, "and reported"
    # The store was shut down and its claim released.
    assert uniko.Uniko.inspect_store_sync(
        str(store), {"vector_metric": "Dot"}
    ).compatible


def test_migrate_refuses_a_schema_path(tmp_path: Path) -> None:
    store = tmp_path / "kb"
    _create(store)
    with pytest.raises(uniko.ConfigError, match="schema_path"):
        uniko.Uniko.migrate_store_sync(
            str(store),
            "rebuild_vector_indexes",
            {"vector_metric": "Dot", "schema_path": str(tmp_path / "x.json")},
        )


def test_an_unsupported_uri_is_refused() -> None:
    uri = "abfss://container@acct.dfs.core.windows.net/kb"
    with pytest.raises(uniko.UnsupportedError):
        uniko.Uniko.inspect_store_sync(uri)
    with pytest.raises(uniko.UnsupportedError):
        uniko.Uniko.migrate_store_sync(
            uri, "rebuild_vector_indexes", {"vector_metric": "Dot"}
        )


async def test_a_store_being_migrated_is_refused_elsewhere(tmp_path: Path) -> None:
    """While a migration runs, another thread can neither inspect nor open
    the store. The progress callback holds the migration in its "schema"
    phase until the other thread has tried both, so the overlap is certain."""
    store = tmp_path / "kb"
    _create(store)
    in_schema = threading.Event()
    tried = threading.Event()
    outcomes: list[str] = []

    def hold(event: MigrationProgressEvent) -> None:
        if event["phase"] == "schema" and not in_schema.is_set():
            in_schema.set()
            tried.wait(timeout=60)

    def other_thread() -> None:
        try:
            assert in_schema.wait(timeout=60), "migration never reached schema"
            try:
                uniko.Uniko.inspect_store_sync(str(store))
                outcomes.append("inspected")
            except uniko.ConfigError as e:
                outcomes.append(str(e))
            try:
                builder = uniko.Uniko.builder().path(str(store)).prefetch("lazy")
                builder.build_sync().shutdown_sync()
                outcomes.append("opened")
            except uniko.ConfigError as e:
                outcomes.append(str(e))
        finally:
            tried.set()

    thread = threading.Thread(target=other_thread)
    thread.start()
    try:
        done = await uniko.Uniko.migrate_store(
            str(store),
            "rebuild_vector_indexes",
            {"vector_metric": "Dot"},
            progress=hold,
        )
    finally:
        tried.set()
        thread.join(timeout=60)
    assert done.completed
    assert len(outcomes) == 2, outcomes
    assert all("being inspected or migrated" in o for o in outcomes), outcomes
    # Released once the migration is done.
    assert uniko.Uniko.inspect_store_sync(
        str(store), {"vector_metric": "Dot"}
    ).compatible
