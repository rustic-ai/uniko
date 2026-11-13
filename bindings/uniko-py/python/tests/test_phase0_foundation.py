"""Phase 0 foundation tests: import, exceptions, Value converter round-trips."""

from __future__ import annotations

import datetime as dt

import uniko


def test_module_imports() -> None:
    assert uniko.__file__ is not None


def test_exception_hierarchy() -> None:
    # Base plus the five branchable subclasses are all present and rooted
    # under the base UnikoError.
    assert issubclass(uniko.UnikoError, Exception)
    for name in (
        "ConfigError",
        "LlmError",
        "UnikoTimeoutError",
        "ConflictError",
        "UnsupportedError",
    ):
        exc = getattr(uniko, name)
        assert issubclass(exc, uniko.UnikoError), name


def test_value_roundtrip_scalars() -> None:
    rt = uniko._uniko._value_roundtrip
    assert rt(None) is None
    assert rt(True) is True
    assert rt(False) is False
    assert rt(7) == 7
    assert rt(3.5) == 3.5
    assert rt("hello") == "hello"
    assert rt(b"bytes") == b"bytes"


def test_value_roundtrip_collections() -> None:
    rt = uniko._uniko._value_roundtrip
    assert rt([1, 2, 3]) == [1, 2, 3]
    assert rt({"a": 1, "b": [2, 3]}) == {"a": 1, "b": [2, 3]}
    # bool-before-int ordering: a bool must not be coerced to 1/0.
    assert rt([True, 1]) == [True, 1]


def test_value_roundtrip_rejects_unsupported() -> None:
    import pytest

    with pytest.raises(TypeError):
        uniko._uniko._value_roundtrip(object())


def test_records_roundtrip() -> None:
    rows = [{"name": "ada", "age": 36}, {"name": "alan", "age": 41}]
    out = uniko._uniko._records_roundtrip(rows)
    assert out == rows


def test_error_mapping_subclasses_and_kind() -> None:
    import pytest

    cases = {
        "config": uniko.ConfigError,
        "llm": uniko.LlmError,
        "timeout": uniko.UnikoTimeoutError,
        "conflict": uniko.ConflictError,
        "unsupported": uniko.UnsupportedError,
    }
    for kind, exc_cls in cases.items():
        with pytest.raises(exc_cls) as info:
            uniko._uniko._raise_error(kind)
        assert info.value.kind == kind
        assert isinstance(info.value, uniko.UnikoError)


def test_error_mapping_storage_variants_fold_into_base() -> None:
    import pytest

    for kind in (
        "storage",
        "search",
        "schema",
        "pipeline",
        "locy",
        "embedding",
        "internal",
    ):
        with pytest.raises(uniko.UnikoError) as info:
            uniko._uniko._raise_error(kind)
        assert info.value.kind == kind
        # These must NOT be one of the dedicated subclasses.
        assert not isinstance(info.value, uniko.ConfigError)


def test_now_utc_is_tz_aware() -> None:
    now = uniko._uniko._now_utc()
    assert isinstance(now, dt.datetime)
    assert now.tzinfo is not None
    assert now.utcoffset() == dt.timedelta(0)
