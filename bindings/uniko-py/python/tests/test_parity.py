# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Dragonscale Industries Inc.
"""The hand-kept Python layers must keep up with the native module.

Three copies of the API are maintained by hand: the native classes, the
``_uniko.pyi`` stub (checked by ``mypy.stubtest`` in CI), and the Pydantic
layer in ``uniko.models``. These tests fail when a native method has no typed
wrapper, when a native result field has no model field, or when
``UnikoConfigModel`` drifts from the engine's configuration.
"""

from __future__ import annotations

import inspect

import pytest

import uniko
from uniko import _uniko
from uniko.models import UnikoConfigModel, outputs, typed

# Native handle -> typed wrapper.
HANDLES = {
    "Uniko": typed.TypedUniko,
    "UnikoBuilder": typed.TypedUnikoBuilder,
    "Agent": typed.TypedAgent,
    "Session": typed.TypedSession,
    "Data": typed.TypedData,
    "Goals": typed.TypedGoals,
}

# Native methods deliberately without a typed counterpart, with the reason.
# Keep this short: an entry here is a gap a typed caller has to work around.
TYPED_EXCEPTIONS: dict[str, set[str]] = {}

# Native classes built by the caller rather than returned, so they need no
# output model even though their public members are all fields.
INPUT_CLASSES = {"Viewer", "ExtractRequest"}

# Methods a result class may carry besides its fields.
RESULT_METHODS = {"citations", "message_node_ids"}

# Native result attributes a model does not mirror, with the reason.
MODEL_EXCEPTIONS: dict[str, set[str]] = {}


def public_members(cls: type) -> set[str]:
    return {name for name in dir(cls) if not name.startswith("_")}


def native_fields(cls: type) -> set[str]:
    """The data attributes of a native pyclass (getters, not methods)."""
    return {
        name
        for name, member in inspect.getmembers(cls)
        if not name.startswith("_") and inspect.isgetsetdescriptor(member)
    }


@pytest.mark.parametrize("native_name", sorted(HANDLES))
def test_every_native_method_has_a_typed_wrapper(native_name: str) -> None:
    native = getattr(_uniko, native_name)
    wrapper = HANDLES[native_name]
    missing = (
        public_members(native)
        - public_members(wrapper)
        - TYPED_EXCEPTIONS.get(native_name, set())
    )
    assert not missing, f"{wrapper.__name__} lacks {sorted(missing)}"


def output_class_names() -> list[str]:
    return sorted(
        name
        for name in outputs.__all__
        if hasattr(_uniko, name) and isinstance(getattr(_uniko, name), type)
    )


def test_every_native_result_class_has_a_model() -> None:
    # A result class is a native class whose public members are all data
    # fields; handles (Agent, Session, ...) and builders carry methods.
    def is_result(cls: object) -> bool:
        if not isinstance(cls, type) or issubclass(cls, BaseException):
            return False
        if cls.__name__ in INPUT_CLASSES:
            return False
        fields = native_fields(cls)
        return bool(fields) and fields >= public_members(cls) - RESULT_METHODS

    native_results = {
        name for name in uniko.__all__ if is_result(getattr(_uniko, name))
    }
    missing = native_results - set(outputs.__all__)
    assert not missing, f"no Pydantic model for {sorted(missing)}"


@pytest.mark.parametrize("name", output_class_names())
def test_result_models_cover_every_native_field(name: str) -> None:
    native = native_fields(getattr(_uniko, name))
    model = set(getattr(outputs, name).model_fields)
    missing = native - model - MODEL_EXCEPTIONS.get(name, set())
    assert not missing, f"outputs.{name} lacks {sorted(missing)}"


def test_config_model_matches_the_engine_config() -> None:
    engine = uniko.Uniko.in_memory_sync().config()
    model = UnikoConfigModel().model_dump()

    def diff(a: object, b: object, path: str = "") -> list[str]:
        if isinstance(a, dict) and isinstance(b, dict):
            out = []
            for key in sorted(a.keys() | b.keys()):
                child = f"{path}.{key}" if path else key
                if key not in a:
                    out.append(f"{child}: only in UnikoConfigModel")
                elif key not in b:
                    out.append(f"{child}: missing from UnikoConfigModel")
                else:
                    out.extend(diff(a[key], b[key], child))
            return out
        return [] if a == b else [f"{path}: engine {a!r} != model {b!r}"]

    problems = diff(engine, model)
    assert not problems, "UnikoConfigModel drifted:\n" + "\n".join(problems)
