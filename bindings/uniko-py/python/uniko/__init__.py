"""uniko — async-first Python SDK for the uniko cognitive memory engine.

The compiled PyO3 extension is imported as ``uniko._uniko`` and its public
surface is re-exported here so callers write ``import uniko`` /
``from uniko import Uniko``.

Almost every method is async (returns an awaitable driven by ``asyncio``);
construction helpers and builder setters are synchronous.
"""

from __future__ import annotations

from . import _uniko
from ._uniko import *  # noqa: F401,F403  (re-export the full native surface)

# The Pydantic IO layer is reachable as ``uniko.models`` (e.g.
# ``from uniko.models import ContextBundle, GoalSpec, TypedUniko``). Importing it
# here only binds the submodule; it does NOT pull model classes into ``uniko.*``.
from . import models  # noqa: F401,E402

# Mirror the native module's __all__ when it defines one; otherwise fall back
# to every public attribute the extension exposes.
__all__ = getattr(
    _uniko, "__all__", [name for name in dir(_uniko) if not name.startswith("_")]
)

# `_uniko.__version__` is `env!("CARGO_PKG_VERSION")` — the crate version, which
# is `version.workspace = true`, i.e. the single workspace version. The fallback
# is a deliberately-invalid PEP 440 local sentinel (never a hardcoded release
# number) so a stale/missing export can never masquerade as a real version.
__version__ = getattr(_uniko, "__version__", "0+unknown")


def __getattr__(name: str) -> object:
    # `uniko.TimeoutError` was renamed `UnikoTimeoutError` so that
    # `from uniko import *` no longer shadows the builtin `TimeoutError`.
    # The old name still resolves, with a warning, until the next minor release.
    if name == "TimeoutError":
        import warnings

        warnings.warn(
            "uniko.TimeoutError is deprecated; use uniko.UnikoTimeoutError",
            DeprecationWarning,
            stacklevel=2,
        )
        return _uniko.UnikoTimeoutError
    raise AttributeError(f"module 'uniko' has no attribute {name!r}")
