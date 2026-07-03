"""Optional trajectory backend loaders."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


class OptionalTrajectoryDependencyError(ImportError):
    """Raised when an optional trajectory backend is requested but missing."""


def require_roboticstoolbox() -> ModuleType:
    """Import and return ``roboticstoolbox`` with a clear install hint."""
    try:
        return import_module("roboticstoolbox")
    except ModuleNotFoundError as exc:
        raise OptionalTrajectoryDependencyError(
            "roboticstoolbox-python is required for the optional trajectory "
            'reference backend. Install it with: pip install -e ".[trajectory]"'
        ) from exc
