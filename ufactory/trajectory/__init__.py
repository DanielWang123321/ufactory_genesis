"""Stable v0.2.5 trajectory application API.

Legacy replay helpers remain implementation modules for internal migration but
are intentionally not re-exported. Real execution requires an ApprovedProgram.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "plan_mixed_waypoints",
    "preflight_program",
    "execute_sim",
    "execute_real",
]

_PUBLIC = {
    "plan_mixed_waypoints": ("ufactory.trajectory.planner", "plan_mixed_waypoints"),
    "preflight_program": ("ufactory.trajectory.preflight", "preflight_program"),
    "execute_sim": ("ufactory.trajectory.execution", "execute_sim"),
    "execute_real": ("ufactory.trajectory.execution", "execute_real"),
}


def __getattr__(name: str) -> Any:
    target = _PUBLIC.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
