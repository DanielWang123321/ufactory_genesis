"""Reusable packaging task geometry, planning, and Genesis scene helpers."""

from __future__ import annotations

from typing import Any

from ufactory.manipulation.packaging.core import (
    OBJECT_RELOCATED_STAGES,
    PackagingLayout,
    build_packaging_program,
    packaging_layout,
    packaging_obstacles,
    packaging_scene_sha256,
    validate_payload_box_clearance,
)

__all__ = [
    "OBJECT_RELOCATED_STAGES",
    "NaturalDropReport",
    "PackagingLayout",
    "build_packaging_program",
    "build_packaging_scene",
    "measure_natural_drop",
    "packaging_layout",
    "packaging_obstacles",
    "packaging_scene_sha256",
    "validate_payload_box_clearance",
]


def __getattr__(name: str) -> Any:
    """Keep Genesis out of geometry-only and CLI configuration imports."""

    if name == "build_packaging_scene":
        from ufactory.manipulation.packaging.scene import build_packaging_scene

        return build_packaging_scene
    if name in {"NaturalDropReport", "measure_natural_drop"}:
        from ufactory.manipulation.packaging.drop import NaturalDropReport, measure_natural_drop

        return {"NaturalDropReport": NaturalDropReport, "measure_natural_drop": measure_natural_drop}[name]
    raise AttributeError(name)
