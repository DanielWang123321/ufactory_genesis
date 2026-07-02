"""Back-compat shim: layered static dynamics analysis moved to the subpackage.

The implementation now lives in :mod:`ufactory.dynamics.analysis` (with the
dataclasses in :mod:`ufactory.dynamics.report`). This module re-exports the
public surface so existing ``from ufactory.dynamics_static_analysis import X``
importers keep working while code migrates to the subpackage.
"""

from __future__ import annotations

from ufactory.dynamics.analysis import (
    LAYER_L2A,
    LAYER_L3A,
    LAYER_L3B,
    STATIC_LAYERS,
    StaticLayerResult,
    StaticPoseAnalysis,
    analyze_genesis_internal,
    analyze_mass_matrices,
    analyze_pd_vs_pinocchio,
    analyze_pin_vs_real,
    build_static_pose_analysis,
    estimate_pd_residual,
    format_torque_row,
    parse_strict_static_layers,
    pinocchio_static_at_q,
    static_layer_l2,
    summarize_static_layers,
    validate_urdf_dynamics,
)
from ufactory.dynamics.analysis import LAYER_L2B  # noqa: F401
from ufactory.dynamics.reference import DEFAULT_GRAVITY_VECTOR

__all__ = [
    "DEFAULT_GRAVITY_VECTOR",
    "LAYER_L2A",
    "LAYER_L2B",
    "LAYER_L3A",
    "LAYER_L3B",
    "STATIC_LAYERS",
    "StaticLayerResult",
    "StaticPoseAnalysis",
    "analyze_genesis_internal",
    "analyze_mass_matrices",
    "analyze_pd_vs_pinocchio",
    "analyze_pin_vs_real",
    "build_static_pose_analysis",
    "estimate_pd_residual",
    "format_torque_row",
    "parse_strict_static_layers",
    "pinocchio_static_at_q",
    "static_layer_l2",
    "summarize_static_layers",
    "validate_urdf_dynamics",
]
