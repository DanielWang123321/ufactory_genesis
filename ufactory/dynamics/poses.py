"""Runtime constants, pose configs and safety filters for dynamics validation.

Single home for the xArm6 runtime singletons (PD gains, effort/abs limits, joint
names), the calibrated pose config lists, and the safe-pose / joint-limit filters.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ufactory.dynamics.report import SafePose
from ufactory.dynamics.poses_config import dynamics_pose_tuples
from ufactory.robots.runtime import (
    XARM6_EFFORT,
    XARM6_KP,
    XARM6_KV,
    get_robot_runtime_profile,
)

_XARM6_RUNTIME = get_robot_runtime_profile("xarm6")
JOINT_NAMES = _XARM6_RUNTIME.arm.joint_names
EE_LINK_NAME = _XARM6_RUNTIME.arm.ee_link

URDF_JOINT_EFFORT = np.array(XARM6_EFFORT, dtype=np.float64)
PD_KP = np.array(XARM6_KP, dtype=np.float32)
PD_KV = np.array(XARM6_KV, dtype=np.float32)
FORCE_LOWER = -np.array(XARM6_EFFORT, dtype=np.float32)
FORCE_UPPER = np.array(XARM6_EFFORT, dtype=np.float32)

# Legacy extra poses for ad-hoc sim checks (not in assets/configs/dynamics_validation_pose.yaml).
DYNAMICS_EXTRA_CONFIGS: list[tuple[str, np.ndarray]] = [
    ("arm_extended", np.array([0.0, -0.8, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)),
    ("arm_sideways", np.array([1.0, -0.4, 0.0, 0.0, 0.3, 0.0], dtype=np.float64)),
]


def merge_test_configs(
    base_configs: Sequence[tuple[str, np.ndarray]],
    extra_configs: Sequence[tuple[str, np.ndarray]] | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Merge config lists, deduplicating by name (first wins)."""
    merged: list[tuple[str, np.ndarray]] = []
    seen: set[str] = set()
    for configs in (base_configs, extra_configs or ()):
        for name, q in configs:
            if name in seen:
                continue
            seen.add(name)
            merged.append((name, np.asarray(q, dtype=np.float64)))
    return merged


def xarm6_default_dynamics_configs(*, include_stress: bool = False) -> list[tuple[str, np.ndarray]]:
    return dynamics_pose_tuples("xarm6", include_stress=include_stress)


def dynamics_default_configs(
    robot_key: str = "xarm6",
    *,
    include_stress: bool = False,
) -> list[tuple[str, np.ndarray]]:
    runtime = get_robot_runtime_profile(robot_key)
    configs = [(name, np.asarray(q, dtype=np.float64)) for name, q in runtime.dynamics.default_configs]
    if include_stress:
        stress = [(name, np.asarray(q, dtype=np.float64)) for name, q in runtime.dynamics.stress_configs]
        configs = merge_test_configs(configs, stress)
    return configs


def filter_safe_configs(
    configs: Sequence[tuple[str, np.ndarray]],
    ee_z_mm_by_name: dict[str, float],
    z_min_mm: float,
) -> tuple[list[SafePose], list[tuple[str, float]]]:
    safe: list[SafePose] = []
    rejected: list[tuple[str, float]] = []
    for name, q in configs:
        if name not in ee_z_mm_by_name:
            rejected.append((name, float("nan")))
            continue
        ee_z = ee_z_mm_by_name[name]
        if ee_z >= z_min_mm:
            safe.append(SafePose(name=name, q=np.asarray(q, dtype=np.float64), ee_z_mm=ee_z))
        else:
            rejected.append((name, ee_z))
    return safe, rejected


def parse_joint_limits(urdf_path: str | Path, joint_names: Sequence[str] = JOINT_NAMES) -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(str(urdf_path)).getroot()
    lower = np.full(len(joint_names), -np.inf, dtype=np.float64)
    upper = np.full(len(joint_names), np.inf, dtype=np.float64)
    by_name = {joint.get("name"): joint for joint in root.findall("joint")}
    for i, name in enumerate(joint_names):
        joint = by_name.get(name)
        if joint is None:
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        lower[i] = float(limit.get("lower", lower[i]))
        upper[i] = float(limit.get("upper", upper[i]))
    return lower, upper


def check_joint_limit_path(
    start_q: Sequence[float],
    target_q: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    margin_rad: float = 0.02,
    steps: int = 25,
) -> list[str]:
    start = np.asarray(start_q, dtype=np.float64)
    target = np.asarray(target_q, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64) + margin_rad
    hi = np.asarray(upper, dtype=np.float64) - margin_rad
    reasons: list[str] = []
    for i in range(steps + 1):
        alpha = i / steps
        q = (1.0 - alpha) * start + alpha * target
        low_bad = np.where(q < lo)[0]
        high_bad = np.where(q > hi)[0]
        for idx in low_bad:
            reasons.append(f"path step {i}: J{idx + 1} below limit margin ({q[idx]:.4f} < {lo[idx]:.4f})")
        for idx in high_bad:
            reasons.append(f"path step {i}: J{idx + 1} above limit margin ({q[idx]:.4f} > {hi[idx]:.4f})")
        if reasons:
            break
    return reasons
