"""Runtime constants, pose configs and safety filters for dynamics validation.

YAML stores five anchor poses per robot (degrees): home, endpoint A, home,
endpoint B, home. Each home-to-endpoint round trip is expanded at runtime into
10 evenly spaced joint configurations (5 forward + 5 return segments). Two legs
yield 20 named poses ``"0"`` … ``"19"`` in radians.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from ufactory.dynamics.report import SafePose
from ufactory.robots.paths import DYNAMICS_VALIDATION_POSES_YAML
from ufactory.robots.registry import get_robot_profile
from ufactory.robots.runtime import get_robot_runtime_profile

NamedPoseTuple = tuple[tuple[str, tuple[float, ...]], ...]

SEGMENTS_PER_LEG = 5
ANCHOR_COUNT = 5


def _resolve_robot_key(robot_key: str) -> str:
    return get_robot_profile(robot_key).key


def _yaml_robot_key(robot_key: str) -> str:
    """Map profile keys to top-level YAML robot names (e.g. xarm6_1305 -> xarm6)."""
    profile = get_robot_profile(robot_key)
    return profile.robot_name


def _lerp_deg(a_deg: Sequence[float], b_deg: Sequence[float], t: float) -> np.ndarray:
    a = np.deg2rad(np.asarray(a_deg, dtype=np.float64))
    b = np.deg2rad(np.asarray(b_deg, dtype=np.float64))
    return a + float(t) * (b - a)


def expand_round_trip(
    home_deg: Sequence[float],
    end_deg: Sequence[float],
    *,
    segments: int = SEGMENTS_PER_LEG,
) -> list[np.ndarray]:
    """Expand one home-endpoint round trip into ``2 * segments`` poses (radians).

    Forward: t = 1/segments … 1 (endpoint). Return: t = (segments-1)/segments … 0 (home).
    """
    if segments < 1:
        raise ValueError(f"segments must be >= 1, got {segments}")
    out: list[np.ndarray] = []
    for k in range(1, segments + 1):
        out.append(_lerp_deg(home_deg, end_deg, k / segments))
    for k in range(segments - 1, -1, -1):
        out.append(_lerp_deg(home_deg, end_deg, k / segments))
    return out


def _parse_anchor_rows(
    rows: Any,
    *,
    robot_key: str,
    dof: int,
) -> list[list[float]]:
    if not isinstance(rows, list) or len(rows) != ANCHOR_COUNT:
        raise ValueError(
            f"{robot_key}.points: expected {ANCHOR_COUNT} anchor rows, got "
            f"{len(rows) if isinstance(rows, list) else type(rows).__name__}"
        )
    out: list[list[float]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, (list, tuple)):
            raise ValueError(f"{robot_key}.points[{i}]: expected list of floats, got {type(row).__name__}")
        if len(row) != dof:
            raise ValueError(f"{robot_key}.points[{i}]: expected {dof} joints, got {len(row)}")
        out.append([float(v) for v in row])
    if out[0] != out[2] or out[0] != out[4]:
        raise ValueError(f"{robot_key}.points: anchors 0, 2, 4 must be identical home poses")
    return out


@lru_cache(maxsize=1)
def _load_yaml(path: Path | None = None) -> dict[str, Any]:
    yaml_path = path or DYNAMICS_VALIDATION_POSES_YAML
    with yaml_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid dynamics poses YAML: {yaml_path}")
    return data


def load_dynamics_pose_lists(
    robot_key: str,
    *,
    yaml_path: Path | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return ``(default_qs, stress_qs)`` for ``robot_key``."""
    key = _resolve_robot_key(robot_key)
    yaml_key = _yaml_robot_key(key)
    entry = _load_yaml(yaml_path).get(yaml_key)
    if entry is None:
        return [], []
    dof = get_robot_profile(key).dof
    points = entry.get("points")
    anchors = _parse_anchor_rows(points, robot_key=yaml_key, dof=dof)
    home_a, end_a, home_b, end_b, home_c = anchors
    if home_a != home_b or home_a != home_c:
        raise ValueError(f"{yaml_key}.points: all home anchors must match")
    default: list[np.ndarray] = []
    default.extend(expand_round_trip(home_a, end_a))
    default.extend(expand_round_trip(home_a, end_b))
    return default, []


def dynamics_pose_tuples(
    robot_key: str,
    *,
    include_stress: bool = False,
    yaml_path: Path | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Named poses with auto-generated index names ``"0"`` … ``"N"``."""
    default, stress = load_dynamics_pose_lists(robot_key, yaml_path=yaml_path)
    configs: list[tuple[str, np.ndarray]] = [(str(i), q) for i, q in enumerate(default)]
    if include_stress:
        base = len(default)
        configs.extend((str(base + j), q) for j, q in enumerate(stress))
    return configs


def default_configs_named_tuple(
    robot_key: str,
    *,
    yaml_path: Path | None = None,
) -> NamedPoseTuple:
    default, _ = load_dynamics_pose_lists(robot_key, yaml_path=yaml_path)
    if not default:
        return ()
    return tuple((str(i), tuple(float(v) for v in q)) for i, q in enumerate(default))


def stress_configs_for_robot(
    robot_key: str,
    *,
    yaml_path: Path | None = None,
) -> NamedPoseTuple:
    default, stress = load_dynamics_pose_lists(robot_key, yaml_path=yaml_path)
    if not stress:
        return ()
    base = len(default)
    return tuple((str(base + j), tuple(float(v) for v in q)) for j, q in enumerate(stress))


_XARM6_RUNTIME = get_robot_runtime_profile("xarm6")
JOINT_NAMES = _XARM6_RUNTIME.arm.joint_names


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


def parse_joint_limits(
    urdf_path: str | Path, joint_names: Sequence[str] = JOINT_NAMES
) -> tuple[np.ndarray, np.ndarray]:
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
