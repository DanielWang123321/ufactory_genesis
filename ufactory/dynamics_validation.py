"""Enterprise-style dynamics validation helpers for UFACTORY Genesis models.

This module separates three concerns that were previously mixed together:

* Genesis controller output (PD hold torque)
* Genesis internal rigid-body state queries (DOF force, mass matrix)
* Real-robot torque/effort measurement and report generation

The public examples should stay thin and call the CLI entry points here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from ufactory.paths import robot_urdf
from ufactory.robot_params import (
    RobotRuntimeProfile,
    XARM6_ABS_ERR_LIMITS as _XARM6_ABS_ERR_LIMITS,
    XARM6_DEFAULT_DYNAMICS_CONFIGS as _XARM6_DEFAULT_DYNAMICS_CONFIGS,
    XARM6_EFFORT as _XARM6_EFFORT,
    XARM6_KP as _XARM6_KP,
    XARM6_KV as _XARM6_KV,
    XARM6_STRESS_DYNAMICS_CONFIGS as _XARM6_STRESS_DYNAMICS_CONFIGS,
    get_robot_runtime_profile,
    robot_runtime_cli_choices,
)

_XARM6_RUNTIME = get_robot_runtime_profile("xarm6")
JOINT_NAMES = _XARM6_RUNTIME.arm.joint_names
EE_LINK_NAME = _XARM6_RUNTIME.arm.ee_link

URDF_JOINT_EFFORT = np.array(_XARM6_EFFORT, dtype=np.float64)

# PD gains aligned with the xArm6 runtime profile; kept for compatibility.
PD_KP = np.array(_XARM6_KP, dtype=np.float32)
PD_KV = np.array(_XARM6_KV, dtype=np.float32)
FORCE_LOWER = -np.array(_XARM6_EFFORT, dtype=np.float32)
FORCE_UPPER = np.array(_XARM6_EFFORT, dtype=np.float32)

SIM_DT = 0.01
SIM_SUBSTEPS = 1
SETTLE_STEPS = 500
POS_ERR_TOL = 0.05  # rad
VEL_TOL = 0.01  # rad/s
SATURATION_MARGIN = 0.995

# Per-joint absolute error limits (Nm), kept as the V1 compatibility gate.
ABS_ERR_LIMITS = np.array(_XARM6_ABS_ERR_LIMITS, dtype=np.float64)
L2_ERR_LIMIT = 5.0  # Nm
REL_ERR_LIMIT = 0.15  # fraction of effort limit


class ValidationStatus(str, Enum):
    PASS = "PASS"
    FAIL_MODEL = "FAIL_MODEL"
    FAIL_BIAS = "FAIL_BIAS"
    NOT_SETTLED = "NOT_SETTLED"
    SATURATED = "SATURATED"
    UNSAFE = "UNSAFE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class SafePose:
    name: str
    q: np.ndarray
    ee_z_mm: float


@dataclass
class UrdfDynamicsIssue:
    severity: str
    item: str
    message: str
    value: Any = None


@dataclass
class DynamicsRunConfig:
    robot_key: str
    urdf_path: str
    urdf_sha256: str | None = None
    kinematics_yaml_path: str | None = None
    kinematics_yaml_sha256: str | None = None
    genesis_version: str | None = None
    genesis_backend: str | None = None
    sim_dt: float = SIM_DT
    sim_substeps: int = SIM_SUBSTEPS
    integrator: str | None = None
    sdk_version: str | None = None
    firmware: str | None = None
    robot_sn: str | None = None
    tcp_load: list[float] | None = None
    gravity_direction: list[float] | None = None
    git_sha: str | None = None
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    mode: str = "dry-run"


@dataclass
class GenesisDynamicsSample:
    q_actual: np.ndarray
    qvel: np.ndarray
    pd_hold_tau: np.ndarray
    actual_dof_force: np.ndarray
    mass_matrix: np.ndarray
    settled: bool
    saturated: bool
    pos_err: float
    vel_mag: float


@dataclass
class DynamicsSample:
    pose: str
    q: np.ndarray
    ee_z_mm: float
    status: ValidationStatus
    settled: bool = False
    saturated: bool = False
    q_actual: np.ndarray | None = None
    qvel: np.ndarray | None = None
    pd_hold_tau: np.ndarray | None = None
    actual_dof_force: np.ndarray | None = None
    mass_matrix: np.ndarray | None = None
    reference_gravity_tau: np.ndarray | None = None
    tau_real: np.ndarray | None = None
    tau_real_median: np.ndarray | None = None
    tau_real_std: np.ndarray | None = None
    tau_real_min: np.ndarray | None = None
    tau_real_max: np.ndarray | None = None
    tau_direct: np.ndarray | None = None
    abs_err: np.ndarray | None = None
    rel_err: np.ndarray | None = None
    signed_err: np.ndarray | None = None
    l2_err: float | None = None
    n_real_samples: int = 0
    skip_reason: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class TorqueCompareResult:
    """Backward-compatible result object for the older real-robot script."""

    name: str
    q: np.ndarray
    ee_z_mm: float
    tau_genesis: np.ndarray
    tau_real: np.ndarray
    abs_err: np.ndarray
    rel_err: np.ndarray
    l2_err: float
    settled: bool
    passed: bool
    skip_reason: str = ""
    status: ValidationStatus = ValidationStatus.INSUFFICIENT_DATA


# Default hardware acceptance poses are now sourced from ufactory.robot_params.
XARM6_DEFAULT_DYNAMICS_CONFIGS: list[tuple[str, np.ndarray]] = [
    (name, np.asarray(q, dtype=np.float64)) for name, q in _XARM6_DEFAULT_DYNAMICS_CONFIGS
]

XARM6_STRESS_DYNAMICS_CONFIGS: list[tuple[str, np.ndarray]] = [
    (name, np.asarray(q, dtype=np.float64)) for name, q in _XARM6_STRESS_DYNAMICS_CONFIGS
]

# Backward-compatible name used by the older script.
DYNAMICS_EXTRA_CONFIGS: list[tuple[str, np.ndarray]] = [
    ("arm_extended", np.array([0.0, -0.8, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)),
    ("arm_sideways", np.array([1.0, -0.4, 0.0, 0.0, 0.3, 0.0], dtype=np.float64)),
]


def sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_sha(root: str | Path | None = None) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root or Path.cwd()),
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return out.strip() or None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def array_or_nan(size: int = 6) -> np.ndarray:
    return np.full(size, np.nan, dtype=np.float64)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


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
    configs = list(XARM6_DEFAULT_DYNAMICS_CONFIGS)
    if include_stress:
        configs = merge_test_configs(configs, XARM6_STRESS_DYNAMICS_CONFIGS)
    return configs


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


def _runtime_dof(runtime: RobotRuntimeProfile) -> int:
    return runtime.model.dof


def _effort_limits(runtime: RobotRuntimeProfile) -> np.ndarray:
    return np.asarray(runtime.arm.effort_limits, dtype=np.float64)


def _abs_err_limits(runtime: RobotRuntimeProfile) -> np.ndarray:
    return np.asarray(runtime.dynamics.abs_err_limits, dtype=np.float64)


def _array_or_nan_for_runtime(runtime: RobotRuntimeProfile) -> np.ndarray:
    return array_or_nan(_runtime_dof(runtime))


def format_torque_row(values: Iterable[float]) -> str:
    return "[" + ", ".join(f"{float(v):8.3f}" for v in values) + "]"


def resolve_entity_name(entity, requested_name: str, kind: str) -> str:
    available = {item.name for item in entity.joints} if kind == "joint" else {item.name for item in entity.links}
    if requested_name in available:
        return requested_name
    fallback = requested_name.split("/")[-1]
    if fallback in available:
        return fallback
    raise KeyError(f"{kind.capitalize()} name not found: {requested_name}. Available: {sorted(available)}")


def build_genesis_scene(
    urdf_path: str,
    *,
    runtime_profile: RobotRuntimeProfile | None = None,
    show_viewer: bool = False,
    sim_dt: float = SIM_DT,
    sim_substeps: int = SIM_SUBSTEPS,
    backend: str = "gpu",
):
    """Create a minimal fixed-base Genesis scene for dynamics checks."""
    import genesis as gs

    runtime = runtime_profile or _XARM6_RUNTIME
    gs_backend = gs.gpu if backend == "gpu" else gs.cpu
    gs.init(backend=gs_backend)
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -1.5, 1.5),
            camera_lookat=(0.0, 0.0, 0.4),
            camera_fov=40,
            refresh_rate=60,
        ),
        sim_options=gs.options.SimOptions(dt=sim_dt, substeps=sim_substeps),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    robot = scene.add_entity(gs.morphs.URDF(file=str(Path(urdf_path).resolve()), pos=(0, 0, 0), fixed=True))
    scene.build()

    available_joints = {j.name: j for j in robot.joints}
    dof_idx = [
        available_joints[resolve_entity_name(robot, name, "joint")].dofs_idx_local[0]
        for name in runtime.arm.joint_names
    ]
    ee_link = robot.get_link(resolve_entity_name(robot, runtime.arm.ee_link, "link"))
    return scene, robot, ee_link, dof_idx


def set_pd_gains(
    robot,
    dof_idx: Sequence[int],
    runtime_profile: RobotRuntimeProfile | None = None,
) -> None:
    """Apply profile-specific arm PD gains and force limits."""
    runtime = runtime_profile or _XARM6_RUNTIME
    robot.set_dofs_kp(np.asarray(runtime.arm.kp, dtype=np.float32), dof_idx)
    robot.set_dofs_kv(np.asarray(runtime.arm.kv, dtype=np.float32), dof_idx)
    robot.set_dofs_force_range(
        np.asarray(runtime.arm.force_lower, dtype=np.float32),
        np.asarray(runtime.arm.force_upper, dtype=np.float32),
        dof_idx,
    )


def _to_np(tensor_or_array) -> np.ndarray:
    if hasattr(tensor_or_array, "cpu"):
        return tensor_or_array.cpu().numpy()
    return np.asarray(tensor_or_array)


def genesis_pd_hold_torque_at_q(
    robot,
    scene,
    dof_idx: Sequence[int],
    target_q: np.ndarray,
    *,
    runtime_profile: RobotRuntimeProfile | None = None,
    settle_steps: int = SETTLE_STEPS,
    pos_tol: float = POS_ERR_TOL,
    vel_tol: float = VEL_TOL,
    effort_limits: np.ndarray | None = None,
) -> GenesisDynamicsSample:
    """Hold ``target_q`` with Genesis PD and return explicitly named physics quantities."""
    runtime = runtime_profile or _XARM6_RUNTIME
    set_pd_gains(robot, dof_idx, runtime)
    target = np.asarray(target_q, dtype=np.float32)
    robot.control_dofs_position(target, dof_idx)
    for _ in range(settle_steps):
        scene.step()

    qpos = _to_np(robot.get_dofs_position(dof_idx)).flatten().astype(np.float64)
    qvel = _to_np(robot.get_dofs_velocity(dof_idx)).flatten().astype(np.float64)
    pd_hold_tau = _to_np(robot.get_dofs_control_force(dof_idx)).flatten().astype(np.float64)
    actual_dof_force = _to_np(robot.get_dofs_force(dof_idx)).flatten().astype(np.float64)
    mass_matrix = _to_np(robot.get_mass_mat()).astype(np.float64)
    if mass_matrix.ndim == 3:
        mass_matrix = mass_matrix[0]

    pos_err = float(np.abs(qpos - target).max())
    vel_mag = float(np.abs(qvel).max())
    settled = pos_err <= pos_tol and vel_mag <= vel_tol
    limits = effort_limits if effort_limits is not None else _effort_limits(runtime)
    saturated = bool(np.any(np.abs(pd_hold_tau) >= np.asarray(limits, dtype=np.float64) * SATURATION_MARGIN))
    return GenesisDynamicsSample(
        q_actual=qpos,
        qvel=qvel,
        pd_hold_tau=pd_hold_tau,
        actual_dof_force=actual_dof_force,
        mass_matrix=mass_matrix,
        settled=settled,
        saturated=saturated,
        pos_err=pos_err,
        vel_mag=vel_mag,
    )


def genesis_gravity_torque_at_q(
    robot,
    scene,
    dof_idx: Sequence[int],
    target_q: np.ndarray,
    *,
    runtime_profile: RobotRuntimeProfile | None = None,
    settle_steps: int = SETTLE_STEPS,
    pos_tol: float = POS_ERR_TOL,
    vel_tol: float = VEL_TOL,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Backward-compatible alias returning Genesis PD hold torque.

    The name is intentionally deprecated: the returned torque is controller
    output from ``get_dofs_control_force()``, not a direct inverse-dynamics
    gravity vector.
    """
    sample = genesis_pd_hold_torque_at_q(
        robot,
        scene,
        dof_idx,
        target_q,
        runtime_profile=runtime_profile,
        settle_steps=settle_steps,
        pos_tol=pos_tol,
        vel_tol=vel_tol,
    )
    return sample.q_actual, sample.pd_hold_tau, sample.settled


def genesis_ee_z_mm_at_q(robot, scene, ee_link, dof_idx, q: np.ndarray, settle_steps: int = 5) -> float:
    """EE z (mm) by setting joint positions and reading link6 world position."""
    _, _, z_mm = genesis_ee_xyz_mm_at_q(robot, scene, ee_link, dof_idx, q, settle_steps=settle_steps)
    return z_mm


def genesis_ee_xyz_mm_at_q(
    robot,
    scene,
    ee_link,
    dof_idx,
    q: np.ndarray,
    settle_steps: int = 5,
) -> tuple[float, float, float]:
    """EE (x, y, z) in mm by setting joint positions and reading link6 world position."""
    robot.set_dofs_position(np.asarray(q, dtype=np.float32), dof_idx)
    for _ in range(settle_steps):
        scene.step()
    pos = ee_link.get_pos()
    if pos.dim() == 1:
        x, y, z = pos[0].item(), pos[1].item(), pos[2].item()
    else:
        x, y, z = pos[0, 0].item(), pos[0, 1].item(), pos[0, 2].item()
    return float(x * 1000.0), float(y * 1000.0), float(z * 1000.0)


def compute_ee_xyz_table_from_sim(
    robot,
    scene,
    ee_link,
    dof_idx,
    configs: Sequence[tuple[str, np.ndarray]],
) -> dict[str, tuple[float, float, float]]:
    table: dict[str, tuple[float, float, float]] = {}
    for name, q in configs:
        table[name] = genesis_ee_xyz_mm_at_q(robot, scene, ee_link, dof_idx, q)
    return table


def compute_ee_z_table_from_sim(
    robot,
    scene,
    ee_link,
    dof_idx,
    configs: Sequence[tuple[str, np.ndarray]],
) -> dict[str, float]:
    table: dict[str, float] = {}
    for name, q in configs:
        table[name] = genesis_ee_z_mm_at_q(robot, scene, ee_link, dof_idx, q)
    return table


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


def check_genesis_path_z(
    robot,
    scene,
    ee_link,
    dof_idx,
    start_q: Sequence[float],
    target_q: Sequence[float],
    *,
    z_min_mm: float,
    steps: int = 25,
) -> list[str]:
    start = np.asarray(start_q, dtype=np.float64)
    target = np.asarray(target_q, dtype=np.float64)
    reasons: list[str] = []
    for i in range(steps + 1):
        alpha = i / steps
        q = (1.0 - alpha) * start + alpha * target
        z = genesis_ee_z_mm_at_q(robot, scene, ee_link, dof_idx, q, settle_steps=1)
        if z < z_min_mm:
            reasons.append(f"Genesis path step {i}: EE z {z:.2f} mm < z_min {z_min_mm:.2f} mm")
            break
    return reasons


def validate_urdf_dynamics(urdf_path: str | Path, *, com_abs_limit_m: float = 2.0) -> list[UrdfDynamicsIssue]:
    """Static URDF dynamics checks for inertial and joint dynamics blocks."""
    root = ET.parse(str(urdf_path)).getroot()
    issues: list[UrdfDynamicsIssue] = []

    for link in root.findall("link"):
        name = link.get("name", "")
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass_el = inertial.find("mass")
        if mass_el is None or "value" not in mass_el.attrib:
            issues.append(UrdfDynamicsIssue("ERROR", name, "missing inertial mass"))
            continue
        mass = float(mass_el.attrib["value"])
        if mass <= 0:
            issues.append(UrdfDynamicsIssue("ERROR", name, "mass must be positive", mass))

        origin = inertial.find("origin")
        if origin is not None:
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            if xyz.size == 3 and float(np.max(np.abs(xyz))) > com_abs_limit_m:
                issues.append(UrdfDynamicsIssue("WARN", name, "COM magnitude looks too large", xyz.tolist()))

        inertia = inertial.find("inertia")
        if inertia is None:
            issues.append(UrdfDynamicsIssue("ERROR", name, "missing inertia matrix"))
            continue
        vals = {k: float(inertia.get(k, 0.0)) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
        mat = np.array(
            [
                [vals["ixx"], vals["ixy"], vals["ixz"]],
                [vals["ixy"], vals["iyy"], vals["iyz"]],
                [vals["ixz"], vals["iyz"], vals["izz"]],
            ],
            dtype=np.float64,
        )
        eig = np.linalg.eigvalsh(mat)
        if float(eig.min()) <= 0.0:
            issues.append(UrdfDynamicsIssue("ERROR", name, "inertia matrix is not positive definite", eig.tolist()))
        if not (
            vals["ixx"] + vals["iyy"] >= vals["izz"]
            and vals["ixx"] + vals["izz"] >= vals["iyy"]
            and vals["iyy"] + vals["izz"] >= vals["ixx"]
        ):
            issues.append(UrdfDynamicsIssue("ERROR", name, "principal inertia triangle inequality failed", vals))

    for joint in root.findall("joint"):
        joint_type = joint.get("type")
        if joint_type not in {"revolute", "continuous", "prismatic"}:
            continue
        name = joint.get("name", "")
        limit = joint.find("limit")
        if joint_type != "continuous" and limit is None:
            issues.append(UrdfDynamicsIssue("ERROR", name, "missing joint limit"))
        if limit is not None:
            for attr in ("effort", "velocity"):
                if attr not in limit.attrib:
                    issues.append(UrdfDynamicsIssue("ERROR", name, f"missing limit {attr}"))
                elif float(limit.attrib[attr]) <= 0:
                    issues.append(UrdfDynamicsIssue("ERROR", name, f"limit {attr} must be positive", limit.attrib[attr]))
        dynamics = joint.find("dynamics")
        if dynamics is None:
            issues.append(UrdfDynamicsIssue("WARN", name, "missing joint dynamics block"))
            continue
        for attr in ("damping", "friction"):
            if attr not in dynamics.attrib:
                issues.append(UrdfDynamicsIssue("WARN", name, f"missing dynamics {attr}"))
            elif float(dynamics.attrib[attr]) < 0:
                issues.append(UrdfDynamicsIssue("ERROR", name, f"dynamics {attr} must be non-negative", dynamics.attrib[attr]))

    return issues


class PinocchioReference:
    """Optional independent rigid-body dynamics reference backend."""

    def __init__(self, urdf_path: str | Path):
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise ImportError(
                "Pinocchio reference backend is unavailable. Install optional dependency: pip install '.[dynamics]'"
            ) from exc

        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()

    @property
    def available(self) -> bool:
        return True

    def mass_matrix(self, q: Sequence[float]) -> np.ndarray:
        q_np = np.asarray(q, dtype=np.float64)
        mat = self.pin.crba(self.model, self.data, q_np)
        return np.asarray((mat + mat.T) * 0.5, dtype=np.float64)

    def gravity(self, q: Sequence[float]) -> np.ndarray:
        q_np = np.asarray(q, dtype=np.float64)
        return np.asarray(self.pin.computeGeneralizedGravity(self.model, self.data, q_np), dtype=np.float64)

    def rnea(self, q: Sequence[float], qd: Sequence[float], qdd: Sequence[float]) -> np.ndarray:
        return np.asarray(
            self.pin.rnea(
                self.model,
                self.data,
                np.asarray(q, dtype=np.float64),
                np.asarray(qd, dtype=np.float64),
                np.asarray(qdd, dtype=np.float64),
            ),
            dtype=np.float64,
        )


def load_reference_backend(urdf_path: str | Path, *, required: bool = False) -> PinocchioReference | None:
    try:
        return PinocchioReference(urdf_path)
    except ImportError:
        if required:
            raise
        return None


def compare_torques(
    tau_genesis: np.ndarray,
    tau_real: np.ndarray,
    *,
    runtime_profile: RobotRuntimeProfile | None = None,
    effort_limits: np.ndarray | None = None,
    abs_limits: np.ndarray | None = None,
    l2_limit: float = L2_ERR_LIMIT,
    rel_limit: float = REL_ERR_LIMIT,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    runtime = runtime_profile or _XARM6_RUNTIME
    limits = effort_limits if effort_limits is not None else _effort_limits(runtime)
    abs_lims = abs_limits if abs_limits is not None else _abs_err_limits(runtime)

    tau_g = np.asarray(tau_genesis, dtype=np.float64).reshape(-1)
    tau_r = np.asarray(tau_real, dtype=np.float64).reshape(-1)
    abs_err = np.abs(tau_g - tau_r)
    rel_err = abs_err / limits
    l2_err = float(np.linalg.norm(abs_err))
    passed = bool((abs_err <= abs_lims).all() and (rel_err <= rel_limit).all() and l2_err <= l2_limit)
    return abs_err, rel_err, l2_err, passed


def classify_torque_result(
    *,
    settled: bool,
    saturated: bool,
    tau_real: np.ndarray | None,
    abs_err: np.ndarray | None = None,
    rel_err: np.ndarray | None = None,
    l2_err: float | None = None,
    runtime_profile: RobotRuntimeProfile | None = None,
    abs_limits: np.ndarray | None = None,
    l2_limit: float = L2_ERR_LIMIT,
    rel_limit: float = REL_ERR_LIMIT,
) -> ValidationStatus:
    if not settled:
        return ValidationStatus.NOT_SETTLED
    if saturated:
        return ValidationStatus.SATURATED
    if tau_real is None or not np.isfinite(np.asarray(tau_real, dtype=np.float64)).all():
        return ValidationStatus.INSUFFICIENT_DATA
    if abs_err is None or rel_err is None or l2_err is None:
        return ValidationStatus.INSUFFICIENT_DATA

    runtime = runtime_profile or _XARM6_RUNTIME
    limits = abs_limits if abs_limits is not None else _abs_err_limits(runtime)
    if bool((abs_err <= limits).all() and (rel_err <= rel_limit).all() and l2_err <= l2_limit):
        return ValidationStatus.PASS
    failed = np.where(abs_err > limits)[0]
    if len(failed) == 1 and l2_err <= l2_limit * 1.25:
        return ValidationStatus.FAIL_BIAS
    return ValidationStatus.FAIL_MODEL


def build_dynamics_sample(
    pose: SafePose,
    genesis_sample: GenesisDynamicsSample,
    *,
    runtime_profile: RobotRuntimeProfile | None = None,
    tau_real: np.ndarray | None = None,
    tau_real_median: np.ndarray | None = None,
    tau_real_std: np.ndarray | None = None,
    tau_real_min: np.ndarray | None = None,
    tau_real_max: np.ndarray | None = None,
    tau_direct: np.ndarray | None = None,
    n_real_samples: int = 0,
    reference_gravity_tau: np.ndarray | None = None,
    skip_reason: str = "",
    notes: list[str] | None = None,
) -> DynamicsSample:
    runtime = runtime_profile or _XARM6_RUNTIME
    abs_err = rel_err = signed_err = None
    l2_err: float | None = None
    if tau_real is not None and np.isfinite(np.asarray(tau_real, dtype=np.float64)).all():
        signed_err = genesis_sample.pd_hold_tau - np.asarray(tau_real, dtype=np.float64)
        abs_err, rel_err, l2_err, _ = compare_torques(
            genesis_sample.pd_hold_tau,
            tau_real,
            runtime_profile=runtime,
            l2_limit=runtime.dynamics.l2_err_limit,
            rel_limit=runtime.dynamics.rel_err_limit,
        )

    status = classify_torque_result(
        settled=genesis_sample.settled,
        saturated=genesis_sample.saturated,
        tau_real=tau_real,
        abs_err=abs_err,
        rel_err=rel_err,
        l2_err=l2_err,
        runtime_profile=runtime,
        l2_limit=runtime.dynamics.l2_err_limit,
        rel_limit=runtime.dynamics.rel_err_limit,
    )
    if skip_reason == "not settled":
        status = ValidationStatus.NOT_SETTLED
    if skip_reason == "dry-run" and status == ValidationStatus.INSUFFICIENT_DATA:
        notes = [*(notes or []), "dry-run: no real robot torque data"]

    return DynamicsSample(
        pose=pose.name,
        q=pose.q,
        ee_z_mm=pose.ee_z_mm,
        status=status,
        settled=genesis_sample.settled,
        saturated=genesis_sample.saturated,
        q_actual=genesis_sample.q_actual,
        qvel=genesis_sample.qvel,
        pd_hold_tau=genesis_sample.pd_hold_tau,
        actual_dof_force=genesis_sample.actual_dof_force,
        mass_matrix=genesis_sample.mass_matrix,
        reference_gravity_tau=reference_gravity_tau,
        tau_real=tau_real,
        tau_real_median=tau_real_median,
        tau_real_std=tau_real_std,
        tau_real_min=tau_real_min,
        tau_real_max=tau_real_max,
        tau_direct=tau_direct,
        abs_err=abs_err,
        rel_err=rel_err,
        signed_err=signed_err,
        l2_err=l2_err,
        n_real_samples=n_real_samples,
        skip_reason=skip_reason,
        notes=notes or [],
    )


def torque_result_from_sample(sample: DynamicsSample) -> TorqueCompareResult:
    passed = sample.status == ValidationStatus.PASS
    n = int(np.asarray(sample.q).reshape(-1).size)
    tau_g = sample.pd_hold_tau if sample.pd_hold_tau is not None else array_or_nan(n)
    tau_r = sample.tau_real if sample.tau_real is not None else array_or_nan(n)
    abs_err = sample.abs_err if sample.abs_err is not None else array_or_nan(n)
    rel_err = sample.rel_err if sample.rel_err is not None else array_or_nan(n)
    return TorqueCompareResult(
        name=sample.pose,
        q=sample.q,
        ee_z_mm=sample.ee_z_mm,
        tau_genesis=tau_g,
        tau_real=tau_r,
        abs_err=abs_err,
        rel_err=rel_err,
        l2_err=float(sample.l2_err) if sample.l2_err is not None else float("nan"),
        settled=sample.settled,
        passed=passed,
        skip_reason=sample.skip_reason,
        status=sample.status,
    )


def write_jsonl_report(
    results: Sequence[DynamicsSample],
    path: str | Path,
    *,
    run_config: DynamicsRunConfig,
    urdf_issues: Sequence[UrdfDynamicsIssue] | None = None,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "run_config", "data": _jsonable(run_config)}, sort_keys=True) + "\n")
        for issue in urdf_issues or ():
            f.write(json.dumps({"type": "urdf_issue", "data": _jsonable(issue)}, sort_keys=True) + "\n")
        for result in results:
            f.write(json.dumps({"type": "sample", "data": _jsonable(result)}, sort_keys=True) + "\n")


def write_csv_report(results: Sequence[DynamicsSample | TorqueCompareResult], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    dof = 0
    for item in results:
        q = item.q if isinstance(item, DynamicsSample) else item.q
        dof = max(dof, int(np.asarray(q).reshape(-1).size))
    if dof == 0:
        dof = len(JOINT_NAMES)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["pose", "ee_z_mm", "settled", "status", "passed", "skip_reason", "l2_err", "n_real_samples"]
        for i in range(1, dof + 1):
            header.extend(
                [
                    f"q{i}",
                    f"pd_hold_tau{i}",
                    f"tau_g{i}",  # legacy alias
                    f"actual_dof_force{i}",
                    f"tau_real{i}",
                    f"tau_real_std{i}",
                    f"abs_err{i}",
                    f"rel_err{i}",
                ]
            )
        writer.writerow(header)
        for result in results:
            if isinstance(result, TorqueCompareResult):
                status = result.status.value if isinstance(result.status, ValidationStatus) else str(result.status)
                row = [
                    result.name,
                    f"{result.ee_z_mm:.3f}",
                    int(result.settled),
                    status,
                    int(result.passed),
                    result.skip_reason,
                    f"{result.l2_err:.6f}" if np.isfinite(result.l2_err) else "",
                    0,
                ]
                for j in range(dof):
                    tau_g = result.tau_genesis[j] if j < len(result.tau_genesis) else np.nan
                    tau_r = result.tau_real[j] if j < len(result.tau_real) else np.nan
                    abs_err = result.abs_err[j] if j < len(result.abs_err) else np.nan
                    rel_err = result.rel_err[j] if j < len(result.rel_err) else np.nan
                    q_val = result.q[j] if j < len(result.q) else np.nan
                    row.extend(
                        [
                            f"{q_val:.6f}" if np.isfinite(q_val) else "",
                            f"{tau_g:.6f}",
                            f"{tau_g:.6f}",
                            "",
                            f"{tau_r:.6f}" if np.isfinite(tau_r) else "",
                            "",
                            f"{abs_err:.6f}" if np.isfinite(abs_err) else "",
                            f"{rel_err:.6f}" if np.isfinite(rel_err) else "",
                        ]
                    )
                writer.writerow(row)
                continue

            row = [
                result.pose,
                f"{result.ee_z_mm:.3f}",
                int(result.settled),
                result.status.value,
                int(result.status == ValidationStatus.PASS),
                result.skip_reason,
                f"{result.l2_err:.6f}" if result.l2_err is not None and np.isfinite(result.l2_err) else "",
                result.n_real_samples,
            ]
            for j in range(dof):
                q_val = result.q[j] if j < len(result.q) else np.nan
                pd_tau = result.pd_hold_tau[j] if result.pd_hold_tau is not None and j < len(result.pd_hold_tau) else np.nan
                dof_force = result.actual_dof_force[j] if result.actual_dof_force is not None and j < len(result.actual_dof_force) else np.nan
                tau_real = result.tau_real[j] if result.tau_real is not None and j < len(result.tau_real) else np.nan
                tau_std = result.tau_real_std[j] if result.tau_real_std is not None and j < len(result.tau_real_std) else np.nan
                abs_err = result.abs_err[j] if result.abs_err is not None and j < len(result.abs_err) else np.nan
                rel_err = result.rel_err[j] if result.rel_err is not None and j < len(result.rel_err) else np.nan
                row.extend(
                    [
                        f"{q_val:.6f}" if np.isfinite(q_val) else "",
                        f"{pd_tau:.6f}" if np.isfinite(pd_tau) else "",
                        f"{pd_tau:.6f}" if np.isfinite(pd_tau) else "",
                        f"{dof_force:.6f}" if np.isfinite(dof_force) else "",
                        f"{tau_real:.6f}" if np.isfinite(tau_real) else "",
                        f"{tau_std:.6f}" if np.isfinite(tau_std) else "",
                        f"{abs_err:.6f}" if np.isfinite(abs_err) else "",
                        f"{rel_err:.6f}" if np.isfinite(rel_err) else "",
                    ]
                )
            writer.writerow(row)


def read_report_records(path: str | Path) -> list[dict[str, Any]]:
    """Read new JSONL reports or legacy CSV reports into comparable records."""
    p = Path(path)
    if p.suffix == ".jsonl":
        records = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if item.get("type") == "sample":
                    records.append(item["data"])
        return records

    with p.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    records = []
    for row in rows:
        rec: dict[str, Any] = {
            "pose": row.get("pose"),
            "status": row.get("status") or ("PASS" if row.get("passed") == "1" else "FAIL_MODEL"),
            "l2_err": float(row["l2_err"]) if row.get("l2_err") else None,
            "abs_err": [],
            "signed_err": [],
        }
        indices = sorted(
            int(key.replace("q", ""))
            for key in row
            if key.startswith("q") and key[1:].isdigit()
        )
        for i in indices:
            abs_key = f"abs_err{i}"
            tau_g_key = f"pd_hold_tau{i}" if row.get(f"pd_hold_tau{i}") not in (None, "") else f"tau_g{i}"
            tau_r_key = f"tau_real{i}"
            rec["abs_err"].append(float(row[abs_key]) if row.get(abs_key) else None)
            if row.get(tau_g_key) and row.get(tau_r_key):
                rec["signed_err"].append(float(row[tau_g_key]) - float(row[tau_r_key]))
            else:
                rec["signed_err"].append(None)
        records.append(rec)
    return records


def compare_report_records(old_records: Sequence[dict[str, Any]], new_records: Sequence[dict[str, Any]]) -> list[dict[str, float]]:
    """Compare per-joint residual distributions between two reports."""
    out: list[dict[str, float]] = []
    dof = 0
    for record in (*old_records, *new_records):
        signed = record.get("signed_err") or []
        dof = max(dof, len(signed))
    for j in range(dof):
        old_vals = [
            float(r["signed_err"][j])
            for r in old_records
            if r.get("signed_err") and r["signed_err"][j] is not None
        ]
        new_vals = [
            float(r["signed_err"][j])
            for r in new_records
            if r.get("signed_err") and r["signed_err"][j] is not None
        ]
        old_arr = np.asarray(old_vals, dtype=np.float64)
        new_arr = np.asarray(new_vals, dtype=np.float64)
        old_rmse = float(np.sqrt(np.mean(old_arr**2))) if old_arr.size else float("nan")
        new_rmse = float(np.sqrt(np.mean(new_arr**2))) if new_arr.size else float("nan")
        old_bias = float(np.mean(old_arr)) if old_arr.size else float("nan")
        new_bias = float(np.mean(new_arr)) if new_arr.size else float("nan")
        out.append(
            {
                "joint": float(j + 1),
                "old_bias": old_bias,
                "new_bias": new_bias,
                "bias_delta": new_bias - old_bias if np.isfinite(old_bias) and np.isfinite(new_bias) else float("nan"),
                "old_rmse": old_rmse,
                "new_rmse": new_rmse,
                "rmse_delta": new_rmse - old_rmse if np.isfinite(old_rmse) and np.isfinite(new_rmse) else float("nan"),
            }
        )
    return out


def make_run_config(
    *,
    robot_key: str,
    urdf_path: str,
    kinematics_yaml_path: str | None = None,
    sim_dt: float = SIM_DT,
    sim_substeps: int = SIM_SUBSTEPS,
    mode: str,
    session: Any | None = None,
) -> DynamicsRunConfig:
    genesis_version = None
    genesis_backend = None
    integrator = None
    try:
        import genesis as gs

        genesis_version = getattr(gs, "__version__", None)
        backend_value = getattr(gs, "backend", None)
        if backend_value == getattr(gs, "gpu", object()):
            genesis_backend = "gpu"
        elif backend_value == getattr(gs, "cpu", object()):
            genesis_backend = "cpu"
        else:
            genesis_backend = str(backend_value)
    except Exception:
        pass

    firmware = robot_sn = None
    tcp_load = gravity_direction = None
    if session is not None:
        arm = session.arm
        firmware = str(getattr(arm, "version", "")) or None
        robot_sn = str(getattr(arm, "sn", "")) or None
        try:
            tcp_load = list(getattr(arm, "tcp_load"))
        except Exception:
            tcp_load = None
        try:
            gravity_direction = list(getattr(arm, "gravity_direction"))
        except Exception:
            gravity_direction = None

    return DynamicsRunConfig(
        robot_key=robot_key,
        urdf_path=str(urdf_path),
        urdf_sha256=sha256_file(urdf_path),
        kinematics_yaml_path=str(kinematics_yaml_path) if kinematics_yaml_path else None,
        kinematics_yaml_sha256=sha256_file(kinematics_yaml_path),
        genesis_version=genesis_version,
        genesis_backend=genesis_backend,
        sim_dt=sim_dt,
        sim_substeps=sim_substeps,
        integrator=integrator,
        sdk_version=package_version("xarm-python-sdk"),
        firmware=firmware,
        robot_sn=robot_sn,
        tcp_load=tcp_load,
        gravity_direction=gravity_direction,
        git_sha=current_git_sha(Path(__file__).resolve().parents[1]),
        mode=mode,
    )


def print_safe_pose_table(safe_poses: Sequence[SafePose], rejected: Sequence[tuple[str, float]]) -> None:
    print(f"\n{'Name':<20} {'EE z (mm)':>12}  Status")
    print("-" * 48)
    for pose in safe_poses:
        print(f"{pose.name:<20} {pose.ee_z_mm:>12.2f}  SAFE")
    for name, ee_z in rejected:
        z_str = f"{ee_z:>12.2f}" if np.isfinite(ee_z) else f"{'n/a':>12}"
        print(f"{name:<20} {z_str}  REJECTED (z < z_min)")


def print_compare_table(results: Sequence[DynamicsSample | TorqueCompareResult]) -> None:
    print(f"\n{'Pose':<20} {'EE z':>8} {'L2 err':>8} {'Settled':>8} {'Sat':>5}  Status")
    print("-" * 72)
    for r in results:
        if isinstance(r, TorqueCompareResult):
            status = r.status.value if isinstance(r.status, ValidationStatus) else str(r.status)
            l2 = f"{r.l2_err:8.3f}" if np.isfinite(r.l2_err) else f"{'n/a':>8}"
            print(f"{r.name:<20} {r.ee_z_mm:8.1f} {l2} {'yes' if r.settled else 'no':>8} {'n/a':>5}  [{status}]")
            continue
        l2 = f"{r.l2_err:8.3f}" if r.l2_err is not None and np.isfinite(r.l2_err) else f"{'n/a':>8}"
        print(
            f"{r.pose:<20} {r.ee_z_mm:8.1f} {l2} "
            f"{'yes' if r.settled else 'no':>8} {'yes' if r.saturated else 'no':>5}  [{r.status.value}]"
        )


def _select_poses(
    configs: Sequence[tuple[str, np.ndarray]],
    names_csv: str | None,
) -> tuple[list[tuple[str, np.ndarray]], set[str]]:
    if not names_csv:
        return list(configs), set()
    wanted = {p.strip() for p in names_csv.split(",") if p.strip()}
    selected = [(name, q) for name, q in configs if name in wanted]
    missing = wanted - {name for name, _ in selected}
    return selected, missing


def _sdk_path_z_reasons(session, start_q: Sequence[float], target_q: Sequence[float], z_min_mm: float, steps: int = 10) -> list[str]:
    reasons: list[str] = []
    start = np.asarray(start_q, dtype=np.float64)
    target = np.asarray(target_q, dtype=np.float64)
    for i in range(steps + 1):
        alpha = i / steps
        q = (1.0 - alpha) * start + alpha * target
        code, sdk_pose = session.arm.get_forward_kinematics(
            angles=q.tolist(),
            input_is_radian=True,
            return_is_radian=True,
        )
        if code != 0:
            reasons.append(f"SDK FK failed at path step {i}: code={code}")
            break
        z_mm = float(sdk_pose[2])
        if z_mm < z_min_mm:
            reasons.append(f"SDK path step {i}: EE z {z_mm:.2f} mm < z_min {z_min_mm:.2f} mm")
            break
    return reasons


def _hardware_path_reasons_by_waypoint(
    *,
    session,
    robot,
    scene,
    ee_link,
    dof_idx,
    start_q: Sequence[float],
    target_q: Sequence[float],
    joint_lower: Sequence[float],
    joint_upper: Sequence[float],
    z_min_mm: float,
    move_strategy: str,
) -> list[str]:
    from ufactory.real_robot_session import build_motion_waypoints

    waypoints = build_motion_waypoints(start_q, target_q, strategy=move_strategy)
    reasons: list[str] = []
    segment_start = np.asarray(start_q, dtype=np.float64)
    for i, waypoint in enumerate(waypoints, start=1):
        segment_reasons: list[str] = []
        segment_reasons.extend(check_joint_limit_path(segment_start, waypoint, joint_lower, joint_upper))
        segment_reasons.extend(
            check_genesis_path_z(
                robot,
                scene,
                ee_link,
                dof_idx,
                segment_start,
                waypoint,
                z_min_mm=z_min_mm,
            )
        )
        segment_reasons.extend(_sdk_path_z_reasons(session, segment_start, waypoint, z_min_mm))
        if segment_reasons:
            reasons.extend(
                f"waypoint {i}/{len(waypoints)} [{move_strategy}]: {reason}"
                for reason in segment_reasons
            )
            break
        segment_start = waypoint
    return reasons


def _run_genesis_samples(
    scene,
    robot,
    ee_link,
    dof_idx,
    safe_poses: Sequence[SafePose],
    reference: PinocchioReference | None,
    runtime_profile: RobotRuntimeProfile | None = None,
) -> dict[str, GenesisDynamicsSample]:
    runtime = runtime_profile or _XARM6_RUNTIME
    out: dict[str, GenesisDynamicsSample] = {}
    for pose in safe_poses:
        sample = genesis_pd_hold_torque_at_q(robot, scene, dof_idx, pose.q, runtime_profile=runtime)
        out[pose.name] = sample
        status = "settled" if sample.settled else "NOT settled"
        sat = " SATURATED" if sample.saturated else ""
        print(f"  [{pose.name}] pd_hold_tau={format_torque_row(sample.pd_hold_tau)}  ({status}{sat})")
        if reference is not None:
            try:
                g_ref = reference.gravity(pose.q)
                print(f"    pinocchio_G={format_torque_row(g_ref)}")
            except Exception as exc:
                print(f"    [WARN] Pinocchio gravity failed: {exc}")
    return out


def cli_hardware_check(argv: Sequence[str] | None = None) -> int:
    from ufactory.kinematics import (
        get_robot_sn,
        log_kinematics_sn_status,
        prepare_robot_model_for_verification,
        validate_kinematics_calibration_request,
    )
    from ufactory.real_robot_session import (
        MOVE_STRATEGIES,
        MOVE_STRATEGY_DIRECT,
        RealRobotSession,
        RobotMotionError,
    )

    parser = argparse.ArgumentParser(description="UFACTORY Genesis vs real static torque validation")
    parser.add_argument("--robot", default="xarm6", choices=robot_runtime_cli_choices())
    parser.add_argument("--ip", type=str, default=None, help="xArm IP (required unless --dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="Genesis torques + safe poses only")
    parser.add_argument("--hold-current-only", action="store_true", help="Read real torque at current pose")
    parser.add_argument("--kinematics-suffix", type=str, default=None)
    parser.add_argument("--kinematics-yaml", type=str, default=None)
    parser.add_argument("--kinematics-yaml-dir", type=str, default=None)
    parser.add_argument("--robot-model", type=str, default=None)
    parser.add_argument("--calibrated-output-dir", type=str, default=None)
    parser.add_argument("--z-min-mm", type=float, default=None, help="Minimum EE z (mm); default from robot profile")
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Joint speed (rad/s) for real moves; default from robot profile",
    )
    parser.add_argument(
        "--move-strategy",
        choices=MOVE_STRATEGIES,
        default=MOVE_STRATEGY_DIRECT,
        help="Real robot joint move strategy; default direct",
    )
    parser.add_argument("--sample-duration", type=float, default=3.0, help="Seconds of hold samples per pose")
    parser.add_argument("--sample-poll", type=float, default=0.1, help="Seconds between hold samples")
    parser.add_argument("--repeats", type=int, default=1, help="Hardware repeats per pose")
    parser.add_argument("--poses", type=str, default=None, help="Comma-separated pose names to run")
    parser.add_argument("--include-stress", action="store_true", help="Include stress/saturation pose set")
    parser.add_argument("--require-reference", action="store_true", help="Fail if Pinocchio is unavailable")
    parser.add_argument("--report", type=str, default=None, help="CSV report output path")
    parser.add_argument("--jsonl-report", type=str, default=None, help="JSONL report output path")
    parser.add_argument("-v", "--vis", action="store_true", help="Genesis viewer")
    args = parser.parse_args(argv)
    runtime = get_robot_runtime_profile(args.robot)
    if args.z_min_mm is None:
        args.z_min_mm = runtime.dynamics.default_z_min_mm
    if args.speed is None:
        args.speed = runtime.dynamics.default_move_speed_rad_s

    if not args.dry_run and not args.hold_current_only and not args.ip:
        parser.error("--ip is required unless --dry-run or --hold-current-only")
    if not args.dry_run and not args.hold_current_only and not runtime.dynamics.supports_hardware_validation:
        parser.error(f"{runtime.model.key} has no hardware dynamics validation pose profile yet")

    if args.z_min_mm < 50.0:
        print(f"[WARN] z_min={args.z_min_mm:.1f} mm is a low-margin hardware mode")

    urdf_path_str, kinematics_yaml_path = prepare_robot_model_for_verification(
        args.robot_model,
        args.kinematics_yaml,
        args.kinematics_suffix,
        args.kinematics_yaml_dir,
        default_base_urdf=robot_urdf(runtime.model.key),
        robot_name=runtime.model.robot_name,
        joint_count=runtime.model.dof,
        output_dir=args.calibrated_output_dir,
    )

    print("=" * 78)
    print(f"{runtime.model.key} Static Dynamics Validation: Genesis PD hold vs Real Robot")
    print("=" * 78)
    print(f"URDF : {urdf_path_str}")
    if kinematics_yaml_path:
        print(f"Calib: {kinematics_yaml_path}")
    print(
        f"z_min: {args.z_min_mm:.1f} mm  speed: {args.speed:.4f} rad/s  sim_dt: {SIM_DT}  "
        f"substeps: {SIM_SUBSTEPS}  move_strategy: {args.move_strategy}"
    )

    urdf_issues = validate_urdf_dynamics(urdf_path_str)
    errors = [i for i in urdf_issues if i.severity == "ERROR"]
    if urdf_issues:
        print("\n--- URDF dynamics static checks ---")
        for issue in urdf_issues:
            print(f"  [{issue.severity}] {issue.item}: {issue.message}")
    if errors:
        print("[FAIL] URDF dynamics static checks contain errors")
        return 1

    configs = dynamics_default_configs(runtime.model.key, include_stress=args.include_stress)
    configs, missing = _select_poses(configs, args.poses)
    if missing:
        print(f"[WARN] Requested poses not found: {sorted(missing)}")
    if not configs:
        print("[FAIL] No poses selected")
        return 1

    print("\n--- Building Genesis scene ---")
    scene, robot, ee_link, dof_idx = build_genesis_scene(
        urdf_path_str,
        runtime_profile=runtime,
        show_viewer=args.vis,
    )
    ee_z_table = compute_ee_z_table_from_sim(robot, scene, ee_link, dof_idx, configs)
    safe_poses, rejected = filter_safe_configs(configs, ee_z_table, args.z_min_mm)

    print("\n--- Safe pose filter (Genesis link6 z) ---")
    print_safe_pose_table(safe_poses, rejected)
    if not safe_poses:
        print("[FAIL] No safe poses after z filter")
        return 1

    reference = load_reference_backend(urdf_path_str, required=args.require_reference)
    if reference is None:
        print("[WARN] Pinocchio reference unavailable; continuing without independent G(q)/M(q)")

    print("\n--- Genesis PD hold torques ---")
    genesis_data = _run_genesis_samples(scene, robot, ee_link, dof_idx, safe_poses, reference, runtime)

    session: RealRobotSession | None = None
    results: list[DynamicsSample] = []
    run_config = make_run_config(
        robot_key=runtime.model.key,
        urdf_path=urdf_path_str,
        kinematics_yaml_path=kinematics_yaml_path,
        sim_dt=SIM_DT,
        sim_substeps=SIM_SUBSTEPS,
        mode="dry-run" if args.dry_run else "hardware",
    )

    try:
        if args.hold_current_only:
            session = RealRobotSession(args.ip, dof=runtime.model.dof, home_qpos=runtime.arm.home_qpos)
            session.configure_for_dynamics()
            session.print_config()
            q, qvel, tau = session.get_joint_states()
            tau_direct = session.get_joints_torque()
            print(f"\nCurrent q          : {format_torque_row(q)}")
            print(f"Current dq         : {format_torque_row(qvel)}")
            print(f"joint_states effort: {format_torque_row(tau)}")
            if tau_direct is not None:
                print(f"get_joints_torque  : {format_torque_row(tau_direct)}")
            return 0

        if args.dry_run:
            for pose in safe_poses:
                ref_g = reference.gravity(pose.q) if reference is not None else None
                results.append(
                    build_dynamics_sample(
                        pose,
                        genesis_data[pose.name],
                        runtime_profile=runtime,
                        reference_gravity_tau=ref_g,
                        skip_reason="dry-run",
                    )
                )
        else:
            session = RealRobotSession(args.ip, dof=runtime.model.dof, home_qpos=runtime.arm.home_qpos)
            session.configure_for_dynamics()
            session.print_config()
            run_config = make_run_config(
                robot_key=runtime.model.key,
                urdf_path=urdf_path_str,
                kinematics_yaml_path=kinematics_yaml_path,
                sim_dt=SIM_DT,
                sim_substeps=SIM_SUBSTEPS,
                mode="hardware",
                session=session,
            )

            sn = get_robot_sn(session.arm)
            run_config.robot_sn = sn or run_config.robot_sn
            validate_kinematics_calibration_request(
                sn,
                runtime.model.robot_name,
                kinematics_yaml=args.kinematics_yaml,
                kinematics_suffix=args.kinematics_suffix,
            )
            log_kinematics_sn_status(
                sn,
                runtime.model.robot_name,
                kinematics_yaml=kinematics_yaml_path,
                kinematics_suffix=args.kinematics_suffix,
            )

            joint_lower, joint_upper = parse_joint_limits(urdf_path_str, runtime.arm.joint_names)
            print("\n--- Hardware sampling ---")
            hardware_abort = False
            for pose in safe_poses:
                if hardware_abort:
                    break
                for repeat_i in range(args.repeats):
                    q_now, _, _ = session.get_joint_states()
                    unsafe = _hardware_path_reasons_by_waypoint(
                        session=session,
                        robot=robot,
                        scene=scene,
                        ee_link=ee_link,
                        dof_idx=dof_idx,
                        start_q=q_now,
                        target_q=pose.q,
                        joint_lower=joint_lower,
                        joint_upper=joint_upper,
                        z_min_mm=args.z_min_mm,
                        move_strategy=args.move_strategy,
                    )
                    genesis_sample = genesis_data[pose.name]
                    if unsafe:
                        print(f"  [{pose.name}] repeat {repeat_i + 1}: UNSAFE; " + "; ".join(unsafe[:2]))
                        results.append(
                            DynamicsSample(
                                pose=pose.name,
                                q=pose.q,
                                ee_z_mm=pose.ee_z_mm,
                                status=ValidationStatus.UNSAFE,
                                settled=False,
                                saturated=genesis_sample.saturated,
                                q_actual=genesis_sample.q_actual,
                                qvel=genesis_sample.qvel,
                                pd_hold_tau=genesis_sample.pd_hold_tau,
                                actual_dof_force=genesis_sample.actual_dof_force,
                                mass_matrix=genesis_sample.mass_matrix,
                                skip_reason="unsafe",
                                notes=unsafe,
                            )
                        )
                        continue
                    if not genesis_sample.settled or genesis_sample.saturated:
                        status = ValidationStatus.NOT_SETTLED if not genesis_sample.settled else ValidationStatus.SATURATED
                        print(f"  [{pose.name}] repeat {repeat_i + 1}: {status.value}; not moving hardware")
                        results.append(
                            DynamicsSample(
                                pose=pose.name,
                                q=pose.q,
                                ee_z_mm=pose.ee_z_mm,
                                status=status,
                                settled=genesis_sample.settled,
                                saturated=genesis_sample.saturated,
                                q_actual=genesis_sample.q_actual,
                                qvel=genesis_sample.qvel,
                                pd_hold_tau=genesis_sample.pd_hold_tau,
                                actual_dof_force=genesis_sample.actual_dof_force,
                                mass_matrix=genesis_sample.mass_matrix,
                                skip_reason=status.value.lower(),
                            )
                        )
                        continue

                    print(f"  Moving to [{pose.name}] repeat {repeat_i + 1}/{args.repeats} ...")
                    try:
                        real = session.sample_at_hold(
                            pose.q,
                            speed_rad_s=args.speed,
                            move_strategy=args.move_strategy,
                            sample_duration_s=args.sample_duration,
                            sample_poll_s=args.sample_poll,
                        )
                    except RobotMotionError as exc:
                        print(f"  [{pose.name}] repeat {repeat_i + 1}: UNSAFE; motion failed: {exc}")
                        results.append(
                            DynamicsSample(
                                pose=pose.name,
                                q=pose.q,
                                ee_z_mm=pose.ee_z_mm,
                                status=ValidationStatus.UNSAFE,
                                settled=False,
                                saturated=genesis_sample.saturated,
                                q_actual=genesis_sample.q_actual,
                                qvel=genesis_sample.qvel,
                                pd_hold_tau=genesis_sample.pd_hold_tau,
                                actual_dof_force=genesis_sample.actual_dof_force,
                                mass_matrix=genesis_sample.mass_matrix,
                                skip_reason="motion_error",
                                notes=[str(exc)],
                            )
                        )
                        hardware_abort = True
                        break
                    ref_g = reference.gravity(pose.q) if reference is not None else None
                    sample = build_dynamics_sample(
                        pose,
                        genesis_sample,
                        runtime_profile=runtime,
                        tau_real=real.tau,
                        tau_real_median=real.tau_median,
                        tau_real_std=real.tau_std,
                        tau_real_min=real.tau_min,
                        tau_real_max=real.tau_max,
                        tau_direct=real.tau_direct,
                        n_real_samples=real.n_samples,
                        reference_gravity_tau=ref_g,
                        skip_reason="" if real.settled else "not settled",
                    )
                    if not real.settled and sample.status == ValidationStatus.INSUFFICIENT_DATA:
                        sample.status = ValidationStatus.NOT_SETTLED
                    print(f"  [{pose.name}] tau_real_mean={format_torque_row(real.tau)}  [{sample.status.value}]")
                    results.append(sample)

    finally:
        if session is not None and not args.hold_current_only:
            try:
                print("\n--- Returning to home ---")
                session.return_home(
                    speed_rad_s=args.speed,
                    move_strategy=args.move_strategy,
                )
            except Exception as exc:
                print(f"[WARN] return_home failed: {exc}")
            session.disconnect()

    print_compare_table(results)

    stamp = now_stamp()
    csv_path = Path(args.report) if args.report else Path("reports") / f"dynamics_verify_real_{stamp}.csv"
    jsonl_path = Path(args.jsonl_report) if args.jsonl_report else csv_path.with_suffix(".jsonl")
    write_csv_report(results, csv_path)
    write_jsonl_report(results, jsonl_path, run_config=run_config, urdf_issues=urdf_issues)
    print(f"\nCSV report  : {csv_path}")
    print(f"JSONL report: {jsonl_path}")

    if args.dry_run:
        print("\n[OK] Dry-run complete (Genesis quantities only)")
        return 0

    eval_results = [r for r in results if r.status not in {ValidationStatus.NOT_SETTLED, ValidationStatus.SATURATED, ValidationStatus.UNSAFE}]
    n_pass = sum(1 for r in eval_results if r.status == ValidationStatus.PASS)
    n_total = len(eval_results)
    n_unsafe = sum(1 for r in results if r.status == ValidationStatus.UNSAFE)
    all_passed = n_total > 0 and n_pass == n_total and n_unsafe == 0
    print("\n" + "=" * 78)
    print(f"SUMMARY: {n_pass}/{n_total} evaluated poses passed")
    if n_unsafe:
        print(f"UNSAFE: {n_unsafe} poses skipped/aborted")
    print(f"Overall: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 78)
    return 0 if all_passed else 1


def cli_sim_check(argv: Sequence[str] | None = None) -> int:
    from ufactory.kinematics import prepare_robot_model_for_verification

    parser = argparse.ArgumentParser(description="UFACTORY Genesis dynamics simulation regression")
    parser.add_argument("--robot", default="xarm6", choices=robot_runtime_cli_choices())
    parser.add_argument("--robot-model", type=str, default=None)
    parser.add_argument("--kinematics-suffix", type=str, default=None)
    parser.add_argument("--kinematics-yaml", type=str, default=None)
    parser.add_argument("--kinematics-yaml-dir", type=str, default=None)
    parser.add_argument("--calibrated-output-dir", type=str, default=None)
    parser.add_argument("--z-min-mm", type=float, default=None)
    parser.add_argument("--random-count", type=int, default=100)
    parser.add_argument("--report", type=str, default=None)
    parser.add_argument("--jsonl-report", type=str, default=None)
    args = parser.parse_args(argv)
    runtime = get_robot_runtime_profile(args.robot)
    if args.z_min_mm is None:
        args.z_min_mm = runtime.dynamics.default_z_min_mm

    urdf_path_str, kinematics_yaml_path = prepare_robot_model_for_verification(
        args.robot_model,
        args.kinematics_yaml,
        args.kinematics_suffix,
        args.kinematics_yaml_dir,
        default_base_urdf=robot_urdf(runtime.model.key),
        robot_name=runtime.model.robot_name,
        joint_count=runtime.model.dof,
        output_dir=args.calibrated_output_dir,
    )
    issues = validate_urdf_dynamics(urdf_path_str)
    errors = [i for i in issues if i.severity == "ERROR"]
    if errors:
        for issue in errors:
            print(f"[ERROR] {issue.item}: {issue.message}")
        return 1

    lower, upper = parse_joint_limits(urdf_path_str, runtime.arm.joint_names)
    lower = np.where(np.isfinite(lower), lower, -1.0)
    upper = np.where(np.isfinite(upper), upper, 1.0)
    rng = np.random.default_rng(42)
    random_configs = [
        (f"random_{i:03d}", rng.uniform(lower + 0.05, upper - 0.05).astype(np.float64))
        for i in range(max(0, args.random_count))
    ]
    configs = merge_test_configs(dynamics_default_configs(runtime.model.key), random_configs)

    scene, robot, ee_link, dof_idx = build_genesis_scene(urdf_path_str, runtime_profile=runtime)
    ee_z_table = compute_ee_z_table_from_sim(robot, scene, ee_link, dof_idx, configs)
    safe_poses, rejected = filter_safe_configs(configs, ee_z_table, args.z_min_mm)
    print_safe_pose_table(safe_poses, rejected[:10])
    reference = load_reference_backend(urdf_path_str, required=False)

    results: list[DynamicsSample] = []
    for pose in safe_poses:
        gs_sample = genesis_pd_hold_torque_at_q(robot, scene, dof_idx, pose.q, runtime_profile=runtime)
        ref_g = reference.gravity(pose.q) if reference is not None else None
        results.append(
            build_dynamics_sample(
                pose,
                gs_sample,
                runtime_profile=runtime,
                reference_gravity_tau=ref_g,
                skip_reason="sim-only",
            )
        )

    stamp = now_stamp()
    csv_path = Path(args.report) if args.report else Path("reports") / f"dynamics_sim_check_{stamp}.csv"
    jsonl_path = Path(args.jsonl_report) if args.jsonl_report else csv_path.with_suffix(".jsonl")
    run_config = make_run_config(
        robot_key=runtime.model.key,
        urdf_path=urdf_path_str,
        kinematics_yaml_path=kinematics_yaml_path,
        mode="sim",
    )
    write_csv_report(results, csv_path)
    write_jsonl_report(results, jsonl_path, run_config=run_config, urdf_issues=issues)
    n_bad = sum(1 for r in results if r.status in {ValidationStatus.NOT_SETTLED, ValidationStatus.SATURATED})
    print(f"Simulation samples: {len(results)}, unstable/saturated: {n_bad}")
    print(f"CSV report  : {csv_path}")
    print(f"JSONL report: {jsonl_path}")
    return 0 if n_bad == 0 else 1


@dataclass
class SimCollisionResult:
    pose_name: str
    passed: bool
    error_code: int | None = None
    waypoint_index: int | None = None
    message: str = ""


def run_sim_collision_chain(
    session,
    poses: Sequence[tuple[str, np.ndarray]],
    *,
    speed_rad_s: float,
    move_strategy: str,
) -> list[SimCollisionResult]:
    """Move through poses in order without returning home between them."""
    from ufactory.real_robot_session import RobotMotionError

    results: list[SimCollisionResult] = []
    for name, q in poses:
        print(f"\n--- [{name}] ---")
        try:
            session.move_to(
                q,
                speed_rad_s=speed_rad_s,
                wait=True,
                move_strategy=move_strategy,
            )
            error_code = int(session.arm.error_code)
            if error_code == 22:
                results.append(
                    SimCollisionResult(
                        pose_name=name,
                        passed=False,
                        error_code=error_code,
                        message="self-collision (error_code=22) after move",
                    )
                )
                print(f"  FAIL: self-collision error_code=22")
                session.recover_after_motion_error()
                continue
            results.append(SimCollisionResult(pose_name=name, passed=True))
            print("  PASS")
        except RobotMotionError as exc:
            error_code = int(exc.code)
            results.append(
                SimCollisionResult(
                    pose_name=name,
                    passed=False,
                    error_code=error_code,
                    waypoint_index=exc.waypoint_index,
                    message=str(exc),
                )
            )
            print(f"  FAIL: {exc}")
            session.recover_after_motion_error()
    return results


def write_sim_collision_report(results: Sequence[SimCollisionResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["pose", "passed", "error_code", "waypoint_index", "message"],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "pose": row.pose_name,
                    "passed": row.passed,
                    "error_code": row.error_code if row.error_code is not None else "",
                    "waypoint_index": row.waypoint_index if row.waypoint_index is not None else "",
                    "message": row.message,
                }
            )


def cli_sim_collision_check(argv: Sequence[str] | None = None) -> int:
    from ufactory.real_robot_session import MOVE_STRATEGIES, MOVE_STRATEGY_DIRECT, RealRobotSession

    parser = argparse.ArgumentParser(
        description="xArm simulation-mode chained self-collision check for dynamics poses",
    )
    parser.add_argument("--ip", type=str, required=True, help="xArm IP (simulation mode)")
    parser.add_argument("--robot", default="xarm6", choices=robot_runtime_cli_choices())
    parser.add_argument("--poses", type=str, default=None, help="Comma-separated pose names to run")
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Joint speed (rad/s); default from robot profile",
    )
    parser.add_argument(
        "--move-strategy",
        choices=MOVE_STRATEGIES,
        default=MOVE_STRATEGY_DIRECT,
        help="Joint move strategy; default direct",
    )
    parser.add_argument("--report", type=str, default=None, help="CSV report output path")
    args = parser.parse_args(argv)

    runtime = get_robot_runtime_profile(args.robot)
    if args.speed is None:
        args.speed = runtime.dynamics.default_move_speed_rad_s
    if not runtime.dynamics.supports_hardware_validation:
        parser.error(f"{runtime.model.key} has no hardware dynamics validation pose profile yet")

    configs = dynamics_default_configs(runtime.model.key)
    configs, missing = _select_poses(configs, args.poses)
    if missing:
        print(f"[WARN] Requested poses not found: {sorted(missing)}")
    if not configs:
        print("[FAIL] No poses selected")
        return 1

    print("=" * 78)
    print(f"{runtime.model.key} Simulation Self-Collision Chain Check")
    print("=" * 78)
    print(f"IP             : {args.ip}")
    print(f"poses          : {len(configs)}")
    print(f"speed          : {args.speed:.4f} rad/s")
    print(f"move_strategy  : {args.move_strategy}")

    session = RealRobotSession(args.ip, dof=runtime.model.dof, home_qpos=runtime.arm.home_qpos)
    try:
        session.configure_for_simulation_collision_check()
        session.print_config()

        print("\n--- Moving to home ---")
        session.move_to(
            runtime.arm.home_qpos,
            speed_rad_s=args.speed,
            wait=True,
            move_strategy=args.move_strategy,
        )

        results = run_sim_collision_chain(
            session,
            configs,
            speed_rad_s=args.speed,
            move_strategy=args.move_strategy,
        )
    finally:
        try:
            session.arm.set_simulation_robot(on_off=False)
        except Exception as exc:
            print(f"[WARN] set_simulation_robot(False) failed: {exc}")
        session.disconnect()

    n_pass = sum(1 for r in results if r.passed)
    n_fail = len(results) - n_pass
    print("\n" + "=" * 78)
    print(f"SUMMARY: {n_pass}/{len(results)} poses passed, {n_fail} failed")
    for row in results:
        if not row.passed:
            wp = f" waypoint={row.waypoint_index}" if row.waypoint_index is not None else ""
            code = f" error_code={row.error_code}" if row.error_code is not None else ""
            print(f"  FAIL [{row.pose_name}]{wp}{code}: {row.message[:120]}")
    print("=" * 78)

    stamp = now_stamp()
    report_path = Path(args.report) if args.report else Path("reports") / f"dynamics_sim_collision_{stamp}.csv"
    write_sim_collision_report(results, report_path)
    print(f"CSV report: {report_path}")

    return 0 if n_fail == 0 else 1


def cli_report_compare(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two dynamics validation reports")
    parser.add_argument("old_report")
    parser.add_argument("new_report")
    args = parser.parse_args(argv)

    old_records = read_report_records(args.old_report)
    new_records = read_report_records(args.new_report)
    stats = compare_report_records(old_records, new_records)
    print(f"{'Joint':>5} {'old_bias':>10} {'new_bias':>10} {'d_bias':>10} {'old_rmse':>10} {'new_rmse':>10} {'d_rmse':>10}")
    for row in stats:
        print(
            f"J{int(row['joint']):<4} "
            f"{row['old_bias']:10.4f} {row['new_bias']:10.4f} {row['bias_delta']:10.4f} "
            f"{row['old_rmse']:10.4f} {row['new_rmse']:10.4f} {row['rmse_delta']:10.4f}"
        )
    return 0
