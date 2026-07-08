"""Validation helpers for trajectory plans.

These checks cover the light, robot-agnostic contract shared by planners and
executors: joint-vector dimensionality, Cartesian target shape, optional z-min,
and gripper segment completeness. Hardware-specific streaming limits remain in
``real_executor.py``.
"""

from __future__ import annotations

import numpy as np

from ufactory.robots.runtime import RobotRuntimeProfile, get_robot_runtime_profile
from ufactory.trajectory.segments import Program, Segment


def runtime_for_robot(robot_key: str) -> RobotRuntimeProfile:
    """Resolve ``robot_key`` and return its runtime profile."""
    return get_robot_runtime_profile(robot_key)


def validate_rate(rate: float) -> float:
    value = float(rate)
    if value <= 0.0:
        raise ValueError(f"trajectory rate must be positive, got {rate!r}")
    return value


def validate_joint_vector(q, *, dof: int, name: str = "q") -> np.ndarray:
    """Return ``q`` as a 1D float array and require exactly ``dof`` joints."""
    arr = np.asarray(q, dtype=np.float64).reshape(-1)
    if arr.size != int(dof):
        raise ValueError(f"{name} expected {int(dof)} joints, got {arr.size}")
    return arr


def validate_cartesian_xyz(xyz, *, name: str = "xyz") -> np.ndarray:
    """Return ``xyz`` as a length-3 float array."""
    arr = np.asarray(xyz, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"{name} expected xyz length 3, got {arr.size}")
    return arr


def validate_segment(
    seg: Segment,
    *,
    runtime: RobotRuntimeProfile | None = None,
    z_min_m: float | None = None,
) -> None:
    """Validate one trajectory segment against the generic trajectory contract."""
    if seg.kind == "movej":
        if runtime is None:
            if seg.q_start is None or seg.q_end is None:
                raise ValueError(f"MoveJ segment {seg.label!r} requires q_start and q_end")
            if np.asarray(seg.q_start).reshape(-1).shape != np.asarray(seg.q_end).reshape(-1).shape:
                raise ValueError(f"MoveJ segment {seg.label!r} q_start/q_end shape mismatch")
            return
        dof = runtime.model.dof
        validate_joint_vector(seg.q_start, dof=dof, name=f"{seg.label or 'movej'}.q_start")
        validate_joint_vector(seg.q_end, dof=dof, name=f"{seg.label or 'movej'}.q_end")
        return

    if seg.kind == "movel":
        p0 = validate_cartesian_xyz(seg.pose_start, name=f"{seg.label or 'movel'}.pose_start")
        p1 = validate_cartesian_xyz(seg.pose_end, name=f"{seg.label or 'movel'}.pose_end")
        if z_min_m is not None:
            z_min = float(z_min_m)
            if min(float(p0[2]), float(p1[2])) < z_min:
                raise ValueError(f"MoveL segment {seg.label!r} has z below {z_min:.4f} m")
        return

    if seg.kind == "gripper":
        if seg.gap_start is None or seg.gap_end is None:
            raise ValueError(f"Gripper segment {seg.label!r} requires gap_start and gap_end")
        if seg.duration < 0.0:
            raise ValueError(f"Gripper segment {seg.label!r} has negative duration")
        return

    raise ValueError(f"unknown segment kind: {seg.kind!r}")


def validate_program(
    program: Program,
    *,
    robot_key: str | None = None,
    z_min_m: float | None = None,
) -> None:
    """Validate every segment in ``program``.

    If ``robot_key`` or ``program.robot_key`` is set, MoveJ segments must match
    that robot's DOF. Cartesian targets are always validated as base-frame xyz.
    """
    validate_rate(program.rate)
    key = robot_key or program.robot_key
    runtime = runtime_for_robot(key) if key else None
    for seg in program.segments:
        validate_segment(seg, runtime=runtime, z_min_m=z_min_m)
