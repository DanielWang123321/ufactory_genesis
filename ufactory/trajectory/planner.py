"""High-level trajectory planners that produce executable ``Program`` objects.

The planner layer is intentionally lightweight: it turns robot-aware waypoints
into time-parameterized MoveJ/MoveL/Gripper segments using the NumPy LSPB
profile in ``profile.py``. Collision-aware planning and heavy reference
toolboxes are optional future/back-end layers, not import-time dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ufactory.robots.runtime import RobotRuntimeProfile, get_robot_runtime_profile
from ufactory.trajectory.segments import (
    JointLimits,
    Program,
    Segment,
    make_gripper,
    make_movej,
    make_movel,
)
from ufactory.trajectory.validation import (
    validate_cartesian_xyz,
    validate_joint_vector,
    validate_program,
    validate_rate,
)


@dataclass(frozen=True)
class TrajectoryPlannerConfig:
    """Robot-aware defaults for waypoint planning.

    Units follow the rest of ``ufactory.trajectory``:
    joint motion uses rad / rad/s / rad/s^2, Cartesian motion uses metres.
    """

    robot_key: str = "xarm6"
    rate: float = 50.0
    speed_rad_s: float = 0.35
    mvacc_rad_s2: float = 2.0
    reach_m: float = 0.4
    z_min_m: float | None = None

    @property
    def runtime(self) -> RobotRuntimeProfile:
        return get_robot_runtime_profile(self.robot_key)

    @property
    def canonical_robot_key(self) -> str:
        return self.runtime.model.key

    @property
    def limits(self) -> JointLimits:
        return JointLimits.from_speed_mvacc(
            self.speed_rad_s,
            self.mvacc_rad_s2,
            reach_m=self.reach_m,
        )

    def validate(self) -> None:
        validate_rate(self.rate)
        if self.speed_rad_s <= 0.0:
            raise ValueError("speed_rad_s must be positive")
        if self.mvacc_rad_s2 <= 0.0:
            raise ValueError("mvacc_rad_s2 must be positive")
        if self.reach_m <= 0.0:
            raise ValueError("reach_m must be positive")
        self.runtime


@dataclass(frozen=True)
class JointWaypoint:
    """A joint-space waypoint in radians."""

    q: Sequence[float]
    label: str = ""


@dataclass(frozen=True)
class CartesianWaypoint:
    """A base-frame Cartesian EE waypoint in metres."""

    xyz: Sequence[float]
    label: str = ""


def _program(config: TrajectoryPlannerConfig, segments: list[Segment], *, kind: str) -> Program:
    metadata: dict[str, object] = {
        "planner": "ufactory.trajectory.planner",
        "kind": kind,
        "robot_key": config.canonical_robot_key,
        "ee_link": config.runtime.arm.ee_link,
    }
    program = Program(
        segments=segments,
        rate=float(config.rate),
        limits=config.limits,
        robot_key=config.canonical_robot_key,
        metadata=metadata,
    )
    validate_program(program, z_min_m=config.z_min_m)
    return program


def plan_joint_waypoints(
    config: TrajectoryPlannerConfig,
    start_q: Sequence[float],
    waypoints: Sequence[JointWaypoint | Sequence[float]],
) -> Program:
    """Plan chained MoveJ segments from ``start_q`` through ``waypoints``."""
    config.validate()
    runtime = config.runtime
    current = validate_joint_vector(start_q, dof=runtime.model.dof, name="start_q")
    segments: list[Segment] = []
    for index, wp in enumerate(waypoints):
        if isinstance(wp, JointWaypoint):
            target = wp.q
            label = wp.label or f"movej-{index}"
        else:
            target = wp
            label = f"movej-{index}"
        q_target = validate_joint_vector(target, dof=runtime.model.dof, name=label)
        segments.append(make_movej(current, q_target, rate=config.rate, limits=config.limits, label=label))
        current = q_target
    return _program(config, segments, kind="joint")


def plan_cartesian_waypoints(
    config: TrajectoryPlannerConfig,
    start_xyz: Sequence[float],
    waypoints: Sequence[CartesianWaypoint | Sequence[float]],
) -> Program:
    """Plan chained MoveL segments through base-frame Cartesian waypoints."""
    config.validate()
    current = validate_cartesian_xyz(start_xyz, name="start_xyz")
    segments: list[Segment] = []
    for index, wp in enumerate(waypoints):
        if isinstance(wp, CartesianWaypoint):
            target = wp.xyz
            label = wp.label or f"movel-{index}"
        else:
            target = wp
            label = f"movel-{index}"
        xyz_target = validate_cartesian_xyz(target, name=label)
        segments.append(make_movel(current, xyz_target, rate=config.rate, limits=config.limits, label=label))
        current = xyz_target
    return _program(config, segments, kind="cartesian")


def plan_mixed_waypoints(
    config: TrajectoryPlannerConfig,
    waypoints: Sequence[JointWaypoint | CartesianWaypoint | Mapping[str, Any]],
    *,
    start_q: Sequence[float] | None = None,
    start_xyz: Sequence[float] | None = None,
) -> Program:
    """Plan mixed MoveJ/MoveL/Gripper waypoints.

    Typed waypoints are chained from ``start_q`` or ``start_xyz``. Dict
    waypoints also support the existing ``build_pickplace_program`` schema:
    ``q_start``/``q_end``, ``pose_start``/``pose_end``, and gripper gap fields.
    """
    config.validate()
    runtime = config.runtime
    current_q = (
        validate_joint_vector(start_q, dof=runtime.model.dof, name="start_q")
        if start_q is not None
        else None
    )
    current_xyz = validate_cartesian_xyz(start_xyz, name="start_xyz") if start_xyz is not None else None
    segments: list[Segment] = []

    for index, wp in enumerate(waypoints):
        if isinstance(wp, JointWaypoint):
            if current_q is None:
                raise ValueError("JointWaypoint requires start_q or a previous joint waypoint")
            target = validate_joint_vector(wp.q, dof=runtime.model.dof, name=wp.label or f"movej-{index}")
            segments.append(
                make_movej(current_q, target, rate=config.rate, limits=config.limits, label=wp.label or f"movej-{index}")
            )
            current_q = target
            continue

        if isinstance(wp, CartesianWaypoint):
            if current_xyz is None:
                raise ValueError("CartesianWaypoint requires start_xyz or a previous Cartesian waypoint")
            target = validate_cartesian_xyz(wp.xyz, name=wp.label or f"movel-{index}")
            segments.append(
                make_movel(current_xyz, target, rate=config.rate, limits=config.limits, label=wp.label or f"movel-{index}")
            )
            current_xyz = target
            continue

        if isinstance(wp, Mapping):
            seg, current_q, current_xyz = _segment_from_mapping(
                config,
                wp,
                index=index,
                current_q=current_q,
                current_xyz=current_xyz,
            )
            segments.append(seg)
            continue

        raise TypeError(f"unsupported waypoint type at index {index}: {type(wp).__name__}")

    return _program(config, segments, kind="mixed")


def _segment_from_mapping(
    config: TrajectoryPlannerConfig,
    wp: Mapping[str, Any],
    *,
    index: int,
    current_q: np.ndarray | None,
    current_xyz: np.ndarray | None,
) -> tuple[Segment, np.ndarray | None, np.ndarray | None]:
    runtime = config.runtime
    wp_type = str(wp["type"]).strip().lower()
    label = str(wp.get("label", f"{wp_type}-{index}"))

    if wp_type == "movej":
        q0_raw = wp.get("q_start", current_q)
        q1_raw = wp.get("q_end", wp.get("q"))
        if q0_raw is None or q1_raw is None:
            raise ValueError(f"MoveJ waypoint {label!r} requires q_start/q_end or chained q")
        q0 = validate_joint_vector(q0_raw, dof=runtime.model.dof, name=f"{label}.q_start")
        q1 = validate_joint_vector(q1_raw, dof=runtime.model.dof, name=f"{label}.q_end")
        return make_movej(q0, q1, rate=config.rate, limits=config.limits, label=label), q1, current_xyz

    if wp_type == "movel":
        p0_raw = wp.get("pose_start", current_xyz)
        p1_raw = wp.get("pose_end", wp.get("xyz"))
        if p0_raw is None or p1_raw is None:
            raise ValueError(f"MoveL waypoint {label!r} requires pose_start/pose_end or chained xyz")
        p0 = validate_cartesian_xyz(p0_raw, name=f"{label}.pose_start")
        p1 = validate_cartesian_xyz(p1_raw, name=f"{label}.pose_end")
        return make_movel(p0, p1, rate=config.rate, limits=config.limits, label=label), current_q, p1

    if wp_type == "gripper":
        duration_s = float(wp.get("duration", 2.0))
        seg = make_gripper(
            float(wp["gap_start"]),
            float(wp["gap_end"]),
            rate=config.rate,
            duration_s=duration_s,
            label=label,
        )
        return seg, current_q, current_xyz

    raise ValueError(f"unknown waypoint type: {wp_type!r}")
