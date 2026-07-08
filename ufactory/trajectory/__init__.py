"""Time-parameterized trajectory planning for UFACTORY robots.

Provides a firmware-aligned (LSPB / trapezoidal) trajectory kernel that is
replayed identically in Genesis (PD) and on the real xArm (MODE_SERVO),
giving sim-to-real alignment by construction (same absolute target stream per
tick). The runtime profile/segment kernel is pure NumPy; heavy simulation and
real-robot executors are imported lazily.

Public surface
---------------
* :class:`TrajectoryPlannerConfig`
* :class:`JointWaypoint`, :class:`CartesianWaypoint`
* :func:`plan_joint_waypoints`, :func:`plan_cartesian_waypoints`,
  :func:`plan_mixed_waypoints`
* :func:`build_pickplace_program`
* :func:`replay_sim`
* :func:`replay_real`
* :class:`Program`, :class:`Segment`, :class:`JointLimits`
* :class:`RealExecutorConfig`, :class:`ServoLimits`, :class:`ServoStreamStats`
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "build_pickplace_program",
    "TrajectoryPlannerConfig",
    "JointWaypoint",
    "CartesianWaypoint",
    "plan_joint_waypoints",
    "plan_cartesian_waypoints",
    "plan_mixed_waypoints",
    "validate_program",
    "validate_segment",
    "validate_joint_vector",
    "validate_cartesian_xyz",
    "OptionalTrajectoryDependencyError",
    "require_roboticstoolbox",
    "replay_sim",
    "replay_real",
    "TrajKinematicMirror",
    "KinematicCarryTracker",
    "Program",
    "Segment",
    "JointLimits",
    "SimReport",
    "PhaseStatus",
    "RealExecutorConfig",
    "ServoLimits",
    "ServoStreamStats",
    "TrajectorySafetyError",
    "EXECUTOR_SERVO_J",
    "EXECUTOR_SERVO_CART",
    "REAL_EXECUTORS",
    "compute_servo_stream_stats",
    "validate_servo_stream",
    "check_segment_safety",
    "make_movej",
    "make_movel",
    "make_gripper",
    "joint_lspb_samples",
    "linear_cartesian_samples",
    "gap_lspb_samples",
    "lspb_duration",
    "joint_limits",
    "linear_limits_from_joint",
    "profile",
    "segments",
    "planner",
    "validation",
    "backends",
    "sim_executor",
    "real_executor",
    "mirror_executor",
]

_LAZY_ATTRS = {
    # profile.py: pure NumPy, no Genesis import.
    "joint_lspb_samples": ("ufactory.trajectory.profile", "joint_lspb_samples"),
    "linear_cartesian_samples": ("ufactory.trajectory.profile", "linear_cartesian_samples"),
    "gap_lspb_samples": ("ufactory.trajectory.profile", "gap_lspb_samples"),
    "lspb_duration": ("ufactory.trajectory.profile", "lspb_duration"),
    "joint_limits": ("ufactory.trajectory.profile", "joint_limits"),
    "linear_limits_from_joint": ("ufactory.trajectory.profile", "linear_limits_from_joint"),
    # segments.py: pure NumPy program assembly.
    "JointLimits": ("ufactory.trajectory.segments", "JointLimits"),
    "Program": ("ufactory.trajectory.segments", "Program"),
    "Segment": ("ufactory.trajectory.segments", "Segment"),
    "build_pickplace_program": ("ufactory.trajectory.segments", "build_pickplace_program"),
    "make_movej": ("ufactory.trajectory.segments", "make_movej"),
    "make_movel": ("ufactory.trajectory.segments", "make_movel"),
    "make_gripper": ("ufactory.trajectory.segments", "make_gripper"),
    # planner.py: robot-aware waypoint planning, no heavy optional backend import.
    "TrajectoryPlannerConfig": ("ufactory.trajectory.planner", "TrajectoryPlannerConfig"),
    "JointWaypoint": ("ufactory.trajectory.planner", "JointWaypoint"),
    "CartesianWaypoint": ("ufactory.trajectory.planner", "CartesianWaypoint"),
    "plan_joint_waypoints": ("ufactory.trajectory.planner", "plan_joint_waypoints"),
    "plan_cartesian_waypoints": ("ufactory.trajectory.planner", "plan_cartesian_waypoints"),
    "plan_mixed_waypoints": ("ufactory.trajectory.planner", "plan_mixed_waypoints"),
    # validation.py: generic Program/Segment contract checks.
    "validate_program": ("ufactory.trajectory.validation", "validate_program"),
    "validate_segment": ("ufactory.trajectory.validation", "validate_segment"),
    "validate_joint_vector": ("ufactory.trajectory.validation", "validate_joint_vector"),
    "validate_cartesian_xyz": ("ufactory.trajectory.validation", "validate_cartesian_xyz"),
    # backends.py: optional third-party loaders.
    "OptionalTrajectoryDependencyError": ("ufactory.trajectory.backends", "OptionalTrajectoryDependencyError"),
    "require_roboticstoolbox": ("ufactory.trajectory.backends", "require_roboticstoolbox"),
    # sim_executor.py: imports Genesis.
    "SimReport": ("ufactory.trajectory.sim_executor", "SimReport"),
    "PhaseStatus": ("ufactory.trajectory.sim_executor", "PhaseStatus"),
    "replay_sim": ("ufactory.trajectory.sim_executor", "replay_sim"),
    # real_executor.py: imports real-robot helpers and optional SDK only on connect.
    "EXECUTOR_SERVO_CART": ("ufactory.trajectory.real_executor", "EXECUTOR_SERVO_CART"),
    "EXECUTOR_SERVO_J": ("ufactory.trajectory.real_executor", "EXECUTOR_SERVO_J"),
    "REAL_EXECUTORS": ("ufactory.trajectory.real_executor", "REAL_EXECUTORS"),
    "RealExecutorConfig": ("ufactory.trajectory.real_executor", "RealExecutorConfig"),
    "ServoLimits": ("ufactory.trajectory.real_executor", "ServoLimits"),
    "ServoStreamStats": ("ufactory.trajectory.real_executor", "ServoStreamStats"),
    "TrajectorySafetyError": ("ufactory.trajectory.real_executor", "TrajectorySafetyError"),
    "compute_servo_stream_stats": ("ufactory.trajectory.real_executor", "compute_servo_stream_stats"),
    "validate_servo_stream": ("ufactory.trajectory.real_executor", "validate_servo_stream"),
    "check_segment_safety": ("ufactory.trajectory.real_executor", "check_segment_safety"),
    "replay_real": ("ufactory.trajectory.real_executor", "replay_real"),
    "TrajKinematicMirror": ("ufactory.trajectory.mirror_executor", "TrajKinematicMirror"),
    "KinematicCarryTracker": ("ufactory.trajectory.mirror_executor", "KinematicCarryTracker"),
    # Submodules for ``from ufactory.trajectory import profile`` compatibility.
    "profile": ("ufactory.trajectory.profile", None),
    "segments": ("ufactory.trajectory.segments", None),
    "planner": ("ufactory.trajectory.planner", None),
    "validation": ("ufactory.trajectory.validation", None),
    "backends": ("ufactory.trajectory.backends", None),
    "sim_executor": ("ufactory.trajectory.sim_executor", None),
    "real_executor": ("ufactory.trajectory.real_executor", None),
    "mirror_executor": ("ufactory.trajectory.mirror_executor", None),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value
