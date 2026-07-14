"""Host-IK compilation of Cartesian segments into the exact servo-j stream."""

from __future__ import annotations

import numpy as np

from ufactory.kinematics.orientation import GRIPPER_DOWN_QUAT_XYZW
from ufactory.safety.interfaces import KinematicsBackend
from ufactory.safety.statistics import signed_velocity_and_acceleration
from ufactory.trajectory.segments import Program, Segment
from ufactory.types import FloatArray


def compile_program_for_servo_j(program: Program, kinematics: KinematicsBackend) -> Program:
    """Replace every MoveL with an explicit, continuous MoveJ sample stream."""

    compiled: list[Segment] = []
    current_q: FloatArray | None = None
    compiled_ticks = 0
    for segment in program.segments:
        samples, _ = segment.samples(program.rate)
        if segment.kind == "movej":
            compiled.append(segment)
            current_q = samples[-1].copy()
            continue
        if segment.kind == "gripper":
            compiled.append(segment)
            continue
        if current_q is None:
            raise ValueError("servo_j compilation requires a preceding MoveJ state")
        q_start = current_q.copy()
        rows: list[FloatArray] = []
        for pose in samples:
            full_pose = np.concatenate((pose, np.asarray(GRIPPER_DOWN_QUAT_XYZW, dtype=np.float64)))
            current_q = np.asarray(kinematics.inverse(full_pose, current_q), dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(current_q)):
                raise ValueError(f"IK returned non-finite joints in segment {segment.label!r}")
            rows.append(current_q.copy())
        stream = np.stack(rows)
        full = np.concatenate((q_start[None, :], stream), axis=0)
        velocity, acceleration = signed_velocity_and_acceleration(full, rate_hz=program.rate)
        compiled.append(
            Segment(
                kind="movej",
                duration=segment.duration,
                v_max=float(np.max(np.abs(velocity), initial=0.0)),
                a_max=float(np.max(np.abs(acceleration), initial=0.0)),
                label=segment.label,
                q_start=q_start,
                q_end=stream[-1].copy(),
                q_samples=stream,
                samples_count=len(stream),
            )
        )
        compiled_ticks += len(stream)
    metadata = {
        **program.metadata,
        "servo_j_compiled": True,
        "servo_j_compiled_ticks": compiled_ticks,
        "servo_j_kinematics_backend": type(kinematics).__name__,
    }
    return Program(
        segments=compiled,
        rate=program.rate,
        limits=program.limits,
        robot_key=program.robot_key,
        metadata=metadata,
    )
