"""Safety gate preflight tests for Robotiq 2F-85 and 2F-140 drive conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ufactory.config import load_runtime_config
from ufactory.grippers.robotiq import robotiq_gap_m_to_sim_drive
from ufactory.robots.assembly import compose_robot_urdf
from ufactory.safety.interfaces import CollisionResult
from ufactory.trajectory.preflight import create_safety_gate
from ufactory.trajectory.segments import Program, Segment


class _StubKinematics:
    def forward(self, q_rad: np.ndarray) -> np.ndarray:
        del q_rad
        return np.array([0.30, 0.00, 0.25, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)


class _RecordingCollision:
    def __init__(self) -> None:
        self.drives: list[float] = []

    def check(self, q_rad: np.ndarray, *, stage: str, gripper_drive: float | None = None) -> CollisionResult:
        del q_rad, stage
        if gripper_drive is not None:
            self.drives.append(float(gripper_drive))
        return CollisionResult(colliding=False, min_distance_m=0.05)


@pytest.mark.parametrize(("variant", "open_m"), [("robotiq_2f_85", 0.085), ("robotiq_2f_140", 0.140)])
def test_safety_gate_maps_robotiq_gap_segments_to_drive_rad(variant: str, open_m: float) -> None:
    config = load_runtime_config("xarm6", gripper=variant)
    collision = _RecordingCollision()
    urdf_path = Path(compose_robot_urdf("xarm6", variant))
    gate = create_safety_gate(
        config,
        kinematics=_StubKinematics(),
        collision=collision,
        calibration_sha256="b" * 64,
        scene_sha256="c" * 64,
        urdf_path=urdf_path,
    )
    home = np.asarray(config.arm.default_qpos_rad, dtype=np.float64)
    program = Program(
        rate=50.0,
        segments=[
            Segment("movej", 0.1, 0.5, 1.0, q_start=home, q_end=home, label="start"),
            Segment("gripper", 0.1, 0.0, 0.0, gap_start=open_m, gap_end=0.022, label="grip"),
        ],
    )
    report = gate.preflight(program, executor="servo_j")
    assert report.passed, report.violations
    expected_close = robotiq_gap_m_to_sim_drive(variant, 0.022)
    assert collision.drives[-1] == pytest.approx(expected_close, abs=1e-6)
