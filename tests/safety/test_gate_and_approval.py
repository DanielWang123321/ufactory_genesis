from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ufactory.config import load_runtime_config
from ufactory.safety import ApprovedProgram, CollisionResult
from ufactory.trajectory.execution import ExecutionRejected, execute_sim
from ufactory.trajectory.preflight import create_safety_gate
from ufactory.trajectory.segments import Program, Segment


class FakeKinematics:
    def forward(self, q_rad: np.ndarray) -> np.ndarray:
        return np.asarray((0.3, 0.0, 0.3))

    def inverse(self, pose: np.ndarray, seed_q_rad: np.ndarray) -> np.ndarray:
        return np.asarray(seed_q_rad, dtype=np.float64)


class ClearCollision:
    def check(self, q_rad: np.ndarray, *, stage: str, gripper_drive: float | None = None) -> CollisionResult:
        return CollisionResult(False, 1.0, "link1", "link3")

    def check_all(
        self, q_rad: np.ndarray, *, stage: str, gripper_drive: float | None = None
    ) -> tuple[CollisionResult, ...]:
        return (self.check(q_rad, stage=stage, gripper_drive=gripper_drive),)


def _program(*, invalid: float | None = None) -> Program:
    q0 = np.zeros(6)
    samples = np.stack((q0 + 0.001, q0 + 0.002))
    if invalid is not None:
        samples[0, 2] = invalid
    segment = Segment(
        "movej",
        0.04,
        1.0,
        12.0,
        "approach",
        q_start=q0,
        q_end=samples[-1],
        q_samples=samples,
        samples_count=2,
    )
    return Program([segment], rate=50.0, robot_key="xarm6_1305")


def _gate():
    config = load_runtime_config("xarm6")
    urdf = Path(config.robot.assets_dir) / config.robot.urdf
    return create_safety_gate(
        config,
        kinematics=FakeKinematics(),
        collision=ClearCollision(),
        calibration_sha256="a" * 64,
        scene_sha256="b" * 64,
        urdf_path=urdf,
    )


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_gate_rejects_non_finite_joint_stream(invalid: float):
    report = _gate().preflight(_program(invalid=invalid), executor="servo_j")
    assert not report.passed
    assert any(item.type.value == "invalid_input" for item in report.violations)


def test_approved_program_constructor_is_private_and_hash_tamper_is_rejected():
    gate = _gate()
    program = _program()
    report = gate.preflight(program, executor="servo_j")
    assert report.passed
    approved = gate.approve(program, report, expected_serial_number="XI130506XXXXXX")
    with pytest.raises(TypeError, match="only be created"):
        ApprovedProgram(program, report, "XI130506XXXXXX", _issuer=object())
    program.metadata["tampered"] = True
    with pytest.raises(ExecutionRejected, match="changed"):
        execute_sim(approved, lambda _: None)


def test_cross_robot_approval_is_rejected():
    program = _program()
    program.robot_key = "lite6"
    report = _gate().preflight(program, executor="servo_j")
    assert not report.passed
    assert any(item.type.value == "identity" for item in report.violations)


def test_cartesian_simulation_approval_does_not_weaken_hardware_evidence_gate():
    gate = _gate()
    program = _program()
    report = gate.preflight(program, executor="servo_cartesian")

    simulated = gate.approve_simulation(program, report)
    assert simulated.expected_serial_number == "SIMULATED-xarm6_1305"
    execute_sim(simulated, lambda _: None)

    with pytest.raises(ValueError, match="requires passing current SDK simulation evidence"):
        gate.approve(program, report, expected_serial_number="XI130506XXXXXX")
