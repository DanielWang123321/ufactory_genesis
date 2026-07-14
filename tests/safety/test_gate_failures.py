"""Failure-path coverage for every predictive safety preflight check."""

from __future__ import annotations

from dataclasses import replace
import json
import time

import numpy as np
import pytest

from ufactory.config import load_runtime_config
from ufactory.safety import CollisionResult, ViolationType
from ufactory.safety.gate import SafetyGate
from ufactory.safety.sdk_sim import (
    SDK_SIMULATION_EVIDENCE_MAX_AGE_S,
    load_sdk_simulation_evidence,
    validate_sdk_simulation,
)
from ufactory.trajectory.segments import Program, Segment


class Kin:
    def __init__(self, *, xyz=(0.3, 0.0, 0.3), quaternion=(0.0, 0.0, 0.0, 1.0), inverse=None, fail=False):
        self.xyz = xyz
        self.quaternion = quaternion
        self.inverse_result = inverse
        self.fail = fail

    def forward(self, q):
        if self.fail:
            raise RuntimeError("FK unavailable")
        quaternion = self.quaternion(q) if callable(self.quaternion) else self.quaternion
        return np.asarray((*self.xyz, *quaternion), dtype=np.float64)

    def inverse(self, pose, seed):
        if self.fail:
            raise RuntimeError("IK unavailable")
        return np.asarray(self.inverse_result if self.inverse_result is not None else seed, dtype=np.float64)


class Collision:
    def __init__(self, *results, fail=False):
        self.results = results or (CollisionResult(False, 1.0, "link1", "link3"),)
        self.fail = fail

    def check(self, q, *, stage, gripper_drive=None):
        if self.fail:
            raise RuntimeError("collision unavailable")
        return self.results[0]

    def check_all(self, q, *, stage, gripper_drive=None):
        if self.fail:
            raise RuntimeError("collision unavailable")
        return self.results


class MarginCollision(Collision):
    def __init__(self, *, fail=False):
        super().__init__()
        self.margin_calls: list[float] = []
        self.fail_margin = fail

    def check_all(self, q, *, stage, gripper_drive=None):
        raise AssertionError("full-distance fallback must not run when margin querying is available")

    def check_all_within_margin(self, q, *, stage, margin_m, gripper_drive=None):
        self.margin_calls.append(float(margin_m))
        if self.fail_margin:
            raise RuntimeError("margin collision unavailable")
        return ()


def program(samples=None, *, rate=50.0, robot="xarm6_1305", label="approach"):
    values = np.asarray(samples if samples is not None else [[0.001] * 6, [0.002] * 6], dtype=np.float64)
    return Program(
        [
            Segment(
                "movej",
                len(values) / rate,
                10.0,
                100.0,
                label,
                q_start=np.zeros(6),
                q_end=values[-1],
                q_samples=values,
                samples_count=len(values),
            )
        ],
        rate=rate,
        robot_key=robot,
    )


def gate(*, kin=Kin(), collision=Collision(), config=None):
    cfg = config or load_runtime_config("xarm6")
    return SafetyGate(
        cfg,
        joint_lower_rad=[-3.0] * 6,
        joint_upper_rad=[3.0] * 6,
        kinematics=kin,
        collision=collision,
        urdf_sha256="u" * 64,
        calibration_sha256="c" * 64,
        scene_sha256="s" * 64,
    )


def kinds(report):
    return {item.type for item in report.violations}


def test_preflight_prefers_margin_collision_capability_and_reports_query_details():
    collision = MarginCollision()
    report = gate(collision=collision).preflight(program(), executor="servo_j")

    assert report.passed
    assert collision.margin_calls == [0.005, 0.005, 0.005]
    collision_check = next(check for check in report.checks if check.name == "collision")
    assert collision_check.details == {
        "query_mode": "security-margin-candidates",
        "margin_m": 0.005,
        "returned_pair_samples": 0,
    }


def test_preflight_margin_query_failure_is_fail_closed_without_exact_fallback():
    report = gate(collision=MarginCollision(fail=True)).preflight(program(), executor="servo_j")

    assert not report.passed
    assert ViolationType.BACKEND_UNAVAILABLE in kinds(report)


def test_constructor_rejects_bad_bounds():
    with pytest.raises(ValueError, match="shape"):
        SafetyGate(
            load_runtime_config("xarm6"),
            joint_lower_rad=[0],
            joint_upper_rad=[1],
            kinematics=None,
            collision=None,
            urdf_sha256="u",
            calibration_sha256="c",
            scene_sha256="s",
        )
    with pytest.raises(ValueError, match="finite"):
        gate_with = load_runtime_config("xarm6")
        SafetyGate(
            gate_with,
            joint_lower_rad=[-3] * 5 + [np.nan],
            joint_upper_rad=[3] * 6,
            kinematics=None,
            collision=None,
            urdf_sha256="u",
            calibration_sha256="c",
            scene_sha256="s",
        )
    with pytest.raises(ValueError, match="below"):
        SafetyGate(
            load_runtime_config("xarm6"),
            joint_lower_rad=[3] * 6,
            joint_upper_rad=[3] * 6,
            kinematics=None,
            collision=None,
            urdf_sha256="u",
            calibration_sha256="c",
            scene_sha256="s",
        )


def test_identity_executor_rate_and_required_backend_fail_closed():
    report = gate(kin=None, collision=None).preflight(program(rate=25.0, robot="lite6"), executor="bad")
    assert {
        ViolationType.IDENTITY,
        ViolationType.TIMING,
        ViolationType.INVALID_INPUT,
        ViolationType.BACKEND_UNAVAILABLE,
    } <= kinds(report)


def test_joint_workspace_orientation_and_motion_limits_are_reported():
    cfg = load_runtime_config("xarm6")
    cfg = replace(
        cfg,
        motion=replace(cfg.motion, joint_speed_rad_s=0.01, joint_acceleration_rad_s2=0.01),
        safety=replace(cfg.safety, max_orientation_step_rad=0.1),
    )
    calls = iter(((0.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)))
    kin = Kin(xyz=(2.0, 0.0, -0.1), quaternion=lambda _q: next(calls))
    report = gate(kin=kin, config=cfg).preflight(program([[2.99] * 6, [2.98] * 6]), executor="servo_j")
    assert {
        ViolationType.JOINT_LIMIT,
        ViolationType.Z_MIN,
        ViolationType.WORKSPACE,
        ViolationType.ORIENTATION,
        ViolationType.VELOCITY,
        ViolationType.ACCELERATION,
    } <= kinds(report)


@pytest.mark.parametrize(
    "result, expected",
    [
        (CollisionResult(True, -0.001, "link1", "link4"), ViolationType.SELF_COLLISION),
        (CollisionResult(True, -0.001, "link1", "table", environment=True), ViolationType.ENVIRONMENT_COLLISION),
        (CollisionResult(False, 0.001, "link1", "link4"), ViolationType.CLEARANCE),
        (CollisionResult(False, np.nan, "link1", "link4"), ViolationType.INVALID_INPUT),
    ],
)
def test_collision_failure_classes(result, expected):
    report = gate(collision=Collision(result)).preflight(program(), executor="servo_j")
    assert expected in kinds(report)


def test_collision_exception_and_runtime_monitor_fail_closed():
    failing = gate(collision=Collision(fail=True))
    assert ViolationType.BACKEND_UNAVAILABLE in kinds(failing.preflight(program(), executor="servo_j"))
    assert failing.runtime_monitor(np.zeros(6), "run").type is ViolationType.BACKEND_UNAVAILABLE
    assert gate().runtime_monitor(np.full(6, np.nan), "run").type is ViolationType.FEEDBACK
    assert gate().runtime_monitor(np.full(6, 3.1), "run").type is ViolationType.JOINT_LIMIT
    assert gate(kin=Kin(xyz=(2.0, 0.0, 0.3))).runtime_monitor(np.zeros(6), "run").type is ViolationType.WORKSPACE
    collision = Collision(CollisionResult(True, -0.01, "link1", "table", environment=True))
    assert gate(collision=collision).runtime_monitor(np.zeros(6), "run").type is ViolationType.ENVIRONMENT_COLLISION
    assert gate().runtime_monitor(np.zeros(6), "run") is None


def test_movel_ik_jump_and_invalid_results():
    start = Segment(
        "movej",
        0.02,
        1.0,
        2.0,
        "start",
        q_start=np.zeros(6),
        q_end=np.zeros(6),
        q_samples=np.zeros((1, 6)),
        samples_count=1,
    )
    line = Segment(
        "movel",
        0.02,
        0.1,
        0.2,
        "line",
        pose_start=np.array([0.3, 0.0, 0.3]),
        pose_end=np.array([0.31, 0.0, 0.3]),
        samples_count=1,
    )
    mixed = Program([start, line], rate=50.0, robot_key="xarm6_1305")
    report = gate(kin=Kin(inverse=[1.0] * 6)).preflight(mixed, executor="servo_j")
    assert ViolationType.IK_DISCONTINUITY in kinds(report)
    report = gate(kin=Kin(inverse=[np.nan] * 6)).preflight(mixed, executor="servo_j")
    assert ViolationType.INVALID_INPUT in kinds(report)
    report = gate(kin=Kin(fail=True)).preflight(mixed, executor="servo_j")
    assert ViolationType.INVALID_INPUT in kinds(report)


def test_segment_order_gripper_range_and_fk_outputs_fail_closed():
    line_only = _line_program()
    assert ViolationType.INVALID_INPUT in kinds(gate().preflight(line_only, executor="servo_j"))

    grip = Segment("gripper", 0.02, 1.0, 1.0, "grasp", gap_start=0.2, gap_end=0.2, samples_count=1)
    assert ViolationType.INVALID_INPUT in kinds(
        gate().preflight(Program([grip], rate=50.0, robot_key="xarm6_1305"), executor="servo_j")
    )

    q = np.zeros(6)
    start = Segment("movej", 0.02, 1.0, 2.0, "start", q_start=q, q_end=q, q_samples=q[None, :], samples_count=1)
    out_of_range = Program([start, grip], rate=50.0, robot_key="xarm6_1305")
    assert ViolationType.INVALID_INPUT in kinds(gate().preflight(out_of_range, executor="servo_j"))
    no_gripper = replace(load_runtime_config("xarm6"), gripper=None)
    assert ViolationType.INVALID_INPUT in kinds(gate(config=no_gripper).preflight(out_of_range, executor="servo_j"))

    for kin in (Kin(fail=True), Kin(xyz=(np.nan, 0.0, 0.0)), Kin(quaternion=(0.0, 0.0, 0.0, 0.0))):
        assert ViolationType.INVALID_INPUT in kinds(gate(kin=kin).preflight(program(), executor="servo_j"))


def _line_program():
    line = Segment(
        "movel",
        0.02,
        0.1,
        0.2,
        "line",
        pose_start=np.array([0.3, 0.0, 0.3]),
        pose_end=np.array([0.31, 0.0, 0.3]),
        samples_count=1,
    )
    return Program([line], rate=50.0, robot_key="xarm6_1305")


def test_allowed_task_contacts_and_optional_collision_backend():
    finger = next(iter(load_runtime_config("xarm6").gripper.allowed_contact_links))
    allowed = Collision(CollisionResult(True, -0.001, finger, "object", environment=True))
    assert gate(collision=allowed).preflight(program(label="grip"), executor="servo_j").passed

    object_table = Collision(CollisionResult(True, -0.001, "object", "table", environment=True))
    assert gate(collision=object_table).preflight(program(label="place-descend"), executor="servo_j").passed

    cfg = load_runtime_config("xarm6")
    optional = replace(cfg, safety=replace(cfg.safety, require_collision=False))
    assert gate(config=optional, collision=None).preflight(program(), executor="servo_j").passed


def test_lite6_contact_exemptions_match_exact_trajectory_stages():
    config = load_runtime_config("lite6")
    urdf_hash = config.safety.exemptions[0].urdf_sha256
    lite_gate = SafetyGate(
        config,
        joint_lower_rad=[-3.0] * 6,
        joint_upper_rad=[3.0] * 6,
        kinematics=Kin(),
        collision=Collision(),
        urdf_sha256=urdf_hash,
        calibration_sha256="c" * 64,
        scene_sha256="s" * 64,
    )
    finger_pair = CollisionResult(True, 0.0, "uflite_finger1", "uflite_finger2")
    expected_stages = {
        "grip",
        "grip-settle",
        "lift",
        "transit",
        "pre-release-settle",
        "place-descend",
        "place-settle",
        "release",
    }
    assert {item.stage for item in config.safety.exemptions} == expected_stages
    assert all(lite_gate.collision_allowed(finger_pair, stage) for stage in expected_stages)
    assert not lite_gate.collision_allowed(finger_pair, "grasp")
    assert not lite_gate.collision_allowed(finger_pair, "place")
    wrong_hash_gate = SafetyGate(
        config,
        joint_lower_rad=[-3.0] * 6,
        joint_upper_rad=[3.0] * 6,
        kinematics=Kin(),
        collision=Collision(),
        urdf_sha256="0" * 64,
        calibration_sha256="c" * 64,
        scene_sha256="s" * 64,
    )
    assert not wrong_hash_gate.collision_allowed(finger_pair, "grip")

    finger_object = CollisionResult(True, 0.0, "uflite_finger1", "object", environment=True)
    shell_object = CollisionResult(True, 0.0, "uflite_gripper_link", "object", environment=True)
    assert not lite_gate.collision_allowed(finger_object, "descend")
    assert lite_gate.collision_allowed(finger_object, "grip")
    assert not lite_gate.collision_allowed(shell_object, "grip")
    assert config.safety.min_collision_distance_m == pytest.approx(0.005)


def test_approval_rejects_empty_serial_and_cartesian_without_current_evidence():
    approved_gate = gate()
    joint_program = program()
    report = approved_gate.preflight(joint_program, executor="servo_j")
    with pytest.raises(ValueError, match="complete robot serial"):
        approved_gate.approve(joint_program, report, expected_serial_number=" ")

    cart_report = approved_gate.preflight(joint_program, executor="servo_cartesian")
    with pytest.raises(ValueError, match="requires passing"):
        approved_gate.approve(joint_program, cart_report, expected_serial_number="XI130506XXXXXX")

    shadow, stages = approved_gate.shadow_joint_stream(joint_program)
    cfg = load_runtime_config("xarm6")
    evidence = validate_sdk_simulation(
        robot_key=cfg.robot.key,
        serial_number="XI130506XXXXXX",
        program_sha256=cart_report.program_sha256,
        config_sha256=cfg.sha256,
        shadow_joint_stream_rad=shadow,
        firmware_joint_stream_rad=shadow.copy(),
        stages=stages,
        policy=cfg.safety,
        kinematics=Kin(),
        collision=Collision(),
    )
    assert (
        approved_gate.approve(
            joint_program, cart_report, expected_serial_number="XI130506XXXXXX", sdk_evidence=evidence
        ).expected_serial_number
        == "XI130506XXXXXX"
    )
    with pytest.raises(ValueError, match="stale"):
        approved_gate.approve(
            joint_program, cart_report, expected_serial_number="DIFFERENT-SERIAL", sdk_evidence=evidence
        )
    expired = replace(
        evidence,
        created_at_unix_s=time.time() - SDK_SIMULATION_EVIDENCE_MAX_AGE_S - 1.0,
    )
    with pytest.raises(ValueError, match="expired"):
        approved_gate.approve(
            joint_program,
            cart_report,
            expected_serial_number="XI130506XXXXXX",
            sdk_evidence=expired,
        )


def test_sdk_simulation_success_failures_and_strict_loader(tmp_path):
    cfg = load_runtime_config("xarm6")
    shadow = np.zeros((2, 6))
    evidence = validate_sdk_simulation(
        robot_key=cfg.robot.key,
        serial_number="XI130506XXXXXX",
        program_sha256="p" * 64,
        config_sha256=cfg.sha256,
        shadow_joint_stream_rad=shadow,
        firmware_joint_stream_rad=shadow.copy(),
        stages=("a", "b"),
        policy=cfg.safety,
        kinematics=Kin(),
        collision=Collision(),
    )
    assert evidence.passed and evidence.samples == 2

    bad = validate_sdk_simulation(
        robot_key=cfg.robot.key,
        serial_number="XI",
        program_sha256="p",
        config_sha256=cfg.sha256,
        shadow_joint_stream_rad=shadow,
        firmware_joint_stream_rad=np.ones((2, 6)),
        stages=("a", "b"),
        policy=cfg.safety,
        kinematics=Kin(),
        collision=Collision(CollisionResult(True, -0.1, "link1", "table", environment=True)),
    )
    assert not bad.passed and bad.max_joint_error_rad == 1.0

    path = tmp_path / "sdk.json"
    path.write_text(json.dumps({**evidence.__dict__, "failures": []}), encoding="utf-8")
    assert load_sdk_simulation_evidence(path) == evidence
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="fields"):
        load_sdk_simulation_evidence(path)
