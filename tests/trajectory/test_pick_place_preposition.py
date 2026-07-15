"""Tests for MODE_POSITION preposition before servo streaming."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ufactory.cli.pick_place import (
    OBJECT_RELOCATED_STAGES,
    _build_program,
    _ensure_cartesian_program_start,
    _ensure_program_start,
    _home_xyz_m,
    _scene_layout_kwargs,
    _sdk_sim_feedback,
)
from ufactory.config import RepositoryAssetStore, load_runtime_config
from ufactory.hardware.transport import XArmTransport
from ufactory.trajectory.segments import Program, Segment


class _FakeArm:
    def __init__(self, q: np.ndarray, *, xyz_mm: np.ndarray | None = None):
        self.q = np.asarray(q, dtype=np.float64).copy()
        self.connected = True
        self.error_code = 0
        self.warn_code = 0
        self.state = 2
        self.mode = 0
        self.axis = int(self.q.size)
        self.calls: list[tuple] = []
        # Default far from Cartesian home so preposition tests exercise set_position.
        self.xyz_mm = (
            np.asarray(xyz_mm, dtype=np.float64).copy()
            if xyz_mm is not None
            else np.asarray([100.0, 0.0, 100.0], dtype=np.float64)
        )
        self.rpy_rad = np.asarray([np.pi, 0.0, 0.0], dtype=np.float64)

    def get_servo_angle(self, is_radian=True):
        return 0, self.q.tolist()

    def get_position(self, is_radian=True):
        return 0, [*self.xyz_mm.tolist(), *self.rpy_rad.tolist()]

    def get_state(self):
        return 0, self.state

    def set_servo_angle(self, angle, speed, mvacc, wait=True, is_radian=True):
        self.calls.append(("set_servo_angle", list(angle), float(speed), float(mvacc), bool(wait)))
        self.q = np.asarray(angle, dtype=np.float64)
        return 0

    def set_position(self, *pose, speed=100.0, mvacc=500.0, wait=True, is_radian=True):
        self.calls.append(("set_position", list(pose), float(speed), float(mvacc), bool(wait)))
        self.xyz_mm = np.asarray(pose[:3], dtype=np.float64)
        self.rpy_rad = np.asarray(pose[3:6], dtype=np.float64)
        return 0

    def motion_enable(self, enable=True):
        self.calls.append(("motion_enable", bool(enable)))
        return 0

    def set_mode(self, mode):
        self.mode = int(mode)
        self.calls.append(("set_mode", int(mode)))
        return 0

    def set_state(self, state):
        self.state = int(state)
        self.calls.append(("set_state", int(state)))
        return 0

    def disconnect(self):
        self.connected = False


class _FakeSdkSimArm(_FakeArm):
    def __init__(self, physical_q: np.ndarray):
        super().__init__(physical_q)
        self.physical_q = np.asarray(physical_q, dtype=np.float64).copy()
        self.virtual_q = self.physical_q.copy()
        self.simulation_enabled = False

    @property
    def is_simulation_robot(self):
        return self.simulation_enabled

    def set_simulation_robot(self, enabled):
        self.simulation_enabled = bool(enabled)
        self.calls.append(("set_simulation_robot", self.simulation_enabled))
        return 0

    def get_servo_angle(self, is_radian=True):
        q = self.virtual_q if self.simulation_enabled else self.physical_q
        return 0, q.tolist()

    def set_servo_angle(self, angle, speed, mvacc, wait=True, is_radian=True):
        assert self.simulation_enabled
        self.calls.append(("set_servo_angle", list(angle), float(speed), float(mvacc), bool(wait)))
        self.virtual_q = np.asarray(angle, dtype=np.float64)
        return 0

    def set_servo_angle_j(self, angles, speed, mvacc, mvtime, is_radian=True):
        assert self.simulation_enabled
        self.calls.append(("set_servo_angle_j", list(angles)))
        self.virtual_q = np.asarray(angles, dtype=np.float64)
        return 0

    def set_state(self, state):
        requested = int(state)
        self.state = 2 if requested == 0 else requested
        self.calls.append(("set_state", requested))
        return 0


class _FakeKinematics:
    def forward(self, q_rad: np.ndarray) -> np.ndarray:
        q = np.asarray(q_rad, dtype=np.float64)
        # Deliberately NOT Cartesian home — program must not use this as start_xyz.
        return np.asarray((0.22 + 0.01 * q[0], 0.0, 0.18 + 0.01 * q[1], 0.0, 0.0, 0.0, 1.0))

    def inverse(self, pose: np.ndarray, seed_q_rad: np.ndarray) -> np.ndarray:
        return np.asarray(seed_q_rad, dtype=np.float64)


def test_preposition_joints_uses_blocking_set_servo_angle_not_servo_stream():
    start = np.asarray([0.1, -0.2, 0.0, 0.0, 0.3, 0.0])
    target = np.asarray([0.0, -0.5, 0.0, 0.0, 0.5, 0.0])
    arm = _FakeArm(start)
    transport = XArmTransport(arm, robot_key="xarm6_1305", serial_number="XI130506XXXXXX")
    err = transport.preposition_joints(target, speed_rad_s=0.2, mvacc_rad_s2=1.5, tolerance_rad=0.02)
    assert err <= 0.02
    assert any(call[0] == "set_servo_angle" for call in arm.calls)
    assert not any(call[0] == "set_servo_angle_j" for call in arm.calls)
    assert np.allclose(arm.q, target)


def test_preposition_cartesian_uses_blocking_set_position_gripper_down():
    arm = _FakeArm(np.zeros(6), xyz_mm=np.asarray([100.0, 50.0, 80.0]))
    transport = XArmTransport(arm, robot_key="xarm6_1305", serial_number="XI130506XXXXXX")
    target = np.asarray([0.30, 0.00, 0.30])
    xyz_err, rpy_err = transport.preposition_cartesian(target, speed_mm_s=80.0, mvacc_mm_s2=400.0)
    assert xyz_err <= 2.0
    assert rpy_err <= 0.05
    assert any(call[0] == "set_position" for call in arm.calls)
    assert np.allclose(arm.xyz_mm, target * 1000.0)
    assert np.allclose(arm.rpy_rad, [np.pi, 0.0, 0.0])


def test_leave_software_stop_clears_state_4_without_clean_error():
    arm = _FakeArm(np.zeros(6))
    arm.state = 4
    transport = XArmTransport(arm, robot_key="xarm6_1305", serial_number="XI130506XXXXXX")
    transport.leave_software_stop()
    assert arm.state == 0
    assert ("set_state", 0) in arm.calls
    assert not any(call[0] == "clean_error" for call in arm.calls)


def test_read_state_truncates_padded_seven_vector_to_axis():
    arm = _FakeArm(np.zeros(6))
    arm.q = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 9.9])
    transport = XArmTransport(arm, robot_key="xarm6_1305", serial_number="XI130506XXXXXX")
    state = transport.read_state()
    assert state.q_rad.shape == (6,)
    assert np.allclose(state.q_rad, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


def test_ensure_program_start_requires_confirm_when_far_from_default():
    start = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    target = np.asarray([0.0, -0.5, 0.0, 0.0, 0.5, 0.0])
    transport = XArmTransport(_FakeArm(start), robot_key="xarm6_1305", serial_number="XI130506XXXXXX")
    with pytest.raises(RuntimeError, match="--confirm-real"):
        _ensure_program_start(transport, target, allow_preposition=False)


def test_ensure_program_start_prepositions_when_allowed():
    start = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    target = np.asarray([0.0, -0.5, 0.0, 0.0, 0.5, 0.0])
    arm = _FakeArm(start)
    transport = XArmTransport(arm, robot_key="xarm6_1305", serial_number="XI130506XXXXXX")
    reached = _ensure_program_start(transport, target, allow_preposition=True)
    assert np.allclose(reached, target)
    assert any(call[0] == "set_servo_angle" for call in arm.calls)


def test_ensure_cartesian_program_start_requires_confirm_when_far():
    transport = XArmTransport(_FakeArm(np.zeros(6)), robot_key="xarm6_1305", serial_number="XI130506XXXXXX")
    with pytest.raises(RuntimeError, match="--confirm-real"):
        _ensure_cartesian_program_start(
            transport,
            np.asarray([0.30, 0.00, 0.30]),
            allow_preposition=False,
        )


def test_ensure_cartesian_program_start_prepositions_when_allowed():
    arm = _FakeArm(np.zeros(6), xyz_mm=np.asarray([100.0, 0.0, 100.0]))
    transport = XArmTransport(arm, robot_key="xarm6_1305", serial_number="XI130506XXXXXX")
    target = np.asarray([0.30, 0.00, 0.30])
    _ensure_cartesian_program_start(transport, target, allow_preposition=True)
    assert any(call[0] == "set_position" for call in arm.calls)
    assert np.allclose(arm.xyz_mm, target * 1000.0)


def test_sdk_sim_feedback_prepositions_only_virtual_robot(monkeypatch):
    monkeypatch.setattr("ufactory.cli.pick_place.time.sleep", lambda _seconds: None)
    physical_start = np.asarray([0.3, -0.2, 0.1, 0.0, 0.2, -0.1])
    virtual_start = np.asarray([0.0, -0.5, 0.0, 0.0, 0.5, 0.0])
    virtual_end = virtual_start + np.asarray([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    segment = Segment(
        kind="movej",
        duration=0.02,
        v_max=1.0,
        a_max=2.0,
        label="move",
        q_start=virtual_start,
        q_end=virtual_end,
        q_samples=np.stack((virtual_start, virtual_end)),
        samples_count=2,
    )
    program = Program([segment], rate=50.0, robot_key="xarm6_1305")
    config = load_runtime_config("xarm6")
    arm = _FakeSdkSimArm(physical_start)

    feedback = _sdk_sim_feedback(arm, program, config, virtual_start)

    assert np.allclose(feedback, np.stack((virtual_start, virtual_start, virtual_end)))
    assert np.allclose(arm.physical_q, physical_start)
    assert ("set_simulation_robot", True) in arm.calls
    assert ("set_simulation_robot", False) in arm.calls
    assert not arm.simulation_enabled
    assert next(call for call in arm.calls if call[0] == "set_servo_angle")[1] == pytest.approx(virtual_start)
    assert any(call[0] == "set_servo_angle_j" for call in arm.calls)
    sim_index = arm.calls.index(("set_simulation_robot", True))
    virtual_move_index = next(i for i, call in enumerate(arm.calls) if call[0] == "set_servo_angle")
    disable_index = arm.calls.index(("set_simulation_robot", False))
    assert sim_index < virtual_move_index < disable_index


def test_sdk_sim_feedback_disables_simulation_after_replay_failure(monkeypatch):
    monkeypatch.setattr("ufactory.cli.pick_place.time.sleep", lambda _seconds: None)
    start = np.zeros(6, dtype=np.float64)
    segment = Segment(
        kind="movej",
        duration=0.02,
        v_max=1.0,
        a_max=2.0,
        label="move",
        q_start=start,
        q_end=start,
        q_samples=start[None, :],
        samples_count=1,
    )
    arm = _FakeSdkSimArm(np.ones(6, dtype=np.float64))
    arm.set_servo_angle_j = lambda **_kwargs: 9

    with pytest.raises(RuntimeError, match="SDK simulation command failed with code 9"):
        _sdk_sim_feedback(
            arm,
            Program([segment], rate=50.0, robot_key="xarm6_1305"),
            load_runtime_config("xarm6"),
            start,
        )

    assert not arm.simulation_enabled
    assert ("set_simulation_robot", False) in arm.calls


def test_build_program_cartesian_starts_at_home_not_default_fk():
    config = load_runtime_config("xarm6")
    kinematics = _FakeKinematics()
    program = _build_program(config, kinematics)
    assert program.segments
    # Pinocchio/fake IK seeds a zero-length MoveJ at home for preflight.
    assert program.segments[0].kind == "movej"
    assert program.segments[0].label == "start"
    home_seg = next(seg for seg in program.segments if seg.kind == "movel" and seg.label == "home->pregrasp")
    assert home_seg.pose_start is not None
    assert np.allclose(home_seg.pose_start[:3], (0.30, 0.00, 0.30), atol=1e-3)
    # Must not start from FK(default_qpos)-like pose (~0.22, 0, 0.18).
    assert not np.allclose(
        home_seg.pose_start[:3],
        kinematics.forward(np.asarray(config.arm.default_qpos_rad))[:3],
        atol=1e-2,
    )
    assert np.allclose(_home_xyz_m(config), (0.30, 0.00, 0.30), atol=1e-3)


def test_build_program_with_q_home_prepends_start_movej():
    config = load_runtime_config("xarm6")
    q_home = np.asarray(config.arm.default_qpos_rad, dtype=np.float64)
    program = _build_program(config, q_home=q_home)
    assert program.segments[0].kind == "movej"
    assert program.segments[0].label == "start"
    assert float(program.segments[0].duration) == 0.0
    home_seg = next(seg for seg in program.segments if seg.kind == "movel" and seg.label == "home->pregrasp")
    assert np.allclose(home_seg.pose_start[:3], (0.30, 0.00, 0.30), atol=1e-3)


def test_default_pick_place_layout_matches_v024_sequence():
    config = load_runtime_config("xarm6")
    params = config.task.parameters
    assert tuple(map(float, params["fixed_object_position_m"])) == pytest.approx((0.30, 0.00, 0.015))
    assert tuple(map(float, params["fixed_target_position_m"])) == pytest.approx((0.30, 0.30, 0.015))
    layout = _scene_layout_kwargs(config)
    assert layout["obj_pos_base"] == pytest.approx((0.30, 0.00, 0.015))
    assert layout["place_pos_base"] == pytest.approx((0.30, 0.30, 0.015))
    assert layout["obj_size"] == pytest.approx((0.030, 0.030, 0.030))
    assert layout["obj_mass_kg"] == pytest.approx(0.017)

    program = _build_program(config, _FakeKinematics())
    labels = [seg.label for seg in program.segments]
    for required in (
        "start",
        "home->pregrasp",
        "descend",
        "grip",
        "grip-settle",
        "lift",
        "transit",
        "place-descend",
        "release",
        "retreat",
        "return-home",
    ):
        assert required in labels
    assert "place-settle" not in labels
    assert not any(seg.kind == "movej" and seg.label == "retreat" for seg in program.segments)
    assert any(seg.kind == "movel" and seg.label == "return-home" for seg in program.segments)

    place_xyz = None
    for segment in program.segments:
        if segment.kind == "movel" and segment.label == "place-descend" and segment.pose_end is not None:
            place_xyz = np.asarray(segment.pose_end[:3], dtype=np.float64)
    assert place_xyz is not None
    assert place_xyz[0] == pytest.approx(0.30, abs=1e-6)
    assert place_xyz[1] == pytest.approx(0.30, abs=1e-6)


def test_lite6_pick_place_program_has_place_settle_not_grip_settle():
    config = load_runtime_config("lite6")

    class _LiteKin(_FakeKinematics):
        def forward(self, q_rad: np.ndarray) -> np.ndarray:
            qv = np.asarray(q_rad, dtype=np.float64)
            return np.asarray((0.2 + 0.01 * qv[0], 0.0, 0.20 + 0.01 * qv[1], 0.0, 0.0, 0.0, 1.0))

    program = _build_program(config, _LiteKin())
    labels = [seg.label for seg in program.segments]
    assert "place-settle" in labels
    assert "grip-settle" not in labels


def test_default_pick_place_program_preflight_passes_servo_j():
    pytest.importorskip("pinocchio")
    pytest.importorskip("genesis")
    from ufactory.cli.pick_place import _compile_servo_j_program
    from ufactory.safety.adapters import EnvironmentObstacle, PinocchioCollisionBackend, PinocchioKinematicsBackend
    from ufactory.safety.adapters.pinocchio import StageAwareObjectCollisionBackend
    from ufactory.simulation import GenesisRuntimeManager
    from ufactory.trajectory.preflight import create_safety_gate

    config = load_runtime_config("xarm6")
    store = RepositoryAssetStore.discover()
    urdf = store.require(Path(config.robot.assets_dir) / config.robot.urdf)
    passive = {config.gripper.drive_joint: config.gripper.open_drive}
    kinematics = PinocchioKinematicsBackend(
        urdf,
        joint_names=config.robot.joint_names,
        ee_link=config.robot.ee_link,
        passive_joint_positions=passive,
    )
    params = config.task.parameters
    object_pos = tuple(map(float, params["fixed_object_position_m"]))
    target_pos = tuple(map(float, params["fixed_target_position_m"]))
    collision = StageAwareObjectCollisionBackend(
        PinocchioCollisionBackend(
            urdf,
            joint_names=config.robot.joint_names,
            ee_link=config.robot.ee_link,
            passive_joint_positions=passive,
            adjacent_link_pairs=config.robot.adjacent_collision_pairs,
            obstacles=(
                EnvironmentObstacle("table", (1.2, 1.2, 0.05), (0.3, 0.0, -0.025)),
                EnvironmentObstacle(
                    "object",
                    tuple(map(float, params["object_size_m"])),
                    object_pos,
                ),
            ),
        ),
        spawn_center_m=object_pos,
        place_center_m=target_pos,
        relocated_stages=OBJECT_RELOCATED_STAGES,
    )
    with GenesisRuntimeManager(config.simulation):
        program, _ctx, q_home = _compile_servo_j_program(config, calibration=None)
    assert program.segments[0].kind == "movej"
    assert program.segments[0].label == "start"
    start_samples, n_start = program.segments[0].samples(program.rate)
    assert n_start >= 1
    assert np.allclose(q_home, np.asarray(start_samples[0], dtype=np.float64), atol=1e-6)
    home_seg = next(seg for seg in program.segments if seg.label == "home->pregrasp")
    # After Genesis compile, home->pregrasp is a MoveJ stream (down-quat IK).
    assert home_seg.kind == "movej"
    gate = create_safety_gate(
        config,
        kinematics=kinematics,
        collision=collision,
        calibration_sha256="a" * 64,
        scene_sha256="b" * 64,
        urdf_path=urdf,
    )
    report = gate.preflight(program, executor="servo_j")
    assert report.passed, [f"{v.type.value}:{v.stage}:{v.message}" for v in report.violations]


@pytest.mark.integration
def test_lite6_cartesian_preflight_preserves_gripper_down_orientation():
    pytest.importorskip("pinocchio")
    pytest.importorskip("coal")
    from ufactory.cli.pick_place import _backends, _model_and_hashes
    from ufactory.kinematics.orientation import GRIPPER_DOWN_QUAT_XYZW
    from ufactory.trajectory.preflight import create_safety_gate

    config = load_runtime_config("lite6")
    urdf, _urdf_hash, calibration_hash = _model_and_hashes(config, None, None)
    kinematics, collision = _backends(config, urdf)
    program = _build_program(config, kinematics)
    gate = create_safety_gate(
        config,
        kinematics=kinematics,
        collision=collision,
        calibration_sha256=calibration_hash,
        scene_sha256="b" * 64,
        urdf_path=urdf,
    )
    report = gate.preflight(program, executor="servo_cartesian")
    assert report.passed, [f"{v.type.value}:{v.stage}:{v.message}" for v in report.violations]

    shadow, _stages = gate.shadow_joint_stream(program)
    desired = np.asarray(GRIPPER_DOWN_QUAT_XYZW, dtype=np.float64)
    errors = []
    for q_rad in shadow:
        quaternion = np.asarray(kinematics.forward(q_rad)[3:7], dtype=np.float64)
        quaternion /= np.linalg.norm(quaternion)
        errors.append(2.0 * np.arccos(np.clip(abs(float(np.dot(quaternion, desired))), 0.0, 1.0)))
    assert max(errors) <= 1e-4
