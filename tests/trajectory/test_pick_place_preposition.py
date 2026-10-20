"""Tests for MODE_POSITION preposition before servo streaming."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ufactory.cli.pick_place import (
    OBJECT_RELOCATED_STAGES,
    _build_program,
    _home_xyz_m,
    _scene_layout_kwargs,
)
from ufactory.config import RepositoryAssetStore, load_runtime_config


class _FakeKinematics:
    def forward(self, q_rad: np.ndarray) -> np.ndarray:
        q = np.asarray(q_rad, dtype=np.float64)
        # Deliberately NOT Cartesian home — program must not use this as start_xyz.
        return np.asarray((0.22 + 0.01 * q[0], 0.0, 0.18 + 0.01 * q[1], 0.0, 0.0, 0.0, 1.0))

    def inverse(self, pose: np.ndarray, seed_q_rad: np.ndarray) -> np.ndarray:
        return np.asarray(seed_q_rad, dtype=np.float64)


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


def test_default_pick_place_layout_matches_reference_sequence():
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


@pytest.mark.gpu
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


def test_xarm7_stationary_seed_offset_stays_inside_cartesian_acceleration_limit():
    pytest.importorskip("pinocchio")
    from ufactory.cli.pick_place import _backends, _model_and_hashes
    from ufactory.kinematics.orientation import GRIPPER_DOWN_QUAT_XYZW
    from ufactory.safety.statistics import compute_motion_statistics
    from ufactory.trajectory.g2_confirmation import prepare_g2_program
    from ufactory.trajectory.preflight import create_safety_gate

    config = load_runtime_config("xarm7")
    urdf, _urdf_hash, calibration_hash = _model_and_hashes(config, None, None)
    kinematics, _collision = _backends(config, urdf)
    home = _home_xyz_m(config)
    quaternion = np.asarray(GRIPPER_DOWN_QUAT_XYZW, dtype=np.float64)
    on_path = np.asarray(
        kinematics.inverse(np.concatenate((home, quaternion)), np.asarray(config.arm.default_qpos_rad)),
        dtype=np.float64,
    )
    # Same micrometre Genesis/Pinocchio mismatch that previously read as 0.802 m/s².
    offset = home + np.array([-4.176e-6, -9.6e-8, -7.43e-7])
    q_home = np.asarray(kinematics.inverse(np.concatenate((offset, quaternion)), on_path), dtype=np.float64)
    program = prepare_g2_program(_build_program(config, q_home=q_home), config.simulation.g2)
    gate = create_safety_gate(
        config,
        kinematics=kinematics,
        collision=None,
        calibration_sha256=calibration_hash,
        scene_sha256="b" * 64,
        urdf_path=urdf,
    )
    timeline, violations = gate._build_timeline(program)
    assert not violations
    stats = compute_motion_statistics(timeline.q_rad, timeline.xyz_m, rate_hz=program.rate)
    limit = config.motion.cartesian_acceleration_m_s2
    assert stats.max_cartesian_acceleration_m_s2 <= limit * 1.001 + 1e-12
