"""Default shared pick-place program shape, start, and preflight tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ufactory.cli.pick_place import (
    OBJECT_RELOCATED_STAGES,
    _build_program,
    _close_gap_m,
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


def _default_program(robot_key: str):
    return _build_program(load_runtime_config(robot_key))


@pytest.mark.parametrize("robot_key", ["xarm5", "xarm6", "xarm7", "uf850"])
def test_g2_pick_place_program_settles_after_grip_before_lift(robot_key: str):
    config = load_runtime_config(robot_key)
    program = _build_program(config)
    labels = [seg.label for seg in program.segments]

    assert labels == [
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
    ]
    grip = program.segments[labels.index("grip")]
    settle = program.segments[labels.index("grip-settle")]
    release = program.segments[labels.index("release")]
    grip_close_m = _close_gap_m(config)

    assert grip.duration == pytest.approx(2.0)
    assert release.duration == pytest.approx(2.0)
    assert settle.kind == "gripper"
    assert settle.duration == pytest.approx(0.5)
    assert settle.gap_start == pytest.approx(grip_close_m)
    assert settle.gap_end == pytest.approx(grip_close_m)
    assert labels.index("grip") < labels.index("grip-settle") < labels.index("lift")


def test_lite6_pick_place_program_uses_fast_gripper_without_grip_settle():
    program = _default_program("lite6")
    labels = [seg.label for seg in program.segments]

    assert labels == [
        "home->pregrasp",
        "descend",
        "grip",
        "lift",
        "transit",
        "place-descend",
        "place-settle",
        "release",
        "retreat",
        "return-home",
    ]
    assert "grip-settle" not in labels
    grip = program.segments[labels.index("grip")]
    place_settle = program.segments[labels.index("place-settle")]
    release = program.segments[labels.index("release")]

    assert grip.duration == pytest.approx(0.5)
    assert release.duration == pytest.approx(0.5)
    assert place_settle.duration == pytest.approx(0.18)


@pytest.mark.parametrize("robot_key", ["xarm6", "uf850", "lite6"])
def test_default_pick_place_arm_segments_use_150mm_s_source_timing(robot_key: str):
    config = load_runtime_config(robot_key)
    program = _default_program(robot_key)
    max_speeds_mm_s: list[float] = []
    for seg in program.segments:
        if seg.kind == "gripper":
            continue
        assert seg.kind == "movel"
        samples, _ = seg.samples(program.rate)
        pts_m = np.vstack([np.asarray(seg.pose_start[:3], dtype=np.float64), samples[:, :3]])
        speeds_m_s = np.linalg.norm(np.diff(pts_m, axis=0), axis=1) * float(program.rate)
        max_speeds_mm_s.append(float(np.max(speeds_m_s)) * 1000.0)

    assert max(max_speeds_mm_s) == pytest.approx(config.motion.cartesian_speed_m_s * 1000.0, abs=2.0)


def test_build_program_cartesian_starts_at_home_not_default_fk():
    config = load_runtime_config("xarm6")
    kinematics = _FakeKinematics()
    program = _build_program(config, kinematics)
    assert program.segments
    assert program.segments[0].kind == "movej"
    assert program.segments[0].label == "start"
    home_seg = next(seg for seg in program.segments if seg.kind == "movel" and seg.label == "home->pregrasp")
    assert home_seg.pose_start is not None
    assert np.allclose(home_seg.pose_start[:3], (0.30, 0.00, 0.30), atol=1e-3)
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
    place_xyz = None
    for segment in program.segments:
        if segment.kind == "movel" and segment.label == "place-descend" and segment.pose_end is not None:
            place_xyz = np.asarray(segment.pose_end[:3], dtype=np.float64)
    assert place_xyz is not None
    assert place_xyz[0] == pytest.approx(0.30, abs=1e-6)
    assert place_xyz[1] == pytest.approx(0.30, abs=1e-6)


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
