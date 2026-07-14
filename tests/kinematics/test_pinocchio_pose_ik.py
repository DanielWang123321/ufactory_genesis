"""Full-pose Pinocchio IK regressions for the fixed gripper-down contract."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ufactory.config import RepositoryAssetStore, load_runtime_config
from ufactory.kinematics.orientation import GRIPPER_DOWN_QUAT_XYZW
from ufactory.safety.adapters.pinocchio import PinocchioKinematicsBackend
from ufactory.trajectory.scene import dry_heights


ROBOTS = ("xarm5", "xarm6", "xarm7", "uf850", "lite6")


def _kinematics(robot: str):
    pytest.importorskip("pinocchio")
    config = load_runtime_config(robot)
    urdf = RepositoryAssetStore.discover().require(f"{config.robot.assets_dir}/{config.robot.urdf}")
    passive = {config.gripper.drive_joint: config.gripper.open_drive}
    return config, PinocchioKinematicsBackend(
        urdf,
        joint_names=config.robot.joint_names,
        ee_link=config.robot.ee_link,
        passive_joint_positions=passive,
    )


@pytest.mark.parametrize("robot", ROBOTS)
def test_gripper_down_key_poses_converge_for_every_robot(robot: str):
    config, kinematics = _kinematics(robot)
    params = config.task.parameters
    heights = dry_heights(config.robot.key)
    obj = params["fixed_object_position_m"]
    target = params["fixed_target_position_m"]
    xyz_targets = (
        (obj[0], obj[1], heights.home_pos_base[2]),
        (obj[0], obj[1], heights.pre_grasp_link6_z),
        (obj[0], obj[1], heights.grasp_link6_z),
        (target[0], target[1], heights.lift_link6_z),
        (target[0], target[1], heights.grasp_link6_z),
    )
    seed = np.asarray(config.arm.default_qpos_rad, dtype=np.float64)
    desired_quaternion = np.asarray(GRIPPER_DOWN_QUAT_XYZW, dtype=np.float64)
    for xyz in xyz_targets:
        target_pose = np.concatenate((np.asarray(xyz, dtype=np.float64), desired_quaternion))
        seed = kinematics.inverse(target_pose, seed)
        actual = kinematics.forward(seed)
        assert np.linalg.norm(actual[:3] - target_pose[:3]) <= 1e-6
        actual_quaternion = actual[3:7] / np.linalg.norm(actual[3:7])
        orientation_error = 2.0 * np.arccos(
            np.clip(abs(float(np.dot(actual_quaternion, desired_quaternion))), 0.0, 1.0)
        )
        assert orientation_error <= 1e-4


@pytest.mark.parametrize(
    "pose,match",
    [
        (np.zeros(4), "xyz or xyz\\+quaternion"),
        (np.asarray([0.3, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0]), "non-zero"),
        (np.asarray([0.3, 0.0, np.nan]), "finite"),
    ],
)
def test_pose_ik_rejects_invalid_pose_contract(pose: np.ndarray, match: str):
    config, kinematics = _kinematics("xarm6")
    with pytest.raises(ValueError, match=match):
        kinematics.inverse(pose, np.asarray(config.arm.default_qpos_rad, dtype=np.float64))


def test_pose_ik_uses_the_fixed_solver_contract():
    _config, kinematics = _kinematics("xarm6")

    assert kinematics.max_iterations == 120
    assert kinematics.tolerance_m == pytest.approx(1e-7)
    assert kinematics.orientation_tolerance_rad == pytest.approx(1e-5)
    assert kinematics.damping == pytest.approx(1e-6)
    assert kinematics.max_step_rad == pytest.approx(0.05)


def test_three_element_target_keeps_position_only_compatibility():
    config, kinematics = _kinematics("xarm6")
    target = np.asarray((0.30, 0.00, 0.30), dtype=np.float64)

    solved = kinematics.inverse(target, np.asarray(config.arm.default_qpos_rad, dtype=np.float64))

    assert np.linalg.norm(kinematics.forward(solved)[:3] - target) <= 1e-6


def test_genesis_ik_quaternion_is_converted_from_shared_xyzw_contract():
    pytest.importorskip("genesis")
    from ufactory.kinematics.genesis import down_quat

    quaternion_wxyz = down_quat(device=torch.device("cpu"), dtype=torch.float64)

    np.testing.assert_allclose(quaternion_wxyz.numpy(), [[0.0, 1.0, 0.0, 0.0]], atol=1e-12)


def test_uf850_gripper_down_shadow_stays_below_joint_acceleration_limit():
    from ufactory.cli.grasp_place import _build_program, _model_and_hashes
    from ufactory.trajectory.preflight import create_safety_gate

    config, kinematics = _kinematics("uf850")
    urdf, _urdf_hash, calibration_hash = _model_and_hashes(config, None, None)
    program = _build_program(config, kinematics)
    gate = create_safety_gate(
        config,
        kinematics=kinematics,
        collision=None,
        calibration_sha256=calibration_hash,
        scene_sha256="s" * 64,
        urdf_path=urdf,
    )
    shadow, _stages = gate.shadow_joint_stream(program)
    velocity = np.diff(shadow, axis=0) * program.rate
    acceleration = np.diff(velocity, axis=0) * program.rate
    assert float(np.max(np.abs(acceleration), initial=0.0)) <= config.motion.joint_acceleration_rad_s2
