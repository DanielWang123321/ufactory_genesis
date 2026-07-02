"""Contract tests for the xArm6 grasp-place RL task."""

from __future__ import annotations

import numpy as np
import pytest

from ufactory.frames import base_to_world_pos, world_to_base_pos
from ufactory.gripper_g2 import (
    GRIPPER_G2_OPEN_GAP_M,
    GRIPPER_G2_SIM_CLOSE_DRIVE,
    gripper_g2_gap_m_to_sdk_pos_mm,
    gripper_g2_gap_m_to_sim_drive,
    gripper_g2_sdk_pos_mm_to_gap_m,
    gripper_g2_sim_drive_to_gap_m,
)


def test_gripper_g2_gap_drive_and_sdk_mappings_round_trip():
    assert gripper_g2_gap_m_to_sim_drive(GRIPPER_G2_OPEN_GAP_M) == pytest.approx(0.0)
    assert gripper_g2_gap_m_to_sim_drive(0.0) == pytest.approx(GRIPPER_G2_SIM_CLOSE_DRIVE)
    assert gripper_g2_sim_drive_to_gap_m(0.0) == pytest.approx(GRIPPER_G2_OPEN_GAP_M)
    assert gripper_g2_sim_drive_to_gap_m(GRIPPER_G2_SIM_CLOSE_DRIVE) == pytest.approx(0.0)

    gap = 0.042
    drive = gripper_g2_gap_m_to_sim_drive(gap)
    assert gripper_g2_sim_drive_to_gap_m(drive) == pytest.approx(gap)
    assert gripper_g2_gap_m_to_sdk_pos_mm(gap) == pytest.approx(42.0)
    assert gripper_g2_sdk_pos_mm_to_gap_m(42.0) == pytest.approx(gap)


def test_base_world_translation_contract():
    base = np.array([0.0, 0.0, 0.4])
    base_xyz = np.array([0.30, -0.10, 0.02])
    world_xyz = base_to_world_pos(base_xyz, base)

    np.testing.assert_allclose(world_xyz, [0.30, -0.10, 0.42])
    np.testing.assert_allclose(world_to_base_pos(world_xyz, base), base_xyz)


def test_grasp_place_task_defaults_are_deploy_friendly():
    from examples.xarm6.xarm6_grasp_place_train import get_task_cfgs

    env_cfg, reward_cfg, robot_cfg = get_task_cfgs("xarm6")

    assert env_cfg["num_obs"] == 30
    assert env_cfg["num_actions"] == 7
    assert env_cfg["action_scale"] == pytest.approx(0.01)
    assert env_cfg["max_joint_delta_rad"] == pytest.approx(0.01)
    assert env_cfg["gripper_open_mm"] == pytest.approx(84.0)
    assert env_cfg["gripper_close_mm"] == pytest.approx(0.0)
    assert env_cfg["gripper_delta_mm"] == pytest.approx(4.0)
    assert robot_cfg["base_pos"] == [0.0, 0.0, env_cfg["table_height"]]
    assert "workspace_violation" in reward_cfg
