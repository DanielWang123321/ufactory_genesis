"""Contract tests for the xArm6 pick-place RL task."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ufactory.config import load_runtime_config, resolve_pick_place_object_spec
from ufactory.manipulation.frames import base_to_world_pos, world_to_base_pos
from ufactory.training import build_pick_place_task_configs
from ufactory.grippers.g2 import (
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


def test_pick_place_task_defaults_are_runtime_bound():
    recipe = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "training" / "pick_place_recipe.yaml"
    env_cfg, reward_cfg, robot_cfg = build_pick_place_task_configs("xarm6", recipe_path=recipe)

    assert env_cfg["num_obs"] == 30
    assert env_cfg["num_actions"] == 4
    assert env_cfg["action_scale"] == pytest.approx(0.01)
    assert env_cfg["max_cartesian_delta_m"] == pytest.approx(0.01)
    assert env_cfg["default_ee_pos"] == pytest.approx((0.30, 0.00, 0.30))
    assert env_cfg["gripper_open_mm"] == pytest.approx(84.0)
    assert env_cfg["gripper_close_mm"] == pytest.approx(22.0)
    assert env_cfg["gripper_delta_mm"] == pytest.approx(4.0)
    assert env_cfg["success_hold_steps"] == 5
    assert env_cfg["substeps"] == 8
    assert env_cfg["fixed_demo_layout"] is True
    assert env_cfg["place_phase_reset_frac"] == pytest.approx(0.25)
    assert env_cfg["place_phase_hover_z_m"] == pytest.approx(0.07)
    object_spec = resolve_pick_place_object_spec(load_runtime_config("xarm6"))
    assert env_cfg["obj_size"] == pytest.approx(object_spec.size_m)
    assert env_cfg["obj_mass_kg"] == pytest.approx(object_spec.mass_kg)
    assert env_cfg["runtime_config_sha256"] == load_runtime_config("xarm6", task="pick_place").sha256
    assert env_cfg["fixed_obj_pos"] == pytest.approx((0.30, 0.00, 0.015))
    assert env_cfg["fixed_target_pos"] == pytest.approx((0.30, 0.30, 0.015))
    assert robot_cfg["base_pos"] == [0.30, 0.0, env_cfg["table_height"]]
    assert "keypoints" in reward_cfg
    assert "place_xy" in reward_cfg
    assert "place_z" in reward_cfg
    assert "lower" in reward_cfg
    assert "holding_table" in reward_cfg
    assert "place" not in reward_cfg
    assert "workspace_violation" in reward_cfg


def test_packaging_showcase_uses_shared_object_spec():
    pytest.importorskip("torch")
    from ufactory.manipulation.packaging.scene import make_layout
    from ufactory.manipulation.packaging import packaging_layout

    layout = make_layout()
    grasp_config = load_runtime_config("xarm6")
    packaging_config = load_runtime_config("xarm6", task="packaging_showcase")
    task_layout = packaging_layout(packaging_config)
    object_spec = resolve_pick_place_object_spec(grasp_config)
    assert layout.obj_size == pytest.approx(object_spec.size_m)
    assert layout.obj_mass_kg == pytest.approx(object_spec.mass_kg)
    assert task_layout.target_position_m[:2] == pytest.approx(
        grasp_config.task.parameters["fixed_target_position_m"][:2]
    )
    assert task_layout.box_center_xy_m == pytest.approx(task_layout.target_position_m[:2])


@pytest.mark.integration
def test_packaging_scene_builds_headless_with_shared_object_spec():
    from ufactory.manipulation.packaging.scene import build_packaging_scene
    from ufactory.simulation import GenesisRuntimeManager
    from ufactory.visualization.glb import enable_glb_pbr_surfaces

    config = load_runtime_config("xarm6")
    enable_glb_pbr_surfaces()
    with GenesisRuntimeManager(config.simulation):
        scene, _robot, block, layout = build_packaging_scene(show_viewer=False)
        position = block.get_pos()[0].detach().cpu().numpy()
        assert np.all(np.isfinite(position))
        assert float(position[2]) == pytest.approx(layout.table_top_z + layout.obj_size[2] / 2.0, abs=1e-4)
        assert layout.obj_mass_kg == pytest.approx(resolve_pick_place_object_spec(config).mass_kg)
        del scene
