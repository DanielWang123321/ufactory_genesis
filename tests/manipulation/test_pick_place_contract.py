"""Contract tests for the xArm6 pick-place RL task."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ufactory.config import load_runtime_config, resolve_pick_place_object_spec
from ufactory.manipulation.frames import base_to_world_pos, world_to_base_pos
from ufactory.simulation import G2_PHYSICS_PROFILE
from ufactory.training import build_pick_place_task_configs, load_pick_place_rl_runtime_config


def test_base_world_translation_contract():
    base = np.array([0.0, 0.0, 0.4])
    base_xyz = np.array([0.30, -0.10, 0.02])
    world_xyz = base_to_world_pos(base_xyz, base)

    np.testing.assert_allclose(world_xyz, [0.30, -0.10, 0.42])
    np.testing.assert_allclose(world_to_base_pos(world_xyz, base), base_xyz)


def test_pick_place_task_defaults_are_runtime_bound():
    pytest.importorskip("trimesh")
    recipe = Path(__file__).resolve().parents[2] / "examples" / "rl" / "pick_place" / "recipe.yaml"
    env_cfg, reward_cfg, robot_cfg = build_pick_place_task_configs("xarm6", recipe_path=recipe)
    assert env_cfg["num_obs"] == 70
    assert env_cfg["num_actions"] == 4
    assert env_cfg["action_scale"] == pytest.approx(0.005)
    assert env_cfg["gap_action_scale"] == pytest.approx(0.003)
    assert env_cfg["success_hold_steps"] == 10
    assert env_cfg["substeps"] == 32
    assert env_cfg["noslip_iterations"] == 0
    assert env_cfg["physics_profile"] == G2_PHYSICS_PROFILE
    config = load_pick_place_rl_runtime_config("xarm6")
    assert env_cfg["obj_mass_kg"] == config.task.parameters["object_mass_kg"]
    assert env_cfg["runtime_config_sha256"] == config.sha256
    assert robot_cfg == {"gripper_damping": config.gripper.damping, "gripper_frictionloss": config.gripper.frictionloss}
    assert "release_potential" in reward_cfg


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
