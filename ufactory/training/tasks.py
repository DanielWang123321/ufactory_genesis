"""Resolved task configuration builders shared by RL entry points and tests."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from ufactory.config import load_runtime_config, resolve_pick_place_object_spec
from ufactory.robots.paths import robot_urdf
from ufactory.training.recipe import load_training_recipe

PICK_PLACE_RL_ROBOTS = ("xarm6", "xarm6_1305")


def build_train_config(
    recipe_path: str | Path,
    *,
    experiment_name: str,
    max_iterations: int,
) -> dict[str, Any]:
    """Resolve PPO settings with command-line run settings taking precedence."""

    train_cfg = deepcopy(load_training_recipe(recipe_path)["train"])
    train_cfg["runner"]["experiment_name"] = experiment_name
    train_cfg["runner"]["max_iterations"] = int(max_iterations)
    return train_cfg


def build_pick_place_task_configs(
    robot: str,
    *,
    recipe_path: str | Path,
    runtime_config_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve the retained xArm6 + Gripper G2 pick-place RL task."""

    if robot not in PICK_PLACE_RL_ROBOTS:
        raise ValueError("the retained v0.2.7 pick-place RL environment supports only xArm6 + Gripper G2")
    config = load_runtime_config(robot, task="pick_place", config_path=runtime_config_path)
    if config.gripper is None or config.gripper.adapter != "g2":
        raise ValueError("the pick-place RL environment requires Gripper G2")
    params = config.task.parameters
    object_spec = resolve_pick_place_object_spec(config)
    recipe = load_training_recipe(recipe_path)
    env_cfg = deepcopy(recipe["environment"])
    env_cfg.update(
        {
            "runtime_config_sha256": config.sha256,
            "episode_length_s": float(params["episode_length_s"]),
            "ctrl_dt": float(params["control_dt_s"]),
            "table_height": float(params["table_height_m"]),
            "default_ee_pos": list(params["default_ee_position_m"]),
            "workspace_lower": list(params["workspace_lower_m"]),
            "workspace_upper": list(params["workspace_upper_m"]),
            "grasp_center_offset_z": float(params["grasp_center_offset_z_m"]),
            "lift_height_m": float(params["lift_height_m"]),
            "place_success_dist_m": float(params["place_success_distance_m"]),
            "success_hold_steps": int(params["success_hold_steps"]),
            "substeps": int(params["substeps"]),
            "gripper_open_mm": config.gripper.open_gap_m * 1000.0,
            # Grasp preload target (~22 mm), not the hardware dead-closed gap (0 mm).
            "gripper_close_mm": float(params["grasp_gap_m"]) * 1000.0,
            "obj_size": list(object_spec.size_m),
            "obj_mass_kg": object_spec.mass_kg,
            "fixed_obj_pos": list(params["fixed_object_position_m"]),
            "fixed_target_pos": list(params["fixed_target_position_m"]),
            "obj_spawn_lower": list(params["object_spawn_lower_m"]),
            "obj_spawn_upper": list(params["object_spawn_upper_m"]),
            "target_spawn_lower": list(params["target_spawn_lower_m"]),
            "target_spawn_upper": list(params["target_spawn_upper_m"]),
            # Recipe may set fixed_demo_layout; default true matches trajectory demo poses.
            "fixed_demo_layout": bool(env_cfg.get("fixed_demo_layout", True)),
            "place_phase_reset_frac": float(env_cfg.get("place_phase_reset_frac", 0.25)),
            "place_phase_hover_z_m": float(env_cfg.get("place_phase_hover_z_m", 0.07)),
        }
    )
    gripper = config.gripper
    robot_cfg = {
        "urdf_path": robot_urdf(config.robot.key, config.robot.urdf),
        # Match trajectory pick-place scene (DEFAULT_ROBOT_BASE_POS xy = 0.30, 0.00).
        "base_pos": [0.30, 0.0, env_cfg["table_height"]],
        "ik_link_name": config.robot.ee_link,
        "gripper_link_names": list(gripper.finger_link_names),
        "arm_joint_names": list(config.robot.joint_names),
        "gripper_joint_name": gripper.drive_joint,
        "default_qpos": list(config.arm.default_qpos_rad),
        "default_gripper_pos": gripper.open_drive,
        "kp": list(config.arm.kp),
        "kv": list(config.arm.kv),
        "force_lower": list(config.arm.force_lower_nm),
        "force_upper": list(config.arm.force_upper_nm),
        "gripper_kp": gripper.kp,
        "gripper_kv": gripper.kv,
        "gripper_force_lower": gripper.force_lower_n,
        "gripper_force_upper": gripper.force_upper_n,
        "all_gripper_joint_names": list(gripper.all_joint_names),
        "gripper_damping": gripper.damping,
        "gripper_frictionloss": gripper.frictionloss,
        "collision_monitor_links": ["link3", "link4", "link5"],
    }
    return env_cfg, deepcopy(recipe["reward"]), robot_cfg
