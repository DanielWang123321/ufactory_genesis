"""Configuration-driven UFACTORY packaging showcase physics execution.

Build a robot-specific movable-gripper scene, physically grasp a red block,
and drop it into the configured open cardboard box.

Usage:
    export NUMBA_CACHE_DIR=~/.cache/numba
    # Windows PowerShell: $env:NUMBA_CACHE_DIR="$env:USERPROFILE\\.cache\\numba"
    python scripts/generate_showcase_textures.py   # first time only
    ufactory-packaging-showcase --robot xarm6 --mode sim
    ufactory-packaging-showcase --robot lite6 --mode sim --cycles 3
    # Without a Genesis-supported GPU:
    ufactory-packaging-showcase --robot xarm6 --mode sim --backend cpu
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass, replace
import math
import time
from pathlib import Path

import numpy as np
import torch

import genesis as gs
from genesis.utils.geom import transform_quat_by_quat, xyz_to_quat

from ufactory.config import load_runtime_config
from ufactory.grippers import create_gripper_adapter
from ufactory.manipulation.packaging.core import build_packaging_program, packaging_layout
from ufactory.manipulation.packaging.scene import (
    FINGER_FRICTION_MU,
    HOME_RPY_DEG,
    ROBOT_BASE_YAW_DEG,
    build_packaging_scene,
    make_layout,
)
from ufactory.robots.runtime import get_robot_runtime_profile, robot_runtime_cli_choices
from ufactory.simulation import GenesisRuntimeManager, override_simulation_backend
from ufactory.simulation.compat import require_genesis_capabilities
from ufactory.trajectory.ik import compile_cartesian_program_to_joint_stream
from ufactory.trajectory.scene import TrajSceneContext
from ufactory.trajectory.sim_executor import SimReport, replay_sim
from ufactory.visualization.viewer import start_deferred_viewer

GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = 0.85
GRIPPER_OPEN_GAP_M = 0.084  # drive=0 → ~84 mm two-finger gap
GRIPPER_GAP_CALIBRATION_OFFSET_M = 0.0053  # linear model under-closes vs G2 pad kinematics
GRASP_SQUEEZE_GAP_MARGIN = 0.0  # flush with block width; calibration offset handles pad error
SIM_DT = 0.02
SETTLE_STEPS = 40
# The shared trajectory keeps the production/dry-run G2 timing. Genesis needs
# the open target to change in one control tick for this high-clearance drop;
# streaming intermediate targets lets the curved linkage pads sweep across the
# cube while it starts falling and adds a large lateral impulse. The joints
# still open through their configured PD/force limits after the target step.
# This override is limited to the showcase and leaves every non-Genesis path
# unchanged.
SIM_RELEASE_DURATION_S = SIM_DT
# Ease most of the grasp preload at the box center before commanding full
# open. This remains below the 30 mm cube width in planned gap coordinates,
# while the physical linkage starts opening visibly before gravity takes over.
SIM_PRE_RELEASE_RELAX_GAP_M = 0.029
SIM_PRE_RELEASE_RELAX_DURATION_S = 0.2
ALL_GRIPPER_JOINTS = (
    "drive_joint",
    "left_finger_joint",
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_finger_joint",
    "right_inner_knuckle_joint",
)


@dataclass
class ShowcaseRobotCtx:
    ik_link: object
    left_finger: object
    right_finger: object
    arm_dof_idx: list[int]
    gripper_dof_idx: list[int]
    all_gripper_dof_idx: list[int]
    down_quat: torch.Tensor
    home_pos: list[float]
    home_qpos_saved: torch.Tensor
    finger_z_offset: float = 0.0
    grasp_drive: float = GRIPPER_CLOSE
    gripper_open_drive: float = GRIPPER_OPEN
    gripper_open_dof_idx: list[int] | None = None
    gripper_initial_open_dof_idx: list[int] | None = None
    gripper_initial_open_clearance_m: float = 0.0
    home_settle_steps: int = SETTLE_STEPS


def gripper_drive_for_gap(gap_m: float) -> float:
    """Map desired two-finger gap (m) to drive_joint command."""
    gap_m = max(0.0, min(GRIPPER_OPEN_GAP_M, gap_m))
    return GRIPPER_CLOSE * (1.0 - gap_m / GRIPPER_OPEN_GAP_M)


def grasp_gripper_drive(obj_size: tuple[float, float, float]) -> float:
    """Partial close target for a block grasped with gripper pointing down (Y axis)."""
    obj_width = obj_size[1]
    target_gap = max(0.0, obj_width + GRASP_SQUEEZE_GAP_MARGIN - GRIPPER_GAP_CALIBRATION_OFFSET_M)
    return gripper_drive_for_gap(target_gap)


def collect_gripper_snapshot(robot) -> dict:
    """All gripper joint q/qd plus knuckle link poses — linkage whip diagnostics."""
    joint_map = {j.name.split("/")[-1]: j for j in robot.joints}
    joints: dict[str, dict[str, float]] = {}
    drive_q: float | None = None
    max_qd = 0.0
    max_mimic_err = 0.0
    for name in ALL_GRIPPER_JOINTS:
        joint = joint_map.get(name)
        if joint is None:
            continue
        idx = joint.dofs_idx_local[0]
        q = robot.get_dofs_position()[0, idx].item()
        qd = robot.get_dofs_velocity()[0, idx].item()
        joints[name] = {"q": q, "qd": qd}
        max_qd = max(max_qd, abs(qd))
        if name == "drive_joint":
            drive_q = q
    if drive_q is not None:
        for name, state in joints.items():
            if name != "drive_joint":
                max_mimic_err = max(max_mimic_err, abs(state["q"] - drive_q))
    knuckles: dict[str, list[float]] = {}
    for link_name in (
        "left_outer_knuckle",
        "left_inner_knuckle",
        "right_outer_knuckle",
        "right_inner_knuckle",
    ):
        pos = robot.get_link(link_name).get_pos()[0].cpu().tolist()
        knuckles[link_name] = pos
    return {
        "joints": joints,
        "max_qd": max_qd,
        "max_mimic_err": max_mimic_err,
        "knuckles": knuckles,
    }


def collect_pose_snapshot(robot, layout) -> dict:
    """Return link6 / finger poses and arm qpos for keyframe metadata."""
    runtime = get_robot_runtime_profile("xarm6")
    ik_link = robot.get_link(runtime.arm.ee_link)
    left_finger = robot.get_link("left_finger")
    right_finger = robot.get_link("right_finger")
    arm_dof_idx = [robot.get_joint(name).dofs_idx_local[0] for name in runtime.arm.joint_names]
    gripper_dof_idx = [robot.get_joint("drive_joint").dofs_idx_local[0]]

    link6 = ik_link.get_pos()[0].cpu().tolist()
    left = left_finger.get_pos()[0].cpu().tolist()
    right = right_finger.get_pos()[0].cpu().tolist()
    finger_center_z = (left[2] + right[2]) / 2
    finger_y_gap_mm = abs(left[1] - right[1]) * 1000.0
    arm_q = robot.get_dofs_position()[0, arm_dof_idx].cpu().tolist()
    grip_q = robot.get_dofs_position()[0, gripper_dof_idx[0]].item()
    table_z = layout.table_top_z

    return {
        "link6_pos": link6,
        "left_finger_pos": left,
        "right_finger_pos": right,
        "finger_center_z": finger_center_z,
        "finger_y_gap_mm": finger_y_gap_mm,
        "table_top_z": table_z,
        "link6_above_table_mm": (link6[2] - table_z) * 1000.0,
        "finger_above_table_mm": (finger_center_z - table_z) * 1000.0,
        "arm_qpos_deg": [math.degrees(q) for q in arm_q],
        "gripper_q": grip_q,
        "gripper_detail": collect_gripper_snapshot(robot),
    }


def _scale_steps(steps: int, speed: float) -> int:
    return max(1, int(round(steps / max(0.25, speed))))


def _world_home(
    layout,
    *,
    cartesian_xy_offset_m: tuple[float, float] | None = None,
) -> list[float]:
    rx, ry = layout.robot_xy
    bx, by, bz = layout.home_position_base
    offset_x, offset_y = (
        layout.simulation_grasp_center_compensation_xy if cartesian_xy_offset_m is None else cartesian_xy_offset_m
    )
    bx += offset_x
    by += offset_y
    yaw = math.radians(ROBOT_BASE_YAW_DEG)
    c, s = math.cos(yaw), math.sin(yaw)
    wx = bx * c - by * s
    wy = bx * s + by * c
    return [rx + wx, ry + wy, layout.table_top_z + bz]


def _world_down_quat() -> torch.Tensor:
    """TCP RPY in base frame → world quat (accounts for ROBOT_BASE_YAW_DEG)."""
    base_yaw_quat = xyz_to_quat(
        torch.tensor([[0.0, 0.0, ROBOT_BASE_YAW_DEG]], device=gs.device, dtype=gs.tc_float),
        rpy=True,
        degrees=True,
    )
    tcp_base_quat = xyz_to_quat(
        torch.tensor([list(HOME_RPY_DEG)], device=gs.device, dtype=gs.tc_float),
        rpy=True,
        degrees=True,
    )
    return transform_quat_by_quat(tcp_base_quat, base_yaw_quat)


def _setup_robot(robot, layout, config, *, arm_kp_scale: float | None = None):
    gripper = config.gripper
    if gripper is None:
        raise RuntimeError(f"{config.robot.key} packaging showcase requires a configured gripper")
    ik_link = robot.get_link(config.robot.ee_link)
    left_finger = robot.get_link(gripper.finger_link_names[0])
    right_finger = robot.get_link(gripper.finger_link_names[1])
    left_finger.set_friction(float(FINGER_FRICTION_MU))
    right_finger.set_friction(float(FINGER_FRICTION_MU))

    arm_dof_idx = [robot.get_joint(name).dofs_idx_local[0] for name in config.robot.joint_names]
    gripper_dof_idx = [robot.get_joint(gripper.drive_joint).dofs_idx_local[0]]
    all_gripper_dof_idx = [robot.get_joint(name).dofs_idx_local[0] for name in gripper.all_joint_names]
    opening_gripper_dof_idx = (
        [gripper_dof_idx[0], all_gripper_dof_idx[3]] if gripper.adapter == "g2" else list(gripper_dof_idx)
    )
    applied_arm_kp_scale = layout.simulation_arm_kp_scale if arm_kp_scale is None else float(arm_kp_scale)
    robot.set_dofs_kp(
        torch.tensor(
            np.asarray(config.arm.kp) * applied_arm_kp_scale,
            device=gs.device,
            dtype=gs.tc_float,
        ),
        arm_dof_idx,
    )
    robot.set_dofs_kv(
        torch.tensor(config.arm.kv, device=gs.device, dtype=gs.tc_float),
        arm_dof_idx,
    )
    robot.set_dofs_force_range(
        torch.tensor(config.arm.force_lower_nm, device=gs.device, dtype=gs.tc_float),
        torch.tensor(config.arm.force_upper_nm, device=gs.device, dtype=gs.tc_float),
        arm_dof_idx,
    )
    robot.set_dofs_kp(
        torch.full((len(opening_gripper_dof_idx),), gripper.kp, device=gs.device, dtype=gs.tc_float),
        opening_gripper_dof_idx,
    )
    robot.set_dofs_kv(
        torch.full((len(opening_gripper_dof_idx),), gripper.kv, device=gs.device, dtype=gs.tc_float),
        opening_gripper_dof_idx,
    )
    robot.set_dofs_force_range(
        torch.full((len(opening_gripper_dof_idx),), gripper.force_lower_n, device=gs.device, dtype=gs.tc_float),
        torch.full((len(opening_gripper_dof_idx),), gripper.force_upper_n, device=gs.device, dtype=gs.tc_float),
        opening_gripper_dof_idx,
    )
    n_grip = len(all_gripper_dof_idx)
    robot.set_dofs_damping(
        torch.full((n_grip,), gripper.damping, device=gs.device, dtype=gs.tc_float),
        all_gripper_dof_idx,
    )
    robot.set_dofs_frictionloss(
        torch.full((n_grip,), gripper.frictionloss, device=gs.device, dtype=gs.tc_float),
        all_gripper_dof_idx,
    )

    down_quat = _world_down_quat()

    return ik_link, left_finger, right_finger, arm_dof_idx, gripper_dof_idx, all_gripper_dof_idx, down_quat


def _init_home_qpos(
    robot,
    ik_link,
    arm_dof_idx,
    all_gripper_dof_idx,
    down_quat,
    home_pos,
    *,
    gripper_open_drive: float,
):
    home_link6_pos = torch.tensor([home_pos], device=gs.device, dtype=gs.tc_float)
    init_qpos = torch.zeros(1, robot.n_dofs, device=gs.device, dtype=gs.tc_float)
    home_qpos_result = robot.inverse_kinematics(
        link=ik_link,
        pos=home_link6_pos,
        quat=down_quat,
        dofs_idx_local=arm_dof_idx,
        init_qpos=init_qpos,
    )
    for i, idx in enumerate(arm_dof_idx):
        init_qpos[:, idx] = home_qpos_result[0, arm_dof_idx[i]]
    for idx in all_gripper_dof_idx:
        init_qpos[:, idx] = gripper_open_drive
    robot.set_qpos(init_qpos)
    return init_qpos.clone()


def _measure_finger_offset(ik_link, left_finger, right_finger):
    link6_pos = ik_link.get_pos()[0]
    fc_pos = ((left_finger.get_pos() + right_finger.get_pos()) / 2)[0]
    return (link6_pos[2] - fc_pos[2]).item()


def init_showcase_robot(
    robot,
    layout,
    scene,
    *,
    runtime_config=None,
    cartesian_xy_offset_m: tuple[float, float] | None = None,
    arm_kp_scale: float | None = None,
) -> ShowcaseRobotCtx:
    """Apply PD gains, set home qpos via IK, and prime finger geometry."""
    config = runtime_config or load_runtime_config(layout.robot_key, task="packaging_showcase")
    task_layout = packaging_layout(config)
    gripper = config.gripper
    if gripper is None:
        raise RuntimeError(f"{config.robot.key} packaging showcase requires a configured gripper")
    ik_link, left_finger, right_finger, arm_dof_idx, gripper_dof_idx, all_gripper_dof_idx, down_quat = _setup_robot(
        robot,
        layout,
        config,
        arm_kp_scale=arm_kp_scale,
    )
    home_pos = _world_home(layout, cartesian_xy_offset_m=cartesian_xy_offset_m)
    home_qpos_saved = _init_home_qpos(
        robot,
        ik_link,
        arm_dof_idx,
        all_gripper_dof_idx,
        down_quat,
        home_pos,
        gripper_open_drive=float(gripper.open_drive),
    )
    scene.step()
    finger_z_offset = _measure_finger_offset(ik_link, left_finger, right_finger)
    grasp_drive = create_gripper_adapter(gripper).gap_to_drive(task_layout.grasp_gap_m)
    is_g2 = gripper.adapter == "g2"
    return ShowcaseRobotCtx(
        ik_link=ik_link,
        left_finger=left_finger,
        right_finger=right_finger,
        arm_dof_idx=arm_dof_idx,
        gripper_dof_idx=gripper_dof_idx,
        all_gripper_dof_idx=all_gripper_dof_idx,
        down_quat=down_quat,
        home_pos=home_pos,
        home_qpos_saved=home_qpos_saved,
        finger_z_offset=finger_z_offset,
        grasp_drive=grasp_drive,
        gripper_open_drive=float(gripper.open_drive),
        gripper_open_dof_idx=([all_gripper_dof_idx[0], all_gripper_dof_idx[3]] if is_g2 else list(gripper_dof_idx)),
        gripper_initial_open_dof_idx=list(all_gripper_dof_idx) if is_g2 else [],
        gripper_initial_open_clearance_m=0.003 if is_g2 else 0.0,
        home_settle_steps=max(1, int(round(task_layout.simulation_home_settle_s * config.motion.rate_hz))),
    )


def hold_robot_home(robot, scene, ctx: ShowcaseRobotCtx, *, steps: int = 1) -> None:
    target_qpos = ctx.home_qpos_saved
    gripper_open = torch.full((1, 1), ctx.gripper_open_drive, device=gs.device, dtype=gs.tc_float)
    for _ in range(steps):
        robot.control_dofs_position(target_qpos[:, ctx.arm_dof_idx], ctx.arm_dof_idx)
        robot.control_dofs_position(gripper_open, ctx.gripper_dof_idx)
        scene.step()


def _reset_block(block, layout) -> None:
    pos = torch.tensor(
        [[layout.obj_spawn_xy[0], layout.obj_spawn_xy[1], layout.table_top_z + layout.obj_spawn_center_z]],
        device=gs.device,
        dtype=gs.tc_float,
    )
    block.set_pos(pos, zero_velocity=True)
    block.set_quat(
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=gs.device, dtype=gs.tc_float),
        zero_velocity=True,
    )


def prepare_packaging_cycle(
    scene,
    robot,
    block,
    layout,
    ctx: ShowcaseRobotCtx,
    *,
    speed: float = 1.0,
) -> None:
    """Restore the cube and passive release joint, then settle at home."""
    release_only_dof_idx = [idx for idx in ctx.all_gripper_dof_idx if idx not in ctx.gripper_dof_idx]
    robot.control_dofs_force(
        torch.zeros((1, len(release_only_dof_idx)), device=gs.device, dtype=gs.tc_float),
        release_only_dof_idx,
    )
    _reset_block(block, layout)
    hold_robot_home(robot, scene, ctx, steps=_scale_steps(getattr(ctx, "home_settle_steps", SETTLE_STEPS), speed))


def _with_sim_release_timing(program, *, speed: float, task_layout=None):
    """Return a copy with YAML-configured packaging-simulation release timing."""
    if task_layout is None:
        task_layout = packaging_layout(load_runtime_config("xarm6", task="packaging_showcase"))
    speed_scale = max(0.25, float(speed))
    duration = task_layout.simulation_release_duration_s / speed_scale
    relax_duration = task_layout.simulation_pre_release_relax_duration_s / speed_scale
    relax_gap = task_layout.simulation_pre_release_relax_gap_m
    segments = []
    for segment in program.segments:
        if segment.label == "pre-release-settle" and relax_duration > 0.0:
            # Preserve the configured stationary hold so the cube can settle
            # after the long transit.  Then ease the clamp preload in a
            # separate Genesis-only phase before the full release target.
            segments.append(segment)
            tuned = replace(
                segment,
                label="pre-release-relax",
                gap_end=relax_gap,
                duration=relax_duration,
            )
        elif segment.label == "release":
            tuned = replace(
                segment,
                gap_start=relax_gap if relax_duration > 0.0 else segment.gap_start,
                duration=duration,
            )
        else:
            segments.append(segment)
            continue
        _, samples_count = tuned.samples(program.rate)
        segments.append(replace(tuned, samples_count=samples_count))
    return replace(program, segments=segments)


def _speed_config(robot_key: str, speed: float, config_path: Path | None = None):
    """Load the versioned task with a simulation-only playback multiplier."""
    mult = max(0.25, float(speed))
    base = load_runtime_config(robot_key, task="packaging_showcase", config_path=config_path)
    p = base.task.parameters
    return load_runtime_config(
        robot_key,
        task="packaging_showcase",
        config_path=config_path,
        overrides={
            "motion.joint_speed_rad_s": base.motion.joint_speed_rad_s * mult,
            "motion.joint_acceleration_rad_s2": base.motion.joint_acceleration_rad_s2 * mult,
            "motion.cartesian_speed_m_s": base.motion.cartesian_speed_m_s * mult,
            "motion.cartesian_acceleration_m_s2": base.motion.cartesian_acceleration_m_s2 * mult,
            "motion.gripper_duration_s": base.motion.gripper_duration_s / mult,
            "task.parameters.grasp_settle_s": float(p["grasp_settle_s"]) / mult,
            "task.parameters.pre_release_settle_s": float(p["pre_release_settle_s"]) / mult,
            "task.parameters.post_release_settle_s": float(p["post_release_settle_s"]) / mult,
        },
    )


def _trajectory_context(
    scene,
    robot,
    block,
    layout,
    ctx: ShowcaseRobotCtx,
    config,
    *,
    cartesian_xy_offset_m: tuple[float, float] | None = None,
) -> TrajSceneContext:
    task_layout = packaging_layout(config)
    runtime = get_robot_runtime_profile(config.robot.key)
    gripper = runtime.gripper
    if gripper is None:
        raise RuntimeError(f"{config.robot.key} packaging showcase requires a configured gripper")
    left = ctx.left_finger.get_pos()[0]
    right = ctx.right_finger.get_pos()[0]
    finger_span = max(abs(float(left[0] - right[0])), abs(float(left[1] - right[1])))
    offset_x, offset_y = (
        task_layout.simulation_grasp_center_compensation_xy_m
        if cartesian_xy_offset_m is None
        else cartesian_xy_offset_m
    )
    home_pos_base = list(task_layout.home_position_m)
    home_pos_base[0] += offset_x
    home_pos_base[1] += offset_y
    return TrajSceneContext(
        scene=scene,
        robot=robot,
        obj=block,
        target_marker=None,
        ik_link=ctx.ik_link,
        left_finger=ctx.left_finger,
        right_finger=ctx.right_finger,
        arm_dof_idx=ctx.arm_dof_idx,
        gripper_dof_idx=ctx.gripper_dof_idx,
        all_gripper_dof_idx=ctx.all_gripper_dof_idx,
        down_quat=ctx.down_quat,
        home_qpos=ctx.home_qpos_saved[0].detach().cpu().numpy().astype(np.float64),
        base_pos_world=(layout.robot_xy[0], layout.robot_xy[1], layout.table_top_z),
        base_yaw_rad=math.radians(ROBOT_BASE_YAW_DEG),
        gripper_open_dof_idx=list(ctx.gripper_open_dof_idx or ctx.gripper_dof_idx),
        gripper_initial_open_dof_idx=list(ctx.gripper_initial_open_dof_idx or ()),
        gripper_initial_open_clearance_m=ctx.gripper_initial_open_clearance_m,
        gripper_binary_commands=gripper.family == "lite6",
        visual_model="glb",
        finger_z_offset=ctx.finger_z_offset,
        finger_y_gap=finger_span,
        grasp_link6_z=task_layout.grasp_link6_z_m,
        pre_grasp_link6_z=task_layout.grasp_link6_z_m + task_layout.pregrasp_clearance_m,
        lift_link6_z=task_layout.release_link6_z_m,
        robot_key=config.robot.key,
        gripper=gripper,
        home_pos_base=home_pos_base,
        table_height=layout.table_top_z,
        obj_xy=task_layout.object_position_m[:2],
        place_xy=task_layout.target_position_m[:2],
        obj_size=task_layout.object_size_m,
        obj_mass_kg=task_layout.object_mass_kg,
        obj_pos_base=task_layout.object_position_m,
        place_pos_base=task_layout.target_position_m,
    )


def run_pick_place_cycle(
    scene,
    robot,
    block,
    layout,
    *,
    speed: float = 1.0,
    ctx: ShowcaseRobotCtx | None = None,
    stop_after_phase0: bool = False,
    capture_hook: callable | None = None,
    config_path: Path | None = None,
    robot_key: str | None = None,
    executor: str = "servo_cartesian",
) -> SimReport | None:
    """Replay the shared physical trajectory without kinematic object carry."""
    selected_robot = robot_key or layout.robot_key
    if ctx is None:
        base_config = load_runtime_config(selected_robot, task="packaging_showcase", config_path=config_path)
        ctx = init_showcase_robot(robot, layout, scene, runtime_config=base_config)
    if stop_after_phase0:
        hold_robot_home(robot, scene, ctx, steps=_scale_steps(SETTLE_STEPS, speed))
        if capture_hook is not None:
            capture_hook("Home")
        return None

    config = _speed_config(selected_robot, speed, config_path)
    task_layout = packaging_layout(config)
    traj_ctx = _trajectory_context(scene, robot, block, layout, ctx, config)
    sim_offset = task_layout.simulation_grasp_center_compensation_xy_m
    if executor == "servo_j":
        q_home = traj_ctx.home_qpos[traj_ctx.arm_dof_idx]
        source = build_packaging_program(
            config,
            q_home=q_home,
            cartesian_xy_offset_m=sim_offset,
            place_xy_offset_m=(0.0, 0.0),
        )
        program = compile_cartesian_program_to_joint_stream(source, traj_ctx)
    elif executor == "servo_cartesian":
        program = build_packaging_program(
            config,
            cartesian_xy_offset_m=sim_offset,
            place_xy_offset_m=(0.0, 0.0),
        )
    else:
        raise ValueError(f"unknown packaging executor: {executor}")
    program = _with_sim_release_timing(program, speed=speed, task_layout=task_layout)

    def _on_phase(status, segment) -> None:
        obj = tuple(value / 1000.0 for value in status.obj_pos_mm)
        link6 = tuple(value / 1000.0 for value in status.link6_mm)
        print(
            f"  [{status.label:20s}] link6=[{link6[0]:.3f}, {link6[1]:.3f}, {link6[2]:.3f}]  "
            f"block=[{obj[0]:.3f}, {obj[1]:.3f}, {obj[2]:.3f}]  "
            f"end_err={status.eside_arm_mm:.1f} mm"
        )
        if capture_hook is not None:
            capture_hook(segment.label or segment.kind)

    print("\n[Physical packaging trajectory]")
    print(
        f"  release object-bottom clearance={task_layout.release_object_bottom_clearance_m * 1000.0:.0f} mm  "
        f"grasp gap={task_layout.grasp_gap_m * 1000.0:.0f} mm"
    )
    report = replay_sim(
        program,
        traj_ctx,
        stabilize_grasp_weld=False,
        on_phase=_on_phase,
    )
    ok = (
        report.place_error_mm < task_layout.place_success_distance_m * 1000.0
        and report.home_drift_mm < task_layout.home_success_distance_m * 1000.0
    )
    print(
        f"\n  Place error: {report.place_error_mm:.1f} mm  "
        f"Home drift: {report.home_drift_mm:.1f} mm  {'OK' if ok else 'CHECK'}"
    )
    return report


def _cycle_failure_reason(report: SimReport, layout, task_layout) -> str | None:
    phases = {phase.label: phase for phase in report.phases}
    lift = phases.get("lift")
    lift_clearance_m = float(getattr(task_layout, "lift_success_clearance_m", 0.100))
    min_lift_z_mm = (layout.table_top_z + lift_clearance_m) * 1000.0
    if lift is None or lift.obj_pos_mm[2] <= min_lift_z_mm:
        actual_z_mm = float("nan") if lift is None else lift.obj_pos_mm[2]
        return f"grasp/lift failed: block_z={actual_z_mm:.1f} mm, required>{min_lift_z_mm:.1f} mm"
    place_limit_mm = task_layout.place_success_distance_m * 1000.0
    if report.place_error_mm >= place_limit_mm:
        return f"place error {report.place_error_mm:.1f} mm exceeds {place_limit_mm:.1f} mm"
    home_limit_mm = float(getattr(task_layout, "home_success_distance_m", 0.010)) * 1000.0
    if report.home_drift_mm >= home_limit_mm:
        return f"home drift {report.home_drift_mm:.1f} mm exceeds {home_limit_mm:.1f} mm"
    return None


def _hold_final_view(scene) -> None:
    visualizer = getattr(scene, "visualizer", None)
    viewer = getattr(visualizer, "viewer", None) if visualizer is not None else None
    if viewer is None:
        return
    try:
        while True:
            scene.step()
            time.sleep(SIM_DT)
    except gs.GenesisException as exc:
        if "Viewer closed" not in str(exc):
            raise


def _positive_cycles(value: str) -> int:
    cycles = int(value)
    if cycles < 1:
        raise argparse.ArgumentTypeError("cycles must be at least 1")
    return cycles


def _resolve_repetition(cycles: int | None, loop: bool | None) -> tuple[bool, int]:
    """Return ``(infinite, finite_limit)`` for the CLI repetition flags."""
    return loop is True, cycles if cycles is not None else 1


def _cycle_indices(*, infinite: bool, cycle_limit: int) -> Iterator[int]:
    cycle = 1
    while infinite or cycle <= cycle_limit:
        yield cycle
        cycle += 1


def _capture_keyframes_interactive(scene, robot, block, layout, speed: float, ctx: ShowcaseRobotCtx) -> None:
    """Capture startup keyframes using the interactive viewer camera."""
    out_dir = Path(__file__).resolve().parents[3] / "debug" / "showcase_keyframes"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCapturing keyframes to {out_dir} (viewer open; inspect then Ctrl+C)")

    hold_robot_home(robot, scene, ctx, steps=_scale_steps(SETTLE_STEPS, speed))
    run_pick_place_cycle(scene, robot, block, layout, speed=speed, ctx=ctx, stop_after_phase0=True)
    print("\nKeyframe workflow complete.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="UFACTORY robot physics packaging showcase")
    parser.add_argument("--robot", default="xarm6", choices=robot_runtime_cli_choices())
    parser.add_argument(
        "--table-height",
        type=float,
        default=None,
        help="Tabletop surface height in meters (default: 0.75)",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Motion speed multiplier")
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Open the interactive viewer and hold the final frame until the window closes",
    )
    parser.add_argument(
        "--executor",
        choices=("servo_j", "servo_cartesian"),
        default="servo_cartesian",
        help="Simulation arm target stream (default: servo_cartesian)",
    )
    parser.add_argument("--config", type=Path, help="Optional packaging task YAML override")
    parser.add_argument(
        "--backend",
        choices=("cpu", "gpu"),
        default=None,
        help="Override simulation.backend (use cpu without a Genesis-supported GPU)",
    )
    repetition = parser.add_mutually_exclusive_group()
    repetition.add_argument(
        "--cycles",
        type=_positive_cycles,
        default=None,
        help="Number of pick-place cycles (default: 1)",
    )
    repetition.add_argument(
        "--loop",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Loop forever; --no-loop is a compatibility alias for one cycle",
    )
    parser.add_argument(
        "--capture-keyframes",
        action="store_true",
        help="Run startup pose debug sequence then exit (use headless script for PNGs)",
    )
    args = parser.parse_args(argv)
    runtime_config = load_runtime_config(args.robot, task="packaging_showcase", config_path=args.config)
    if args.backend is not None:
        runtime_config = override_simulation_backend(runtime_config, args.backend)

    require_genesis_capabilities(gs, pbr=True, deferred_viewer=True)
    with GenesisRuntimeManager(runtime_config.simulation):
        return _run_packaging_sim_body(args, runtime_config)


def _run_packaging_sim_body(args: argparse.Namespace, runtime_config) -> int:
    table_top_z = (
        args.table_height if args.table_height is not None else make_layout(runtime_config=runtime_config).table_top_z
    )
    scene, robot, block, layout = build_packaging_scene(
        table_top_z,
        sim_dt=SIM_DT,
        build_scene=True,
        show_viewer=False,
        runtime_config=runtime_config,
    )

    print(f"{runtime_config.robot.key} packaging showcase — Ctrl+C to exit")
    print(f"  simulation.backend={runtime_config.simulation.backend}")
    infinite, cycle_limit = _resolve_repetition(args.cycles, args.loop)
    task_layout = packaging_layout(runtime_config)
    cycle_text = "infinite" if infinite else str(cycle_limit)
    print(f"  table_top_z={layout.table_top_z:.2f}m  speed={args.speed}  cycles={cycle_text}")

    ctx = init_showcase_robot(robot, layout, scene, runtime_config=runtime_config)
    hold_robot_home(robot, scene, ctx, steps=_scale_steps(SETTLE_STEPS, args.speed))

    if args.visual:
        start_deferred_viewer(scene)

    if args.capture_keyframes:
        _reset_block(block, layout)
        _capture_keyframes_interactive(scene, robot, block, layout, args.speed, ctx)
        return 0

    failed = False
    cycle = 0
    try:
        for cycle in _cycle_indices(infinite=infinite, cycle_limit=cycle_limit):
            total = "∞" if infinite else str(cycle_limit)
            print(f"\n=== Cycle {cycle}/{total} ===")
            prepare_packaging_cycle(scene, robot, block, layout, ctx, speed=args.speed)
            report = run_pick_place_cycle(
                scene,
                robot,
                block,
                layout,
                speed=args.speed,
                ctx=ctx,
                config_path=args.config,
                robot_key=runtime_config.robot.key,
                executor=args.executor,
            )
            assert report is not None
            failure_reason = _cycle_failure_reason(report, layout, task_layout)
            if failure_reason is not None:
                failed = True
                print(f"\nCycle {cycle}/{total} FAILED: {failure_reason}")
                break
            print(f"Cycle {cycle}/{total} complete.")

        if args.visual and (failed or not infinite):
            if failed:
                print("No further cycles will run. Viewer stays open for inspection.")
            else:
                print(f"\nCompleted {cycle_limit} cycle(s). Viewer stays open.")
            _hold_final_view(scene)
        elif failed:
            print("No further cycles will run.")
        elif not infinite:
            print(f"\nCompleted {cycle_limit} cycle(s).")
    except KeyboardInterrupt:
        return 1 if failed else 0
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
