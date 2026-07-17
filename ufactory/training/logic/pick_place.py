"""Genesis-free numeric core for the pick-place RL task (CPU-testable)."""

from __future__ import annotations

from typing import NamedTuple

import torch

from ufactory.grippers.g2 import GRIPPER_G2_SIM_CLOSE_DRIVE
from ufactory.training.logic.common import action_penalty

__all__ = [
    "PickPlaceTaskState",
    "action_penalty",
    "arm_target_ee_pos",
    "arm_target_qpos",
    "desired_grasp_pos_base",
    "drive_to_gap_m",
    "gap_m_to_drive",
    "gripper_target_gap",
    "next_curriculum_stage",
    "pick_place_observation",
    "reward_action_penalty",
    "reward_align",
    "reward_close_gripper",
    "near_target_attenuate",
    "reward_grasp",
    "reward_holding_table",
    "reward_keypoints",
    "reward_lift",
    "reward_lower",
    "reward_place",
    "reward_place_xy",
    "reward_place_z",
    "reward_reach",
    "reward_release",
    "reward_success",
    "reward_table_collision",
    "reward_workspace_violation",
    "sample_object_and_target",
    "update_task_state",
]


class PickPlaceTaskState(NamedTuple):
    holding: torch.Tensor
    carry: torch.Tensor
    grasped: torch.Tensor  # alias of carry (obs / metrics)
    ever_lifted: torch.Tensor
    ever_grasped: torch.Tensor
    place_stable_steps: torch.Tensor
    episode_place_success: torch.Tensor
    episode_success: torch.Tensor


def gap_m_to_drive(gap_m: torch.Tensor, close_gap_m: float, open_gap_m: float) -> torch.Tensor:
    clipped = torch.clamp(gap_m, close_gap_m, open_gap_m)
    return GRIPPER_G2_SIM_CLOSE_DRIVE * (1.0 - clipped / open_gap_m)


def drive_to_gap_m(drive: torch.Tensor, open_gap_m: float) -> torch.Tensor:
    clipped = torch.clamp(drive, 0.0, GRIPPER_G2_SIM_CLOSE_DRIVE)
    return open_gap_m * (1.0 - clipped / GRIPPER_G2_SIM_CLOSE_DRIVE)


def arm_target_qpos(
    current_q: torch.Tensor,
    actions_arm: torch.Tensor,
    action_scale: float,
    max_joint_delta_rad: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (target_q, joint_delta_unclipped). Kept for unit-test / joint-space baselines."""
    joint_delta_unclipped = actions_arm * action_scale
    joint_delta = torch.clamp(joint_delta_unclipped, -max_joint_delta_rad, max_joint_delta_rad)
    return current_q + joint_delta, joint_delta_unclipped


def arm_target_ee_pos(
    current_ee: torch.Tensor,
    actions_xyz: torch.Tensor,
    action_scale: float,
    max_cartesian_delta_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (target_ee_base, cartesian_delta_unclipped) for relative Δxyz arm control."""
    delta_unclipped = actions_xyz * action_scale
    delta = torch.clamp(delta_unclipped, -max_cartesian_delta_m, max_cartesian_delta_m)
    return current_ee + delta, delta_unclipped


def gripper_target_gap(
    current_gap: torch.Tensor,
    action_gripper: torch.Tensor,
    gripper_delta_m: float,
    close_gap_m: float,
    open_gap_m: float,
) -> torch.Tensor:
    return torch.clamp(
        current_gap + action_gripper * gripper_delta_m,
        min=close_gap_m,
        max=open_gap_m,
    )


def pick_place_observation(
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    ee_base: torch.Tensor,
    gripper_gap: torch.Tensor,
    obj_base: torch.Tensor,
    target_base: torch.Tensor,
    grasped: torch.Tensor,
    ever_grasped: torch.Tensor,
) -> torch.Tensor:
    ee_to_obj = obj_base - ee_base
    obj_to_target = target_base - obj_base
    return torch.cat(
        [
            joint_pos,
            joint_vel,
            ee_base,
            gripper_gap.unsqueeze(-1),
            obj_base,
            target_base,
            ee_to_obj,
            obj_to_target,
            grasped.unsqueeze(-1).float(),
            ever_grasped.unsqueeze(-1).float(),
        ],
        dim=-1,
    )


def desired_grasp_pos_base(obj_base: torch.Tensor, grasp_center_offset_z: float) -> torch.Tensor:
    offset = torch.zeros_like(obj_base)
    offset[..., 2] = grasp_center_offset_z
    return obj_base + offset


def update_task_state(
    obj_base: torch.Tensor,
    ee_base: torch.Tensor,
    gap: torch.Tensor,
    obj_vel: torch.Tensor,
    target_pos: torch.Tensor,
    *,
    ever_grasped: torch.Tensor,
    ever_lifted: torch.Tensor,
    episode_place_success: torch.Tensor,
    episode_success: torch.Tensor,
    place_stable_steps: torch.Tensor,
    obj_rest_z_base: float,
    object_width: float,
    place_success_dist_m: float,
    success_hold_steps: int,
) -> PickPlaceTaskState:
    lifted = obj_base[:, 2] > (obj_rest_z_base + 0.02)
    gripper_holding_gap = gap < (object_width + 0.02)
    ee_near_obj = torch.norm(ee_base - obj_base, dim=-1) < 0.07

    holding = gripper_holding_gap & ee_near_obj
    carry = holding & lifted
    # Metrics / obs "grasped" means a real lift-carry (not table-touch hold).
    grasped = carry
    ever_lifted_out = ever_lifted | lifted
    ever_grasped_out = ever_grasped | carry

    xy_dist = torch.norm(obj_base[:, :2] - target_pos[:, :2], dim=-1)
    at_table = torch.abs(obj_base[:, 2] - obj_rest_z_base) < 0.025
    released = gap > (object_width + 0.02)
    stable = torch.norm(obj_vel, dim=-1) < 0.15
    place_candidate = (xy_dist < place_success_dist_m) & at_table & released & stable & ever_grasped_out
    place_stable_steps_out = torch.where(
        place_candidate,
        place_stable_steps + 1,
        torch.zeros_like(place_stable_steps),
    )
    episode_place_success_out = episode_place_success | place_candidate
    episode_success_out = episode_success | (place_stable_steps_out >= success_hold_steps)
    return PickPlaceTaskState(
        holding=holding,
        carry=carry,
        grasped=grasped,
        ever_lifted=ever_lifted_out,
        ever_grasped=ever_grasped_out,
        place_stable_steps=place_stable_steps_out,
        episode_place_success=episode_place_success_out,
        episode_success=episode_success_out,
    )


def reward_reach(
    ee_base: torch.Tensor,
    grasp_pos_base: torch.Tensor,
    ever_grasped: torch.Tensor,
) -> torch.Tensor:
    dist = torch.norm(ee_base - grasp_pos_base, dim=-1)
    return (1.0 / (1.0 + 8.0 * dist)) * (~ever_grasped).float()


def reward_align(
    ee_base: torch.Tensor,
    grasp_pos_base: torch.Tensor,
    ever_grasped: torch.Tensor,
) -> torch.Tensor:
    """Legacy align; prefer ``reward_keypoints``. Kept for scale=0 compatibility."""
    xy_dist = torch.norm(ee_base[:, :2] - grasp_pos_base[:, :2], dim=-1)
    z_diff = torch.abs(ee_base[:, 2] - grasp_pos_base[:, 2])
    return torch.exp(-25.0 * xy_dist) * torch.exp(-25.0 * z_diff) * (~ever_grasped).float()


def reward_keypoints(
    ee_base: torch.Tensor,
    grasp_pos_base: torch.Tensor,
    ever_grasped: torch.Tensor,
    *,
    unit_length: float = 0.03,
) -> torch.Tensor:
    """Genesis-style keypoint alignment (simplified 3-axis offsets, gripper-down).

    Compares finger-center + local offsets against the desired grasp pose offsets.
    """
    # Local offsets in base frame assuming fixed gripper-down (no rotation term).
    offsets = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [unit_length, 0.0, 0.0],
            [-unit_length, 0.0, 0.0],
            [0.0, unit_length, 0.0],
            [0.0, -unit_length, 0.0],
            [0.0, 0.0, unit_length],
            [0.0, 0.0, -unit_length],
        ],
        device=ee_base.device,
        dtype=ee_base.dtype,
    )
    ee_kp = ee_base.unsqueeze(1) + offsets.unsqueeze(0)
    obj_kp = grasp_pos_base.unsqueeze(1) + offsets.unsqueeze(0)
    dist = torch.norm(ee_kp - obj_kp, dim=-1).sum(dim=-1)
    return torch.exp(-dist) * (~ever_grasped).float()


def reward_close_gripper(
    ee_base: torch.Tensor,
    grasp_pos_base: torch.Tensor,
    gap: torch.Tensor,
    open_gap_m: float,
    ever_grasped: torch.Tensor,
    target_gap_m: float,
) -> torch.Tensor:
    """Reward closing toward ``target_gap_m`` (grasp preload), not dead-closed 0 mm."""
    dist = torch.norm(ee_base - grasp_pos_base, dim=-1)
    proximity = torch.clamp(1.0 - dist / 0.06, min=0.0)
    gap_scale = max(0.015, 0.25 * (open_gap_m - target_gap_m))
    gap_match = torch.exp(-torch.abs(gap - target_gap_m) / gap_scale)
    return proximity * gap_match * (~ever_grasped).float()


def near_target_attenuate(
    obj_base: torch.Tensor,
    target_pos: torch.Tensor,
    place_success_dist_m: float,
) -> torch.Tensor:
    """1 when far from place target, 0 when xy <= place_success_dist (ramp to 2x).

    Used to kill dense grasp/lift credit once the cube is already over the target,
    so hovering is no longer the optimal dense-reward policy.
    """
    xy_dist = torch.norm(obj_base[:, :2] - target_pos[:, :2], dim=-1)
    d = max(float(place_success_dist_m), 1e-6)
    return ((xy_dist - d) / d).clamp(0.0, 1.0)


def reward_lift(
    obj_z: torch.Tensor,
    obj_rest_z_base: float,
    lift_height_m: float,
    carry: torch.Tensor,
    obj_base: torch.Tensor | None = None,
    target_pos: torch.Tensor | None = None,
    place_success_dist_m: float = 0.04,
) -> torch.Tensor:
    height_gain = (obj_z - obj_rest_z_base).clamp(0.0, lift_height_m) / lift_height_m
    scale = torch.ones_like(height_gain)
    if obj_base is not None and target_pos is not None:
        scale = near_target_attenuate(obj_base, target_pos, place_success_dist_m)
    return height_gain * carry.float() * scale


def reward_grasp(
    carry: torch.Tensor,
    obj_base: torch.Tensor | None = None,
    target_pos: torch.Tensor | None = None,
    place_success_dist_m: float = 0.04,
) -> torch.Tensor:
    """Dense carry bonus; attenuated near the place target to break hover basins."""
    carry_f = carry.float()
    if obj_base is None or target_pos is None:
        return carry_f
    return carry_f * near_target_attenuate(obj_base, target_pos, place_success_dist_m)


def reward_place_xy(
    obj_base: torch.Tensor,
    target_pos: torch.Tensor,
    carry: torch.Tensor,
    ever_grasped: torch.Tensor,
) -> torch.Tensor:
    """Horizontal transport while carrying; approach after set-down."""
    xy_dist = torch.norm(obj_base[:, :2] - target_pos[:, :2], dim=-1)
    transport = (1.0 / (1.0 + 8.0 * xy_dist)) * carry.float()
    approach = (1.0 / (1.0 + 8.0 * xy_dist)) * ever_grasped.float() * (~carry).float()
    return transport + approach


def reward_place_z(
    obj_base: torch.Tensor,
    target_pos: torch.Tensor,
    ever_grasped: torch.Tensor,
    obj_rest_z_base: float,
    place_success_dist_m: float,
) -> torch.Tensor:
    """Dense lower-to-table shaping once XY is near the place target."""
    xy_dist = torch.norm(obj_base[:, :2] - target_pos[:, :2], dim=-1)
    near = (xy_dist < (2.0 * place_success_dist_m)).float()
    height_err = (obj_base[:, 2] - obj_rest_z_base).clamp(min=0.0)
    return (1.0 / (1.0 + 12.0 * height_err)) * near * ever_grasped.float()


def reward_lower(
    obj_z: torch.Tensor,
    prev_obj_z: torch.Tensor,
    obj_base: torch.Tensor,
    target_pos: torch.Tensor,
    carry: torch.Tensor,
    place_success_dist_m: float,
) -> torch.Tensor:
    """Reward downward object motion while carrying near the place target (Δxyz-aligned)."""
    xy_dist = torch.norm(obj_base[:, :2] - target_pos[:, :2], dim=-1)
    near = (xy_dist < (2.0 * place_success_dist_m)).float()
    descent = (prev_obj_z - obj_z).clamp(min=0.0, max=0.02) / 0.01
    return near * carry.float() * descent


def reward_holding_table(
    holding: torch.Tensor,
    carry: torch.Tensor,
    obj_base: torch.Tensor,
    target_pos: torch.Tensor,
    ever_grasped: torch.Tensor,
    obj_rest_z_base: float,
    place_success_dist_m: float,
) -> torch.Tensor:
    """Small bridge reward when holding on the table near the target (avoids carry cliff vacuum)."""
    xy_dist = torch.norm(obj_base[:, :2] - target_pos[:, :2], dim=-1)
    near = (xy_dist < (2.0 * place_success_dist_m)).float()
    at_table = (torch.abs(obj_base[:, 2] - obj_rest_z_base) < 0.04).float()
    return holding.float() * (~carry).float() * near * at_table * ever_grasped.float()


def reward_place(
    obj_base: torch.Tensor,
    target_pos: torch.Tensor,
    carry: torch.Tensor,
    ever_grasped: torch.Tensor,
    obj_rest_z_base: float,
    place_success_dist_m: float,
) -> torch.Tensor:
    """Backward-compatible sum of place_xy + place_z."""
    return reward_place_xy(obj_base, target_pos, carry, ever_grasped) + reward_place_z(
        obj_base, target_pos, ever_grasped, obj_rest_z_base, place_success_dist_m
    )


def reward_release(
    obj_base: torch.Tensor,
    target_pos: torch.Tensor,
    gap: torch.Tensor,
    open_gap_m: float,
    ever_grasped: torch.Tensor,
    obj_rest_z_base: float,
    place_success_dist_m: float,
    *,
    release_height_m: float = 0.05,
) -> torch.Tensor:
    """Open-gripper credit only when low over the place target (no high-altitude release)."""
    xy_dist = torch.norm(obj_base[:, :2] - target_pos[:, :2], dim=-1)
    near_target = (xy_dist < place_success_dist_m).float()
    low = (obj_base[:, 2] < (obj_rest_z_base + release_height_m)).float()
    at_table = torch.exp(-20.0 * torch.abs(obj_base[:, 2] - obj_rest_z_base))
    open_fraction = (gap / open_gap_m).clamp(0.0, 1.0)
    return near_target * low * at_table * open_fraction * ever_grasped.float()


def reward_success(episode_success: torch.Tensor) -> torch.Tensor:
    return episode_success.float()


def reward_action_penalty(joint_vel: torch.Tensor) -> torch.Tensor:
    return action_penalty(joint_vel)


def reward_table_collision(
    link_zs: torch.Tensor,
    table_top_z: float,
    margin: float = 0.04,
) -> torch.Tensor:
    """link_zs: (N, L) z positions of monitored links."""
    violation = torch.clamp((table_top_z + margin - link_zs) / margin, min=0.0)
    return -violation.sum(dim=-1)


def reward_workspace_violation(
    ee_base: torch.Tensor,
    obj_base: torch.Tensor,
    workspace_lower: torch.Tensor,
    workspace_upper: torch.Tensor,
) -> torch.Tensor:
    ee_low = (ee_base < workspace_lower).float().sum(dim=-1)
    ee_high = (ee_base > workspace_upper).float().sum(dim=-1)
    obj_low = (obj_base < workspace_lower).float().sum(dim=-1)
    obj_high = (obj_base > workspace_upper).float().sum(dim=-1)
    dropped = (obj_base[:, 2] < -0.02).float()
    return -(ee_low + ee_high + obj_low + obj_high + dropped)


def sample_object_and_target(
    stage: int,
    n: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    fixed_obj: torch.Tensor,
    fixed_target: torch.Tensor,
    obj_spawn_lower: torch.Tensor,
    obj_spawn_upper: torch.Tensor,
    target_spawn_lower: torch.Tensor,
    target_spawn_upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if stage == 0:
        obj = fixed_obj.unsqueeze(0).expand(n, 3).clone()
        target = fixed_target.unsqueeze(0).expand(n, 3).clone()
        return obj, target

    rand_obj = torch.rand(n, 3, device=device, dtype=dtype)
    obj = obj_spawn_lower + rand_obj * (obj_spawn_upper - obj_spawn_lower)

    if stage == 1:
        target = obj.clone()
        target[:, 0] += 0.10
    elif stage == 2:
        rand = torch.rand(n, 2, device=device, dtype=dtype)
        target = obj.clone()
        target[:, 0] += 0.05 + rand[:, 0] * 0.10
        target[:, 1] += (rand[:, 1] - 0.5) * 0.06
    elif stage == 3:
        rand = torch.rand(n, 2, device=device, dtype=dtype)
        target = obj.clone()
        target[:, 0] += 0.10 + rand[:, 0] * 0.15
        target[:, 1] += (rand[:, 1] - 0.5) * 0.10
    else:
        rand_target = torch.rand(n, 3, device=device, dtype=dtype)
        target = target_spawn_lower + rand_target * (target_spawn_upper - target_spawn_lower)
    return obj, target


def next_curriculum_stage(
    stage: int,
    grasp_rate: float,
    place_rate: float,
    grasp_count: int,
    place_count: int,
) -> int:
    """Pure stage transition; counters reset remain the env's responsibility."""
    if grasp_count >= 500:
        if stage == 0 and grasp_rate > 0.50:
            return 1
        if stage == 1 and grasp_rate > 0.70:
            return 2
    if place_count >= 500:
        if stage == 2 and place_rate > 0.60:
            return 3
        if stage == 3 and place_rate > 0.50:
            return 4
    return stage
