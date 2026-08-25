"""Genesis-free numeric core for the pick-place RL task (CPU-testable)."""

from __future__ import annotations

from typing import NamedTuple

import torch

from ufactory.grippers.g2 import GRIPPER_G2_SIM_CLOSE_DRIVE
from ufactory.training.logic.common import action_penalty

__all__ = [
    "PickPlaceTaskState",
    "action_penalty",
    "action_to_cartesian_delta",
    "annealed_frac",
    "arm_target_ee_pos",
    "arm_target_qpos",
    "cartesian_delta_to_action",
    "desired_grasp_pos_base",
    "drive_to_gap_m",
    "gap_m_to_drive",
    "gripper_target_gap",
    "leashed_ee_setpoint",
    "next_curriculum_stage",
    "normalized_ee_setpoint_residual",
    "normalized_pick_place_contact_features",
    "normalized_pick_place_quality_features",
    "near_table_down_penalty",
    "pick_place_observation",
    "reward_action_penalty",
    "reward_align",
    "reward_approach_potential",
    "reward_clean_lift_bonus",
    "reward_clean_lift_quality",
    "reward_close_gripper",
    "reward_descent_progress",
    "reward_drop_far",
    "near_target_attenuate",
    "reward_grasp",
    "reward_grasp_bonus",
    "reward_grasp_centering",
    "reward_grasp_gap_progress",
    "reward_grasp_lift_action",
    "reward_grasp_lift_progress",
    "reward_grasp_ready_closing",
    "reward_grasp_settle_action",
    "reward_hard_landing",
    "reward_holding_table",
    "reward_invalid_release",
    "reward_keypoints",
    "reward_landing_quality",
    "reward_lift",
    "reward_lower",
    "reward_near_table_xy_action",
    "reward_near_table_down_speed_margin",
    "reward_near_table_xy_speed_margin",
    "reward_near_target_speed",
    "reward_place",
    "reward_place_xy",
    "reward_place_z",
    "reward_post_release_clearance_progress",
    "reward_post_release_contact",
    "reward_post_release_recontact",
    "reward_premature_opening",
    "reward_precision_progress",
    "reward_pre_lift_xy_progress",
    "reward_push_after_release",
    "reward_push_before_grasp",
    "reward_release_clearance_opening",
    "reward_ready_opening",
    "reward_reach",
    "reward_release",
    "reward_release_quality",
    "reward_release_readiness_progress",
    "reward_setdown_action",
    "reward_throw_release",
    "reward_transport_progress",
    "reward_success",
    "reward_table_collision",
    "reward_valid_release",
    "reward_workspace_violation",
    "sample_object_and_target",
    "sample_reset_phase_masks",
    "update_task_state",
]


class PickPlaceTaskState(NamedTuple):
    holding: torch.Tensor
    carry: torch.Tensor
    grasped: torch.Tensor  # alias of carry (obs / metrics)
    ever_lifted: torch.Tensor
    ever_grasped: torch.Tensor
    ever_carried_near: torch.Tensor
    place_stable_steps: torch.Tensor
    episode_place_success: torch.Tensor
    episode_success: torch.Tensor
    release_started: torch.Tensor
    release_valid: torch.Tensor
    release_violation: torch.Tensor
    near_table_entered: torch.Tensor
    hard_landing_violation: torch.Tensor
    max_pre_lift_xy_m: torch.Tensor
    max_landing_xy_speed_m_s: torch.Tensor
    max_landing_down_speed_m_s: torch.Tensor
    release_xy_dist_m: torch.Tensor
    release_height_error_m: torch.Tensor
    release_speed_m_s: torch.Tensor
    release_obj_xy: torch.Tensor
    post_release_drift_m: torch.Tensor
    grasp_offset_xy_m: torch.Tensor
    quality_ok: torch.Tensor


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


def action_to_cartesian_delta(
    actions_xyz: torch.Tensor,
    action_scale: float,
    response_exponent: float = 1.0,
) -> torch.Tensor:
    """Map bounded actions to a Cartesian delta, optionally with a fine-grained centre.

    An exponent above one keeps the full reach at |action|=1 while shrinking the delta
    per unit action near zero, so the sub-millimetre corrections the near-table speed
    limits demand no longer live inside a few percent of the action range.
    """
    if response_exponent == 1.0:
        return actions_xyz * action_scale
    if response_exponent <= 0.0:
        raise ValueError("response_exponent must be positive")
    magnitude = actions_xyz.abs() ** response_exponent
    return torch.sign(actions_xyz) * magnitude * action_scale


def cartesian_delta_to_action(
    delta_m: torch.Tensor,
    action_scale: float,
    response_exponent: float = 1.0,
) -> torch.Tensor:
    """Invert :func:`action_to_cartesian_delta` for scripted controllers."""
    normalized = (delta_m / action_scale).clamp(-1.0, 1.0)
    if response_exponent == 1.0:
        return normalized
    return torch.sign(normalized) * normalized.abs() ** (1.0 / response_exponent)


POLICY_PREFIX_OBJ_Z = 18
POLICY_PREFIX_GRASPED = 28


def near_table_down_penalty(
    observations: torch.Tensor,
    predicted_actions: torch.Tensor,
    *,
    obj_rest_z_m: float,
    height_m: float,
    max_down_action: float,
) -> torch.Tensor:
    """Penalize faster-than-brake downward actions while holding near the table.

    The 30-dimensional observation prefix puts object z at index 18 and the
    grasped flag at 28. The clone that already carries to the target still
    commands a coarse descent there; this term is the Z counterpart of the
    far-open gripper penalty.
    """

    if height_m <= 0.0:
        raise ValueError("height_m must be positive")
    if max_down_action < 0.0:
        raise ValueError("max_down_action must be non-negative")
    height = observations[:, POLICY_PREFIX_OBJ_Z] - float(obj_rest_z_m)
    grasped = observations[:, POLICY_PREFIX_GRASPED] > 0.5
    near = grasped & (height <= float(height_m))
    excess = torch.relu(-predicted_actions[:, 2] - float(max_down_action))
    return near.float() * excess.square()


def arm_target_ee_pos(
    current_ee: torch.Tensor,
    actions_xyz: torch.Tensor,
    action_scale: float,
    max_cartesian_delta_m: float,
    response_exponent: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (target_ee_base, cartesian_delta_unclipped) for relative Δxyz arm control."""
    delta_unclipped = action_to_cartesian_delta(actions_xyz, action_scale, response_exponent)
    delta = torch.clamp(delta_unclipped, -max_cartesian_delta_m, max_cartesian_delta_m)
    return current_ee + delta, delta_unclipped


def leashed_ee_setpoint(
    setpoint_ee: torch.Tensor,
    measured_ee: torch.Tensor,
    leash_m: float,
) -> torch.Tensor:
    """Keep a commanded setpoint within ``leash_m`` of where the arm actually is.

    Integrating deltas onto the commanded setpoint rather than the measured pose is what
    makes a zero action mean "hold": adding to the measured pose subtracts the standing
    PD droop from every command, so the wrist sinks whenever the policy stops pushing.
    The leash stops the setpoint from running away while the arm is blocked.
    """
    if leash_m <= 0.0:
        raise ValueError("leash_m must be positive")
    error = setpoint_ee - measured_ee
    return measured_ee + error.clamp(-leash_m, leash_m)


def normalized_ee_setpoint_residual(
    setpoint_ee: torch.Tensor,
    measured_ee: torch.Tensor,
    leash_m: float,
) -> torch.Tensor:
    """Expose the Cartesian controller's integrated-state error to the policy."""

    scale = float(leash_m)
    if scale <= 0.0:
        raise ValueError("leash_m must be positive")
    if setpoint_ee.shape != measured_ee.shape or setpoint_ee.shape[-1] != 3:
        raise ValueError("setpoint_ee and measured_ee must have matching (..., 3) shapes")
    return ((setpoint_ee - measured_ee) / scale).clamp(-2.0, 2.0)


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


def normalized_pick_place_contact_features(
    finger_span_m: torch.Tensor,
    left_contact_force_n: torch.Tensor,
    right_contact_force_n: torch.Tensor,
    *,
    object_width_m: float,
    contact_force_scale_n: float = 5.0,
    contact_force_threshold_n: float = 0.05,
) -> torch.Tensor:
    """Return the three appended contact features: span error, bilateral flag, grip force.

    The nominal gap the policy already sees is the drive angle mapped through a straight
    line, so it reads several millimetres away from the true pad separation and says
    nothing about whether the pads actually touch anything. These three come from link
    poses and solver contact forces instead.
    """
    scale = float(contact_force_scale_n)
    if scale <= 0.0:
        raise ValueError("contact_force_scale_n must be positive")
    span_error = ((finger_span_m - float(object_width_m)) / float(object_width_m)).clamp(-2.0, 4.0)
    grip_force = torch.minimum(left_contact_force_n, right_contact_force_n)
    bilateral = (grip_force > float(contact_force_threshold_n)).float()
    return torch.stack(
        [
            span_error,
            bilateral,
            (grip_force / scale).clamp(0.0, 4.0),
        ],
        dim=-1,
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
    commanded_gap: torch.Tensor | None = None,
    previous_action: torch.Tensor | None = None,
    normalized_layout_offsets: torch.Tensor | None = None,
    quality_features: torch.Tensor | None = None,
    contact_features: torch.Tensor | None = None,
    normalized_setpoint_residual: torch.Tensor | None = None,
    scripted_action_hint: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the policy observation.

    The original 30 values remain an unchanged prefix for artifact
    compatibility.  New policies append the integrated gripper command and
    previous action, removing two otherwise hidden pieces of controller state.
    """
    ee_to_obj = obj_base - ee_base
    obj_to_target = target_base - obj_base
    parts = [
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
    ]
    if commanded_gap is not None:
        parts.append(commanded_gap.unsqueeze(-1))
    if previous_action is not None:
        parts.append(previous_action)
    if quality_features is not None:
        parts.append(quality_features)
    if contact_features is not None:
        parts.append(contact_features)
    if normalized_setpoint_residual is not None:
        parts.append(normalized_setpoint_residual)
    if normalized_layout_offsets is not None:
        parts.append(normalized_layout_offsets)
    if scripted_action_hint is not None:
        parts.append(scripted_action_hint)
    return torch.cat(parts, dim=-1)


def normalized_pick_place_quality_features(
    obj_vel: torch.Tensor,
    holding: torch.Tensor,
    ever_carried_near: torch.Tensor,
    release_started: torch.Tensor,
    *,
    velocity_scale_m_s: float = 0.25,
) -> torch.Tensor:
    """Return the six appended policy features used by quality training.

    Ordering is fixed: normalized vx/vy/vz, holding, ever-carried-near, and
    release-started.  Velocity is clipped only to keep extreme contact spikes
    from dominating the actor input; the critic still receives raw velocity in
    its privileged group.
    """

    scale = float(velocity_scale_m_s)
    if scale <= 0.0:
        raise ValueError("velocity_scale_m_s must be positive")
    normalized_velocity = (obj_vel / scale).clamp(-4.0, 4.0)
    return torch.cat(
        [
            normalized_velocity,
            holding.unsqueeze(-1).float(),
            ever_carried_near.unsqueeze(-1).float(),
            release_started.unsqueeze(-1).float(),
        ],
        dim=-1,
    )


def normalized_pick_place_layout_offsets(
    episode_obj_start_base: torch.Tensor,
    episode_target_base: torch.Tensor,
    fixed_obj: torch.Tensor,
    fixed_target: torch.Tensor,
    obj_spawn_lower: torch.Tensor,
    obj_spawn_upper: torch.Tensor,
    target_spawn_lower: torch.Tensor,
    target_spawn_upper: torch.Tensor,
) -> torch.Tensor:
    """Return pickup-layout offsets without changing the legacy prefix.

    These inputs are the sampled task endpoints, not the object's live pose. Otherwise
    a fixed-layout episode would acquire a changing "layout" signal while transporting
    the cube, defeating zero-effect transfer and entangling phase with spawn position.
    The immutable offsets remain available through set-down and release. A random
    pickup changes the arm/object state that reaches the target even when the target
    itself is fixed, so switching the signal off at that boundary creates an abrupt
    observation alias and prevents a protected residual from correcting the arrival.
    Fixed-layout episodes still produce six exact zeros for their entire lifetime.
    """

    obj_half_span = (obj_spawn_upper[:2] - obj_spawn_lower[:2]).abs().mul(0.5).clamp_min(1e-6)
    target_half_span = (target_spawn_upper[:2] - target_spawn_lower[:2]).abs().mul(0.5).clamp_min(1e-6)
    obj_offset = (episode_obj_start_base[:, :2] - fixed_obj[:2]) / obj_half_span
    target_offset = (episode_target_base[:, :2] - fixed_target[:2]) / target_half_span
    canonical_transport = fixed_target[:2] - fixed_obj[:2]
    transport_offset = ((episode_target_base[:, :2] - episode_obj_start_base[:, :2]) - canonical_transport) / (
        obj_half_span + target_half_span
    )
    return torch.cat([obj_offset, target_offset, transport_offset], dim=-1)


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
    ever_carried_near: torch.Tensor,
    episode_place_success: torch.Tensor,
    episode_success: torch.Tensor,
    place_stable_steps: torch.Tensor,
    obj_rest_z_base: float,
    object_width: float,
    place_success_dist_m: float,
    success_hold_steps: int,
    success_table_z_tolerance_m: float = 0.025,
    success_max_obj_speed_m_s: float = 0.15,
    carry_near_dist_m: float = 0.04,
    initial_obj_pos: torch.Tensor | None = None,
    commanded_gap: torch.Tensor | None = None,
    action_gripper: torch.Tensor | None = None,
    release_started: torch.Tensor | None = None,
    release_valid: torch.Tensor | None = None,
    release_violation: torch.Tensor | None = None,
    near_table_entered: torch.Tensor | None = None,
    hard_landing_violation: torch.Tensor | None = None,
    max_pre_lift_xy_m: torch.Tensor | None = None,
    max_landing_xy_speed_m_s: torch.Tensor | None = None,
    max_landing_down_speed_m_s: torch.Tensor | None = None,
    release_xy_dist_m: torch.Tensor | None = None,
    release_height_error_m: torch.Tensor | None = None,
    release_speed_m_s: torch.Tensor | None = None,
    release_obj_xy: torch.Tensor | None = None,
    post_release_drift_m: torch.Tensor | None = None,
    grasp_offset_xy_m: torch.Tensor | None = None,
    release_success_dist_m: float | None = None,
    release_height_tolerance_m: float = 0.005,
    release_max_obj_speed_m_s: float = 0.02,
    pre_lift_max_drag_m: float = 0.005,
    post_release_max_drift_m: float = 0.003,
    landing_near_table_height_m: float = 0.02,
    landing_max_xy_speed_m_s: float = 0.03,
    landing_max_down_speed_m_s: float = 0.05,
    release_action_threshold: float = 0.05,
    release_command_margin_m: float = 0.0005,
    gripper_close_gap_m: float = 0.0,
    bilateral_contact: torch.Tensor | None = None,
) -> PickPlaceTaskState:
    n = obj_base.shape[0]
    quality_checks_enabled = (
        initial_obj_pos is not None
        or commanded_gap is not None
        or action_gripper is not None
        or release_started is not None
    )
    bool_zeros = torch.zeros(n, dtype=torch.bool, device=obj_base.device)
    float_zeros = torch.zeros(n, dtype=obj_base.dtype, device=obj_base.device)
    release_started = bool_zeros if release_started is None else release_started
    release_valid = bool_zeros if release_valid is None else release_valid
    release_violation = bool_zeros if release_violation is None else release_violation
    near_table_entered = bool_zeros if near_table_entered is None else near_table_entered
    hard_landing_violation = bool_zeros if hard_landing_violation is None else hard_landing_violation
    max_pre_lift_xy_m = float_zeros if max_pre_lift_xy_m is None else max_pre_lift_xy_m
    max_landing_xy_speed_m_s = float_zeros if max_landing_xy_speed_m_s is None else max_landing_xy_speed_m_s
    max_landing_down_speed_m_s = float_zeros if max_landing_down_speed_m_s is None else max_landing_down_speed_m_s
    release_xy_dist_m = float_zeros if release_xy_dist_m is None else release_xy_dist_m
    release_height_error_m = float_zeros if release_height_error_m is None else release_height_error_m
    release_speed_m_s = float_zeros if release_speed_m_s is None else release_speed_m_s
    release_obj_xy = (
        torch.zeros(n, 2, dtype=obj_base.dtype, device=obj_base.device) if release_obj_xy is None else release_obj_xy
    )
    post_release_drift_m = float_zeros if post_release_drift_m is None else post_release_drift_m
    grasp_offset_xy_m = float_zeros if grasp_offset_xy_m is None else grasp_offset_xy_m

    lifted = obj_base[:, 2] > (obj_rest_z_base + 0.02)
    gripper_holding_gap = gap < (object_width + 0.02)
    ee_near_obj = torch.norm(ee_base - obj_base, dim=-1) < 0.07

    # ``gap`` is the drive angle mapped through a straight line, so the width-plus-20 mm
    # test alone calls it a grasp while the pads are still a centimetre off each face.
    # When the solver can report contact, require the pads to actually be loaded.
    holding = gripper_holding_gap & ee_near_obj
    if bilateral_contact is not None:
        holding = holding & bilateral_contact
    carry = holding & lifted
    # Metrics / obs "grasped" means a real lift-carry (not table-touch hold).
    grasped = carry
    ever_lifted_out = ever_lifted | lifted
    ever_grasped_out = ever_grasped | carry

    xy_dist = torch.norm(obj_base[:, :2] - target_pos[:, :2], dim=-1)
    # Require a real carry inside the near-target neighborhood before place/release credit.
    # Blocks the mid-path "drop then slide into the circle" shortcut.
    ever_carried_near_out = ever_carried_near | (carry & (xy_dist < float(carry_near_dist_m)))

    if initial_obj_pos is not None:
        pre_lift_displacement = torch.norm(
            obj_base[:, :2] - initial_obj_pos[:, :2],
            dim=-1,
        )
        max_pre_lift_xy_m_out = torch.where(
            ~ever_grasped,
            torch.maximum(max_pre_lift_xy_m, pre_lift_displacement),
            max_pre_lift_xy_m,
        )
    else:
        max_pre_lift_xy_m_out = max_pre_lift_xy_m

    # How far off centre the cube sits between the pads at the moment it leaves the
    # table. Whatever offset is locked in here is carried all the way to the target and
    # shows up unchanged in the final placement error.
    first_lift = carry & (~ever_grasped)
    grasp_offset_xy_m_out = torch.where(
        first_lift,
        torch.norm(obj_base[:, :2] - ee_base[:, :2], dim=-1),
        grasp_offset_xy_m,
    )

    opening_intent = bool_zeros
    if commanded_gap is not None and action_gripper is not None:
        opening_intent = (action_gripper > float(release_action_threshold)) & (
            commanded_gap > (float(gripper_close_gap_m) + float(release_command_margin_m))
        )
    release_event = ever_grasped_out & (~release_started) & opening_intent
    release_success_dist = (
        float(place_success_dist_m) if release_success_dist_m is None else float(release_success_dist_m)
    )
    current_height_error = torch.abs(obj_base[:, 2] - float(obj_rest_z_base))
    current_speed = torch.norm(obj_vel, dim=-1)
    current_release_valid = (
        (xy_dist <= release_success_dist)
        & (current_height_error <= float(release_height_tolerance_m))
        & (current_speed <= float(release_max_obj_speed_m_s))
        & ever_carried_near_out
    )
    release_started_out = release_started | release_event
    release_valid_out = release_valid | (release_event & current_release_valid)
    release_violation_out = release_violation | (release_event & (~current_release_valid))
    release_xy_dist_m_out = torch.where(
        release_event,
        xy_dist,
        release_xy_dist_m,
    )
    release_height_error_m_out = torch.where(
        release_event,
        current_height_error,
        release_height_error_m,
    )
    release_speed_m_s_out = torch.where(
        release_event,
        current_speed,
        release_speed_m_s,
    )
    release_obj_xy_out = torch.where(
        release_event.unsqueeze(-1),
        obj_base[:, :2],
        release_obj_xy,
    )
    current_release_drift = torch.norm(obj_base[:, :2] - release_obj_xy_out, dim=-1)
    post_release_drift_m_out = torch.where(
        release_started_out,
        torch.maximum(post_release_drift_m, current_release_drift),
        post_release_drift_m,
    )

    landing_near_table = ever_carried_near_out & (
        obj_base[:, 2] <= (float(obj_rest_z_base) + float(landing_near_table_height_m))
    )
    near_table_entered_out = near_table_entered | landing_near_table
    xy_speed = torch.norm(obj_vel[:, :2], dim=-1)
    down_speed = torch.clamp(-obj_vel[:, 2], min=0.0)
    max_landing_xy_speed_m_s_out = torch.where(
        landing_near_table,
        torch.maximum(max_landing_xy_speed_m_s, xy_speed),
        max_landing_xy_speed_m_s,
    )
    max_landing_down_speed_m_s_out = torch.where(
        landing_near_table,
        torch.maximum(max_landing_down_speed_m_s, down_speed),
        max_landing_down_speed_m_s,
    )
    hard_landing_now = landing_near_table & (
        (xy_speed > float(landing_max_xy_speed_m_s)) | (down_speed > float(landing_max_down_speed_m_s))
    )
    hard_landing_violation_out = hard_landing_violation | hard_landing_now

    quality_ok = (
        (max_pre_lift_xy_m_out <= float(pre_lift_max_drag_m))
        & release_started_out
        & release_valid_out
        & (~release_violation_out)
        & (~hard_landing_violation_out)
        & (post_release_drift_m_out <= float(post_release_max_drift_m))
    )
    if not quality_checks_enabled:
        quality_ok = torch.ones_like(quality_ok)
    at_table = torch.abs(obj_base[:, 2] - obj_rest_z_base) < success_table_z_tolerance_m
    released = gap > (object_width + 0.02)
    stable = torch.norm(obj_vel, dim=-1) < success_max_obj_speed_m_s
    place_candidate = (
        (xy_dist < place_success_dist_m) & at_table & released & stable & ever_carried_near_out & quality_ok
    )
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
        ever_carried_near=ever_carried_near_out,
        place_stable_steps=place_stable_steps_out,
        episode_place_success=episode_place_success_out,
        episode_success=episode_success_out,
        release_started=release_started_out,
        release_valid=release_valid_out,
        release_violation=release_violation_out,
        near_table_entered=near_table_entered_out,
        hard_landing_violation=hard_landing_violation_out,
        max_pre_lift_xy_m=max_pre_lift_xy_m_out,
        max_landing_xy_speed_m_s=max_landing_xy_speed_m_s_out,
        max_landing_down_speed_m_s=max_landing_down_speed_m_s_out,
        release_xy_dist_m=release_xy_dist_m_out,
        release_height_error_m=release_height_error_m_out,
        release_speed_m_s=release_speed_m_s_out,
        release_obj_xy=release_obj_xy_out,
        post_release_drift_m=post_release_drift_m_out,
        grasp_offset_xy_m=grasp_offset_xy_m_out,
        quality_ok=quality_ok,
    )


def reward_reach(
    ee_base: torch.Tensor,
    grasp_pos_base: torch.Tensor,
    carry: torch.Tensor,
) -> torch.Tensor:
    dist = torch.norm(ee_base - grasp_pos_base, dim=-1)
    return (1.0 / (1.0 + 8.0 * dist)) * (~carry).float()


def reward_approach_potential(
    prev_dist: torch.Tensor,
    cur_dist: torch.Tensor,
    carry: torch.Tensor,
) -> torch.Tensor:
    """Reward net approach progress without paying a stationary hover."""

    return (prev_dist - cur_dist) * (~carry).float()


def reward_align(
    ee_base: torch.Tensor,
    grasp_pos_base: torch.Tensor,
    carry: torch.Tensor,
) -> torch.Tensor:
    """Legacy align; prefer ``reward_keypoints``. Kept for scale=0 compatibility."""
    xy_dist = torch.norm(ee_base[:, :2] - grasp_pos_base[:, :2], dim=-1)
    z_diff = torch.abs(ee_base[:, 2] - grasp_pos_base[:, 2])
    return torch.exp(-25.0 * xy_dist) * torch.exp(-25.0 * z_diff) * (~carry).float()


def reward_keypoints(
    ee_base: torch.Tensor,
    grasp_pos_base: torch.Tensor,
    carry: torch.Tensor,
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
    return torch.exp(-dist) * (~carry).float()


def reward_close_gripper(
    ee_base: torch.Tensor,
    grasp_pos_base: torch.Tensor,
    gap: torch.Tensor,
    open_gap_m: float,
    carry: torch.Tensor,
    target_gap_m: float,
) -> torch.Tensor:
    """Reward closing toward ``target_gap_m`` (grasp preload), not dead-closed 0 mm."""
    dist = torch.norm(ee_base - grasp_pos_base, dim=-1)
    proximity = torch.clamp(1.0 - dist / 0.06, min=0.0)
    gap_scale = max(0.015, 0.25 * (open_gap_m - target_gap_m))
    gap_match = torch.exp(-torch.abs(gap - target_gap_m) / gap_scale)
    return proximity * gap_match * (~carry).float()


def reward_grasp_ready_closing(
    action_gripper: torch.Tensor,
    grasp_dist_m: torch.Tensor,
    measured_gap_m: torch.Tensor,
    commanded_gap_m: torch.Tensor,
    holding: torch.Tensor,
    ever_grasped: torch.Tensor,
    *,
    activation_dist_m: float = 0.012,
    recovery_dist_m: float = 0.040,
    open_gap_m: float = 0.084,
    contact_gap_m: float = 0.050,
    command_commit_margin_m: float = 0.005,
) -> torch.Tensor:
    """Keep closing until measured finger contact, including actuator lag."""

    aligned = grasp_dist_m <= float(activation_dist_m)
    closing_committed = commanded_gap_m < (float(open_gap_m) - float(command_commit_margin_m))
    active = (
        (aligned | (closing_committed & (grasp_dist_m <= float(recovery_dist_m))))
        & (measured_gap_m > float(contact_gap_m))
        & (~holding)
        & (~ever_grasped)
    )
    return torch.clamp(-action_gripper, min=-1.0, max=1.0) * active.float()


def reward_grasp_settle_action(
    cartesian_action: torch.Tensor,
    grasp_dist_m: torch.Tensor,
    measured_gap_m: torch.Tensor,
    commanded_gap_m: torch.Tensor,
    ever_grasped: torch.Tensor,
    *,
    activation_dist_m: float = 0.012,
    recovery_dist_m: float = 0.040,
    open_gap_m: float = 0.084,
    contact_gap_m: float = 0.031,
    command_commit_margin_m: float = 0.005,
) -> torch.Tensor:
    """Hold XY still until lift and hold Z still until physical finger contact."""

    aligned = grasp_dist_m <= float(activation_dist_m)
    closing_committed = (commanded_gap_m < (float(open_gap_m) - float(command_commit_margin_m))) & (
        grasp_dist_m <= float(recovery_dist_m)
    )
    active = (aligned | closing_committed) & (~ever_grasped)
    xy_cost = torch.sum(torch.square(cartesian_action[:, :2]), dim=-1)
    z_cost = torch.square(cartesian_action[:, 2]) * (measured_gap_m > float(contact_gap_m)).float()
    return -(xy_cost + z_cost) * active.float()


def reward_grasp_gap_progress(
    previous_gap_m: torch.Tensor,
    current_gap_m: torch.Tensor,
    ever_grasped: torch.Tensor,
    *,
    contact_gap_m: float = 0.031,
    normalization_m: float = 0.020,
) -> torch.Tensor:
    """Reward net measured-gap closure; reopening cancels prior closure credit."""

    scale = float(normalization_m)
    if scale <= 0.0:
        raise ValueError("normalization_m must be positive")
    contact_gap = float(contact_gap_m)
    active = ((previous_gap_m > contact_gap) | (current_gap_m > contact_gap)) & (~ever_grasped)
    return (previous_gap_m - current_gap_m) / scale * active.float()


def reward_grasp_lift_action(
    action_z: torch.Tensor,
    measured_gap_m: torch.Tensor,
    holding: torch.Tensor,
    ever_grasped: torch.Tensor,
    max_pre_lift_drag_m: torch.Tensor,
    *,
    contact_gap_m: float = 0.031,
    max_drag_m: float = 0.005,
) -> torch.Tensor:
    """Give signed upward-action feedback once contact is clean but before lift."""

    active = (
        holding
        & (measured_gap_m <= float(contact_gap_m))
        & (~ever_grasped)
        & (max_pre_lift_drag_m <= float(max_drag_m))
    )
    return torch.clamp(action_z, min=-1.0, max=1.0) * active.float()


def reward_grasp_lift_progress(
    previous_obj_z_m: torch.Tensor,
    current_obj_z_m: torch.Tensor,
    measured_gap_m: torch.Tensor,
    holding: torch.Tensor,
    prev_ever_grasped: torch.Tensor,
    max_pre_lift_drag_m: torch.Tensor,
    *,
    contact_gap_m: float = 0.031,
    max_drag_m: float = 0.005,
    normalization_m: float = 0.020,
) -> torch.Tensor:
    """Reward signed physical object-height progress after a clean finger contact."""

    scale = float(normalization_m)
    if scale <= 0.0:
        raise ValueError("normalization_m must be positive")
    active = (
        holding
        & (measured_gap_m <= float(contact_gap_m))
        & (~prev_ever_grasped)
        & (max_pre_lift_drag_m <= float(max_drag_m))
    )
    return (current_obj_z_m - previous_obj_z_m) / scale * active.float()


def near_target_attenuate(
    obj_base: torch.Tensor,
    target_pos: torch.Tensor,
    place_success_dist_m: float,
    ramp_mult: float = 4.5,
) -> torch.Tensor:
    """Attenuate carry credit across a wide band near the place target.

    The wider ramp removes the historical policy's profitable parking point just
    outside the old two-radius attenuation region.
    """
    xy_dist = torch.norm(obj_base[:, :2] - target_pos[:, :2], dim=-1)
    d = max(float(place_success_dist_m), 1e-6)
    far = d * max(float(ramp_mult), 1.0 + 1e-6)
    return ((xy_dist - d) / (far - d)).clamp(0.0, 1.0)


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
    place_success_dist_m: float = 0.04,
    ever_carried_near: torch.Tensor | None = None,
) -> torch.Tensor:
    """Horizontal transport while carrying; post-set-down bridge only near target.

    Giving the same dense credit after an early drop makes dropping neutral
    relative to carrying. Restrict that bridge to the placement neighborhood so
    transport credit is actually conditional on retaining the object. The bridge
    further requires ``ever_carried_near`` so mid-path slide-ins earn nothing.
    """
    xy_dist = torch.norm(obj_base[:, :2] - target_pos[:, :2], dim=-1)
    proximity = 1.0 / (1.0 + 8.0 * xy_dist)
    transport = proximity * carry.float()
    near = (xy_dist < place_success_dist_m).float()
    near_gate = ever_grasped if ever_carried_near is None else ever_carried_near
    approach = proximity * near_gate.float() * (~carry).float() * near
    return transport + approach


def reward_place_z(
    obj_base: torch.Tensor,
    target_pos: torch.Tensor,
    ever_grasped: torch.Tensor,
    obj_rest_z_base: float,
    place_success_dist_m: float,
    *,
    near_factor: float = 1.5,
    ever_carried_near: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense lower-to-table shaping once XY is near the place target."""
    xy_dist = torch.norm(obj_base[:, :2] - target_pos[:, :2], dim=-1)
    near = (xy_dist < (float(near_factor) * place_success_dist_m)).float()
    height_err = (obj_base[:, 2] - obj_rest_z_base).clamp(min=0.0)
    near_gate = ever_grasped if ever_carried_near is None else ever_carried_near
    return (1.0 / (1.0 + 12.0 * height_err)) * near * near_gate.float()


def reward_lower(
    obj_z: torch.Tensor,
    prev_obj_z: torch.Tensor,
    obj_base: torch.Tensor,
    target_pos: torch.Tensor,
    carry: torch.Tensor,
    place_success_dist_m: float,
    *,
    near_factor: float = 1.5,
) -> torch.Tensor:
    """Reward downward object motion while carrying near the place target (Δxyz-aligned)."""
    xy_dist = torch.norm(obj_base[:, :2] - target_pos[:, :2], dim=-1)
    near = (xy_dist < (float(near_factor) * place_success_dist_m)).float()
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
    *,
    near_factor: float = 1.5,
    ever_carried_near: torch.Tensor | None = None,
) -> torch.Tensor:
    """Small bridge reward when holding on the table near the target (avoids carry cliff vacuum)."""
    xy_dist = torch.norm(obj_base[:, :2] - target_pos[:, :2], dim=-1)
    near = (xy_dist < (float(near_factor) * place_success_dist_m)).float()
    at_table = (torch.abs(obj_base[:, 2] - obj_rest_z_base) < 0.04).float()
    near_gate = ever_grasped if ever_carried_near is None else ever_carried_near
    return holding.float() * (~carry).float() * near * at_table * near_gate.float()


def reward_drop_far(
    obj_base: torch.Tensor,
    target_pos: torch.Tensor,
    carry: torch.Tensor,
    ever_grasped: torch.Tensor,
    place_success_dist_m: float,
    *,
    obj_rest_z_base: float | None = None,
) -> torch.Tensor:
    """Penalize early loss / early table contact before the placement neighborhood.

    Losing ``carry`` while still far remains the primary signal. When
    ``obj_rest_z_base`` is provided, also penalize bringing the object down to
    the table far from the target even if the gripper is still near it (the
    mid-path "place on the table then slide" shortcut).
    """

    xy_dist = torch.norm(obj_base[:, :2] - target_pos[:, :2], dim=-1)
    far = xy_dist >= place_success_dist_m
    # ``holding`` is deliberately not used for the lost-object branch: its
    # gap+proximity heuristic can remain true while the object rests on the
    # table. ``ever_grasped`` latches only after a real lift-carry.
    lost = ever_grasped & (~carry)
    penalty = (far & lost).float()
    if obj_rest_z_base is not None:
        at_table = torch.abs(obj_base[:, 2] - float(obj_rest_z_base)) < 0.04
        early_table = ever_grasped & far & at_table
        penalty = torch.maximum(penalty, early_table.float())
    return -penalty


def reward_place(
    obj_base: torch.Tensor,
    target_pos: torch.Tensor,
    carry: torch.Tensor,
    ever_grasped: torch.Tensor,
    obj_rest_z_base: float,
    place_success_dist_m: float,
    *,
    near_factor: float = 1.5,
    ever_carried_near: torch.Tensor | None = None,
) -> torch.Tensor:
    """Backward-compatible sum of place_xy + place_z."""
    return reward_place_xy(
        obj_base,
        target_pos,
        carry,
        ever_grasped,
        place_success_dist_m,
        ever_carried_near=ever_carried_near,
    ) + reward_place_z(
        obj_base,
        target_pos,
        ever_grasped,
        obj_rest_z_base,
        place_success_dist_m,
        near_factor=near_factor,
        ever_carried_near=ever_carried_near,
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
    ever_carried_near: torch.Tensor | None = None,
    obj_vel: torch.Tensor | None = None,
    release_speed_k: float = 20.0,
) -> torch.Tensor:
    """Open-gripper credit only when low and slow over the place target.

    When ``obj_vel`` is provided, multiplies by ``exp(-k * ||v||)`` so a throw-style
    open while the object is still moving earns near-zero dense release credit.
    """
    xy_dist = torch.norm(obj_base[:, :2] - target_pos[:, :2], dim=-1)
    near_target = (xy_dist < place_success_dist_m).float()
    low = (obj_base[:, 2] < (obj_rest_z_base + release_height_m)).float()
    at_table = torch.exp(-20.0 * torch.abs(obj_base[:, 2] - obj_rest_z_base))
    open_fraction = (gap / open_gap_m).clamp(0.0, 1.0)
    near_gate = ever_grasped if ever_carried_near is None else ever_carried_near
    rew = near_target * low * at_table * open_fraction * near_gate.float()
    if obj_vel is not None:
        speed = torch.norm(obj_vel, dim=-1)
        rew = rew * torch.exp(-float(release_speed_k) * speed)
    return rew


def reward_throw_release(
    obj_vel: torch.Tensor,
    gap: torch.Tensor,
    ever_carried_near: torch.Tensor,
    object_width: float,
    *,
    throw_speed_m_s: float = 0.15,
) -> torch.Tensor:
    """Penalize opening/releasing after a near-target carry while XY speed is high.

    Blocks the "throw into the circle and hope hold catches the bounce" shortcut.
    """
    opening = gap > (object_width + 0.01)
    horiz_speed = torch.norm(obj_vel[:, :2], dim=-1)
    fast = horiz_speed > float(throw_speed_m_s)
    return -(ever_carried_near & opening & fast).float()


def reward_push_before_grasp(
    obj_base: torch.Tensor,
    initial_obj_pos: torch.Tensor,
    ee_base: torch.Tensor,
    gap: torch.Tensor,
    ever_grasped: torch.Tensor,
    object_width: float,
    *,
    push_dist_m: float = 0.02,
    ee_near_m: float = 0.07,
) -> torch.Tensor:
    """Penalize shoving the cube sideways before a real grasp-and-lift.

    Blocks the "single finger pushes the object tens of mm, then closes" habit.
    """
    not_yet = ~ever_grasped
    horiz = torch.norm(obj_base[:, :2] - initial_obj_pos[:, :2], dim=-1)
    pushed = horiz > float(push_dist_m)
    ee_near = torch.norm(ee_base - obj_base, dim=-1) < float(ee_near_m)
    openish = gap > (float(object_width) + 0.005)
    return -(not_yet & pushed & ee_near & openish).float()


def reward_push_after_release(
    ee_base: torch.Tensor,
    obj_base: torch.Tensor,
    gap: torch.Tensor,
    ever_carried_near: torch.Tensor,
    carry: torch.Tensor,
    obj_rest_z_base: float,
    object_width: float,
) -> torch.Tensor:
    """Penalize keeping the open gripper against the object after a near-target carry.

    Blocks the post-release "use the fingers as a pusher" overshoot.
    """
    released = gap > (object_width + 0.02)
    at_table = torch.abs(obj_base[:, 2] - obj_rest_z_base) < 0.04
    ee_near_obj = torch.norm(ee_base - obj_base, dim=-1) < 0.07
    return -(ever_carried_near & released & at_table & ee_near_obj & (~carry)).float()


def reward_success(episode_success: torch.Tensor) -> torch.Tensor:
    return episode_success.float()


def reward_grasp_bonus(carry: torch.Tensor, prev_ever_grasped: torch.Tensor) -> torch.Tensor:
    """One-shot bonus on the first successful grasp-and-lift."""

    return (carry & (~prev_ever_grasped)).float()


def reward_pre_lift_xy_progress(
    previous_max_drag_m: torch.Tensor,
    current_max_drag_m: torch.Tensor,
    *,
    normalization_m: float = 0.01,
) -> torch.Tensor:
    """Penalize every newly reached pre-lift drag distance, independent of gripper gap."""

    scale = float(normalization_m)
    if scale <= 0.0:
        raise ValueError("normalization_m must be positive")
    return -torch.clamp(current_max_drag_m - previous_max_drag_m, min=0.0) / scale


def reward_clean_lift_bonus(
    carry: torch.Tensor,
    prev_ever_grasped: torch.Tensor,
    max_pre_lift_xy_m: torch.Tensor,
    *,
    max_drag_m: float = 0.005,
) -> torch.Tensor:
    """One-shot first-lift bonus only when the object was not dragged into the grasp."""

    return (carry & (~prev_ever_grasped) & (max_pre_lift_xy_m <= float(max_drag_m))).float()


def reward_clean_lift_quality(
    carry: torch.Tensor,
    prev_ever_grasped: torch.Tensor,
    max_pre_lift_xy_m: torch.Tensor,
    *,
    distance_scale_m: float = 0.04,
) -> torch.Tensor:
    """Continuous first-lift credit that improves smoothly as drag decreases."""

    scale = float(distance_scale_m)
    if scale <= 0.0:
        raise ValueError("distance_scale_m must be positive")
    first_lift = carry & (~prev_ever_grasped)
    return first_lift.float() * torch.exp(-max_pre_lift_xy_m / scale)


def reward_grasp_centering(
    carry: torch.Tensor,
    prev_ever_grasped: torch.Tensor,
    grasp_offset_xy_m: torch.Tensor,
    *,
    distance_scale_m: float = 0.005,
) -> torch.Tensor:
    """One-shot credit for lifting the cube centred between the pads.

    An off-centre grasp is carried unchanged to the target, so this is the term that
    ties clean grasping to placement accuracy rather than treating them separately.
    """

    scale = float(distance_scale_m)
    if scale <= 0.0:
        raise ValueError("distance_scale_m must be positive")
    first_lift = carry & (~prev_ever_grasped)
    return first_lift.float() * torch.exp(-grasp_offset_xy_m / scale)


def reward_valid_release(
    release_started: torch.Tensor,
    prev_release_started: torch.Tensor,
    release_valid: torch.Tensor,
) -> torch.Tensor:
    """One-shot credit for the first opening command when all release limits hold."""

    return (release_started & (~prev_release_started) & release_valid).float()


def reward_invalid_release(
    release_started: torch.Tensor,
    prev_release_started: torch.Tensor,
    release_violation: torch.Tensor,
) -> torch.Tensor:
    """One-shot penalty at opening intent; measured finger opening is deliberately irrelevant."""

    return -(release_started & (~prev_release_started) & release_violation).float()


def reward_release_quality(
    release_started: torch.Tensor,
    prev_release_started: torch.Tensor,
    release_xy_dist_m: torch.Tensor,
    release_height_error_m: torch.Tensor,
    release_speed_m_s: torch.Tensor,
    *,
    xy_scale_m: float = 0.04,
    height_scale_m: float = 0.04,
    speed_scale_m_s: float = 0.10,
) -> torch.Tensor:
    """Continuous one-shot release score used before strict release is reachable."""

    scales = (float(xy_scale_m), float(height_scale_m), float(speed_scale_m_s))
    if min(scales) <= 0.0:
        raise ValueError("release quality scales must be positive")
    release_event = release_started & (~prev_release_started)
    score = (
        torch.exp(-release_xy_dist_m / scales[0])
        + torch.exp(-release_height_error_m / scales[1])
        + torch.exp(-release_speed_m_s / scales[2])
    ) / 3.0
    return release_event.float() * score


def reward_release_readiness_progress(
    previous_xy_dist_m: torch.Tensor,
    previous_height_error_m: torch.Tensor,
    previous_speed_m_s: torch.Tensor,
    current_xy_dist_m: torch.Tensor,
    current_height_error_m: torch.Tensor,
    current_speed_m_s: torch.Tensor,
    ever_carried_near: torch.Tensor,
    prev_release_started: torch.Tensor,
    *,
    xy_scale_m: float = 0.04,
    height_scale_m: float = 0.04,
    speed_scale_m_s: float = 0.10,
) -> torch.Tensor:
    """Telescoping progress toward a jointly near, low, and slow release state."""

    scales = (float(xy_scale_m), float(height_scale_m), float(speed_scale_m_s))
    if min(scales) <= 0.0:
        raise ValueError("release readiness scales must be positive")

    def score(
        xy_dist_m: torch.Tensor,
        height_error_m: torch.Tensor,
        speed_m_s: torch.Tensor,
    ) -> torch.Tensor:
        return torch.exp(-xy_dist_m / scales[0] - height_error_m / scales[1] - speed_m_s / scales[2])

    progress = score(
        current_xy_dist_m,
        current_height_error_m,
        current_speed_m_s,
    ) - score(
        previous_xy_dist_m,
        previous_height_error_m,
        previous_speed_m_s,
    )
    active = ever_carried_near & (~prev_release_started)
    return progress * active.float()


def reward_premature_opening(
    action_gripper: torch.Tensor,
    ever_grasped: torch.Tensor,
    prev_release_started: torch.Tensor,
    release_ready: torch.Tensor,
) -> torch.Tensor:
    """Penalize positive opening command while the strict release state is not ready."""

    opening_amount = torch.clamp(action_gripper, min=0.0, max=1.0)
    active = ever_grasped & (~prev_release_started) & (~release_ready)
    return -opening_amount * active.float()


def reward_ready_opening(
    action_gripper: torch.Tensor,
    release_ready: torch.Tensor,
    prev_release_started: torch.Tensor,
) -> torch.Tensor:
    """Give signed dense feedback for the gripper action only when release-ready."""

    active = release_ready & (~prev_release_started)
    return torch.clamp(action_gripper, min=-1.0, max=1.0) * active.float()


def reward_release_clearance_opening(
    action_gripper: torch.Tensor,
    commanded_gap_m: torch.Tensor,
    release_valid: torch.Tensor,
    *,
    target_commanded_gap_m: float,
) -> torch.Tensor:
    """Continue opening after a valid release until actuator lag has safe clearance."""

    target_gap = float(target_commanded_gap_m)
    if target_gap <= 0.0:
        raise ValueError("target_commanded_gap_m must be positive")
    active = release_valid & (commanded_gap_m < target_gap)
    return torch.clamp(action_gripper, min=-1.0, max=1.0) * active.float()


def reward_setdown_action(
    action_z: torch.Tensor,
    obj_height_error_m: torch.Tensor,
    ever_carried_near: torch.Tensor,
    holding: torch.Tensor,
    prev_release_started: torch.Tensor,
    *,
    release_height_tolerance_m: float = 0.005,
) -> torch.Tensor:
    """Penalize upward Cartesian commands while a held object still needs set-down."""

    tolerance = float(release_height_tolerance_m)
    if tolerance <= 0.0:
        raise ValueError("release_height_tolerance_m must be positive")
    active = ever_carried_near & holding & (~prev_release_started) & (obj_height_error_m > tolerance)
    return -torch.clamp(action_z, min=0.0, max=1.0) * active.float()


def reward_landing_quality(
    near_table_entered: torch.Tensor,
    prev_near_table_entered: torch.Tensor,
    obj_vel: torch.Tensor,
    *,
    xy_scale_m_s: float = 0.03,
    down_scale_m_s: float = 0.05,
) -> torch.Tensor:
    """One-shot soft-landing score with severity information at first near-table entry."""

    scales = (float(xy_scale_m_s), float(down_scale_m_s))
    if min(scales) <= 0.0:
        raise ValueError("landing quality scales must be positive")
    first_entry = near_table_entered & (~prev_near_table_entered)
    xy_speed = torch.norm(obj_vel[:, :2], dim=-1)
    down_speed = torch.clamp(-obj_vel[:, 2], min=0.0)
    score = torch.exp(-xy_speed / scales[0] - down_speed / scales[1])
    return first_entry.float() * score


def reward_hard_landing(
    hard_landing_violation: torch.Tensor,
    prev_hard_landing_violation: torch.Tensor,
) -> torch.Tensor:
    """One-shot penalty on the first unsafe near-table velocity sample."""

    return -(hard_landing_violation & (~prev_hard_landing_violation)).float()


def reward_precision_progress(
    previous_xy_dist_m: torch.Tensor,
    current_xy_dist_m: torch.Tensor,
    ever_carried_near: torch.Tensor,
    *,
    normalization_m: float = 0.01,
) -> torch.Tensor:
    """Telescoping XY progress after entering the target neighborhood."""

    scale = float(normalization_m)
    if scale <= 0.0:
        raise ValueError("normalization_m must be positive")
    improvement = previous_xy_dist_m - current_xy_dist_m
    return improvement / scale * ever_carried_near.float()


def reward_transport_progress(
    previous_xy_dist_m: torch.Tensor,
    current_xy_dist_m: torch.Tensor,
    ever_grasped: torch.Tensor,
    holding: torch.Tensor,
    prev_release_started: torch.Tensor,
    *,
    normalization_m: float = 0.10,
) -> torch.Tensor:
    """Telescoping transport progress after a real lift, with no hover reward."""

    scale = float(normalization_m)
    if scale <= 0.0:
        raise ValueError("normalization_m must be positive")
    active = ever_grasped & holding & (~prev_release_started)
    return (previous_xy_dist_m - current_xy_dist_m) / scale * active.float()


def reward_descent_progress(
    previous_obj_z_m: torch.Tensor,
    current_obj_z_m: torch.Tensor,
    ever_carried_near: torch.Tensor,
    holding: torch.Tensor,
    prev_release_started: torch.Tensor,
    *,
    normalization_m: float = 0.04,
) -> torch.Tensor:
    """Telescoping held-object descent near target, penalizing upward oscillation."""

    scale = float(normalization_m)
    if scale <= 0.0:
        raise ValueError("normalization_m must be positive")
    active = ever_carried_near & holding & (~prev_release_started)
    return (previous_obj_z_m - current_obj_z_m) / scale * active.float()


def reward_near_target_speed(
    obj_base: torch.Tensor,
    target_pos: torch.Tensor,
    obj_vel: torch.Tensor,
    *,
    near_dist_m: float = 0.04,
    normalization_m_s: float = 0.03,
    max_height_error_m: float | None = None,
) -> torch.Tensor:
    """Penalize only speed above the safe set-down limit near the target."""

    speed_scale = float(normalization_m_s)
    if speed_scale <= 0.0:
        raise ValueError("normalization_m_s must be positive")
    xy_dist = torch.norm(obj_base[:, :2] - target_pos[:, :2], dim=-1)
    speed_cost = torch.clamp(
        torch.norm(obj_vel, dim=-1) / speed_scale - 1.0,
        min=0.0,
        max=10.0,
    )
    active = xy_dist <= float(near_dist_m)
    if max_height_error_m is not None:
        max_height = float(max_height_error_m)
        if max_height <= 0.0:
            raise ValueError("max_height_error_m must be positive")
        active &= torch.abs(obj_base[:, 2] - target_pos[:, 2]) <= max_height
    return -speed_cost * active.float()


def reward_near_table_xy_speed_margin(
    obj_base: torch.Tensor,
    obj_vel: torch.Tensor,
    ever_carried_near: torch.Tensor,
    *,
    obj_rest_z_base: float,
    near_table_height_m: float = 0.020,
    speed_limit_m_s: float = 0.030,
    max_normalized_speed: float = 4.0,
    safety_zone_frac: float = 0.0,
) -> torch.Tensor:
    """Continuous horizontal-speed cost on the exact hard-landing region.

    Unlike the one-shot landing score, this remains informative at every
    near-table transition.  Squaring speed normalized by the acceptance limit
    rewards a real safety margin even while the trajectory is still legal.

    With ``safety_zone_frac`` in (0, 1), speeds below that fraction of the
    limit are free; the cost then ramps quadratically to -1 exactly at the
    limit and keeps growing above it.  This pushes the optimal policy to hold
    a real margin under the acceptance limit (robust to execution noise)
    instead of riding right at the cliff.
    """

    height = float(near_table_height_m)
    limit = float(speed_limit_m_s)
    max_speed = float(max_normalized_speed)
    safety = float(safety_zone_frac)
    if min(height, limit, max_speed) <= 0.0:
        raise ValueError("near-table height, speed limit, and clamp must be positive")
    if not (0.0 <= safety < 1.0):
        raise ValueError("safety_zone_frac must be in [0, 1)")
    if obj_vel.shape[-1] != 3 or obj_base.shape != obj_vel.shape:
        raise ValueError("obj_base and obj_vel must have matching (N, 3) shapes")
    active = ever_carried_near & (obj_base[:, 2] <= float(obj_rest_z_base) + height)
    speed = torch.norm(obj_vel[:, :2], dim=-1)
    excess = (speed - safety * limit).clamp(min=0.0)
    normalized = (excess / (limit * (1.0 - safety))).clamp(max=max_speed)
    return -torch.square(normalized) * active.float()


def reward_near_table_down_speed_margin(
    obj_base: torch.Tensor,
    obj_vel: torch.Tensor,
    ever_carried_near: torch.Tensor,
    *,
    obj_rest_z_base: float,
    near_table_height_m: float = 0.020,
    speed_limit_m_s: float = 0.050,
    max_normalized_speed: float = 4.0,
    safety_zone_frac: float = 0.0,
) -> torch.Tensor:
    """Continuous downward-speed cost separated from horizontal velocity.

    Same safety-zone semantics as :func:`reward_near_table_xy_speed_margin`:
    with ``safety_zone_frac`` in (0, 1), descent speeds below that fraction of
    the limit are free and the cost ramps to -1 exactly at the limit.
    """

    height = float(near_table_height_m)
    limit = float(speed_limit_m_s)
    max_speed = float(max_normalized_speed)
    safety = float(safety_zone_frac)
    if min(height, limit, max_speed) <= 0.0:
        raise ValueError("near-table height, speed limit, and clamp must be positive")
    if not (0.0 <= safety < 1.0):
        raise ValueError("safety_zone_frac must be in [0, 1)")
    if obj_vel.shape[-1] != 3 or obj_base.shape != obj_vel.shape:
        raise ValueError("obj_base and obj_vel must have matching (N, 3) shapes")
    active = ever_carried_near & (obj_base[:, 2] <= float(obj_rest_z_base) + height)
    down_speed = torch.clamp(-obj_vel[:, 2], min=0.0)
    excess = (down_speed - safety * limit).clamp(min=0.0)
    normalized = (excess / (limit * (1.0 - safety))).clamp(max=max_speed)
    return -torch.square(normalized) * active.float()


def reward_near_table_xy_action(
    action_xy: torch.Tensor,
    obj_height_error_m: torch.Tensor,
    ever_carried_near: torch.Tensor,
    *,
    max_height_error_m: float = 0.025,
) -> torch.Tensor:
    """Penalize horizontal commands only in the final set-down/contact region."""

    max_height = float(max_height_error_m)
    if max_height <= 0.0:
        raise ValueError("max_height_error_m must be positive")
    if action_xy.ndim != 2 or action_xy.shape[-1] != 2:
        raise ValueError("action_xy must have shape (N, 2)")
    active = ever_carried_near & (obj_height_error_m <= max_height)
    return -torch.sum(torch.square(action_xy), dim=-1) * active.float()


def reward_post_release_contact(
    ee_base: torch.Tensor,
    obj_base: torch.Tensor,
    release_started: torch.Tensor,
    carry: torch.Tensor,
    obj_rest_z_base: float,
    *,
    ee_near_m: float = 0.07,
    table_height_tolerance_m: float = 0.04,
) -> torch.Tensor:
    """Penalize continued gripper contact from opening intent onward."""

    at_table = torch.abs(obj_base[:, 2] - float(obj_rest_z_base)) <= float(table_height_tolerance_m)
    ee_near_obj = torch.norm(ee_base - obj_base, dim=-1) < float(ee_near_m)
    return -(release_started & at_table & ee_near_obj & (~carry)).float()


def reward_post_release_clearance_progress(
    previous_clearance_m: torch.Tensor,
    current_clearance_m: torch.Tensor,
    release_valid: torch.Tensor,
    *,
    max_clearance_m: float = 0.050,
) -> torch.Tensor:
    """Reward only net upward gripper/object separation after a valid release.

    The clipped potential is bounded to one unit over an episode.  Moving back
    toward the object reverses earlier credit, and remaining still earns nothing.
    """

    maximum = float(max_clearance_m)
    if maximum <= 0.0:
        raise ValueError("max_clearance_m must be positive")
    if previous_clearance_m.shape != current_clearance_m.shape:
        raise ValueError("previous and current clearance tensors must have matching shapes")
    previous = previous_clearance_m.clamp(min=0.0, max=maximum)
    current = current_clearance_m.clamp(min=0.0, max=maximum)
    return (current - previous) / maximum * release_valid.float()


def reward_post_release_recontact(recontact_event: torch.Tensor) -> torch.Tensor:
    """One-shot penalty when a fingertip touches the object after full clearance."""

    return -recontact_event.float()


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


def annealed_frac(initial: float, final: float, progress: float) -> float:
    """Linearly interpolate a fraction with progress clamped to [0, 1]."""

    p = min(1.0, max(0.0, float(progress)))
    return float(initial) + (float(final) - float(initial)) * p


def sample_reset_phase_masks(
    u: torch.Tensor,
    place_frac: float,
    grasp_frac: float,
    carry_frac: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Partition reset samples into disjoint place, carry, and grasp phases."""

    total = float(place_frac) + float(carry_frac) + float(grasp_frac)
    if min(place_frac, carry_frac, grasp_frac) < 0.0 or total > 1.0 + 1e-9:
        raise ValueError("reset phase fractions must be non-negative and sum to at most 1")
    place_mask = u < place_frac
    carry_lo = place_frac
    carry_mask = (u >= carry_lo) & (u < carry_lo + carry_frac)
    grasp_lo = place_frac + carry_frac
    grasp_mask = (u >= grasp_lo) & (u < grasp_lo + grasp_frac)
    return place_mask, carry_mask, grasp_mask


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
    edge_fraction: float = 0.0,
    randomize_target: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 <= edge_fraction <= 1.0:
        raise ValueError("edge_fraction must be in [0, 1]")
    if stage == 0:
        obj = fixed_obj.unsqueeze(0).expand(n, 3).clone()
        target = fixed_target.unsqueeze(0).expand(n, 3).clone()
        return obj, target

    # All stages expand around the canonical task geometry.  In particular, a
    # +Y pick-place task never detours through the legacy +X curriculum.
    if stage == 1:
        obj_jitter = (torch.rand(n, 3, device=device, dtype=dtype) - 0.5) * 0.02
        target_jitter = (torch.rand(n, 3, device=device, dtype=dtype) - 0.5) * 0.02
        obj_jitter[:, 2] = 0.0
        target_jitter[:, 2] = 0.0
        obj = torch.clamp(fixed_obj.unsqueeze(0) + obj_jitter, obj_spawn_lower, obj_spawn_upper)
        target = torch.clamp(fixed_target.unsqueeze(0) + target_jitter, target_spawn_lower, target_spawn_upper)
    elif stage == 2:
        obj_half_span = 0.25 * (obj_spawn_upper - obj_spawn_lower)
        target_half_span = 0.25 * (target_spawn_upper - target_spawn_lower)
        obj = fixed_obj.unsqueeze(0) + (torch.rand(n, 3, device=device, dtype=dtype) * 2.0 - 1.0) * obj_half_span
        target = (
            fixed_target.unsqueeze(0) + (torch.rand(n, 3, device=device, dtype=dtype) * 2.0 - 1.0) * target_half_span
        )
        obj = torch.clamp(obj, obj_spawn_lower, obj_spawn_upper)
        target = torch.clamp(target, target_spawn_lower, target_spawn_upper)
    elif stage == 3:
        obj_half_span = 0.5 * (obj_spawn_upper - obj_spawn_lower)
        target_half_span = 0.5 * (target_spawn_upper - target_spawn_lower)
        obj = fixed_obj.unsqueeze(0) + (torch.rand(n, 3, device=device, dtype=dtype) * 2.0 - 1.0) * obj_half_span
        target = (
            fixed_target.unsqueeze(0) + (torch.rand(n, 3, device=device, dtype=dtype) * 2.0 - 1.0) * target_half_span
        )
        obj = torch.clamp(obj, obj_spawn_lower, obj_spawn_upper)
        target = torch.clamp(target, target_spawn_lower, target_spawn_upper)
    else:
        rand_obj = torch.rand(n, 3, device=device, dtype=dtype)
        obj = obj_spawn_lower + rand_obj * (obj_spawn_upper - obj_spawn_lower)
        rand_target = torch.rand(n, 3, device=device, dtype=dtype)
        target = target_spawn_lower + rand_target * (target_spawn_upper - target_spawn_lower)
    if edge_fraction > 0.0:
        if stage == 1:
            obj_half_span = target_half_span = torch.full_like(fixed_obj, 0.01)
            obj_half_span[2] = 0.0
            target_half_span[2] = 0.0
            obj_lower = torch.maximum(obj_spawn_lower, fixed_obj - obj_half_span)
            obj_upper = torch.minimum(obj_spawn_upper, fixed_obj + obj_half_span)
            target_lower = torch.maximum(target_spawn_lower, fixed_target - target_half_span)
            target_upper = torch.minimum(target_spawn_upper, fixed_target + target_half_span)
        elif stage in (2, 3):
            scale = 0.25 if stage == 2 else 0.5
            obj_half_span = scale * (obj_spawn_upper - obj_spawn_lower)
            target_half_span = scale * (target_spawn_upper - target_spawn_lower)
            obj_lower = torch.maximum(obj_spawn_lower, fixed_obj - obj_half_span)
            obj_upper = torch.minimum(obj_spawn_upper, fixed_obj + obj_half_span)
            target_lower = torch.maximum(target_spawn_lower, fixed_target - target_half_span)
            target_upper = torch.minimum(target_spawn_upper, fixed_target + target_half_span)
        else:
            obj_lower, obj_upper = obj_spawn_lower, obj_spawn_upper
            target_lower, target_upper = target_spawn_lower, target_spawn_upper
        edge_mask = torch.rand(n, device=device) < edge_fraction
        obj_edges = torch.where(
            torch.rand(n, 2, device=device) < 0.5,
            obj_lower[:2],
            obj_upper[:2],
        )
        target_edges = torch.where(
            torch.rand(n, 2, device=device) < 0.5,
            target_lower[:2],
            target_upper[:2],
        )
        obj[edge_mask, :2] = obj_edges[edge_mask]
        target[edge_mask, :2] = target_edges[edge_mask]
    if not randomize_target:
        target = fixed_target.unsqueeze(0).expand(n, 3).clone()
    return obj, target


def next_curriculum_stage(
    stage: int,
    grasp_rate: float,
    place_rate: float,
    grasp_count: int,
    place_count: int,
    *,
    min_count: int = 500,
    stage0_grasp_rate: float = 0.50,
    stage1_grasp_rate: float = 0.70,
    stage2_place_rate: float = 0.60,
    stage3_place_rate: float = 0.50,
    max_stage: int = 4,
) -> int:
    """Pure stage transition; counters reset remain the env's responsibility."""
    if not 0 <= max_stage <= 4:
        raise ValueError("max_stage must be in [0, 4]")
    candidate = stage
    if grasp_count >= min_count:
        if stage == 0 and grasp_rate >= stage0_grasp_rate:
            candidate = 1
        elif stage == 1 and grasp_rate >= stage1_grasp_rate:
            candidate = 2
    if place_count >= min_count:
        if stage == 2 and place_rate >= stage2_place_rate:
            candidate = 3
        elif stage == 3 and place_rate >= stage3_place_rate:
            candidate = 4
    return min(candidate, max_stage)
