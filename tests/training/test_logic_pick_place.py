"""CPU tests for the Genesis-free pick-place numeric core."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ufactory.grippers.g2 import GRIPPER_G2_OPEN_GAP_M, GRIPPER_G2_SIM_CLOSE_DRIVE
from ufactory.training.logic import (
    action_penalty,
    action_to_cartesian_delta,
    annealed_frac,
    arm_target_ee_pos,
    arm_target_qpos,
    cartesian_delta_to_action,
    drive_to_gap_m,
    gap_m_to_drive,
    leashed_ee_setpoint,
    near_target_attenuate,
    next_curriculum_stage,
    normalized_pick_place_contact_features,
    normalized_pick_place_layout_offsets,
    normalized_pick_place_quality_features,
    normalized_ee_setpoint_residual,
    reward_grasp_centering,
    pick_place_observation,
    reward_clean_lift_bonus,
    reward_clean_lift_quality,
    reward_descent_progress,
    reward_grasp,
    reward_grasp_bonus,
    reward_grasp_gap_progress,
    reward_grasp_lift_action,
    reward_grasp_lift_progress,
    reward_grasp_ready_closing,
    reward_grasp_settle_action,
    reward_drop_far,
    reward_approach_potential,
    reward_keypoints,
    reward_lift,
    reward_lower,
    reward_hard_landing,
    reward_invalid_release,
    reward_landing_quality,
    reward_near_table_xy_action,
    reward_near_table_down_speed_margin,
    reward_near_table_xy_speed_margin,
    reward_near_target_speed,
    reward_place,
    reward_place_xy,
    reward_place_z,
    reward_post_release_clearance_progress,
    reward_post_release_contact,
    reward_post_release_recontact,
    reward_premature_opening,
    reward_precision_progress,
    reward_pre_lift_xy_progress,
    reward_push_after_release,
    reward_push_before_grasp,
    reward_release_clearance_opening,
    reward_ready_opening,
    reward_reach,
    reward_release,
    reward_release_quality,
    reward_release_readiness_progress,
    reward_setdown_action,
    reward_success,
    reward_throw_release,
    reward_transport_progress,
    reward_valid_release,
    sample_object_and_target,
    sample_reset_phase_masks,
    update_task_state,
)

_LOGIC_DIR = Path(__file__).resolve().parents[2] / "ufactory" / "training" / "logic"
OPEN = GRIPPER_G2_OPEN_GAP_M
CLOSE_DRIVE = GRIPPER_G2_SIM_CLOSE_DRIVE


def test_logic_pick_place_does_not_import_genesis() -> None:
    for name in ("pick_place.py",):
        text = (_LOGIC_DIR / name).read_text(encoding="utf-8")
        assert "import genesis" not in text
        assert "from genesis" not in text


def test_action_penalty_is_non_positive() -> None:
    assert torch.all(action_penalty(torch.randn(3, 6)) <= 0.0)


def test_observation_dim_is_30() -> None:
    n = 2
    obs = pick_place_observation(
        torch.zeros(n, 6),
        torch.zeros(n, 6),
        torch.zeros(n, 3),
        torch.zeros(n),
        torch.zeros(n, 3),
        torch.ones(n, 3),
        torch.zeros(n, dtype=torch.bool),
        torch.zeros(n, dtype=torch.bool),
    )
    assert obs.shape == (n, 30)


def test_observation_dim_is_35_with_controller_state() -> None:
    n = 2
    previous_action = torch.arange(8, dtype=torch.float32).reshape(n, 4)
    commanded_gap = torch.tensor([0.084, 0.022])
    obs = pick_place_observation(
        torch.zeros(n, 6),
        torch.zeros(n, 6),
        torch.zeros(n, 3),
        torch.zeros(n),
        torch.zeros(n, 3),
        torch.ones(n, 3),
        torch.zeros(n, dtype=torch.bool),
        torch.zeros(n, dtype=torch.bool),
        commanded_gap,
        previous_action,
    )
    assert obs.shape == (n, 35)
    assert torch.allclose(
        obs[:, :30],
        pick_place_observation(
            torch.zeros(n, 6),
            torch.zeros(n, 6),
            torch.zeros(n, 3),
            torch.zeros(n),
            torch.zeros(n, 3),
            torch.ones(n, 3),
            torch.zeros(n, dtype=torch.bool),
            torch.zeros(n, dtype=torch.bool),
        ),
    )
    assert torch.allclose(obs[:, 30], commanded_gap)
    assert torch.allclose(obs[:, 31:], previous_action)


def test_observation_appends_normalized_setpoint_residual_after_existing_prefix() -> None:
    n = 2
    base = pick_place_observation(
        torch.zeros(n, 6),
        torch.zeros(n, 6),
        torch.zeros(n, 3),
        torch.zeros(n),
        torch.zeros(n, 3),
        torch.ones(n, 3),
        torch.zeros(n, dtype=torch.bool),
        torch.zeros(n, dtype=torch.bool),
        torch.zeros(n),
        torch.zeros(n, 4),
        quality_features=torch.zeros(n, 6),
        contact_features=torch.zeros(n, 3),
    )
    residual = normalized_ee_setpoint_residual(
        torch.tensor([[0.01, -0.02, 0.03], [0.05, 0.00, -0.05]]),
        torch.zeros(n, 3),
        0.02,
    )
    obs = pick_place_observation(
        torch.zeros(n, 6),
        torch.zeros(n, 6),
        torch.zeros(n, 3),
        torch.zeros(n),
        torch.zeros(n, 3),
        torch.ones(n, 3),
        torch.zeros(n, dtype=torch.bool),
        torch.zeros(n, dtype=torch.bool),
        torch.zeros(n),
        torch.zeros(n, 4),
        quality_features=torch.zeros(n, 6),
        contact_features=torch.zeros(n, 3),
        normalized_setpoint_residual=residual,
    )
    assert base.shape == (n, 44)
    assert obs.shape == (n, 47)
    assert torch.equal(obs[:, :44], base)
    assert torch.allclose(
        obs[:, 44:],
        torch.tensor([[0.5, -1.0, 1.5], [2.0, 0.0, -2.0]]),
    )


def test_normalized_setpoint_residual_validates_shape_and_scale() -> None:
    with pytest.raises(ValueError, match="positive"):
        normalized_ee_setpoint_residual(torch.zeros(1, 3), torch.zeros(1, 3), 0.0)
    with pytest.raises(ValueError, match="matching"):
        normalized_ee_setpoint_residual(torch.zeros(1, 3), torch.zeros(2, 3), 0.02)


def test_observation_dim_is_41_with_quality_state_and_legacy_prefix_unchanged() -> None:
    n = 2
    base = pick_place_observation(
        torch.zeros(n, 6),
        torch.zeros(n, 6),
        torch.zeros(n, 3),
        torch.zeros(n),
        torch.zeros(n, 3),
        torch.ones(n, 3),
        torch.zeros(n, dtype=torch.bool),
        torch.zeros(n, dtype=torch.bool),
        torch.tensor([0.084, 0.022]),
        torch.zeros(n, 4),
    )
    features = normalized_pick_place_quality_features(
        torch.tensor([[0.25, -0.50, 0.0], [0.0, 0.0, 1.25]]),
        torch.tensor([True, False]),
        torch.tensor([False, True]),
        torch.tensor([False, True]),
        velocity_scale_m_s=0.25,
    )
    obs = pick_place_observation(
        torch.zeros(n, 6),
        torch.zeros(n, 6),
        torch.zeros(n, 3),
        torch.zeros(n),
        torch.zeros(n, 3),
        torch.ones(n, 3),
        torch.zeros(n, dtype=torch.bool),
        torch.zeros(n, dtype=torch.bool),
        torch.tensor([0.084, 0.022]),
        torch.zeros(n, 4),
        quality_features=features,
    )
    assert obs.shape == (n, 41)
    assert torch.equal(obs[:, :35], base)
    assert torch.allclose(obs[:, 35:38], torch.tensor([[1.0, -2.0, 0.0], [0.0, 0.0, 4.0]]))
    assert torch.equal(obs[:, 38:], torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]))


def test_normalized_layout_offsets_append_six_order_one_features() -> None:
    obj = torch.tensor([[0.315, -0.025, 0.015]])
    target = torch.tensor([[0.285, 0.325, 0.015]])
    fixed_obj = torch.tensor([0.300, 0.000, 0.015])
    fixed_target = torch.tensor([0.300, 0.300, 0.015])
    obj_lower = torch.tensor([0.280, -0.050, 0.015])
    obj_upper = torch.tensor([0.340, 0.050, 0.015])
    target_lower = torch.tensor([0.280, 0.250, 0.015])
    target_upper = torch.tensor([0.340, 0.350, 0.015])
    offsets = normalized_pick_place_layout_offsets(
        obj,
        target,
        fixed_obj,
        fixed_target,
        obj_lower,
        obj_upper,
        target_lower,
        target_upper,
    )
    assert offsets.shape == (1, 6)
    assert torch.allclose(offsets[0, :4], torch.tensor([0.5, -0.5, -0.5, 0.5]))
    assert torch.allclose(offsets[0, 4:], torch.tensor([-0.5, 0.5]))

    obs = pick_place_observation(
        torch.zeros(1, 6),
        torch.zeros(1, 6),
        torch.zeros(1, 3),
        torch.zeros(1),
        obj,
        target,
        torch.zeros(1, dtype=torch.bool),
        torch.zeros(1, dtype=torch.bool),
        torch.zeros(1),
        torch.zeros(1, 4),
        offsets,
    )
    assert obs.shape == (1, 41)
    assert torch.allclose(
        obs[:, :35],
        pick_place_observation(
            torch.zeros(1, 6),
            torch.zeros(1, 6),
            torch.zeros(1, 3),
            torch.zeros(1),
            obj,
            target,
            torch.zeros(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
            torch.zeros(1),
            torch.zeros(1, 4),
        ),
    )
    assert torch.allclose(obs[:, 35:], offsets)


def test_scripted_action_hint_appends_after_complete_legacy_and_layout_prefix() -> None:
    n = 2
    layout = torch.tensor([[0.25] * 6, [-0.5] * 6])
    guide = torch.tensor([[0.1, -0.2, 0.3, -0.4], [-1.0, 0.5, 0.0, 1.0]])
    prefix = pick_place_observation(
        torch.zeros(n, 6),
        torch.zeros(n, 6),
        torch.zeros(n, 3),
        torch.zeros(n),
        torch.zeros(n, 3),
        torch.ones(n, 3),
        torch.zeros(n, dtype=torch.bool),
        torch.zeros(n, dtype=torch.bool),
        torch.zeros(n),
        torch.zeros(n, 4),
        normalized_layout_offsets=layout,
        quality_features=torch.zeros(n, 6),
        contact_features=torch.zeros(n, 3),
    )
    obs = pick_place_observation(
        torch.zeros(n, 6),
        torch.zeros(n, 6),
        torch.zeros(n, 3),
        torch.zeros(n),
        torch.zeros(n, 3),
        torch.ones(n, 3),
        torch.zeros(n, dtype=torch.bool),
        torch.zeros(n, dtype=torch.bool),
        torch.zeros(n),
        torch.zeros(n, 4),
        normalized_layout_offsets=layout,
        quality_features=torch.zeros(n, 6),
        contact_features=torch.zeros(n, 3),
        scripted_action_hint=guide,
    )

    assert prefix.shape == (n, 50)
    assert obs.shape == (n, 54)
    assert torch.equal(obs[:, :50], prefix)
    assert torch.equal(obs[:, 50:], guide)


def test_fixed_episode_layout_offsets_stay_zero_after_object_motion() -> None:
    fixed_obj = torch.tensor([0.300, 0.000, 0.015])
    fixed_target = torch.tensor([0.300, 0.300, 0.015])
    offsets = normalized_pick_place_layout_offsets(
        fixed_obj.unsqueeze(0),
        fixed_target.unsqueeze(0),
        fixed_obj,
        fixed_target,
        torch.tensor([0.280, -0.050, 0.015]),
        torch.tensor([0.340, 0.050, 0.015]),
        torch.tensor([0.280, 0.250, 0.015]),
        torch.tensor([0.340, 0.350, 0.015]),
    )
    live_object_after_transport = fixed_target.unsqueeze(0)
    assert not torch.equal(live_object_after_transport, fixed_obj.unsqueeze(0))
    assert torch.equal(offsets, torch.zeros_like(offsets))


def test_episode_layout_offsets_remain_available_in_target_neighbourhood() -> None:
    fixed_obj = torch.tensor([0.300, 0.000, 0.015])
    fixed_target = torch.tensor([0.300, 0.300, 0.015])
    object_starts = torch.tensor([[0.330, 0.025, 0.015], [0.330, 0.025, 0.015]])
    targets = fixed_target.unsqueeze(0).expand(2, 3)
    offsets = normalized_pick_place_layout_offsets(
        object_starts,
        targets,
        fixed_obj,
        fixed_target,
        torch.tensor([0.280, -0.050, 0.015]),
        torch.tensor([0.340, 0.050, 0.015]),
        torch.tensor([0.280, 0.250, 0.015]),
        torch.tensor([0.340, 0.350, 0.015]),
    )
    assert torch.count_nonzero(offsets[0]).item() == 4
    assert torch.equal(offsets[1], offsets[0])


def test_layout_offsets_follow_the_complete_existing_observation_prefix() -> None:
    n = 2
    common = dict(
        joint_pos=torch.zeros(n, 6),
        joint_vel=torch.zeros(n, 6),
        ee_base=torch.zeros(n, 3),
        gripper_gap=torch.zeros(n),
        obj_base=torch.zeros(n, 3),
        target_base=torch.ones(n, 3),
        grasped=torch.zeros(n, dtype=torch.bool),
        ever_grasped=torch.zeros(n, dtype=torch.bool),
        commanded_gap=torch.zeros(n),
        previous_action=torch.zeros(n, 4),
        quality_features=torch.randn(n, 6),
        contact_features=torch.randn(n, 3),
    )
    base = pick_place_observation(**common)
    offsets = torch.randn(n, 6)
    expanded = pick_place_observation(**common, normalized_layout_offsets=offsets)
    assert base.shape == (n, 44)
    assert expanded.shape == (n, 50)
    assert torch.equal(expanded[:, :44], base)
    assert torch.equal(expanded[:, 44:], offsets)


def test_gap_drive_roundtrip_endpoints() -> None:
    open_t = torch.tensor([OPEN])
    closed_t = torch.tensor([0.0])
    assert gap_m_to_drive(open_t, 0.0, OPEN).item() == pytest.approx(0.0, abs=1e-6)
    assert gap_m_to_drive(closed_t, 0.0, OPEN).item() == pytest.approx(CLOSE_DRIVE, abs=1e-6)
    assert drive_to_gap_m(torch.tensor([0.0]), OPEN).item() == pytest.approx(OPEN, abs=1e-6)
    assert drive_to_gap_m(torch.tensor([CLOSE_DRIVE]), OPEN).item() == pytest.approx(0.0, abs=1e-6)


def test_arm_target_qpos_clips_joint_delta() -> None:
    current = torch.zeros(1, 6)
    huge = torch.full((1, 6), 10.0)
    target, unclipped = arm_target_qpos(current, huge, action_scale=0.05, max_joint_delta_rad=0.02)
    assert torch.all(unclipped.abs() > 0.02)
    assert torch.all((target - current).abs() <= 0.02 + 1e-6)


def test_arm_target_ee_pos_clips_cartesian_delta() -> None:
    current = torch.zeros(1, 3)
    huge = torch.full((1, 3), 10.0)
    target, unclipped = arm_target_ee_pos(current, huge, action_scale=0.05, max_cartesian_delta_m=0.02)
    assert torch.all(unclipped.abs() > 0.02)
    assert torch.all((target - current).abs() <= 0.02 + 1e-6)


def test_reward_close_gripper_peaks_at_target_gap() -> None:
    from ufactory.training.logic import reward_close_gripper

    ee = torch.zeros(3, 3)
    grasp = torch.zeros(3, 3)  # at grasp (full proximity)
    gaps = torch.tensor([0.022, 0.0, 0.084])
    ever = torch.zeros(3, dtype=torch.bool)
    rew = reward_close_gripper(ee, grasp, gaps, OPEN, ever, target_gap_m=0.022)
    assert rew[0] > rew[1]
    assert rew[0] > rew[2]
    # Far from grasp: no close credit even at target gap
    far_ee = torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    far_rew = reward_close_gripper(far_ee, grasp, gaps, OPEN, ever, target_gap_m=0.022)
    assert torch.all(far_rew == 0.0)


def test_reward_reach_closer_is_larger_and_zero_after_grasp() -> None:
    ee = torch.zeros(2, 3)
    grasp = torch.tensor([[0.01, 0.0, 0.0], [0.5, 0.0, 0.0]])
    ever = torch.zeros(2, dtype=torch.bool)
    rew = reward_reach(ee, grasp, ever)
    assert rew[0] > rew[1]
    ever_true = torch.ones(2, dtype=torch.bool)
    assert torch.all(reward_reach(ee, grasp, ever_true) == 0.0)


def test_approach_potential_rewards_progress_and_penalizes_retreat() -> None:
    previous = torch.tensor([0.20, 0.10, 0.10])
    current = torch.tensor([0.10, 0.20, 0.05])
    carry = torch.tensor([False, False, True])
    reward = reward_approach_potential(previous, current, carry)
    assert reward.tolist() == pytest.approx([0.10, -0.10, 0.0])


def test_grasp_bonus_fires_once() -> None:
    carry = torch.tensor([True, True, False])
    previous = torch.tensor([False, True, False])
    assert reward_grasp_bonus(carry, previous).tolist() == pytest.approx([1.0, 0.0, 0.0])


def test_reward_success_is_unit() -> None:
    ok = torch.tensor([True, False])
    rew = reward_success(ok)
    assert rew[0].item() == pytest.approx(1.0)
    assert rew[1].item() == pytest.approx(0.0)


def test_reward_keypoints_peaks_when_aligned() -> None:
    grasp = torch.zeros(2, 3)
    near = torch.zeros(2, 3)
    far = torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    ever = torch.zeros(2, dtype=torch.bool)
    near_rew = reward_keypoints(near, grasp, ever)
    far_rew = reward_keypoints(far, grasp, ever)
    assert near_rew[0] > far_rew[0]
    assert torch.all(reward_keypoints(near, grasp, torch.ones(2, dtype=torch.bool)) == 0.0)


def test_reward_place_survives_setdown() -> None:
    """After ever_grasped, place reward must stay positive when object is on the table."""
    obj = torch.tensor([[0.30, 0.28, 0.015]])  # near target, on table
    target = torch.tensor([[0.30, 0.30, 0.015]])
    carry = torch.zeros(1, dtype=torch.bool)
    ever = torch.ones(1, dtype=torch.bool)
    rew = reward_place(obj, target, carry, ever, obj_rest_z_base=0.015, place_success_dist_m=0.04)
    assert rew[0].item() > 0.0
    assert reward_place_z(obj, target, ever, 0.015, 0.04)[0].item() > 0.0


def test_reward_grasp_is_carry_only() -> None:
    carry = torch.tensor([True, False])
    rew = reward_grasp(carry)
    assert rew.tolist() == pytest.approx([1.0, 0.0])


def test_reward_grasp_attenuates_near_place_target() -> None:
    carry = torch.ones(2, dtype=torch.bool)
    target = torch.tensor([[0.30, 0.30, 0.015], [0.30, 0.30, 0.015]])
    far = torch.tensor([[0.30, 0.00, 0.09], [0.30, 0.30, 0.09]])  # second is on-target XY
    rew = reward_grasp(carry, far, target, place_success_dist_m=0.04)
    assert rew[0].item() == pytest.approx(1.0)
    assert rew[1].item() == pytest.approx(0.0)


def test_hover_basin_lower_beats_hover_dense_terms() -> None:
    """Descending 1 cm near target must beat staying put (anti-hover invariant)."""
    target = torch.tensor([[0.30, 0.30, 0.015]])
    hover = torch.tensor([[0.30, 0.30, 0.090]])
    lower_pos = torch.tensor([[0.30, 0.30, 0.080]])
    carry = torch.ones(1, dtype=torch.bool)
    ever = torch.ones(1, dtype=torch.bool)
    rest = 0.015
    d = 0.04

    def dense(obj: torch.Tensor, prev_z: torch.Tensor) -> float:
        g = reward_grasp(carry, obj, target, d)
        lift = reward_lift(obj[:, 2], rest, 0.08, carry, obj, target, d)
        pxy = reward_place_xy(obj, target, carry, ever)
        pz = reward_place_z(obj, target, ever, rest, d)
        low = reward_lower(obj[:, 2], prev_z, obj, target, carry, d)
        return float((g + lift + pxy + pz + low)[0].item())

    hover_rew = dense(hover, hover[:, 2])
    lower_rew = dense(lower_pos, hover[:, 2])
    assert lower_rew > hover_rew


def test_place_xy_does_not_reward_early_drop_far_from_target() -> None:
    target = torch.tensor([[0.30, 0.30, 0.015]])
    far = torch.tensor([[0.27, 0.04, 0.015]])
    near = torch.tensor([[0.30, 0.28, 0.015]])
    dropped = torch.zeros(1, dtype=torch.bool)
    ever = torch.ones(1, dtype=torch.bool)
    near_carry = torch.ones(1, dtype=torch.bool)
    no_near = torch.zeros(1, dtype=torch.bool)

    assert reward_place_xy(far, target, dropped, ever, 0.04).item() == 0.0
    assert reward_place_xy(near, target, dropped, ever, 0.04).item() > 0.0
    assert reward_place_xy(far, target, ~dropped, ever, 0.04).item() > 0.0
    # Slide-in without ever carrying near must not earn the post-drop bridge.
    assert reward_place_xy(near, target, dropped, ever, 0.04, ever_carried_near=no_near).item() == 0.0
    assert reward_place_xy(near, target, dropped, ever, 0.04, ever_carried_near=near_carry).item() > 0.0


def test_drop_far_penalizes_only_lost_object_outside_place_neighborhood() -> None:
    target = torch.tensor([[0.30, 0.30, 0.015]]).expand(3, 3)
    obj = torch.tensor(
        [
            [0.27, 0.04, 0.015],
            [0.30, 0.28, 0.015],
            [0.27, 0.04, 0.08],
        ]
    )
    carry = torch.tensor([False, False, True])
    ever = torch.ones(3, dtype=torch.bool)
    penalty = reward_drop_far(obj, target, carry, ever, 0.04)
    assert penalty.tolist() == [-1.0, 0.0, 0.0]

    boundary = torch.tensor([[0.30, 0.41, 0.015]])
    assert (
        reward_drop_far(
            boundary,
            target[:1],
            torch.zeros(1, dtype=torch.bool),
            torch.ones(1, dtype=torch.bool),
            0.04,
        ).item()
        == -1.0
    )


def test_drop_far_penalizes_early_table_contact_while_still_holding() -> None:
    """Mid-path table contact must be penalized even if carry/holding stays true."""
    target = torch.tensor([[0.30, 0.30, 0.015]])
    mid_table = torch.tensor([[0.30, 0.15, 0.015]])  # 150 mm away, on table
    hover = torch.tensor([[0.30, 0.15, 0.08]])
    carry = torch.ones(1, dtype=torch.bool)
    ever = torch.ones(1, dtype=torch.bool)
    assert reward_drop_far(mid_table, target, carry, ever, 0.02, obj_rest_z_base=0.015).item() == -1.0
    assert reward_drop_far(hover, target, carry, ever, 0.02, obj_rest_z_base=0.015).item() == 0.0


def test_place_z_and_lower_gated_by_near_factor() -> None:
    target = torch.tensor([[0.30, 0.30, 0.015]])
    # 50 mm away: inside old 3x*20mm=60mm gate, outside new 1.5x*20mm=30mm gate
    mid = torch.tensor([[0.30, 0.25, 0.08]])
    near = torch.tensor([[0.30, 0.29, 0.08]])
    ever = torch.ones(1, dtype=torch.bool)
    carry = torch.ones(1, dtype=torch.bool)
    assert reward_place_z(mid, target, ever, 0.015, 0.02, near_factor=1.5).item() == 0.0
    assert reward_place_z(near, target, ever, 0.015, 0.02, near_factor=1.5).item() > 0.0
    assert reward_lower(mid[:, 2] - 0.01, mid[:, 2], mid, target, carry, 0.02, near_factor=1.5).item() == 0.0
    assert reward_lower(near[:, 2] - 0.01, near[:, 2], near, target, carry, 0.02, near_factor=1.5).item() > 0.0


def test_reward_release_gated_low_only() -> None:
    target = torch.tensor([[0.30, 0.30, 0.015]])
    gap = torch.tensor([0.08])
    ever = torch.ones(1, dtype=torch.bool)
    near = torch.ones(1, dtype=torch.bool)
    high = torch.tensor([[0.30, 0.30, 0.090]])
    low = torch.tensor([[0.30, 0.30, 0.020]])
    high_rew = reward_release(high, target, gap, 0.084, ever, 0.015, 0.04, ever_carried_near=near)
    low_rew = reward_release(low, target, gap, 0.084, ever, 0.015, 0.04, ever_carried_near=near)
    assert high_rew[0].item() == pytest.approx(0.0)
    assert low_rew[0].item() > 0.0


def test_reward_release_zero_when_far() -> None:
    obj = torch.tensor([[0.30, 0.0, 0.015]])
    target = torch.tensor([[0.30, 0.30, 0.015]])
    gap = torch.tensor([0.08])
    ever = torch.ones(1, dtype=torch.bool)
    near = torch.ones(1, dtype=torch.bool)
    rew = reward_release(obj, target, gap, 0.084, ever, 0.015, 0.04, ever_carried_near=near)
    assert rew[0].item() == pytest.approx(0.0)


def test_reward_release_requires_ever_carried_near() -> None:
    target = torch.tensor([[0.30, 0.30, 0.015]])
    obj = torch.tensor([[0.30, 0.30, 0.020]])
    gap = torch.tensor([0.08])
    ever = torch.ones(1, dtype=torch.bool)
    no_near = torch.zeros(1, dtype=torch.bool)
    rew = reward_release(obj, target, gap, 0.084, ever, 0.015, 0.04, ever_carried_near=no_near)
    assert rew[0].item() == pytest.approx(0.0)


def test_reward_push_after_release_penalizes_open_ee_near_object() -> None:
    ee = torch.tensor([[0.30, 0.30, 0.02]])
    obj = torch.tensor([[0.30, 0.30, 0.015]])
    gap = torch.tensor([0.08])
    near = torch.ones(1, dtype=torch.bool)
    carry = torch.zeros(1, dtype=torch.bool)
    pen = reward_push_after_release(ee, obj, gap, near, carry, 0.015, 0.03)
    assert pen.item() == -1.0
    assert reward_push_after_release(ee, obj, gap, near, ~carry, 0.015, 0.03).item() == 0.0
    assert reward_push_after_release(ee, obj, gap, ~near, carry, 0.015, 0.03).item() == 0.0


def test_reward_release_attenuates_when_object_fast() -> None:
    target = torch.tensor([[0.30, 0.30, 0.015]])
    obj = torch.tensor([[0.30, 0.30, 0.020]])
    gap = torch.tensor([0.08])
    ever = torch.ones(1, dtype=torch.bool)
    near = torch.ones(1, dtype=torch.bool)
    still = torch.zeros(1, 3)
    fast = torch.tensor([[0.40, 0.0, 0.0]])
    slow_rew = reward_release(obj, target, gap, 0.084, ever, 0.015, 0.04, ever_carried_near=near, obj_vel=still)
    fast_rew = reward_release(obj, target, gap, 0.084, ever, 0.015, 0.04, ever_carried_near=near, obj_vel=fast)
    assert slow_rew[0].item() > 0.0
    assert fast_rew[0].item() == pytest.approx(0.0, abs=1e-3)
    assert fast_rew[0].item() < 0.05 * slow_rew[0].item()


def test_reward_release_height_configurable() -> None:
    target = torch.tensor([[0.30, 0.30, 0.015]])
    # 40 mm above rest: inside default 50 mm gate, outside tightened 30 mm gate
    obj = torch.tensor([[0.30, 0.30, 0.055]])
    gap = torch.tensor([0.08])
    ever = torch.ones(1, dtype=torch.bool)
    near = torch.ones(1, dtype=torch.bool)
    wide = reward_release(obj, target, gap, 0.084, ever, 0.015, 0.04, release_height_m=0.05, ever_carried_near=near)
    tight = reward_release(obj, target, gap, 0.084, ever, 0.015, 0.04, release_height_m=0.03, ever_carried_near=near)
    assert wide[0].item() > 0.0
    assert tight[0].item() == pytest.approx(0.0)


def test_reward_throw_release_penalizes_fast_open() -> None:
    gap = torch.tensor([0.08])
    near = torch.ones(1, dtype=torch.bool)
    fast = torch.tensor([[0.30, 0.0, 0.0]])
    still = torch.zeros(1, 3)
    pen = reward_throw_release(fast, gap, near, 0.03, throw_speed_m_s=0.15)
    assert pen[0].item() == pytest.approx(-1.0)
    assert reward_throw_release(still, gap, near, 0.03, throw_speed_m_s=0.15).item() == 0.0
    assert reward_throw_release(fast, gap, ~near, 0.03, throw_speed_m_s=0.15).item() == 0.0
    closed = torch.tensor([0.02])
    assert reward_throw_release(fast, closed, near, 0.03, throw_speed_m_s=0.15).item() == 0.0


def test_reward_push_before_grasp_penalizes_sideways_shove() -> None:
    start = torch.tensor([[0.30, 0.0, 0.015]])
    shoved = torch.tensor([[0.30, 0.05, 0.015]])
    ee = torch.tensor([[0.30, 0.05, 0.02]])
    gap = torch.tensor([0.08])
    not_grasped = torch.zeros(1, dtype=torch.bool)
    pen = reward_push_before_grasp(shoved, start, ee, gap, not_grasped, 0.03, push_dist_m=0.02)
    assert pen[0].item() == pytest.approx(-1.0)
    assert reward_push_before_grasp(start, start, ee, gap, not_grasped, 0.03).item() == 0.0
    grasped = torch.ones(1, dtype=torch.bool)
    assert reward_push_before_grasp(shoved, start, ee, gap, grasped, 0.03).item() == 0.0


def test_pre_lift_drag_progress_penalizes_even_with_closed_gripper() -> None:
    start = torch.tensor([[0.30, 0.00, 0.015]])
    dragged = torch.tensor([[0.30, 0.02, 0.015]])
    false = torch.zeros(1, dtype=torch.bool)
    state = update_task_state(
        dragged,
        dragged,
        torch.tensor([0.024]),
        torch.zeros(1, 3),
        torch.tensor([[0.30, 0.30, 0.015]]),
        ever_grasped=false,
        ever_lifted=false,
        ever_carried_near=false,
        episode_place_success=false,
        episode_success=false,
        place_stable_steps=torch.zeros(1, dtype=torch.int64),
        obj_rest_z_base=0.015,
        object_width=0.03,
        place_success_dist_m=0.01,
        success_hold_steps=25,
        initial_obj_pos=start,
        commanded_gap=torch.tensor([0.024]),
        action_gripper=torch.tensor([-0.5]),
    )
    assert state.max_pre_lift_xy_m.item() == pytest.approx(0.02)
    penalty = reward_pre_lift_xy_progress(
        torch.zeros(1),
        state.max_pre_lift_xy_m,
        normalization_m=0.01,
    )
    assert penalty.item() == pytest.approx(-2.0)
    assert (
        reward_clean_lift_bonus(
            torch.tensor([True]),
            false,
            state.max_pre_lift_xy_m,
        ).item()
        == 0.0
    )


@pytest.mark.parametrize(
    "velocity",
    ([0.03, 0.0, 0.0], [0.0, 0.03, 0.0], [0.0, 0.0, 0.03]),
)
def test_release_intent_before_40mm_detects_three_axis_high_speed(velocity) -> None:
    true = torch.ones(1, dtype=torch.bool)
    false = torch.zeros(1, dtype=torch.bool)
    state = update_task_state(
        torch.tensor([[0.30, 0.30, 0.015]]),
        torch.tensor([[0.30, 0.30, 0.05]]),
        torch.tensor([0.024]),  # measured gap is still well below 40 mm
        torch.tensor([velocity]),
        torch.tensor([[0.30, 0.30, 0.015]]),
        ever_grasped=true,
        ever_lifted=true,
        ever_carried_near=true,
        episode_place_success=false,
        episode_success=false,
        place_stable_steps=torch.zeros(1, dtype=torch.int64),
        obj_rest_z_base=0.015,
        object_width=0.03,
        place_success_dist_m=0.01,
        success_hold_steps=25,
        initial_obj_pos=torch.tensor([[0.30, 0.00, 0.015]]),
        commanded_gap=torch.tensor([0.025]),
        action_gripper=torch.tensor([0.5]),
        gripper_close_gap_m=0.022,
    )
    assert bool(state.release_started.item())
    assert bool(state.release_violation.item())
    assert reward_invalid_release(
        state.release_started,
        false,
        state.release_violation,
    ).item() == pytest.approx(-1.0)


def test_fast_downward_near_table_latches_hard_landing_once() -> None:
    true = torch.ones(1, dtype=torch.bool)
    false = torch.zeros(1, dtype=torch.bool)
    state = update_task_state(
        torch.tensor([[0.30, 0.30, 0.020]]),
        torch.tensor([[0.30, 0.30, 0.05]]),
        torch.tensor([0.03]),
        torch.tensor([[0.0, 0.0, -0.06]]),
        torch.tensor([[0.30, 0.30, 0.015]]),
        ever_grasped=true,
        ever_lifted=true,
        ever_carried_near=true,
        episode_place_success=false,
        episode_success=false,
        place_stable_steps=torch.zeros(1, dtype=torch.int64),
        obj_rest_z_base=0.015,
        object_width=0.03,
        place_success_dist_m=0.01,
        success_hold_steps=25,
        initial_obj_pos=torch.tensor([[0.30, 0.00, 0.015]]),
        commanded_gap=torch.tensor([0.022]),
        action_gripper=torch.tensor([0.0]),
    )
    assert bool(state.near_table_entered.item())
    assert bool(state.hard_landing_violation.item())
    assert state.max_landing_down_speed_m_s.item() == pytest.approx(0.06)
    assert reward_hard_landing(state.hard_landing_violation, false).item() == -1.0
    assert reward_hard_landing(state.hard_landing_violation, true).item() == 0.0


def test_valid_release_is_one_shot_and_quality_allows_stable_success() -> None:
    true = torch.ones(1, dtype=torch.bool)
    false = torch.zeros(1, dtype=torch.bool)
    common = dict(
        obj_base=torch.tensor([[0.30, 0.30, 0.015]]),
        ee_base=torch.tensor([[0.30, 0.30, 0.08]]),
        gap=torch.tensor([0.08]),
        obj_vel=torch.zeros(1, 3),
        target_pos=torch.tensor([[0.30, 0.30, 0.015]]),
        ever_grasped=true,
        ever_lifted=true,
        ever_carried_near=true,
        episode_place_success=false,
        episode_success=false,
        place_stable_steps=torch.zeros(1, dtype=torch.int64),
        obj_rest_z_base=0.015,
        object_width=0.03,
        place_success_dist_m=0.01,
        success_hold_steps=2,
        initial_obj_pos=torch.tensor([[0.30, 0.00, 0.015]]),
        commanded_gap=torch.tensor([0.026]),
        action_gripper=torch.tensor([0.5]),
        gripper_close_gap_m=0.022,
    )
    first = update_task_state(**common)
    assert bool(first.release_valid.item())
    assert bool(first.quality_ok.item())
    assert reward_valid_release(first.release_started, false, first.release_valid).item() == 1.0
    second = update_task_state(
        **{
            **common,
            "episode_place_success": first.episode_place_success,
            "place_stable_steps": first.place_stable_steps,
            "action_gripper": torch.tensor([0.0]),
            "release_started": first.release_started,
            "release_valid": first.release_valid,
            "release_violation": first.release_violation,
            "near_table_entered": first.near_table_entered,
            "hard_landing_violation": first.hard_landing_violation,
            "max_pre_lift_xy_m": first.max_pre_lift_xy_m,
            "max_landing_xy_speed_m_s": first.max_landing_xy_speed_m_s,
            "max_landing_down_speed_m_s": first.max_landing_down_speed_m_s,
            "release_xy_dist_m": first.release_xy_dist_m,
            "release_height_error_m": first.release_height_error_m,
            "release_speed_m_s": first.release_speed_m_s,
            "release_obj_xy": first.release_obj_xy,
            "post_release_drift_m": first.post_release_drift_m,
        }
    )
    assert (
        reward_valid_release(
            second.release_started,
            first.release_started,
            second.release_valid,
        ).item()
        == 0.0
    )
    assert bool(second.episode_success.item())


def test_continuous_lift_and_release_quality_rewards_keep_intermediate_signal() -> None:
    first_lift = torch.tensor([True, True, True])
    never_grasped = torch.tensor([False, False, False])
    lift_quality = reward_clean_lift_quality(
        first_lift,
        never_grasped,
        torch.tensor([0.005, 0.04, 0.08]),
        distance_scale_m=0.04,
    )
    assert lift_quality[0] > lift_quality[1] > lift_quality[2] > 0.0

    release_quality = reward_release_quality(
        torch.tensor([True, True]),
        torch.tensor([False, False]),
        torch.tensor([0.01, 0.04]),
        torch.tensor([0.005, 0.04]),
        torch.tensor([0.02, 0.10]),
    )
    assert release_quality[0] > release_quality[1] > 0.0
    assert (
        reward_release_quality(
            torch.tensor([True]),
            torch.tensor([True]),
            torch.tensor([0.01]),
            torch.tensor([0.005]),
            torch.tensor([0.02]),
        ).item()
        == 0.0
    )


def test_release_readiness_opening_and_landing_rewards_have_severity_signal() -> None:
    readiness = reward_release_readiness_progress(
        torch.tensor([0.04, 0.01, 0.04]),
        torch.tensor([0.04, 0.01, 0.04]),
        torch.tensor([0.10, 0.02, 0.10]),
        torch.tensor([0.01, 0.04, 0.01]),
        torch.tensor([0.01, 0.04, 0.01]),
        torch.tensor([0.02, 0.10, 0.02]),
        torch.tensor([True, True, False]),
        torch.tensor([False, False, False]),
    )
    assert readiness[0] > 0.0
    assert readiness[1] < 0.0
    assert readiness[2] == 0.0
    assert (
        reward_release_readiness_progress(
            torch.tensor([0.04]),
            torch.tensor([0.04]),
            torch.tensor([0.10]),
            torch.tensor([0.01]),
            torch.tensor([0.01]),
            torch.tensor([0.02]),
            torch.tensor([True]),
            torch.tensor([True]),
        ).item()
        == 0.0
    )

    opening = reward_premature_opening(
        torch.tensor([0.5, 0.5, -0.5]),
        torch.tensor([True, True, True]),
        torch.tensor([False, False, False]),
        torch.tensor([False, True, False]),
    )
    assert opening.tolist() == pytest.approx([-0.5, 0.0, 0.0])
    ready_opening = reward_ready_opening(
        torch.tensor([-0.5, 0.5, 0.5]),
        torch.tensor([True, True, False]),
        torch.tensor([False, False, False]),
    )
    assert ready_opening.tolist() == pytest.approx([-0.5, 0.5, 0.0])
    clearance_opening = reward_release_clearance_opening(
        torch.tensor([-0.5, 0.5, 0.5]),
        torch.tensor([0.049, 0.049, 0.056]),
        torch.tensor([True, True, True]),
        target_commanded_gap_m=0.055,
    )
    assert clearance_opening.tolist() == pytest.approx([-0.5, 0.5, 0.0])
    closing = reward_grasp_ready_closing(
        torch.tensor([-0.6, 0.6, -0.6, -0.6]),
        torch.tensor([0.005, 0.020, 0.005, 0.050]),
        torch.tensor([0.060, 0.060, 0.049, 0.060]),
        torch.tensor([0.040, 0.040, 0.040, 0.040]),
        torch.tensor([False, False, False, False]),
        torch.tensor([False, False, False, False]),
    )
    assert closing.tolist() == pytest.approx([0.6, -0.6, 0.0, 0.0])
    settle = reward_grasp_settle_action(
        torch.tensor(
            [
                [0.3, -0.4, 0.5],
                [0.3, -0.4, 0.5],
                [0.3, -0.4, 0.5],
                [0.3, -0.4, 0.5],
            ]
        ),
        torch.tensor([0.005, 0.020, 0.050, 0.005]),
        torch.tensor([0.060, 0.030, 0.060, 0.030]),
        torch.tensor([0.084, 0.040, 0.084, 0.040]),
        torch.tensor([False, False, False, True]),
    )
    assert settle.tolist() == pytest.approx([-0.50, -0.25, 0.0, 0.0])
    gap_progress = reward_grasp_gap_progress(
        torch.tensor([0.080, 0.060, 0.030, 0.030]),
        torch.tensor([0.060, 0.080, 0.029, 0.036]),
        torch.tensor([False, False, False, False]),
    )
    assert gap_progress.tolist() == pytest.approx([1.0, -1.0, 0.0, -0.3])
    lift_action = reward_grasp_lift_action(
        torch.tensor([0.5, -0.5, 0.5]),
        torch.tensor([0.030, 0.040, 0.030]),
        torch.tensor([True, True, True]),
        torch.tensor([False, False, False]),
        torch.tensor([0.004, 0.004, 0.006]),
    )
    assert lift_action.tolist() == pytest.approx([0.5, 0.0, 0.0])
    lift_progress = reward_grasp_lift_progress(
        torch.tensor([0.015, 0.020, 0.020, 0.020]),
        torch.tensor([0.020, 0.015, 0.025, 0.025]),
        torch.tensor([0.030, 0.030, 0.040, 0.030]),
        torch.tensor([True, True, True, True]),
        torch.tensor([False, False, False, True]),
        torch.tensor([0.004, 0.004, 0.004, 0.004]),
    )
    assert lift_progress.tolist() == pytest.approx([0.25, -0.25, 0.0, 0.0])

    landing = reward_landing_quality(
        torch.tensor([True, True, True]),
        torch.tensor([False, False, True]),
        torch.tensor(
            [
                [0.005, 0.0, -0.01],
                [0.20, 0.0, -0.50],
                [0.0, 0.0, 0.0],
            ]
        ),
    )
    assert landing[0] > landing[1] > 0.0
    assert landing[2] == 0.0

    setdown = reward_setdown_action(
        torch.tensor([0.5, -0.5, 0.5]),
        torch.tensor([0.04, 0.04, 0.004]),
        torch.tensor([True, True, True]),
        torch.tensor([True, True, True]),
        torch.tensor([False, False, False]),
    )
    assert setdown.tolist() == pytest.approx([-0.5, 0.0, 0.0])


def test_quality_violation_prevents_geometric_success_without_ending_correction() -> None:
    true = torch.ones(1, dtype=torch.bool)
    false = torch.zeros(1, dtype=torch.bool)
    state = update_task_state(
        torch.tensor([[0.30, 0.30, 0.015]]),
        torch.tensor([[0.30, 0.30, 0.08]]),
        torch.tensor([0.08]),
        torch.zeros(1, 3),
        torch.tensor([[0.30, 0.30, 0.015]]),
        ever_grasped=true,
        ever_lifted=true,
        ever_carried_near=true,
        episode_place_success=false,
        episode_success=false,
        place_stable_steps=torch.zeros(1, dtype=torch.int64),
        obj_rest_z_base=0.015,
        object_width=0.03,
        place_success_dist_m=0.01,
        success_hold_steps=1,
        initial_obj_pos=torch.tensor([[0.30, 0.00, 0.015]]),
        commanded_gap=torch.tensor([0.026]),
        action_gripper=torch.tensor([0.5]),
        release_violation=true,
        max_pre_lift_xy_m=torch.tensor([0.02]),
        gripper_close_gap_m=0.022,
    )
    assert not bool(state.quality_ok.item())
    assert not bool(state.episode_place_success.item())
    assert not bool(state.episode_success.item())


def test_precision_speed_and_post_release_rewards_have_expected_signs() -> None:
    assert reward_precision_progress(
        torch.tensor([0.02]),
        torch.tensor([0.01]),
        torch.tensor([True]),
    ).item() == pytest.approx(1.0)
    assert reward_precision_progress(
        torch.tensor([0.01]),
        torch.tensor([0.02]),
        torch.tensor([True]),
    ).item() == pytest.approx(-1.0)
    transport = reward_transport_progress(
        torch.tensor([0.20, 0.20]),
        torch.tensor([0.10, 0.25]),
        torch.tensor([True, True]),
        torch.tensor([True, True]),
        torch.tensor([False, False]),
    )
    assert transport.tolist() == pytest.approx([1.0, -0.5])
    descent = reward_descent_progress(
        torch.tensor([0.08, 0.04]),
        torch.tensor([0.04, 0.08]),
        torch.tensor([True, True]),
        torch.tensor([True, True]),
        torch.tensor([False, False]),
    )
    assert descent.tolist() == pytest.approx([1.0, -1.0])
    speed_penalty = reward_near_target_speed(
        torch.tensor([[0.0, 0.0, 0.015]]),
        torch.tensor([[0.0, 0.0, 0.015]]),
        torch.tensor([[0.04, 0.0, 0.0]]),
        max_height_error_m=0.025,
    )
    assert speed_penalty.item() < 0.0
    assert (
        reward_near_target_speed(
            torch.tensor([[0.0, 0.0, 0.015]]),
            torch.tensor([[0.0, 0.0, 0.015]]),
            torch.tensor([[0.02, 0.0, 0.0]]),
            max_height_error_m=0.025,
        ).item()
        == 0.0
    )
    assert (
        reward_near_target_speed(
            torch.tensor([[0.0, 0.0, 0.060]]),
            torch.tensor([[0.0, 0.0, 0.015]]),
            torch.tensor([[0.04, 0.0, 0.0]]),
            max_height_error_m=0.025,
        ).item()
        == 0.0
    )
    assert reward_near_table_xy_action(
        torch.tensor([[0.3, -0.4], [0.3, -0.4]]),
        torch.tensor([0.01, 0.04]),
        torch.tensor([True, True]),
        max_height_error_m=0.025,
    ).tolist() == pytest.approx([-0.25, 0.0])
    contact_penalty = reward_post_release_contact(
        torch.tensor([[0.0, 0.0, 0.02]]),
        torch.tensor([[0.0, 0.0, 0.015]]),
        torch.tensor([True]),
        torch.tensor([False]),
        0.015,
    )
    assert contact_penalty.item() == -1.0


def test_post_release_clearance_is_bounded_reversible_and_not_farmable() -> None:
    valid = torch.tensor([True, True, True, False])
    progress = reward_post_release_clearance_progress(
        torch.tensor([0.000, 0.030, 0.050, 0.000]),
        torch.tensor([0.010, 0.020, 0.080, 0.010]),
        valid,
        max_clearance_m=0.050,
    )
    assert progress.tolist() == pytest.approx([0.2, -0.2, 0.0, 0.0])
    assert (
        reward_post_release_clearance_progress(
            torch.tensor([0.02]),
            torch.tensor([0.02]),
            torch.tensor([True]),
        ).item()
        == 0.0
    )
    assert reward_post_release_recontact(torch.tensor([False, True, True])).tolist() == pytest.approx([0.0, -1.0, -1.0])


def test_near_table_velocity_margins_are_separated_continuous_and_gated() -> None:
    obj = torch.tensor(
        [
            [0.30, 0.30, 0.025],  # active
            [0.30, 0.30, 0.025],  # never carried near
            [0.30, 0.30, 0.050],  # above near-table region
        ]
    )
    velocity = torch.tensor(
        [
            [0.018, 0.024, -0.025],  # xy=0.03, down=0.025
            [0.030, 0.000, -0.050],
            [0.030, 0.000, -0.050],
        ]
    )
    carried_near = torch.tensor([True, False, True])
    xy_cost = reward_near_table_xy_speed_margin(
        obj,
        velocity,
        carried_near,
        obj_rest_z_base=0.015,
        near_table_height_m=0.020,
        speed_limit_m_s=0.030,
    )
    down_cost = reward_near_table_down_speed_margin(
        obj,
        velocity,
        carried_near,
        obj_rest_z_base=0.015,
        near_table_height_m=0.020,
        speed_limit_m_s=0.050,
    )
    assert xy_cost.tolist() == pytest.approx([-1.0, 0.0, 0.0])
    assert down_cost.tolist() == pytest.approx([-0.25, 0.0, 0.0])
    upward = velocity[:1].clone()
    upward[:, 2] = 0.10
    assert (
        reward_near_table_down_speed_margin(
            obj[:1],
            upward,
            carried_near[:1],
            obj_rest_z_base=0.015,
        ).item()
        == 0.0
    )


def test_near_table_speed_margin_safety_zone_ramps_to_limit() -> None:
    """Safety zone makes speeds below half the limit free, ramping to -1 at the limit."""
    obj = torch.tensor([[0.30, 0.30, 0.025]])  # active near-table row
    carried = torch.tensor([True])
    base_kwargs = dict(obj_rest_z_base=0.015, near_table_height_m=0.035)
    # xy margin with safety_zone_frac=0.5 on limit 0.03 -> free below 0.015.
    free_xy = reward_near_table_xy_speed_margin(
        obj,
        torch.tensor([[0.010, 0.005, -0.01]]),
        carried,
        speed_limit_m_s=0.030,
        safety_zone_frac=0.5,
        **base_kwargs,
    )
    assert free_xy.item() == 0.0
    # Exactly at the limit the cost is -1.0 (same as the legacy shape).
    at_limit_xy = reward_near_table_xy_speed_margin(
        obj,
        torch.tensor([[0.030, 0.000, -0.01]]),
        carried,
        speed_limit_m_s=0.030,
        safety_zone_frac=0.5,
        **base_kwargs,
    )
    assert at_limit_xy.item() == pytest.approx(-1.0)
    # Above the limit the cost keeps growing beyond -1.
    over_limit_xy = reward_near_table_xy_speed_margin(
        obj,
        torch.tensor([[0.040, 0.000, -0.01]]),
        carried,
        speed_limit_m_s=0.030,
        safety_zone_frac=0.5,
        **base_kwargs,
    )
    assert over_limit_xy.item() < -1.0
    # Down margin: free below 0.025 (half of 0.05), -0.25 at half-way of the
    # danger band, -1.0 exactly at the limit.
    down_free = reward_near_table_down_speed_margin(
        obj,
        torch.tensor([[0.0, 0.0, -0.010]]),
        carried,
        speed_limit_m_s=0.050,
        safety_zone_frac=0.5,
        **base_kwargs,
    )
    assert down_free.item() == 0.0
    down_mid = reward_near_table_down_speed_margin(
        obj,
        torch.tensor([[0.0, 0.0, -0.0375]]),
        carried,
        speed_limit_m_s=0.050,
        safety_zone_frac=0.5,
        **base_kwargs,
    )
    assert down_mid.item() == pytest.approx(-0.25)
    down_limit = reward_near_table_down_speed_margin(
        obj,
        torch.tensor([[0.0, 0.0, -0.050]]),
        carried,
        speed_limit_m_s=0.050,
        safety_zone_frac=0.5,
        **base_kwargs,
    )
    assert down_limit.item() == pytest.approx(-1.0)
    # Legacy default (no safety zone) is unchanged: cost starts at any nonzero speed.
    legacy = reward_near_table_xy_speed_margin(
        obj,
        torch.tensor([[0.015, 0.000, -0.01]]),
        carried,
        speed_limit_m_s=0.030,
        **base_kwargs,
    )
    assert legacy.item() == pytest.approx(-0.25)


def test_near_table_speed_margin_safety_zone_validation() -> None:
    obj = torch.tensor([[0.30, 0.30, 0.025]])
    carried = torch.tensor([True])
    vel = torch.zeros(1, 3)
    for bad in (-0.1, 1.0, 2.0):
        with pytest.raises(ValueError):
            reward_near_table_xy_speed_margin(
                obj,
                vel,
                carried,
                obj_rest_z_base=0.015,
                safety_zone_frac=bad,
            )
        with pytest.raises(ValueError):
            reward_near_table_down_speed_margin(
                obj,
                vel,
                carried,
                obj_rest_z_base=0.015,
                safety_zone_frac=bad,
            )


def test_near_target_attenuate_ramp() -> None:
    target = torch.tensor([[0.0, 0.0, 0.0]])
    on = torch.tensor([[0.0, 0.0, 0.0]])
    mid = torch.tensor([[0.06, 0.0, 0.0]])  # 1.5 * 0.04
    far = torch.tensor([[0.20, 0.0, 0.0]])
    assert near_target_attenuate(on, target, 0.04)[0].item() == pytest.approx(0.0)
    assert near_target_attenuate(far, target, 0.04)[0].item() == pytest.approx(1.0)
    assert 0.0 < near_target_attenuate(mid, target, 0.04)[0].item() < 1.0


def test_update_task_state_grasp_and_success() -> None:
    n = 1
    obj_base = torch.tensor([[0.3, 0.0, 0.05]])  # lifted above rest 0.015
    ee_base = torch.tensor([[0.3, 0.0, 0.05]])
    gap = torch.tensor([0.03])  # holding (< object_width+0.02 = 0.05)
    obj_vel = torch.zeros(1, 3)
    target = torch.tensor([[0.3, 0.0, 0.015]])
    false = torch.zeros(n, dtype=torch.bool)
    zeros = torch.zeros(n, dtype=torch.int64)
    state = update_task_state(
        obj_base,
        ee_base,
        gap,
        obj_vel,
        target,
        ever_grasped=false,
        ever_lifted=false,
        ever_carried_near=false,
        episode_place_success=false,
        episode_success=false,
        place_stable_steps=zeros,
        obj_rest_z_base=0.015,
        object_width=0.03,
        place_success_dist_m=0.04,
        success_hold_steps=3,
        carry_near_dist_m=0.04,
    )
    assert bool(state.grasped[0].item())
    assert bool(state.ever_carried_near[0].item())

    # Place success path: on table, released, stable, near target, previously carried near
    obj_place = torch.tensor([[0.3, 0.0, 0.015]])
    gap_open = torch.tensor([0.08])
    ever = torch.ones(n, dtype=torch.bool)
    near = torch.ones(n, dtype=torch.bool)
    place_false = torch.zeros(n, dtype=torch.bool)
    steps = torch.zeros(n, dtype=torch.int64)
    for _ in range(3):
        st = update_task_state(
            obj_place,
            ee_base,
            gap_open,
            obj_vel,
            target,
            ever_grasped=ever,
            ever_lifted=ever,
            ever_carried_near=near,
            episode_place_success=place_false,
            episode_success=place_false,
            place_stable_steps=steps,
            obj_rest_z_base=0.015,
            object_width=0.03,
            place_success_dist_m=0.04,
            success_hold_steps=3,
            carry_near_dist_m=0.04,
        )
        steps = st.place_stable_steps
        place_false = st.episode_place_success
        ever = st.ever_grasped
        near = st.ever_carried_near
    assert bool(st.episode_success[0].item())


def test_update_task_state_slide_in_without_near_carry_cannot_place() -> None:
    """Mid-path drop then slide into the circle must not latch place success."""
    n = 1
    obj = torch.tensor([[0.30, 0.30, 0.015]])
    ee = torch.tensor([[0.30, 0.30, 0.05]])
    gap = torch.tensor([0.08])
    vel = torch.zeros(1, 3)
    target = torch.tensor([[0.30, 0.30, 0.015]])
    false = torch.zeros(n, dtype=torch.bool)
    ever = torch.ones(n, dtype=torch.bool)
    zeros = torch.zeros(n, dtype=torch.int64)
    st = update_task_state(
        obj,
        ee,
        gap,
        vel,
        target,
        ever_grasped=ever,
        ever_lifted=ever,
        ever_carried_near=false,
        episode_place_success=false,
        episode_success=false,
        place_stable_steps=zeros,
        obj_rest_z_base=0.015,
        object_width=0.03,
        place_success_dist_m=0.04,
        success_hold_steps=3,
        carry_near_dist_m=0.04,
    )
    assert not bool(st.episode_place_success[0].item())
    assert not bool(st.ever_carried_near[0].item())


def test_curriculum_transitions() -> None:
    assert next_curriculum_stage(0, grasp_rate=0.6, place_rate=0.0, grasp_count=500, place_count=0) == 1
    assert next_curriculum_stage(2, grasp_rate=0.0, place_rate=0.65, grasp_count=0, place_count=500) == 3
    assert next_curriculum_stage(0, grasp_rate=0.6, place_rate=0.0, grasp_count=100, place_count=0) == 0


def test_curriculum_supports_strict_from_home_thresholds() -> None:
    kwargs = {
        "min_count": 1024,
        "stage0_grasp_rate": 0.95,
        "stage1_grasp_rate": 0.90,
        "stage2_place_rate": 0.90,
        "stage3_place_rate": 0.85,
    }
    assert next_curriculum_stage(0, 0.94, 0.0, 2048, 0, **kwargs) == 0
    assert next_curriculum_stage(0, 0.95, 0.0, 1024, 0, **kwargs) == 1
    assert next_curriculum_stage(2, 0.0, 0.89, 0, 2048, **kwargs) == 2
    assert next_curriculum_stage(2, 0.0, 0.90, 0, 1024, **kwargs) == 3


def test_curriculum_max_stage_holds_a_training_phase() -> None:
    assert (
        next_curriculum_stage(
            2,
            grasp_rate=1.0,
            place_rate=1.0,
            grasp_count=4096,
            place_count=4096,
            max_stage=2,
        )
        == 2
    )
    with pytest.raises(ValueError, match="max_stage"):
        next_curriculum_stage(0, 1.0, 1.0, 4096, 4096, max_stage=5)


def test_reset_phase_masks_are_disjoint_and_anneal_is_clamped() -> None:
    samples = torch.tensor([0.05, 0.20, 0.50, 0.90])
    place, carry, grasp = sample_reset_phase_masks(samples, 0.15, 0.15, 0.35)
    assert place.tolist() == [True, False, False, False]
    assert carry.tolist() == [False, True, False, False]
    assert grasp.tolist() == [False, False, True, False]
    assert not torch.any(place & carry)
    assert not torch.any(place & grasp)
    assert not torch.any(carry & grasp)
    assert annealed_frac(0.15, 0.0, -1.0) == pytest.approx(0.15)
    assert annealed_frac(0.15, 0.0, 0.5) == pytest.approx(0.075)
    assert annealed_frac(0.15, 0.0, 2.0) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="sum to at most 1"):
        sample_reset_phase_masks(samples, 0.5, 0.5, 0.5)


def test_sample_object_and_target_stage0_is_fixed_demo_layout() -> None:
    """Stage 0 matches trajectory pick-place fixed object/target (used by fixed_demo_layout)."""
    fixed_obj = torch.tensor([0.30, 0.00, 0.015])
    fixed_target = torch.tensor([0.30, 0.30, 0.015])
    spawn_lo = torch.tensor([0.28, -0.05, 0.015])
    spawn_hi = torch.tensor([0.34, 0.05, 0.015])
    tgt_lo = torch.tensor([0.28, 0.25, 0.015])
    tgt_hi = torch.tensor([0.34, 0.35, 0.015])
    obj, target = sample_object_and_target(
        0,
        4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        fixed_obj=fixed_obj,
        fixed_target=fixed_target,
        obj_spawn_lower=spawn_lo,
        obj_spawn_upper=spawn_hi,
        target_spawn_lower=tgt_lo,
        target_spawn_upper=tgt_hi,
    )
    assert obj.shape == (4, 3)
    assert torch.allclose(obj, fixed_obj.unsqueeze(0).expand(4, 3))
    assert torch.allclose(target, fixed_target.unsqueeze(0).expand(4, 3))


@pytest.mark.parametrize("stage", [1, 2, 3, 4])
def test_random_curriculum_preserves_canonical_positive_y_geometry(stage: int) -> None:
    fixed_obj = torch.tensor([0.30, 0.00, 0.015])
    fixed_target = torch.tensor([0.30, 0.30, 0.015])
    obj_lo = torch.tensor([0.28, -0.05, 0.015])
    obj_hi = torch.tensor([0.34, 0.05, 0.015])
    target_lo = torch.tensor([0.28, 0.25, 0.015])
    target_hi = torch.tensor([0.34, 0.35, 0.015])
    obj, target = sample_object_and_target(
        stage,
        512,
        device=torch.device("cpu"),
        dtype=torch.float32,
        fixed_obj=fixed_obj,
        fixed_target=fixed_target,
        obj_spawn_lower=obj_lo,
        obj_spawn_upper=obj_hi,
        target_spawn_lower=target_lo,
        target_spawn_upper=target_hi,
    )
    assert torch.all((obj >= obj_lo) & (obj <= obj_hi))
    assert torch.all((target >= target_lo) & (target <= target_hi))
    assert torch.all(target[:, 1] > obj[:, 1])


def test_curriculum_edge_fraction_samples_stage_boundary_corners() -> None:
    fixed_obj = torch.tensor([0.30, 0.00, 0.015])
    fixed_target = torch.tensor([0.30, 0.30, 0.015])
    obj, target = sample_object_and_target(
        1,
        512,
        device=torch.device("cpu"),
        dtype=torch.float32,
        fixed_obj=fixed_obj,
        fixed_target=fixed_target,
        obj_spawn_lower=torch.tensor([0.28, -0.05, 0.015]),
        obj_spawn_upper=torch.tensor([0.34, 0.05, 0.015]),
        target_spawn_lower=torch.tensor([0.28, 0.25, 0.015]),
        target_spawn_upper=torch.tensor([0.34, 0.35, 0.015]),
        edge_fraction=1.0,
    )
    assert torch.allclose(torch.abs(obj[:, :2] - fixed_obj[:2]), torch.full((512, 2), 0.01))
    assert torch.allclose(torch.abs(target[:, :2] - fixed_target[:2]), torch.full((512, 2), 0.01))
    with pytest.raises(ValueError, match="edge_fraction"):
        sample_object_and_target(
            1,
            1,
            device=torch.device("cpu"),
            dtype=torch.float32,
            fixed_obj=fixed_obj,
            fixed_target=fixed_target,
            obj_spawn_lower=torch.tensor([0.28, -0.05, 0.015]),
            obj_spawn_upper=torch.tensor([0.34, 0.05, 0.015]),
            target_spawn_lower=torch.tensor([0.28, 0.25, 0.015]),
            target_spawn_upper=torch.tensor([0.34, 0.35, 0.015]),
            edge_fraction=1.1,
        )


@pytest.mark.parametrize("stage", [1, 2, 3, 4])
def test_random_curriculum_can_keep_target_fixed(stage: int) -> None:
    fixed_obj = torch.tensor([0.30, 0.00, 0.015])
    fixed_target = torch.tensor([0.30, 0.30, 0.015])
    obj, target = sample_object_and_target(
        stage,
        512,
        device=torch.device("cpu"),
        dtype=torch.float32,
        fixed_obj=fixed_obj,
        fixed_target=fixed_target,
        obj_spawn_lower=torch.tensor([0.28, -0.05, 0.015]),
        obj_spawn_upper=torch.tensor([0.34, 0.05, 0.015]),
        target_spawn_lower=torch.tensor([0.28, 0.25, 0.015]),
        target_spawn_upper=torch.tensor([0.34, 0.35, 0.015]),
        edge_fraction=0.25,
        randomize_target=False,
    )
    assert torch.allclose(target, fixed_target.unsqueeze(0).expand_as(target))
    assert not torch.allclose(obj, fixed_obj.unsqueeze(0).expand_as(obj))


def test_observation_golden_values() -> None:
    """Lock observation layout against hand-computed values."""
    joint_pos = torch.arange(6, dtype=torch.float32).unsqueeze(0)
    joint_vel = torch.arange(6, 12, dtype=torch.float32).unsqueeze(0)
    ee = torch.tensor([[1.0, 2.0, 3.0]])
    gap = torch.tensor([0.04])
    obj = torch.tensor([[4.0, 5.0, 6.0]])
    target = torch.tensor([[7.0, 8.0, 9.0]])
    grasped = torch.tensor([True])
    ever = torch.tensor([False])
    obs = pick_place_observation(joint_pos, joint_vel, ee, gap, obj, target, grasped, ever)
    expected = torch.tensor(
        [
            [
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                1.0,
                2.0,
                3.0,
                0.04,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                3.0,
                3.0,
                3.0,  # ee_to_obj
                3.0,
                3.0,
                3.0,  # obj_to_target
                1.0,
                0.0,
            ]
        ]
    )
    assert torch.allclose(obs, expected)


def test_squared_action_response_keeps_reach_and_refines_the_centre() -> None:
    """A squared response must keep full reach at the bound and shrink small commands."""
    actions = torch.tensor([[1.0, 0.25, -0.5]])
    linear = action_to_cartesian_delta(actions, 0.01, 1.0)
    squared = action_to_cartesian_delta(actions, 0.01, 2.0)
    assert linear[0, 0].item() == pytest.approx(0.01)
    assert squared[0, 0].item() == pytest.approx(0.01)
    assert squared[0, 1].item() == pytest.approx(0.000625)
    assert squared[0, 2].item() == pytest.approx(-0.0025)
    roundtrip = cartesian_delta_to_action(squared, 0.01, 2.0)
    assert torch.allclose(roundtrip, actions, atol=1e-6)


def test_commanded_setpoint_holds_still_where_measured_integration_sinks() -> None:
    """Zero action must not move the setpoint even when the arm droops behind it."""
    setpoint = torch.tensor([[0.30, 0.0, 0.20]])
    measured = torch.tensor([[0.30, 0.0, 0.1966]])  # 3.4 mm of standing PD droop
    leashed = leashed_ee_setpoint(setpoint, measured, 0.02)
    held, _ = arm_target_ee_pos(leashed, torch.zeros(1, 3), 0.01, 0.01)
    assert held[0, 2].item() == pytest.approx(0.20)
    sunk, _ = arm_target_ee_pos(measured, torch.zeros(1, 3), 0.01, 0.01)
    assert sunk[0, 2].item() == pytest.approx(0.1966)


def test_setpoint_leash_bounds_runaway_when_the_arm_is_blocked() -> None:
    setpoint = torch.tensor([[0.30, 0.0, 0.30]])
    measured = torch.tensor([[0.30, 0.0, 0.20]])
    leashed = leashed_ee_setpoint(setpoint, measured, 0.02)
    assert leashed[0, 2].item() == pytest.approx(0.22)


def test_contact_features_separate_a_loaded_grasp_from_a_closed_but_empty_gripper() -> None:
    span = torch.tensor([0.0305, 0.0305])
    left = torch.tensor([2.0, 0.0])
    right = torch.tensor([1.5, 0.0])
    features = normalized_pick_place_contact_features(
        span,
        left,
        right,
        object_width_m=0.03,
        contact_force_scale_n=5.0,
    )
    assert features[0, 1].item() == pytest.approx(1.0)
    assert features[0, 2].item() == pytest.approx(0.3)
    assert features[1, 1].item() == pytest.approx(0.0)
    assert features[1, 2].item() == pytest.approx(0.0)


def test_holding_requires_loaded_pads_when_contact_is_available() -> None:
    """The nominal-gap test alone calls a 48 mm gap a grasp; contact must veto that."""
    obj = torch.tensor([[0.30, 0.00, 0.015]])
    false = torch.zeros(1, dtype=torch.bool)
    common = dict(
        gap=torch.tensor([0.048]),
        obj_vel=torch.zeros(1, 3),
        target_pos=torch.tensor([[0.30, 0.30, 0.015]]),
        ever_grasped=false,
        ever_lifted=false,
        ever_carried_near=false,
        episode_place_success=false,
        episode_success=false,
        place_stable_steps=torch.zeros(1, dtype=torch.int64),
        obj_rest_z_base=0.015,
        object_width=0.03,
        place_success_dist_m=0.01,
        success_hold_steps=25,
    )
    gap_only = update_task_state(obj, obj, **common)
    assert bool(gap_only.holding.item())
    untouched = update_task_state(obj, obj, bilateral_contact=false, **common)
    assert not bool(untouched.holding.item())
    loaded = update_task_state(
        obj,
        obj,
        bilateral_contact=torch.ones(1, dtype=torch.bool),
        **common,
    )
    assert bool(loaded.holding.item())


def test_grasp_offset_latches_at_first_lift_and_rewards_a_centred_grasp() -> None:
    obj = torch.tensor([[0.30, 0.00, 0.05]])
    off_centre_ee = torch.tensor([[0.30, 0.004, 0.05]])
    false = torch.zeros(1, dtype=torch.bool)
    state = update_task_state(
        obj,
        off_centre_ee,
        torch.tensor([0.024]),
        torch.zeros(1, 3),
        torch.tensor([[0.30, 0.30, 0.015]]),
        ever_grasped=false,
        ever_lifted=false,
        ever_carried_near=false,
        episode_place_success=false,
        episode_success=false,
        place_stable_steps=torch.zeros(1, dtype=torch.int64),
        obj_rest_z_base=0.015,
        object_width=0.03,
        place_success_dist_m=0.01,
        success_hold_steps=25,
    )
    assert bool(state.carry.item())
    assert state.grasp_offset_xy_m.item() == pytest.approx(0.004)
    off_centre = reward_grasp_centering(
        state.carry,
        false,
        state.grasp_offset_xy_m,
        distance_scale_m=0.005,
    )
    centred = reward_grasp_centering(
        state.carry,
        false,
        torch.zeros(1),
        distance_scale_m=0.005,
    )
    assert centred.item() == pytest.approx(1.0)
    assert off_centre.item() < centred.item()
    # Already-lifted episodes keep the latched value instead of re-measuring.
    assert (
        reward_grasp_centering(
            state.carry,
            torch.ones(1, dtype=torch.bool),
            state.grasp_offset_xy_m,
        ).item()
        == 0.0
    )
