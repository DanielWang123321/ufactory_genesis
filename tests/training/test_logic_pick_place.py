"""CPU tests for the Genesis-free pick-place numeric core."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ufactory.grippers.g2 import GRIPPER_G2_OPEN_GAP_M, GRIPPER_G2_SIM_CLOSE_DRIVE
from ufactory.training.logic import (
    action_penalty,
    arm_target_ee_pos,
    arm_target_qpos,
    drive_to_gap_m,
    gap_m_to_drive,
    near_target_attenuate,
    next_curriculum_stage,
    pick_place_observation,
    reward_grasp,
    reward_keypoints,
    reward_lift,
    reward_lower,
    reward_place,
    reward_place_xy,
    reward_place_z,
    reward_reach,
    reward_release,
    reward_success,
    sample_object_and_target,
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


def test_reward_release_gated_low_only() -> None:
    target = torch.tensor([[0.30, 0.30, 0.015]])
    gap = torch.tensor([0.08])
    ever = torch.ones(1, dtype=torch.bool)
    high = torch.tensor([[0.30, 0.30, 0.090]])
    low = torch.tensor([[0.30, 0.30, 0.020]])
    high_rew = reward_release(high, target, gap, 0.084, ever, 0.015, 0.04)
    low_rew = reward_release(low, target, gap, 0.084, ever, 0.015, 0.04)
    assert high_rew[0].item() == pytest.approx(0.0)
    assert low_rew[0].item() > 0.0


def test_reward_release_zero_when_far() -> None:
    obj = torch.tensor([[0.30, 0.0, 0.015]])
    target = torch.tensor([[0.30, 0.30, 0.015]])
    gap = torch.tensor([0.08])
    ever = torch.ones(1, dtype=torch.bool)
    rew = reward_release(obj, target, gap, 0.084, ever, 0.015, 0.04)
    assert rew[0].item() == pytest.approx(0.0)


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
        episode_place_success=false,
        episode_success=false,
        place_stable_steps=zeros,
        obj_rest_z_base=0.015,
        object_width=0.03,
        place_success_dist_m=0.04,
        success_hold_steps=3,
    )
    assert bool(state.grasped[0].item())

    # Place success path: on table, released, stable, near target, previously grasped
    obj_place = torch.tensor([[0.3, 0.0, 0.015]])
    gap_open = torch.tensor([0.08])
    ever = torch.ones(n, dtype=torch.bool)
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
            episode_place_success=place_false,
            episode_success=place_false,
            place_stable_steps=steps,
            obj_rest_z_base=0.015,
            object_width=0.03,
            place_success_dist_m=0.04,
            success_hold_steps=3,
        )
        steps = st.place_stable_steps
        place_false = st.episode_place_success
        ever = st.ever_grasped
    assert bool(st.episode_success[0].item())


def test_curriculum_transitions() -> None:
    assert next_curriculum_stage(0, grasp_rate=0.6, place_rate=0.0, grasp_count=500, place_count=0) == 1
    assert next_curriculum_stage(2, grasp_rate=0.0, place_rate=0.65, grasp_count=0, place_count=500) == 3
    assert next_curriculum_stage(0, grasp_rate=0.6, place_rate=0.0, grasp_count=100, place_count=0) == 0


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
