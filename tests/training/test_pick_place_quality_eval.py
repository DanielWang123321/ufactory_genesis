"""CPU tests for quality-evaluation snapshots and deterministic perturbations."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.rl.pick_place.evaluate import _resolve_eval_performance_mode
from examples.rl.pick_place.env import XArm6PickPlaceEnv
from examples.rl.pick_place.expert import (
    PHASE_APPROACH,
    PHASE_RELEASE,
    PHASE_RETREAT,
    PHASE_SETDOWN,
    PHASE_SETTLE,
    expert_phase_sample_weights,
)
from examples.rl.pick_place.trace_utils import (
    action_noise_episode_mask,
    action_noise_axis_mask,
    apply_action_noise_bank,
    apply_deterministic_action_noise,
    build_action_noise_bank,
    confidence_intervals_applicable,
    disable_training_action_noise_for_evaluation,
    env0_trace_row,
    hard_event_trace_row,
    save_rgb_frame,
    scheduled_action_noise_std,
    scenario_layout_key,
    task_phase_label,
    trace_fieldnames,
)


class _FakeEnv:
    action_clip = 1.0
    reward_scales = {"valid_release": 20.0, "hard_landing": 50.0}

    def __init__(self) -> None:
        one = torch.tensor([1.0])
        zero = torch.tensor([0.0])
        true = torch.tensor([True])
        false = torch.tensor([False])
        self.extras = {
            "step_snapshot": {
                "ee_base": torch.tensor([[0.30, 0.30, 0.08]]),
                "ik_ee_base": torch.tensor([[0.30, 0.30, 0.09]]),
                "ee_setpoint_base": torch.tensor([[0.301, 0.298, 0.093]]),
                "obj_base": torch.tensor([[0.30, 0.30, 0.015]]),
                "obj_vel": torch.tensor([[0.01, 0.02, -0.03]]),
                "target_pos": torch.tensor([[0.30, 0.30, 0.015]]),
                "grasp_pos": torch.tensor([[0.30, 0.30, 0.08]]),
                "gap_m": torch.tensor([0.05]),
                "commanded_gap_m": torch.tensor([0.06]),
                "actions": torch.tensor([[0.96, 0.0, -0.96, 0.5]]),
                "holding": false,
                "carry": false,
                "ever_grasped": true,
                "ever_carried_near": true,
                "release_started": true,
                "release_valid": true,
                "release_violation": false,
                "near_table_entered": true,
                "hard_landing_event": false,
                "hard_landing_violation": false,
                "quality_ok": true,
                "max_pre_lift_xy_m": torch.tensor([0.004]),
                "release_xy_dist_m": torch.tensor([0.006]),
                "release_height_error_m": torch.tensor([0.003]),
                "release_speed_m_s": torch.tensor([0.015]),
                "max_landing_xy_speed_m_s": torch.tensor([0.025]),
                "max_landing_down_speed_m_s": torch.tensor([0.04]),
                "post_release_drift_m": torch.tensor([0.002]),
                "first_push_event": false,
                "first_lift_event": false,
                "release_event": true,
                "table_contact_event": true,
                "final_stable_event": true,
                "reward": one,
                "done": true,
                "reward_terms": {
                    "valid_release": torch.tensor([20.0]),
                    "hard_landing": zero,
                },
            }
        }


@pytest.mark.parametrize(
    ("headless", "cli_override", "saved_mode", "expected_mode", "expected_source"),
    (
        (False, None, True, False, "viewer fast-start default"),
        (False, True, False, True, "CLI override"),
        (False, False, True, False, "CLI override"),
        (True, None, True, True, "saved training config (headless default)"),
        (True, None, False, False, "saved training config (headless default)"),
        (True, False, True, False, "CLI override"),
    ),
)
def test_evaluation_resolves_genesis_performance_mode_without_mutating_artifact(
    headless: bool,
    cli_override: bool | None,
    saved_mode: bool,
    expected_mode: bool,
    expected_source: str,
) -> None:
    args = SimpleNamespace(headless=headless, performance_mode=cli_override)
    env_cfg = {"genesis_performance_mode": saved_mode}

    resolved = _resolve_eval_performance_mode(args, env_cfg)

    assert resolved == (expected_mode, expected_source)
    assert env_cfg == {"genesis_performance_mode": saved_mode}


def test_terminal_trace_uses_pre_reset_snapshot_and_actual_reward_terms() -> None:
    env = _FakeEnv()
    row = env0_trace_row(
        env,
        episode=7,
        step=123,
        reward_value=20.0,
        done_flag=True,
    )
    assert row["done"] == 1
    assert row["obj_x"] == pytest.approx(0.30)
    assert row["obj_vz"] == pytest.approx(-0.03)
    assert row["obj_down_speed_m_s"] == pytest.approx(0.03)
    assert row["setpoint_residual_x"] == pytest.approx(0.001, abs=1e-7)
    assert row["setpoint_residual_y"] == pytest.approx(-0.002, abs=1e-7)
    assert row["setpoint_residual_z"] == pytest.approx(0.003, abs=1e-7)
    assert row["release_event"] == 1
    assert row["task_phase"] == "release_step"
    assert row["final_stable_event"] == 1
    assert row["rew_valid_release"] == pytest.approx(20.0)
    assert row["rew_hard_landing"] == pytest.approx(0.0)
    assert row["action_near_bound_fraction"] == pytest.approx(0.5)
    assert trace_fieldnames(env)[-2:] == ["rew_valid_release", "rew_hard_landing"]


def test_trace_separates_policy_action_from_executed_noise() -> None:
    env = _FakeEnv()
    executed = torch.tensor([0.12, -0.08, 0.03, 0.40])
    clean = torch.tensor([0.10, -0.10, 0.04, 0.35])
    row = env0_trace_row(
        env,
        episode=1,
        step=1,
        reward_value=0.0,
        done_flag=False,
        action=executed,
        policy_action=clean,
    )
    assert row["policy_action_x"] == pytest.approx(0.10)
    assert row["action_x"] == pytest.approx(0.12)
    assert row["action_noise_x"] == pytest.approx(0.02)
    assert row["action_noise_gripper"] == pytest.approx(0.05)


def test_hard_event_trace_captures_actions_motion_and_residual() -> None:
    snapshot = _FakeEnv().extras["step_snapshot"]
    row = hard_event_trace_row(
        snapshot,
        index=0,
        step=42,
        action_noise_episode_id=9,
        policy_action=torch.tensor([0.10, 0.20, 0.30, 0.40]),
        executed_action=torch.tensor([0.11, 0.18, 0.33, 0.35]),
    )
    assert row["action_noise_episode_id"] == 9
    assert row["step"] == 42
    assert row["action_noise_x"] == pytest.approx(0.01)
    assert row["action_noise_gripper"] == pytest.approx(-0.05)
    assert row["obj_vz"] == pytest.approx(-0.03)
    assert row["setpoint_residual_z"] == pytest.approx(0.003, abs=1e-7)


def test_task_phase_labels_pre_release_and_post_release_events() -> None:
    snapshot = _FakeEnv().extras["step_snapshot"]
    assert task_phase_label(snapshot) == "release_step"
    snapshot["release_event"] = torch.tensor([False])
    assert task_phase_label(snapshot) == "post_release"
    snapshot["release_started"] = torch.tensor([False])
    snapshot["table_contact_event"] = torch.tensor([True])
    assert task_phase_label(snapshot) == "near_table_entry_pre_release"


def test_action_noise_sequence_is_reproducible_and_bounded() -> None:
    actions = torch.zeros(8, 4)
    generator_a = torch.Generator().manual_seed(17)
    generator_b = torch.Generator().manual_seed(17)
    noisy_a = apply_deterministic_action_noise(
        actions,
        std=0.02,
        generator=generator_a,
        action_clip=1.0,
    )
    noisy_b = apply_deterministic_action_noise(
        actions,
        std=0.02,
        generator=generator_b,
        action_clip=1.0,
    )
    assert torch.equal(noisy_a, noisy_b)
    assert not torch.equal(noisy_a, actions)
    assert torch.all(noisy_a.abs() <= 1.0)


def test_action_noise_axis_ablation_keeps_other_coordinates_clean_and_paired() -> None:
    actions = torch.zeros(32, 4)
    all_axes = apply_deterministic_action_noise(
        actions,
        std=0.02,
        generator=torch.Generator().manual_seed(19),
        action_clip=1.0,
        axis_mask=action_noise_axis_mask(("x", "y", "z", "gripper")),
    )
    xy_only = apply_deterministic_action_noise(
        actions,
        std=0.02,
        generator=torch.Generator().manual_seed(19),
        action_clip=1.0,
        axis_mask=action_noise_axis_mask(("x", "y")),
    )
    assert torch.equal(xy_only[:, :2], all_axes[:, :2])
    assert torch.count_nonzero(xy_only[:, 2:]) == 0


def test_action_noise_axis_mask_rejects_empty_unknown_and_duplicate_axes() -> None:
    with pytest.raises(ValueError, match="at least one"):
        action_noise_axis_mask(())
    with pytest.raises(ValueError, match="unknown"):
        action_noise_axis_mask(("roll",))
    with pytest.raises(ValueError, match="duplicates"):
        action_noise_axis_mask(("x", "x"))


def test_action_noise_bank_is_batch_order_independent() -> None:
    bank = build_action_noise_bank(
        20260817,
        episode_count=8,
        max_steps=12,
        action_dim=4,
    )
    clean = torch.zeros(3, 4)
    episode_ids = torch.tensor([5, 1, 7])
    step_indices = torch.tensor([3, 9, 0])
    together = apply_action_noise_bank(
        clean,
        std=0.02,
        bank=bank,
        episode_ids=episode_ids,
        step_indices=step_indices,
        action_clip=1.0,
    )
    reordered = apply_action_noise_bank(
        clean[[2, 0, 1]],
        std=0.02,
        bank=bank,
        episode_ids=episode_ids[[2, 0, 1]],
        step_indices=step_indices[[2, 0, 1]],
        action_clip=1.0,
    )
    assert torch.equal(together, reordered[[1, 2, 0]])
    assert (
        bank.sha256
        == build_action_noise_bank(
            20260817,
            episode_count=8,
            max_steps=12,
            action_dim=4,
        ).sha256
    )


def test_action_noise_bank_file_round_trip(tmp_path) -> None:
    values = np.arange(3 * 5 * 4, dtype=np.float32).reshape(3, 5, 4)
    path = tmp_path / "noise.npz"
    np.savez(path, noise=values)
    bank = build_action_noise_bank(
        str(path),
        episode_count=2,
        max_steps=4,
        action_dim=4,
    )
    sample = bank.sample(
        torch.tensor([1]),
        torch.tensor([3]),
        device="cpu",
        dtype=torch.float32,
    )
    assert torch.equal(sample[0], torch.from_numpy(values[1, 3]))


def test_action_noise_episode_cohort_is_fixed_by_reset_sample() -> None:
    samples = torch.tensor([0.10, 0.49, 0.50, 0.90])
    assert action_noise_episode_mask(
        samples,
        clean_episode_frac=0.5,
        noise_available=True,
    ).tolist() == [False, False, True, True]
    assert not action_noise_episode_mask(
        samples,
        clean_episode_frac=0.0,
        noise_available=False,
    ).any()
    with pytest.raises(ValueError, match="clean_episode_frac"):
        action_noise_episode_mask(samples, clean_episode_frac=1.1, noise_available=True)


def test_evaluation_disables_saved_training_action_noise() -> None:
    env_cfg = {"train_action_noise_std": 0.02}
    saved = disable_training_action_noise_for_evaluation(env_cfg)
    assert saved == pytest.approx(0.02)
    assert env_cfg["train_action_noise_std"] == 0.0
    assert env_cfg["train_action_noise_clean_episode_frac"] == 1.0


def test_scheduled_action_noise_std_anneals_and_clamps() -> None:
    # No curriculum -> fixed start value regardless of step count.
    assert scheduled_action_noise_std(0.02, 0.02, 0, 0) == pytest.approx(0.02)
    assert scheduled_action_noise_std(0.02, 0.02, 0, 9999) == pytest.approx(0.02)
    # Linear interpolation start -> end over anneal_steps.
    assert scheduled_action_noise_std(0.002, 0.02, 100, 0) == pytest.approx(0.002)
    assert scheduled_action_noise_std(0.002, 0.02, 100, 50) == pytest.approx(0.011)
    assert scheduled_action_noise_std(0.002, 0.02, 100, 100) == pytest.approx(0.02)
    # Clamped at full strength past the anneal horizon and at step 0.
    assert scheduled_action_noise_std(0.002, 0.02, 100, 500) == pytest.approx(0.02)
    assert scheduled_action_noise_std(0.02, 0.002, 100, 0) == pytest.approx(0.02)
    # Negative inputs are rejected.
    for bad in (
        (-0.1, 0.02, 100, 0),
        (0.002, -0.1, 100, 0),
        (0.002, 0.02, -1, 0),
    ):
        with pytest.raises(ValueError):
            scheduled_action_noise_std(*bad)


def test_evaluation_disables_training_noise_curriculum_keys() -> None:
    env_cfg = {
        "train_action_noise_std": 0.002,
        "train_action_noise_std_end": 0.02,
        "noise_anneal_steps": 20000,
        "noise_std_uniform_sample": True,
    }
    saved = disable_training_action_noise_for_evaluation(env_cfg)
    assert saved == pytest.approx(0.002)
    assert env_cfg["train_action_noise_std"] == 0.0
    assert env_cfg["train_action_noise_std_end"] == 0.0
    assert env_cfg["noise_anneal_steps"] == 0


def test_environment_training_state_restores_noise_progress() -> None:
    source = XArm6PickPlaceEnv.__new__(XArm6PickPlaceEnv)
    source.total_env_steps = 1234
    source.curriculum_stage = 0
    source.curriculum_max_stage = 0
    source.train_action_noise_std = 0.02
    source.train_action_noise_std_end = 0.02
    source.noise_anneal_steps = 0
    source.train_action_noise_clean_episode_frac = 0.5
    state = source.training_state_dict()

    restored = XArm6PickPlaceEnv.__new__(XArm6PickPlaceEnv)
    restored.total_env_steps = 0
    restored.curriculum_stage = 0
    restored.curriculum_max_stage = 0
    restored.train_action_noise_std = 0.02
    restored.train_action_noise_std_end = 0.02
    restored.noise_anneal_steps = 0
    restored.train_action_noise_clean_episode_frac = 0.5
    restored.load_training_state_dict(state)
    assert restored.total_env_steps == 1234

    restored.train_action_noise_clean_episode_frac = 0.25
    with pytest.raises(ValueError, match="schedule differs"):
        restored.load_training_state_dict(state)


def test_unique_layout_count_deduplicates_fixed_pose_copies() -> None:
    obj = torch.tensor([0.30, 0.00, 0.015])
    target = torch.tensor([0.30, 0.30, 0.015])
    keys = {scenario_layout_key(obj.clone(), target.clone()) for _ in range(512)}
    assert len(keys) == 1
    shifted = target.clone()
    shifted[0] += 0.001
    keys.add(scenario_layout_key(obj, shifted))
    assert len(keys) == 2
    assert not confidence_intervals_applicable(
        unique_scenario_count=1,
        episode_count=512,
        action_noise_std=0.0,
    )
    assert confidence_intervals_applicable(
        unique_scenario_count=1,
        episode_count=512,
        action_noise_std=0.02,
    )


def test_event_frame_is_saved_as_uint8_png(tmp_path) -> None:
    frame = np.zeros((12, 16, 3), dtype=np.float32)
    frame[..., 0] = 1.0
    path = tmp_path / "events" / "episode_0001_release_start.png"
    save_rgb_frame(frame, path)
    assert path.is_file()
    from PIL import Image

    saved = np.asarray(Image.open(path))
    assert saved.dtype == np.uint8
    assert saved.shape == (12, 16, 3)
    assert np.all(saved[..., 0] == 255)


def test_expert_quality_phase_weights_match_disturbed_bc_recipe() -> None:
    phases = torch.tensor(
        [
            PHASE_APPROACH,
            PHASE_SETDOWN,
            PHASE_SETTLE,
            PHASE_RELEASE,
            PHASE_RETREAT,
        ]
    )
    assert expert_phase_sample_weights(phases).tolist() == pytest.approx([1.0, 4.0, 4.0, 2.0, 2.0])
    with pytest.raises(ValueError, match="positive"):
        expert_phase_sample_weights(phases, near_table_weight=0.0)
