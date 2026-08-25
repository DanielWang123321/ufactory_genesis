"""CPU tests for quality-evaluation snapshots and deterministic perturbations."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ufactory.training.logic.pick_place import pick_place_observation

from examples.rl.pick_place.evaluate import (
    _acceptance_target_results,
    _pin_eval_curriculum_stage,
    _resolve_eval_performance_mode,
)
from examples.rl.pick_place.env import XArm6PickPlaceEnv
from examples.rl.pick_place.expert import (
    PHASE_APPROACH,
    PHASE_CLOSE,
    PHASE_DESCEND,
    PHASE_LIFT,
    PHASE_RELEASE,
    PHASE_RETREAT,
    PHASE_SETDOWN,
    PHASE_SETTLE,
    PHASE_TRANSPORT,
    ScriptedPickPlaceExpert,
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


@pytest.mark.parametrize("stage", range(5))
def test_evaluation_pins_initial_and_maximum_curriculum_stage(stage: int) -> None:
    env_cfg = {"curriculum_initial_stage": 0, "curriculum_max_stage": 4}
    assert _pin_eval_curriculum_stage(env_cfg, stage) == stage
    assert env_cfg["curriculum_initial_stage"] == stage
    assert env_cfg["curriculum_max_stage"] == stage


def test_evaluation_rejects_invalid_inferred_curriculum_stage() -> None:
    with pytest.raises(ValueError, match="curriculum stage"):
        _pin_eval_curriculum_stage({}, 5)


def test_robustness_aggregate_gate_uses_99_percent_point_estimate_and_quality() -> None:
    stats = {
        "episode_count": 512,
        "success_count": 507,
        "quality_count": 507,
        "final_xy_errors_m": [0.001] * 512,
        "max_pre_lift_xy_values_m": [0.001] * 512,
        "post_release_drift_values_m": [0.001] * 512,
        "max_action_clip": 0.0,
        "max_ik_failure": 0.0,
        "max_ik_jump_reject": 0.0,
    }
    assert _acceptance_target_results(stats) == {"standard": False, "robustness": True}
    stats["success_count"] = 506
    assert not _acceptance_target_results(stats)["robustness"]
    stats["success_count"] = 507
    stats["quality_count"] = 506
    assert not _acceptance_target_results(stats)["robustness"]


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
    source.curriculum_stage_enter_step = 1000
    source.curriculum_max_stage = 0
    source.train_action_noise_std = 0.02
    source.train_action_noise_std_end = 0.02
    source.noise_anneal_steps = 0
    source.train_action_noise_clean_episode_frac = 0.5
    state = source.training_state_dict()

    restored = XArm6PickPlaceEnv.__new__(XArm6PickPlaceEnv)
    restored.total_env_steps = 0
    restored.curriculum_stage = 0
    restored.curriculum_stage_enter_step = 0
    restored.curriculum_max_stage = 0
    restored.train_action_noise_std = 0.02
    restored.train_action_noise_std_end = 0.02
    restored.noise_anneal_steps = 0
    restored.train_action_noise_clean_episode_frac = 0.5
    restored.load_training_state_dict(state)
    assert restored.total_env_steps == 1234
    assert restored.curriculum_stage_enter_step == 1000

    restored.train_action_noise_clean_episode_frac = 0.25
    with pytest.raises(ValueError, match="schedule differs"):
        restored.load_training_state_dict(state)


def test_curriculum_stage_dwell_resets_histories_and_blocks_same_step_skip() -> None:
    env = XArm6PickPlaceEnv.__new__(XArm6PickPlaceEnv)
    env.fixed_demo_layout = False
    env.curriculum_stage = 0
    env.curriculum_max_stage = 4
    env.curriculum_stage_enter_step = 0
    env.curriculum_min_stage_steps = 128
    env.curriculum_min_count = 4
    env.curriculum_stage0_grasp_rate = 0.95
    env.curriculum_stage1_grasp_rate = 0.90
    env.curriculum_stage2_place_rate = 0.90
    env.curriculum_stage3_place_rate = 0.85
    env.total_env_steps = 127
    for name in (
        "learned_grasp_success_history",
        "learned_lift_success_history",
        "learned_place_success_history",
        "learned_success_history",
    ):
        setattr(env, name, torch.ones(4))
    env.learned_grasp_history_idx = 3
    env.learned_grasp_history_count = 4
    env.learned_lift_history_idx = 3
    env.learned_lift_history_count = 4
    env.learned_place_history_idx = 3
    env.learned_place_history_count = 4
    env.learned_success_history_idx = 3
    env.learned_success_history_count = 4
    env._metric_cohorts = ("learned_clean",)
    env.cohort_grasp_histories = {"learned_clean": torch.ones(4)}
    env.cohort_success_histories = {"learned_clean": torch.ones(4)}
    env.cohort_history_indices = {"learned_clean": 3}
    env.cohort_history_counts = {"learned_clean": 4}

    env._maybe_update_curriculum()
    assert env.curriculum_stage == 0

    env.total_env_steps = 128
    env._maybe_update_curriculum()
    assert env.curriculum_stage == 1
    assert env.curriculum_stage_enter_step == 128
    assert env.learned_grasp_history_count == 0
    assert env.learned_grasp_history_idx == 0
    assert not bool(env.learned_grasp_success_history.any())
    assert env.cohort_history_counts["learned_clean"] == 0

    env.learned_grasp_success_history.fill_(1.0)
    env.learned_grasp_history_count = 4
    env._maybe_update_curriculum()
    assert env.curriculum_stage == 1

    env.total_env_steps = 256
    env._maybe_update_curriculum()
    assert env.curriculum_stage == 2


def test_late_previous_stage_episode_does_not_repopulate_current_gate() -> None:
    env = XArm6PickPlaceEnv.__new__(XArm6PickPlaceEnv)
    for prefix in ("grasp", "lift", "place", "success"):
        setattr(env, f"{prefix}_success_history" if prefix != "success" else "success_history", torch.zeros(8))
        setattr(env, f"{prefix}_history_idx", 0)
        setattr(env, f"{prefix}_history_count", 0)
        setattr(
            env,
            f"learned_{prefix}_success_history" if prefix != "success" else "learned_success_history",
            torch.zeros(8),
        )
        setattr(env, f"learned_{prefix}_history_idx", 0)
        setattr(env, f"learned_{prefix}_history_count", 0)
    env.curriculum_stage = 1
    env.episode_curriculum_stage = torch.tensor([0, 1])
    env.ever_grasped = torch.tensor([False, True])
    env.ever_lifted = torch.tensor([False, True])
    env.episode_place_success = torch.tensor([False, True])
    env.episode_success = torch.tensor([False, True])
    env.is_bootstrap_episode = torch.tensor([False, False])
    env.episode_action_noise_enabled = torch.tensor([False, False])
    env._metric_cohorts = ("learned_clean", "learned_noisy", "bootstrap_clean", "bootstrap_noisy")
    env.cohort_grasp_histories = {label: torch.zeros(8) for label in env._metric_cohorts}
    env.cohort_success_histories = {label: torch.zeros(8) for label in env._metric_cohorts}
    env.cohort_history_indices = {label: 0 for label in env._metric_cohorts}
    env.cohort_history_counts = {label: 0 for label in env._metric_cohorts}

    env._record_episode_outcomes(torch.tensor([0, 1]))

    assert env.grasp_history_count == 2
    assert env.learned_grasp_history_count == 1
    assert env.learned_grasp_success_history[0].item() == 1.0
    assert env.learned_success_history_count == 1
    assert env.learned_success_history[0].item() == 1.0
    assert env.cohort_history_counts["learned_clean"] == 1


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
    with pytest.raises(ValueError, match="positive"):
        expert_phase_sample_weights(phases, transport_weight=0.0)


def test_expert_phase_weights_can_boost_transport_without_near_table() -> None:
    phases = torch.tensor(
        [PHASE_APPROACH, PHASE_TRANSPORT, PHASE_SETDOWN, PHASE_RELEASE],
        dtype=torch.long,
    )
    weights = expert_phase_sample_weights(
        phases,
        near_table_weight=1.0,
        release_retreat_weight=1.0,
        transport_weight=4.0,
    )
    assert weights.tolist() == pytest.approx([1.0, 4.0, 1.0, 1.0])

    balanced_phases = torch.tensor(
        [PHASE_APPROACH, PHASE_TRANSPORT, PHASE_TRANSPORT, PHASE_SETDOWN],
        dtype=torch.long,
    )
    balanced = expert_phase_sample_weights(
        balanced_phases,
        near_table_weight=1.0,
        release_retreat_weight=1.0,
        transport_weight=4.0,
        balance_phases=True,
    )
    totals = {
        phase: float(balanced[balanced_phases == phase].sum())
        for phase in (PHASE_APPROACH, PHASE_TRANSPORT, PHASE_SETDOWN)
    }
    assert balanced.mean().item() == pytest.approx(1.0)
    assert totals[PHASE_TRANSPORT] / totals[PHASE_APPROACH] == pytest.approx(4.0)
    assert totals[PHASE_SETDOWN] / totals[PHASE_APPROACH] == pytest.approx(1.0)


def test_expert_phase_weights_can_invert_release_versus_close() -> None:
    phases = torch.tensor(
        [PHASE_CLOSE, PHASE_LIFT, PHASE_TRANSPORT, PHASE_RELEASE],
        dtype=torch.long,
    )
    weights = expert_phase_sample_weights(
        phases,
        near_table_weight=1.0,
        release_retreat_weight=0.25,
        transport_weight=4.0,
        close_lift_weight=4.0,
    )
    assert weights.tolist() == pytest.approx([4.0, 4.0, 4.0, 0.25])


def test_policy_prefix_puts_obj_to_target_before_grasped() -> None:
    obs = pick_place_observation(
        joint_pos=torch.zeros(1, 6),
        joint_vel=torch.zeros(1, 6),
        ee_base=torch.tensor([[0.30, 0.00, 0.10]]),
        gripper_gap=torch.tensor([0.022]),
        obj_base=torch.tensor([[0.30, 0.00, 0.02]]),
        target_base=torch.tensor([[0.30, 0.30, 0.02]]),
        grasped=torch.tensor([True]),
        ever_grasped=torch.tensor([True]),
    )
    assert obs.shape[-1] == 30
    assert float(obs[0, 25]) == pytest.approx(0.0)
    assert float(obs[0, 26]) == pytest.approx(0.30)
    assert float(obs[0, 28]) == pytest.approx(1.0)


def test_expert_phase_weights_can_remove_duration_bias() -> None:
    phases = torch.tensor(
        [PHASE_APPROACH, PHASE_APPROACH, PHASE_SETDOWN, PHASE_RELEASE],
        dtype=torch.long,
    )
    weights = expert_phase_sample_weights(phases, balance_phases=True)

    totals = {phase: float(weights[phases == phase].sum()) for phase in (PHASE_APPROACH, PHASE_SETDOWN, PHASE_RELEASE)}
    assert weights.mean().item() == pytest.approx(1.0)
    assert totals[PHASE_SETDOWN] / totals[PHASE_APPROACH] == pytest.approx(4.0)
    assert totals[PHASE_RELEASE] / totals[PHASE_APPROACH] == pytest.approx(2.0)


def _cpu_scripted_expert(**overrides) -> ScriptedPickPlaceExpert:
    env = SimpleNamespace(
        device=torch.device("cpu"),
        num_envs=1,
        obj_size=[0.03, 0.03, 0.03],
        grasp_center_offset_z=0.065,
        obj_rest_z_base=0.015,
        gripper_close_gap_m=0.022,
        gripper_min_command_gap_m=0.012,
        gripper_open_gap_m=0.084,
        action_scale=0.005,
        max_cartesian_delta_m=0.005,
        ee_command_integration="commanded",
        gripper_delta_m=0.004,
        commanded_gap=torch.tensor([0.022]),
        ctrl_dt=0.02,
        holding=torch.tensor([False]),
        ever_carried_near=torch.tensor([False]),
    )
    return ScriptedPickPlaceExpert(env, **overrides)


def test_latched_layout_setpoints_ignore_live_cube_jitter() -> None:
    expert = _cpu_scripted_expert(latch_layout_setpoints=True, close_gap_m=0.022)
    expert.phase[:] = PHASE_APPROACH
    expert._latched_obj_xy[:] = torch.tensor([[0.30, 0.00]])
    expert._obj_xy_valid[:] = True
    finger = torch.tensor([[0.30, 0.00, 0.14]])
    obj = torch.tensor([[0.305, 0.004, 0.015]])
    target = torch.tensor([[0.30, 0.30, 0.015]])
    vel = torch.zeros(1, 3)
    height = obj[:, 2] - 0.015
    gap = torch.tensor([0.05])
    desired, *_ = expert._phase_setpoints(finger, obj, target, vel, height, gap)
    assert desired[0, 0].item() == pytest.approx(0.30)
    assert desired[0, 1].item() == pytest.approx(-0.001)

    expert.phase[:] = PHASE_TRANSPORT
    expert._latched_place_xy[:] = torch.tensor([[0.30, 0.29]])
    expert._place_xy_valid[:] = True
    desired, *_ = expert._phase_setpoints(finger, obj, target, vel, height, gap)
    assert desired[0, 0].item() == pytest.approx(0.30)
    assert desired[0, 1].item() == pytest.approx(0.29)


@pytest.mark.parametrize(("object_y", "expected_bias"), [(-0.05, 0.003), (0.0, 0.0), (0.05, -0.003)])
def test_layout_centering_bias_is_bounded_and_shared_by_phase_gate(
    object_y: float,
    expected_bias: float,
) -> None:
    expert = _cpu_scripted_expert(
        latch_layout_setpoints=True,
        grasp_y_compensation_m=0.0,
        grasp_y_centering_gain=0.06,
        grasp_y_compensation_limit_m=0.003,
    )
    expert._latched_obj_xy[:] = torch.tensor([[0.30, object_y]])
    expert._obj_xy_valid[:] = True
    approach_xy = expert._latched_obj_xy.clone()
    assert expert._grasp_y_compensation(approach_xy).item() == pytest.approx(expected_bias)

    expert.phase[:] = PHASE_DESCEND
    finger = torch.tensor([[0.30, object_y + expected_bias, 0.080]])
    obj = torch.tensor([[0.30, object_y, 0.015]])
    expert._advance_phases(
        finger,
        obj,
        torch.tensor([[0.30, 0.30, 0.015]]),
        torch.tensor([0.050]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
    )
    assert expert.phase.item() == PHASE_CLOSE


def test_brake_only_near_table_keeps_coarse_steps_until_the_landing_band() -> None:
    expert = _cpu_scripted_expert(
        latch_layout_setpoints=True,
        brake_only_near_table=True,
        close_gap_m=0.022,
        near_table_margin_m=0.020,
        near_table_xy_step_m=0.0006,
        near_table_z_step_m=0.001,
        coarse_step_m=0.005,
    )
    expert.phase[:] = PHASE_SETDOWN
    expert._latched_place_xy[:] = torch.tensor([[0.30, 0.30]])
    expert._place_xy_valid[:] = True
    finger = torch.tensor([[0.30, 0.30, 0.10]])
    obj = torch.tensor([[0.30, 0.30, 0.095]])
    target = torch.tensor([[0.30, 0.30, 0.015]])
    vel = torch.zeros(1, 3)
    height = obj[:, 2] - 0.015
    gap = torch.tensor([0.022])
    _, max_xy, max_z, _ = expert._phase_setpoints(finger, obj, target, vel, height, gap)
    assert max_xy[0, 0].item() == pytest.approx(0.005)
    assert max_z[0, 0].item() == pytest.approx(0.003)

    obj = torch.tensor([[0.30, 0.30, 0.025]])
    height = obj[:, 2] - 0.015
    _, max_xy, max_z, _ = expert._phase_setpoints(finger, obj, target, vel, height, gap)
    assert max_xy[0, 0].item() == pytest.approx(0.0006)
    assert max_z[0, 0].item() <= 0.001 + 1e-12
