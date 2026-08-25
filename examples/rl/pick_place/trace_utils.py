"""Genesis-free per-step diagnostic trace helpers for pick-place evaluation.

Kept import-light (torch only) so the trace logic is unit-testable without a
Genesis runtime. ``evaluate.py`` imports these and feeds a live environment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch

ACTION_AXIS_NAMES = ("x", "y", "z", "gripper")


def action_noise_episode_mask(
    uniform_samples: torch.Tensor,
    *,
    clean_episode_frac: float,
    noise_available: bool,
) -> torch.Tensor:
    """Assign a fixed clean/noisy cohort for the lifetime of each episode."""

    fraction = float(clean_episode_frac)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("clean_episode_frac must be in [0, 1]")
    if not noise_available:
        return torch.zeros_like(uniform_samples, dtype=torch.bool)
    return uniform_samples >= fraction


@dataclass(frozen=True)
class ActionNoiseBank:
    """A standard-normal action perturbation indexed by episode and step.

    Unlike a stateful random generator, these samples do not change when evaluation
    batch size, checkpoint order, or episode completion order changes.
    """

    values: np.ndarray
    source: str
    sha256: str
    seed: int | None

    def sample(
        self,
        episode_ids: torch.Tensor,
        step_indices: torch.Tensor,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ids = episode_ids.detach().to(device="cpu", dtype=torch.int64).numpy()
        steps = step_indices.detach().to(device="cpu", dtype=torch.int64).numpy()
        if ids.shape != steps.shape:
            raise ValueError("noise-bank episode IDs and step indices must have matching shapes")
        if ids.size and (ids.min() < 0 or ids.max() >= self.values.shape[0]):
            raise IndexError("noise-bank episode ID is out of range")
        if steps.size and (steps.min() < 0 or steps.max() >= self.values.shape[1]):
            raise IndexError("noise-bank step index is out of range")
        samples = np.ascontiguousarray(self.values[ids, steps])
        return torch.from_numpy(samples).to(device=device, dtype=dtype)


def _noise_bank_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def build_action_noise_bank(
    spec: str | int,
    *,
    episode_count: int,
    max_steps: int,
    action_dim: int,
) -> ActionNoiseBank:
    """Load an ``.npz`` bank or generate one from an integer seed."""

    if min(int(episode_count), int(max_steps), int(action_dim)) <= 0:
        raise ValueError("noise-bank dimensions must be positive")
    path = Path(str(spec)).expanduser()
    seed: int | None
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            if "noise" not in archive:
                raise ValueError(f"action-noise bank {path} has no 'noise' array")
            values = np.asarray(archive["noise"], dtype=np.float32)
        required = (int(episode_count), int(max_steps), int(action_dim))
        if values.ndim != 3 or any(values.shape[i] < required[i] for i in range(3)):
            raise ValueError(f"action-noise bank shape {values.shape} is smaller than required {required}")
        values = np.ascontiguousarray(
            values[: required[0], : required[1], : required[2]],
            dtype=np.float32,
        )
        source = str(path.resolve())
        seed = None
    else:
        try:
            seed = int(spec)
        except (TypeError, ValueError) as exc:
            raise FileNotFoundError(
                f"action-noise bank must be an integer seed or an existing .npz file: {spec}"
            ) from exc
        if seed < 0:
            raise ValueError("action-noise bank seed must be non-negative")
        rng = np.random.default_rng(seed)
        values = rng.standard_normal(
            (int(episode_count), int(max_steps), int(action_dim)),
            dtype=np.float32,
        )
        source = f"seed:{seed}"
    return ActionNoiseBank(
        values=values,
        source=source,
        sha256=_noise_bank_sha256(values),
        seed=seed,
    )


def apply_action_noise_bank(
    actions: torch.Tensor,
    *,
    std: float,
    bank: ActionNoiseBank,
    episode_ids: torch.Tensor,
    step_indices: torch.Tensor,
    action_clip: float,
    axis_mask: Sequence[bool] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the bank samples selected by stable episode/step identifiers."""

    if std < 0.0:
        raise ValueError("action noise std must be non-negative")
    if actions.ndim != 2:
        raise ValueError("actions must have shape (N, action_dim)")
    if episode_ids.numel() != actions.shape[0] or step_indices.numel() != actions.shape[0]:
        raise ValueError("one noise-bank episode ID and step index is required per action row")
    if std == 0.0:
        return actions
    noise = bank.sample(
        episode_ids,
        step_indices,
        device=actions.device,
        dtype=actions.dtype,
    )
    if axis_mask is not None:
        mask = torch.as_tensor(axis_mask, device=actions.device, dtype=actions.dtype)
        if mask.shape != (actions.shape[-1],):
            raise ValueError(
                "action noise axis mask must have one value per action: "
                f"got {tuple(mask.shape)}, expected ({actions.shape[-1]},)"
            )
        noise = noise * mask
    return torch.clamp(
        actions + float(std) * noise,
        -float(action_clip),
        float(action_clip),
    )


TRACE_BASE_FIELDS = [
    "episode",
    "step",
    "ee_x",
    "ee_y",
    "ee_z",
    "ik_ee_x",
    "ik_ee_y",
    "ik_ee_z",
    "setpoint_x",
    "setpoint_y",
    "setpoint_z",
    "setpoint_residual_x",
    "setpoint_residual_y",
    "setpoint_residual_z",
    "setpoint_residual_norm_m",
    "obj_x",
    "obj_y",
    "obj_z",
    "obj_vx",
    "obj_vy",
    "obj_vz",
    "obj_speed_m_s",
    "obj_xy_speed_m_s",
    "obj_down_speed_m_s",
    "target_x",
    "target_y",
    "target_z",
    "target_xy_dist",
    "gap_m",
    "commanded_gap_m",
    "applied_gripper_gap_m",
    "gripper_force_hold_latched",
    "action_x",
    "action_y",
    "action_z",
    "action_gripper",
    "policy_action_x",
    "policy_action_y",
    "policy_action_z",
    "policy_action_gripper",
    "action_noise_x",
    "action_noise_y",
    "action_noise_z",
    "action_noise_gripper",
    "left_contact_force_n",
    "right_contact_force_n",
    "grasp_x",
    "grasp_y",
    "grasp_z",
    "dist_to_grasp",
    "holding",
    "carry",
    "ever_grasped",
    "ever_carried_near",
    "release_started",
    "release_valid",
    "release_violation",
    "task_phase",
    "hard_landing_event",
    "hard_landing_violation",
    "quality_ok",
    "max_pre_lift_xy_m",
    "release_xy_dist_m",
    "release_height_error_m",
    "release_speed_m_s",
    "max_landing_xy_speed_m_s",
    "max_landing_down_speed_m_s",
    "post_release_drift_m",
    "post_release_clearance_m",
    "release_clearance_achieved",
    "post_release_recontact_event",
    "post_release_recontact",
    "first_push_event",
    "first_lift_event",
    "release_event",
    "table_contact_event",
    "final_stable_event",
    "action_near_bound_fraction",
    "reward",
    "done",
]

HARD_EVENT_FIELDS = [
    "action_noise_episode_id",
    "env_index",
    "step",
    "task_phase",
    "policy_action_x",
    "policy_action_y",
    "policy_action_z",
    "policy_action_gripper",
    "action_x",
    "action_y",
    "action_z",
    "action_gripper",
    "action_noise_x",
    "action_noise_y",
    "action_noise_z",
    "action_noise_gripper",
    "left_contact_force_n",
    "right_contact_force_n",
    "obj_vx",
    "obj_vy",
    "obj_vz",
    "ee_vx",
    "ee_vy",
    "ee_vz",
    "setpoint_residual_x",
    "setpoint_residual_y",
    "setpoint_residual_z",
    "obj_x",
    "obj_y",
    "obj_z",
    "ee_x",
    "ee_y",
    "ee_z",
]


def trace_fieldnames(env) -> list[str]:
    """Base kinematic/flag columns plus one rew_* column per active reward term."""
    return TRACE_BASE_FIELDS + ["rew_" + name for name in env.reward_scales.keys()]


def hard_event_trace_row(
    snapshot: dict,
    *,
    index: int,
    step: int,
    action_noise_episode_id: int,
    policy_action: torch.Tensor | None = None,
    executed_action: torch.Tensor | None = None,
) -> dict:
    """Build one event row before the terminal environment can be reset."""

    clean = snapshot.get("policy_actions", snapshot["actions"])[index] if policy_action is None else policy_action
    executed = snapshot["actions"][index] if executed_action is None else executed_action
    noise = executed - clean
    ee = snapshot["ee_base"][index]
    obj = snapshot["obj_base"][index]
    obj_vel = snapshot["obj_vel"][index]
    ee_vel = snapshot.get("ee_velocity_base", torch.zeros_like(snapshot["ee_base"]))[index]
    residual = snapshot["ee_setpoint_base"][index] - snapshot["ik_ee_base"][index]
    left_force = snapshot.get("left_contact_force_n", torch.zeros_like(snapshot["gap_m"]))
    right_force = snapshot.get("right_contact_force_n", torch.zeros_like(snapshot["gap_m"]))
    return {
        "action_noise_episode_id": int(action_noise_episode_id),
        "env_index": int(index),
        "step": int(step),
        "task_phase": task_phase_label(snapshot, index),
        "policy_action_x": float(clean[0].item()),
        "policy_action_y": float(clean[1].item()),
        "policy_action_z": float(clean[2].item()),
        "policy_action_gripper": float(clean[3].item()),
        "action_x": float(executed[0].item()),
        "action_y": float(executed[1].item()),
        "action_z": float(executed[2].item()),
        "action_gripper": float(executed[3].item()),
        "action_noise_x": float(noise[0].item()),
        "action_noise_y": float(noise[1].item()),
        "action_noise_z": float(noise[2].item()),
        "action_noise_gripper": float(noise[3].item()),
        "left_contact_force_n": float(left_force[index].item()),
        "right_contact_force_n": float(right_force[index].item()),
        "obj_vx": float(obj_vel[0].item()),
        "obj_vy": float(obj_vel[1].item()),
        "obj_vz": float(obj_vel[2].item()),
        "ee_vx": float(ee_vel[0].item()),
        "ee_vy": float(ee_vel[1].item()),
        "ee_vz": float(ee_vel[2].item()),
        "setpoint_residual_x": float(residual[0].item()),
        "setpoint_residual_y": float(residual[1].item()),
        "setpoint_residual_z": float(residual[2].item()),
        "obj_x": float(obj[0].item()),
        "obj_y": float(obj[1].item()),
        "obj_z": float(obj[2].item()),
        "ee_x": float(ee[0].item()),
        "ee_y": float(ee[1].item()),
        "ee_z": float(ee[2].item()),
    }


def apply_deterministic_action_noise(
    actions: torch.Tensor,
    *,
    std: float,
    generator: torch.Generator,
    action_clip: float,
    axis_mask: Sequence[bool] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply one reproducible Gaussian action-noise sample and enforce action bounds.

    A full noise tensor is sampled before masking.  Consequently, evaluations
    using the same seed are paired across all-axis and single-axis ablations.
    """

    if std < 0.0:
        raise ValueError("action noise std must be non-negative")
    mask = None
    if axis_mask is not None:
        mask = torch.as_tensor(axis_mask, device=actions.device, dtype=actions.dtype)
        if mask.ndim != 1 or mask.shape[0] != actions.shape[-1]:
            raise ValueError(
                "action noise axis mask must have one value per action: "
                f"got {tuple(mask.shape)}, expected ({actions.shape[-1]},)"
            )
    if std == 0.0:
        return actions
    noise = torch.randn(
        actions.shape,
        generator=generator,
        device=actions.device,
        dtype=actions.dtype,
    )
    if mask is not None:
        noise = noise * mask
    return torch.clamp(actions + float(std) * noise, -float(action_clip), float(action_clip))


def action_noise_axis_mask(
    axes: Sequence[str],
    *,
    axis_names: Sequence[str] = ACTION_AXIS_NAMES,
) -> tuple[bool, ...]:
    """Convert named action axes into a validated Boolean mask."""

    names = tuple(str(name) for name in axis_names)
    selected = tuple(str(axis) for axis in axes)
    if not selected:
        raise ValueError("at least one action-noise axis must be selected")
    unknown = sorted(set(selected) - set(names))
    if unknown:
        raise ValueError(f"unknown action-noise axes: {unknown}; expected a subset of {names}")
    if len(set(selected)) != len(selected):
        raise ValueError("action-noise axes must not contain duplicates")
    return tuple(name in selected for name in names)


def scheduled_action_noise_std(
    start_std: float,
    end_std: float,
    anneal_steps: int,
    total_env_steps: int,
) -> float:
    """Annealed execution-noise std for a given batch-step count.

    Fixed at ``start_std`` when no curriculum is configured (``anneal_steps <= 0``),
    otherwise linear interpolation ``start_std -> end_std`` clamped to full strength.
    """

    if start_std < 0.0 or end_std < 0.0:
        raise ValueError("noise std values must be non-negative")
    if anneal_steps < 0:
        raise ValueError("noise_anneal_steps must be non-negative")
    if anneal_steps <= 0:
        return float(start_std)
    progress = total_env_steps / float(anneal_steps)
    p = min(1.0, max(0.0, progress))
    return float(start_std) + (float(end_std) - float(start_std)) * p


def disable_training_action_noise_for_evaluation(env_cfg: dict) -> float:
    """Disable saved training-time noise and return its configured value.

    Also clears the noise-curriculum keys (start/end/anneal), because the
    evaluation env must run deterministically unless --action-noise-std adds
    the explicit robustness-trial perturbation.
    """

    saved_std = float(env_cfg.get("train_action_noise_std", 0.0))
    if saved_std < 0.0:
        raise ValueError("train_action_noise_std must be non-negative")
    env_cfg["train_action_noise_std"] = 0.0
    env_cfg["train_action_noise_std_end"] = 0.0
    env_cfg["noise_anneal_steps"] = 0
    env_cfg["train_action_noise_clean_episode_frac"] = 1.0
    return saved_std


def task_phase_label(snapshot: dict, index: int = 0) -> str:
    """Classify a transition into a compact diagnostic task phase."""

    def flag(name: str) -> bool:
        return bool(snapshot[name][index].item())

    if flag("release_event"):
        return "release_step"
    if flag("release_started"):
        return "post_release"
    if flag("table_contact_event"):
        return "near_table_entry_pre_release"
    if flag("near_table_entered"):
        return "near_table_pre_release"
    if flag("ever_carried_near"):
        return "setdown"
    if flag("ever_grasped"):
        return "transport"
    return "pre_grasp"


def scenario_layout_key(
    initial_obj_pos: torch.Tensor,
    target_pos: torch.Tensor,
    *,
    decimals: int = 9,
) -> tuple[float, ...]:
    """Canonical hashable key for counting physically unique scene layouts."""

    values = torch.cat([initial_obj_pos, target_pos]).detach().cpu().tolist()
    return tuple(round(float(value), decimals) for value in values)


def confidence_intervals_applicable(
    *,
    unique_scenario_count: int,
    episode_count: int,
    action_noise_std: float,
) -> bool:
    """Whether episodes contain independent layout or action perturbations."""

    return bool(episode_count > 0 and (action_noise_std > 0.0 or unique_scenario_count == episode_count))


def save_rgb_frame(frame, output_path: Path) -> None:
    """Normalize a Genesis RGB buffer and save it as an 8-bit PNG."""

    from PIL import Image

    image = np.asarray(frame)
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 255.0)
        if float(image.max(initial=0.0)) <= 1.0:
            image = image * 255.0
        image = image.astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path)


def env0_trace_row(
    env,
    *,
    episode: int,
    step: int,
    reward_value: float,
    done_flag: bool,
    action: torch.Tensor | None = None,
    policy_action: torch.Tensor | None = None,
) -> dict:
    """Build env-0 trace data from the environment's pre-reset step snapshot."""

    snapshot = env.extras.get("step_snapshot")
    if not isinstance(snapshot, dict):
        raise RuntimeError("environment did not provide a pre-reset step_snapshot")
    ee = snapshot["ee_base"][0]
    ik_ee = snapshot["ik_ee_base"][0]
    setpoint = snapshot["ee_setpoint_base"][0]
    setpoint_residual = setpoint - ik_ee
    obj = snapshot["obj_base"][0]
    obj_vel = snapshot["obj_vel"][0]
    target = snapshot["target_pos"][0]
    grasp = snapshot["grasp_pos"][0]
    gap = float(snapshot["gap_m"][0].item())
    dist = float(torch.norm(ee - grasp).item())
    step_action = snapshot["actions"][0] if action is None else action
    clean_action = snapshot.get("policy_actions", snapshot["actions"])[0] if policy_action is None else policy_action
    action_noise = (
        snapshot.get("action_noise", snapshot["actions"] - clean_action)[0]
        if action is None and policy_action is None
        else step_action - clean_action
    )
    row = {
        "episode": episode,
        "step": step,
        "ee_x": float(ee[0].item()),
        "ee_y": float(ee[1].item()),
        "ee_z": float(ee[2].item()),
        "ik_ee_x": float(ik_ee[0].item()),
        "ik_ee_y": float(ik_ee[1].item()),
        "ik_ee_z": float(ik_ee[2].item()),
        "setpoint_x": float(setpoint[0].item()),
        "setpoint_y": float(setpoint[1].item()),
        "setpoint_z": float(setpoint[2].item()),
        "setpoint_residual_x": float(setpoint_residual[0].item()),
        "setpoint_residual_y": float(setpoint_residual[1].item()),
        "setpoint_residual_z": float(setpoint_residual[2].item()),
        "setpoint_residual_norm_m": float(torch.norm(setpoint_residual).item()),
        "obj_x": float(obj[0].item()),
        "obj_y": float(obj[1].item()),
        "obj_z": float(obj[2].item()),
        "obj_vx": float(obj_vel[0].item()),
        "obj_vy": float(obj_vel[1].item()),
        "obj_vz": float(obj_vel[2].item()),
        "obj_speed_m_s": float(torch.norm(obj_vel).item()),
        "obj_xy_speed_m_s": float(torch.norm(obj_vel[:2]).item()),
        "obj_down_speed_m_s": float(torch.clamp(-obj_vel[2], min=0.0).item()),
        "target_x": float(target[0].item()),
        "target_y": float(target[1].item()),
        "target_z": float(target[2].item()),
        "target_xy_dist": float(torch.norm(obj[:2] - target[:2]).item()),
        "gap_m": gap,
        "commanded_gap_m": float(snapshot["commanded_gap_m"][0].item()),
        "applied_gripper_gap_m": float(snapshot.get("applied_gripper_gap_m", snapshot["commanded_gap_m"])[0].item()),
        "gripper_force_hold_latched": int(
            bool(snapshot.get("gripper_force_hold_latched", torch.zeros(1, dtype=torch.bool))[0].item())
        ),
        "action_x": float(step_action[0].item()),
        "action_y": float(step_action[1].item()),
        "action_z": float(step_action[2].item()),
        "action_gripper": float(step_action[3].item()),
        "policy_action_x": float(clean_action[0].item()),
        "policy_action_y": float(clean_action[1].item()),
        "policy_action_z": float(clean_action[2].item()),
        "policy_action_gripper": float(clean_action[3].item()),
        "action_noise_x": float(action_noise[0].item()),
        "action_noise_y": float(action_noise[1].item()),
        "action_noise_z": float(action_noise[2].item()),
        "action_noise_gripper": float(action_noise[3].item()),
        "left_contact_force_n": float(snapshot.get("left_contact_force_n", torch.zeros(1))[0].item()),
        "right_contact_force_n": float(snapshot.get("right_contact_force_n", torch.zeros(1))[0].item()),
        "grasp_x": float(grasp[0].item()),
        "grasp_y": float(grasp[1].item()),
        "grasp_z": float(grasp[2].item()),
        "dist_to_grasp": dist,
        "holding": int(bool(snapshot["holding"][0].item())),
        "carry": int(bool(snapshot["carry"][0].item())),
        "ever_grasped": int(bool(snapshot["ever_grasped"][0].item())),
        "ever_carried_near": int(bool(snapshot["ever_carried_near"][0].item())),
        "release_started": int(bool(snapshot["release_started"][0].item())),
        "release_valid": int(bool(snapshot["release_valid"][0].item())),
        "release_violation": int(bool(snapshot["release_violation"][0].item())),
        "task_phase": task_phase_label(snapshot),
        "hard_landing_event": int(bool(snapshot["hard_landing_event"][0].item())),
        "hard_landing_violation": int(bool(snapshot["hard_landing_violation"][0].item())),
        "quality_ok": int(bool(snapshot["quality_ok"][0].item())),
        "max_pre_lift_xy_m": float(snapshot["max_pre_lift_xy_m"][0].item()),
        "release_xy_dist_m": float(snapshot["release_xy_dist_m"][0].item()),
        "release_height_error_m": float(snapshot["release_height_error_m"][0].item()),
        "release_speed_m_s": float(snapshot["release_speed_m_s"][0].item()),
        "max_landing_xy_speed_m_s": float(snapshot["max_landing_xy_speed_m_s"][0].item()),
        "max_landing_down_speed_m_s": float(snapshot["max_landing_down_speed_m_s"][0].item()),
        "post_release_drift_m": float(snapshot["post_release_drift_m"][0].item()),
        "post_release_clearance_m": float(snapshot.get("post_release_clearance_m", torch.zeros(1))[0].item()),
        "release_clearance_achieved": int(bool(snapshot.get("release_clearance_achieved", torch.zeros(1))[0].item())),
        "post_release_recontact_event": int(
            bool(snapshot.get("post_release_recontact_event", torch.zeros(1))[0].item())
        ),
        "post_release_recontact": int(bool(snapshot.get("post_release_recontact", torch.zeros(1))[0].item())),
        "first_push_event": int(bool(snapshot["first_push_event"][0].item())),
        "first_lift_event": int(bool(snapshot["first_lift_event"][0].item())),
        "release_event": int(bool(snapshot["release_event"][0].item())),
        "table_contact_event": int(bool(snapshot["table_contact_event"][0].item())),
        "final_stable_event": int(bool(snapshot["final_stable_event"][0].item())),
        "action_near_bound_fraction": float((step_action.abs() >= 0.95 * float(env.action_clip)).float().mean().item()),
        "reward": float(reward_value),
        "done": int(bool(done_flag)),
    }
    reward_terms = snapshot["reward_terms"]
    for name in env.reward_scales.keys():
        row["rew_" + name] = float(reward_terms[name][0].item())
    return row
