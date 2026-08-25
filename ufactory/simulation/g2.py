"""Genesis 1.3.3 stability policy for physical Gripper G2 contact scenes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any, Mapping

import numpy as np
import torch


G2_PHYSICS_PROFILE = "g2_stable_v1_3_3"
G2_MAX_RIGID_SUBSTEP_DT_S = 0.000625
# Genesis solver tuple:
# (timeconst, dampratio, dmin, dmax, width, midpoint, power).
#
# The historical project value used dmax=0.001 and dampratio=0.1.  In Genesis
# 1.3.3 that creates reference-acceleration coefficients several orders of
# magnitude above the defaults and can overflow when all five mimic equalities
# meet bilateral finger contact.  This deliberately compliant profile passed
# the same 128-environment perturbed-contact case at both 8 and 32 substeps.
G2_MIMIC_CONSTRAINT_SOL_PARAMS = (0.02, 1.0, 0.9, 0.95, 0.001, 0.5, 2.0)
G2_MIMIC_EQUALITY_NAMES = frozenset(
    {
        "mimic_left_finger_joint_to_drive_joint",
        "mimic_left_inner_knuckle_joint_to_drive_joint",
        "mimic_right_outer_knuckle_joint_to_drive_joint",
        "mimic_right_finger_joint_to_drive_joint",
        "mimic_right_inner_knuckle_joint_to_drive_joint",
    }
)


@dataclass(frozen=True)
class G2ContactHoldPolicy:
    """Fixed physical force-hold policy shared by trajectory and RL control."""

    contact_threshold_n: float = 0.2
    contact_confirm_steps: int = 2
    # Empirical Genesis 1.3.3 G2 working point.  A 3 N target let the delayed
    # position-driven linkage open through the load-bearing point during lift;
    # 10 +/- 2 N retains transient margin while remaining well below the
    # roughly 23 N produced by the former open-loop over-close command.
    target_force_n: float = 10.0
    force_deadband_n: float = 2.0
    max_gap_step_m: float = 0.00025
    latch_window_m: float = 0.002
    contact_loss_steps: int = 3

    def __post_init__(self) -> None:
        finite_positive = (
            self.contact_threshold_n,
            self.target_force_n,
            self.force_deadband_n,
            self.max_gap_step_m,
            self.latch_window_m,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in finite_positive):
            raise ValueError("G2 contact-hold force and gap parameters must be finite and positive")
        if self.force_deadband_n >= self.target_force_n:
            raise ValueError("G2 contact-hold deadband must be smaller than its force target")
        if self.contact_confirm_steps < 1 or self.contact_loss_steps < 1:
            raise ValueError("G2 contact-hold confirmation counts must be positive")


G2_CONTACT_HOLD_POLICY = G2ContactHoldPolicy()


class G2ContactHoldController:
    """Batched contact-confirmed gap regulator for a physical G2 drive joint.

    The public command remains a physical two-finger gap.  Before contact it is
    passed through unchanged.  Once both object-facing pads have carried load
    for two consecutive steps, the measured contact gap is latched and a small,
    bounded gap correction regulates the weaker pad around 10 N.  Release always
    has priority and clears the latch immediately.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        initial_gap_m: float,
        policy: G2ContactHoldPolicy = G2_CONTACT_HOLD_POLICY,
    ) -> None:
        if int(num_envs) < 1:
            raise ValueError("G2 contact-hold controller requires at least one environment")
        if not math.isfinite(float(initial_gap_m)) or float(initial_gap_m) < 0.0:
            raise ValueError("G2 contact-hold initial gap must be finite and non-negative")
        self.num_envs = int(num_envs)
        self.policy = policy
        self.device = torch.device(device)
        self.dtype = dtype
        self.latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.contact_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.loss_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.latch_gap_m = torch.full(
            (self.num_envs,),
            float(initial_gap_m),
            dtype=self.dtype,
            device=self.device,
        )
        self.applied_gap_m = self.latch_gap_m.clone()

    def reset(self, envs_idx: Any = None, *, gap_m: float | torch.Tensor | None = None) -> None:
        """Clear selected latches and seed their next applied gap."""

        idx = slice(None) if envs_idx is None else envs_idx
        self.latched[idx] = False
        self.contact_count[idx] = 0
        self.loss_count[idx] = 0
        if gap_m is not None:
            value = torch.as_tensor(gap_m, device=self.device, dtype=self.dtype)
            self.latch_gap_m[idx] = value
            self.applied_gap_m[idx] = value

    def prime(self, envs_idx: Any, gap_m: float | torch.Tensor) -> None:
        """Start selected reset environments already holding at a known gap."""

        value = torch.as_tensor(gap_m, device=self.device, dtype=self.dtype)
        self.latched[envs_idx] = True
        self.contact_count[envs_idx] = self.policy.contact_confirm_steps
        self.loss_count[envs_idx] = 0
        self.latch_gap_m[envs_idx] = value
        self.applied_gap_m[envs_idx] = value

    def update(
        self,
        *,
        requested_gap_m: torch.Tensor,
        measured_gap_m: torch.Tensor,
        left_force_n: torch.Tensor,
        right_force_n: torch.Tensor,
        closing: torch.Tensor,
        release: torch.Tensor,
    ) -> torch.Tensor:
        """Return the next bounded applied-gap target for each environment."""

        requested = self._vector("requested_gap_m", requested_gap_m, self.dtype)
        measured = self._vector("measured_gap_m", measured_gap_m, self.dtype)
        left_force = self._vector("left_force_n", left_force_n, self.dtype)
        right_force = self._vector("right_force_n", right_force_n, self.dtype)
        closing_mask = self._vector("closing", closing, torch.bool)
        release_mask = self._vector("release", release, torch.bool)
        self._require_finite(requested, measured, left_force, right_force)

        # Release has priority over every hold/reacquire path.
        if bool(release_mask.any().item()):
            self.reset(release_mask, gap_m=requested[release_mask])

        bilateral = (left_force >= self.policy.contact_threshold_n) & (right_force >= self.policy.contact_threshold_n)
        confirming = closing_mask & (~release_mask) & (~self.latched) & bilateral
        self.contact_count = torch.where(
            confirming,
            self.contact_count + 1,
            torch.zeros_like(self.contact_count),
        )
        newly_latched = confirming & (self.contact_count >= self.policy.contact_confirm_steps)
        if bool(newly_latched.any().item()):
            self.latched[newly_latched] = True
            self.latch_gap_m[newly_latched] = measured[newly_latched]
            # Preserve the existing closing preload.  Jumping straight from the
            # requested close command to the larger measured contact gap removes
            # position stiffness exactly when the lift begins.
            self.applied_gap_m[newly_latched] = requested[newly_latched]

        free = (~self.latched) & (~release_mask)
        self.applied_gap_m[free] = requested[free]

        holding = self.latched & (~release_mask)
        lost = holding & (~bilateral)
        self.loss_count = torch.where(lost, self.loss_count + 1, torch.zeros_like(self.loss_count))

        weaker_force = torch.minimum(left_force, right_force)
        # Position control has several frames of response delay during a lift.
        # Preserve the prior target inside the deadband so contact spring-back
        # cannot drag the target open, but cap every opening correction both by
        # target slew and by measured-gap lead.  This is the anti-windup boundary:
        # the regulator retains physical preload without accumulating millimetres
        # of unseen opening command while the linkage is still catching up.
        regulated = self.applied_gap_m.clone()
        loaded = holding & bilateral
        underloaded = loaded & (weaker_force < self.policy.target_force_n - self.policy.force_deadband_n)
        overloaded = loaded & (weaker_force > self.policy.target_force_n + self.policy.force_deadband_n)
        regulated[underloaded] = self.applied_gap_m[underloaded] - self.policy.max_gap_step_m
        regulated[overloaded] = torch.minimum(
            self.applied_gap_m[overloaded] + self.policy.max_gap_step_m,
            measured[overloaded] + self.policy.max_gap_step_m,
        )
        reacquire = lost & (self.loss_count >= self.policy.contact_loss_steps)
        regulated[reacquire] = self.applied_gap_m[reacquire] - self.policy.max_gap_step_m

        lower = self.latch_gap_m - self.policy.latch_window_m
        upper = self.latch_gap_m + self.policy.latch_window_m
        regulated = torch.maximum(lower, torch.minimum(upper, regulated))
        self.applied_gap_m[holding] = regulated[holding]
        return self.applied_gap_m.clone()

    def _vector(self, name: str, value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=self.device, dtype=dtype)
        if tensor.shape != (self.num_envs,):
            raise ValueError(f"{name} must have shape ({self.num_envs},), got {tuple(tensor.shape)}")
        return tensor

    @staticmethod
    def _require_finite(*values: torch.Tensor) -> None:
        for value in values:
            if not bool(torch.isfinite(value).all().item()):
                raise RuntimeError("G2 contact-hold input contains NaN or Inf")


def object_finger_contact_forces_n(
    contacts: Mapping[str, Any],
    *,
    left_link_idx: int,
    right_link_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the object-only net contact load carried by each finger link.

    ``RigidEntity.get_contacts`` returns dense ``(B, C, ...)`` tensors for a
    parallel scene and ``(C, ...)`` tensors for one environment.  This helper
    normalizes both forms to one force value per environment, selects the force
    applied to the requested finger regardless of whether it is contact side A
    or B, and vector-sums all of that finger's object contacts before taking the
    magnitude.  Vector summation avoids overstating the pad load when Genesis
    emits several nearby contact points for one physical patch.
    """

    link_a = torch.as_tensor(contacts.get("link_a", torch.empty(0, dtype=torch.int64)))
    link_b = torch.as_tensor(contacts.get("link_b", torch.empty(0, dtype=torch.int64)), device=link_a.device)
    unbatched = link_a.ndim == 1
    if unbatched:
        link_a = link_a.unsqueeze(0)
        link_b = link_b.unsqueeze(0)
    if link_a.ndim != 2 or link_b.shape != link_a.shape:
        raise ValueError("contact link arrays must have shape (contacts,) or (envs, contacts)")

    force_a = torch.as_tensor(contacts.get("force_a", torch.zeros((*link_a.shape, 3))), device=link_a.device)
    if unbatched and force_a.ndim == 2:
        force_a = force_a.unsqueeze(0)
    if force_a.shape != (*link_a.shape, 3):
        raise ValueError("contact force array does not match contact link arrays")
    raw_force_b = contacts.get("force_b")
    force_b = -force_a if raw_force_b is None else torch.as_tensor(raw_force_b, device=link_a.device)
    if unbatched and force_b.ndim == 2:
        force_b = force_b.unsqueeze(0)
    if force_b.shape != force_a.shape:
        raise ValueError("contact force_b array does not match contact force_a")

    raw_valid = contacts.get("valid_mask")
    if raw_valid is None:
        valid = torch.ones_like(link_a, dtype=torch.bool)
    else:
        valid = torch.as_tensor(raw_valid, device=link_a.device, dtype=torch.bool)
        if unbatched and valid.ndim == 1:
            valid = valid.unsqueeze(0)
        if valid.shape != link_a.shape:
            raise ValueError("contact valid_mask does not match contact link arrays")

    def force_for(link_idx: int) -> torch.Tensor:
        mask_a = valid & (link_a == int(link_idx))
        mask_b = valid & (link_b == int(link_idx))
        applied = torch.where(mask_a.unsqueeze(-1), force_a, torch.zeros_like(force_a))
        applied = applied + torch.where(mask_b.unsqueeze(-1), force_b, torch.zeros_like(force_b))
        return torch.linalg.vector_norm(applied.sum(dim=1), dim=-1)

    return force_for(left_link_idx), force_for(right_link_idx)


def validate_g2_contact_substeps(*, dt: float, substeps: int) -> float:
    """Enforce the project's conservative G2 production-step policy.

    A 0.625 ms rigid step is retained as an empirical safety margin for the
    validated production profile.  It is not a claim that every Genesis 1.3.3
    G2 model needs this step size: the mimic solver profile is the primary fix.
    """

    dt = float(dt)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("G2 contact scene dt must be finite and positive")
    if isinstance(substeps, bool) or not isinstance(substeps, Integral):
        raise ValueError("G2 contact scene substeps must be a positive integer")
    normalized_substeps = int(substeps)
    if normalized_substeps < 1:
        raise ValueError("G2 contact scene substeps must be a positive integer")
    substep_dt = dt / normalized_substeps
    if substep_dt > G2_MAX_RIGID_SUBSTEP_DT_S + 1e-15:
        required_substeps = math.ceil(dt / G2_MAX_RIGID_SUBSTEP_DT_S)
        raise ValueError(
            f"The {G2_PHYSICS_PROFILE} project policy requires a rigid substep no longer than "
            f"0.625 ms; dt={dt:g}s with substeps={normalized_substeps} "
            f"gives {substep_dt * 1000.0:g} ms. Use at least {required_substeps} substeps."
        )
    return substep_dt


def configure_g2_mimic_constraints(robot: Any) -> int:
    """Apply and verify the one project-wide policy on all five G2 equalities."""

    sol_params = np.asarray(G2_MIMIC_CONSTRAINT_SOL_PARAMS, dtype=np.float64)
    equalities = {str(equality.name): equality for equality in robot.equalities}
    missing = sorted(G2_MIMIC_EQUALITY_NAMES - equalities.keys())
    if missing:
        raise RuntimeError(f"{G2_PHYSICS_PROFILE} is missing G2 mimic equalities: {', '.join(missing)}")
    for name in sorted(G2_MIMIC_EQUALITY_NAMES):
        equalities[name].set_sol_params(sol_params)
    return len(G2_MIMIC_EQUALITY_NAMES)


__all__ = [
    "G2_CONTACT_HOLD_POLICY",
    "G2ContactHoldController",
    "G2ContactHoldPolicy",
    "G2_MAX_RIGID_SUBSTEP_DT_S",
    "G2_MIMIC_CONSTRAINT_SOL_PARAMS",
    "G2_MIMIC_EQUALITY_NAMES",
    "G2_PHYSICS_PROFILE",
    "configure_g2_mimic_constraints",
    "object_finger_contact_forces_n",
    "validate_g2_contact_substeps",
]
