"""G2 physical stability policy: rigid substep guard, mimic equalities, pad contact forces."""

from __future__ import annotations

import math
from numbers import Integral
from typing import Any, Mapping

import numpy as np
import torch


from ufactory.config.models import G2_PHYSICS_PROFILE

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
    "G2_MAX_RIGID_SUBSTEP_DT_S",
    "G2_MIMIC_CONSTRAINT_SOL_PARAMS",
    "G2_MIMIC_EQUALITY_NAMES",
    "G2_PHYSICS_PROFILE",
    "configure_g2_mimic_constraints",
    "object_finger_contact_forces_n",
    "validate_g2_contact_substeps",
]
