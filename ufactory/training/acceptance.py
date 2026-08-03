"""Immutable pick-place evaluation profiles.

Training recipes are free to change reward shaping, reset mixtures, and policy
observations.  Evaluation quality limits are not: loading them from a candidate's
training artifact makes the candidate redefine the test it is being compared on.
"""

from __future__ import annotations

import math
from collections.abc import MutableMapping
from typing import Any


PICK_PLACE_ACCEPTANCE_PROFILES = ("contact_v1",)


def apply_pick_place_acceptance_profile(
    env_cfg: MutableMapping[str, Any],
    profile: str,
) -> dict[str, Any]:
    """Apply one named, versioned quality profile to an evaluation config.

    The returned mapping is the fully resolved profile and is suitable for reports.
    ``success_hold_steps`` is derived from the environment control period so the
    required stable duration remains 0.5 seconds.
    """

    if profile not in PICK_PLACE_ACCEPTANCE_PROFILES:
        raise ValueError(
            f"unknown pick-place acceptance profile {profile!r}; expected one of {PICK_PLACE_ACCEPTANCE_PROFILES}"
        )
    ctrl_dt = float(env_cfg.get("ctrl_dt", 0.0))
    if not math.isfinite(ctrl_dt) or ctrl_dt <= 0.0:
        raise ValueError("evaluation ctrl_dt must be finite and positive")

    if profile == "contact_v1":
        resolved: dict[str, Any] = {
            "place_success_dist_m": 0.010,
            "release_success_dist_m": 0.010,
            "success_hold_steps": max(1, round(0.5 / ctrl_dt)),
            "success_table_z_tolerance_m": 0.005,
            "success_max_obj_speed_m_s": 0.020,
            "release_height_tolerance_m": 0.005,
            "release_max_obj_speed_m_s": 0.020,
            "pre_lift_max_drag_m": 0.005,
            "post_release_max_drift_m": 0.003,
            # This height defines the hard-landing event and therefore belongs to
            # the profile.  Reward shaping uses landing_speed_margin_height_m.
            "landing_near_table_height_m": 0.020,
            "landing_max_xy_speed_m_s": 0.030,
            "landing_max_down_speed_m_s": 0.050,
        }
    else:  # pragma: no cover - the profile validation above is exhaustive.
        raise AssertionError(profile)

    env_cfg.update(resolved)
    return resolved


__all__ = [
    "PICK_PLACE_ACCEPTANCE_PROFILES",
    "apply_pick_place_acceptance_profile",
]
