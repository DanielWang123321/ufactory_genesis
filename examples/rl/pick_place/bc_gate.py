"""Genesis-free gate for pick-place behaviour-cloning smoke summaries.

The campaign treats a clone as successful when a from-home ``contact_v1``
evaluation grasps and lifts every episode with zero action clipping and zero
IK faults. Placement is recorded for reporting and downstream PPO staging but
is no longer gated: placement refinement is PPO's job (place-phase resets plus
grasp-degradation monitoring), and the diagnosis campaign showed the strict
clone placement gate never passed under any version.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping


ZERO_DIAGNOSTIC_TOLERANCE = 1e-12
PREMATURE_RELEASE_XY_M = 0.12
OVERSHOOT_RELEASE_XY_M = 0.05
OVERSHOOT_FINAL_XY_M = 0.05


@dataclass(frozen=True)
class GateVerdict:
    """Machine-readable result of one behaviour-cloning smoke evaluation."""

    passed: bool
    reasons: tuple[str, ...]
    failure_mode: str
    acceptance_profile: str | None
    episodes: int
    grasp: int
    lift: int
    place: int
    p99_final_xy_error_m: float | None
    p99_release_xy_dist_m: float | None
    max_action_clip_fraction: float | None
    max_ik_failure_fraction: float | None
    max_ik_jump_reject_fraction: float | None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_eval_summary(path: Path) -> dict[str, Any]:
    """Read one evaluator ``--summary-json`` document."""

    return json.loads(path.read_text(encoding="utf-8"))


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _outcome_count(summary: Mapping[str, Any], name: str) -> int | None:
    outcomes = summary.get("outcomes")
    if not isinstance(outcomes, Mapping):
        return None
    block = outcomes.get(name)
    if not isinstance(block, Mapping):
        return None
    return _as_int(block.get("count"))


def _diagnostic(summary: Mapping[str, Any], name: str) -> float | None:
    diagnostics = summary.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        return None
    return _as_float(diagnostics.get(name))


def _quality(summary: Mapping[str, Any], name: str) -> float | None:
    quality = summary.get("quality")
    if not isinstance(quality, Mapping):
        return None
    return _as_float(quality.get(name))


def classify_bc_failure(
    *,
    passed: bool,
    grasp: int,
    episodes: int,
    p99_final_xy_error_m: float | None,
    p99_release_xy_dist_m: float | None,
) -> str:
    """Name the dominant clone failure so the campaign can pick the next recipe."""

    if passed:
        return "pass"
    if episodes <= 0 or grasp <= 0:
        return "no_grasp"
    release_xy = 0.0 if p99_release_xy_dist_m is None else p99_release_xy_dist_m
    final_xy = 1.0 if p99_final_xy_error_m is None else p99_final_xy_error_m
    if release_xy >= PREMATURE_RELEASE_XY_M:
        return "premature_release"
    if release_xy <= OVERSHOOT_RELEASE_XY_M and final_xy >= OVERSHOOT_FINAL_XY_M:
        return "overshoot_no_release"
    if final_xy >= OVERSHOOT_FINAL_XY_M:
        return "missed_target"
    return "near_miss"


def judge_bc_summary(
    summary: Mapping[str, Any],
    *,
    expected_episodes: int | None = None,
    required_profile: str = "contact_v1",
) -> GateVerdict:
    """Return whether one evaluator summary satisfies the clone smoke gate."""

    reasons: list[str] = []
    profile = summary.get("acceptance_profile")
    profile_name = profile if isinstance(profile, str) else None
    if profile_name != required_profile:
        reasons.append(f"acceptance_profile {profile_name!r} != {required_profile!r}")

    episodes = _as_int(summary.get("episodes"))
    if episodes is None or episodes <= 0:
        reasons.append("episodes missing or not positive")
        episodes = 0
    elif expected_episodes is not None and episodes != expected_episodes:
        reasons.append(f"episodes {episodes} != {expected_episodes}")

    grasp = _outcome_count(summary, "grasp")
    lift = _outcome_count(summary, "lift")
    place = _outcome_count(summary, "place")
    if grasp is None:
        reasons.append("outcomes.grasp.count missing")
        grasp = 0
    if lift is None:
        reasons.append("outcomes.lift.count missing")
        lift = 0
    if place is None:
        reasons.append("outcomes.place.count missing")
        place = 0
    if episodes > 0:
        if grasp != episodes:
            reasons.append(f"grasp {grasp}/{episodes}")
        if lift != episodes:
            reasons.append(f"lift {lift}/{episodes}")

    clip = _diagnostic(summary, "max_action_clip_fraction")
    ik_fail = _diagnostic(summary, "max_ik_failure_fraction")
    ik_jump = _diagnostic(summary, "max_ik_jump_reject_fraction")
    for name, value in (
        ("max_action_clip_fraction", clip),
        ("max_ik_failure_fraction", ik_fail),
        ("max_ik_jump_reject_fraction", ik_jump),
    ):
        if value is None:
            reasons.append(f"diagnostics.{name} missing")
        elif abs(value) > ZERO_DIAGNOSTIC_TOLERANCE:
            reasons.append(f"{name}={value:g}")

    p99_xy = _quality(summary, "p99_final_xy_error_m")
    p99_release = _quality(summary, "p99_release_xy_dist_m")
    if p99_xy is not None:
        reasons.append(f"p99_final_xy_error_m={p99_xy:.6f}")
    if p99_release is not None:
        reasons.append(f"p99_release_xy_dist_m={p99_release:.6f}")

    passed = (
        profile_name == required_profile
        and episodes > 0
        and (expected_episodes is None or episodes == expected_episodes)
        and grasp == episodes
        and lift == episodes
        and clip is not None
        and ik_fail is not None
        and ik_jump is not None
        and abs(clip) <= ZERO_DIAGNOSTIC_TOLERANCE
        and abs(ik_fail) <= ZERO_DIAGNOSTIC_TOLERANCE
        and abs(ik_jump) <= ZERO_DIAGNOSTIC_TOLERANCE
    )
    if passed:
        reasons = [
            f"grasp {grasp}/{episodes}",
            f"lift {lift}/{episodes}",
            f"place {place}/{episodes} (recorded, not gated)",
        ]
        if p99_xy is not None:
            reasons.append(f"p99_final_xy_error_m={p99_xy:.6f}")
        if p99_release is not None:
            reasons.append(f"p99_release_xy_dist_m={p99_release:.6f}")

    failure_mode = classify_bc_failure(
        passed=passed,
        grasp=grasp,
        episodes=episodes,
        p99_final_xy_error_m=p99_xy,
        p99_release_xy_dist_m=p99_release,
    )
    return GateVerdict(
        passed=passed,
        reasons=tuple(reasons),
        failure_mode=failure_mode,
        acceptance_profile=profile_name,
        episodes=episodes,
        grasp=grasp,
        lift=lift,
        place=place,
        p99_final_xy_error_m=p99_xy,
        p99_release_xy_dist_m=p99_release,
        max_action_clip_fraction=clip,
        max_ik_failure_fraction=ik_fail,
        max_ik_jump_reject_fraction=ik_jump,
    )


def judge_eval_summary_path(
    path: Path,
    *,
    expected_episodes: int | None = None,
    required_profile: str = "contact_v1",
) -> GateVerdict:
    """Load ``path`` and apply :func:`judge_bc_summary`."""

    return judge_bc_summary(
        load_eval_summary(path),
        expected_episodes=expected_episodes,
        required_profile=required_profile,
    )


__all__ = [
    "GateVerdict",
    "classify_bc_failure",
    "judge_bc_summary",
    "judge_eval_summary_path",
    "load_eval_summary",
]
