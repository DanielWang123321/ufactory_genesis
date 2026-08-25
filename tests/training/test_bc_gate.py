"""CPU-only tests for the behaviour-cloning smoke gate."""

from __future__ import annotations

import json
from pathlib import Path

from examples.rl.pick_place.bc_gate import classify_bc_failure, judge_bc_summary, judge_eval_summary_path


def _summary(
    *,
    profile: str | None = "contact_v1",
    episodes: int = 8,
    grasp: int = 8,
    lift: int = 8,
    place: int = 8,
    clip: float = 0.0,
    ik_fail: float = 0.0,
    ik_jump: float = 0.0,
    p99_xy: float = 0.002,
    p99_release: float = 0.003,
) -> dict:
    payload = {
        "acceptance_profile": profile,
        "episodes": episodes,
        "outcomes": {
            "grasp": {"count": grasp, "rate": grasp / episodes if episodes else 0.0},
            "lift": {"count": lift, "rate": lift / episodes if episodes else 0.0},
            "place": {"count": place, "rate": place / episodes if episodes else 0.0},
        },
        "quality": {
            "p99_final_xy_error_m": p99_xy,
            "p99_release_xy_dist_m": p99_release,
        },
        "diagnostics": {
            "max_action_clip_fraction": clip,
            "max_ik_failure_fraction": ik_fail,
            "max_ik_jump_reject_fraction": ik_jump,
        },
    }
    return payload


def test_gate_passes_contact_v1_full_counts_and_zero_diagnostics() -> None:
    verdict = judge_bc_summary(_summary(), expected_episodes=8)
    assert verdict.passed
    assert verdict.failure_mode == "pass"
    assert verdict.grasp == verdict.lift == verdict.place == 8
    assert verdict.reasons[0] == "grasp 8/8"


def test_gate_records_place_misses_without_failing() -> None:
    missed = judge_bc_summary(_summary(place=0, p99_xy=0.198, p99_release=0.285), expected_episodes=8)
    assert missed.passed
    assert missed.failure_mode == "pass"
    assert "place 0/8 (recorded, not gated)" in missed.reasons
    assert any("p99_release_xy_dist_m=0.285000" in reason for reason in missed.reasons)
    assert (
        classify_bc_failure(
            passed=False,
            grasp=8,
            episodes=8,
            p99_final_xy_error_m=0.198,
            p99_release_xy_dist_m=0.285,
        )
        == "premature_release"
    )

    wrong_profile = judge_bc_summary(_summary(profile=None), expected_episodes=8)
    assert not wrong_profile.passed
    assert any("acceptance_profile" in reason for reason in wrong_profile.reasons)


def test_gate_rejects_clip_or_ik_faults_even_when_counts_match() -> None:
    clipped = judge_bc_summary(_summary(clip=0.01), expected_episodes=8)
    assert not clipped.passed
    assert any("max_action_clip_fraction" in reason for reason in clipped.reasons)

    jumped = judge_bc_summary(_summary(ik_jump=0.23), expected_episodes=8)
    assert not jumped.passed
    assert jumped.failure_mode == "near_miss"


def test_gate_classifies_overshoot_and_no_grasp() -> None:
    overshoot = classify_bc_failure(
        passed=False,
        grasp=8,
        episodes=8,
        p99_final_xy_error_m=0.208,
        p99_release_xy_dist_m=0.0,
    )
    assert overshoot == "overshoot_no_release"
    empty = judge_bc_summary(_summary(grasp=0, lift=0, place=0, p99_xy=0.30), expected_episodes=8)
    assert not empty.passed
    assert empty.failure_mode == "no_grasp"
    assert (
        classify_bc_failure(
            passed=False,
            grasp=8,
            episodes=8,
            p99_final_xy_error_m=0.04,
            p99_release_xy_dist_m=0.08,
        )
        == "near_miss"
    )


def test_gate_reads_summary_json_from_disk(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(_summary(place=0, p99_xy=0.211, p99_release=0.0)), encoding="utf-8")
    verdict = judge_eval_summary_path(path, expected_episodes=8)
    assert verdict.passed
    assert verdict.place == 0
    assert verdict.failure_mode == "pass"
