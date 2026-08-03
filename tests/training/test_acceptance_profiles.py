"""Tests for immutable pick-place evaluation profiles."""

from __future__ import annotations

import pytest

from ufactory.training import apply_pick_place_acceptance_profile


def test_contact_v1_overrides_candidate_quality_definition() -> None:
    env = {
        "ctrl_dt": 0.02,
        "place_success_dist_m": 0.2,
        "landing_near_table_height_m": 0.035,
        "landing_max_xy_speed_m_s": 1.0,
    }
    resolved = apply_pick_place_acceptance_profile(env, "contact_v1")
    assert env["place_success_dist_m"] == pytest.approx(0.010)
    assert env["release_success_dist_m"] == pytest.approx(0.010)
    assert env["landing_near_table_height_m"] == pytest.approx(0.020)
    assert env["landing_max_xy_speed_m_s"] == pytest.approx(0.030)
    assert env["success_hold_steps"] == 25
    assert resolved["post_release_max_drift_m"] == pytest.approx(0.003)


def test_acceptance_profile_rejects_unknown_name_and_invalid_period() -> None:
    with pytest.raises(ValueError, match="unknown"):
        apply_pick_place_acceptance_profile({"ctrl_dt": 0.02}, "future")
    with pytest.raises(ValueError, match="ctrl_dt"):
        apply_pick_place_acceptance_profile({"ctrl_dt": 0.0}, "contact_v1")
