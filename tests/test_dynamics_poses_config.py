"""Tests for YAML-backed dynamics validation pose loading."""

from __future__ import annotations

import numpy as np

from ufactory.dynamics.poses_config import (
    default_configs_named_tuple,
    dynamics_pose_tuples,
    expand_round_trip,
    load_dynamics_pose_lists,
)


def test_all_robots_have_twenty_expanded_poses():
    for robot_key in ("xarm5", "xarm6", "xarm7", "lite6", "uf850"):
        configs = dynamics_pose_tuples(robot_key)
        assert len(configs) == 20
        assert [name for name, _ in configs] == [str(i) for i in range(20)]


def test_xarm6_include_stress_has_no_extra_poses():
    configs = dynamics_pose_tuples("xarm6", include_stress=True)
    assert len(configs) == 20
    assert configs[-1][0] == "19"


def test_expand_round_trip_xarm6_first_leg():
    home = [0, 0, 0, 0, 0, 0]
    end = [90, -90, -60, 170, 160, -179]
    leg = expand_round_trip(home, end)
    assert len(leg) == 10
    np.testing.assert_allclose(leg[0], np.deg2rad(np.asarray(home) + 0.2 * (np.asarray(end) - np.asarray(home))))
    np.testing.assert_allclose(leg[4], np.deg2rad(end))
    np.testing.assert_allclose(leg[9], np.zeros(6))


def test_xarm6_second_leg_starts_after_first():
    default, _ = load_dynamics_pose_lists("xarm6")
    assert len(default) == 20
    np.testing.assert_allclose(default[9], np.zeros(6))
    np.testing.assert_allclose(
        default[10],
        np.deg2rad(np.asarray([-90, -90, -60, -170, -20, 179]) * 0.2),
    )


def test_default_configs_named_tuple_matches_loader():
    tuples = default_configs_named_tuple("lite6")
    assert len(tuples) == 20
    assert tuples[0][0] == "0"
