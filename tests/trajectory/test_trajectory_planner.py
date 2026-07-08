"""Robot-aware trajectory planner tests."""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import numpy as np
import pytest

from ufactory.robots.registry import ROBOT_PROFILES
from ufactory.robots.runtime import get_robot_runtime_profile
from ufactory.trajectory import (
    CartesianWaypoint,
    JointWaypoint,
    OptionalTrajectoryDependencyError,
    TrajectoryPlannerConfig,
    plan_cartesian_waypoints,
    plan_joint_waypoints,
    plan_mixed_waypoints,
    require_roboticstoolbox,
    validate_program,
)


def test_planner_import_does_not_load_heavy_modules():
    code = (
        "import sys; "
        "from ufactory.trajectory import TrajectoryPlannerConfig, plan_joint_waypoints; "
        "print('genesis' in sys.modules, 'roboticstoolbox' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False False"


@pytest.mark.parametrize("robot_key", sorted(ROBOT_PROFILES))
def test_plan_joint_waypoints_supports_all_robot_dofs(robot_key: str):
    runtime = get_robot_runtime_profile(robot_key)
    dof = runtime.model.dof
    start = np.zeros(dof)
    waypoint = np.asarray(runtime.arm.default_qpos, dtype=np.float64)
    config = TrajectoryPlannerConfig(robot_key=robot_key, rate=50.0, speed_rad_s=0.35, mvacc_rad_s2=2.0)

    program = plan_joint_waypoints(config, start, [JointWaypoint(waypoint, label="default")])

    assert program.robot_key == runtime.model.key
    assert program.metadata["ee_link"] == runtime.arm.ee_link
    assert len(program.segments) == 1
    seg = program.segments[0]
    assert seg.kind == "movej"
    assert seg.q_start.shape == (dof,)
    assert seg.q_end.shape == (dof,)
    np.testing.assert_allclose(seg.q_start, start)
    np.testing.assert_allclose(seg.q_end, waypoint)
    validate_program(program)


def test_plan_joint_waypoints_rejects_wrong_dof_for_xarm5():
    config = TrajectoryPlannerConfig(robot_key="xarm5")

    with pytest.raises(ValueError, match="expected 5 joints"):
        plan_joint_waypoints(config, np.zeros(6), [np.zeros(6)])


def test_plan_cartesian_waypoints_generates_chained_movel_segments():
    config = TrajectoryPlannerConfig(robot_key="uf850", rate=20.0, speed_rad_s=0.5, mvacc_rad_s2=1.0)
    start = [0.30, 0.00, 0.30]
    wp1 = CartesianWaypoint([0.35, 0.00, 0.30], label="x")
    wp2 = CartesianWaypoint([0.35, 0.10, 0.25], label="y-down")

    program = plan_cartesian_waypoints(config, start, [wp1, wp2])

    assert [seg.label for seg in program.segments] == ["x", "y-down"]
    np.testing.assert_allclose(program.segments[0].pose_start, start)
    np.testing.assert_allclose(program.segments[0].pose_end, wp1.xyz)
    np.testing.assert_allclose(program.segments[1].pose_start, wp1.xyz)
    np.testing.assert_allclose(program.segments[1].pose_end, wp2.xyz)
    assert program.total_ticks == sum(seg.samples_count for seg in program.segments)
    validate_program(program)


def test_plan_cartesian_waypoints_checks_z_min():
    config = TrajectoryPlannerConfig(robot_key="xarm6", z_min_m=0.0)

    with pytest.raises(ValueError, match="below 0.0000 m"):
        plan_cartesian_waypoints(config, [0.30, 0.0, 0.10], [[0.30, 0.0, -0.01]])


def test_plan_mixed_waypoints_chains_typed_and_dict_waypoints():
    config = TrajectoryPlannerConfig(robot_key="xarm6", rate=50.0)
    start_q = np.zeros(6)
    start_xyz = [0.30, 0.0, 0.30]

    program = plan_mixed_waypoints(
        config,
        [
            JointWaypoint([0.1, -0.1, 0.0, 0.0, 0.1, 0.0], label="joint"),
            CartesianWaypoint([0.30, 0.10, 0.30], label="line"),
            {"type": "gripper", "gap_start": 0.084, "gap_end": 0.024, "duration": 1.0, "label": "grip"},
            {"type": "movel", "xyz": [0.30, 0.20, 0.30], "label": "line2"},
        ],
        start_q=start_q,
        start_xyz=start_xyz,
    )

    assert [seg.kind for seg in program.segments] == ["movej", "movel", "gripper", "movel"]
    assert [seg.label for seg in program.segments] == ["joint", "line", "grip", "line2"]
    np.testing.assert_allclose(program.segments[-1].pose_start, [0.30, 0.10, 0.30])
    np.testing.assert_allclose(program.segments[-1].pose_end, [0.30, 0.20, 0.30])
    validate_program(program)


def test_optional_roboticstoolbox_backend_has_clear_install_hint():
    if importlib.util.find_spec("roboticstoolbox") is not None:
        pytest.skip("roboticstoolbox installed in this environment")

    with pytest.raises(OptionalTrajectoryDependencyError, match=r'\.\[trajectory\]'):
        require_roboticstoolbox()
