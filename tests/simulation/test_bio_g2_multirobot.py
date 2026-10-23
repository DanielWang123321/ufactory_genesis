"""Common flange poses, complete mounting transforms, and shared task targets."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from ufactory.trajectory import bio_g2_replay as replay
from ufactory.trajectory.bio_g2_frames import (
    BioG2TaskFrames,
    HOME_FLANGE_XYZ_M,
    SUPPORTED_ROBOTS,
    pose_components,
    pose_matrix,
    quaternion_slerp,
    rotation_error_rad,
)


@pytest.fixture
def cli_module():
    pytest.importorskip("torch")
    pytest.importorskip("genesis")
    from ufactory.cli import pick_place

    return pick_place


@pytest.mark.parametrize("robot", SUPPORTED_ROBOTS)
def test_common_flange_home_and_complete_tcp_roundtrip(robot):
    frames = BioG2TaskFrames.load(robot)
    home = frames.home_flange()
    np.testing.assert_array_equal(home[:3, 3], HOME_FLANGE_XYZ_M)
    np.testing.assert_allclose(replay.pose_report(home, "robot_base")["rpy_deg"], [180, 0, -0.404582])
    # An arbitrary tilted pose exercises the rotated translation as well as the mounting yaw.
    q = np.array([0.4, 0.3, -0.2, 0.1])
    flange = pose_matrix([0.3, 0.1, 0.5], q)
    tcp = frames.flange_to_tcp_pose(flange)
    np.testing.assert_allclose(frames.tcp_to_flange(tcp), flange, atol=2e-16)
    np.testing.assert_allclose(tcp[:3, 3], flange[:3, 3] + flange[:3, :3] @ frames.flange_to_tcp[:3, 3])
    assert len(frames.ik_seed) == frames.runtime.model.dof
    assert replay.arm_joint_limits(robot).shape == (frames.runtime.model.dof, 3)
    if robot == "xarm7":
        np.testing.assert_array_equal(frames.ik_seed, [0, 0, 0, 0.5, -np.pi, -0.5, np.pi])


def test_four_models_share_every_public_task_target_and_command():
    reference = None
    for robot in SUPPORTED_ROBOTS:
        ticks = replay.sample_task_trajectory(BioG2TaskFrames.load(robot), replay.default_phases())
        public = [t for t in ticks if t["phase"] not in {"enter_task", "return_flange", "home_settle"}]
        if reference is None:
            reference = public
        for first, second in zip(reference, public, strict=True):
            for key in first:
                np.testing.assert_array_equal(first[key], second[key], err_msg=f"{robot}: {key}")


@pytest.mark.parametrize("robot", SUPPORTED_ROBOTS)
def test_transitions_use_quintic_positions_and_shortest_slerp_before_final_settle(robot):
    frames = BioG2TaskFrames.load(robot)
    ticks = replay.sample_task_trajectory(frames, replay.default_phases())
    home_xyz, home_quat = pose_components(frames.flange_to_tcp_pose(frames.home_flange(replay.ROBOT_BASE_POS)))
    task_xyz, task_quat = np.asarray(replay.TASK_START_TCP_WORLD_M), replay.DOWN_QUAT_WXYZ
    for name, start_xyz, end_xyz, start_quat, end_quat in (
        ("enter_task", home_xyz, task_xyz, home_quat, task_quat),
        ("return_flange", task_xyz, home_xyz, task_quat, home_quat),
    ):
        transition = [t for t in ticks if t["phase"] == name]
        assert len(transition) * replay.DT == pytest.approx(0.4)
        for idx, tick in enumerate(transition):
            u = (idx + 1) / len(transition)
            s = 6 * u**5 - 15 * u**4 + 10 * u**3
            np.testing.assert_allclose(tick["target_grasp_xyz"], start_xyz + s * (end_xyz - start_xyz), atol=1e-15)
            assert rotation_error_rad(start_quat, tick["target_grasp_quat_wxyz"]) == pytest.approx(
                s * rotation_error_rad(start_quat, end_quat), abs=5e-8
            )
            assert np.linalg.norm(tick["target_grasp_quat_wxyz"]) == pytest.approx(1)
            assert tick["gripper_target_mm"] is None
        np.testing.assert_allclose(transition[-1]["target_grasp_xyz"], end_xyz, atol=1e-16)
        assert rotation_error_rad(transition[-1]["target_grasp_quat_wxyz"], end_quat) < 5e-8
    assert list(dict.fromkeys(t["phase"] for t in ticks))[-2:] == ["return_flange", "home_settle"]
    for tick in (t for t in ticks if t["phase"] == "home_settle"):
        flange = frames.tcp_to_flange(pose_matrix(tick["target_grasp_xyz"], tick["target_grasp_quat_wxyz"]))
        np.testing.assert_allclose(flange, frames.home_flange(replay.ROBOT_BASE_POS), atol=2e-16)


def test_slerp_handles_antipodal_and_half_turn_quaternions():
    np.testing.assert_array_equal(quaternion_slerp([1, 0, 0, 0], [-1, 0, 0, 0], 0.5), [1, 0, 0, 0])
    q = quaternion_slerp([1, 0, 0, 0], [0, 0, 0, 1], 0.5)
    np.testing.assert_allclose(q, [2**-0.5, 0, 0, 2**-0.5])


@pytest.mark.parametrize("robot", SUPPORTED_ROBOTS)
def test_bio_cli_accepts_four_robots_and_defaults_to_cartesian(monkeypatch, robot, cli_module):
    pick_place = cli_module
    calls = []
    monkeypatch.setattr(
        replay, "run_bio_g2_pick_place", lambda **kw: calls.append(kw) or {"passed": True, "metrics": {}}
    )
    assert pick_place.main(["--robot", robot, "--gripper", "bio_gripper_g2", "--mode", "sim"]) == 0
    assert calls[0]["robot"] == robot
    assert calls[0]["executor"] == "servo_cartesian"


def test_other_grippers_still_require_executor_and_lite6_rejects_bio(cli_module):
    pick_place = cli_module
    for argv in (
        ["--robot", "xarm6", "--mode", "sim"],
        ["--robot", "lite6", "--mode", "sim", "--gripper", "bio_gripper_g2"],
    ):
        with pytest.raises(SystemExit) as exc:
            pick_place.main(argv)
        assert exc.value.code == 2
    with pytest.raises(ValueError, match="robot"):
        BioG2TaskFrames.load("lite6")


@pytest.mark.parametrize("position,rotation", [(0.5001, 0), (0, 0.001001)])
def test_initial_pose_check_prevents_viewer_opening(position, rotation):
    sim = object.__new__(replay.BioG2PickPlaceSimulation)
    sim.executor = "servo_cartesian"
    sim.frames = BioG2TaskFrames.load()
    sim.transport_speed_mm_s = 250
    sim._solve_home = lambda: setattr(sim, "home_arm_q", np.zeros(6))
    sim.reset = lambda *a, **kw: None
    sim.measured_pose = lambda: {"flange_position_error_mm": position, "flange_rotation_error_rad": rotation}
    sim.start_viewer = lambda: pytest.fail("viewer opened outside flange home tolerance")
    with pytest.raises(replay.HomePoseError):
        sim.prepare(replay.default_phases())


@pytest.mark.parametrize("field", ["target_grasp_xyz", "target_grasp_quat_wxyz"])
def test_invalid_pose_still_saves_finite_json_and_separate_joint_artifact(tmp_path, field):
    frames = BioG2TaskFrames.load()
    tick = replay.sample_task_trajectory(frames, replay.default_phases())[0]
    tick[field][0] = np.nan
    sim = SimpleNamespace(
        frames=frames,
        runtime=frames.runtime,
        arm_joint_names=frames.runtime.arm.joint_names,
        executor="servo_cartesian",
        cartesian_targets=[tick],
        joint_targets=[],
        joint_rows=[],
    )
    files = replay.save_trajectory_artifacts(sim, tmp_path / "failure.json")

    def read(path):
        return json.loads(path.read_text(), parse_constant=lambda value: pytest.fail(f"Invalid JSON: {value}"))

    from pathlib import Path

    targets, joints = read(Path(files["cartesian_targets"])), read(Path(files["joint_targets"]))
    assert "invalid_pose" in targets["samples"][0]["tcp"]
    assert targets["samples"][0]["flange"] is None
    assert joints["compiled"] == joints["executed"] == []


def test_collision_audit_expands_both_fingers_and_detects_table_penetration():
    pytest.importorskip("pinocchio")
    pytest.importorskip("coal")
    from ufactory.simulation.bio_g2_collision import audit_collisions

    frames = BioG2TaskFrames.load()
    result = audit_collisions(frames, {"initial_arm_q_rad": [0, -0.2, 0.04, 0, 0.65, 0], "executed": []})
    exhaustive = audit_collisions(
        frames, {"initial_arm_q_rad": [0, -0.2, 0.04, 0, 0.65, 0], "executed": []}, exhaustive=True
    )
    assert result["min_distance_mm"] == exhaustive["min_distance_mm"]
    assert result["collision_count"] == exhaustive["collision_count"]
    assert result["mimic_expanded"]
    assert result["sample_count"] == 1
    assert not result["passed"]
    assert any(c["category"] == "table" for c in result["collisions"])
