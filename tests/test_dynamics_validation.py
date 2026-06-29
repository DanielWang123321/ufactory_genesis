"""Unit tests for enterprise dynamics validation helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from ufactory.dynamics_validation import (
    ABS_ERR_LIMITS,
    GenesisDynamicsSample,
    SafePose,
    ValidationStatus,
    build_dynamics_sample,
    classify_torque_result,
    compare_report_records,
    read_report_records,
    run_sim_collision_chain,
    validate_urdf_dynamics,
    write_csv_report,
    write_jsonl_report,
    DynamicsRunConfig,
    dynamics_default_configs,
    xarm6_default_dynamics_configs,
)
from ufactory.kinematics import build_calibrated_urdf
from ufactory.paths import xarm6_1305_urdf
from ufactory.robot_params import DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S
from ufactory.real_robot_session import RealRobotSession


def test_xarm6_urdf_dynamics_static_checks_have_no_errors():
    issues = validate_urdf_dynamics(xarm6_1305_urdf())
    assert [issue for issue in issues if issue.severity == "ERROR"] == []


def test_default_pose_set_is_enterprise_scale_and_excludes_stress_pose():
    configs = xarm6_default_dynamics_configs()
    names = {name for name, _ in configs}
    assert len(configs) == 21
    assert "config_H" not in names
    assert "home" in names
    assert "calib_002" in names
    assert "calib_034" in names
    assert "calib_001" not in names
    assert "config_H" in {name for name, _ in xarm6_default_dynamics_configs(include_stress=True)}


def test_torque_classification_distinguishes_bias_like_single_joint_failure():
    abs_err = np.array([0.1, 0.2, 0.1, 0.1, 0.1, ABS_ERR_LIMITS[5] + 0.05])
    rel_err = abs_err / np.array([50.0, 50.0, 32.0, 32.0, 32.0, 20.0])
    status = classify_torque_result(
        settled=True,
        saturated=False,
        tau_real=np.zeros(6),
        abs_err=abs_err,
        rel_err=rel_err,
        l2_err=float(np.linalg.norm(abs_err)),
    )
    assert status == ValidationStatus.FAIL_BIAS


def test_report_jsonl_and_csv_roundtrip(tmp_path: Path):
    pose = SafePose("home", np.zeros(6), 108.0)
    genesis = GenesisDynamicsSample(
        q_actual=np.zeros(6),
        qvel=np.zeros(6),
        pd_hold_tau=np.ones(6),
        actual_dof_force=np.zeros(6),
        mass_matrix=np.eye(6),
        settled=True,
        saturated=False,
        pos_err=0.0,
        vel_mag=0.0,
    )
    sample = build_dynamics_sample(pose, genesis, tau_real=np.ones(6), n_real_samples=3)
    run_config = DynamicsRunConfig(robot_key="xarm6_1305", urdf_path=xarm6_1305_urdf(), mode="unit")

    jsonl = tmp_path / "report.jsonl"
    csv = tmp_path / "report.csv"
    write_jsonl_report([sample], jsonl, run_config=run_config)
    write_csv_report([sample], csv)

    assert read_report_records(jsonl)[0]["pose"] == "home"
    assert read_report_records(csv)[0]["pose"] == "home"


def test_compare_report_records_uses_signed_residuals():
    old = [{"signed_err": [1, 2, 3, 4, 5, 6]}]
    new = [{"signed_err": [2, 2, 3, 4, 5, 9]}]
    stats = compare_report_records(old, new)
    assert stats[0]["bias_delta"] == 1
    assert stats[5]["new_bias"] == 9


def test_report_compare_supports_seven_dof():
    old = [{"signed_err": [1, 2, 3, 4, 5, 6, 7]}]
    new = [{"signed_err": [1, 2, 3, 4, 5, 6, 9]}]
    stats = compare_report_records(old, new)
    assert len(stats) == 7
    assert stats[6]["bias_delta"] == 2


def test_non_xarm6_dynamics_configs_match_dof():
    configs = dynamics_default_configs("xarm7")
    assert configs
    assert {len(q) for _, q in configs} == {7}


def test_calibrated_urdf_defaults_to_cache(tmp_path: Path):
    kinematics = {
        f"joint{i}": {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        for i in range(1, 7)
    }
    out = Path(build_calibrated_urdf(xarm6_1305_urdf(), kinematics, suffix="unit", output_dir=str(tmp_path)))
    assert out.parent == tmp_path
    assert out.name.endswith("_calib.urdf")
    assert out.exists()


def test_real_robot_collect_hold_samples_uses_statistics():
    session = object.__new__(RealRobotSession)
    arm = MagicMock()
    arm.get_joint_states.side_effect = [
        (0, [np.zeros(7).tolist(), np.zeros(7).tolist(), np.ones(7).tolist()]),
        (0, [np.zeros(7).tolist(), np.zeros(7).tolist(), (np.ones(7) * 3).tolist()]),
    ]
    arm.get_joints_torque.side_effect = [
        (0, np.ones(7).tolist()),
        (0, (np.ones(7) * 3).tolist()),
    ]
    session.arm = arm
    session.dof = 6
    session.home_qpos = np.zeros(6)

    sample = session.collect_hold_samples(duration_s=0.0)
    assert sample.n_samples == 1
    assert np.allclose(sample.tau, np.ones(6))
    assert np.allclose(sample.tau_direct, np.ones(6))


def test_run_sim_collision_chain_moves_through_poses_in_order(monkeypatch):
    monkeypatch.setattr("ufactory.xarm_control.time.sleep", lambda _: None)
    arm = MagicMock()
    arm.error_code = 0
    session = object.__new__(RealRobotSession)
    session.arm = arm
    session.dof = 6
    session.home_qpos = np.zeros(6)

    calls: list[tuple[str, np.ndarray]] = []

    def move_to(q, *, speed_rad_s, wait, move_strategy):
        del speed_rad_s, wait, move_strategy
        calls.append(("move", np.asarray(q, dtype=np.float64)))
        return 0

    session.move_to = move_to
    session.recover_after_motion_error = lambda **kwargs: True

    poses = [
        ("home", np.zeros(6)),
        ("pose_a", np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ("pose_b", np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])),
    ]
    results = run_sim_collision_chain(
        session,
        poses,
        speed_rad_s=DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S,
        move_strategy="direct",
    )

    assert len(results) == 3
    assert all(r.passed for r in results)
    assert [name for name, _ in poses] == [r.pose_name for r in results]
    assert len(calls) == 3
    assert np.allclose(calls[0][1], np.zeros(6))
    assert np.allclose(calls[1][1], poses[1][1])
    assert np.allclose(calls[2][1], poses[2][1])


def test_run_sim_collision_chain_records_failure_and_recovers(monkeypatch):
    from ufactory.real_robot_session import RobotMotionError

    monkeypatch.setattr("ufactory.xarm_control.time.sleep", lambda _: None)
    arm = MagicMock()
    arm.error_code = 0
    session = object.__new__(RealRobotSession)
    session.arm = arm
    session.dof = 6
    session.home_qpos = np.zeros(6)

    recover_calls = {"n": 0}

    def move_to(q, *, speed_rad_s, wait, move_strategy):
        del speed_rad_s, wait, move_strategy
        if float(np.asarray(q)[0]) == 0.2:
            raise RobotMotionError(
                "collision",
                code=22,
                waypoint_index=1,
                waypoint_count=1,
                target_q=q,
                waypoint_q=q,
            )
        return 0

    session.move_to = move_to

    def recover_after_motion_error(**kwargs):
        del kwargs
        recover_calls["n"] += 1
        return True

    session.recover_after_motion_error = recover_after_motion_error

    poses = [
        ("home", np.zeros(6)),
        ("bad", np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ("next", np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0])),
    ]
    results = run_sim_collision_chain(
        session,
        poses,
        speed_rad_s=DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S,
        move_strategy="direct",
    )

    assert results[0].passed
    assert not results[1].passed
    assert results[1].error_code == 22
    assert results[2].passed
    assert recover_calls["n"] == 1
