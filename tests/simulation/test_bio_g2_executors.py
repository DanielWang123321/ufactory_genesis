"""Bio G2 executor selection, online IK continuity, validation, and failure reports."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from ufactory.trajectory import bio_g2_replay as replay


@pytest.fixture
def ik_sim():
    sim = object.__new__(replay.XArm6BioG2PickPlaceSimulation)
    sim.frames = replay.BioG2TaskFrames.load()
    sim.runtime = sim.frames.runtime
    sim.arm_joint_names = sim.runtime.arm.joint_names
    sim.arm_dof_idx = list(range(6))
    sim.arm_limits = replay.arm_joint_limits()
    sim.last_arm_q = np.asarray(replay.XARM6_DEFAULT_QPOS)
    sim.max_ik_position_error_m = sim.max_ik_rotation_error_rad = 0.0
    sim.ik_total_ms = 0.0
    sim.ik_calls = 0
    sim.adapter = SimpleNamespace(indices=[6, 7])
    sim.controller = SimpleNamespace(reference_q_m=0.02)
    sim.flange_link = SimpleNamespace(get_pos=lambda: np.zeros(3), get_quat=lambda: replay.DOWN_QUAT_WXYZ)
    sim.measured_pose = lambda: {"flange_position_error_mm": 0, "flange_rotation_error_rad": 0}
    sim.joint_rows = []
    sim.last_environment_contacts = {"robot_table_contacts": 0, "robot_nest_contacts": 0}
    return sim


@pytest.mark.parametrize("executor", replay.EXECUTORS)
def test_cli_passes_selected_executor(monkeypatch, executor, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("genesis")
    from ufactory.cli import pick_place

    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return {"passed": True, "metrics": {}}

    monkeypatch.setattr(replay, "run_bio_g2_pick_place", run)
    assert (
        pick_place.main(
            [
                "--robot",
                "xarm6",
                "--gripper",
                "bio_gripper_g2",
                "--mode",
                "sim",
                "--executor",
                executor,
                "--substeps",
                "32",
                "--visual",
                "--csv",
                str(tmp_path / "samples.csv"),
            ]
        )
        == 0
    )
    assert calls[0]["executor"] == executor
    assert calls[0]["substeps"] == 32
    assert calls[0]["headless"] is False


def test_python_and_parser_default_to_servo_cartesian():
    import inspect

    for entry in (replay.run_bio_g2_pick_place, replay.XArm6BioG2PickPlaceSimulation):
        assert inspect.signature(entry).parameters["executor"].default == "servo_cartesian"
        assert inspect.signature(entry).parameters["robot"].default == "xarm6"
    assert replay.parse_args([]).executor == "servo_cartesian"
    assert replay.parse_args(["--executor", "servo_cartesian"]).executor == "servo_cartesian"
    with pytest.raises(SystemExit):
        replay.parse_args(["--executor", "unknown"])
    with pytest.raises(ValueError, match="executor"):
        replay.run_bio_g2_pick_place(executor="unknown")


@pytest.mark.parametrize("executor", replay.EXECUTORS)
def test_preparation_and_stream_match_centered_joint_velocities(ik_sim, executor):
    calls = []
    events = []
    ik_sim.executor = executor
    ik_sim.transport_speed_mm_s = 250.0

    def solve(xyz, seed_q, grasp_quat):
        calls.append((np.asarray(xyz).copy(), seed_q.copy()))
        # A nonlinear, deterministic IK surrogate exposes incorrect velocity differences.
        return np.r_[np.asarray(xyz) ** 2, np.sin(xyz)]

    ik_sim.solve_arm_ik = solve

    def reset(q, initial_gap_mm):
        assert initial_gap_mm == 150.0
        events.append("reset")
        ik_sim._joint_stream = ik_sim._new_joint_stream(q)

    ik_sim.reset = reset
    ik_sim.start_viewer = lambda: events.append("viewer")
    ticks = ik_sim.prepare(replay.default_phases())
    prepared_calls = len(calls)
    moving_count = sum(not tick["stationary"] for tick in ticks)
    assert prepared_calls == (1 if executor == "servo_cartesian" else moving_count + 1)
    assert events == ["reset", "viewer"]
    assert all("target_arm_q" not in tick for tick in ticks)
    actual = []
    for idx, tick in enumerate(ticks):
        if executor == "servo_cartesian":
            actual.append(ik_sim._joint_stream.targets(tick, ticks[idx + 1] if idx + 1 < len(ticks) else None))
        else:
            actual.append(ik_sim.joint_targets[idx])
    assert len(calls) == moving_count + 1
    for (_, seed), (previous_xyz, _) in zip(calls[1:], calls):
        np.testing.assert_array_equal(seed, np.r_[previous_xyz**2, np.sin(previous_xyz)])
    previous = ik_sim.home_arm_q
    for block in sorted({tick["motion_block"] for tick in ticks}):
        indices = [idx for idx, tick in enumerate(ticks) if tick["motion_block"] == block]
        q = np.array([actual[idx][0] for idx in indices])
        expected = np.gradient(np.vstack((previous, q)), replay.DT, axis=0)[1:]
        expected[[0, -1]] = 0
        np.testing.assert_allclose([actual[idx][1] for idx in indices], expected, atol=1e-14)
        previous = q[-1]


def test_runtime_ik_precedes_physics_and_prefetches_only_one_sample(ik_sim, monkeypatch):
    calls = []
    ik_sim.executor, ik_sim.headless, ik_sim.time_s = "servo_cartesian", True, 0.0
    home_q = np.zeros(6)
    ticks = replay.sample_cartesian_trajectory(replay.default_phases())
    for idx, tick in enumerate(ticks):
        tick["target_index"] = idx

    def solve(xyz, seed_q, grasp_quat):
        calls.append(("ik", np.asarray(xyz).copy(), seed_q.copy()))
        ik_sim.ik_total_ms += 1.0
        ik_sim.ik_calls += 1
        return np.r_[np.asarray(xyz) - replay.HOME_GRASP_POS, [0, 0, 0]]

    ik_sim.solve_arm_ik = solve
    ik_sim._joint_stream = ik_sim._new_joint_stream(home_q)
    ik_sim.control_arm = lambda q, dq: calls.append(("arm", q.copy(), dq.copy())) or np.zeros(6)
    ik_sim.scene = SimpleNamespace(step=lambda **kwargs: calls.append(("physics",)))
    ik_sim.controller = SimpleNamespace(
        position_mm=150.0,
        command=lambda **kwargs: None,
        apply=lambda dt: None,
        telemetry=lambda: {},
    )
    ik_sim.adapter = SimpleNamespace(
        drive_force=lambda: (0.0, 0.0), position_resets=1, positions=lambda: np.zeros(8), indices=[6, 7]
    )
    ik_sim.gripper_poses = lambda: (np.zeros((3, 3)), np.tile(replay.DOWN_QUAT_WXYZ, (3, 1)))
    ik_sim.contact_telemetry = lambda quat: (0, 0, 0)
    ik_sim.reagent_kit = SimpleNamespace(get_pos=lambda: np.zeros(3), get_quat=lambda: np.array([1, 0, 0, 0]))
    ik_sim.gripping_planes = []
    monkeypatch.setattr(replay, "signed_plate_gaps_mm", lambda *args: (32, 32))

    first = ik_sim.step(ticks[0], 200, 50, next_tick=ticks[1])
    assert [call[0] for call in calls] == ["ik", "ik", "arm", "physics"]
    assert first["executor"] == "servo_cartesian"
    assert first["ik_solve_count"] == 2
    assert first["ik_ms"] == 2.0
    np.testing.assert_array_equal(calls[2][2], 0.0)
    second = ik_sim.step(ticks[1], 200, 50, next_tick=ticks[2])
    assert second["ik_solve_count"] == 1
    assert np.linalg.norm(calls[-2][2]) > 0
    assert ik_sim._joint_stream.pending_tick is ticks[2]

    ik_sim.executor = "servo_j"
    ik_sim.joint_targets = [(np.zeros(6), np.zeros(6))] * len(ticks)
    third = ik_sim.step(ticks[2], 200, 50)
    assert third["ik_solve_count"] == 0
    assert third["ik_ms"] == 0.0


def test_ik_converts_world_tcp_to_flange_with_explicit_seed(ik_sim):
    target = np.array([0.38, -0.12, 0.44])
    seed = np.linspace(0, 0.05, 6)
    solution = np.r_[seed, 0.02, -0.02]

    def inverse(**kwargs):
        expected = ik_sim.frames.tcp_to_flange(replay.pose_matrix(target, replay.DOWN_QUAT_WXYZ))
        xyz, quat = replay.pose_components(expected)
        np.testing.assert_array_equal(kwargs["pos"], xyz)
        np.testing.assert_array_equal(kwargs["quat"], quat)
        assert kwargs["link"] is ik_sim.flange_link
        np.testing.assert_array_equal(kwargs["init_qpos"], solution)
        assert kwargs["respect_joint_limit"] is True
        assert kwargs["pos_tol"] == 1e-5
        assert kwargs["rot_tol"] == 1e-4
        return solution, np.zeros(6)

    ik_sim.robot = SimpleNamespace(n_dofs=8, inverse_kinematics=inverse)
    np.testing.assert_array_equal(ik_sim.solve_arm_ik(target, seed), seed)
    np.testing.assert_array_equal(ik_sim.last_arm_q, seed)
    assert ik_sim.ik_calls == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["target", "seed", "solution", "residual", "gripper"])
def test_ik_rejects_nonfinite_inputs_and_results(ik_sim, bad, field):
    target, seed, solution, residual = np.zeros(3), np.zeros(6), np.zeros(8), np.zeros(6)
    values = {"target": target, "seed": seed, "solution": solution, "residual": residual}
    if field == "gripper":
        ik_sim.controller.reference_q_m = bad
    else:
        values[field][0] = bad
    ik_sim.robot = SimpleNamespace(n_dofs=8, inverse_kinematics=lambda **kwargs: (solution, residual))
    with pytest.raises((RuntimeError, ValueError), match="IK failed"):
        ik_sim.solve_arm_ik(target, seed)
    assert ik_sim.ik_calls == 1
    np.testing.assert_array_equal(ik_sim.last_arm_q, replay.XARM6_DEFAULT_QPOS)


@pytest.mark.parametrize("joint", range(6))
@pytest.mark.parametrize("bound,offset", [(0, -1e-6), (1, 1e-6)])
def test_ik_rejects_urdf_joint_limit_violations(ik_sim, joint, bound, offset):
    solution = np.zeros(8)
    solution[joint] = ik_sim.arm_limits[joint, bound] + offset
    ik_sim.robot = SimpleNamespace(n_dofs=8, inverse_kinematics=lambda **kwargs: (solution, np.zeros(6)))
    with pytest.raises(ValueError, match=f"{replay.ARM_JOINT_NAMES[joint]} position"):
        ik_sim.solve_arm_ik(replay.HOME_GRASP_POS)


def test_joint_velocity_failure_reports_lookahead_target_and_prevents_motion():
    ticks = replay.sample_cartesian_trajectory(replay.default_phases())
    seeds = []

    def solve(xyz, seed_q, grasp_quat):
        seeds.append(seed_q.copy())
        return seed_q + (0.001 if len(seeds) == 1 else 0.1)

    stream = replay._JointTargetStream(solve, np.zeros(6), replay.arm_joint_limits()[:, 2])
    with pytest.raises(replay.IKExecutionError, match="target velocity") as exc:
        stream.targets(ticks[0], ticks[1])
    assert exc.value.report["lookahead"] is True
    assert exc.value.report["phase_step"] == 1
    assert exc.value.report["target_grasp_xyz_m"] == ticks[1]["target_grasp_xyz"].tolist()
    np.testing.assert_array_equal(stream.previous_q, 0.0)


@pytest.mark.parametrize("stage", ["preparation", "execution"])
def test_ik_failure_preserves_csv_and_structured_json(monkeypatch, tmp_path, stage):
    tick = {"phase": "transit", "phase_step": 4, "target_grasp_xyz": [0.38, 0.12, float("nan")]}
    failure = replay.IKExecutionError(tick, "unreachable target")
    sim = SimpleNamespace(
        executor="servo_cartesian",
        ik_total_ms=0.0,
        ik_calls=0,
        max_ik_position_error_m=0.0,
        max_ik_rotation_error_rad=0.0,
        parameters=replay.ServoParameters(),
        initial_pose=None,
        frames=replay.BioG2TaskFrames.load(),
        cartesian_targets=[],
        joint_targets=[],
        joint_rows=[],
    )
    sim.runtime, sim.arm_joint_names = sim.frames.runtime, sim.frames.runtime.arm.joint_names

    def prepare(phases):
        sim.ik_total_ms += 2
        sim.ik_calls += 1
        if stage == "preparation":
            raise failure
        return []

    def execute(*args):
        yield {
            "phase": "pregrasp",
            "time_s": 0.02,
            "gap_mm": 150,
            "kit_x_mm": 380,
            "kit_y_mm": -120,
            "kit_z_mm": 437,
            "left_normal_n": 0,
            "right_normal_n": 0,
            "compute_ms": 21,
            "ee_error_mm": 0.2,
            "ee_rotation_error_rad": 0.001,
            "ik_ms": 1.0,
            "executor": "servo_cartesian",
        }
        sim.ik_total_ms += 3
        sim.ik_calls += 1
        raise failure

    sim.prepare = prepare
    monkeypatch.setattr(replay, "BioG2PickPlaceSimulation", lambda **kwargs: sim)
    monkeypatch.setattr(replay, "execute_trajectory", execute)
    monkeypatch.setattr(replay, "build_microplate", lambda: {})
    csv_path, report_path = tmp_path / "samples.csv", tmp_path / "report.json"
    report = replay.run_bio_g2_pick_place(executor="servo_cartesian", csv_path=csv_path, summary_path=report_path)
    saved = json.loads(report_path.read_text(), parse_constant=lambda value: pytest.fail(f"Invalid JSON {value}"))
    assert saved == report
    assert report["passed"] is False
    assert report["checks"] == {"ik_solution_valid": False}
    assert report["failure_details"]["stage"] == stage
    assert report["failure_details"]["phase"] == "transit"
    assert report["failure_details"]["target_grasp_xyz_m"][-1] is None
    assert report["preparation_ik_calls"] == 1
    assert report["runtime_ik_calls"] == (1 if stage == "execution" else 0)
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == (1 if stage == "execution" else 0)
    assert report["metrics"]["sample_count"] == len(rows)
    assert report["steps_over_20ms"] == len(rows)


@pytest.mark.parametrize("executor", replay.EXECUTORS)
def test_substep_comparison_requires_same_executor_and_accepts_legacy_servo_j(executor):
    first = {"substeps": 8, "executor": executor}
    second = {"substeps": 32, "executor": "servo_cartesian" if executor == "servo_j" else "servo_j"}
    with pytest.raises(ValueError, match="executor"):
        replay.compare_substeps(first, second)
    first.pop("executor")
    second["executor"] = "servo_cartesian"
    with pytest.raises(ValueError, match="executor"):
        replay.compare_substeps(first, second)
