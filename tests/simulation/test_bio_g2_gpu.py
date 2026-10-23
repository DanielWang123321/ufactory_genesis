"""Complete Bio G2 executor acceptance and 8/32-substep agreement in isolated GPU processes."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from ufactory.trajectory.bio_g2_replay import compare_substeps
from ufactory.trajectory.bio_g2_frames import BioG2TaskFrames, SUPPORTED_ROBOTS
from ufactory.simulation.bio_g2_collision import audit_collisions

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.mark.parametrize("executor", ["servo_j", "servo_cartesian"])
@pytest.mark.parametrize("robot", SUPPORTED_ROBOTS)
def test_microplate_executor_physics_and_substep_agreement(robot, executor, tmp_path):
    reports = []
    for substeps in (8, 32):
        report_path = tmp_path / f"{executor}_{substeps}.json"
        csv_path = report_path.with_suffix(".csv")
        result = subprocess.run(
            [
                sys.executable,
                "examples/pick_place/run.py",
                "--robot",
                robot,
                "--gripper",
                "bio_gripper_g2",
                "--mode",
                "sim",
                "--backend",
                "gpu",
                "--executor",
                executor,
                "--substeps",
                str(substeps),
                "--csv",
                str(csv_path),
                "--sim-report",
                str(report_path),
            ],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        report = json.loads(report_path.read_text()) if report_path.exists() else None
        assert result.returncode == 0, (report, result.stdout[-3000:], result.stderr[-3000:])
        assert report["passed"], report
        assert report["executor"] == executor
        assert report["robot"] == BioG2TaskFrames.load(robot).runtime.model.key
        for pose in (report["initial_pose"], report["final_pose"]):
            assert pose["flange_position_error_mm"] <= 0.5
            assert pose["flange_rotation_error_rad"] <= 1e-3
        np.testing.assert_allclose(report["home_flange_target"]["xyz_mm"], [204.977625, -0.524202, 197.308], atol=1e-12)
        assert report["max_ik_position_error_mm"] <= 0.01
        assert report["max_ik_rotation_error_rad"] <= 1e-4
        assert report["metrics"]["place_xy_error_mm"] <= 5
        assert report["metrics"]["place_z_error_mm"] <= 2
        assert report["metrics"]["transport_vertical_slip_mm"] <= 3
        assert report["metrics"]["total_nest_finger_contacts"] == 0
        if executor == "servo_cartesian":
            assert report["preparation_ik_calls"] == 1
            assert report["runtime_ik_calls"] > 100
        else:
            assert report["preparation_ik_calls"] > 100
            assert report["runtime_ik_calls"] == 0
        with csv_path.open() as stream:
            rows = list(csv.DictReader(stream))
        assert all(row["executor"] == executor for row in rows)
        assert len(rows) == report["metrics"]["sample_count"]
        assert len({row["position_resets"] for row in rows}) == 1
        assert sum(int(row["ik_solve_count"]) for row in rows) == report["runtime_ik_calls"]
        assert sum(float(row["compute_ms"]) > 20 for row in rows) == report["steps_over_20ms"]
        assert np.isfinite([float(row["ee_error_mm"]) for row in rows]).all()
        joints = json.loads(Path(report["trajectory_files"]["joint_targets"]).read_text())
        collision_audit = audit_collisions(BioG2TaskFrames.load(robot), joints)
        report_path.with_suffix(".collision.json").write_text(json.dumps(collision_audit, indent=2) + "\n")
        assert collision_audit["passed"], collision_audit
        assert collision_audit["max_mounting_fk_error_mm"] < 1e-8
        reports.append(report)
    comparison = compare_substeps(*reports)
    (tmp_path / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    assert comparison["passed"], comparison


@pytest.mark.parametrize("executor", ["servo_j", "servo_cartesian"])
def test_unreachable_target_stops_and_saves_samples(executor, tmp_path):
    script = """
import sys
from pathlib import Path
import numpy as np
from ufactory.trajectory import bio_g2_replay as replay

sample = replay.sample_cartesian_trajectory
def unreachable_sample(*args, **kwargs):
    ticks = sample(*args, **kwargs)
    ticks[20]['target_grasp_xyz'] = np.array([10.0, 0.0, 1.0])
    return ticks
replay.sample_cartesian_trajectory = unreachable_sample
output = Path(sys.argv[2])
report = replay.run_bio_g2_pick_place(
    executor=sys.argv[1], csv_path=output / 'samples.csv', summary_path=output / 'failure.json'
)
raise SystemExit(0 if not report['passed'] else 1)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, executor, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, (result.stdout[-3000:], result.stderr[-3000:])
    report = json.loads((tmp_path / "failure.json").read_text())
    assert report["passed"] is False
    assert report["checks"] == {"ik_solution_valid": False}
    failure = report["failure_details"]
    assert failure["phase"] == "pregrasp"
    assert failure["phase_step"] == 20
    assert failure["target_grasp_xyz_m"] == [10.0, 0.0, 1.0]
    assert failure["lookahead"] is True
    assert "position=" in failure["reason"]
    assert failure["stage"] == ("preparation" if executor == "servo_j" else "execution")
    with (tmp_path / "samples.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == report["metrics"]["sample_count"]
    if executor == "servo_cartesian":
        assert 0 < len(rows) < 40
        assert rows[-1]["phase"] == "pregrasp"
    else:
        assert not rows
