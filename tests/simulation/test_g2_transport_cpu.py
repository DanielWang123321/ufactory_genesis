"""CPU transport-quality regression for the default G2 pick-place (about 20 s)."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[2]


def test_default_pick_place_holds_the_cube_without_creep(tmp_path: Path) -> None:
    pytest.importorskip("genesis")
    pytest.importorskip("pinocchio")
    report = tmp_path / "pick.json"
    result = subprocess.run(
        [
            sys.executable,
            "examples/pick_place/run.py",
            "--robot",
            "xarm6",
            "--mode",
            "sim",
            "--executor",
            "servo_cartesian",
            "--backend",
            "cpu",
            "--sim-report",
            str(report),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    data = json.loads(report.read_text())
    acceptance = data["acceptance"]
    assert data["physics_parameters"] and data["completed"] and data["timing_passed"]
    # Without the no-slip pass the cube creeps ~7.4 mm in the pad frame.
    assert acceptance["carry_max_slip_mm"] <= 1.0, acceptance["carry_max_slip_mm"]
    assert data["transport_passed"], acceptance["failed_checks"]
