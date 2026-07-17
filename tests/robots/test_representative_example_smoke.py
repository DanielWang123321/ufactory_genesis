"""xArm6 representative subprocess smoke for public example entry points.

Five-robot physical CLI evidence belongs in ``project-check sdk-sim`` (parallel by IP).
Real-robot FK/IK/dynamics checks live in ``tests/hardware/test_xarm6_real.py``;
release evidence still uses ``project-check hardware`` inventory.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))
from conftest import PROJECT_ROOT

pytestmark = pytest.mark.integration

PYTHON = sys.executable
NUMBA_CACHE_DIR = os.path.expanduser("~/.cache/numba")
REPRESENTATIVE = "xarm6"
PICK_PLACE = "examples/pick_place/run.py"


def _run(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.setdefault("NUMBA_CACHE_DIR", NUMBA_CACHE_DIR)
    return subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _run_pick_place(robot: str, args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return _run([PYTHON, PICK_PLACE, "--robot", robot, *args], timeout=timeout)


def _run_dynamics_cli(fn_name: str, args: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    """Run a dynamics CLI entry in a fresh process (Genesis allows one init per process)."""
    code = f"import sys; from ufactory.dynamics.cli import {fn_name}; raise SystemExit({fn_name}(sys.argv[1:]))"
    return _run([PYTHON, "-c", code, *args], timeout=timeout)


def test_verify_robot_headless_representative():
    result = _run([PYTHON, "examples/kinematics/verify_robot.py", "--robot", REPRESENTATIVE])
    assert result.returncode == 0, result.stderr[-3000:]


def test_view_robot_headless_representative():
    result = _run([PYTHON, "examples/visualization/view_robot.py", "--robot", REPRESENTATIVE, "--headless"])
    assert result.returncode == 0, result.stderr[-3000:]


def test_view_robot_gripper_demo_headless_representative():
    result = _run(
        [
            PYTHON,
            "examples/visualization/view_robot.py",
            "--robot",
            "xarm6_1305",
            "--gripper-g2",
            "--movable",
            "--gripper-demo",
            "--headless",
        ]
    )
    assert result.returncode == 0, result.stderr[-3000:]


def test_pick_place_headless_sim_representative():
    result = _run_pick_place(REPRESENTATIVE, ["--mode", "sim", "--executor", "servo_cartesian"])
    assert result.returncode == 0, (
        f"{REPRESENTATIVE} failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}"
    )
    place_match = re.search(r"place_error_mm=([\d.]+)", result.stdout)
    drift_match = re.search(r"home_drift_mm=([\d.]+)", result.stdout)
    assert "preflight=PASS" in result.stdout
    assert place_match is not None, f"no place error reported\n{result.stdout[-2000:]}"
    assert drift_match is not None, f"no home drift reported\n{result.stdout[-2000:]}"
    assert float(place_match.group(1)) < 50.0
    assert float(drift_match.group(1)) < 10.0


def test_pick_place_servo_j_dry_run_compiles_host_ik_representative():
    result = _run_pick_place(REPRESENTATIVE, ["--mode", "dry-run", "--executor", "servo_j"], timeout=600)
    assert result.returncode == 0, (
        f"{REPRESENTATIVE} servo_j dry-run failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout[-4000:]}\n"
        f"stderr:\n{result.stderr[-4000:]}"
    )
    assert "[ik-compile] building Genesis scene" in result.stdout
    assert "[ik-compile] complete samples=" in result.stdout
    assert "[preflight] checking samples=" in result.stdout
    assert "preflight=PASS" in result.stdout
    assert "[preflight] complete status=PASS" in result.stdout


@pytest.mark.parametrize("robot", ["xarm5", "xarm6", "xarm7", "uf850", "lite6"])
def test_pick_place_print_config_resolves_each_robot(robot: str):
    result = _run_pick_place(robot, ["--mode", "dry-run", "--executor", "servo_cartesian", "--print-config"])
    assert result.returncode == 0, result.stderr[-2000:]
    assert f"key: {robot}" in result.stdout


def test_dynamics_sim_cli_representative():
    result = _run_dynamics_cli("cli_sim_check", ["--robot", REPRESENTATIVE, "--random-count", "5"])
    assert result.returncode in {0, 1}, (
        f"cli_sim_check unexpected exit {result.returncode}\n"
        f"stdout:\n{result.stdout[-4000:]}\n"
        f"stderr:\n{result.stderr[-4000:]}"
    )


def test_dynamics_hardware_dry_run_cli_representative():
    result = _run_dynamics_cli("cli_hardware_check", ["--robot", REPRESENTATIVE, "--dry-run"])
    assert result.returncode == 0, (
        f"cli_hardware_check --dry-run exited {result.returncode}\n"
        f"stdout:\n{result.stdout[-4000:]}\n"
        f"stderr:\n{result.stderr[-4000:]}"
    )
