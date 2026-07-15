"""Smoke tests for the task-oriented pick-place example entry point."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import pytest

from conftest import PROJECT_ROOT

PYTHON = sys.executable
SCRIPT = "examples/pick_place/run.py"
NUMBA_CACHE_DIR = os.path.expanduser("~/.cache/numba")
pytestmark = pytest.mark.integration


def _run_example(robot: str, args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    script_path = PROJECT_ROOT / SCRIPT
    env = os.environ.copy()
    env.setdefault("NUMBA_CACHE_DIR", NUMBA_CACHE_DIR)
    return subprocess.run(
        [PYTHON, str(script_path), "--robot", robot, *args],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


@pytest.mark.parametrize("robot", ["xarm5", "xarm6", "xarm7", "uf850", "lite6"])
def test_pick_place_headless_sim(robot: str):
    result = _run_example(robot, ["--mode", "sim", "--executor", "servo_cartesian"])
    assert result.returncode == 0, (
        f"{robot} failed (exit {result.returncode})\nstdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}"
    )
    place_match = re.search(r"place_error_mm=([\d.]+)", result.stdout)
    drift_match = re.search(r"home_drift_mm=([\d.]+)", result.stdout)
    assert "preflight=PASS" in result.stdout
    assert place_match is not None, f"no place error reported\n{result.stdout[-2000:]}"
    assert drift_match is not None, f"no home drift reported\n{result.stdout[-2000:]}"
    assert float(place_match.group(1)) < 50.0
    assert float(drift_match.group(1)) < 10.0


@pytest.mark.parametrize("robot", ["xarm6", "uf850", "lite6"])
def test_pick_place_servo_j_dry_run_compiles_host_ik(robot: str):
    result = _run_example(robot, ["--mode", "dry-run", "--executor", "servo_j"], timeout=600)
    assert result.returncode == 0, (
        f"{robot} servo_j dry-run failed (exit {result.returncode})\n"
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
    result = _run_example(robot, ["--mode", "dry-run", "--executor", "servo_cartesian", "--print-config"])
    assert result.returncode == 0, result.stderr[-2000:]
    assert f"key: {robot}" in result.stdout
