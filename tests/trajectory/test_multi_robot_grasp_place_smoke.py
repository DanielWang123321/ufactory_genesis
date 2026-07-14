"""Smoke tests for the per-robot grasp-place trajectory example scripts.

Runs each robot's unified ``--mode sim --executor servo_cartesian`` entry point
end-to-end and checks it reports place_error_mm / home_drift_mm within the
command's own pass thresholds (exit code 0). Covers all 5 supported robots:
xArm5/6/7 and UF850 (Gripper G2, 30 mm cube) and Lite6 (reversed parallel
gripper, 30 mm cube).
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

PYTHON = sys.executable
NUMBA_CACHE_DIR = os.path.expanduser("~/.cache/numba")
pytestmark = pytest.mark.integration


def _run_example(script: str, extra_args: list[str] | None = None, timeout: int = 300) -> subprocess.CompletedProcess:
    script_path = PROJECT_ROOT / script
    cmd = [PYTHON, str(script_path), *(extra_args or [])]
    env = os.environ.copy()
    env.setdefault("NUMBA_CACHE_DIR", NUMBA_CACHE_DIR)
    return subprocess.run(
        cmd,
        cwd=script_path.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "script",
    [
        "examples/xarm5/xarm5_grasp_place_traj.py",
        "examples/xarm6/xarm6_grasp_place_traj.py",
        "examples/xarm7/xarm7_grasp_place_traj.py",
        "examples/uf850/uf850_grasp_place_traj.py",
        "examples/lite6/lite6_grasp_place_traj.py",
    ],
)
def test_grasp_place_traj_headless_sim(script: str):
    result = _run_example(script, ["--mode", "sim", "--executor", "servo_cartesian"])
    assert result.returncode == 0, (
        f"{script} failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout[-4000:]}\n"
        f"stderr:\n{result.stderr[-4000:]}"
    )
    place_match = re.search(r"place_error_mm=([\d.]+)", result.stdout)
    drift_match = re.search(r"home_drift_mm=([\d.]+)", result.stdout)
    assert "preflight=PASS" in result.stdout
    assert place_match is not None, f"no place error reported\n{result.stdout[-2000:]}"
    assert drift_match is not None, f"no home drift reported\n{result.stdout[-2000:]}"
    assert float(place_match.group(1)) < 50.0
    assert float(drift_match.group(1)) < 10.0


@pytest.mark.integration
@pytest.mark.parametrize(
    "script,expect_gap_mm",
    [
        ("examples/xarm6/xarm6_grasp_place_traj.py", 84.0),
        ("examples/lite6/lite6_grasp_place_traj.py", 38.0),
    ],
)
def test_grasp_place_traj_real_dry_run_reports_gripper_family(script: str, expect_gap_mm: float):
    """Real-path dry-run must resolve each robot's own gripper gap (no G2 hardcode leak)."""
    result = _run_example(script, ["--executor", "servo_cartesian", "--dry-run", "--rate", "50", "--z-min-mm", "0"])
    assert result.returncode == 0, (
        f"{script} dry-run failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout[-4000:]}\n"
        f"stderr:\n{result.stderr[-4000:]}"
    )
    assert f"gripper gap {expect_gap_mm:.1f}mm ->" in result.stdout


@pytest.mark.integration
def test_xarm6_grasp_place_servo_j_dry_run_compiles_host_ik():
    result = _run_example(
        "examples/xarm6/xarm6_grasp_place_traj.py",
        ["--mode", "dry-run", "--executor", "servo_j"],
        timeout=600,
    )
    assert result.returncode == 0, (
        f"xarm6 servo_j dry-run failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout[-4000:]}\n"
        f"stderr:\n{result.stderr[-4000:]}"
    )
    assert "[ik-compile] building Genesis scene" in result.stdout
    assert "[ik-compile] complete samples=" in result.stdout
    assert "[preflight] checking samples=" in result.stdout
    assert "preflight=PASS" in result.stdout
    assert "[preflight] complete status=PASS" in result.stdout


@pytest.mark.integration
def test_uf850_servo_j_dry_run_uses_stable_ik_damping():
    result = _run_example(
        "examples/uf850/uf850_grasp_place_traj.py",
        [
            "--executor",
            "servo_j",
            "--dry-run",
            "--rate",
            "50",
            "--z-min-mm",
            "0",
        ],
        timeout=600,
    )
    assert result.returncode == 0, (
        f"uf850 servo_j dry-run failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout[-4000:]}\n"
        f"stderr:\n{result.stderr[-4000:]}"
    )
    assert "timing=joint-lspb-retime" in result.stdout
    assert "damping=0.05" in result.stdout
    assert "retimed=False" in result.stdout
    assert "[servo_j:descend]" in result.stdout


@pytest.mark.integration
def test_lite6_grasp_place_servo_j_dry_run_passes_default_150mm_s_timing():
    result = _run_example(
        "examples/lite6/lite6_grasp_place_traj.py",
        ["--executor", "servo_j", "--dry-run", "--rate", "50", "--z-min-mm", "0"],
        timeout=600,
    )
    assert result.returncode == 0, (
        f"lite6 servo_j dry-run failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout[-4000:]}\n"
        f"stderr:\n{result.stderr[-4000:]}"
    )
    assert "[ik-compile] executor=servo_j" in result.stdout
    assert "timing=joint-lspb-retime" in result.stdout
    assert "retimed=False" in result.stdout
    assert "[servo_j:descend]" in result.stdout


def test_lite6_grasp_place_rejects_gap_below_reversed_gripper_range():
    result = _run_example(
        "examples/lite6/lite6_grasp_place_traj.py",
        ["--headless", "--grip-gap-mm", "10"],
    )
    assert result.returncode != 0
    assert "--grip-gap-mm must be between 20.0 and 38.0" in result.stderr


def test_xarm5_grasp_place_real_dry_run_reports_gripper_gap():
    """xArm5 real-path dry-run resolves G2 gripper gap like other xArm arms."""
    result = _run_example(
        "examples/xarm5/xarm5_grasp_place_traj.py",
        ["--executor", "servo_cartesian", "--dry-run", "--rate", "50", "--z-min-mm", "0"],
    )
    assert result.returncode == 0, (
        f"xarm5 dry-run failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout[-4000:]}\n"
        f"stderr:\n{result.stderr[-4000:]}"
    )
    assert "gripper gap 84.0mm ->" in result.stdout
