"""Smoke tests for the per-robot grasp-place trajectory example scripts.

Runs each robot's ``<robot>_grasp_place_traj.py --headless`` sim entry point
end-to-end and checks it reports place_error_mm / home_drift_mm within the
script's own pass thresholds (exit code 0). Covers all 5 supported robots:
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
LITE6_TABLE_CUBE_CENTER_Z_MM = 415.0


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


def _phase_obj_z_mm(output: str, label: str) -> float:
    phase_match = re.search(
        rf"\[{re.escape(label)}\s*\].*obj=\[\s*-?\d+,\s*-?\d+,\s*(-?\d+)\]",
        output,
    )
    assert phase_match is not None, f"no {label!r} phase obj z in output\n{output[-2000:]}"
    return float(phase_match.group(1))


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
    result = _run_example(script, ["--headless", "--rate", "50"])
    assert result.returncode == 0, (
        f"{script} failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout[-4000:]}\n"
        f"stderr:\n{result.stderr[-4000:]}"
    )
    place_match = re.search(r"Place error:\s*([\d.]+)\s*mm", result.stdout)
    drift_match = re.search(r"Home drift:\s*([\d.]+)\s*mm", result.stdout)
    assert "sim_grasp_weld=False" in result.stdout
    assert place_match is not None, f"no place error reported\n{result.stdout[-2000:]}"
    assert drift_match is not None, f"no home drift reported\n{result.stdout[-2000:]}"
    assert float(place_match.group(1)) < 50.0
    assert float(drift_match.group(1)) < 10.0
    if "lite6_grasp_place_traj.py" in script:
        assert "sim_hold_bias=2.0mm" in result.stdout
        assert "place-standoff" not in result.stdout
        settle_z_mm = _phase_obj_z_mm(result.stdout, "place-settle")
        assert LITE6_TABLE_CUBE_CENTER_Z_MM - 1.0 <= settle_z_mm <= LITE6_TABLE_CUBE_CENTER_Z_MM + 1.0


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
        ["--executor", "servo_j", "--dry-run", "--rate", "50", "--z-min-mm", "0"],
        timeout=600,
    )
    assert result.returncode == 0, (
        f"xarm6 servo_j dry-run failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout[-4000:]}\n"
        f"stderr:\n{result.stderr[-4000:]}"
    )
    assert "[ik-compile] executor=servo_j" in result.stdout
    assert "timing=preserve-cartesian" in result.stdout
    assert "retimed=False" in result.stdout
    assert "[real DRY-RUN] executor=servo_j" in result.stdout
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
    assert "timing=preserve-cartesian" in result.stdout
    assert "retimed=False" in result.stdout
    assert "[servo_j:descend]" in result.stdout


def test_lite6_grasp_place_rejects_gap_below_reversed_gripper_range():
    result = _run_example(
        "examples/lite6/lite6_grasp_place_traj.py",
        ["--headless", "--grip-gap-mm", "10"],
    )
    assert result.returncode != 0
    assert "--grip-gap-mm must be between 20.0 and 38.0" in result.stderr


@pytest.mark.integration
def test_lite6_contact_diagnose_reports_bilateral_hold():
    result = _run_example(
        "examples/lite6_contact_grasp_diagnose.py",
        ["--gaps-mm", "20"],
    )
    assert result.returncode == 0, (
        f"lite6 contact diagnose failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout[-4000:]}\n"
        f"stderr:\n{result.stderr[-4000:]}"
    )
    hold_lines = [line for line in result.stdout.splitlines() if " hold " in line]
    assert hold_lines, f"no hold summary in output\n{result.stdout[-2000:]}"
    assert "bilateral_pct=100.0" in hold_lines[-1]
    assert "touched_end=[9, 10]" in hold_lines[-1]
    contact_z_match = re.search(r"contact_z_mm=\[([\d.]+),([\d.]+)\]", hold_lines[-1])
    assert contact_z_match is not None, f"no contact z range in hold summary\n{hold_lines[-1]}"
    assert float(contact_z_match.group(1)) > 20.0
    side_match = re.search(r"gap_neg_mm=\s*(-?[\d.]+).*gap_pos_mm=\s*(-?[\d.]+)", hold_lines[-1])
    assert side_match is not None, f"no side clearance in hold summary\n{hold_lines[-1]}"
    assert float(side_match.group(1)) > -0.5
    assert float(side_match.group(2)) > -0.5


@pytest.mark.integration
def test_lite6_standalone_gripper_cube_raw_contact_is_clean():
    result = _run_example(
        "examples/lite6_gripper_cube_diagnose.py",
        ["--collision-mode", "raw", "--rate", "50", "--substeps", "8", "--hold-s", "0.5"],
    )
    assert result.returncode == 0, (
        f"lite6 standalone gripper diagnose failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout[-4000:]}\n"
        f"stderr:\n{result.stderr[-4000:]}"
    )
    open_lines = [line for line in result.stdout.splitlines() if "phase=open" in line]
    hold_lines = [line for line in result.stdout.splitlines() if "phase=hold" in line]
    assert open_lines and hold_lines, f"missing standalone samples\n{result.stdout[-2000:]}"
    assert "bilateral=False" in open_lines[-1]
    assert "rows=0" in open_lines[-1]
    assert "bilateral=True" in hold_lines[-1]
    side_match = re.search(r"side_mm=\[(-?[\d.]+),(-?[\d.]+)\]", hold_lines[-1])
    assert side_match is not None, f"no side clearance in hold line\n{hold_lines[-1]}"
    assert float(side_match.group(1)) > -0.5
    assert float(side_match.group(2)) > -0.5
    contact_z_match = re.search(r"contact_local_z_mm=\[([\d.]+),([\d.]+)\]", hold_lines[-1])
    assert contact_z_match is not None, f"no local contact z in hold line\n{hold_lines[-1]}"
    assert float(contact_z_match.group(1)) > 20.0


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
