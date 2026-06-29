"""Smoke tests for xArm 6 Genesis simulation examples."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
NUMBA_CACHE_DIR = os.path.expanduser("~/.cache/numba")


def _run_example(script: str, extra_args: list[str] | None = None, timeout: int = 600) -> subprocess.CompletedProcess:
    script_path = PROJECT_ROOT / script
    cmd = [PYTHON, str(script_path), *(extra_args or [])]
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


def _require_xarm_ip() -> str:
    ip = os.environ.get("XARM_IP")
    if not ip:
        pytest.skip("Set XARM_IP to run hardware tests")
    return ip


@pytest.mark.parametrize(
    "script,extra_args",
    [
        ("examples/xarm6/verify_xarm6.py", []),
        ("examples/xarm6/verify_xarm6_dynamics.py", []),
        ("examples/xarm6/dynamics_verify_real.py", ["--dry-run"]),
        ("examples/xarm6/xarm6_reach_train.py", ["-B", "1", "--max_iterations", "3"]),
        ("examples/xarm6/xarm6_grasp_place_train.py", ["-B", "1", "--max_iterations", "2"]),
    ],
)
def test_xarm6_smoke(script: str, extra_args: list[str]):
    result = _run_example(script, extra_args)
    assert result.returncode == 0, (
        f"{script} failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout[-4000:]}\n"
        f"stderr:\n{result.stderr[-4000:]}"
    )


@pytest.mark.hardware
def test_dynamics_verify_real():
    ip = _require_xarm_ip()
    args = [
        "--ip", ip,
        "--poses", "home,calib_002,calib_004,calib_030",
        "--move-strategy", "direct",
        "--z-min-mm", "0",
    ]
    suffix = os.environ.get("XARM_KINEMATICS_SUFFIX")
    if suffix:
        args.extend(["--kinematics-suffix", suffix])
    result = _run_example(
        "examples/xarm6/dynamics_verify_real.py",
        args,
        timeout=1200,
    )
    assert result.returncode == 0, (
        f"dynamics_verify_real failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout[-4000:]}\n"
        f"stderr:\n{result.stderr[-4000:]}"
    )


@pytest.mark.hardware
def test_fk_verify():
    ip = _require_xarm_ip()
    result = _run_example("examples/xarm6/fk_verify.py", ["--ip", ip])
    assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.hardware
def test_ik_verify():
    ip = _require_xarm_ip()
    result = _run_example("examples/xarm6/ik_verify.py", ["--ip", ip], timeout=900)
    assert result.returncode == 0, result.stderr[-2000:]
