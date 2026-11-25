"""Ad-hoc real-robot checks (FK / IK / dynamics) on the arm at ``XARM_IP``.

Release evidence uses ``project-check hardware --inventory …``; these cases remain for
single-cabinet debugging. The robot model is read from the controller; set ``XARM_ROBOT``
(xarm5, xarm6, xarm7, lite6, uf850) only when the controller model cannot be read.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
NUMBA_CACHE_DIR = os.path.expanduser("~/.cache/numba")
ROBOT_HINT = "set XARM_ROBOT to one of xarm5, xarm6, xarm7, lite6, uf850 and rerun"


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


def _tail(result: subprocess.CompletedProcess, limit: int = 2000) -> str:
    """Last output of a failed example run (FAIL lines go to stdout, tracebacks to stderr)."""
    return (result.stdout + result.stderr)[-limit:]


def _read_controller_robot(ip: str) -> str | None:
    """Connect read-only and return the robot name the controller reports (``None`` if unknown)."""
    from xarm.wrapper import XArmAPI

    from ufactory.hardware.xarm import controller_robot_name

    arm = XArmAPI(ip, is_radian=True)
    try:
        if not arm.connected:
            pytest.fail(f"cannot connect to the controller at {ip}", pytrace=False)
        return controller_robot_name(arm)
    finally:
        arm.disconnect()


@pytest.fixture(scope="module")
def real_robot() -> tuple[str, str]:
    """``(ip, robot_name)`` of the arm at ``XARM_IP``."""
    from ufactory.robots.registry import get_robot_profile

    ip = os.environ.get("XARM_IP")
    if not ip:
        pytest.skip("Set XARM_IP to run hardware tests")
    manual = os.environ.get("XARM_ROBOT", "").strip()
    try:
        declared = get_robot_profile(manual).robot_name if manual else None
    except KeyError:
        pytest.fail(f"unknown XARM_ROBOT={manual!r}; {ROBOT_HINT}", pytrace=False)
    detected = _read_controller_robot(ip)
    if detected is None:
        if declared is None:
            pytest.fail(f"cannot read the robot model from the controller at {ip}; {ROBOT_HINT}", pytrace=False)
        return ip, declared
    if declared is not None and declared != detected:
        pytest.fail(f"XARM_ROBOT={manual} but the controller at {ip} reports {detected}", pytrace=False)
    return ip, detected


@pytest.mark.hardware
def test_dynamics_verify_real(real_robot):
    from ufactory.dynamics.cli import cli_hardware_check

    ip, robot = real_robot
    args = [
        "--robot",
        robot,
        "--ip",
        ip,
        "--poses",
        "0,1,3",
        "--move-strategy",
        "direct",
        "--confirm-real",
    ]
    suffix = os.environ.get("XARM_KINEMATICS_SUFFIX")
    if suffix:
        args.extend(["--kinematics-suffix", suffix])
    rc = cli_hardware_check(args)
    assert rc == 0, f"cli_hardware_check failed for {robot} (exit {rc})"


@pytest.mark.hardware
def test_fk_verify(real_robot):
    ip, robot = real_robot
    result = _run_example("examples/kinematics/verify_fk.py", ["--robot", robot, "--ip", ip])
    assert result.returncode == 0, _tail(result)


@pytest.mark.hardware
def test_ik_verify(real_robot):
    ip, robot = real_robot
    result = _run_example("examples/kinematics/verify_ik.py", ["--robot", robot, "--ip", ip], timeout=900)
    assert result.returncode == 0, _tail(result)
