"""Smoke tests for xArm 6 Genesis simulation examples."""

from __future__ import annotations

import os
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


@pytest.mark.integration
@pytest.mark.parametrize(
    "script,extra_args",
    [
        ("examples/kinematics/verify_robot.py", ["--robot", "xarm6"]),
        (
            "examples/pick_place/run.py",
            ["--robot", "xarm6", "--mode", "sim", "--executor", "servo_cartesian"],
        ),
    ],
)
def test_xarm6_smoke(script: str, extra_args: list[str]):
    result = _run_example(script, extra_args)
    assert result.returncode == 0, (
        f"{script} failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout[-4000:]}\n"
        f"stderr:\n{result.stderr[-4000:]}"
    )


@pytest.mark.integration
def test_xarm6_dynamics_sim_cli():
    from ufactory.dynamics.cli import cli_sim_check

    rc = cli_sim_check(["--robot", "xarm6", "--random-count", "5"])
    assert rc in {0, 1}, f"cli_sim_check unexpected exit {rc}"


@pytest.mark.integration
def test_xarm6_dynamics_hardware_dry_run_cli():
    from ufactory.dynamics.cli import cli_hardware_check

    rc = cli_hardware_check(["--robot", "xarm6", "--dry-run"])
    assert rc == 0, f"cli_hardware_check --dry-run exited {rc}"


@pytest.mark.hardware
def test_dynamics_verify_real():
    from ufactory.dynamics.cli import cli_hardware_check

    ip = _require_xarm_ip()
    args = [
        "--robot",
        "xarm6",
        "--ip",
        ip,
        "--poses",
        "0,1,3",
        "--move-strategy",
        "direct",
        "--confirm-real",
        "--z-min-mm",
        "0",
    ]
    suffix = os.environ.get("XARM_KINEMATICS_SUFFIX")
    if suffix:
        args.extend(["--kinematics-suffix", suffix])
    rc = cli_hardware_check(args)
    assert rc == 0, f"cli_hardware_check failed (exit {rc})"


@pytest.mark.hardware
def test_fk_verify():
    ip = _require_xarm_ip()
    result = _run_example("examples/kinematics/verify_fk.py", ["--robot", "xarm6", "--ip", ip])
    assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.hardware
def test_ik_verify():
    ip = _require_xarm_ip()
    result = _run_example("examples/kinematics/verify_ik.py", ["--robot", "xarm6", "--ip", ip], timeout=900)
    assert result.returncode == 0, result.stderr[-2000:]
