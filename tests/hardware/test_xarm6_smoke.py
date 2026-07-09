"""Smoke tests for xArm 6 Genesis simulation examples."""

from __future__ import annotations

import os
import pickle
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
        ("examples/xarm6/verify_xarm6.py", []),
        ("examples/xarm6/xarm6_grasp_place_traj.py", ["--headless", "--rate", "50"]),
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
    if script == "examples/xarm6/xarm6_grasp_place_train.py":
        cfgs_path = PROJECT_ROOT / "logs" / "xarm6-grasp-place-joint-g2" / "cfgs.pkl"
        metrics_path = PROJECT_ROOT / "logs" / "xarm6-grasp-place-joint-g2" / "metrics.csv"
        assert cfgs_path.exists()
        assert metrics_path.exists()
        with cfgs_path.open("rb") as f:
            env_cfg, _reward_cfg, _robot_cfg, _train_cfg = pickle.load(f)
        assert env_cfg["num_envs"] == 1
        assert env_cfg["num_obs"] == 30
        assert env_cfg["num_actions"] == 7
        assert env_cfg["action_scale"] == pytest.approx(0.01)
        assert env_cfg["max_joint_delta_rad"] == pytest.approx(0.01)


@pytest.mark.integration
def test_xarm6_dynamics_sim_cli():
    from ufactory.dynamics import cli_sim_check

    rc = cli_sim_check(["--robot", "xarm6", "--random-count", "5"])
    assert rc in {0, 1}, f"cli_sim_check unexpected exit {rc}"


@pytest.mark.integration
def test_xarm6_dynamics_hardware_dry_run_cli():
    from ufactory.dynamics import cli_hardware_check

    rc = cli_hardware_check(["--robot", "xarm6", "--dry-run"])
    assert rc == 0, f"cli_hardware_check --dry-run exited {rc}"


@pytest.mark.hardware
def test_dynamics_verify_real():
    from ufactory.dynamics import cli_hardware_check

    ip = _require_xarm_ip()
    args = [
        "--robot", "xarm6",
        "--ip", ip,
        "--poses", "0,1,3",
        "--move-strategy", "direct",
        "--z-min-mm", "0",
    ]
    suffix = os.environ.get("XARM_KINEMATICS_SUFFIX")
    if suffix:
        args.extend(["--kinematics-suffix", suffix])
    rc = cli_hardware_check(args)
    assert rc == 0, f"cli_hardware_check failed (exit {rc})"


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
