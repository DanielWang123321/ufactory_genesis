"""Smoke tests for multi-robot Genesis integration."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
NUMBA_CACHE_DIR = os.path.expanduser("~/.cache/numba")

ROBOTS = ["lite6", "uf850", "xarm5_1305", "xarm7_1305"]


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


@pytest.mark.parametrize("robot", ROBOTS)
def test_verify_robot_headless(robot: str):
  result = _run([PYTHON, "examples/verify_robot.py", "--robot", robot])
  assert result.returncode == 0, result.stderr[-3000:]


@pytest.mark.parametrize("robot", ROBOTS)
def test_view_robot_headless(robot: str):
  result = _run([PYTHON, "examples/view_robot_glb.py", "--robot", robot, "--headless"])
  assert result.returncode == 0, result.stderr[-3000:]
