"""Verify Lite6 URDF in Genesis."""

import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

if __name__ == "__main__":
  script = Path(__file__).resolve().parents[1] / "verify_robot.py"
  raise SystemExit(subprocess.call([sys.executable, str(script), "--robot", "lite6", *sys.argv[1:]]))
