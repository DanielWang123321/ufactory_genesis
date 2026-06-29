"""Generic IK verification: Genesis URDF vs xArm Python SDK."""

from __future__ import annotations

import sys

import _bootstrap  # noqa: F401
from ufactory.kinematics_validation import cli_ik


if __name__ == "__main__":
  sys.exit(cli_ik())
