"""Compatibility wrapper for xArm6 IK verification."""

from __future__ import annotations

import sys

import _bootstrap  # noqa: F401
from ufactory.kinematics_validation import cli_ik


if __name__ == "__main__":
    sys.exit(cli_ik(default_robot="xarm6", require_robot=False))
