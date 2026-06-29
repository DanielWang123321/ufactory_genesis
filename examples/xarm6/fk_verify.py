"""Compatibility wrapper for xArm6 FK verification."""

from __future__ import annotations

import sys

import _bootstrap  # noqa: F401
from ufactory.kinematics_validation import cli_fk


if __name__ == "__main__":
    sys.exit(cli_fk(default_robot="xarm6", require_robot=False))
