"""Thin compatibility wrapper for xArm6 hardware dynamics validation."""

from __future__ import annotations

import sys

import _bootstrap  # noqa: F401
from ufactory.dynamics_validation import cli_hardware_check


if __name__ == "__main__":
    sys.exit(cli_hardware_check())

