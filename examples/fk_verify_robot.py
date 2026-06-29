"""Generic FK verification: Genesis URDF vs xArm Python SDK."""

from __future__ import annotations

import sys

import _bootstrap  # noqa: F401
from ufactory.kinematics_validation import cli_fk


if __name__ == "__main__":
  sys.exit(cli_fk())
