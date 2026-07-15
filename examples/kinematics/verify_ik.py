"""Generic IK verification: Genesis URDF vs xArm Python SDK."""

from __future__ import annotations

import sys

from ufactory.kinematics.validation import cli_ik


if __name__ == "__main__":
    sys.exit(cli_ik())
