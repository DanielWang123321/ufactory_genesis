"""Generic FK verification: Genesis URDF vs xArm Python SDK."""

from __future__ import annotations

import sys

from ufactory.kinematics.validation import cli_fk


if __name__ == "__main__":
    sys.exit(cli_fk())
