"""Compatibility entry point for the xArm6 + Gripper G2 showcase.

The implementation is robot-generic as of v0.2.6.  Importers receive the
shared module so existing diagnostic imports and monkeypatches keep working;
direct execution delegates to the generic implementation with xArm6 selected.
"""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples import _packaging_showcase as _implementation  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(_implementation.main(["--robot", "xarm6", *sys.argv[1:]]))

# Preserve the old import surface as an alias of the shared implementation.
sys.modules[__name__] = _implementation
