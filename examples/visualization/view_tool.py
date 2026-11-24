"""Standalone viewer for supported UFACTORY flange tools (grippers and vacuum cups)."""

from __future__ import annotations

from ufactory.visualization.standalone_gripper import (
    MOUNTED_QUAT_WXYZ,
    STANDALONE_TOOL_SPECS,
    run_standalone_tool_viewer,
)

__all__ = ["MOUNTED_QUAT_WXYZ", "STANDALONE_TOOL_SPECS", "main"]


def main() -> dict | None:
    return run_standalone_tool_viewer()


if __name__ == "__main__":
    main()
