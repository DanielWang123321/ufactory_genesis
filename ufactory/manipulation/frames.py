"""Small frame conversion helpers for task code."""

from __future__ import annotations


def world_to_base_pos(pos, base_pos):
    """Translate a world-frame xyz value into a robot-base-frame xyz value.

    The current xArm6 pick-place setup mounts the robot base without yaw or
    pitch/roll, so frame conversion is a pure translation.
    """
    return pos - base_pos


def base_to_world_pos(pos, base_pos):
    """Translate a robot-base-frame xyz value into world-frame xyz value."""
    return pos + base_pos
