"""Collision and kinematics backends for the packaging task."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ufactory.manipulation.packaging.core import (
    OBJECT_RELOCATED_STAGES,
    packaging_layout,
    packaging_obstacles,
    validate_payload_box_clearance,
)
from ufactory.safety.adapters import PinocchioCollisionBackend, PinocchioKinematicsBackend
from ufactory.safety.adapters.pinocchio import StageAwareObjectCollisionBackend


def build_packaging_backends(config: Any, urdf: Path, *, calibrated_g2: bool = False) -> tuple[Any, Any]:
    if config.gripper is None:
        raise ValueError("packaging requires a configured gripper")
    layout = packaging_layout(config)
    validate_payload_box_clearance(layout, margin_m=float(config.safety.min_collision_distance_m))
    passive = {config.gripper.drive_joint: config.gripper.open_drive}
    kinematics = PinocchioKinematicsBackend(
        urdf,
        joint_names=config.robot.joint_names,
        ee_link=config.robot.ee_link,
        passive_joint_positions=passive,
    )
    collision = StageAwareObjectCollisionBackend(
        PinocchioCollisionBackend(
            urdf,
            joint_names=config.robot.joint_names,
            ee_link=config.robot.ee_link,
            passive_joint_positions=passive,
            adjacent_link_pairs=config.robot.adjacent_collision_pairs,
            excluded_geometry_pairs=config.safety.excluded_geometry_pairs,
            obstacles=packaging_obstacles(layout),
        ),
        spawn_center_m=layout.object_position_m,
        place_center_m=layout.target_position_m,
        relocated_stages=OBJECT_RELOCATED_STAGES,
    )
    if calibrated_g2 and config.gripper.adapter == "g2":
        from ufactory.simulation.g2_adapter import G2ContactCollisionBackend

        collision = G2ContactCollisionBackend(collision, object_width_m=layout.object_size_m[1])
    return kinematics, collision
