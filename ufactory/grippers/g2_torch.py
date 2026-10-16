"""Tensor G2 kinematics using the scalar linkage geometry and joint angles."""

from __future__ import annotations

import torch

from ufactory.grippers.g2 import G2_LINKAGE_GEOMETRY, G2LinkageGeometry, GRIPPER_G2_OPEN_GAP_M


def drive_for_gap_m(gap_m: torch.Tensor, geometry: G2LinkageGeometry = G2_LINKAGE_GEOMETRY) -> torch.Tensor:
    g = geometry
    cosine = (gap_m.clamp(0, GRIPPER_G2_OPEN_GAP_M) / 2 - g.base_joint_y_m - g.pad_inner_y_m) / g.radius_m
    return (cosine.clamp(-1, 1).acos() - g.phase_rad).clamp(g.drive_lower_rad, g.drive_upper_rad)


def pad_gap_m(drive_rad: torch.Tensor, geometry: G2LinkageGeometry = G2_LINKAGE_GEOMETRY) -> torch.Tensor:
    g = geometry
    return 2 * (
        g.base_joint_y_m + g.finger_joint_y_m * drive_rad.cos() - g.finger_joint_z_m * drive_rad.sin() + g.pad_inner_y_m
    )
