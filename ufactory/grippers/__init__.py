"""Gripper control and command-space conversion helpers."""

from ufactory.grippers.bio_g2 import BioGripperG2
from ufactory.grippers.g2 import (
    GRIPPER_G2_CLOSED_GAP_MM,
    GRIPPER_G2_OPEN_GAP_M,
    GRIPPER_G2_OPEN_GAP_MM,
    GRIPPER_G2_SIM_CLOSE_DRIVE,
    GRIPPER_G2_SIM_OPEN_DRIVE,
    clamp_gripper_g2_gap_m,
    gripper_g2_gap_m_to_sdk_pos_mm,
    gripper_g2_gap_m_to_sim_drive,
    gripper_g2_sdk_pos_mm_to_gap_m,
    gripper_g2_sim_drive_to_gap_m,
)

__all__ = [
    "BioGripperG2",
    "GRIPPER_G2_CLOSED_GAP_MM",
    "GRIPPER_G2_OPEN_GAP_M",
    "GRIPPER_G2_OPEN_GAP_MM",
    "GRIPPER_G2_SIM_CLOSE_DRIVE",
    "GRIPPER_G2_SIM_OPEN_DRIVE",
    "clamp_gripper_g2_gap_m",
    "gripper_g2_gap_m_to_sdk_pos_mm",
    "gripper_g2_gap_m_to_sim_drive",
    "gripper_g2_sdk_pos_mm_to_gap_m",
    "gripper_g2_sim_drive_to_gap_m",
]
