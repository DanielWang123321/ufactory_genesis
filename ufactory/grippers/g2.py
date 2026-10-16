"""G2 pad geometry and conversions between metres, joint radians and SDK millimetres."""

from __future__ import annotations

from dataclasses import dataclass
import math

GRIPPER_G2_CLOSED_GAP_MM = 0.0
GRIPPER_G2_OPEN_GAP_MM = 84.0
GRIPPER_G2_OPEN_GAP_M = GRIPPER_G2_OPEN_GAP_MM / 1000.0

GRIPPER_G2_SIM_OPEN_DRIVE = 0.057210812728026743
GRIPPER_G2_SIM_CLOSE_DRIVE = 0.86509464599328212
GRIPPER_G2_JOINT_VELOCITY_RAD_S = 2.608067454724


@dataclass(frozen=True)
class G2LinkageGeometry:
    """Parallel-pad linkage dimensions from the project URDF and collision mesh."""

    base_joint_y_m: float = 0.035
    finger_joint_y_m: float = 0.035465
    finger_joint_z_m: float = 0.042039
    pad_inner_y_m: float = -0.02600320242345333
    pad_center_z_m: float = 0.041003202088177204
    drive_lower_rad: float = GRIPPER_G2_SIM_OPEN_DRIVE
    drive_upper_rad: float = GRIPPER_G2_SIM_CLOSE_DRIVE

    @property
    def radius_m(self) -> float:
        return math.hypot(self.finger_joint_y_m, self.finger_joint_z_m)

    @property
    def phase_rad(self) -> float:
        return math.atan2(self.finger_joint_z_m, self.finger_joint_y_m)

    def pad_gap_m(self, drive_rad: float) -> float:
        q = float(drive_rad)
        return 2.0 * (
            self.base_joint_y_m
            + self.finger_joint_y_m * math.cos(q)
            - self.finger_joint_z_m * math.sin(q)
            + self.pad_inner_y_m
        )

    def gap_jacobian_m_per_rad(self, drive_rad: float) -> float:
        """Absolute derivative of the full pad gap with respect to joint angle."""
        q = float(drive_rad)
        return abs(2.0 * (self.finger_joint_y_m * math.sin(q) + self.finger_joint_z_m * math.cos(q)))

    def drive_for_gap_m(self, gap_m: float) -> float:
        gap = float(gap_m)
        if not math.isfinite(gap) or not self.pad_gap_m(self.drive_upper_rad) <= gap <= self.pad_gap_m(
            self.drive_lower_rad
        ):
            raise ValueError("calibration gap is outside the collision pads' physical range")
        cosine = (gap / 2.0 - self.base_joint_y_m - self.pad_inner_y_m) / self.radius_m
        q = math.acos(max(-1.0, min(1.0, cosine))) - self.phase_rad
        return max(self.drive_lower_rad, min(self.drive_upper_rad, q))


G2_LINKAGE_GEOMETRY = G2LinkageGeometry()


def clamp_gripper_g2_gap_m(gap_m: float) -> float:
    """Clamp a two-finger gap in metres to the Gripper G2 physical range."""
    return max(GRIPPER_G2_CLOSED_GAP_MM / 1000.0, min(GRIPPER_G2_OPEN_GAP_M, float(gap_m)))


def gripper_g2_gap_m_to_sim_drive(gap_m: float) -> float:
    """Map a clamped pad gap in metres to the actual ``drive_joint`` angle in radians."""
    return G2_LINKAGE_GEOMETRY.drive_for_gap_m(clamp_gripper_g2_gap_m(gap_m))


def gripper_g2_sim_drive_to_gap_m(drive: float) -> float:
    """Map the actual joint angle in radians to a clamped pad gap in metres."""
    clipped = max(GRIPPER_G2_SIM_OPEN_DRIVE, min(GRIPPER_G2_SIM_CLOSE_DRIVE, float(drive)))
    return clamp_gripper_g2_gap_m(G2_LINKAGE_GEOMETRY.pad_gap_m(clipped))


def gripper_g2_gap_m_to_sdk_pos_mm(gap_m: float) -> float:
    """Map a physical gap in metres to ``set_gripper_g2_position`` millimetres."""
    return clamp_gripper_g2_gap_m(gap_m) * 1000.0


def gripper_g2_sdk_pos_mm_to_gap_m(pos_mm: float) -> float:
    """Map ``get_gripper_g2_position`` millimetres to physical gap in metres."""
    return clamp_gripper_g2_gap_m(float(pos_mm) / 1000.0)
