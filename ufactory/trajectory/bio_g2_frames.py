"""Full flange/TCP transforms for the Bio G2 microplate simulation (metres, WXYZ)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from ufactory.kinematics.tcp_offset import matrix_to_rpy, rpy_matrix
from ufactory.robots.paths import robot_visual_glb_urdf
from ufactory.robots.runtime import RobotRuntimeProfile, get_robot_runtime_profile
from ufactory.simulation.bio_g2_adapter import origin_transform, quaternion_rotation
from ufactory.simulation.bio_g2_model import GRASP_OFFSET_IN_BASE_LINK

SUPPORTED_ROBOTS = ("xarm5", "xarm6", "xarm7", "uf850")
# Flange pose in the robot base frame; the scene translates the base by +400 mm Z.
HOME_FLANGE_XYZ_M = (0.204977625, -0.000524202, 0.197308)
HOME_FLANGE_RPY_DEG = (180.0, 0.0, -0.404582)
TRANSITION_DURATION_S = 0.4
HOME_POSITION_TOL_MM = 0.5
HOME_ROTATION_TOL_RAD = 1e-3


def rotation_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a normalized WXYZ quaternion."""
    roll, pitch, yaw = matrix_to_rpy(rotation)
    cr, cp, cy = np.cos(np.array([roll, pitch, yaw]) / 2)
    sr, sp, sy = np.sin(np.array([roll, pitch, yaw]) / 2)
    q = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )
    return q / np.linalg.norm(q)


def pose_matrix(xyz, quat) -> np.ndarray:
    xyz, quat = np.asarray(xyz, dtype=np.float64), np.asarray(quat, dtype=np.float64)
    if xyz.shape != (3,) or quat.shape != (4,) or not np.all(np.isfinite(np.r_[xyz, quat])):
        raise ValueError("pose must contain finite XYZ and WXYZ quaternion")
    if np.linalg.norm(quat) < 1e-12:
        raise ValueError("pose quaternion must have nonzero norm")
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = quaternion_rotation(quat), xyz
    return result


def pose_components(matrix) -> tuple[np.ndarray, np.ndarray]:
    return matrix[:3, 3].copy(), rotation_quaternion(matrix[:3, :3])


def rotation_error_rad(first, second) -> float:
    a, b = np.asarray(first), np.asarray(second)
    return float(2 * np.arccos(np.clip(abs(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))), 0, 1)))


def quaternion_slerp(first, second, progress: float) -> np.ndarray:
    """Interpolate the shortest quaternion arc, including equal and antipodal inputs."""
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    u = float(np.clip(progress, 0, 1))
    if dot > 0.9995:
        q = (1 - u) * a + u * b
    else:
        angle = math.acos(np.clip(dot, -1, 1))
        q = (np.sin((1 - u) * angle) * a + np.sin(u * angle) * b) / np.sin(angle)
    return q / np.linalg.norm(q)


def pose_report(matrix, frame: str) -> dict:
    xyz, quat = pose_components(matrix)
    return {
        "frame": frame,
        "xyz_mm": (xyz * 1000).tolist(),
        "quaternion_wxyz": quat.tolist(),
        "rpy_deg": np.degrees(matrix_to_rpy(matrix[:3, :3])).tolist(),
    }


@dataclass(frozen=True)
class BioG2TaskFrames:
    runtime: RobotRuntimeProfile
    urdf_path: Path
    flange_to_gripper: np.ndarray
    flange_to_tcp: np.ndarray

    @classmethod
    def load(cls, robot: str = "xarm6") -> BioG2TaskFrames:
        runtime = get_robot_runtime_profile(robot)
        if runtime.model.robot_name not in SUPPORTED_ROBOTS:
            raise ValueError(f"Bio G2 simulation robot must be one of {SUPPORTED_ROBOTS}")
        path = Path(robot_visual_glb_urdf(runtime.model.key, with_bio_gripper_g2=True, movable=True))
        root = ET.parse(path).getroot()
        attach = root.find("joint[@name='bio_gripper_g2_attach']")
        if (
            attach is None
            or attach.get("type") != "fixed"
            or attach.find("parent").get("link") != runtime.arm.ee_link
            or attach.find("child").get("link") != "bio_gripper_g2_base_link"
        ):
            raise ValueError("Bio G2 URDF must attach the gripper directly to the configured flange")
        mounting = origin_transform(attach.find("origin"))
        offset = np.eye(4)
        offset[:3, 3] = GRASP_OFFSET_IN_BASE_LINK
        return cls(runtime, path, mounting, mounting @ offset)

    @property
    def ik_seed(self) -> np.ndarray:
        if self.runtime.model.robot_name == "xarm7":
            return np.array([0, 0, 0, 0.5, -math.pi, -0.5, math.pi])
        return np.asarray(self.runtime.arm.default_qpos, dtype=np.float64)

    def home_flange(self, base_xyz=(0, 0, 0)) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(HOME_FLANGE_XYZ_M) + base_xyz
        pose[:3, :3] = rpy_matrix(*np.radians(HOME_FLANGE_RPY_DEG))
        return pose

    def tcp_to_flange(self, tcp_pose: np.ndarray) -> np.ndarray:
        return tcp_pose @ np.linalg.inv(self.flange_to_tcp)

    def flange_to_tcp_pose(self, flange_pose: np.ndarray) -> np.ndarray:
        return flange_pose @ self.flange_to_tcp
