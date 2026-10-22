"""Genesis integration, right-finger-only servo adapter, URDF gravity feedforward, and grasp geometry for Bio Gripper G2."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from ufactory.simulation.bio_g2_controller import (
    BioG2ControlOutput,
    BioGripperG2Controller,
    ServoParameters,
)
from ufactory.simulation.bio_g2_model import (
    BIO_G2_DT,
    GRASP_OFFSET_IN_BASE_LINK,
    LEFT_JOINT,
    RIGHT_JOINT,
    BioGripperG2Model,
    gap_to_q,
)

DT = BIO_G2_DT
# Genesis solver tuple: (timeconst, dampratio, dmin, dmax, width, mid, power)
BIO_G2_CONTACT_SOL_PARAMS: tuple[float, ...] = (0.005, 1.0, 0.99, 0.99, 0.001, 0.5, 2.0)
BIO_G2_MIMIC_SOL_PARAMS: tuple[float, ...] = (0.00125, 1.0, 0.9999, 0.9999, 0.001, 0.5, 2.0)
BIO_G2_PICK_PLACE_MIMIC_SOL_PARAMS: tuple[float, ...] = (0.005, 1.0, 0.9999, 0.9999, 0.001, 0.5, 2.0)

CONTACT_SOL_PARAMS = list(BIO_G2_CONTACT_SOL_PARAMS)
MIMIC_SOL_PARAMS = list(BIO_G2_MIMIC_SOL_PARAMS)
PICK_PLACE_MIMIC_SOL_PARAMS = list(BIO_G2_PICK_PLACE_MIMIC_SOL_PARAMS)

BIO_G2_MIMIC_EQUALITY_NAMES = frozenset(
    {
        "mimic_bio_gripper_g2_left_finger_joint_to_bio_gripper_g2_right_finger_joint",
    }
)

DOWN_QUAT_WXYZ = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


def configure_bio_g2_mimic_constraints(
    robot: Any,
    *,
    sol_params: tuple[float, ...] | list[float] = BIO_G2_MIMIC_SOL_PARAMS,
    contact_sol_params: tuple[float, ...] | list[float] | None = BIO_G2_CONTACT_SOL_PARAMS,
) -> int:
    """Configure mimic equality and optional gripper link contact solver parameters."""
    sol = np.asarray(sol_params, dtype=np.float64)
    count = 0
    for equality in getattr(robot, "equalities", ()):
        name = str(getattr(equality, "name", ""))
        if name in BIO_G2_MIMIC_EQUALITY_NAMES or "bio_gripper_g2" in name or not name:
            equality.set_sol_params(sol)
            count += 1
    if contact_sol_params is not None:
        c_sol = np.asarray(contact_sol_params, dtype=np.float64)
        for link in getattr(robot, "links", ()):
            if str(getattr(link, "name", "")).split("/")[-1].startswith("bio_gripper_g2_"):
                for geom in getattr(link, "geoms", ()):
                    geom.set_sol_params(c_sol)
    return count


def _resolve_joint(robot: Any, name: str) -> Any:
    joint_map = {j.name.split("/")[-1]: j for j in getattr(robot, "joints", ())}
    if name in joint_map:
        return joint_map[name]
    return robot.get_joint(name)


def configure_bio_g2_drive(
    robot: Any,
    parameters: ServoParameters = ServoParameters(),
    *,
    damping: float = 0.0,
    frictionloss: float = 0.0,
) -> tuple[list[int], list[int]]:
    """Configure right-finger-only actuation; left finger remains a passive mimic."""
    right_idx = int(_resolve_joint(robot, RIGHT_JOINT).dofs_idx_local[0])
    left_idx = int(_resolve_joint(robot, LEFT_JOINT).dofs_idx_local[0])
    indices = [right_idx, left_idx]
    robot.set_dofs_kp([parameters.kp_n_m], [right_idx])
    robot.set_dofs_kv([parameters.kv_n_s_m], [right_idx])
    robot.set_dofs_kp([0.0], [left_idx])
    robot.set_dofs_kv([0.0], [left_idx])
    robot.set_dofs_damping([damping, damping], indices)
    robot.set_dofs_frictionloss([frictionloss, frictionloss], indices)
    if hasattr(robot, "set_dofs_armature"):
        robot.set_dofs_armature([0.0, 0.0], indices)
    if hasattr(robot, "control_dofs_force"):
        robot.control_dofs_force([0.0], [left_idx])
    return [right_idx], indices


class BioG2GenesisAdapter:
    """Low-level Genesis adapter for Bio Gripper G2 right-finger servo."""

    def __init__(self, entity: Any, parameters: ServoParameters = ServoParameters()) -> None:
        self.entity, self.parameters = entity, parameters
        drive_idx, indices = configure_bio_g2_drive(entity, parameters)
        self.right = drive_idx[0]
        self.left = indices[1]
        self.indices = indices
        self.position_resets = 0
        self.last_servo = (gap_to_q(150.0), 0.0)
        self._fallback_q = (gap_to_q(150.0), -gap_to_q(150.0))

    def state(self) -> tuple[float, float, float, float]:
        if not hasattr(self.entity, "get_dofs_position"):
            return self._fallback_q[0], 0.0, self._fallback_q[1], 0.0
        q = to_numpy(self.entity.get_dofs_position(self.indices)).reshape(-1)
        v = to_numpy(self.entity.get_dofs_velocity(self.indices)).reshape(-1)
        return float(q[0]), float(v[0]), float(q[1]), float(v[1])

    def set_force_limit(self, limit_n: float) -> None:
        self.entity.set_dofs_force_range([-limit_n], [limit_n], [self.right])

    def servo(self, position_m: float, velocity_m_s: float) -> None:
        self.last_servo = (position_m, velocity_m_s)
        self._fallback_q = (float(position_m), -float(position_m))
        if hasattr(self.entity, "control_dofs_position_velocity"):
            self.entity.control_dofs_position_velocity([position_m], [velocity_m_s], [self.right])
        elif hasattr(self.entity, "control_dofs_position"):
            self.entity.control_dofs_position([position_m], [self.right])

    def initialize(self, q_m: float) -> None:
        self.position_resets += 1
        self._fallback_q = (float(q_m), -float(q_m))
        self.entity.set_dofs_position([q_m, -q_m], self.indices)
        if hasattr(self.entity, "set_dofs_velocity"):
            self.entity.set_dofs_velocity([0.0, 0.0], self.indices)

    def drive_force(self) -> tuple[float, float]:
        values = to_numpy(self.entity.get_dofs_control_force(self.indices)).reshape(-1)
        return float(values[0]), float(values[1])


GenesisAdapter = BioG2GenesisAdapter


class BioG2GenesisController:
    """High-level single-environment Bio Gripper G2 controller aligned with G2GenesisController."""

    def __init__(
        self,
        robot: Any,
        *,
        initial_position_mm: float = 150.0,
        parameters: ServoParameters = ServoParameters(),
    ) -> None:
        self.robot = robot
        self.model = BioGripperG2Model.load()
        self.adapter = BioG2GenesisAdapter(robot, parameters)
        self.controller = BioGripperG2Controller(self.adapter, parameters)
        self.drive_idx = [self.adapter.right]
        self.all_dof_idx = list(self.adapter.indices)
        self.output: BioG2ControlOutput | None = None
        self.reset(initial_position_mm)

    def gap_mm(self) -> float:
        return float(self.controller.telemetry()["gap_mm"])

    def reset(self, position_mm: float = 150.0) -> None:
        self.controller.reset(position_mm)
        self.output = None

    def command(
        self,
        position_mm: float,
        speed_percent: float | None = None,
        force_percent: float = 50.0,
        *,
        speed_mm_s: float | None = None,
        speed_rpm: float | None = None,
    ) -> None:
        self.controller.command(
            position_mm,
            speed_mm_s=speed_mm_s,
            force_percent=force_percent,
            speed_percent=speed_percent,
            speed_rpm=speed_rpm,
        )

    def pause(self) -> None:
        self.controller.pause()

    def resume(self) -> None:
        self.controller.resume()

    def apply(self, dt: float = BIO_G2_DT) -> BioG2ControlOutput:
        self.output = self.controller.apply(dt)
        return self.output

    def telemetry(self) -> dict:
        return self.controller.telemetry()


def quaternion_rotation(quat: Any) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def axis_rotation(axis: Any, angle: float) -> np.ndarray:
    vec = np.asarray(axis, dtype=np.float64)
    vec = vec / np.linalg.norm(vec)
    x, y, z = vec
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def origin_transform(node: ET.Element | None) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    if node is not None:
        result[:3, 3] = np.fromstring(node.get("xyz", "0 0 0"), sep=" ")
        roll, pitch, yaw = np.fromstring(node.get("rpy", "0 0 0"), sep=" ")
        result[:3, :3] = (
            axis_rotation([0, 0, 1], yaw) @ axis_rotation([0, 1, 0], pitch) @ axis_rotation([1, 0, 0], roll)
        )
    return result


class URDFGravity:
    """Sum COM Jacobian transposes times weight, returning holding torques."""

    def __init__(self, path: str | Path, controlled_joints: tuple[str, ...]) -> None:
        root = ET.parse(path).getroot()
        self.names = controlled_joints
        self.children: dict[str, list] = {}
        child_links = set()
        for joint in root.findall("joint"):
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            child_links.add(child)
            axis_node = joint.find("axis")
            axis = np.fromstring(axis_node.get("xyz"), sep=" ") if axis_node is not None else np.array([1.0, 0.0, 0.0])
            self.children.setdefault(parent, []).append(
                (
                    joint.get("name"),
                    child,
                    joint.get("type"),
                    origin_transform(joint.find("origin")),
                    axis,
                )
            )
        self.roots = [link.get("name") for link in root.findall("link") if link.get("name") not in child_links]
        self.inertials = {}
        for link in root.findall("link"):
            inertial = link.find("inertial")
            if inertial is not None:
                self.inertials[link.get("name")] = (
                    float(inertial.find("mass").get("value")),
                    origin_transform(inertial.find("origin"))[:3, 3],
                )

    def compute(self, positions: dict[str, float]) -> tuple[np.ndarray, float]:
        torque = dict.fromkeys(self.names, 0.0)
        potential = 0.0

        def visit(link: str, transform: np.ndarray, ancestors: list) -> None:
            nonlocal potential
            if link in self.inertials:
                mass, local_com = self.inertials[link]
                com = transform[:3, :3] @ local_com + transform[:3, 3]
                weight = np.array([0.0, 0.0, -9.81 * mass])
                potential += 9.81 * mass * com[2]
                for name, kind, point, axis in ancestors:
                    if name in torque:
                        jac = np.cross(axis, com - point) if kind in ("revolute", "continuous") else axis
                        torque[name] -= float(jac @ weight)
            for name, child, kind, origin, axis in self.children.get(link, []):
                at_joint = transform @ origin
                active = kind in ("revolute", "continuous", "prismatic")
                chain = ancestors
                if active:
                    chain = ancestors + [(name, kind, at_joint[:3, 3], at_joint[:3, :3] @ axis)]
                motion = np.eye(4)
                value = positions.get(name, 0.0)
                if kind in ("revolute", "continuous"):
                    motion[:3, :3] = axis_rotation(axis, value)
                elif kind == "prismatic":
                    motion[:3, 3] = axis * value
                visit(child, at_joint @ motion, chain)

        for link in self.roots:
            visit(link, np.eye(4), [])
        return np.array(list(torque.values()), dtype=np.float64), potential


def collision_gripping_planes(path: str | Path) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Read the retained planar rod faces in each finger link frame."""
    import trimesh

    urdf_path = Path(path)
    root = ET.parse(urdf_path).getroot()
    planes = []
    for side, direction in (("left", 1.0), ("right", -1.0)):
        collision = root.find(f"link[@name='bio_gripper_g2_{side}_finger']/collision")
        mesh = collision.find("geometry/mesh")
        vertices = trimesh.load((urdf_path.parent / mesh.get("filename")).resolve(), force="mesh").vertices.copy()
        vertices *= np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
        transform = origin_transform(collision.find("origin"))
        vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
        extreme = vertices[:, 1].max() if direction > 0 else vertices[:, 1].min()
        face = vertices[np.abs(vertices[:, 1] - extreme) < 1e-8]
        if len(face) < 3:
            raise ValueError(f"Missing planar collision gripping face: {side}")
        planes.append((face.mean(axis=0), np.array([0.0, direction, 0.0])))
    return tuple(planes)


def signed_plate_gaps_mm(
    planes: tuple[tuple[np.ndarray, np.ndarray], ...],
    finger_poses: list[tuple[np.ndarray, np.ndarray]] | tuple[tuple[np.ndarray, np.ndarray], ...],
    plate_pos: np.ndarray,
    plate_quat: np.ndarray,
    size: np.ndarray,
) -> np.ndarray:
    """Minimum signed distances from oriented box corners to inward finger planes."""
    corners = np.array([(x, y, z) for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)]) * np.asarray(
        size, dtype=np.float64
    )
    corners = corners @ quaternion_rotation(plate_quat).T + np.asarray(plate_pos, dtype=np.float64)
    gaps = []
    for (point, normal), (position, quat) in zip(planes, finger_poses):
        rotation = quaternion_rotation(quat)
        world_point = rotation @ point + position
        gaps.append(float(np.min((corners - world_point) @ (rotation @ normal))) * 1000.0)
    return np.asarray(gaps, dtype=np.float64)


def grasp_xyz_to_base_link_xyz(grasp_xyz: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    """Convert desired world grasp-center XYZ to bio_gripper_g2_base_link origin XYZ under DOWN_QUAT_WXYZ."""
    g = np.asarray(grasp_xyz, dtype=np.float64)
    offset = np.asarray(GRASP_OFFSET_IN_BASE_LINK, dtype=np.float64)
    world_offset = np.array([offset[0], -offset[1], -offset[2]], dtype=np.float64)
    return g - world_offset


def base_link_xyz_to_grasp_xyz(base_xyz: np.ndarray, base_quat_wxyz: np.ndarray) -> np.ndarray:
    """Compute actual world grasp-center XYZ from bio_gripper_g2_base_link pose."""
    rot = quaternion_rotation(base_quat_wxyz)
    return np.asarray(base_xyz, dtype=np.float64).reshape(3) + rot @ np.asarray(
        GRASP_OFFSET_IN_BASE_LINK, dtype=np.float64
    )
