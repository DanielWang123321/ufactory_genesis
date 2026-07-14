"""Versioned xArm6 packaging task geometry and trajectory assembly."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

from ufactory.config import ResolvedRuntimeConfig, resolve_manipulation_object_spec
from ufactory.kinematics.orientation import GRIPPER_DOWN_QUAT_XYZW
from ufactory.safety.adapters import EnvironmentObstacle
from ufactory.trajectory.planner import CartesianWaypoint, JointWaypoint, TrajectoryPlannerConfig, plan_mixed_waypoints
from ufactory.trajectory.scene import (
    FINGER_CLOSE_DESCENT,
    FINGER_PAD_BELOW_FC,
    FINGER_Z_OFFSET_G2,
    GRASP_TABLE_CLEARANCE,
)

PACKAGING_TASK_NAME = "packaging_showcase"
OBJECT_RELOCATED_STAGES = ("post-release-settle", "return-transit", "return-home")


def _vec3(value: Any) -> tuple[float, float, float]:
    items = tuple(map(float, value))
    if len(items) != 3:
        raise ValueError("expected a three-value vector")
    return items[0], items[1], items[2]


def _vec2(value: Any) -> tuple[float, float]:
    items = tuple(map(float, value))
    if len(items) != 2:
        raise ValueError("expected a two-value vector")
    return items[0], items[1]


@dataclass(frozen=True)
class PackagingLayout:
    object_size_m: tuple[float, float, float]
    object_mass_kg: float
    object_position_m: tuple[float, float, float]
    target_position_m: tuple[float, float, float]
    home_position_m: tuple[float, float, float]
    table_size_m: tuple[float, float, float]
    table_center_m: tuple[float, float, float]
    box_outer_size_m: tuple[float, float, float]
    box_center_xy_m: tuple[float, float]
    box_wall_m: float
    box_floor_top_z_m: float
    pregrasp_clearance_m: float
    release_object_bottom_clearance_m: float
    grasp_gap_m: float
    grasp_settle_s: float
    pre_release_settle_s: float
    post_release_settle_s: float
    place_success_distance_m: float
    simulation_grasp_center_compensation_xy_m: tuple[float, float]
    simulation_arm_kp_scale: float

    @property
    def object_support_z_m(self) -> float:
        return self.object_position_m[2] - self.object_size_m[2] / 2.0

    @property
    def box_rim_z_m(self) -> float:
        return self.box_floor_top_z_m + self.box_outer_size_m[2]

    @property
    def box_inner_size_xy_m(self) -> tuple[float, float]:
        return (
            self.box_outer_size_m[0] - 2.0 * self.box_wall_m,
            self.box_outer_size_m[1] - 2.0 * self.box_wall_m,
        )

    @property
    def grasp_link6_z_m(self) -> float:
        return (
            self.object_support_z_m
            + GRASP_TABLE_CLEARANCE
            + FINGER_CLOSE_DESCENT
            + FINGER_PAD_BELOW_FC
            + FINGER_Z_OFFSET_G2
        )

    @property
    def release_link6_z_m(self) -> float:
        # Preserve the grasped object's nominal vertical offset from link6 so
        # the configured clearance is explicitly the cube bottom to box rim.
        link6_to_object_bottom = self.grasp_link6_z_m - self.object_support_z_m
        return self.box_rim_z_m + self.release_object_bottom_clearance_m + link6_to_object_bottom

    @property
    def release_object_center_m(self) -> tuple[float, float, float]:
        return (
            self.box_center_xy_m[0],
            self.box_center_xy_m[1],
            self.box_rim_z_m + self.release_object_bottom_clearance_m + self.object_size_m[2] / 2.0,
        )


def packaging_layout(config: ResolvedRuntimeConfig) -> PackagingLayout:
    if config.task.name != PACKAGING_TASK_NAME:
        raise ValueError(f"expected task {PACKAGING_TASK_NAME!r}, got {config.task.name!r}")
    p = config.task.parameters
    obj = resolve_manipulation_object_spec(config)
    sim_compensation = _vec2(p.get("simulation_grasp_center_compensation_xy_m", (0.0, 0.0)))
    sim_arm_kp_scale = float(p.get("simulation_arm_kp_scale", 1.0))
    if not all(np.isfinite(value) for value in sim_compensation):
        raise ValueError("simulation grasp-center compensation must be finite")
    if not np.isfinite(sim_arm_kp_scale) or sim_arm_kp_scale <= 0.0:
        raise ValueError("simulation arm kp scale must be finite and positive")
    return PackagingLayout(
        object_size_m=obj.size_m,
        object_mass_kg=obj.mass_kg,
        object_position_m=_vec3(p["fixed_object_position_m"]),
        target_position_m=_vec3(p["fixed_target_position_m"]),
        home_position_m=_vec3(p["default_ee_position_m"]),
        table_size_m=_vec3(p["table_size_m"]),
        table_center_m=_vec3(p["table_center_m"]),
        box_outer_size_m=_vec3(p["box_outer_size_m"]),
        box_center_xy_m=_vec2(p["box_center_xy_m"]),
        box_wall_m=float(p["box_wall_thickness_m"]),
        box_floor_top_z_m=float(p["box_floor_top_z_m"]),
        pregrasp_clearance_m=float(p["pregrasp_clearance_m"]),
        release_object_bottom_clearance_m=float(p["release_object_bottom_clearance_m"]),
        grasp_gap_m=float(p["grasp_gap_m"]),
        grasp_settle_s=float(p["grasp_settle_s"]),
        pre_release_settle_s=float(p["pre_release_settle_s"]),
        post_release_settle_s=float(p["post_release_settle_s"]),
        place_success_distance_m=float(p["place_success_distance_m"]),
        simulation_grasp_center_compensation_xy_m=sim_compensation,
        simulation_arm_kp_scale=sim_arm_kp_scale,
    )


def validate_payload_box_clearance(layout: PackagingLayout, *, margin_m: float = 0.0) -> None:
    inner_x, inner_y = layout.box_inner_size_xy_m
    obj_x, obj_y, _ = layout.object_size_m
    if obj_x + 2.0 * margin_m >= inner_x or obj_y + 2.0 * margin_m >= inner_y:
        raise ValueError("object plus safety margin does not fit through the configured box opening")
    dx = abs(layout.target_position_m[0] - layout.box_center_xy_m[0])
    dy = abs(layout.target_position_m[1] - layout.box_center_xy_m[1])
    if dx + obj_x / 2.0 + margin_m >= inner_x / 2.0:
        raise ValueError("target object X footprint is outside the box opening")
    if dy + obj_y / 2.0 + margin_m >= inner_y / 2.0:
        raise ValueError("target object Y footprint is outside the box opening")
    release_bottom = layout.release_object_center_m[2] - layout.object_size_m[2] / 2.0
    expected = layout.box_rim_z_m + layout.release_object_bottom_clearance_m
    if not np.isclose(release_bottom, expected, rtol=0.0, atol=1e-12):
        raise ValueError("release height does not preserve the configured object-bottom clearance")


def packaging_obstacles(layout: PackagingLayout) -> tuple[EnvironmentObstacle, ...]:
    sx, sy, sz = layout.box_outer_size_m
    cx, cy = layout.box_center_xy_m
    wall = layout.box_wall_m
    wall_center_z = layout.box_floor_top_z_m + sz / 2.0
    floor_center_z = layout.box_floor_top_z_m - wall / 2.0
    return (
        EnvironmentObstacle("table", layout.table_size_m, layout.table_center_m),
        EnvironmentObstacle("box_floor", (sx, sy, wall), (cx, cy, floor_center_z)),
        EnvironmentObstacle(
            "box_wall_x_min", (wall, sy - 2.0 * wall, sz), (cx - sx / 2.0 + wall / 2.0, cy, wall_center_z)
        ),
        EnvironmentObstacle(
            "box_wall_x_max", (wall, sy - 2.0 * wall, sz), (cx + sx / 2.0 - wall / 2.0, cy, wall_center_z)
        ),
        EnvironmentObstacle("box_wall_y_min", (sx, wall, sz), (cx, cy - sy / 2.0 + wall / 2.0, wall_center_z)),
        EnvironmentObstacle("box_wall_y_max", (sx, wall, sz), (cx, cy + sy / 2.0 - wall / 2.0, wall_center_z)),
        EnvironmentObstacle("object", layout.object_size_m, layout.object_position_m),
    )


def build_packaging_program(
    config: ResolvedRuntimeConfig,
    *,
    q_home: np.ndarray | None = None,
    kinematics: Any | None = None,
    cartesian_xy_offset_m: tuple[float, float] = (0.0, 0.0),
    place_xy_offset_m: tuple[float, float] | None = None,
) -> Any:
    layout = packaging_layout(config)
    validate_payload_box_clearance(layout, margin_m=float(config.safety.min_collision_distance_m))
    if config.gripper is None:
        raise ValueError("packaging showcase requires a configured gripper")

    offset = np.asarray(cartesian_xy_offset_m, dtype=np.float64).reshape(-1)
    if offset.shape != (2,) or not np.all(np.isfinite(offset)):
        raise ValueError("cartesian_xy_offset_m must contain two finite values")
    place_offset = offset if place_xy_offset_m is None else np.asarray(place_xy_offset_m, dtype=np.float64).reshape(-1)
    if place_offset.shape != (2,) or not np.all(np.isfinite(place_offset)):
        raise ValueError("place_xy_offset_m must contain two finite values")
    offset_x, offset_y = map(float, offset)
    place_offset_x, place_offset_y = map(float, place_offset)
    obj_x = layout.object_position_m[0] + offset_x
    obj_y = layout.object_position_m[1] + offset_y
    box_x = layout.box_center_xy_m[0] + place_offset_x
    box_y = layout.box_center_xy_m[1] + place_offset_y
    home = np.asarray(layout.home_position_m, dtype=np.float64)
    home[:2] += offset
    grasp_z = layout.grasp_link6_z_m
    release_z = layout.release_link6_z_m
    open_gap = float(config.gripper.open_gap_m)
    close_gap = layout.grasp_gap_m

    waypoints: list[Any] = [
        CartesianWaypoint([obj_x, obj_y, grasp_z + layout.pregrasp_clearance_m], label="home->pregrasp"),
        CartesianWaypoint([obj_x, obj_y, grasp_z], label="descend"),
        {
            "type": "gripper",
            "gap_start": open_gap,
            "gap_end": close_gap,
            "duration": config.motion.gripper_duration_s,
            "label": "grip",
        },
        {
            "type": "gripper",
            "gap_start": close_gap,
            "gap_end": close_gap,
            "duration": layout.grasp_settle_s,
            "label": "grip-settle",
        },
        CartesianWaypoint([obj_x, obj_y, release_z], label="lift"),
        CartesianWaypoint([box_x, box_y, release_z], label="transit"),
        {
            "type": "gripper",
            "gap_start": close_gap,
            "gap_end": close_gap,
            "duration": layout.pre_release_settle_s,
            "label": "pre-release-settle",
        },
        {
            "type": "gripper",
            "gap_start": close_gap,
            "gap_end": open_gap,
            "duration": config.motion.gripper_duration_s,
            "label": "release",
        },
        {
            "type": "gripper",
            "gap_start": open_gap,
            "gap_end": open_gap,
            "duration": layout.post_release_settle_s,
            "label": "post-release-settle",
        },
        CartesianWaypoint([home[0], home[1], release_z], label="return-transit"),
        CartesianWaypoint(home.tolist(), label="return-home"),
    ]

    start_q: list[float] | None = None
    if q_home is not None:
        start_q = np.asarray(q_home, dtype=np.float64).reshape(-1).tolist()
        waypoints.insert(0, JointWaypoint(start_q, label="start"))
    elif kinematics is not None:
        seed = np.asarray(config.arm.default_qpos_rad, dtype=np.float64)
        home_pose = np.concatenate((home, np.asarray(GRIPPER_DOWN_QUAT_XYZW, dtype=np.float64)))
        start_q = np.asarray(kinematics.inverse(home_pose, seed), dtype=np.float64).reshape(-1).tolist()
        waypoints.insert(0, JointWaypoint(start_q, label="start"))

    planner = TrajectoryPlannerConfig(
        robot_key=config.robot.key,
        rate=config.motion.rate_hz,
        speed_rad_s=config.motion.joint_speed_rad_s,
        mvacc_rad_s2=config.motion.joint_acceleration_rad_s2,
        linear_speed_m_s=config.motion.cartesian_speed_m_s,
        linear_acc_m_s2=config.motion.cartesian_acceleration_m_s2,
        z_min_m=config.safety.z_min_m,
        runtime_config=config,
    )
    return plan_mixed_waypoints(planner, waypoints, start_q=start_q, start_xyz=home.tolist())


def packaging_scene_sha256(config: ResolvedRuntimeConfig) -> str:
    layout = packaging_layout(config)
    payload = {
        "task": dict(config.task.parameters),
        "contacts": dict(config.task.allowed_contacts),
        "obstacles": [obstacle.__dict__ for obstacle in packaging_obstacles(layout)],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=list).encode()).hexdigest()
