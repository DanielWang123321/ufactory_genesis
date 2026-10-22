"""xArm6 + Bio Gripper G2 SBS 96-well microplate pick-and-place simulation and evaluation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np

from ufactory.robots.paths import (
    bio_gripper_g2_urdf,
    robot_visual_glb_urdf,
)
from ufactory.simulation.bio_g2_adapter import (
    BIO_G2_CONTACT_SOL_PARAMS as CONTACT_SOL_PARAMS,
    BIO_G2_DT as DT,
    BIO_G2_MIMIC_SOL_PARAMS as MIMIC_SOL_PARAMS,
    DOWN_QUAT_WXYZ,
    BioG2GenesisAdapter as GenesisAdapter,
    URDFGravity,
    base_link_xyz_to_grasp_xyz,
    collision_gripping_planes,
    grasp_xyz_to_base_link_xyz,
    quaternion_rotation,
    signed_plate_gaps_mm,
    to_numpy as numpy,
)
from ufactory.simulation.bio_g2_controller import (
    BioGripperG2Controller as GripperController,
    ServoParameters,
)
from ufactory.simulation.bio_g2_microplate import (
    PLATE_URDF,
    build_microplate,
    plate_size_m,
)
from ufactory.simulation.bio_g2_model import (
    LEFT_JOINT as LEFT,
    MAX_SPEED_MM_S,
    MIN_SPEED_MM_S,
    RIGHT_JOINT as RIGHT,
    bio_g2_speed_mm_s_from_percent,
    bounded,
    finger_force_n,
    sha256,
)

COMBO_MODEL_PATH = Path(robot_visual_glb_urdf("xarm6_1305", with_bio_gripper_g2=True, movable=True))
STANDALONE_BIO_G2_PATH = Path(bio_gripper_g2_urdf())

GLB_FINGER_VISUAL_Y_OFFSET_M = {
    "bio_gripper_g2_left_finger": 0.0069565922021865845,
    "bio_gripper_g2_right_finger": -0.008117303252220154,
}

ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
XARM6_KP = (3000.0, 3000.0, 2000.0, 2000.0, 1000.0, 1000.0)
XARM6_KV = (300.0, 300.0, 200.0, 200.0, 100.0, 100.0)
XARM6_EFFORT_LIMITS = (179.4445, 179.4445, 92.0, 81.558, 30.6, 30.6)
XARM6_DEFAULT_QPOS = (0.0, -0.5, 0.0, 0.0, 0.5, 0.0)

# Scene & workstation geometry (meters).
TABLE_TOP_Z = 0.40
ROBOT_BASE_POS = (0.0, 0.0, TABLE_TOP_Z)
NEST_SIZE_M = (0.070, 0.070, 0.030)
NEST_TOP_Z = TABLE_TOP_Z + NEST_SIZE_M[2]
REAGENT_KIT_SIZE_M = plate_size_m()
REAGENT_KIT_MASS_KG = 0.10
REAGENT_KIT_REST_Z = NEST_TOP_Z + REAGENT_KIT_SIZE_M[2] / 2.0

PICK_XY = (0.38, -0.12)
PLACE_XY = (0.38, 0.12)
HOME_GRASP_POS = (0.36, 0.0, 0.548)
TRANSPORT_Z = 0.523
TRANSPORT_PHASE_NAMES = ("lift", "transit", "place_descend")
RETURN_PHASE_NAMES = ("retreat_up", "return_home")
TRANSPORT_BLEND_M = 0.020
DEFAULT_TRANSPORT_SPEED_MM_S = 250.0
DEFAULT_PICK_PLACE_SPEED_MM_S = 200.0
GRIPPER_KIT_FRICTION = 0.8
TRANSPORT_RAMP_S = 0.40
DEFAULT_SUBSTEPS = 8
CONSTRAINT_TOLERANCE = 1e-6
CONSTRAINT_ITERATIONS = 100
IK_POS_TOL_M = 1e-5
IK_ROT_TOL_RAD = 1e-4
ALIGN_TIMEOUT_S = 0.40
ALIGN_GAP_DIFFERENCE_MM = 0.20
ALIGN_EE_ERROR_MM = 0.50
ALIGN_STABLE_STEPS = 3
GRIP_TIMEOUT_S = 1.60
GRIP_HOLD_DURATION_S = 0.10
GRIP_HOLD_STEPS = round(GRIP_HOLD_DURATION_S / DT)
GRIP_FORCE_THRESHOLD_RATIO = 0.80
GRIP_STEADY_SPEED_TOL_MM_S = 5.0
PICK_PLACE_MIMIC_SOL_PARAMS = [0.005, *MIMIC_SOL_PARAMS[1:]]


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    duration_s: float
    start_grasp_xyz: tuple[float, float, float]
    end_grasp_xyz: tuple[float, float, float]
    gripper_target_mm: float | None = None


def build_xarm6_combo_model() -> dict:
    """Return provenance metadata for the public xArm6 + Bio Gripper G2 movable GLB URDF."""
    root = ET.parse(COMBO_MODEL_PATH).getroot()
    arm_masses = {}
    gripper_masses = {}
    for link in root.findall("link"):
        mass_node = link.find("inertial/mass")
        if mass_node is None:
            continue
        name = link.get("name") or ""
        mass = float(mass_node.get("value"))
        if name.startswith("bio_gripper_g2_"):
            gripper_masses[name] = mass
        else:
            arm_masses[name] = mass
    attach_joint = root.find("joint[@name='bio_gripper_g2_attach']")
    attach_origin = attach_joint.find("origin") if attach_joint is not None else None
    return {
        "xarm6_bio_glb_urdf_sha256": sha256(COMBO_MODEL_PATH),
        "gripper_dynamics_urdf_sha256": sha256(STANDALONE_BIO_G2_PATH),
        "combo_urdf_sha256": sha256(COMBO_MODEL_PATH),
        "visual_format": "glb",
        "arm_mass_kg": sum(arm_masses.values()),
        "gripper_mass_kg": sum(gripper_masses.values()),
        "total_mass_kg": sum(arm_masses.values()) + sum(gripper_masses.values()),
        "attach_origin_xyz": attach_origin.get("xyz") if attach_origin is not None else "0 0 0",
        "attach_origin_rpy": attach_origin.get("rpy") if attach_origin is not None else "0 0 0",
        "glb_finger_visual_y_offset_m": GLB_FINGER_VISUAL_Y_OFFSET_M,
    }


def smooth_step(alpha: float) -> float:
    """Quintic smoothstep with zero first and second derivatives at endpoints."""
    u = max(0.0, min(1.0, float(alpha)))
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def bezier_positions(controls: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """Evaluate a line or a quintic Bezier segment at the supplied parameters."""
    u = np.asarray(parameters, dtype=np.float64).reshape(-1, 1)
    degree = len(controls) - 1
    return sum(
        math.comb(degree, idx) * (1.0 - u) ** (degree - idx) * u**idx * point for idx, point in enumerate(controls)
    )


def transport_path_segments(phases: tuple[PhaseSpec, ...]) -> tuple[np.ndarray, ...]:
    """Round the two transport corners, trimming 20 mm from each adjacent line."""
    if tuple(phase.name for phase in phases) != TRANSPORT_PHASE_NAMES:
        raise ValueError("Transport requires lift, transit, place_descend in sequence")
    points = np.asarray([phases[0].start_grasp_xyz, *(phase.end_grasp_xyz for phase in phases)])
    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    if np.any(lengths <= 2.0 * TRANSPORT_BLEND_M):
        raise ValueError("Transport legs must be longer than two corner trims")
    directions = deltas / lengths[:, None]
    segments = []
    start = points[0]
    for idx in (1, 2):
        corner = points[idx]
        incoming, outgoing = directions[idx - 1], directions[idx]
        entry = corner - TRANSPORT_BLEND_M * incoming
        exit_point = corner + TRANSPORT_BLEND_M * outgoing
        segments.append(np.array([start, entry]))
        segments.append(
            np.array(
                [
                    entry,
                    entry + 0.5 * TRANSPORT_BLEND_M * incoming,
                    corner,
                    corner,
                    exit_point - 0.5 * TRANSPORT_BLEND_M * outgoing,
                    exit_point,
                ]
            )
        )
        start = exit_point
    segments.append(np.array([start, points[-1]]))
    return tuple(segments)


def sample_transport_path(phases: tuple[PhaseSpec, ...], progress: np.ndarray) -> np.ndarray:
    """Sample the whole rounded path by normalized arc length, without corner stops."""
    tables = []
    offset = 0.0
    for controls in transport_path_segments(phases):
        u = np.linspace(0.0, 1.0, 1001 if len(controls) == 6 else 2)
        xyz = bezier_positions(controls, u)
        distances = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
        tables.append((controls, u, distances + offset))
        offset += distances[-1]
    requested = np.clip(np.asarray(progress, dtype=np.float64), 0.0, 1.0) * offset
    result = np.empty((len(requested), 3), dtype=np.float64)
    for controls, u, distances in tables:
        mask = (requested >= distances[0]) & (requested <= distances[-1])
        result[mask] = bezier_positions(controls, np.interp(requested[mask], distances, u))
    return result


def line_timing(length_m: float, speed_mm_s: float) -> dict:
    """Quintic velocity ramps and a constant-speed plateau over a straight segment."""
    if not math.isfinite(speed_mm_s) or speed_mm_s <= 0:
        raise ValueError("transport_speed_mm_s must be finite and positive")
    length = float(length_m)
    velocity = speed_mm_s / 1000.0
    ramp_s = min(TRANSPORT_RAMP_S, length / velocity)
    duration_s = math.ceil((length / velocity + ramp_s) / DT) * DT
    if length / velocity >= TRANSPORT_RAMP_S:
        ramp_s = duration_s - length / velocity
    else:
        ramp_s = duration_s / 2.0
        velocity = length / ramp_s
    times = np.arange(1, round(duration_s / DT) + 1) * DT

    def integral(u):
        return u**4 * (2.5 - 3.0 * u + u * u)

    distance = velocity * (times - ramp_s / 2.0)
    speeds = np.full(len(times), velocity)
    ramp_up, ramp_down = times < ramp_s, times > duration_s - ramp_s
    u = times[ramp_up] / ramp_s
    distance[ramp_up] = velocity * ramp_s * integral(u)
    speeds[ramp_up] = velocity * np.array([smooth_step(value) for value in u])
    u = np.maximum(0.0, (duration_s - times[ramp_down]) / ramp_s)
    distance[ramp_down] = length - velocity * ramp_s * integral(u)
    speeds[ramp_down] = velocity * np.array([smooth_step(value) for value in u])
    return {
        "duration_s": duration_s,
        "length_m": length,
        "ramp_s": ramp_s,
        "progress": np.clip(distance / length, 0.0, 1.0),
        "speeds_m_s": speeds,
        "cruise_speed_mm_s": velocity * 1000.0,
    }


def transport_timing(phases: tuple[PhaseSpec, ...], speed_mm_s: float) -> dict:
    """Quintic velocity ramps and a constant-speed plateau over the whole arc."""
    if not math.isfinite(speed_mm_s) or speed_mm_s <= 0:
        raise ValueError("transport_speed_mm_s must be finite and positive")
    lengths = []
    for controls in transport_path_segments(phases):
        points = bezier_positions(controls, np.linspace(0, 1, 1001 if len(controls) == 6 else 2))
        lengths.append(float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()))
    profile = line_timing(sum(lengths), speed_mm_s)
    distance = profile["progress"] * profile["length_m"]
    boundaries = [lengths[0] + lengths[1] / 2.0, sum(lengths[:3]) + lengths[3] / 2.0]
    profile["labels"] = np.searchsorted(boundaries, distance, side="right")
    return profile


def default_phases(transport_speed_mm_s: float = DEFAULT_TRANSPORT_SPEED_MM_S) -> tuple[PhaseSpec, ...]:
    pick_grasp = (PICK_XY[0], PICK_XY[1], REAGENT_KIT_REST_Z)
    pick_above = (PICK_XY[0], PICK_XY[1], TRANSPORT_Z)
    place_above = (PLACE_XY[0], PLACE_XY[1], TRANSPORT_Z)
    place_grasp = (PLACE_XY[0], PLACE_XY[1], REAGENT_KIT_REST_Z + 0.0005)
    place_retreat = (PLACE_XY[0], PLACE_XY[1], HOME_GRASP_POS[2])
    phases = (
        PhaseSpec("pregrasp", 0.90, HOME_GRASP_POS, pick_above, gripper_target_mm=150.0),
        PhaseSpec("descend", 0.70, pick_above, pick_grasp, gripper_target_mm=None),
        PhaseSpec("grasp_align", ALIGN_TIMEOUT_S, pick_grasp, pick_grasp, gripper_target_mm=None),
        PhaseSpec("grip", GRIP_TIMEOUT_S, pick_grasp, pick_grasp, gripper_target_mm=71.0),
        PhaseSpec("lift", 0.80, pick_grasp, pick_above, gripper_target_mm=None),
        PhaseSpec("transit", 1.20, pick_above, place_above, gripper_target_mm=None),
        PhaseSpec("place_descend", 0.70, place_above, place_grasp, gripper_target_mm=None),
        PhaseSpec("place_settle", 0.08, place_grasp, place_grasp, gripper_target_mm=None),
        PhaseSpec("release", 0.46, place_grasp, place_grasp, gripper_target_mm=150.0),
        PhaseSpec("retreat_up", 0.70, place_grasp, place_retreat, gripper_target_mm=None),
        PhaseSpec("return_home", 0.90, place_retreat, HOME_GRASP_POS, gripper_target_mm=None),
        PhaseSpec("home_settle", 0.30, HOME_GRASP_POS, HOME_GRASP_POS, gripper_target_mm=None),
    )
    block = tuple(p for p in phases if p.name in TRANSPORT_PHASE_NAMES)
    timing = transport_timing(block, transport_speed_mm_s)
    resolved = []
    for p in phases:
        if p.name in TRANSPORT_PHASE_NAMES:
            dur = float(np.count_nonzero(timing["labels"] == TRANSPORT_PHASE_NAMES.index(p.name))) * DT
            resolved.append(replace(p, duration_s=dur))
        elif p.name in RETURN_PHASE_NAMES:
            seg_len = float(np.linalg.norm(np.asarray(p.end_grasp_xyz) - np.asarray(p.start_grasp_xyz)))
            resolved.append(replace(p, duration_s=line_timing(seg_len, transport_speed_mm_s)["duration_s"]))
        else:
            resolved.append(p)
    return tuple(resolved)


class AlignmentError(RuntimeError):
    pass


class GripTimeoutError(RuntimeError):
    pass


class AlignmentGuard:
    """Require three consecutive measured aligned samples before closing."""

    def __init__(self) -> None:
        self.stable_steps = 0
        self.samples = 0
        self.last_row: dict | None = None

    @property
    def ready(self) -> bool:
        return self.stable_steps >= ALIGN_STABLE_STEPS

    def update(self, row: dict) -> None:
        self.last_row = row
        self.samples += 1
        values = [row["left_gap_mm"], row["right_gap_mm"], row["ee_error_mm"]]
        aligned = (
            np.all(np.isfinite(values))
            and min(values[:2]) >= 0
            and abs(values[0] - values[1]) <= ALIGN_GAP_DIFFERENCE_MM
            and values[2] <= ALIGN_EE_ERROR_MM
        )
        self.stable_steps = self.stable_steps + 1 if aligned else 0

    def require_ready(self) -> None:
        if not self.ready:
            row = self.last_row or {}
            raise AlignmentError(
                f"grasp_align timeout after {self.samples * DT:.2f}s: "
                f"left_gap={row.get('left_gap_mm', float('nan')):.3f}mm, "
                f"right_gap={row.get('right_gap_mm', float('nan')):.3f}mm, "
                f"ee_error={row.get('ee_error_mm', float('nan')):.3f}mm"
            )


class GripForceGuard:
    """Require bilateral contact force above threshold sustained for GRIP_HOLD_DURATION_S."""

    def __init__(self, expected_force_n: float, hold_duration_s: float = GRIP_HOLD_DURATION_S) -> None:
        self.expected_force_n = float(expected_force_n)
        self.threshold_n = GRIP_FORCE_THRESHOLD_RATIO * self.expected_force_n
        self.hold_steps_required = max(1, int(round(float(hold_duration_s) / DT)))
        self.stable_steps = 0
        self.samples = 0
        self.last_row: dict | None = None

    @property
    def ready(self) -> bool:
        return self.stable_steps >= self.hold_steps_required

    def update(self, row: dict) -> None:
        self.last_row = row
        self.samples += 1
        left_n = float(row["left_normal_n"])
        right_n = float(row["right_normal_n"])
        gap_speed = float(row.get("gap_speed_mm_s", row.get("speed_mm_s", 0.0)))
        clamped = (
            np.isfinite(left_n)
            and np.isfinite(right_n)
            and np.isfinite(gap_speed)
            and min(left_n, right_n) >= self.threshold_n
            and abs(gap_speed) <= GRIP_STEADY_SPEED_TOL_MM_S
        )
        self.stable_steps = self.stable_steps + 1 if clamped else 0

    def require_ready(self) -> None:
        if not self.ready:
            row = self.last_row or {}
            raise GripTimeoutError(
                f"grip timeout after {self.samples * DT:.2f}s: "
                f"left_normal={row.get('left_normal_n', float('nan')):.2f}N, "
                f"right_normal={row.get('right_normal_n', float('nan')):.2f}N, "
                f"threshold={self.threshold_n:.2f}N, "
                f"held={self.stable_steps * DT * 1000:.0f}/{self.hold_steps_required * DT * 1000:.0f}ms"
            )


class PickPlaceAdapter(GenesisAdapter):
    """Share measured joint state within a physics step, never across steps."""

    def __init__(self, entity, scene, parameters: ServoParameters = ServoParameters()) -> None:
        super().__init__(entity, parameters)
        self.scene = scene
        self._sample_step = None

    def initialize(self, q_m: float) -> None:
        super().initialize(q_m)
        self._sample_step = None

    def positions(self) -> np.ndarray:
        step = self.scene.sim.cur_step_global
        if self._sample_step != step:
            self._q = numpy(self.entity.get_dofs_position()).reshape(-1)
            self._v = numpy(self.entity.get_dofs_velocity(self.indices)).reshape(-1)
            self._sample_step = step
        return self._q

    def state(self) -> tuple[float, float, float, float]:
        q = self.positions()
        return float(q[self.right]), float(self._v[0]), float(q[self.left]), float(self._v[1])


class XArm6BioG2PickPlaceSimulation:
    """Full physics simulation of xArm6 + Bio Gripper G2 picking and placing an SBS 96-well microplate."""

    def __init__(
        self,
        backend: str = "gpu",
        headless: bool = True,
        parameters: ServoParameters = ServoParameters(),
        substeps: int = DEFAULT_SUBSTEPS,
        transport_speed_mm_s: float = DEFAULT_TRANSPORT_SPEED_MM_S,
    ) -> None:
        import importlib.metadata
        import genesis as gs

        if importlib.metadata.version("genesis-world") != "1.4.4":
            raise RuntimeError("This simulation requires genesis-world==1.4.4")
        if backend not in ("cpu", "gpu"):
            raise ValueError("backend must be cpu or gpu")
        if substeps not in (8, 32):
            raise ValueError("substeps must be 8 or 32")
        if not math.isfinite(transport_speed_mm_s) or transport_speed_mm_s <= 0:
            raise ValueError("transport_speed_mm_s must be finite and positive")
        if not PLATE_URDF.exists():
            build_microplate()

        if not gs._initialized:
            gs.init(
                backend=gs.cuda if backend == "gpu" else gs.cpu,
                precision="64",
                seed=0,
                logging_level="warning",
                performance_mode=True,
            )
        expected = gs.cuda if backend == "gpu" else gs.cpu
        if gs.backend != expected or (backend == "gpu" and gs.device.type != "cuda"):
            raise RuntimeError(f"Requested {backend}, got {gs.backend}/{gs.device}; refusing fallback")
        if np.dtype(gs.np_float) != np.dtype("float64"):
            raise RuntimeError("This simulation requires Genesis precision=64")

        self.gs, self.backend, self.headless, self.parameters = gs, backend, headless, parameters
        self.substeps, self.transport_speed_mm_s = substeps, transport_speed_mm_s
        self.gravity = URDFGravity(COMBO_MODEL_PATH, ARM_JOINT_NAMES)
        self.gripping_planes = collision_gripping_planes(COMBO_MODEL_PATH)
        self.max_ik_position_error_m = self.max_ik_rotation_error_rad = 0.0
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=DT, substeps=substeps, gravity=(0.0, 0.0, -9.81)),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                enable_adjacent_collision=False,
                constraint_timeconst=0.005,
                iterations=CONSTRAINT_ITERATIONS,
                tolerance=CONSTRAINT_TOLERANCE,
                friction_cone=gs.friction_cone.elliptic,
                impratio=100,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.95, -0.75, 0.92),
                camera_lookat=(0.28, 0.0, TABLE_TOP_Z + 0.08),
                camera_fov=40,
                res=(1280, 720),
                refresh_rate=30,
                realtime_factor=None,
            ),
            show_viewer=False,
        )

        self.scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(friction=0.5))
        self.table = self.scene.add_entity(
            gs.morphs.Box(size=(0.65, 0.70, TABLE_TOP_Z), pos=(0.25, 0.0, TABLE_TOP_Z / 2.0), fixed=True),
            material=gs.materials.Rigid(friction=0.6),
            surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.82, 0.84, 0.86))),
        )
        self.pick_nest = self.scene.add_entity(
            gs.morphs.Box(
                size=NEST_SIZE_M,
                pos=(PICK_XY[0], PICK_XY[1], TABLE_TOP_Z + NEST_SIZE_M[2] / 2.0),
                fixed=True,
            ),
            material=gs.materials.Rigid(friction=0.6),
            surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.25, 0.35, 0.48))),
        )
        self.place_nest = self.scene.add_entity(
            gs.morphs.Box(
                size=NEST_SIZE_M,
                pos=(PLACE_XY[0], PLACE_XY[1], TABLE_TOP_Z + NEST_SIZE_M[2] / 2.0),
                fixed=True,
            ),
            material=gs.materials.Rigid(friction=0.6),
            surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.20, 0.55, 0.38))),
        )
        from ufactory.visualization.glb import glb_pbr_surfaces, glb_view_surface

        with glb_pbr_surfaces():
            self.reagent_kit = self.scene.add_entity(
                gs.morphs.URDF(
                    file=str(PLATE_URDF),
                    pos=(PICK_XY[0], PICK_XY[1], REAGENT_KIT_REST_Z),
                    fixed=False,
                    default_armature=None,
                    merge_fixed_links=False,
                    decimate=False,
                    convexify=True,
                    decompose_robot_error_threshold=float("inf"),
                ),
                material=gs.materials.Rigid(friction=GRIPPER_KIT_FRICTION),
                surface=glb_view_surface(),
            )
            self.robot = self.scene.add_entity(
                gs.morphs.URDF(
                    file=str(COMBO_MODEL_PATH),
                    pos=ROBOT_BASE_POS,
                    fixed=True,
                    default_armature=None,
                    merge_fixed_links=False,
                    decimate=False,
                    convexify=True,
                    decompose_robot_error_threshold=float("inf"),
                ),
                material=gs.materials.Rigid(friction=GRIPPER_KIT_FRICTION),
                surface=glb_view_surface(),
            )

        for entity in (self.robot, self.reagent_kit, self.pick_nest, self.place_nest):
            for geom in entity.geoms:
                geom.set_sol_params(CONTACT_SOL_PARAMS)
        for equality in self.robot.equalities:
            equality.set_sol_params(PICK_PLACE_MIMIC_SOL_PARAMS)

        self.scene.build()

        self.arm_dof_idx = [self.robot.get_joint(name).dofs_idx_local[0] for name in ARM_JOINT_NAMES]
        self.robot.set_dofs_kp(XARM6_KP, self.arm_dof_idx)
        self.robot.set_dofs_kv(XARM6_KV, self.arm_dof_idx)
        self.robot.set_dofs_force_range(
            [-limit for limit in XARM6_EFFORT_LIMITS],
            list(XARM6_EFFORT_LIMITS),
            self.arm_dof_idx,
        )
        self.robot.set_dofs_armature([0.0] * len(self.arm_dof_idx), self.arm_dof_idx)

        self.adapter = PickPlaceAdapter(self.robot, self.scene, parameters)
        self.controller = GripperController(self.adapter, parameters)
        self.gripper_base_link = self.robot.get_link("bio_gripper_g2_base_link")
        self.left_link = self.robot.get_link("bio_gripper_g2_left_finger")
        self.right_link = self.robot.get_link("bio_gripper_g2_right_finger")
        self.gripper_link_indices = [
            link.idx_local for link in (self.gripper_base_link, self.left_link, self.right_link)
        ]
        self.gripper_poses()
        self.kit_link_idx = self.reagent_kit.links[0].idx
        self.nest_link_ids = {self.pick_nest.links[0].idx, self.place_nest.links[0].idx}
        self.time_s = 0.0
        self.last_arm_q = np.asarray(XARM6_DEFAULT_QPOS, dtype=np.float64)

    def solve_arm_ik(
        self, grasp_xyz: tuple[float, float, float] | np.ndarray, seed_q: np.ndarray | None = None
    ) -> np.ndarray:
        """Solve 6-DOF arm joint positions for the desired world grasp-center position."""
        target_base_xyz = grasp_xyz_to_base_link_xyz(grasp_xyz)
        init_qpos = np.zeros(self.robot.n_dofs, dtype=np.float64)
        arm_seed = self.last_arm_q if seed_q is None else np.asarray(seed_q, dtype=np.float64)
        init_qpos[self.arm_dof_idx] = arm_seed
        q_right = self.controller.reference_q_m
        init_qpos[self.adapter.indices] = [q_right, -q_right]
        q_result, pose_error = self.robot.inverse_kinematics(
            link=self.gripper_base_link,
            pos=target_base_xyz,
            quat=DOWN_QUAT_WXYZ,
            init_qpos=init_qpos,
            dofs_idx_local=self.arm_dof_idx,
            max_samples=20,
            max_solver_iters=50,
            damping=0.01,
            pos_tol=IK_POS_TOL_M,
            rot_tol=IK_ROT_TOL_RAD,
            return_error=True,
        )
        q_full = numpy(q_result).reshape(-1)
        error = numpy(pose_error).reshape(6)
        position_error, rotation_error = np.linalg.norm(error[:3]), np.linalg.norm(error[3:])
        if (
            not np.all(np.isfinite(q_full))
            or not np.all(np.isfinite(error))
            or position_error > IK_POS_TOL_M
            or rotation_error > IK_ROT_TOL_RAD
        ):
            raise RuntimeError(f"IK failed: position={position_error * 1000:.6f}mm, rotation={rotation_error:.6g}rad")
        self.max_ik_position_error_m = max(self.max_ik_position_error_m, float(position_error))
        self.max_ik_rotation_error_rad = max(self.max_ik_rotation_error_rad, float(rotation_error))
        arm_q = q_full[self.arm_dof_idx].astype(np.float64).copy()
        self.last_arm_q = arm_q
        return arm_q

    def compile_trajectory(self, phases: tuple[PhaseSpec, ...]) -> list[dict]:
        """Compile joint streams, treating the three transport phases as one motion."""
        ticks: list[dict] = []
        self.home_arm_q = self.solve_arm_ik(
            HOME_GRASP_POS, seed_q=np.asarray(XARM6_DEFAULT_QPOS, dtype=np.float64)
        ).copy()
        q_prev = self.home_arm_q.copy()
        phase_idx = 0
        while phase_idx < len(phases):
            block = phases[phase_idx : phase_idx + 1]
            if block[0].name == "lift":
                block = phases[phase_idx : phase_idx + 3]
                if tuple(phase.name for phase in block) != TRANSPORT_PHASE_NAMES:
                    raise ValueError("Transport requires lift, transit, place_descend in sequence")
            labels = [(phase, idx) for phase in block for idx in range(max(1, int(round(phase.duration_s / DT))))]
            n_steps = len(labels)
            progress = np.array([smooth_step((idx + 1) / n_steps) for idx in range(n_steps)])
            start_xyz = np.asarray(block[0].start_grasp_xyz, dtype=np.float64)
            end_xyz = np.asarray(block[-1].end_grasp_xyz, dtype=np.float64)
            timing = None
            if len(block) == 3:
                timing = transport_timing(block, self.transport_speed_mm_s)
                progress = timing["progress"]
                counts = [0, 0, 0]
                labels = []
                for label in timing["labels"]:
                    labels.append((block[label], counts[label]))
                    counts[label] += 1
                n_steps = len(labels)
                xyz_block = sample_transport_path(block, progress)
            elif block[0].name in RETURN_PHASE_NAMES:
                seg_len = float(np.linalg.norm(end_xyz - start_xyz))
                timing = line_timing(seg_len, self.transport_speed_mm_s)
                progress = timing["progress"]
                labels = [(block[0], idx) for idx in range(len(progress))]
                n_steps = len(labels)
                xyz_block = start_xyz + progress[:, None] * (end_xyz - start_xyz)
            else:
                xyz_block = start_xyz + progress[:, None] * (end_xyz - start_xyz)
            stationary = float(np.linalg.norm(end_xyz - start_xyz)) < 1e-9
            q_start = q_prev.copy()
            q_block = np.empty((n_steps, len(self.arm_dof_idx)), dtype=np.float64)
            for idx, xyz in enumerate(xyz_block):
                if not stationary:
                    q_prev = self.solve_arm_ik(xyz, seed_q=q_prev)
                q_block[idx] = q_prev
            if stationary or n_steps == 1:
                dq_block = np.zeros_like(q_block)
            else:
                dq_block = np.gradient(np.vstack((q_start, q_block)), DT, axis=0)[1:]
                dq_block[0] = 0.0
                dq_block[-1] = 0.0
            for idx, (phase, phase_step) in enumerate(labels):
                ticks.append(
                    {
                        "phase": phase.name,
                        "phase_step": phase_step,
                        "target_grasp_xyz": xyz_block[idx],
                        "target_arm_q": q_block[idx],
                        "target_arm_dq": dq_block[idx],
                        "gripper_target_mm": phase.gripper_target_mm if phase_step == 0 else None,
                        "target_speed_mm_s": float(timing["speeds_m_s"][idx] * 1000) if timing is not None else 0.0,
                    }
                )
            phase_idx += len(block)
        return ticks

    def prepare(self, phases: tuple[PhaseSpec, ...]) -> list[dict]:
        """Compile, reset to exact home and warm up before opening any window."""
        ticks = self.compile_trajectory(phases)
        self.reset(self.home_arm_q, initial_gap_mm=150.0)
        self.start_viewer()
        return ticks

    def start_viewer(self) -> None:
        """Wait for the initialized home frame to be presented before task motion."""
        if self.headless:
            return
        from ufactory.visualization.render_patch import is_frame_ready
        from ufactory.visualization.viewer import start_deferred_viewer

        start_deferred_viewer(self.scene)
        viewer = self.scene.visualizer.viewer
        deadline = time.monotonic() + 30.0
        while not is_frame_ready(viewer._pyrender_viewer):
            if not viewer.is_alive():
                raise RuntimeError("Viewer closed before presenting the initial frame")
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for the initial viewer frame")
            viewer.update(force=True)
            time.sleep(0.01)

    def reset(self, home_arm_q: np.ndarray, initial_gap_mm: float = 150.0) -> None:
        self.scene.reset()
        for entity in (self.robot, self.reagent_kit):
            for geom in entity.geoms:
                geom.set_friction(GRIPPER_KIT_FRICTION)
        self.reagent_kit.set_pos((PICK_XY[0], PICK_XY[1], REAGENT_KIT_REST_Z))
        self.reagent_kit.set_quat((1.0, 0.0, 0.0, 0.0))
        self.reagent_kit.set_dofs_velocity(np.zeros(6, dtype=np.float64))
        self.robot.set_dofs_position(home_arm_q, self.arm_dof_idx)
        self.robot.set_dofs_velocity(np.zeros(len(self.arm_dof_idx), dtype=np.float64), self.arm_dof_idx)
        self.controller.reset(initial_gap_mm)
        self.time_s = 0.0
        for _ in range(10):
            self.control_arm(home_arm_q, np.zeros(len(self.arm_dof_idx)))
            self.controller.apply(DT)
            self.scene.step(update_visualizer=False)

    def control_arm(self, target_q, target_dq):
        actual = self.adapter.positions()
        positions = dict(zip(ARM_JOINT_NAMES, actual[self.arm_dof_idx]))
        positions.update(zip((RIGHT, LEFT), actual[self.adapter.indices]))
        torque, _ = self.gravity.compute(positions)
        reference = np.asarray(target_q) + torque / np.asarray(XARM6_KP)
        self.robot.control_dofs_position_velocity(reference, target_dq, self.arm_dof_idx)
        return torque

    def gripper_poses(self):
        positions = numpy(self.robot.get_links_pos(self.gripper_link_indices)).reshape(3, 3)
        quaternions = numpy(self.robot.get_links_quat(self.gripper_link_indices)).reshape(3, 4)
        return positions, quaternions

    def contact_telemetry(self, base_quat) -> tuple[float, float, int]:
        """Read bilateral finger-kit normal forces and finger-nest contact count in one call."""
        contacts = self.robot.get_contacts(exclude_self_contact=True)
        la, lb = numpy(contacts["link_a"]).reshape(-1), numpy(contacts["link_b"]).reshape(-1)
        fa, fb = numpy(contacts["force_a"]).reshape(-1, 3), numpy(contacts["force_b"]).reshape(-1, 3)
        quat = np.asarray(base_quat).reshape(4)
        quat = quat / np.linalg.norm(quat)
        w, x, y, z = quat
        y_axis_world = np.array(
            [2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)],
            dtype=np.float64,
        )
        forces = []
        for link in (self.left_link, self.right_link):
            mask_a = (la == link.idx) & (lb == self.kit_link_idx)
            mask_b = (lb == link.idx) & (la == self.kit_link_idx)
            f_world = fa[mask_a].sum(axis=0) + fb[mask_b].sum(axis=0)
            proj = float(np.dot(f_world, y_axis_world))
            forces.append(proj * (-1.0 if link == self.left_link else 1.0))
        finger_ids = {self.left_link.idx, self.right_link.idx}
        nest_contacts = int(
            sum(
                (int(a) in finger_ids and int(b) in self.nest_link_ids)
                or (int(b) in finger_ids and int(a) in self.nest_link_ids)
                for a, b in zip(la, lb)
            )
        )
        return forces[0], forces[1], nest_contacts

    def step(self, tick: dict, speed_mm_s: float, force_percent: float) -> dict:
        compute_started = time.perf_counter()
        if tick["gripper_target_mm"] is not None:
            target_mm = float(tick["gripper_target_mm"])
            if target_mm > self.controller.position_mm:
                self.controller.pause()
                self.controller.command(
                    position_mm=target_mm,
                    speed_mm_s=speed_mm_s,
                    force_percent=force_percent,
                )
                self.controller.resume()
            else:
                self.controller.command(
                    position_mm=target_mm,
                    speed_mm_s=speed_mm_s,
                    force_percent=force_percent,
                )
        gravity_torque = self.control_arm(tick["target_arm_q"], tick["target_arm_dq"])
        self.controller.apply(DT)
        self.scene.step(update_visualizer=False)
        self.time_s += DT

        row = self.controller.telemetry()
        gripper_positions, gripper_quaternions = self.gripper_poses()
        base_pos, base_quat = gripper_positions[0], gripper_quaternions[0]
        left_n, right_n, nest_contacts = self.contact_telemetry(base_quat)
        right_drive, left_drive = self.adapter.drive_force()
        kit_pos = numpy(self.reagent_kit.get_pos()).reshape(3)
        actual_grasp = base_link_xyz_to_grasp_xyz(base_pos, base_quat)
        target_grasp = np.asarray(tick["target_grasp_xyz"], dtype=np.float64)
        ee_error_mm = float(np.linalg.norm(actual_grasp - target_grasp) * 1000.0)
        kit_quat = numpy(self.reagent_kit.get_quat()).reshape(4)
        finger_poses = list(zip(gripper_positions[1:], gripper_quaternions[1:]))
        left_gap, right_gap = signed_plate_gaps_mm(
            self.gripping_planes,
            finger_poses,
            kit_pos,
            kit_quat,
            np.asarray(REAGENT_KIT_SIZE_M),
        )
        relative = quaternion_rotation(base_quat).T @ (kit_pos - actual_grasp) * 1000.0

        row.update(
            time_s=round(self.time_s, 6),
            phase=tick["phase"],
            left_normal_n=left_n,
            right_normal_n=right_n,
            right_drive_n=right_drive,
            left_drive_n=left_drive,
            kit_x_mm=float(kit_pos[0] * 1000.0),
            kit_y_mm=float(kit_pos[1] * 1000.0),
            kit_z_mm=float(kit_pos[2] * 1000.0),
            grasp_x_mm=float(actual_grasp[0] * 1000.0),
            grasp_y_mm=float(actual_grasp[1] * 1000.0),
            grasp_z_mm=float(actual_grasp[2] * 1000.0),
            ee_error_mm=ee_error_mm,
            left_gap_mm=float(left_gap),
            right_gap_mm=float(right_gap),
            gap_difference_mm=float(abs(left_gap - right_gap)),
            relative_x_mm=float(relative[0]),
            relative_y_mm=float(relative[1]),
            relative_z_mm=float(relative[2]),
            target_grasp_x_mm=float(target_grasp[0] * 1000.0),
            target_grasp_y_mm=float(target_grasp[1] * 1000.0),
            target_grasp_z_mm=float(target_grasp[2] * 1000.0),
            target_speed_mm_s=tick["target_speed_mm_s"],
            nest_finger_contacts=nest_contacts,
            position_resets=self.adapter.position_resets,
        )
        row.update({f"gravity_joint{idx + 1}_nm": float(value) for idx, value in enumerate(gravity_torque)})
        if not self.headless:
            self.scene.visualizer.viewer.update()
        row["compute_ms"] = (time.perf_counter() - compute_started) * 1000.0
        return row


XArm6PickPlaceSimulation = XArm6BioG2PickPlaceSimulation


def execute_trajectory(sim, ticks, speed_mm_s: float, force_percent: float):
    """Enforce alignment before closing, sustain contact force before lifting, and pace playback."""
    guard = AlignmentGuard()
    grip_guard = GripForceGuard(finger_force_n(force_percent))
    next_frame_s = time.perf_counter()
    for tick in ticks:
        if tick["phase"] == "grasp_align" and guard.ready:
            continue
        if tick["phase"] == "grip":
            guard.require_ready()
            if grip_guard.ready:
                continue
        if tick["phase"] == "lift":
            grip_guard.require_ready()
        row = sim.step(tick, speed_mm_s=speed_mm_s, force_percent=force_percent)
        if tick["phase"] == "grasp_align":
            guard.update(row)
        elif tick["phase"] == "grip" and "left_normal_n" in row and "right_normal_n" in row:
            grip_guard.update(row)
        yield row
        if not sim.headless:
            next_frame_s += DT
            now = time.perf_counter()
            remaining = next_frame_s - now
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_frame_s = now
    guard.require_ready()


def evaluate_run(rows: list[dict], force_percent: float) -> dict:
    """Evaluate pick-and-place physical metrics and pass/fail checks."""
    expected_force_n = finger_force_n(force_percent)
    initial_kit_z_mm = REAGENT_KIT_REST_Z * 1000.0
    grip_rows = [r for r in rows if r["phase"] == "grip"]
    align_rows = [r for r in rows if r["phase"] == "grasp_align"]
    lift_rows = [r for r in rows if r["phase"] == "lift"]
    transit_rows = [r for r in rows if r["phase"] == "transit"]
    transport_rows = [r for r in rows if r["phase"] in TRANSPORT_PHASE_NAMES]
    place_settle_rows = [r for r in rows if r["phase"] == "place_settle"]
    release_rows = [r for r in rows if r["phase"] == "release"]
    retreat_rows = [r for r in rows if r["phase"] == "retreat_up"]
    final_row = rows[-1]

    grip_tail = grip_rows[-GRIP_HOLD_STEPS:] if len(grip_rows) >= GRIP_HOLD_STEPS else grip_rows
    mean_left_n = float(np.mean([r["left_normal_n"] for r in grip_tail]))
    mean_right_n = float(np.mean([r["right_normal_n"] for r in grip_tail]))
    max_lift_gain_mm = max(r["kit_z_mm"] for r in lift_rows + transit_rows) - initial_kit_z_mm
    relative_z_mm = [r["kit_z_mm"] - r["grasp_z_mm"] for r in transport_rows]
    transport_vertical_slip_mm = max(relative_z_mm) - min(relative_z_mm)
    close_start = np.array([align_rows[-1][f"kit_{axis}_mm"] for axis in "xy"])
    closing_push_mm = max(
        float(np.linalg.norm(np.array([r[f"kit_{axis}_mm"] for axis in "xy"]) - close_start)) for r in grip_rows
    )
    relative_xyz = np.array([[r[f"relative_{axis}_mm"] for axis in "xyz"] for r in transport_rows])
    transport_relative_displacement_mm = float(np.linalg.norm(relative_xyz - relative_xyz[0], axis=1).max())
    transport_relative_range_mm = np.ptp(relative_xyz, axis=0)
    aligned = align_rows[-ALIGN_STABLE_STEPS:]
    alignment_passed = len(aligned) == ALIGN_STABLE_STEPS and all(
        min(r["left_gap_mm"], r["right_gap_mm"]) >= 0
        and r["gap_difference_mm"] <= ALIGN_GAP_DIFFERENCE_MM
        and r["ee_error_mm"] <= ALIGN_EE_ERROR_MM
        for r in aligned
    )
    place_xy_error_mm = float(
        math.hypot(
            final_row["kit_x_mm"] - PLACE_XY[0] * 1000.0,
            final_row["kit_y_mm"] - PLACE_XY[1] * 1000.0,
        )
    )
    place_z_error_mm = abs(final_row["kit_z_mm"] - initial_kit_z_mm)
    pre_release_gap_mm = place_settle_rows[-1]["gap_mm"]
    release_first_step_gain_mm = release_rows[0]["gap_mm"] - pre_release_gap_mm
    post_release_force_n = max(abs(release_rows[-1]["left_normal_n"]), abs(release_rows[-1]["right_normal_n"]))
    retreat_z_gain_mm = retreat_rows[-1]["grasp_z_mm"] - release_rows[-1]["grasp_z_mm"]
    retreat_xy_drift_mm = float(
        math.hypot(
            retreat_rows[-1]["grasp_x_mm"] - PLACE_XY[0] * 1000.0,
            retreat_rows[-1]["grasp_y_mm"] - PLACE_XY[1] * 1000.0,
        )
    )
    home_return_error_mm = float(
        np.linalg.norm(
            np.array([final_row["grasp_x_mm"], final_row["grasp_y_mm"], final_row["grasp_z_mm"]])
            - np.asarray(HOME_GRASP_POS) * 1000.0
        )
    )
    total_nest_contacts = sum(int(r["nest_finger_contacts"]) for r in rows)
    resets_during_motion = final_row["position_resets"] - rows[0]["position_resets"]

    checks = {
        "preclose_alignment": alignment_passed,
        "closing_horizontal_push_le_0_30mm": closing_push_mm <= 0.30,
        "bilateral_grasp_force": (mean_left_n >= 0.7 * expected_force_n and mean_right_n >= 0.7 * expected_force_n),
        "lift_clearance_ge_70mm": max_lift_gain_mm >= 70.0,
        "transit_vertical_slip_le_3mm": transport_vertical_slip_mm <= 3.0,
        "place_xy_error_le_5mm": place_xy_error_mm <= 5.0,
        "place_z_error_le_2mm": place_z_error_mm <= 2.0,
        "immediate_release_response": release_first_step_gain_mm >= 0.5,
        "clean_release": post_release_force_n <= 0.05 and release_rows[-1]["gap_mm"] >= 145.0,
        "vertical_retreat_before_home": retreat_z_gain_mm >= 70.0 and retreat_xy_drift_mm <= 5.0,
        "return_home_le_5mm": home_return_error_mm <= 5.0,
        "zero_nest_finger_contacts": total_nest_contacts == 0,
        "no_joint_teleport_during_motion": resets_during_motion == 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "expected_single_finger_force_n": expected_force_n,
            "grip_steady_left_normal_n": mean_left_n,
            "grip_steady_right_normal_n": mean_right_n,
            "max_lift_gain_mm": max_lift_gain_mm,
            "transport_vertical_slip_mm": transport_vertical_slip_mm,
            "transport_relative_displacement_3d_mm": transport_relative_displacement_mm,
            "transport_relative_range_xyz_mm": transport_relative_range_mm.tolist(),
            "closing_horizontal_push_mm": closing_push_mm,
            "preclose_left_gap_mm": align_rows[-1]["left_gap_mm"],
            "preclose_right_gap_mm": align_rows[-1]["right_gap_mm"],
            "preclose_gap_difference_mm": align_rows[-1]["gap_difference_mm"],
            "preclose_ee_error_mm": align_rows[-1]["ee_error_mm"],
            "alignment_duration_s": len(align_rows) * DT,
            "grip_duration_s": len(grip_rows) * DT,
            "final_kit_xyz_mm": [final_row["kit_x_mm"], final_row["kit_y_mm"], final_row["kit_z_mm"]],
            "target_place_xyz_mm": [PLACE_XY[0] * 1000.0, PLACE_XY[1] * 1000.0, initial_kit_z_mm],
            "place_xy_error_mm": place_xy_error_mm,
            "place_z_error_mm": place_z_error_mm,
            "release_first_step_gain_mm": release_first_step_gain_mm,
            "post_release_normal_force_n": post_release_force_n,
            "retreat_z_gain_mm": retreat_z_gain_mm,
            "retreat_xy_drift_mm": retreat_xy_drift_mm,
            "home_return_error_mm": home_return_error_mm,
            "total_nest_finger_contacts": total_nest_contacts,
            "sample_count": len(rows),
        },
    }


def compare_substeps(first: dict, second: dict) -> dict:
    """Check numerical agreement of complete runs at 8 and 32 substeps."""
    if {first["substeps"], second["substeps"]} != {8, 32}:
        raise ValueError("Comparison requires substeps 8 and 32")
    for key in ("backend", "speed_mm_s", "transport_speed_mm_s", "force_percent"):
        if first[key] != second[key]:
            raise ValueError(f"Comparison settings differ: {key}")
    if first["microplate"]["sha256"] != second["microplate"]["sha256"]:
        raise ValueError("Comparison microplate assets differ")
    a, b = first["metrics"], second["metrics"]
    position_difference = float(np.linalg.norm(np.asarray(a["final_kit_xyz_mm"]) - b["final_kit_xyz_mm"]))
    force_difference = [
        abs(a[f"grip_steady_{side}_normal_n"] - b[f"grip_steady_{side}_normal_n"]) for side in ("left", "right")
    ]
    checks = {
        "both_complete_runs_pass": first["passed"] and second["passed"],
        "final_position_difference_le_0_50mm": position_difference <= 0.50,
        "steady_single_finger_force_difference_le_0_20n": max(force_difference) <= 0.20,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "position_difference_mm": position_difference,
        "force_difference_left_right_n": force_difference,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--headless", action="store_true", help="Run without opening the interactive viewer window")
    parser.add_argument("--speed-mm-s", type=float, default=DEFAULT_PICK_PLACE_SPEED_MM_S)
    parser.add_argument("--speed-percent", type=float, default=None)
    parser.add_argument("--transport-speed-mm-s", type=float, default=DEFAULT_TRANSPORT_SPEED_MM_S)
    parser.add_argument("--substeps", type=int, choices=(8, 32), default=DEFAULT_SUBSTEPS)
    parser.add_argument("--force-percent", type=float, default=50.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        if args.speed_percent is not None:
            args.speed_mm_s = bio_g2_speed_mm_s_from_percent(args.speed_percent)
        bounded(args.speed_mm_s, MIN_SPEED_MM_S, MAX_SPEED_MM_S, "speed_mm_s")
        if not math.isfinite(args.transport_speed_mm_s) or args.transport_speed_mm_s <= 0:
            raise ValueError("transport_speed_mm_s must be finite and positive")
        finger_force_n(args.force_percent)
    except ValueError as error:
        parser.error(str(error))
    return args


def run_bio_g2_pick_place(
    *,
    backend: str = "gpu",
    headless: bool = True,
    speed_mm_s: float = DEFAULT_PICK_PLACE_SPEED_MM_S,
    transport_speed_mm_s: float = DEFAULT_TRANSPORT_SPEED_MM_S,
    substeps: int = DEFAULT_SUBSTEPS,
    force_percent: float = 50.0,
    csv_path: Path | None = None,
    summary_path: Path | None = None,
) -> dict:
    """Execute the calibrated xArm6 + Bio Gripper G2 SBS 96-well microplate pick-and-place task."""
    bounded(speed_mm_s, MIN_SPEED_MM_S, MAX_SPEED_MM_S, "speed_mm_s")
    if not math.isfinite(transport_speed_mm_s) or transport_speed_mm_s <= 0:
        raise ValueError("transport_speed_mm_s must be finite and positive")
    finger_force_n(force_percent)

    started = time.perf_counter()
    manifest = build_xarm6_combo_model()
    plate_manifest = build_microplate()
    sim = XArm6BioG2PickPlaceSimulation(
        backend=backend,
        headless=headless,
        substeps=substeps,
        transport_speed_mm_s=transport_speed_mm_s,
    )
    phases = default_phases(transport_speed_mm_s)
    ticks = sim.prepare(phases)

    rows: list[dict] = []
    last_phase = None
    loop_started = time.perf_counter()
    failure = None
    stream = None
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        stream = csv_path.open("w", newline="")
    try:
        writer = None
        try:
            for row in execute_trajectory(sim, ticks, speed_mm_s, force_percent):
                if stream is not None:
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                rows.append(row)
                if row["phase"] != last_phase:
                    last_phase = row["phase"]
                    print(
                        f"[{row['time_s']:5.2f}s] phase={row['phase']:<14} "
                        f"gap={row['gap_mm']:6.2f}mm  "
                        f"plate=({row['kit_x_mm']:6.1f}, {row['kit_y_mm']:6.1f}, {row['kit_z_mm']:5.1f})mm  "
                        f"normal L/R={row['left_normal_n']:5.2f}/{row['right_normal_n']:5.2f}N",
                        flush=True,
                    )
        except (AlignmentError, GripTimeoutError) as error:
            failure = str(error)
            print(f"FAIL {failure}", flush=True)
    finally:
        if stream is not None:
            stream.close()

    loop_wall_s = time.perf_counter() - loop_started
    evaluation = (
        evaluate_run(rows, force_percent=force_percent)
        if failure is None
        else {
            "passed": False,
            "checks": {"preclose_alignment": False},
            "metrics": {"sample_count": len(rows)},
        }
    )
    summary = {
        "passed": evaluation["passed"],
        "backend": backend,
        "headless": headless,
        "speed_mm_s": speed_mm_s,
        "transport_speed_mm_s": transport_speed_mm_s,
        "substeps": substeps,
        "dt_s": DT,
        "constraint_timeconst_s": 0.005,
        "constraint_tolerance": CONSTRAINT_TOLERANCE,
        "constraint_max_iterations": CONSTRAINT_ITERATIONS,
        "contact_sol_params": list(CONTACT_SOL_PARAMS),
        "mimic_sol_params": list(PICK_PLACE_MIMIC_SOL_PARAMS),
        "viewer_refresh_hz": 30,
        "force_percent": force_percent,
        "elapsed_wall_s": time.perf_counter() - started,
        "loop_wall_s": loop_wall_s,
        "ms_per_step": (loop_wall_s * 1000.0) / max(len(rows), 1),
        "compute_ms_per_step": float(np.mean([r["compute_ms"] for r in rows])) if rows else None,
        "realtime_factor": len(rows) * DT / max(loop_wall_s, 1e-9),
        "failure": failure,
        "combo_model": manifest,
        "microplate": plate_manifest,
        "max_ik_position_error_mm": sim.max_ik_position_error_m * 1000.0,
        "max_ik_rotation_error_rad": sim.max_ik_rotation_error_rad,
        "servo_parameters": asdict(sim.parameters),
        "checks": evaluation["checks"],
        "metrics": evaluation["metrics"],
    }
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
