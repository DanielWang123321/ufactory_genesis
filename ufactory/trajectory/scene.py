"""Scene construction for the trajectory grasp-place pipeline.

Mirrors :mod:`examples.xarm6.xarm6_grasp_place_demo` scene/robot setup but as a
reusable builder returning a typed context (entities, dof indices, PD gains,
finger offsets, key heights). Single environment, robot base at table height.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch

import genesis as gs
from genesis.utils.geom import xyz_to_quat
from ufactory.visualization.glb import enable_glb_pbr_surfaces, glb_view_surface
from ufactory.grippers.g2 import GRIPPER_G2_OPEN_GAP_M
from ufactory.robots.paths import robot_visual_glb_urdf, xarm6_urdf

TABLE_HEIGHT = 0.4  # meters; robot base sits on the table surface
DEFAULT_ROBOT_BASE_POS = (0.30, 0.00, TABLE_HEIGHT)
OBJ_SIZE = (0.04, 0.04, 0.04)

# Object and target xy are expressed in the robot base frame.
OBJ_XY = (0.30, 0.00)
PLACE_XY = (0.30, 0.30)
LIFT_Z = 0.30
HOME_Z = 0.30

XARM6_GRIPPER_STL_URDF = xarm6_urdf("xarm6_with_gripper.urdf")
XARM6_GRIPPER_GLB_URDF = robot_visual_glb_urdf("xarm6_1305", with_gripper_g2=True, movable=True)
XARM6_GRIPPER_URDF = XARM6_GRIPPER_GLB_URDF

GRIPPER_OPEN_DRIVE = 0.0
GRIPPER_CLOSE_DRIVE = 0.85
DEFAULT_GRIPPER_GRASP_GAP_M = 0.024

# Finger geometry constants measured/URDF-derived (same as the demo).
FINGER_PAD_BELOW_FC = 0.061
FINGER_CLOSE_DESCENT = 0.015
GRASP_TABLE_CLEARANCE = 0.010

ARM_KP = (3000.0, 3000.0, 2000.0, 2000.0, 1000.0, 1000.0)
ARM_KV = (300.0, 300.0, 200.0, 200.0, 100.0, 100.0)
ARM_FORCE = (50.0, 50.0, 32.0, 32.0, 32.0, 20.0)
# Gripper G2 gains are paired with DEFAULT_GRIPPER_GRASP_GAP_M. Commanding a
# full 0 mm close makes the angled G2 linkage keep driving after cube contact
# and eject the cube; a 24 mm target keeps enough PD error to clamp the 40 mm
# cube without forcing the fingers to chase the mechanical hard close.
GRIPPER_KP = 20.0
GRIPPER_KV = 10.0
GRIPPER_FORCE = 5.0
GRIPPER_DAMPING = 0.1

# Sim object tuning: a light, high-friction cube so the partial-close G2 grip
# reliably holds it through lift/transit. Real G2 firmware grasp is out-of-scope v1.
OBJ_INERTIAL_MASS_KG = 0.03
OBJ_FRICTION = 2.5

ALL_GRIPPER_JOINTS = (
    "drive_joint",
    "left_finger_joint",
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_finger_joint",
    "right_inner_knuckle_joint",
)


@dataclass
class TrajSceneContext:
    scene: object
    robot: object
    obj: object
    target_marker: object
    ik_link: object
    left_finger: object
    right_finger: object
    arm_dof_idx: list[int]
    gripper_dof_idx: list[int]
    all_gripper_dof_idx: list[int]
    down_quat: torch.Tensor
    home_qpos: np.ndarray
    base_pos_world: tuple[float, float, float]
    visual_model: str
    finger_z_offset: float
    finger_y_gap: float
    # Key link6 heights in the **robot base frame** (z above the base / table).
    grasp_link6_z: float
    pre_grasp_link6_z: float
    lift_link6_z: float
    home_pos_base: list[float] = field(default_factory=list)
    table_height: float = TABLE_HEIGHT
    obj_xy: tuple[float, float] = OBJ_XY
    place_xy: tuple[float, float] = PLACE_XY
    obj_size: tuple[float, float, float] = OBJ_SIZE

    def base_to_world(self, pos_base) -> list[float]:
        """Convert a base-frame xyz (m) to the Genesis world frame (m)."""
        p = [float(pos_base[0]), float(pos_base[1]), float(pos_base[2])]
        return [p[0] + self.base_pos_world[0], p[1] + self.base_pos_world[1], p[2] + self.base_pos_world[2]]


def _base_to_world(base_pos_world: tuple[float, float, float], pos_base) -> list[float]:
    p = [float(pos_base[0]), float(pos_base[1]), float(pos_base[2])]
    return [p[0] + base_pos_world[0], p[1] + base_pos_world[1], p[2] + base_pos_world[2]]


def _robot_urdf_for_visual_model(visual_model: str) -> tuple[str, bool]:
    model = str(visual_model).strip().lower()
    if model == "glb":
        return XARM6_GRIPPER_GLB_URDF, True
    if model == "stl":
        return XARM6_GRIPPER_STL_URDF, False
    raise ValueError("visual_model must be 'glb' or 'stl'")


def build_scene(
    *,
    rate: float = 50.0,
    show_viewer: bool = False,
    substeps: int = 8,
    base_pos: tuple[float, float, float] | None = None,
    visual_model: Literal["glb", "stl"] = "glb",
    obj_xy: tuple[float, float] = OBJ_XY,
    place_xy: tuple[float, float] = PLACE_XY,
    obj_size: tuple[float, float, float] = OBJ_SIZE,
) -> TrajSceneContext:
    """Build the single-env Genesis scene and configure PD gains + finger offsets."""
    robot_base = tuple(float(v) for v in (base_pos if base_pos is not None else DEFAULT_ROBOT_BASE_POS))
    robot_urdf, use_glb_visual = _robot_urdf_for_visual_model(visual_model)
    if use_glb_visual:
        enable_glb_pbr_surfaces()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=1)

    dt = 1.0 / float(rate)
    camera_lookat = _base_to_world(robot_base, (0.30, 0.15, 0.10))
    camera_pos = (camera_lookat[0] + 0.90, camera_lookat[1] - 1.35, camera_lookat[2] + 0.60)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=substeps),
        rigid_options=gs.options.RigidOptions(
            dt=dt,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            refresh_rate=50,
            camera_pos=camera_pos,
            camera_lookat=tuple(camera_lookat),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    scene.add_entity(
        gs.morphs.Box(
            size=(0.5, 0.8, TABLE_HEIGHT),
            pos=(0.45, 0.0, TABLE_HEIGHT / 2),
            fixed=True,
        ),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ColorTexture(color=(0.6, 0.6, 0.6)),
        ),
    )

    robot_morph = gs.morphs.URDF(
        file=robot_urdf,
        pos=robot_base,
        fixed=True,
        requires_jac_and_IK=True,
    )
    if use_glb_visual:
        robot = scene.add_entity(robot_morph, surface=glb_view_surface())
    else:
        robot = scene.add_entity(robot_morph)

    obj_half_z = obj_size[2] / 2
    obj_pos_world = _base_to_world(robot_base, (obj_xy[0], obj_xy[1], obj_half_z))
    place_pos_world = _base_to_world(robot_base, (place_xy[0], place_xy[1], obj_half_z))
    obj = scene.add_entity(
        gs.morphs.Box(
            size=tuple(obj_size),
            pos=tuple(obj_pos_world),
            fixed=False,
        ),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ColorTexture(color=(0.9, 0.1, 0.1)),
        ),
    )
    target_marker = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.02,
            pos=tuple(place_pos_world),
            fixed=True,
            collision=False,
        ),
        surface=gs.surfaces.Rough(
            diffuse_texture=gs.textures.ColorTexture(color=(0.0, 1.0, 0.0)),
        ),
    )

    scene.build(n_envs=1)

    # Sim object tuning: light + high-friction cube so the G2 grip reliably
    # holds it through lift/transit (rubbery block). Set before any stepping.
    obj.set_friction(float(OBJ_FRICTION))
    obj.set_links_inertial_mass(
        torch.tensor([OBJ_INERTIAL_MASS_KG], device=gs.device, dtype=gs.tc_float),
    )

    ik_link = robot.get_link("link6")
    left_finger = robot.get_link("left_finger")
    right_finger = robot.get_link("right_finger")
    arm_dof_idx = [robot.get_joint(f"joint{i+1}").dofs_idx_local[0] for i in range(6)]
    gripper_dof_idx = [robot.get_joint("drive_joint").dofs_idx_local[0]]
    all_gripper_dof_idx = [robot.get_joint(n).dofs_idx_local[0] for n in ALL_GRIPPER_JOINTS]

    robot.set_dofs_kp(
        torch.tensor(ARM_KP, device=gs.device, dtype=gs.tc_float), arm_dof_idx,
    )
    robot.set_dofs_kv(
        torch.tensor(ARM_KV, device=gs.device, dtype=gs.tc_float), arm_dof_idx,
    )
    robot.set_dofs_force_range(
        torch.tensor([-f for f in ARM_FORCE], device=gs.device, dtype=gs.tc_float),
        torch.tensor(list(ARM_FORCE), device=gs.device, dtype=gs.tc_float),
        arm_dof_idx,
    )
    robot.set_dofs_kp(
        torch.tensor([GRIPPER_KP], device=gs.device, dtype=gs.tc_float), gripper_dof_idx,
    )
    robot.set_dofs_kv(
        torch.tensor([GRIPPER_KV], device=gs.device, dtype=gs.tc_float), gripper_dof_idx,
    )
    robot.set_dofs_force_range(
        torch.tensor([-GRIPPER_FORCE], device=gs.device, dtype=gs.tc_float),
        torch.tensor([GRIPPER_FORCE], device=gs.device, dtype=gs.tc_float),
        gripper_dof_idx,
    )
    n_grip = len(all_gripper_dof_idx)
    robot.set_dofs_damping(
        torch.full((n_grip,), GRIPPER_DAMPING, device=gs.device, dtype=gs.tc_float),
        all_gripper_dof_idx,
    )
    robot.set_dofs_frictionloss(
        torch.zeros(n_grip, device=gs.device, dtype=gs.tc_float), all_gripper_dof_idx,
    )

    dq = xyz_to_quat(
        torch.tensor([[math.pi, 0.0, 0.0]], device=gs.device, dtype=gs.tc_float),
        rpy=True,
        degrees=False,
    )
    home_pos_base = [0.3, 0.0, HOME_Z]
    home_pos = _base_to_world(robot_base, home_pos_base)
    home_link6_pos = torch.tensor([home_pos], device=gs.device, dtype=gs.tc_float)
    home_qpos_result = robot.inverse_kinematics(
        link=ik_link, pos=home_link6_pos, quat=dq, dofs_idx_local=arm_dof_idx,
    )
    init_qpos = torch.zeros(1, robot.n_dofs, device=gs.device, dtype=gs.tc_float)
    for i, idx in enumerate(arm_dof_idx):
        init_qpos[:, idx] = home_qpos_result[0, arm_dof_idx[i]]
    init_qpos[:, gripper_dof_idx[0]] = GRIPPER_OPEN_DRIVE
    robot.set_qpos(init_qpos)
    scene.step()

    link6_pos = ik_link.get_pos()[0]
    lf_pos = left_finger.get_pos()[0]
    rf_pos = right_finger.get_pos()[0]
    fc_pos = (lf_pos + rf_pos) / 2
    finger_z_offset = (link6_pos[2] - fc_pos[2]).item()
    # Finger-span across the open gap: the G2 fingers separate mainly along y
    # when the gripper points down, so use the larger of |dy|/|dx| as the span.
    finger_y_gap = max(abs(lf_pos[1] - rf_pos[1]).item(), abs(lf_pos[0] - rf_pos[0]).item())

    grasp_link6_z = (
        GRASP_TABLE_CLEARANCE + FINGER_CLOSE_DESCENT + FINGER_PAD_BELOW_FC + finger_z_offset
    )
    pre_grasp_link6_z = grasp_link6_z + 0.10
    lift_link6_z = LIFT_Z
    home_qpos_np = init_qpos[0].detach().cpu().numpy().astype(np.float64).copy()

    return TrajSceneContext(
        scene=scene,
        robot=robot,
        obj=obj,
        target_marker=target_marker,
        ik_link=ik_link,
        left_finger=left_finger,
        right_finger=right_finger,
        arm_dof_idx=arm_dof_idx,
        gripper_dof_idx=gripper_dof_idx,
        all_gripper_dof_idx=all_gripper_dof_idx,
        down_quat=dq,
        home_qpos=home_qpos_np,
        base_pos_world=robot_base,
        visual_model=str(visual_model).strip().lower(),
        finger_z_offset=finger_z_offset,
        finger_y_gap=finger_y_gap,
        grasp_link6_z=grasp_link6_z,
        pre_grasp_link6_z=pre_grasp_link6_z,
        lift_link6_z=lift_link6_z,
        home_pos_base=home_pos_base,
        table_height=TABLE_HEIGHT,
        obj_xy=obj_xy,
        place_xy=place_xy,
        obj_size=obj_size,
    )


def drive_for_gap_m(gap_m: float) -> float:
    """Convert physical two-finger gap (m) to the Genesis drive_joint value."""
    open_fraction = max(0.0, min(1.0, float(gap_m) / GRIPPER_G2_OPEN_GAP_M))
    return GRIPPER_CLOSE_DRIVE * (1.0 - open_fraction)
