"""Scene construction for the trajectory grasp-place pipeline.

Robot-generic Genesis scene builder for the grasp-place trajectory pipeline
(originally written for xArm6; see :mod:`examples.xarm6.xarm6_grasp_place_demo`
for the scene/robot setup this mirrors). Returns a typed context (entities, dof
indices, PD gains, finger offsets, key heights) reused across xArm5/6/7, UF850
(Gripper G2) and Lite6 (parallel gripper). Single environment, robot base at
table height.
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
from ufactory.robots.paths import robot_visual_glb_urdf, robot_urdf
from ufactory.robots.registry import joint_names
from ufactory.robots.runtime import GripperControlParams, RobotRuntimeProfile, get_robot_runtime_profile

TABLE_HEIGHT = 0.4  # meters; robot base sits on the table surface
DEFAULT_ROBOT_BASE_POS = (0.30, 0.00, TABLE_HEIGHT)
OBJ_SIZE = (0.030, 0.030, 0.030)

# Object and target xy are expressed in the robot base frame. Defaults are
# tuned for the 700 mm-reach xArm5/6/7 and 850 mm-reach UF850; Lite6 (440 mm
# reach) uses smaller LITE6_* defaults (see below).
OBJ_XY = (0.30, 0.00)
PLACE_XY = (0.30, 0.30)
LIFT_Z = 0.30
HOME_Z = 0.30

# Lite6 has a 440 mm reach (vs. 700-850 mm for the other arms). The installed
# reversed Lite6 gripper has a 20-38 mm physical opening, so the default task
# uses a 30 mm cube and closer-in workspace coordinates.
LITE6_OBJ_SIZE = (0.030, 0.030, 0.030)
LITE6_OBJ_XY = (0.20, 0.00)
LITE6_PLACE_XY = (0.20, 0.15)
LITE6_LIFT_Z = 0.20
LITE6_HOME_Z = 0.20

# The G2 drive->gap model under-closes relative to the visible/collision pads:
# a 34.7 mm command yields an actual ~40 mm finger inner span for the default
# cube, matching the earlier xArm6 showcase calibration.
GRIPPER_G2_GAP_CALIBRATION_OFFSET_M = 0.0053
DEFAULT_GRIPPER_GRASP_GAP_M = OBJ_SIZE[1] - GRIPPER_G2_GAP_CALIBRATION_OFFSET_M

# A tiny preload gives the binary Lite6 gripper a stable sim grasp without
# visible finger penetration into the 30 mm cube.
LITE6_DEFAULT_GRIPPER_GRASP_GAP_M = LITE6_OBJ_SIZE[1] - 0.0002

# Finger geometry constants measured/URDF-derived (same as the demo). These
# describe the Gripper G2 pad geometry; Lite6 uses its own LITE6_* constants.
FINGER_PAD_BELOW_FC = 0.061
FINGER_CLOSE_DESCENT = 0.015
GRASP_TABLE_CLEARANCE = 0.010

# Lite6 parallel-jaw fingers: pad extends ~27 mm below the joint/finger-center
# plane (URDF mesh-derived). Keep the fingertip about 6 mm above the table so
# the 30 mm cube intersects the flat inner pad region instead of the tapered
# fingertip end.
LITE6_FINGER_PAD_BELOW_FC = 0.027
LITE6_FINGER_CLOSE_DESCENT = 0.0
LITE6_GRASP_TABLE_CLEARANCE = 0.006
# Descend push-down is handled by sim/mirror spawn-freeze; keep extra Z at zero so
# GLB finger visuals stay aligned with the cube at grasp height.
LITE6_GRASP_LINK6_Z_EXTRA_M = 0.0
# Lite6 place: lift the tool slightly before opening so binary/real fingers and
# collision STLs clear the block top (GLB visuals already show clearance).
LITE6_PLACE_RELEASE_STANDOFF_M = 0.012

# Empirical link6->finger-center z offset (gripper open, "down" orientation),
# measured once per gripper family by build_scene and stable across xArm5/6/7/
# UF850 (all Gripper G2) since the arm's ee_link and the G2 mount are identical
# geometry. Used by real-path dry-run height estimates (no sim scene needed).
FINGER_Z_OFFSET_G2 = 0.1011
FINGER_Z_OFFSET_LITE6 = 0.0543

RIGID_SOLVER_ITERATIONS = 100
RIGID_NOSLIP_ITERATIONS = 5
RIGID_CONSTRAINT_TIMECONST = 0.005
MIMIC_CONSTRAINT_SOL_PARAMS = (0.01, 0.1, 0.0001, 0.001, 0.001, 0.5, 2.0)

# Sim object tuning: a light, high-friction cube so the partial-close grip
# reliably holds it through lift/transit. Real gripper firmware grasp physics
# are out-of-scope v1.
OBJ_INERTIAL_MASS_KG = 0.01
OBJ_FRICTION = 2.5
LITE6_OBJ_INERTIAL_MASS_KG = 0.006


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
    robot_key: str = "xarm6_1305"
    gripper: GripperControlParams | None = None
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


def _robot_urdf_for_visual_model(runtime: RobotRuntimeProfile, visual_model: str) -> tuple[str, bool]:
    """Resolve the sim URDF for ``robot_key`` at the requested visual fidelity.

    ``glb`` (default): GLB visuals + STL collision, movable gripper included.
    ``stl``: legacy all-STL visuals + collision (xArm6 only; kept for geometry
    alignment checks against the original demo).
    """
    model = str(visual_model).strip().lower()
    profile = runtime.model
    gripper = runtime.gripper
    if model == "stl":
        if profile.key != "xarm6_1305":
            raise ValueError(f"visual_model='stl' is only supported for xarm6_1305, got {profile.key!r}")
        return robot_urdf(profile.key, "xarm6_with_gripper.urdf"), False
    if model != "glb":
        raise ValueError("visual_model must be 'glb' or 'stl'")
    if gripper is not None and gripper.family == "lite6":
        return robot_visual_glb_urdf(profile.key, with_lite6_gripper=True, movable=True), True
    if gripper is not None and gripper.family == "g2":
        return robot_visual_glb_urdf(profile.key, with_gripper_g2=True, movable=True), True
    raise ValueError(f"{profile.key} has no supported gripper for the trajectory scene")


def _robot_defaults(robot_key: str) -> dict:
    """Per-robot default scene geometry (object/place xy, key heights)."""
    if get_robot_runtime_profile(robot_key).model.key == "lite6":
        return {
            "base_pos": DEFAULT_ROBOT_BASE_POS,
            "obj_xy": LITE6_OBJ_XY,
            "place_xy": LITE6_PLACE_XY,
            "obj_size": LITE6_OBJ_SIZE,
            "lift_z": LITE6_LIFT_Z,
            "home_z": LITE6_HOME_Z,
            "grasp_gap_m": LITE6_DEFAULT_GRIPPER_GRASP_GAP_M,
        }
    return {
        "base_pos": DEFAULT_ROBOT_BASE_POS,
        "obj_xy": OBJ_XY,
        "place_xy": PLACE_XY,
        "obj_size": OBJ_SIZE,
        "lift_z": LIFT_Z,
        "home_z": HOME_Z,
        "grasp_gap_m": DEFAULT_GRIPPER_GRASP_GAP_M,
    }


def default_grasp_gap_m(robot_key: str) -> float:
    """Tuned default two-finger grasp gap (m) for ``robot_key``'s default object."""
    return _robot_defaults(get_robot_runtime_profile(robot_key).model.key)["grasp_gap_m"]


@dataclass
class DryHeights:
    """Off-sim height/geometry estimate for the real-path dry-run/mirror flow.

    Mirrors the physically-derived constants :func:`build_scene` measures at
    runtime (finger offset, key link6 heights) without needing a live Genesis
    scene. The real arm uses per-unit calibrated forward kinematics at deploy
    time, so these base-frame heights match the sim plan up to the per-arm
    kinematics offset.
    """

    finger_z_offset: float
    home_pos_base: list[float]
    obj_xy: tuple[float, float]
    place_xy: tuple[float, float]
    grasp_link6_z: float
    pre_grasp_link6_z: float
    lift_link6_z: float


def dry_heights(robot_key: str) -> DryHeights:
    """Build a :class:`DryHeights` for ``robot_key`` without a Genesis scene."""
    runtime = get_robot_runtime_profile(robot_key)
    gripper = runtime.gripper
    if gripper is None:
        raise ValueError(f"{runtime.model.key} has no supported gripper for the trajectory scene")
    defaults = _robot_defaults(runtime.model.key)
    obj_xy = defaults["obj_xy"]
    if gripper.family == "lite6":
        finger_z_offset = FINGER_Z_OFFSET_LITE6
        pad_below_fc = LITE6_FINGER_PAD_BELOW_FC
        close_descent = LITE6_FINGER_CLOSE_DESCENT
        table_clearance = LITE6_GRASP_TABLE_CLEARANCE
        grasp_extra = LITE6_GRASP_LINK6_Z_EXTRA_M
    else:
        finger_z_offset = FINGER_Z_OFFSET_G2
        pad_below_fc = FINGER_PAD_BELOW_FC
        close_descent = FINGER_CLOSE_DESCENT
        table_clearance = GRASP_TABLE_CLEARANCE
        grasp_extra = 0.0
    grasp_link6_z = table_clearance + close_descent + pad_below_fc + finger_z_offset + grasp_extra
    return DryHeights(
        finger_z_offset=finger_z_offset,
        home_pos_base=[obj_xy[0], obj_xy[1], defaults["home_z"]],
        obj_xy=tuple(obj_xy),
        place_xy=tuple(defaults["place_xy"]),
        grasp_link6_z=grasp_link6_z,
        pre_grasp_link6_z=grasp_link6_z + 0.10,
        lift_link6_z=defaults["lift_z"],
    )


def build_scene(
    *,
    robot_key: str = "xarm6",
    rate: float = 50.0,
    show_viewer: bool = False,
    substeps: int = 8,
    solver_iterations: int = RIGID_SOLVER_ITERATIONS,
    noslip_iterations: int = RIGID_NOSLIP_ITERATIONS,
    constraint_timeconst: float = RIGID_CONSTRAINT_TIMECONST,
    use_gjk_collision: bool | None = None,
    stiffen_gripper_mimic: bool = True,
    contact_sol_params: tuple[float, float, float, float, float, float, float] | None = None,
    base_pos: tuple[float, float, float] | None = None,
    visual_model: Literal["glb", "stl"] = "glb",
    obj_xy: tuple[float, float] | None = None,
    place_xy: tuple[float, float] | None = None,
    obj_size: tuple[float, float, float] | None = None,
    lift_z: float | None = None,
    home_z: float | None = None,
) -> TrajSceneContext:
    """Build the single-env Genesis scene and configure PD gains + finger offsets.

    ``robot_key`` selects the arm profile (``xarm5``/``xarm6``/``xarm7``/``lite6``/``uf850``);
    the gripper (Gripper G2 or Lite6 parallel gripper) is resolved from that
    robot's runtime profile. Any of ``obj_xy``/``place_xy``/``obj_size``/``lift_z``/``home_z``
    left as ``None`` fall back to the robot's tuned default (Lite6's smaller
    reach and gripper opening use different defaults than the other arms).
    """
    runtime = get_robot_runtime_profile(robot_key)
    profile = runtime.model
    gripper = runtime.gripper
    if gripper is None:
        raise ValueError(f"{profile.key} has no supported gripper for the trajectory scene")
    defaults = _robot_defaults(profile.key)

    robot_base = tuple(float(v) for v in (base_pos if base_pos is not None else defaults["base_pos"]))
    obj_xy = tuple(obj_xy) if obj_xy is not None else defaults["obj_xy"]
    place_xy = tuple(place_xy) if place_xy is not None else defaults["place_xy"]
    obj_size = tuple(obj_size) if obj_size is not None else defaults["obj_size"]
    lift_z = float(lift_z) if lift_z is not None else defaults["lift_z"]
    home_z = float(home_z) if home_z is not None else defaults["home_z"]

    robot_urdf_path, use_glb_visual = _robot_urdf_for_visual_model(runtime, visual_model)
    if use_glb_visual:
        enable_glb_pbr_surfaces()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=1)

    dt = 1.0 / float(rate)
    substep_dt = dt / int(substeps)
    constraint_timeconst = max(float(constraint_timeconst), 2.0 * substep_dt)
    camera_lookat = _base_to_world(robot_base, (0.30, 0.15, 0.10))
    camera_pos = (camera_lookat[0] + 0.90, camera_lookat[1] - 1.35, camera_lookat[2] + 0.60)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=substeps),
        rigid_options=gs.options.RigidOptions(
            dt=dt,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True,
            iterations=int(solver_iterations),
            noslip_iterations=int(noslip_iterations),
            constraint_timeconst=float(constraint_timeconst),
            use_gjk_collision=use_gjk_collision,
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
        file=robot_urdf_path,
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

    ee_link_name = profile.ee_link
    ik_link = robot.get_link(ee_link_name)
    left_finger = robot.get_link(gripper.finger_link_names[0])
    right_finger = robot.get_link(gripper.finger_link_names[1])

    if stiffen_gripper_mimic:
        _stiffen_gripper_mimic_constraints(robot)
    if contact_sol_params is not None:
        _set_contact_sol_params((obj, left_finger, right_finger), contact_sol_params)

    # Sim object tuning: light + high-friction cube so the grip reliably holds
    # it through lift/transit. Set before any stepping.
    obj_mass_kg = LITE6_OBJ_INERTIAL_MASS_KG if gripper.family == "lite6" else OBJ_INERTIAL_MASS_KG
    obj.set_friction(float(OBJ_FRICTION))
    obj.set_links_inertial_mass(
        torch.tensor([obj_mass_kg], device=gs.device, dtype=gs.tc_float),
    )

    jnames = joint_names(profile)
    arm_dof_idx = [robot.get_joint(n).dofs_idx_local[0] for n in jnames]
    gripper_dof_idx = [robot.get_joint(gripper.drive_joint).dofs_idx_local[0]]
    all_gripper_dof_idx = [robot.get_joint(n).dofs_idx_local[0] for n in gripper.all_joint_names]

    runtime_arm = runtime.arm
    robot.set_dofs_kp(
        torch.tensor(runtime_arm.kp, device=gs.device, dtype=gs.tc_float), arm_dof_idx,
    )
    robot.set_dofs_kv(
        torch.tensor(runtime_arm.kv, device=gs.device, dtype=gs.tc_float), arm_dof_idx,
    )
    robot.set_dofs_force_range(
        torch.tensor(runtime_arm.force_lower, device=gs.device, dtype=gs.tc_float),
        torch.tensor(runtime_arm.force_upper, device=gs.device, dtype=gs.tc_float),
        arm_dof_idx,
    )
    robot.set_dofs_kp(
        torch.tensor([gripper.kp], device=gs.device, dtype=gs.tc_float), gripper_dof_idx,
    )
    robot.set_dofs_kv(
        torch.tensor([gripper.kv], device=gs.device, dtype=gs.tc_float), gripper_dof_idx,
    )
    robot.set_dofs_force_range(
        torch.tensor([gripper.force_lower], device=gs.device, dtype=gs.tc_float),
        torch.tensor([gripper.force_upper], device=gs.device, dtype=gs.tc_float),
        gripper_dof_idx,
    )
    n_grip = len(all_gripper_dof_idx)
    robot.set_dofs_damping(
        torch.full((n_grip,), gripper.damping, device=gs.device, dtype=gs.tc_float),
        all_gripper_dof_idx,
    )
    robot.set_dofs_frictionloss(
        torch.full((n_grip,), gripper.frictionloss, device=gs.device, dtype=gs.tc_float), all_gripper_dof_idx,
    )

    dq = xyz_to_quat(
        torch.tensor([[math.pi, 0.0, 0.0]], device=gs.device, dtype=gs.tc_float),
        rpy=True,
        degrees=False,
    )
    home_pos_base = [obj_xy[0], obj_xy[1], home_z]
    home_pos = _base_to_world(robot_base, home_pos_base)
    home_link6_pos = torch.tensor([home_pos], device=gs.device, dtype=gs.tc_float)
    home_qpos_result = robot.inverse_kinematics(
        link=ik_link, pos=home_link6_pos, quat=dq, dofs_idx_local=arm_dof_idx,
    )
    init_qpos = torch.zeros(1, robot.n_dofs, device=gs.device, dtype=gs.tc_float)
    for i, idx in enumerate(arm_dof_idx):
        init_qpos[:, idx] = home_qpos_result[0, arm_dof_idx[i]]
    for idx in all_gripper_dof_idx:
        init_qpos[:, idx] = gripper.open_pos
    robot.set_qpos(init_qpos)
    scene.step()

    link6_pos = ik_link.get_pos()[0]
    lf_pos = left_finger.get_pos()[0]
    rf_pos = right_finger.get_pos()[0]
    fc_pos = (lf_pos + rf_pos) / 2
    finger_z_offset = (link6_pos[2] - fc_pos[2]).item()
    # Finger-span across the open gap: fingers separate mainly along y when
    # the gripper points down, so use the larger of |dy|/|dx| as the span.
    finger_y_gap = max(abs(lf_pos[1] - rf_pos[1]).item(), abs(lf_pos[0] - rf_pos[0]).item())

    if gripper.family == "lite6":
        pad_below_fc = LITE6_FINGER_PAD_BELOW_FC
        close_descent = LITE6_FINGER_CLOSE_DESCENT
        table_clearance = LITE6_GRASP_TABLE_CLEARANCE
        grasp_extra = LITE6_GRASP_LINK6_Z_EXTRA_M
    else:
        pad_below_fc = FINGER_PAD_BELOW_FC
        close_descent = FINGER_CLOSE_DESCENT
        table_clearance = GRASP_TABLE_CLEARANCE
        grasp_extra = 0.0

    grasp_link6_z = table_clearance + close_descent + pad_below_fc + finger_z_offset + grasp_extra
    pre_grasp_link6_z = grasp_link6_z + 0.10
    lift_link6_z = lift_z
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
        robot_key=profile.key,
        gripper=gripper,
        home_pos_base=home_pos_base,
        table_height=TABLE_HEIGHT,
        obj_xy=obj_xy,
        place_xy=place_xy,
        obj_size=obj_size,
    )


def drive_for_gap_m(gap_m: float, gripper: GripperControlParams) -> float:
    """Convert a physical two-finger gap (m) to the Genesis drive-DOF value.

    Interpolates linearly between ``gripper.close_pos``
    (gap=``closed_gap_m``) and ``gripper.open_pos``
    (gap=``open_gap_m``); works for both Gripper G2 (open=0.0 < close=0.85)
    and Lite6 (close=0.0 < open=0.0089) sign conventions.
    """
    closed_gap = float(getattr(gripper, "closed_gap_m", 0.0))
    open_gap = float(gripper.open_gap_m)
    span = open_gap - closed_gap
    if span <= 0.0:
        return float(gripper.close_pos)
    open_fraction = max(0.0, min(1.0, (float(gap_m) - closed_gap) / span))
    return gripper.close_pos + open_fraction * (gripper.open_pos - gripper.close_pos)


def _stiffen_gripper_mimic_constraints(robot) -> None:
    """Tighten G2 mimic equalities so contact loads do not shear the linkage."""
    sol_params = np.asarray(MIMIC_CONSTRAINT_SOL_PARAMS, dtype=np.float64)
    for eq in robot.equalities:
        if "finger" in eq.name or "knuckle" in eq.name:
            eq.set_sol_params(sol_params)


def _set_contact_sol_params(entities_or_links, sol_params) -> None:
    """Apply contact solver parameters to selected entities or links."""
    sol = np.asarray(sol_params, dtype=np.float64)
    for item in entities_or_links:
        links = getattr(item, "links", None)
        if links is None:
            links = (item,)
        for link in links:
            for geom in link.geoms:
                geom.set_sol_params(sol)
