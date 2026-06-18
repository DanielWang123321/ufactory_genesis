"""Genesis xArm6 + G2 IL environment with wrist fisheye camera."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

import genesis as gs
from genesis.utils.geom import xyz_to_quat
from ufactory.paths import xarm6_urdf

import constants
import camera_mount
from camera_mount import (
    CameraMountConfig,
    add_g2_wrist_camera,
    attach_camera_to_g2,
)
from xarm6_lerobot_features import gripper_openness_from_drive, pack_ee_state, quat_wxyz_to_rotvec

CTRL_DT = constants.CTRL_DT
OBJ_SIZE = constants.OBJ_SIZE
TABLE_HEIGHT = constants.TABLE_HEIGHT

XARM6_GRIPPER_URDF = xarm6_urdf("xarm6_with_gripper.urdf")


@dataclass
class SceneRandomization:
    """Domain randomization for scaled dataset recording."""

    obj_xy_range: tuple[tuple[float, float], tuple[float, float]] | None = None
    place_xy_range: tuple[tuple[float, float], tuple[float, float]] | None = None
    table_color: tuple[float, float, float] | None = None
    obj_color: tuple[float, float, float] | None = None
    seed: int | None = None


@dataclass
class XArm6G2ILEnv:
    table_height: float = TABLE_HEIGHT
    obj_size: tuple[float, float, float] = OBJ_SIZE
    obj_xy: tuple[float, float] = (0.30, 0.00)
    place_xy: tuple[float, float] = (0.30, 0.30)
    show_viewer: bool = False
    camera_mount: CameraMountConfig = field(default_factory=CameraMountConfig)

    scene: object = field(init=False, repr=False)
    robot: object = field(init=False, repr=False)
    obj: object = field(init=False, repr=False)
    camera: object = field(init=False, repr=False)
    ik_link: object = field(init=False, repr=False)
    left_finger: object = field(init=False, repr=False)
    right_finger: object = field(init=False, repr=False)
    arm_dof_idx: list[int] = field(init=False, repr=False)
    gripper_dof_idx: list[int] = field(init=False, repr=False)
    down_quat: torch.Tensor = field(init=False, repr=False)
    base_pos: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._build()

    def _build(self) -> None:
        self.scene = gs.Scene(
            show_viewer=self.show_viewer,
            sim_options=gs.options.SimOptions(dt=CTRL_DT, substeps=4),
            rigid_options=gs.options.RigidOptions(
                dt=CTRL_DT,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=60,
                camera_pos=(1.2, -1.2, 1.1),
                camera_lookat=(0.3, 0.15, self.table_height + 0.1),
                camera_fov=40,
            ),
        )
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        table_color = (0.6, 0.6, 0.6)
        self.scene.add_entity(
            gs.morphs.Box(
                size=(0.5, 0.8, self.table_height),
                pos=(0.45, 0.0, self.table_height / 2),
                fixed=True,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(color=table_color),
            ),
        )
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=XARM6_GRIPPER_URDF,
                pos=(0.0, 0.0, self.table_height),
                fixed=True,
                requires_jac_and_IK=True,
            ),
        )
        half_z = self.obj_size[2] / 2
        self.obj = self.scene.add_entity(
            gs.morphs.Box(
                size=tuple(self.obj_size),
                pos=(self.obj_xy[0], self.obj_xy[1], self.table_height + half_z),
                fixed=False,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(color=(0.9, 0.1, 0.1)),
            ),
        )
        self.scene.add_entity(
            gs.morphs.Sphere(
                radius=0.02,
                pos=(self.place_xy[0], self.place_xy[1], self.table_height + half_z),
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(color=(0.0, 1.0, 0.0)),
            ),
        )
        self.camera = add_g2_wrist_camera(self.scene, self.camera_mount)
        self.scene.build(n_envs=1)
        attach_camera_to_g2(self.robot, self.camera, self.camera_mount)

        self.ik_link = self.robot.get_link("link6")
        self.left_finger = self.robot.get_link("left_finger")
        self.right_finger = self.robot.get_link("right_finger")
        self.arm_dof_idx = [
            self.robot.get_joint(f"joint{i+1}").dofs_idx_local[0] for i in range(6)
        ]
        self.gripper_dof_idx = [
            self.robot.get_joint("drive_joint").dofs_idx_local[0],
        ]
        self._setup_pd()
        self.down_quat = xyz_to_quat(
            torch.tensor([[math.pi, 0.0, 0.0]], device=gs.device, dtype=gs.tc_float),
            rpy=True,
            degrees=False,
        )
        self.base_pos = np.array([0.0, 0.0, self.table_height], dtype=np.float64)

    def _setup_pd(self) -> None:
        robot = self.robot
        arm = self.arm_dof_idx
        grip = self.gripper_dof_idx
        robot.set_dofs_kp(
            torch.tensor([3000, 3000, 2000, 2000, 1000, 1000], device=gs.device, dtype=gs.tc_float),
            arm,
        )
        robot.set_dofs_kv(
            torch.tensor([300, 300, 200, 200, 100, 100], device=gs.device, dtype=gs.tc_float),
            arm,
        )
        robot.set_dofs_force_range(
            torch.tensor([-50, -50, -32, -32, -32, -20], device=gs.device, dtype=gs.tc_float),
            torch.tensor([50, 50, 32, 32, 32, 20], device=gs.device, dtype=gs.tc_float),
            arm,
        )
        robot.set_dofs_kp(torch.tensor([2.0], device=gs.device, dtype=gs.tc_float), grip)
        robot.set_dofs_kv(torch.tensor([3.0], device=gs.device, dtype=gs.tc_float), grip)
        robot.set_dofs_force_range(
            torch.tensor([-1.0], device=gs.device, dtype=gs.tc_float),
            torch.tensor([1.0], device=gs.device, dtype=gs.tc_float),
            grip,
        )
        all_gripper = [
            "drive_joint",
            "left_finger_joint",
            "left_inner_knuckle_joint",
            "right_outer_knuckle_joint",
            "right_finger_joint",
            "right_inner_knuckle_joint",
        ]
        all_idx = [robot.get_joint(n).dofs_idx_local[0] for n in all_gripper]
        n = len(all_idx)
        robot.set_dofs_damping(
            torch.full((n,), 0.1, device=gs.device, dtype=gs.tc_float),
            all_idx,
        )
        robot.set_dofs_frictionloss(
            torch.zeros(n, device=gs.device, dtype=gs.tc_float),
            all_idx,
        )

    def reset_object(self, obj_xy: tuple[float, float] | None = None) -> None:
        xy = obj_xy or self.obj_xy
        half_z = self.obj_size[2] / 2
        pos = torch.tensor(
            [[xy[0], xy[1], self.table_height + half_z]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.obj.set_pos(pos)

    def get_ee_state_base(self) -> np.ndarray:
        """7D EE in robot-base frame (link6 pose + gripper openness)."""
        pos_w = self.ik_link.get_pos()[0].detach().cpu().numpy()
        quat_w = self.ik_link.get_quat()[0].detach().cpu().numpy()
        pos_base = pos_w - self.base_pos
        rotvec = quat_wxyz_to_rotvec(quat_w)
        drive = self.robot.get_dofs_position(self.gripper_dof_idx)[0, 0].item()
        openness = gripper_openness_from_drive(drive)
        return pack_ee_state(pos_base, rotvec, openness)

    def render_wrist_rgb(self) -> np.ndarray:
        self.camera.move_to_attach()
        rgb, _, _, _ = self.camera.render()
        frame = rgb[0] if isinstance(rgb, np.ndarray) and rgb.ndim == 4 else rgb
        return np.asarray(frame, dtype=np.uint8)

    def finger_world_positions(self) -> tuple[np.ndarray, np.ndarray]:
        lf = self.left_finger.get_pos()[0].detach().cpu().numpy()
        rf = self.right_finger.get_pos()[0].detach().cpu().numpy()
        return lf, rf

    def apply_randomization(self, rnd: SceneRandomization) -> tuple[tuple[float, float], tuple[float, float]]:
        rng = np.random.default_rng(rnd.seed)
        if rnd.obj_xy_range:
            x_lo, x_hi = rnd.obj_xy_range[0]
            y_lo, y_hi = rnd.obj_xy_range[1]
            ox = (float(rng.uniform(x_lo, x_hi)), float(rng.uniform(y_lo, y_hi)))
        else:
            ox = self.obj_xy
        if rnd.place_xy_range:
            x_lo, x_hi = rnd.place_xy_range[0]
            y_lo, y_hi = rnd.place_xy_range[1]
            px = (float(rng.uniform(x_lo, x_hi)), float(rng.uniform(y_lo, y_hi)))
        else:
            px = self.place_xy
        self.obj_xy = ox
        self.place_xy = px
        self.reset_object(ox)
        return ox, px

    @staticmethod
    def default_randomization_ranges() -> SceneRandomization:
        return SceneRandomization(
            obj_xy_range=((0.28, 0.32), (-0.05, 0.05)),
            place_xy_range=((0.40, 0.55), (-0.10, 0.10)),
        )

    @staticmethod
    def sample_randomization(seed: int) -> SceneRandomization:
        rng = np.random.default_rng(seed)
        hue = float(rng.uniform(0.0, 1.0))
        import colorsys

        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        return SceneRandomization(
            obj_xy_range=((0.27, 0.33), (-0.08, 0.08)),
            place_xy_range=((0.38, 0.56), (-0.12, 0.12)),
            obj_color=(r, g, b),
            seed=seed,
        )
