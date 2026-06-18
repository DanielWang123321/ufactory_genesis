"""G2-mounted fisheye camera extrinsics and framing helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as ScipyR

import constants

CAMERA_FOV_DEG = constants.CAMERA_FOV_DEG
CAMERA_HEIGHT_M = constants.CAMERA_HEIGHT_M
CAMERA_LATERAL_Y_M = constants.CAMERA_LATERAL_Y_M
CAMERA_MODEL = constants.CAMERA_MODEL
CAMERA_WIDTH = constants.CAMERA_WIDTH
CAMERA_HEIGHT = constants.CAMERA_HEIGHT
FINGER_BAND_Y_END = constants.FINGER_BAND_Y_END
FINGER_BAND_Y_START = constants.FINGER_BAND_Y_START

# Phase 0: Ry(180°) look-down on link6 + z along gripper axis (see plan §1.2).
R_DOWN_LINK6 = np.array(
    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    dtype=np.float64,
)


def rotation_look_down() -> np.ndarray:
    """Optical axis toward workspace (-world Z at home pose)."""
    return R_DOWN_LINK6.copy()


def make_offset_T(
    height_m: float = CAMERA_HEIGHT_M,
    lateral_x: float = 0.0,
    lateral_y: float = CAMERA_LATERAL_Y_M,
    roll_deg: float = 0.0,
) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    R = rotation_look_down()
    if abs(roll_deg) > 1e-6:
        R = R @ ScipyR.from_euler("z", roll_deg, degrees=True).as_matrix()
    T[:3, :3] = R
    T[:3, 3] = [lateral_x, lateral_y, height_m]
    return T


DEFAULT_CAMERA_OFFSET_T = make_offset_T()


@dataclass(frozen=True)
class CameraMountConfig:
    res: tuple[int, int] = (CAMERA_WIDTH, CAMERA_HEIGHT)
    model: str = CAMERA_MODEL
    fov: float = CAMERA_FOV_DEG
    offset_T: np.ndarray = field(default_factory=lambda: DEFAULT_CAMERA_OFFSET_T.copy())


def add_g2_wrist_camera(scene, mount: CameraMountConfig | None = None):
    """Create fisheye camera (not yet attached — call after scene.build)."""
    mount = mount or CameraMountConfig()
    return scene.add_camera(
        res=mount.res,
        model=mount.model,
        fov=mount.fov,
        GUI=False,
    )


def attach_camera_to_g2(robot, camera, mount: CameraMountConfig | None = None) -> str:
    """Attach camera to G2 / link6. Returns link name used."""
    mount = mount or CameraMountConfig()
    link_name = "link6"
    link = robot.get_link(link_name)
    camera.attach(link, mount.offset_T)
    return link_name


def project_world_points_to_image(
    world_pts: np.ndarray,
    cam_pos: np.ndarray,
    cam_forward: np.ndarray,
    cam_up: np.ndarray,
    fov_deg: float,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Equidistant fisheye projection for framing validation (y grows downward).
    Returns (N, 2) pixel coords; NaN if behind camera.
    """
    world_pts = np.asarray(world_pts, dtype=np.float64).reshape(-1, 3)
    cam_pos = np.asarray(cam_pos, dtype=np.float64).reshape(3)
    forward = np.asarray(cam_forward, dtype=np.float64).reshape(3)
    forward = forward / (np.linalg.norm(forward) + 1e-9)
    up = np.asarray(cam_up, dtype=np.float64).reshape(3)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-9)
    up_c = np.cross(right, forward)
    up_c = up_c / (np.linalg.norm(up_c) + 1e-9)

    rel = world_pts - cam_pos
    x_cam = rel @ right
    y_cam = rel @ up_c
    z_cam = rel @ forward
    pixels = np.full((len(world_pts), 2), np.nan, dtype=np.float64)
    valid = z_cam > 1e-4
    theta = np.arctan2(np.hypot(x_cam[valid], y_cam[valid]), z_cam[valid])
    phi = np.arctan2(y_cam[valid], x_cam[valid])
    max_theta = np.deg2rad(fov_deg) / 2.0
    r_max = min(width, height) / 2.0
    r = r_max * (theta / max_theta)
    pixels[valid, 0] = width / 2.0 + r * np.cos(phi)
    pixels[valid, 1] = height / 2.0 - r * np.sin(phi)
    return pixels


def fingers_in_bottom_band(pixel_y: np.ndarray) -> bool:
    y = np.asarray(pixel_y, dtype=np.float64)
    if np.any(np.isnan(y)):
        return False
    return bool(np.all((y >= FINGER_BAND_Y_START) & (y < FINGER_BAND_Y_END)))


def gripper_visible_in_bottom_band(rgb: np.ndarray, min_pixels: int = 80) -> bool:
    """Heuristic: gripper metal/white pixels in bottom 20% (complements 3D projection)."""
    band = np.asarray(rgb, dtype=np.uint8)[FINGER_BAND_Y_START:FINGER_BAND_Y_END, :, :]
    bright = band.astype(np.int16).sum(axis=2) > 380
    return int(bright.sum()) >= min_pixels


def check_finger_framing(
    env,
    offset_T: np.ndarray,
    *,
    min_band_pixels: int = 80,
) -> dict:
    """Project finger links + render heuristic for Phase 0 acceptance."""
    env.camera.attach(env.ik_link, offset_T)
    lf, rf = env.finger_world_positions()
    env.camera.move_to_attach()
    cam_pos = env.camera.get_pos().detach().cpu().numpy().reshape(3)
    look = env.camera.get_lookat().detach().cpu().numpy().reshape(3)
    up = env.camera.get_up().detach().cpu().numpy().reshape(3)
    forward = look - cam_pos
    forward /= np.linalg.norm(forward) + 1e-9
    pixels = project_world_points_to_image(
        np.stack([lf, rf]),
        cam_pos,
        forward,
        up,
        env.camera_mount.fov,
        env.camera_mount.res[0],
        env.camera_mount.res[1],
    )
    ly, ry = float(pixels[0, 1]), float(pixels[1, 1])
    rgb = env.render_wrist_rgb()
    render_ok = gripper_visible_in_bottom_band(rgb, min_pixels=min_band_pixels)
    proj_ok = fingers_in_bottom_band(np.array([ly, ry]))
    ok = proj_ok
    return {
        "left_y": ly,
        "right_y": ry,
        "band_start": FINGER_BAND_Y_START,
        "proj_ok": proj_ok,
        "render_ok": render_ok,
        "ok": ok,
    }
