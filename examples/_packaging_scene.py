"""Scene builders for xArm6 packaging showcase (yellow table, cardboard box)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import genesis as gs
from ufactory.glb_visual import glb_view_surface
from ufactory.paths import robot_visual_glb_urdf

REPO_ROOT = Path(__file__).resolve().parents[1]
TEXTURE_DIR = REPO_ROOT / "assets" / "scenes" / "packaging_showcase" / "textures"
MESH_DIR = REPO_ROOT / "assets" / "scenes" / "packaging_showcase" / "meshes"

# Layout (meters, world frame). Table X=1.2 m (long), Y=0.8 m (short).
DEFAULT_TABLE_TOP_Z = 0.75
TABLE_TOP_SIZE = (1.2, 0.8, 0.04)
TABLE_ORIGIN_X = 0.05  # tabletop min corner offset
LEG_SIZE = (0.06, 0.06, None)

# link_base collision mesh max XY radius ≈ 63 mm (assets/.../collision/link_base.obj)
ROBOT_BASE_RADIUS = 0.063
TRANSFER_CLEARANCE_ABOVE_BOX = 0.10  # cruise 100 mm above box top


def table_top_center() -> tuple[float, float, float]:
  sx, sy, sz = TABLE_TOP_SIZE
  return (TABLE_ORIGIN_X + sx / 2, 0.0, 0.0)


# Robot centered on negative-Y long edge; base circle tangent to table edge (center inset by radius).
def robot_xy_on_long_edge() -> tuple[float, float]:
  tcx, tcy, _ = table_top_center()
  _, top_sy, _ = TABLE_TOP_SIZE
  long_edge_y = tcy - top_sy / 2
  return (tcx, long_edge_y + ROBOT_BASE_RADIUS)


ROBOT_XY = robot_xy_on_long_edge()
# Base yaw (deg, extrinsic Z): rotate so robot +X faces the box (+Y in world).
ROBOT_BASE_YAW_DEG = 90.0
# TCP home in robot-base frame (mm / deg): [300, 0, 300, 180, 0, 0]
HOME_XY = (0.30, 0.0)
HOME_Z = 0.30
HOME_RPY_DEG = (180.0, 0.0, 0.0)

OBJ_SIZE = (0.03, 0.03, 0.03)  # matching real reference block: 30 mm, 17 g
OBJ_SPAWN_XY = (ROBOT_XY[0], 0.0)  # 0.1 m toward robot from prior y=0.10

BOX_OUTER = (0.28, 0.22, 0.16)
BOX_CENTER_XY = (ROBOT_XY[0], 0.28)
BOX_WALL = 0.003

# 黄褐色纸箱：颜色烘焙进 cardboard.jpg，image_color 保持 1.0 避免二次压暗
CARDBOARD_COLOR_SCALE = (1.0, 1.0, 1.0)

TABLE_TOP_COLOR = (0.95, 0.82, 0.15)
TABLE_LEG_COLOR = (0.78, 0.65, 0.12)
BLOCK_COLOR = (0.86, 0.14, 0.12)


def box_top_z(layout: PackagingLayout) -> float:
  """World Z of the cardboard box top rim (outer)."""
  return layout.table_top_z + layout.box_wall + layout.box_outer[2]


def packaging_camera(table_top_z: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
  """Viewer / headless camera aligned to long-edge robot workspace."""
  cx = (ROBOT_XY[0] + OBJ_SPAWN_XY[0] + BOX_CENTER_XY[0]) / 3
  cy = (ROBOT_XY[1] + OBJ_SPAWN_XY[1] + BOX_CENTER_XY[1]) / 3
  pos = (cx + 0.12, cy - 1.12, table_top_z + 0.55)
  lookat = (cx, cy + 0.04, table_top_z + 0.08)
  return pos, lookat


@dataclass(frozen=True)
class PackagingLayout:
  table_top_z: float
  robot_xy: tuple[float, float]
  obj_spawn_xy: tuple[float, float]
  obj_size: tuple[float, float, float]
  box_center_xy: tuple[float, float]
  box_outer: tuple[float, float, float]
  box_wall: float
  box_inner_floor_z: float
  place_xy: tuple[float, float]


def texture_path(name: str) -> str:
  path = TEXTURE_DIR / name
  if not path.exists():
    raise FileNotFoundError(
      f"Missing texture {path}. Run: python scripts/generate_showcase_textures.py"
    )
  return str(path.resolve())


def mesh_path(name: str) -> str:
  path = MESH_DIR / name
  if not path.exists():
    raise FileNotFoundError(
      f"Missing mesh {path}. Run: python scripts/generate_showcase_textures.py"
    )
  return str(path.resolve())


def add_textured_box(
  scene: gs.Scene,
  size: tuple[float, float, float],
  pos: tuple[float, float, float],
  texture_file: str,
  *,
  color_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
  fixed: bool = True,
  collision: bool = True,
) -> None:
  scene.add_entity(
    gs.morphs.Mesh(
      file=mesh_path("uv_box.obj"),
      scale=size,
      pos=pos,
      fixed=fixed,
      decimate=False,
      convexify=not fixed,
    ),
    surface=image_rough(texture_file, color_scale),
  )


def image_rough(
  filename: str,
  color_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> gs.surfaces.Rough:
  surf = gs.surfaces.Rough(
    diffuse_texture=gs.textures.ImageTexture(
      image_path=texture_path(filename),
      image_color=color_scale,
      encoding="srgb",
    )
  )
  return surf


def color_rough(color: tuple[float, float, float]) -> gs.surfaces.Rough:
  return gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=color))


def make_layout(table_top_z: float = DEFAULT_TABLE_TOP_Z) -> PackagingLayout:
  wall = BOX_WALL
  inner_floor_z = table_top_z + wall
  return PackagingLayout(
    table_top_z=table_top_z,
    robot_xy=ROBOT_XY,
    obj_spawn_xy=OBJ_SPAWN_XY,
    obj_size=OBJ_SIZE,
    box_center_xy=BOX_CENTER_XY,
    box_outer=BOX_OUTER,
    box_wall=wall,
    box_inner_floor_z=inner_floor_z,
    place_xy=BOX_CENTER_XY,
  )


def add_table(scene: gs.Scene, layout: PackagingLayout) -> None:
  top_z = layout.table_top_z
  top_sx, top_sy, top_sz = TABLE_TOP_SIZE
  leg_h = top_z - top_sz
  leg_sx, leg_sy, _ = LEG_SIZE
  tcx, tcy, _ = table_top_center()

  scene.add_entity(
    gs.morphs.Box(
      size=TABLE_TOP_SIZE,
      pos=(tcx, tcy, top_z - top_sz / 2),
      fixed=True,
    ),
    surface=color_rough(TABLE_TOP_COLOR),
  )

  leg_z = leg_h / 2
  corners = (
    (TABLE_ORIGIN_X + 0.03, tcy - top_sy / 2 + 0.06),
    (TABLE_ORIGIN_X + 0.03, tcy + top_sy / 2 - 0.06),
    (TABLE_ORIGIN_X + top_sx - 0.03, tcy - top_sy / 2 + 0.06),
    (TABLE_ORIGIN_X + top_sx - 0.03, tcy + top_sy / 2 - 0.06),
  )
  for cx, cy in corners:
    scene.add_entity(
      gs.morphs.Box(size=(leg_sx, leg_sy, leg_h), pos=(cx, cy, leg_z), fixed=True),
      surface=color_rough(TABLE_LEG_COLOR),
    )


def add_cardboard_box(scene: gs.Scene, layout: PackagingLayout) -> None:
  cx, cy = layout.box_center_xy
  ox, oy, oz = layout.box_outer
  t = layout.box_wall
  floor_z = layout.table_top_z

  cardboard_surf = image_rough("cardboard.jpg", CARDBOARD_COLOR_SCALE)

  add_textured_box(
    scene,
    (ox, oy, t),
    (cx, cy, floor_z + t / 2),
    "cardboard.jpg",
    color_scale=CARDBOARD_COLOR_SCALE,
    fixed=True,
  )
  inner_z0 = floor_z + t
  add_textured_box(
    scene, (ox, t, oz), (cx, cy - oy / 2 + t / 2, inner_z0 + oz / 2), "cardboard.jpg",
    color_scale=CARDBOARD_COLOR_SCALE, fixed=True,
  )
  add_textured_box(
    scene, (ox, t, oz), (cx, cy + oy / 2 - t / 2, inner_z0 + oz / 2), "cardboard.jpg",
    color_scale=CARDBOARD_COLOR_SCALE, fixed=True,
  )
  add_textured_box(
    scene,
    (t, oy - 2 * t, oz),
    (cx - ox / 2 + t / 2, cy, inner_z0 + oz / 2),
    "cardboard.jpg",
    color_scale=CARDBOARD_COLOR_SCALE,
    fixed=True,
  )
  add_textured_box(
    scene,
    (t, oy - 2 * t, oz),
    (cx + ox / 2 - t / 2, cy, inner_z0 + oz / 2),
    "cardboard.jpg",
    color_scale=CARDBOARD_COLOR_SCALE,
    fixed=True,
  )

  scene.add_entity(
    gs.morphs.Mesh(
      file=mesh_path("uv_box.obj"),
      scale=(ox * 0.55, 0.012, 0.002),
      pos=(cx, cy + oy / 2 - 0.006, inner_z0 + oz - 0.001),
      fixed=True,
      decimate=False,
      convexify=False,
    ),
    surface=image_rough("cardboard.jpg", CARDBOARD_COLOR_SCALE),
  )


def build_packaging_scene(
  table_top_z: float = DEFAULT_TABLE_TOP_Z,
  *,
  sim_dt: float = 0.02,
  show_viewer: bool = True,
  build_scene: bool = True,
  renderer=None,
  add_capture_camera: bool = False,
  capture_res: tuple[int, int] = (1280, 720),
):
  layout = make_layout(table_top_z)
  cam_pos, cam_lookat = packaging_camera(table_top_z)
  scene_kwargs: dict = {}
  if renderer is not None:
    scene_kwargs["renderer"] = renderer
  scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=sim_dt, substeps=8),
    rigid_options=gs.options.RigidOptions(
      dt=sim_dt,
      constraint_solver=gs.constraint_solver.Newton,
      enable_collision=True,
      enable_joint_limit=True,
      iterations=100,
      noslip_iterations=5,
    ),
    viewer_options=gs.options.ViewerOptions(
      camera_pos=cam_pos,
      camera_lookat=cam_lookat,
      camera_fov=35,
      refresh_rate=60,
    ),
    show_viewer=show_viewer,
    **scene_kwargs,
  )

  if add_capture_camera:
    scene.add_camera(pos=cam_pos, lookat=cam_lookat, res=capture_res, fov=35)

  scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

  add_table(scene, layout)
  add_cardboard_box(scene, layout)

  urdf_path = robot_visual_glb_urdf("xarm6_1305", with_gripper_g2=True, movable=True)
  robot = scene.add_entity(
    gs.morphs.URDF(
      file=urdf_path,
      pos=(layout.robot_xy[0], layout.robot_xy[1], layout.table_top_z),
      euler=(0.0, 0.0, ROBOT_BASE_YAW_DEG),
      fixed=True,
      requires_jac_and_IK=True,
    ),
    surface=glb_view_surface(),
  )

  obj_half_z = layout.obj_size[2] / 2
  block = scene.add_entity(
    gs.morphs.Box(
      size=layout.obj_size,
      pos=(
        layout.obj_spawn_xy[0],
        layout.obj_spawn_xy[1],
        layout.table_top_z + obj_half_z,
      ),
      fixed=False,
    ),
    material=gs.materials.Rigid(rho=630, friction=1.0),  # ~17 g for 30 mm cube
    surface=color_rough(BLOCK_COLOR),
  )

  if build_scene:
    scene.build(n_envs=1)
  return scene, robot, block, layout
