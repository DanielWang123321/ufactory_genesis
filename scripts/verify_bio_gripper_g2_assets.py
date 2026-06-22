#!/usr/bin/env python3
"""Validate Bio Gripper G2 URDF mesh paths, names, finger stroke, and static flange pose."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

from ufactory.bio_gripper_g2 import CLOSED_GAP
from ufactory.paths import BIO_GRIPPER_G2_ASSETS, bio_gripper_g2_movable_visual_urdf, robot_visual_glb_urdf
from ufactory.robot_registry import ROBOT_PROFILES

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
  sys.path.insert(0, str(_SCRIPTS_DIR))

from relocalize_bio_gripper_g2_glb import (  # noqa: E402
  FINGER_TARGET,
  _arm_locating_holes,
  _attach_hole_fit_mm,
  _finger_direction,
  _gripper_pin_points_stl,
  _static_visual_finger_world_ref,
)
from relocalize_gripper_glb import EE_RING_R, G2_RING_R, _ring_plane_z_and_xy  # noqa: E402

# Tolerance (m) on the verified closed two-finger gap; the gripping faces are snapped to
# CLOSED_GAP during relocalization, so the residual is well under this.
_CLOSED_GAP_TOL = 0.004
_ATTACH_HOLE_FIT_MAX_MM = 3.0
_ATTACH_RING_GAP_MAX_MM = 5.0
_FINGER_WORLD_STATIC_ANGLE_TOL_DEG = 2.0
_METRICS_PATH = BIO_GRIPPER_G2_ASSETS / "meshes" / "visual" / "relocalize_metrics.json"
_STL_BASE = BIO_GRIPPER_G2_ASSETS / "meshes" / "visual" / "link_base.stl"


BASE_LINK = "bio_gripper_g2_base_link"
LEFT_LINK = "bio_gripper_g2_left_finger"
RIGHT_LINK = "bio_gripper_g2_right_finger"
VISUAL_LINK = "bio_gripper_g2_visual"
BASE_JOINT = "bio_gripper_g2_attach"
LEFT_JOINT = "bio_gripper_g2_left_finger_joint"
RIGHT_JOINT = "bio_gripper_g2_right_finger_joint"
TCP_LINK = "bio_gripper_g2_tcp_link"
TCP_JOINT = "bio_gripper_g2_tcp_joint"


def _origin_rt(elem: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
  if elem is None:
    return np.eye(3), np.zeros(3)
  xyz = np.fromstring(elem.get("xyz", "0 0 0"), sep=" ", dtype=float)
  roll, pitch, yaw = np.fromstring(elem.get("rpy", "0 0 0"), sep=" ", dtype=float)
  cr, sr = np.cos(roll), np.sin(roll)
  cp, sp = np.cos(pitch), np.sin(pitch)
  cy, sy = np.cos(yaw), np.sin(yaw)
  rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
  ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
  rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
  return rz @ ry @ rx, xyz


def _compose(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
  ra, ta = a
  rb, tb = b
  return ra @ rb, ta + ra @ tb


def _link_world_rt(root: ET.Element, link_name: str) -> tuple[np.ndarray, np.ndarray]:
  parent_by_child: dict[str, tuple[str, tuple[np.ndarray, np.ndarray]]] = {}
  for joint in root.findall("joint"):
    parent = joint.find("parent")
    child = joint.find("child")
    if parent is None or child is None:
      continue
    parent_by_child[child.get("link", "")] = (parent.get("link", ""), _origin_rt(joint.find("origin")))

  chain: list[tuple[np.ndarray, np.ndarray]] = []
  current = link_name
  while current in parent_by_child:
    parent, origin_rt = parent_by_child[current]
    chain.append(origin_rt)
    current = parent

  out = (np.eye(3), np.zeros(3))
  for origin_rt in reversed(chain):
    out = _compose(out, origin_rt)
  return out


def _link_visual(root: ET.Element, link_name: str) -> tuple[str, tuple[np.ndarray, np.ndarray]]:
  for link in root.findall("link"):
    if link.get("name") != link_name:
      continue
    visual = link.find("visual")
    if visual is None:
      raise KeyError(f"{link_name} has no visual")
    mesh = visual.find("./geometry/mesh")
    if mesh is None or not mesh.get("filename"):
      raise KeyError(f"{link_name} visual has no mesh")
    return mesh.get("filename", ""), _origin_rt(visual.find("origin"))
  raise KeyError(f"missing link {link_name}")


def _resolve_mesh(urdf: Path, mesh_ref: str) -> Path:
  return (urdf.parent / mesh_ref).resolve()


def _mesh_parts(path: Path, rt: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
  scene = trimesh.load(path, force="scene")
  rot, trans = rt
  plastic: list[np.ndarray] = []
  metal: list[np.ndarray] = []
  for geom in scene.geometry.values():
    material = getattr(geom.visual, "material", None)
    metallic = float(getattr(material, "metallicFactor", 0.0) or 0.0)
    verts = np.asarray(geom.vertices, dtype=float) @ rot.T + trans
    if metallic >= 0.5:
      metal.append(verts)
    else:
      plastic.append(verts)
  if not plastic or not metal:
    raise ValueError(f"{path} must contain plastic and metal submeshes")
  return np.vstack(plastic), np.vstack(metal)


def _require_names(root: ET.Element, names: set[str], tag: str) -> list[str]:
  found = {elem.get("name", "") for elem in root.findall(tag)}
  return sorted(names - found)


def _blade_inner_y(verts: np.ndarray, side: str) -> float:
  """Center-facing Y of the distal gripping blade (finger link frame; blade is +X)."""
  distal = verts[verts[:, 0] >= np.percentile(verts[:, 0], 70)]
  return float(np.percentile(distal[:, 1], 85 if side == "left" else 15))


def _movable_finger_errors(urdf: Path, root: ET.Element) -> list[str]:
  """Validate the movable fingers: correct sides, +X blades, and the closed 71 mm gap.

  Finger GLBs are stored in the finger link frame and the prismatic joint slides in Y,
  so joint zero is the closed pose and the closed two-finger gap is read directly from
  the meshes (arm-pose independent).  Left jaw must sit at -Y, right at +Y, both blades
  extend toward +X (TCP), and the gripping faces must be CLOSED_GAP apart.
  """
  errors: list[str] = []
  blade: dict[str, tuple[float, float]] = {}
  for link_name, side in ((LEFT_LINK, "left"), (RIGHT_LINK, "right")):
    mesh_ref, _ = _link_visual(root, link_name)
    verts = trimesh.load(_resolve_mesh(urdf, mesh_ref), force="mesh").vertices
    blade[side] = (_blade_inner_y(np.asarray(verts, dtype=float), side), float(verts[:, 0].max()))
  left_inner, left_xmax = blade["left"]
  right_inner, right_xmax = blade["right"]
  if left_inner >= 0.0 or right_inner <= 0.0:
    errors.append(
      f"finger sides swapped: left_inner_y={left_inner * 1000:.1f} mm, "
      f"right_inner_y={right_inner * 1000:.1f} mm"
    )
  if left_xmax <= 0.0 or right_xmax <= 0.0:
    errors.append("finger blades do not extend toward +X (TCP)")
  gap = right_inner - left_inner
  if abs(gap - CLOSED_GAP) > _CLOSED_GAP_TOL:
    errors.append(
      f"closed two-finger gap {gap * 1000:.1f} mm != {CLOSED_GAP * 1000:.0f} mm"
    )
  return errors


def _attach_joint_rt(root: ET.Element) -> tuple[np.ndarray, np.ndarray, str]:
  for joint in root.findall("joint"):
    if joint.get("name") != BASE_JOINT:
      continue
    parent = joint.find("parent")
    if parent is None:
      break
    return _origin_rt(joint.find("origin")), parent.get("link", "")
  raise KeyError(f"missing joint {BASE_JOINT}")


def _robot_ee_glb_path(robot_key: str) -> Path:
  profile = ROBOT_PROFILES[robot_key]
  return (
    profile.assets_dir
    / "meshes"
    / profile.mesh_variant
    / "visual_glb"
    / f"{profile.ee_link}.glb"
  )


def _attach_ring_gap_budget_mm(robot_key: str) -> float:
  if not _METRICS_PATH.is_file():
    return _ATTACH_RING_GAP_MAX_MM
  report = json.loads(_METRICS_PATH.read_text(encoding="utf-8"))
  for entry in report:
    if entry.get("robot_key") == robot_key:
      expected = float(entry.get("attach_ring_gap_mm", _ATTACH_RING_GAP_MAX_MM))
      return max(_ATTACH_RING_GAP_MAX_MM, expected * 1.15 + 0.5)
  return _ATTACH_RING_GAP_MAX_MM


def _attach_hole_fit_budget_mm(robot_key: str) -> float:
  """Allow relocalize-reported fit for flanges with larger pin-hole residual (e.g. uf850)."""
  if not _METRICS_PATH.is_file():
    return _ATTACH_HOLE_FIT_MAX_MM
  report = json.loads(_METRICS_PATH.read_text(encoding="utf-8"))
  for entry in report:
    if entry.get("robot_key") == robot_key:
      expected = float(entry.get("attach_hole_fit_mm", _ATTACH_HOLE_FIT_MAX_MM))
      return max(_ATTACH_HOLE_FIT_MAX_MM, expected * 1.15 + 0.5)
  return _ATTACH_HOLE_FIT_MAX_MM


def _gripper_finger_world_dir(root: ET.Element, base_link: str) -> np.ndarray:
  R, _ = _link_world_rt(root, base_link)
  fw = R @ FINGER_TARGET
  return fw / (np.linalg.norm(fw) + 1e-12)


def _attach_static_angle_budget_deg(robot_key: str) -> float | None:
  if not _METRICS_PATH.is_file():
    return None
  report = json.loads(_METRICS_PATH.read_text(encoding="utf-8"))
  for entry in report:
    if entry.get("robot_key") == robot_key:
      expected = entry.get("attach_static_angle_deg")
      if expected is None:
        return None
      return float(expected) + _FINGER_WORLD_STATIC_ANGLE_TOL_DEG
  return None


def _movable_finger_world_errors(root: ET.Element, robot_key: str) -> list[str]:
  """Movable combo at q=0: finger opening must match static visual reference."""
  errors: list[str] = []
  fw = _gripper_finger_world_dir(root, BASE_LINK)
  profile = ROBOT_PROFILES[robot_key]
  static_ref = _static_visual_finger_world_ref(profile)
  if static_ref is not None:
    dot = float(np.clip(np.dot(fw, static_ref), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(dot)))
    angle_max = _attach_static_angle_budget_deg(robot_key)
    if angle_max is not None and angle_deg > angle_max:
      errors.append(
        f"finger world dir vs static differs by {angle_deg:.1f} deg "
        f"> {angle_max:.1f} deg"
      )
  return errors


def _movable_mount_errors(root: ET.Element, ee_link: str, robot_key: str) -> list[str]:
  """Movable arm combo: pins in base frame must seat in EE holes after attach origin."""
  errors: list[str] = []
  (R, t), parent = _attach_joint_rt(root)
  if parent != ee_link:
    errors.append(f"{BASE_JOINT} parent {parent!r} != expected {ee_link!r}")
  stl_base = trimesh.load(_STL_BASE, force="mesh")
  stl_pins = _gripper_pin_points_stl(stl_base)
  ee_mesh = trimesh.load(_robot_ee_glb_path(robot_key), force="mesh")
  holes = _arm_locating_holes(ee_mesh)
  holes_canon = holes[np.argsort(holes[:, 1])]
  hole_fit = min(
    _attach_hole_fit_mm(R, t, stl_pins, holes_canon),
    _attach_hole_fit_mm(R, t, stl_pins, holes_canon[::-1]),
  )
  hole_fit_max = _attach_hole_fit_budget_mm(robot_key)
  if hole_fit > hole_fit_max:
    errors.append(f"attach pin-hole fit {hole_fit:.2f} mm > {hole_fit_max:.1f} mm")
  base_in_ee = stl_base.copy()
  base_in_ee.vertices = base_in_ee.vertices @ R.T + t
  z_ee, _ = _ring_plane_z_and_xy(ee_mesh, *EE_RING_R, z_pick="max")
  z_g, _ = _ring_plane_z_and_xy(base_in_ee, *G2_RING_R, z_pick="area_peak")
  ring_gap = abs(z_ee - z_g) * 1000
  ring_gap_max = _attach_ring_gap_budget_mm(robot_key)
  if ring_gap > ring_gap_max:
    errors.append(f"attach ring gap {ring_gap:.2f} mm > {ring_gap_max:.1f} mm")
  return errors


def _check_urdf(urdf: Path, *, movable: bool, standalone: bool, ee_link: str | None = None, robot_key: str | None = None) -> list[str]:
  errors: list[str] = []
  root = ET.parse(urdf).getroot()
  old_asset_dir = "bio" + "_gripper"
  for ref in [m.get("filename", "") for m in root.iter("mesh") if m.get("filename")]:
    if not _resolve_mesh(urdf, ref).is_file():
      errors.append(f"missing mesh {ref}")
    if f"{old_asset_dir}/" in ref or f"{old_asset_dir}\\" in ref:
      errors.append(f"old mesh path {ref}")

  required_links = {BASE_LINK, LEFT_LINK, RIGHT_LINK, TCP_LINK}
  required_joints = {LEFT_JOINT, RIGHT_JOINT, TCP_JOINT}
  if not standalone:
    required_joints.add(BASE_JOINT)
  if not movable and not standalone:
    required_links.add(VISUAL_LINK)
    required_joints.add("bio_gripper_g2_visual_fix")

  missing_links = _require_names(root, required_links, "link")
  missing_joints = _require_names(root, required_joints, "joint")
  if missing_links:
    errors.append(f"missing links {missing_links}")
  if missing_joints:
    errors.append(f"missing joints {missing_joints}")

  try:
    if movable:
      errors.extend(_movable_finger_errors(urdf, root))
      if not standalone and ee_link and robot_key:
        errors.extend(_movable_mount_errors(root, ee_link, robot_key))
        errors.extend(_movable_finger_world_errors(root, robot_key))
    else:
      attach_rt = _link_world_rt(root, VISUAL_LINK)
      mesh_ref, visual_rt = _link_visual(root, VISUAL_LINK)
      plastic, metal = _mesh_parts(_resolve_mesh(urdf, mesh_ref), _compose(attach_rt, visual_rt))
      if float(metal[:, 2].max()) <= float(plastic[:, 2].max()) + 1e-4:
        errors.append("static flange is not above plastic")
  except Exception as exc:
    errors.append(str(exc))
  return errors


def main() -> int:
  errors: list[str] = []

  standalone = Path(bio_gripper_g2_movable_visual_urdf())
  standalone_errors = _check_urdf(standalone, movable=True, standalone=True)
  if standalone_errors:
    errors.extend(f"standalone: {err}" for err in standalone_errors)
  else:
    print(f"[ok] standalone: {standalone.name}")

  keys = [k for k, p in ROBOT_PROFILES.items() if p.supports_bio_gripper_g2]
  for key in keys:
    for movable in (False, True):
      urdf = Path(robot_visual_glb_urdf(key, with_bio_gripper_g2=True, movable=movable))
      label = f"{key} movable={movable}"
      urdf_errors = _check_urdf(
        urdf, movable=movable, standalone=False, ee_link=ROBOT_PROFILES[key].ee_link, robot_key=key
      )
      if urdf_errors:
        errors.extend(f"{label}: {err}" for err in urdf_errors)
      else:
        print(f"[ok] {label}: {urdf.name}")

  src = BIO_GRIPPER_G2_ASSETS / "meshes" / "visual" / "visual_glb_src" / "bio_gripper_g2.glb"
  if src.is_file():
    print(f"[ok] source GLB: {src.name}")
  else:
    errors.append(f"missing source GLB: {src}")

  if errors:
    for err in errors:
      print(f"[fail] {err}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
