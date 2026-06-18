#!/usr/bin/env python3
"""Validate Bio Gripper G2 URDF mesh paths, names, and flange-up visual pose."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

from ufactory.paths import BIO_GRIPPER_G2_ASSETS, bio_gripper_g2_movable_visual_urdf, robot_visual_glb_urdf
from ufactory.robot_registry import ROBOT_PROFILES


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


def _mesh_vertices(path: Path, rt: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
  mesh = trimesh.load(path, force="mesh")
  rot, trans = rt
  return np.asarray(mesh.vertices, dtype=float) @ rot.T + trans


def _require_names(root: ET.Element, names: set[str], tag: str) -> list[str]:
  found = {elem.get("name", "") for elem in root.findall(tag)}
  return sorted(names - found)


def _check_urdf(urdf: Path, *, movable: bool, standalone: bool) -> list[str]:
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
      base_rt = _link_world_rt(root, BASE_LINK)
      mesh_ref, visual_rt = _link_visual(root, BASE_LINK)
      plastic, metal = _mesh_parts(_resolve_mesh(urdf, mesh_ref), _compose(base_rt, visual_rt))
      metal_top = float(metal[:, 2].max())
      plastic_top = float(plastic[:, 2].max())
      if metal_top <= plastic_top + 1e-4:
        errors.append(f"base flange is not above plastic: metal_top={metal_top:.4f}, plastic_top={plastic_top:.4f}")
      for link_name in (LEFT_LINK, RIGHT_LINK):
        mesh_ref, visual_rt = _link_visual(root, link_name)
        finger_rt = _compose(_link_world_rt(root, link_name), visual_rt)
        finger = _mesh_vertices(_resolve_mesh(urdf, mesh_ref), finger_rt)
        if float(finger[:, 2].mean()) >= metal_top:
          errors.append(f"{link_name} is above flange after URDF transforms")
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
      urdf_errors = _check_urdf(urdf, movable=movable, standalone=False)
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
