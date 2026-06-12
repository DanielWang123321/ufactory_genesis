#!/usr/bin/env python3
"""Test Rz90 on static knuckles in link6 before link-local split."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "xarm6"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _bootstrap  # noqa: F401
import genesis as gs
from ufactory.glb_visual import enable_glb_pbr_surfaces, glb_view_surface
from ufactory.paths import xarm6_1305_visual_glb_urdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from relocalize_gripper_glb import (  # noqa: E402
    GRIPPER_G2_DIR,
    KNUCKLE_GLBS,
    VISUAL_GLB_OUT,
    _gripper_link_poses_in_link6,
    _movable_from_static_parts,
    _split_assembly_to_link,
    _submesh_for_vertex_mask,
    _vertex_labels_via_stl_cloud,
    MOVABLE_SOLID_GLBS,
    relocalize_static_assembly,
)

LOG = Path(__file__).resolve().parents[1] / ".cursor" / "debug-a21d90.log"
RZ = Rotation.from_euler("z", 90, degrees=True).as_matrix()


def _log(hypothesis_id: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": "a21d90",
        "runId": "rz90-link6-test",
        "hypothesisId": hypothesis_id,
        "location": "diagnose_knuckle_rz90_link6.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _knuckles_rz90_link6(static_parts, link_poses) -> dict[str, trimesh.Trimesh]:
    combined = trimesh.util.concatenate(static_parts)
    masks = _vertex_labels_via_stl_cloud(combined, link_poses)
    out = {}
    for glb_name in KNUCKLE_GLBS:
        sub = _submesh_for_vertex_mask(combined, masks[glb_name])
        c = sub.centroid.copy()
        sub.vertices = (sub.vertices - c) @ RZ.T + c
        out[glb_name] = _split_assembly_to_link(sub, link_poses[glb_name])
    return out


def _world_cloud(robot, scene, meshes: dict[str, trimesh.Trimesh]) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    jm = {j.name.split("/")[-1]: j for j in robot.joints}
    names = (
        "drive_joint",
        "left_finger_joint",
        "left_inner_knuckle_joint",
        "right_outer_knuckle_joint",
        "right_finger_joint",
        "right_inner_knuckle_joint",
    )
    dofs = [jm[n].dofs_idx_local[0] for n in names if n in jm]
    if dofs:
        robot.set_dofs_position(np.zeros(len(dofs)), dofs)
    for _ in range(20):
        scene.step()
    links = {l.name.split("/")[-1]: l for l in robot.links}
    from relocalize_gripper_glb import GLB_TO_LINK

    chunks = []
    for glb_name, mesh in meshes.items():
        link = links[GLB_TO_LINK[glb_name]]
        q = np.asarray(link.get_quat().cpu().numpy()).reshape(4)
        rot = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        t = np.asarray(link.get_pos().cpu().numpy()).reshape(3)
        chunks.append(np.asarray(mesh.vertices) @ rot.T + t)
    return np.vstack(chunks)


def main() -> None:
    _, static_parts = relocalize_static_assembly("link6", dry_run=True)
    link_poses = _gripper_link_poses_in_link6()
    cur = {g: trimesh.load(VISUAL_GLB_OUT / g, force="mesh") for g in KNUCKLE_GLBS}
    rz = _knuckles_rz90_link6(static_parts, link_poses)
    base = trimesh.load(VISUAL_GLB_OUT / "base.glb", force="mesh")

    enable_glb_pbr_surfaces()
    try:
        gs.init(backend=gs.gpu, logging_level="error")
    except gs.GenesisException:
        gs.init(backend=gs.cpu, logging_level="error")

    sc = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.01))
    rb = sc.add_entity(
        gs.morphs.URDF(file=xarm6_1305_visual_glb_urdf(with_gripper_g2=True, movable=True), fixed=True),
        surface=glb_view_surface(),
    )
    sc.build()

    # distance knuckle world cloud to base shell surface
    tree_base = cKDTree(np.asarray(base.vertices))  # link-local only approx
    for label, meshes in (("current", cur), ("rz90_link6", rz)):
        # compose knuckle+base in link6 via link poses (analytical)
        pts = []
        for glb_name, mesh in meshes.items():
            t = link_poses[glb_name]
            ones = np.ones((len(mesh.vertices), 1))
            homog = np.hstack([mesh.vertices, ones])
            pts.append((homog @ t.T)[:, :3])
        knuckle_link6 = np.vstack(pts)
        base_link6 = np.asarray(base.vertices)  # base on link6 in URDF
        tree_k = cKDTree(knuckle_link6)
        d, _ = tree_k.query(base_link6, k=1)
        _log(
            "H5",
            f"knuckle vs base shell link6 {label}",
            {
                "label": label,
                "base_to_knuckle_mean_mm": round(float(d.mean() * 1000), 2),
                "base_to_knuckle_p95_mm": round(float(np.percentile(d, 95) * 1000), 2),
            },
        )
        print(label, "base->knuckle mean mm", d.mean() * 1000)


if __name__ == "__main__":
    main()
