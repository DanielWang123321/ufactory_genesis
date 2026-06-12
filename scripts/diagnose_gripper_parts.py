#!/usr/bin/env python3
"""Per-part Genesis world chamfer: movable vs static assembly."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "xarm6"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _bootstrap  # noqa: F401
import genesis as gs
from scipy.spatial.transform import Rotation
from ufactory.glb_visual import enable_glb_pbr_surfaces, glb_view_surface
from ufactory.paths import xarm6_1305_visual_glb_urdf

LOG = Path(__file__).resolve().parents[1] / ".cursor" / "debug-a21d90.log"
PARTS = (
    "base.glb",
    "left_finger.glb",
    "right_finger.glb",
    "left_outer_knuckle.glb",
    "right_outer_knuckle.glb",
    "left_inner_knuckle.glb",
    "right_inner_knuckle.glb",
)


def _log(hypothesis_id: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": "a21d90",
        "runId": "part-chamfer",
        "hypothesisId": hypothesis_id,
        "location": "diagnose_gripper_parts.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _quat_R(quat) -> np.ndarray:
    q = np.asarray(quat.cpu().numpy()).reshape(4)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def _part_cloud(robot, scene, q: float, part: str) -> np.ndarray:
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
        robot.set_dofs_position(np.full(len(dofs), q), dofs)
        robot.control_dofs_position(np.full(len(dofs), q), dofs)
    for _ in range(30):
        scene.step()
    links = {l.name.split("/")[-1]: l for l in robot.links}
    for vg in robot.vgeoms:
        meta = vg.metadata if isinstance(vg.metadata, dict) else {}
        name = Path(str(meta.get("mesh_path", ""))).name
        if name != part:
            continue
        link = links[vg.link.name.split("/")[-1]]
        v = vg.get_trimesh().vertices
        R = _quat_R(link.get_quat())
        t = np.asarray(link.get_pos().cpu().numpy()).reshape(3)
        pts = v @ R.T + t
        n = min(4000, len(pts))
        return pts[np.random.default_rng(0).choice(len(pts), n, replace=False)]
    raise RuntimeError(f"part not found: {part}")


def _static_cloud(robot, scene, q: float) -> np.ndarray:
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
        robot.set_dofs_position(np.full(len(dofs), q), dofs)
        robot.control_dofs_position(np.full(len(dofs), q), dofs)
    for _ in range(30):
        scene.step()
    links = {l.name.split("/")[-1]: l for l in robot.links}
    for vg in robot.vgeoms:
        meta = vg.metadata if isinstance(vg.metadata, dict) else {}
        name = Path(str(meta.get("mesh_path", ""))).name
        if name != "gripper_g2_static_link6.glb":
            continue
        link = links[vg.link.name.split("/")[-1]]
        v = vg.get_trimesh().vertices
        R = _quat_R(link.get_quat())
        t = np.asarray(link.get_pos().cpu().numpy()).reshape(3)
        pts = v @ R.T + t
        n = min(12000, len(pts))
        return pts[np.random.default_rng(0).choice(len(pts), n, replace=False)]
    raise RuntimeError("static mesh not found")


def main() -> None:
    enable_glb_pbr_surfaces()
    try:
        gs.init(backend=gs.gpu, logging_level="error")
    except gs.GenesisException:
        gs.init(backend=gs.cpu, logging_level="error")

    sc_m = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.01))
    rb_m = sc_m.add_entity(
        gs.morphs.URDF(file=xarm6_1305_visual_glb_urdf(with_gripper_g2=True, movable=True), fixed=True),
        surface=glb_view_surface(),
    )
    sc_m.build()

    sc_s = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.01))
    rb_s = sc_s.add_entity(
        gs.morphs.URDF(file=xarm6_1305_visual_glb_urdf(with_gripper_g2=True, movable=False), fixed=True),
        surface=glb_view_surface(),
    )
    sc_s.build()

    q = 0.0
    static_pts = _static_cloud(rb_s, sc_s, q)
    tree_static = cKDTree(static_pts)

    for part in PARTS:
        pm = _part_cloud(rb_m, sc_m, q, part)
        d, _ = tree_static.query(pm, k=1)
        entry = {
            "part": part,
            "q": q,
            "to_static_mean_mm": round(float(d.mean() * 1000), 2),
            "to_static_p95_mm": round(float(np.percentile(d, 95) * 1000), 2),
            "to_static_max_mm": round(float(d.max() * 1000), 2),
        }
        _log("H3", f"part vs static @ open", entry)
        print(entry)


if __name__ == "__main__":
    main()
