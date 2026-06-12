#!/usr/bin/env python3
"""Verify knuckle Rz90 frame mismatch vs base shell."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from relocalize_gripper_glb import (  # noqa: E402
    GLB_TO_LINK,
    GRIPPER_G2_DIR,
    GRIPPER_STL_DIR,
    KNUCKLE_GLBS,
    STL_FOR_GLB,
    VISUAL_GLB_OUT,
    _gripper_link_poses_in_link6,
    _split_assembly_to_link,
    _transform_points_row,
    mean_surface_distance,
)

LOG = Path(__file__).resolve().parents[1] / ".cursor" / "debug-a21d90.log"
STATIC_GLB = GRIPPER_G2_DIR / "gripper_g2_static_link6.glb"
SEMANTIC_ROT = Rotation.from_euler("z", 90, degrees=True).as_matrix()


def _log(hypothesis_id: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": "a21d90",
        "runId": "rz90-diagnose",
        "hypothesisId": hypothesis_id,
        "location": "diagnose_knuckle_rz90.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _to_link6(mesh_local: trimesh.Trimesh, t_link6_link: np.ndarray) -> np.ndarray:
    return _transform_points_row(np.asarray(mesh_local.vertices), t_link6_link)


def main() -> None:
    link_poses = _gripper_link_poses_in_link6()
    static_whole = trimesh.load(STATIC_GLB, force="mesh")
    base_glb = trimesh.load(VISUAL_GLB_OUT / "base.glb", force="mesh")

    for glb_name in sorted(KNUCKLE_GLBS):
        knuckle_glb = trimesh.load(VISUAL_GLB_OUT / glb_name, force="mesh")
        stl = trimesh.load(GRIPPER_STL_DIR / STL_FOR_GLB[glb_name], force="mesh")
        t = link_poses[glb_name]
        pts_link6 = _to_link6(knuckle_glb, t)
        pts_stl_link6 = _transform_points_row(np.asarray(stl.vertices), t)

        knuckle_rz = knuckle_glb.copy()
        c = knuckle_rz.centroid.copy()
        knuckle_rz.vertices = (knuckle_rz.vertices - c) @ SEMANTIC_ROT.T + c
        pts_rz_link6 = _to_link6(knuckle_rz, t)

        # distance from knuckle cloud to static assembly surface
        tree_static = cKDTree(static_whole.vertices)
        d_curr, _ = tree_static.query(pts_link6, k=1)
        d_rz, _ = tree_static.query(pts_rz_link6, k=1)
        d_stl, _ = tree_static.query(pts_stl_link6, k=1)

        _log(
            "H1",
            f"knuckle vs static assembly {glb_name}",
            {
                "glb": glb_name,
                "current_to_static_mean_mm": round(float(d_curr.mean() * 1000), 2),
                "current_to_static_p95_mm": round(float(np.percentile(d_curr, 95) * 1000), 2),
                "rz90_to_static_mean_mm": round(float(d_rz.mean() * 1000), 2),
                "rz90_to_static_p95_mm": round(float(np.percentile(d_rz, 95) * 1000), 2),
                "stl_to_static_mean_mm": round(float(d_stl.mean() * 1000), 2),
                "link_local_vs_stl_mm": round(
                    float(mean_surface_distance(knuckle_glb, stl) * 1000), 2
                ),
            },
        )
        print(
            f"{glb_name}: current->static {d_curr.mean()*1000:.1f}mm, "
            f"rz90->static {d_rz.mean()*1000:.1f}mm, stl->static {d_stl.mean()*1000:.1f}mm"
        )

    # base shell vs static
    tree_base = cKDTree(base_glb.vertices)
    d_base_static, _ = cKDTree(static_whole.vertices).query(base_glb.vertices, k=1)
    _log(
        "H1",
        "base.glb vs static assembly",
        {
            "base_to_static_mean_mm": round(float(d_base_static.mean() * 1000), 2),
            "base_to_static_p95_mm": round(float(np.percentile(d_base_static, 95) * 1000), 2),
        },
    )
    print(f"base->static mean {d_base_static.mean()*1000:.1f}mm")


if __name__ == "__main__":
    main()
