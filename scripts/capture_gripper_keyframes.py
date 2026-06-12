#!/usr/bin/env python3
"""Capture gripper keyframe renders and static-vs-movable metrics."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "xarm6"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _bootstrap  # noqa: F401
import genesis as gs
from scipy.spatial.transform import Rotation
from ufactory.glb_visual import enable_glb_pbr_surfaces, glb_view_surface
from ufactory.paths import xarm6_1305_visual_glb_urdf

OUT_DIR = Path(__file__).resolve().parents[1] / ".cursor" / "gripper_keyframes"
LOG = Path(__file__).resolve().parents[1] / ".cursor" / "debug-a21d90.log"
GRIPPER_GLB_DIR = (
    Path(__file__).resolve().parents[1]
    / "assets/urdf/gripper_g2/meshes/visual/visual_glb"
)
GRIPPER_STL_DIR = Path(__file__).resolve().parents[1] / "assets/urdf/gripper_g2/meshes/collision"
LINKAGE_GLBS = (
    "left_finger.glb",
    "right_finger.glb",
    "left_outer_knuckle.glb",
    "right_outer_knuckle.glb",
    "left_inner_knuckle.glb",
    "right_inner_knuckle.glb",
)
STL_FOR_GLB = {
    "left_finger.glb": "left_finger.STL",
    "right_finger.glb": "right_finger.STL",
    "left_outer_knuckle.glb": "left_outer_knuckle.STL",
    "right_outer_knuckle.glb": "right_outer_knuckle.STL",
    "left_inner_knuckle.glb": "left_inner_knuckle.STL",
    "right_inner_knuckle.glb": "right_inner_knuckle.STL",
}
DRIVE_Q = {"open": 0.0, "closed": 0.85}
GRIPPER_JOINTS = (
    "drive_joint",
    "left_finger_joint",
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_finger_joint",
    "right_inner_knuckle_joint",
)
CAMERA_POS = (0.34, -0.42, 0.28)
CAMERA_LOOKAT = (0.08, 0.0, 0.11)
CAMERA_FOV = 30


def _log(hypothesis_id: str, message: str, data: dict, run_id: str = "keyframe-capture") -> None:
    # #region agent log
    payload = {
        "sessionId": "a21d90",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": "capture_gripper_keyframes.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    # #endregion


def _pca_axes(mesh_vertices: np.ndarray) -> np.ndarray:
    centered = mesh_vertices - mesh_vertices.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return vh


def _axis_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    dot = float(np.clip(np.abs(a @ b), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def _log_link_local_orientation() -> None:
    """Compare movable GLB vs STL principal axes in each link-local frame."""
    import trimesh

    for glb_name in LINKAGE_GLBS:
        glb = trimesh.load(GRIPPER_GLB_DIR / glb_name, force="mesh")
        stl = trimesh.load(GRIPPER_STL_DIR / STL_FOR_GLB[glb_name], force="mesh")
        glb_axes = _pca_axes(np.asarray(glb.vertices))
        stl_axes = _pca_axes(np.asarray(stl.vertices))
        axis_angles = [
            round(_axis_angle_deg(glb_axes[i], stl_axes[j]), 2)
            for i in range(3)
            for j in range(3)
        ]
        best_pairs = []
        for i in range(3):
            j = int(np.argmin([_axis_angle_deg(glb_axes[i], stl_axes[k]) for k in range(3)]))
            best_pairs.append(
                {
                    "glb_axis": i,
                    "stl_axis": j,
                    "angle_deg": round(_axis_angle_deg(glb_axes[i], stl_axes[j]), 2),
                }
            )
        extent_glb = (glb.vertices.max(0) - glb.vertices.min(0)).tolist()
        extent_stl = (stl.vertices.max(0) - stl.vertices.min(0)).tolist()
        _log(
            "A",
            f"link-local PCA axes {glb_name}",
            {
                "glb": glb_name,
                "best_axis_pairs": best_pairs,
                "min_axis_angle_deg": min(p["angle_deg"] for p in best_pairs),
                "extent_glb_mm": [round(float(x) * 1000, 2) for x in extent_glb],
                "extent_stl_mm": [round(float(x) * 1000, 2) for x in extent_stl],
                "all_pair_angles_deg": axis_angles,
            },
            run_id="orientation-diagnose",
        )


def _quat_R(quat) -> np.ndarray:
    q = np.asarray(quat.cpu().numpy()).reshape(4)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def _gripper_cloud(robot, scene, q: float, movable: bool) -> np.ndarray:
    jm = {j.name.split("/")[-1]: j for j in robot.joints}
    dofs = [jm[name].dofs_idx_local[0] for name in GRIPPER_JOINTS if name in jm]
    if dofs:
        robot.set_dofs_position(np.full(len(dofs), q), dofs)
        robot.control_dofs_position(np.full(len(dofs), q), dofs)
    for _ in range(30):
        scene.step()
    links = {l.name.split("/")[-1]: l for l in robot.links}
    chunks = []
    for vg in robot.vgeoms:
        meta = vg.metadata if isinstance(vg.metadata, dict) else {}
        name = Path(str(meta.get("mesh_path", ""))).name
        if movable:
            if name not in {
                "base.glb",
                "left_finger.glb",
                "right_finger.glb",
                "left_outer_knuckle.glb",
                "right_outer_knuckle.glb",
                "left_inner_knuckle.glb",
                "right_inner_knuckle.glb",
            }:
                continue
        elif name != "gripper_g2_static_link6.glb":
            continue
        link = links[vg.link.name.split("/")[-1]]
        v = vg.get_trimesh().vertices
        R = _quat_R(link.get_quat())
        t = np.asarray(link.get_pos().cpu().numpy()).reshape(3)
        chunks.append(v @ R.T + t)
    pts = np.vstack(chunks)
    n = min(12000, len(pts))
    return pts[np.random.default_rng(0).choice(len(pts), n, replace=False)]


def _chamfer(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    ta, tb = cKDTree(b), cKDTree(a)
    da, _ = ta.query(a, k=1)
    db, _ = tb.query(b, k=1)
    return {
        "a_to_b_mean_mm": float(da.mean() * 1000),
        "a_to_b_p95_mm": float(np.percentile(da, 95) * 1000),
        "b_to_a_mean_mm": float(db.mean() * 1000),
        "b_to_a_p95_mm": float(np.percentile(db, 95) * 1000),
    }


def main() -> None:
    if LOG.exists():
        LOG.unlink()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_link_local_orientation()

    enable_glb_pbr_surfaces()
    gs.init(backend=gs.gpu, logging_level="error")

    for movable in (False, True):
        label = "movable" if movable else "static"
        scene = gs.Scene(
            show_viewer=False,
            sim_options=gs.options.SimOptions(dt=0.01),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=CAMERA_POS,
                camera_lookat=CAMERA_LOOKAT,
                camera_fov=CAMERA_FOV,
            ),
        )
        scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        robot = scene.add_entity(
            gs.morphs.URDF(
                file=xarm6_1305_visual_glb_urdf(with_gripper_g2=True, movable=movable),
                fixed=True,
            ),
            surface=glb_view_surface(),
        )
        cam = scene.add_camera(
            res=(960, 720),
            pos=CAMERA_POS,
            lookat=CAMERA_LOOKAT,
            fov=CAMERA_FOV,
        )
        scene.build()

        clouds: dict[str, np.ndarray] = {}
        for pose, q in DRIVE_Q.items():
            cloud = _gripper_cloud(robot, scene, q, movable)
            clouds[pose] = cloud
            rgb, _, _, _ = cam.render()
            frame = rgb[0] if isinstance(rgb, np.ndarray) and rgb.ndim == 4 else rgb
            out = OUT_DIR / f"{label}_{pose}.png"
            imageio.imwrite(out, np.asarray(frame))
            _log("J", f"render {label} {pose}", {"file": str(out), "verts_sampled": int(len(cloud))})
            print(f"wrote {out}")

    # chamfer: separate lightweight scenes (avoid scene.clear — not always available)
    scenes = {}
    for movable in (False, True):
        sc = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.01))
        rb = sc.add_entity(
            gs.morphs.URDF(
                file=xarm6_1305_visual_glb_urdf(with_gripper_g2=True, movable=movable),
                fixed=True,
            ),
            surface=glb_view_surface(),
        )
        sc.build()
        scenes[movable] = (rb, sc)

    for pose, q in DRIVE_Q.items():
        ps = _gripper_cloud(*scenes[False], q, False)
        pm = _gripper_cloud(*scenes[True], q, True)
        metrics = _chamfer(ps, pm)
        _log("K", f"chamfer static vs movable @ {pose}", {"pose": pose, "drive_q": q, **metrics})
        print(f"chamfer {pose}:", metrics)

    print(f"keyframes: {OUT_DIR}")
    print(f"log: {LOG}")


if __name__ == "__main__":
    main()
