#!/usr/bin/env python3
"""Capture Bio Gripper G2 static vs movable keyframes and runtime mesh metrics."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import _bootstrap  # noqa: F401
import genesis as gs
from ufactory.glb_visual import enable_glb_pbr_surfaces, glb_view_surface
from ufactory.paths import robot_visual_glb_urdf

OUT_DIR = Path(__file__).resolve().parents[1] / ".cursor" / "bio_gripper_keyframes"
LOG = Path(__file__).resolve().parents[1] / ".cursor" / "debug-e97626.log"
ROBOT = "xarm5_1305"
CAMERA_POS = (0.55, -0.65, 0.42)
CAMERA_LOOKAT = (0.12, 0.0, 0.18)
CAMERA_FOV = 32

BIO_STATIC_MESH = "bio_gripper_g2_visual_link5.glb"
BIO_MOVABLE_MESHES = frozenset(
    {
        "bio_gripper_base.glb",
        "bio_left_finger.glb",
        "bio_right_finger.glb",
    }
)


def _log(hypothesis_id: str, message: str, data: dict, run_id: str = "bio-keyframe") -> None:
    # #region agent log
    payload = {
        "sessionId": "e97626",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": "capture_bio_gripper_keyframes.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")
    # #endregion


def _quat_R(quat) -> np.ndarray:
    q = np.asarray(quat.cpu().numpy()).reshape(4)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def _bio_gripper_cloud(robot, scene, *, movable: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    for _ in range(30):
        scene.step()
    links = {l.name.split("/")[-1]: l for l in robot.links}
    chunks: list[np.ndarray] = []
    by_mesh: dict[str, list[np.ndarray]] = {}
    for vg in robot.vgeoms:
        meta = vg.metadata if isinstance(vg.metadata, dict) else {}
        name = Path(str(meta.get("mesh_path", ""))).name
        if movable:
            if name not in BIO_MOVABLE_MESHES:
                continue
        elif name != BIO_STATIC_MESH:
            continue
        link = links[vg.link.name.split("/")[-1]]
        v = np.asarray(vg.get_trimesh().vertices)
        R = _quat_R(link.get_quat())
        t = np.asarray(link.get_pos().cpu().numpy()).reshape(3)
        world = v @ R.T + t
        chunks.append(world)
        by_mesh.setdefault(name, []).append(world)
    pts = np.vstack(chunks)
    n = min(16000, len(pts))
    rng = np.random.default_rng(0)
    sample = pts[rng.choice(len(pts), n, replace=False)]
    merged_by_mesh = {k: np.vstack(v) for k, v in by_mesh.items()}
    return sample, merged_by_mesh


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


def _finger_stats(by_mesh: dict[str, np.ndarray], label: str) -> None:
    if label == "static":
        if BIO_STATIC_MESH not in by_mesh:
            return
        pts = by_mesh[BIO_STATIC_MESH]
        pos_y = pts[pts[:, 1] > 0.003]
        neg_y = pts[pts[:, 1] < -0.003]
        finger = pts[pts[:, 0] > 0.05]
        _log(
            "H3",
            f"{label} runtime finger centroids",
            {
                "all_centroid_mm": [round(float(x) * 1000, 2) for x in pts.mean(0)],
                "pos_y_centroid_mm": [round(float(x) * 1000, 2) for x in pos_y.mean(0)]
                if len(pos_y)
                else None,
                "neg_y_centroid_mm": [round(float(x) * 1000, 2) for x in neg_y.mean(0)]
                if len(neg_y)
                else None,
                "finger_x_gt_50mm_centroid_mm": [round(float(x) * 1000, 2) for x in finger.mean(0)]
                if len(finger)
                else None,
                "verts": int(len(pts)),
            },
        )
        return
    for side, fname in (("left", "left_finger.glb"), ("right", "right_finger.glb")):
        if fname not in by_mesh:
            continue
        pts = by_mesh[fname]
        _log(
            "H3",
            f"{label} runtime {side} finger",
            {
                "centroid_mm": [round(float(x) * 1000, 2) for x in pts.mean(0)],
                "extent_mm": [round(float(x) * 1000, 2) for x in (pts.max(0) - pts.min(0))],
                "verts": int(len(pts)),
            },
        )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    enable_glb_pbr_surfaces()
    gs.init(backend=gs.gpu, logging_level="error")

    clouds: dict[str, np.ndarray] = {}
    mesh_parts: dict[str, dict[str, np.ndarray]] = {}

    for movable in (False, True):
        label = "movable" if movable else "static"
        urdf = robot_visual_glb_urdf(
            ROBOT, with_bio_gripper_g2=True, movable=movable
        )
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
            gs.morphs.URDF(file=urdf, fixed=True),
            surface=glb_view_surface(),
        )
        cam = scene.add_camera(
            res=(960, 720),
            pos=CAMERA_POS,
            lookat=CAMERA_LOOKAT,
            fov=CAMERA_FOV,
        )
        scene.build()
        cloud, by_mesh = _bio_gripper_cloud(robot, scene, movable=movable)
        clouds[label] = cloud
        mesh_parts[label] = by_mesh
        _finger_stats(by_mesh, label)
        rgb, _, _, _ = cam.render()
        frame = rgb[0] if isinstance(rgb, np.ndarray) and rgb.ndim == 4 else rgb
        out = OUT_DIR / f"{label}_open.png"
        imageio.imwrite(out, np.asarray(frame))
        _log("H4", f"render {label}", {"file": str(out), "urdf": urdf, "sample_verts": len(cloud)})
        print(f"wrote {out}")

    metrics = _chamfer(clouds["static"], clouds["movable"])
    _log("H5", "chamfer static vs movable @ open", metrics)
    print("chamfer:", metrics)

  # finger-only chamfer if we can isolate finger regions
    static_pts = clouds["static"]
    static_finger = static_pts[static_pts[:, 0] > 0.05]
    mov_parts = []
    for fname in ("bio_left_finger.glb", "bio_right_finger.glb"):
        if fname in mesh_parts["movable"]:
            mov_parts.append(mesh_parts["movable"][fname])
    if mov_parts and len(static_finger):
        mov_finger = np.vstack(mov_parts)
        n = min(8000, len(static_finger), len(mov_finger))
        rng = np.random.default_rng(1)
        sf = static_finger[rng.choice(len(static_finger), n, replace=False)]
        mf = mov_finger[rng.choice(len(mov_finger), n, replace=False)]
        finger_metrics = _chamfer(sf, mf)
        _log("H5", "chamfer finger regions static vs movable", finger_metrics)
        print("finger chamfer:", finger_metrics)

    print(f"keyframes: {OUT_DIR}")
    print(f"log: {LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
