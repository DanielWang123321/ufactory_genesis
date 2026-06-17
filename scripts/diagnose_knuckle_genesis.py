#!/usr/bin/env python3
"""Genesis runtime diagnostics: knuckle vgeoms, world AABB, drive modes."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "xarm6"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _bootstrap  # noqa: F401
import genesis as gs
from ufactory.glb_visual import enable_glb_pbr_surfaces, glb_view_surface
from ufactory.paths import gripper_g2_movable_visual_urdf, xarm6_1305_visual_glb_urdf

LOG = Path(__file__).resolve().parents[1] / ".cursor" / "debug-a21d90.log"
KNUCKLE_NAMES = {
    "left_outer_knuckle.glb",
    "right_outer_knuckle.glb",
    "left_inner_knuckle.glb",
    "right_inner_knuckle.glb",
}
ALL_GRIPPER_JOINTS = (
    "drive_joint",
    "left_finger_joint",
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_finger_joint",
    "right_inner_knuckle_joint",
)


def _log(hypothesis_id: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": "a21d90",
        "runId": "genesis-knuckle",
        "hypothesisId": hypothesis_id,
        "location": "diagnose_knuckle_genesis.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _quat_R(quat) -> np.ndarray:
    q = np.asarray(quat.cpu().numpy()).reshape(4)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def _set_drive(robot, q: float, mode: str) -> None:
    jm = {j.name.split("/")[-1]: j for j in robot.joints}
    if mode == "drive_only":
        idx = jm["drive_joint"].dofs_idx_local[0]
        robot.set_dofs_position(np.array([q]), [idx])
        robot.control_dofs_position(np.array([q]), [idx])
    else:
        dofs = [jm[n].dofs_idx_local[0] for n in ALL_GRIPPER_JOINTS if n in jm]
        if dofs:
            robot.set_dofs_position(np.full(len(dofs), q), dofs)
            robot.control_dofs_position(np.full(len(dofs), q), dofs)


def _knuckle_vgeom_report(robot, scene, q: float, mode: str) -> dict:
    _set_drive(robot, q, mode)
    for _ in range(30):
        scene.step()
    links = {l.name.split("/")[-1]: l for l in robot.links}
    parts: dict[str, list] = {n: [] for n in KNUCKLE_NAMES}
    for vg in robot.vgeoms:
        meta = vg.metadata if isinstance(vg.metadata, dict) else {}
        name = Path(str(meta.get("mesh_path", ""))).name
        if name not in KNUCKLE_NAMES:
            continue
        link = links[vg.link.name.split("/")[-1]]
        v = vg.get_trimesh().vertices
        R = _quat_R(link.get_quat())
        t = np.asarray(link.get_pos().cpu().numpy()).reshape(3)
        world = v @ R.T + t
        lo, hi = world.min(0), world.max(0)
        parts[name].append(
            {
                "link": link.name.split("/")[-1],
                "verts": int(len(v)),
                "world_aabb_mm": {
                    "min": (lo * 1000).round(2).tolist(),
                    "max": (hi * 1000).round(2).tolist(),
                    "extent": ((hi - lo) * 1000).round(2).tolist(),
                },
            }
        )
    return {
        "mode": mode,
        "q": q,
        "knuckle_vgeom_count": sum(len(v) for v in parts.values()),
        "parts": {k: v for k, v in parts.items() if v},
    }


def _cloud(robot, scene, q: float, mode: str, movable: bool, knuckle_only: bool) -> np.ndarray:
    _set_drive(robot, q, mode)
    for _ in range(30):
        scene.step()
    links = {l.name.split("/")[-1]: l for l in robot.links}
    chunks = []
    for vg in robot.vgeoms:
        meta = vg.metadata if isinstance(vg.metadata, dict) else {}
        name = Path(str(meta.get("mesh_path", ""))).name
        if movable:
            if knuckle_only:
                if name not in KNUCKLE_NAMES:
                    continue
            elif name not in KNUCKLE_NAMES | {"base.glb", "left_finger.glb", "right_finger.glb"}:
                continue
        else:
            if name != "gripper_g2_static_link6.glb":
                continue
        link = links[vg.link.name.split("/")[-1]]
        v = vg.get_trimesh().vertices
        R = _quat_R(link.get_quat())
        t = np.asarray(link.get_pos().cpu().numpy()).reshape(3)
        chunks.append(v @ R.T + t)
    pts = np.vstack(chunks)
    n = min(8000, len(pts))
    return pts[np.random.default_rng(0).choice(len(pts), n, replace=False)]


def _chamfer_mm(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    ta, tb = cKDTree(b), cKDTree(a)
    da, _ = ta.query(a, k=1)
    db, _ = tb.query(b, k=1)
    return {
        "a_to_b_mean_mm": float(da.mean() * 1000),
        "a_to_b_p95_mm": float(np.percentile(da, 95) * 1000),
        "b_to_a_mean_mm": float(db.mean() * 1000),
    }


def main() -> None:
    enable_glb_pbr_surfaces()
    try:
        gs.init(backend=gs.gpu, logging_level="error")
    except gs.GenesisException:
        gs.init(backend=gs.cpu, logging_level="error")

    standalone = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.01))
    standalone_robot = standalone.add_entity(
        gs.morphs.URDF(file=gripper_g2_movable_visual_urdf(), fixed=True),
        surface=glb_view_surface(),
    )
    standalone.build()

    for mode in ("drive_only", "all_joints"):
        for q, label in ((0.0, "open"), (0.85, "closed")):
            report = _knuckle_vgeom_report(standalone_robot, standalone, q, mode)
            _log("H4", f"standalone vgeom @ {label} {mode}", report)
            print(f"standalone {label} {mode}: vgeoms={report['knuckle_vgeom_count']}")
            for part, entries in sorted(report["parts"].items()):
                e = entries[0]
                print(f"  {part}: verts={e['verts']} extent={e['world_aabb_mm']['extent']}")

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

    for q, label in ((0.0, "open"), (0.85, "closed")):
        for mode in ("drive_only", "all_joints"):
            ps_k = _cloud(*scenes[False], q, mode, False, knuckle_only=True)
            pm_k = _cloud(*scenes[True], q, mode, True, knuckle_only=True)
            m_k = _chamfer_mm(pm_k, ps_k)
            _log("H2", f"genesis chamfer @ {label} {mode}", {"q": q, "mode": mode, "knuckle_only": m_k})
            print(f"q={label} {mode} knuckle chamfer:", m_k)


if __name__ == "__main__":
    main()
