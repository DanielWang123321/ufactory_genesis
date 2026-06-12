#!/usr/bin/env python3
"""Genesis world-space knuckle vs static assembly at drive q."""

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

import _bootstrap  # noqa: F401
import genesis as gs
from ufactory.glb_visual import enable_glb_pbr_surfaces, glb_view_surface
from ufactory.paths import xarm6_1305_visual_glb_urdf

LOG = Path(__file__).resolve().parents[1] / ".cursor" / "debug-a21d90.log"
KNUCKLE_NAMES = {
    "left_outer_knuckle.glb",
    "right_outer_knuckle.glb",
    "left_inner_knuckle.glb",
    "right_inner_knuckle.glb",
}
GRIPPER_JOINTS = (
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


def _cloud(robot, scene, q: float, movable: bool, knuckle_only: bool) -> np.ndarray:
    jm = {j.name.split("/")[-1]: j for j in robot.joints}
    dofs = [jm[n].dofs_idx_local[0] for n in GRIPPER_JOINTS if n in jm]
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
        ps_k = _cloud(*scenes[False], q, False, knuckle_only=True)
        pm_k = _cloud(*scenes[True], q, True, knuckle_only=True)
        ps_all = _cloud(*scenes[False], q, False, knuckle_only=False)
        pm_all = _cloud(*scenes[True], q, True, knuckle_only=False)
        m_k = _chamfer_mm(pm_k, ps_k)
        m_all = _chamfer_mm(pm_all, ps_all)
        _log("H2", f"genesis chamfer @ {label}", {"q": q, "knuckle_only": m_k, "full_gripper": m_all})
        print(f"q={label} knuckle chamfer:", m_k)
        print(f"q={label} full chamfer:", m_all)


if __name__ == "__main__":
    main()
