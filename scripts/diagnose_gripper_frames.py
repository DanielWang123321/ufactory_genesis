#!/usr/bin/env python3
"""Compare Genesis link frames static vs movable gripper."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "xarm6"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _bootstrap  # noqa: F401
import genesis as gs
from ufactory.glb_visual import enable_glb_pbr_surfaces, glb_view_surface
from ufactory.paths import xarm6_1305_visual_glb_urdf

LOG = Path(__file__).resolve().parents[1] / ".cursor" / "debug-a21d90.log"
LINKS = (
    "link6",
    "link_eef",
    "gripper_g2_visual",
    "xarm_gripper_base_link",
    "left_outer_knuckle",
    "left_finger",
)


def _log(hypothesis_id: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": "a21d90",
        "runId": "frame-diagnose",
        "hypothesisId": hypothesis_id,
        "location": "diagnose_gripper_frames.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _link_pose(robot) -> dict[str, list]:
    out = {}
    for l in robot.links:
        name = l.name.split("/")[-1]
        if name not in LINKS:
            continue
        pos = np.asarray(l.get_pos().cpu().numpy()).reshape(3).tolist()
        quat = np.asarray(l.get_quat().cpu().numpy()).reshape(4).tolist()
        out[name] = {"pos_m": [round(x, 5) for x in pos], "quat_wxyz": [round(x, 5) for x in quat]}
    return out


def _mesh_aabb(robot, mesh_name: str) -> dict | None:
    links = {l.name.split("/")[-1]: l for l in robot.links}
    for vg in robot.vgeoms:
        meta = vg.metadata if isinstance(vg.metadata, dict) else {}
        name = Path(str(meta.get("mesh_path", ""))).name
        if name != mesh_name:
            continue
        link = links[vg.link.name.split("/")[-1]]
        v = vg.get_trimesh().vertices
        from scipy.spatial.transform import Rotation

        q = np.asarray(link.get_quat().cpu().numpy()).reshape(4)
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        t = np.asarray(link.get_pos().cpu().numpy()).reshape(3)
        world = v @ R.T + t
        return {
            "mesh": mesh_name,
            "link": vg.link.name.split("/")[-1],
            "local_centroid_m": [round(float(x), 5) for x in v.mean(0)],
            "world_centroid_m": [round(float(x), 5) for x in world.mean(0)],
            "world_extent_m": [round(float(x), 5) for x in (world.max(0) - world.min(0))],
        }
    return None


def main() -> None:
    enable_glb_pbr_surfaces()
    try:
        gs.init(backend=gs.gpu, logging_level="error")
    except gs.GenesisException:
        gs.init(backend=gs.cpu, logging_level="error")

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
        label = "movable" if movable else "static"
        poses = _link_pose(rb)
        _log("H4", f"link poses {label}", {"movable": movable, "links": poses})
        print(f"=== {label} ===")
        for k, v in poses.items():
            print(k, v)
        for mesh in ("gripper_g2_static_link6.glb", "link6/base.glb", "left_outer_knuckle.glb"):
            info = _mesh_aabb(rb, mesh)
            if info:
                _log("H4", f"mesh aabb {label}", info)
                print(info)


if __name__ == "__main__":
    main()
