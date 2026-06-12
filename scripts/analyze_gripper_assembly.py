#!/usr/bin/env python3
"""Compare static vs movable G2 gripper assembly in Genesis (headless)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "xarm6"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _bootstrap  # noqa: F401
import genesis as gs
from ufactory.glb_visual import enable_glb_pbr_surfaces, glb_view_surface
from ufactory.paths import xarm6_1305_visual_glb_urdf

LOG = Path(__file__).resolve().parents[1] / ".cursor" / "debug-906f67.log"
GRIPPER_PARTS = (
    "base.glb",
    "gripper_g2_static_link6.glb",
    "left_outer_knuckle.glb",
    "left_finger.glb",
    "left_inner_knuckle.glb",
    "right_outer_knuckle.glb",
    "right_finger.glb",
    "right_inner_knuckle.glb",
)
DRIVE_Q = {"open": 0.0, "closed": 0.85}


def _log(hypothesis_id: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": "906f67",
        "runId": "assembly-analysis",
        "hypothesisId": hypothesis_id,
        "location": "analyze_gripper_assembly.py",
        "message": message,
        "data": data,
        "timestamp": int(__import__("time").time() * 1000),
    }
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _link_map(robot) -> dict[str, object]:
    return {link.name.split("/")[-1]: link for link in robot.links}


def _set_drive(robot, value: float) -> None:
    joint_map = {j.name.split("/")[-1]: j for j in robot.joints}
    if "drive_joint" not in joint_map:
        return
    idx = joint_map["drive_joint"].dofs_idx_local[0]
    robot.set_dofs_position(np.array([value]), [idx])


def _quat_to_R(quat) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    q = np.asarray(quat.cpu().numpy() if hasattr(quat, "cpu") else quat).reshape(-1)[:4]
    # Genesis uses (w, x, y, z)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def _world_samples(robot) -> dict[str, dict]:
    """Per gripper GLB: world centroid using full link pose (R,t)."""
    links = _link_map(robot)
    link6 = links.get("link6")
    base_pos = np.asarray(link6.get_pos().cpu().numpy()).reshape(3) if link6 else np.zeros(3)
    out: dict[str, dict] = {}
    for vg in robot.vgeoms:
        mesh_path = str((vg.metadata or {}).get("mesh_path", ""))
        part = Path(mesh_path).name
        if part not in GRIPPER_PARTS:
            continue
        link_name = vg.link.name.split("/")[-1]
        link = links.get(link_name)
        if link is None:
            continue
        tm = vg.get_trimesh()
        pts_local = tm.vertices
        link_pos = np.asarray(link.get_pos().cpu().numpy()).reshape(3)
        link_R = _quat_to_R(link.get_quat())
        pts_world = pts_local @ link_R.T + link_pos
        centroid = pts_world.mean(0)
        base_R = _quat_to_R(link6.get_quat()) if link6 is not None else np.eye(3)
        rel_link6 = (centroid - base_pos) @ base_R * 1000
        out[part] = {
            "link": link_name,
            "centroid_world_mm": (centroid * 1000).round(2).tolist(),
            "rel_link6_mm": rel_link6.round(2).tolist(),
            "extent_mm": ((pts_local.max(0) - pts_local.min(0)) * 1000).round(2).tolist(),
            "verts": int(len(pts_local)),
        }
    return out


def _load_robot(urdf: str):
    scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.01))
    robot = scene.add_entity(
        gs.morphs.URDF(file=urdf, fixed=True),
        surface=glb_view_surface(),
    )
    scene.build()
    return robot, scene


def _pairwise_dists(samples: dict[str, dict], a: str, b: str) -> float | None:
    if a not in samples or b not in samples:
        return None
    pa = np.array(samples[a]["centroid_world_mm"])
    pb = np.array(samples[b]["centroid_world_mm"])
    return float(np.linalg.norm(pa - pb))


def main() -> None:
    if LOG.exists():
        LOG.unlink()

    enable_glb_pbr_surfaces()
    gs.init(backend=gs.gpu, logging_level="error")

    static_urdf = xarm6_1305_visual_glb_urdf(with_gripper_g2=True, movable=False)
    movable_urdf = xarm6_1305_visual_glb_urdf(with_gripper_g2=True, movable=True)

    for label, urdf in (
        ("static", static_urdf),
        ("movable", movable_urdf),
    ):
        robot, scene = _load_robot(urdf)
        for pose, q in DRIVE_Q.items():
            _set_drive(robot, q)
            for _ in range(30):
                scene.step()
            samples = _world_samples(robot)
            _log(
                "G",
                f"{label} gripper @ drive_joint={q}",
                {
                    "mode": label,
                    "pose": pose,
                    "drive_q": q,
                    "parts": samples,
                    "lf_rf_mm": _pairwise_dists(samples, "left_finger.glb", "right_finger.glb"),
                    "lo_ro_mm": _pairwise_dists(
                        samples, "left_outer_knuckle.glb", "right_outer_knuckle.glb"
                    ),
                },
            )
            print(f"\n=== {label} @ {pose} (q={q}) ===")
            for part, info in sorted(samples.items()):
                print(f"  {part}: link={info['link']} rel_link6={info['rel_link6_mm']}")

    # CAD reference: genesis world assembly + global Rz90 + flange shift
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from relocalize_gripper_glb import (  # noqa: E402
        GENESIS_PART_INDICES,
        MOVABLE_SRC,
        _init_genesis,
        align_parts_to_link6_flange,
    )
    from relocalize_arm_glb import bake_glb_genesis_parts  # noqa: E402
    from scipy.spatial.transform import Rotation  # noqa: E402

    baked = bake_glb_genesis_parts(MOVABLE_SRC)
    _, flange = align_parts_to_link6_flange([baked[0].copy()])
    tg = np.array(
        [
            flange["xy_shift_mm"][0] / 1000,
            flange["xy_shift_mm"][1] / 1000,
            flange["z_shift_mm"] / 1000,
        ]
    )
    Rg = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    link6 = trimesh.load(
        Path(__file__).resolve().parents[1]
        / "assets/urdf/xarm6/meshes/xarm6_1305/visual_glb/link6.glb",
        force="mesh",
    )
    link6_c = link6.centroid
    cad_ref = {}
    for part in GRIPPER_PARTS:
        if part == "gripper_g2_static_link6.glb":
            continue
        if part == "base.glb":
            mesh = baked[0].copy()
            mesh.vertices += tg
        else:
            idx = GENESIS_PART_INDICES[part][0]
            mesh = baked[idx].copy()
            mesh.vertices = (mesh.vertices @ Rg.T) + tg
        c = mesh.centroid
        cad_ref[part] = {
            "rel_link6_mm": ((c - link6_c) * 1000).round(2).tolist(),
            "extent_mm": ((mesh.vertices.max(0) - mesh.vertices.min(0)) * 1000).round(2).tolist(),
        }
    _log("H", "CAD assembly reference (global Rz90+flange)", {"parts": cad_ref})
    print("\n=== CAD reference (Rz90+flange, rel link6) ===")
    for part, info in sorted(cad_ref.items()):
        print(f"  {part}: {info['rel_link6_mm']}")

    print(f"\nWrote analysis log: {LOG}")


if __name__ == "__main__":
    main()
