#!/usr/bin/env python3
"""Compare gripper shell (base) orientation: static vs movable in Genesis."""

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _bootstrap  # noqa: F401
import genesis as gs
from relocalize_arm_glb import bake_glb_genesis_parts, mean_surface_distance, _init_genesis
from relocalize_gripper_glb import (
    GENESIS_PART_INDICES,
    MOVABLE_SRC,
    VISUAL_GLB_OUT,
    align_parts_to_link6_flange,
    relocalize_static_assembly,
    _movable_from_static_parts,
    _gripper_link_poses_in_link6,
    _apply_rigid_to_parts,
)
from ufactory.glb_visual import enable_glb_pbr_surfaces, glb_view_surface
from ufactory.paths import xarm6_1305_visual_glb_urdf

LOG = Path(__file__).resolve().parents[1] / ".cursor" / "debug-a21d90.log"
DRIVE_Q = 0.85


def _log(hypothesis_id: str, message: str, data: dict) -> None:
    # #region agent log
    payload = {
        "sessionId": "a21d90",
        "runId": "shell-diagnose",
        "hypothesisId": hypothesis_id,
        "location": "diagnose_gripper_shell.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    # #endregion


def _quat_R(quat) -> np.ndarray:
    q = np.asarray(quat.cpu().numpy()).reshape(4)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def _world_mesh(robot, mesh_names: set[str]) -> dict[str, trimesh.Trimesh]:
    links = {l.name.split("/")[-1]: l for l in robot.links}
    out: dict[str, trimesh.Trimesh] = {}
    for vg in robot.vgeoms:
        p = Path(str((vg.metadata or {}).get("mesh_path", ""))).name
        if p not in mesh_names:
            continue
        link = links[vg.link.name.split("/")[-1]]
        tm = vg.get_trimesh().copy()
        R = _quat_R(link.get_quat())
        t = np.asarray(link.get_pos().cpu().numpy()).reshape(3)
        tm.vertices = tm.vertices @ R.T + t
        out[p] = tm
    return out


def _euler(a: trimesh.Trimesh, b: trimesh.Trimesh) -> list[float]:
    _, nn = cKDTree(b.vertices).query(a.vertices, k=1)
    pa, pb = a.vertices, b.vertices[nn]
    ca, cb = pa.mean(0), pb.mean(0)
    H = (pa - ca).T @ (pb - cb)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return [round(float(x), 2) for x in Rotation.from_matrix(R).as_euler("xyz", degrees=True)]


def _set_gripper(robot, scene, q: float) -> None:
    joints = (
        "drive_joint",
        "left_finger_joint",
        "left_inner_knuckle_joint",
        "right_outer_knuckle_joint",
        "right_finger_joint",
        "right_inner_knuckle_joint",
    )
    jm = {j.name.split("/")[-1]: j for j in robot.joints}
    dofs = [jm[n].dofs_idx_local[0] for n in joints if n in jm]
    if dofs:
        robot.set_dofs_position(np.full(len(dofs), q), dofs)
    for _ in range(30):
        scene.step()


def _runtime_shell_compare() -> None:
    enable_glb_pbr_surfaces()
    gs.init(backend=gs.gpu, logging_level="error")
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
        _set_gripper(rb, sc, DRIVE_Q)
        scenes[movable] = _world_mesh(
            rb,
            {"gripper_g2_static_link6.glb"} if not movable else {"link6/base.glb"},
        )
    static_shell = scenes[False]["gripper_g2_static_link6.glb"]
    movable_base = scenes[True]["base.glb"]
    # isolate shell-like vertices (white case: high z extent region)
    static_case = static_shell.copy()
    mov_case = movable_base.copy()
    surf = mean_surface_distance(mov_case, static_case) * 1000
    ext_s = (static_case.vertices.max(0) - static_case.vertices.min(0)) * 1000
    ext_m = (mov_case.vertices.max(0) - mov_case.vertices.min(0)) * 1000
    _log(
        "S1",
        "runtime world shell static vs movable base",
        {
            "mean_surface_mm": round(float(surf), 2),
            "euler_xyz_deg": _euler(mov_case, static_case),
            "static_extent_mm": ext_s.round(1).tolist(),
            "movable_extent_mm": ext_m.round(1).tolist(),
        },
    )
    print("runtime shell vs base:", surf, "mm euler", _euler(mov_case, static_case))
    print("  static extent", ext_s.round(1).tolist())
    print("  movable extent", ext_m.round(1).tolist())


def _pipeline_base_sweep() -> None:
    _init_genesis()
    baked = bake_glb_genesis_parts(MOVABLE_SRC)
    base_raw = baked[GENESIS_PART_INDICES["base.glb"][0]].copy()
    base_aligned, flange = align_parts_to_link6_flange([base_raw.copy()])
    base_t = np.array(
        [
            flange["xy_shift_mm"][0] / 1000.0,
            flange["xy_shift_mm"][1] / 1000.0,
            flange["z_shift_mm"] / 1000.0,
        ]
    )
    _, static_parts = relocalize_static_assembly("link6", True)
    link_poses = _gripper_link_poses_in_link6()
    static_cands, _ = _movable_from_static_parts(static_parts, link_poses)
    static_case = static_cands["base.glb"]
    on_disk = trimesh.load(VISUAL_GLB_OUT / "base.glb", force="mesh")

    for z_deg in (0, 90, -90, 180):
        parts = _apply_rigid_to_parts(
            [base_raw.copy()],
            rotation=Rotation.from_euler("z", z_deg, degrees=True).as_matrix(),
            translation=base_t,
        )
        mesh = parts[0]
        surf_static = mean_surface_distance(mesh, static_case) * 1000
        surf_disk = mean_surface_distance(on_disk, mesh) * 1000
        _log(
            "S2",
            f"base semantic flange + Rz{z_deg}",
            {
                "z_deg": z_deg,
                "vs_static_case_mm": round(float(surf_static), 2),
                "on_disk_vs_candidate_mm": round(float(surf_disk), 2),
                "euler_vs_static": _euler(mesh, static_case),
            },
        )
        print(f"base Rz{z_deg:4}: vs_static_case={surf_static:.1f}mm euler={_euler(mesh, static_case)}")


def main() -> None:
    if LOG.exists():
        LOG.unlink()
    _runtime_shell_compare()
    _pipeline_base_sweep()
    print(f"log: {LOG}")


if __name__ == "__main__":
    main()
