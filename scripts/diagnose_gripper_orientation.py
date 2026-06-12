#!/usr/bin/env python3
"""Deep orientation diagnosis: static vs movable G2 gripper in Genesis + relocalize pipeline."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "xarm6"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _bootstrap  # noqa: F401
import genesis as gs
from relocalize_arm_glb import bake_glb_genesis_parts, mean_surface_distance
from relocalize_gripper_glb import (
    GENESIS_PART_INDICES,
    GRIPPER_STL_DIR,
    LINK_LOCAL_VISUAL_RPY_DEG,
    MOVABLE_SRC,
    STL_FOR_GLB,
    VISUAL_GLB_OUT,
    _align_centroid_to_stl,
    _apply_link_local_visual_rotation,
    _apply_rigid_to_parts,
    _gripper_link_poses_in_link6,
    _split_assembly_to_link,
    align_parts_to_link6_flange,
)
from ufactory.glb_visual import enable_glb_pbr_surfaces, glb_view_surface
from ufactory.paths import xarm6_1305_visual_glb_urdf

LOG = Path(__file__).resolve().parents[1] / ".cursor" / "debug-a21d90.log"
GRIPPER_JOINTS = (
    "drive_joint",
    "left_finger_joint",
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_finger_joint",
    "right_inner_knuckle_joint",
)
LINKAGE_GLBS = (
    "left_finger.glb",
    "right_finger.glb",
    "left_outer_knuckle.glb",
    "right_outer_knuckle.glb",
    "left_inner_knuckle.glb",
    "right_inner_knuckle.glb",
)
DRIVE_Q = 0.0


def _log(hypothesis_id: str, message: str, data: dict, run_id: str = "orientation-deep") -> None:
    # #region agent log
    payload = {
        "sessionId": "a21d90",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": "diagnose_gripper_orientation.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    # #endregion


def _quat_R(quat) -> np.ndarray:
    q = np.asarray(quat.cpu().numpy() if hasattr(quat, "cpu") else quat).reshape(-1)[:4]
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def _euler_xyz_deg(R: np.ndarray) -> list[float]:
    return [round(float(x), 2) for x in Rotation.from_matrix(R).as_euler("xyz", degrees=True)]


def _sample_points(mesh: trimesh.Trimesh, n: int = 2000) -> np.ndarray:
    if len(mesh.vertices) >= n:
        idx = np.linspace(0, len(mesh.vertices) - 1, n, dtype=int)
        return mesh.vertices[idx]
    return mesh.vertices


def _rigid_delta(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh) -> dict:
    """Proper rotation (Kabsch) from A centroid frame to B."""
    pa = _sample_points(mesh_a)
    pb = _sample_points(mesh_b)
    # nearest-neighbor pairing for orientation estimate
    from scipy.spatial import cKDTree

    _, nn = cKDTree(pb).query(pa, k=1)
    pb_matched = pb[nn]
    ca, cb = pa.mean(0), pb_matched.mean(0)
    H = (pa - ca).T @ (pb_matched - cb)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cb - ca @ R.T
    aligned = pa @ R.T + t
    rms = float(np.linalg.norm(aligned - pb_matched) / np.sqrt(len(pa)))
    return {
        "cost": round(rms, 6),
        "euler_xyz_deg": _euler_xyz_deg(R),
        "translation_mm": [round(float(x) * 1000, 3) for x in t],
        "det_R": round(float(np.linalg.det(R)), 4),
    }


def _set_gripper(robot, scene, q: float) -> None:
    jm = {j.name.split("/")[-1]: j for j in robot.joints}
    dofs = [jm[n].dofs_idx_local[0] for n in GRIPPER_JOINTS if n in jm]
    if dofs:
        robot.set_dofs_position(np.full(len(dofs), q), dofs)
        robot.control_dofs_position(np.full(len(dofs), q), dofs)
    for _ in range(30):
        scene.step()


def _world_meshes(robot, names: set[str]) -> dict[str, trimesh.Trimesh]:
    links = {l.name.split("/")[-1]: l for l in robot.links}
    out: dict[str, trimesh.Trimesh] = {}
    for vg in robot.vgeoms:
        mesh_path = str((vg.metadata or {}).get("mesh_path", ""))
        part = Path(mesh_path).name
        if part not in names:
            continue
        link = links[vg.link.name.split("/")[-1]]
        tm = vg.get_trimesh().copy()
        R = _quat_R(link.get_quat())
        t = np.asarray(link.get_pos().cpu().numpy()).reshape(3)
        tm.vertices = tm.vertices @ R.T + t
        out[part] = tm
    return out


def _genesis_runtime_compare() -> None:
    enable_glb_pbr_surfaces()
    gs.init(backend=gs.gpu, logging_level="error")

    sc = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.01))
    rb = sc.add_entity(
        gs.morphs.URDF(
            file=xarm6_1305_visual_glb_urdf(with_gripper_g2=True, movable=True),
            fixed=True,
        ),
        surface=glb_view_surface(),
    )
    sc.build()
    _set_gripper(rb, sc, DRIVE_Q)
    world = _world_meshes(rb, set(LINKAGE_GLBS))
    links = {l.name.split("/")[-1]: l for l in rb.links}

    # STL collision reference in world frame (ground truth from URDF kinematics)
    for part in LINKAGE_GLBS:
        link_name = part.replace(".glb", "").replace("_", "_")  # unused fallback
        link_map = {
            "left_finger.glb": "left_finger",
            "right_finger.glb": "right_finger",
            "left_outer_knuckle.glb": "left_outer_knuckle",
            "right_outer_knuckle.glb": "right_outer_knuckle",
            "left_inner_knuckle.glb": "left_inner_knuckle",
            "right_inner_knuckle.glb": "right_inner_knuckle",
        }
        lname = link_map[part]
        if part not in world or lname not in links:
            continue
        link = links[lname]
        stl = trimesh.load(GRIPPER_STL_DIR / STL_FOR_GLB[part], force="mesh")
        R = _quat_R(link.get_quat())
        t = np.asarray(link.get_pos().cpu().numpy()).reshape(3)
        stl_world = stl.copy()
        stl_world.vertices = stl.vertices @ R.T + t
        mov = world[part]
        delta = _rigid_delta(mov, stl_world)
        surf_mm = mean_surface_distance(mov, stl_world) * 1000
        _log(
            "R1",
            f"runtime world: movable GLB vs STL collision {part}",
            {"part": part, "link": lname, "mean_surface_mm": round(float(surf_mm), 2), **delta},
        )
        print(f"runtime {part} vs STL: surf={surf_mm:.1f}mm rigid_euler={delta['euler_xyz_deg']}")


def _pipeline_rot_sweep() -> None:
    """Sweep semantic_rot z and link_local rpy; compare each movable GLB to STL in link frame."""
    baked = bake_glb_genesis_parts(MOVABLE_SRC)
    _, flange = align_parts_to_link6_flange([baked[0].copy()])
    base_t = np.array(
        [
            flange["xy_shift_mm"][0] / 1000.0,
            flange["xy_shift_mm"][1] / 1000.0,
            flange["z_shift_mm"] / 1000.0,
        ]
    )
    link_poses = _gripper_link_poses_in_link6()

    for z_sem in (0, 90, -90, 180):
        sem_R = Rotation.from_euler("z", z_sem, degrees=True).as_matrix()
        for glb_name in LINKAGE_GLBS:
            indices = GENESIS_PART_INDICES[glb_name]
            parts = _apply_rigid_to_parts(
                [baked[i] for i in indices],
                rotation=sem_R,
                translation=base_t,
            )
            mesh_link6 = trimesh.util.concatenate(parts)
            mesh_local = _split_assembly_to_link(mesh_link6, link_poses[glb_name])
            stl = trimesh.load(GRIPPER_STL_DIR / STL_FOR_GLB[glb_name], force="mesh")
            mesh_local = _align_centroid_to_stl(mesh_local, stl)
            for z_loc in (0, 90, -90):
                mesh_try = _apply_link_local_visual_rotation(mesh_local, glb_name)
                if z_loc != 0:
                    mesh_try = mesh_try.copy()
                    c = mesh_try.centroid
                    loc_R = Rotation.from_euler("z", z_loc, degrees=True).as_matrix()
                    mesh_try.vertices = (mesh_try.vertices - c) @ loc_R.T + c
                surf = mean_surface_distance(mesh_try, stl) * 1000
                rigid = _rigid_delta(mesh_try, stl)
                _log(
                    "R2",
                    f"pipeline sweep {glb_name}",
                    {
                        "part": glb_name,
                        "semantic_z_deg": z_sem,
                        "extra_link_z_deg": z_loc,
                        "link_local_cfg_z": LINK_LOCAL_VISUAL_RPY_DEG.get(glb_name, (0, 0, 0))[2],
                        "mean_surface_mm": round(float(surf), 2),
                        "rigid_euler_xyz_deg": rigid["euler_xyz_deg"],
                        "rigid_cost": rigid["cost"],
                    },
                )
            # only log best per semantic_z for stdout brevity
        best = []
        for glb_name in LINKAGE_GLBS:
            # read from nothing - recompute best for this z_sem
            pass
    # print summary for current config (z_sem=90, link_local from dict)
    print("\npipeline current config (semantic_z=90, LINK_LOCAL knuckle z90):")
    sem_R = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    for glb_name in LINKAGE_GLBS:
        indices = GENESIS_PART_INDICES[glb_name]
        parts = _apply_rigid_to_parts([baked[i] for i in indices], rotation=sem_R, translation=base_t)
        mesh_local = _split_assembly_to_link(trimesh.util.concatenate(parts), link_poses[glb_name])
        stl = trimesh.load(GRIPPER_STL_DIR / STL_FOR_GLB[glb_name], force="mesh")
        mesh_local = _apply_link_local_visual_rotation(_align_centroid_to_stl(mesh_local, stl), glb_name)
        surf = mean_surface_distance(mesh_local, stl) * 1000
        rigid = _rigid_delta(mesh_local, stl)
        on_disk = trimesh.load(VISUAL_GLB_OUT / glb_name, force="mesh")
        surf_disk = mean_surface_distance(on_disk, stl) * 1000
        rigid_disk = _rigid_delta(on_disk, stl)
        _log(
            "R3",
            f"current pipeline vs on-disk {glb_name}",
            {
                "part": glb_name,
                "recomputed_surface_mm": round(float(surf), 2),
                "recomputed_rigid_euler": rigid["euler_xyz_deg"],
                "on_disk_surface_mm": round(float(surf_disk), 2),
                "on_disk_rigid_euler": rigid_disk["euler_xyz_deg"],
            },
        )
        print(
            f"  {glb_name}: recomputed surf={surf:.1f}mm euler={rigid['euler_xyz_deg']} | "
            f"on_disk surf={surf_disk:.1f}mm euler={rigid_disk['euler_xyz_deg']}"
        )


def main() -> None:
    if LOG.exists():
        LOG.unlink()
    _genesis_runtime_compare()
    _pipeline_rot_sweep()
    print(f"\nlog: {LOG}")


if __name__ == "__main__":
    main()
