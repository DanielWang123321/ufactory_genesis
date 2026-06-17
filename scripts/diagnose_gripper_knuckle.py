#!/usr/bin/env python3
"""Compare knuckle (传动连杆) assembly: movable vs static in Genesis world frame."""

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
    GRIPPER_STL_DIR,
    KNUCKLE_GLBS,
    MOVABLE_SRC,
    STL_FOR_GLB,
    VISUAL_GLB_OUT,
    _align_centroid_to_stl,
    _apply_rigid_to_parts,
    _gripper_link_poses_in_link6,
    _split_assembly_to_link,
    _ee_glb_path,
    align_parts_to_ee_flange,
    relocalize_static_assembly,
    _movable_from_static_parts,
    refine_rigid_to_stl,
)
from ufactory.glb_visual import enable_glb_pbr_surfaces, glb_view_surface
from ufactory.paths import xarm6_1305_visual_glb_urdf

LOG = Path(__file__).resolve().parents[1] / ".cursor" / "debug-a21d90.log"
KNUCKLE_NAMES = tuple(KNUCKLE_GLBS)
POSES = {"open": 0.0, "closed": 0.85}


def _log(hypothesis_id: str, message: str, data: dict) -> None:
    # #region agent log
    payload = {
        "sessionId": "a21d90",
        "runId": "knuckle-diagnose",
        "hypothesisId": hypothesis_id,
        "location": "diagnose_gripper_knuckle.py",
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


def _sample_cloud(robot, names: set[str], n: int = 6000) -> np.ndarray:
    links = {l.name.split("/")[-1]: l for l in robot.links}
    chunks = []
    for vg in robot.vgeoms:
        p = Path(str((vg.metadata or {}).get("mesh_path", ""))).name
        if p not in names:
            continue
        l = links[vg.link.name.split("/")[-1]]
        v = vg.get_trimesh().vertices
        R = _quat_R(l.get_quat())
        t = np.asarray(l.get_pos().cpu().numpy()).reshape(3)
        chunks.append(v @ R.T + t)
    pts = np.vstack(chunks)
    if len(pts) > n:
        pts = pts[np.random.default_rng(0).choice(len(pts), n, replace=False)]
    return pts


def _chamfer(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    ta, tb = cKDTree(b), cKDTree(a)
    da, _ = ta.query(a, k=1)
    db, _ = tb.query(b, k=1)
    return {
        "mean_mm": float((da.mean() + db.mean()) / 2 * 1000),
        "p95_mm": float(max(np.percentile(da, 95), np.percentile(db, 95)) * 1000),
    }


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
        robot.control_dofs_position(np.full(len(dofs), q), dofs)
    for _ in range(30):
        scene.step()


def _runtime_knuckle_vs_static() -> None:
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

    finger_names = {"left_finger.glb", "right_finger.glb"}
    for pose, q in POSES.items():
        _set_gripper(*scenes[False], q)
        _set_gripper(*scenes[True], q)
        ps = _sample_cloud(scenes[False][0], {"gripper_g2_static_link6.glb"})
        pm_all = _sample_cloud(
            scenes[True][0],
            finger_names | set(KNUCKLE_NAMES) | {"base.glb"},
        )
        pm_knuckle = _sample_cloud(scenes[True][0], set(KNUCKLE_NAMES))
        pm_finger = _sample_cloud(scenes[True][0], finger_names)
        # knuckle-only region in static: whole-part reference in link6 frame
        try:
            _init_genesis()
        except gs.GenesisException:
            pass
        _, static_parts = relocalize_static_assembly("link6", True)
        link_poses = _gripper_link_poses_in_link6()
        from relocalize_gripper_glb import _knuckle_from_static_whole

        static_knuckle_meshes = _knuckle_from_static_whole(static_parts, link_poses)
        static_knuckle_pts = []
        for glb in KNUCKLE_NAMES:
            sk = static_knuckle_meshes[glb]
            T = link_poses[glb]
            v = sk.vertices
            ones = np.ones((len(v), 1))
            static_knuckle_pts.append((np.hstack([v, ones]) @ T.T)[:, :3])
        static_knuckle = np.vstack(static_knuckle_pts)
        if len(static_knuckle) > 6000:
            static_knuckle = static_knuckle[
                np.random.default_rng(1).choice(len(static_knuckle), 6000, replace=False)
            ]

        metrics = {
            "movable_all_vs_static": _chamfer(pm_all, ps),
            "movable_knuckle_vs_static_knuckle_ref": _chamfer(pm_knuckle, static_knuckle),
            "movable_finger_vs_static": _chamfer(pm_finger, ps),
        }
        _log("K1", f"runtime @ {pose}", {"pose": pose, "drive_q": q, **metrics})
        print(f"\n{pose}: knuckle vs static_ref {metrics['movable_knuckle_vs_static_knuckle_ref']}")


def _pipeline_knuckle_variants() -> None:
    try:
        _init_genesis()
    except gs.GenesisException:
        pass
    baked = bake_glb_genesis_parts(MOVABLE_SRC)
    _, flange = align_parts_to_ee_flange([baked[0].copy()], _ee_glb_path("link6"), "link6")
    base_t = np.array(
        [flange["xy_shift_mm"][0] / 1000, flange["xy_shift_mm"][1] / 1000, flange["z_shift_mm"] / 1000]
    )
    sem_R = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    link_poses = _gripper_link_poses_in_link6()
    _, static_parts = relocalize_static_assembly("link6", True)
    static_cands, _ = _movable_from_static_parts(static_parts, link_poses)

    for glb in KNUCKLE_NAMES:
        idx = GENESIS_PART_INDICES[glb]
        stl = trimesh.load(GRIPPER_STL_DIR / STL_FOR_GLB[glb], force="mesh")
        static_ref = static_cands[glb]
        parts = _apply_rigid_to_parts([baked[i] for i in idx], rotation=sem_R, translation=base_t)
        mesh0 = _align_centroid_to_stl(
            _split_assembly_to_link(trimesh.util.concatenate(parts), link_poses[glb]), stl
        )
        variants = {"semantic_centroid": mesh0}
        aligned, meta = refine_rigid_to_stl([mesh0], stl)
        if not meta.get("rejected"):
            variants["icp_stl"] = aligned[0]
        aligned2, meta2 = refine_rigid_to_stl([mesh0], static_ref)
        if not meta2.get("rejected"):
            variants["icp_static"] = aligned2[0]
        on_disk = trimesh.load(VISUAL_GLB_OUT / glb, force="mesh")
        for name, mesh in variants.items():
            _log(
                "K2",
                f"pipeline {glb} {name}",
                {
                    "part": glb,
                    "variant": name,
                    "vs_stl_mm": round(mean_surface_distance(mesh, stl) * 1000, 2),
                    "vs_static_ref_mm": round(mean_surface_distance(mesh, static_ref) * 1000, 2),
                    "on_disk_vs_variant_mm": round(mean_surface_distance(on_disk, mesh) * 1000, 2),
                },
            )
        print(
            f"{glb}: on_disk vs semantic {mean_surface_distance(on_disk, mesh0)*1000:.1f}mm | "
            f"icp_stl static {variants.get('icp_stl') is not None}"
        )


def main() -> None:
    if LOG.exists():
        LOG.unlink()
    _runtime_knuckle_vs_static()
    try:
        _pipeline_knuckle_variants()
    except gs.GenesisException:
        pass
    print(f"log: {LOG}")


if __name__ == "__main__":
    main()
