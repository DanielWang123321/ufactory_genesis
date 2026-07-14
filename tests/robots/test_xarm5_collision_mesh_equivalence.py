"""xArm collision mesh OBJ baseline vs STL geometry equivalence."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

import sys

_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))
from conftest import PROJECT_ROOT

BASELINE_COMMIT = "54aacdd6f41b91cb9e83ec7eace9c7b53eb59188"
ROBOT_MESHES = {
    "xarm5": ("xarm5_1305", ("link1", "link2", "link3", "link4", "link5", "link_base")),
    "xarm6": (
        "xarm6_1305",
        ("link1", "link2", "link3", "link4", "link5", "link6", "link_base"),
    ),
    "xarm7": (
        "xarm7_1305",
        ("link1", "link2", "link3", "link4", "link5", "link6", "link7", "link_base"),
    ),
}
TEST_CASES = [(robot_key, link_name) for robot_key, (_, links) in ROBOT_MESHES.items() for link_name in links]

HAUSDORFF_TOL_M = 1e-4
VOLUME_REL_TOL = 0.01


def _collision_dir(robot_key: str) -> Path:
    variant = ROBOT_MESHES[robot_key][0]
    return PROJECT_ROOT / "assets" / "urdf" / robot_key / "meshes" / variant / "collision"


def _load_baseline_obj(robot_key: str, link_name: str) -> trimesh.Trimesh:
    variant = ROBOT_MESHES[robot_key][0]
    rel = f"assets/urdf/{robot_key}/meshes/{variant}/collision/{link_name}.obj"
    try:
        obj_bytes = subprocess.check_output(
            ["git", "show", f"{BASELINE_COMMIT}:{rel}"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        pytest.fail(f"baseline OBJ unavailable: {BASELINE_COMMIT}:{rel}; {exc.stderr.decode().strip()}")
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp:
        tmp.write(obj_bytes)
        tmp_path = Path(tmp.name)
    try:
        return trimesh.load(tmp_path, force="mesh", process=False)
    finally:
        tmp_path.unlink(missing_ok=True)


def _hausdorff_m(a: trimesh.Trimesh, b: trimesh.Trimesh) -> float:
    da = trimesh.proximity.ProximityQuery(a).signed_distance(b.vertices)
    db = trimesh.proximity.ProximityQuery(b).signed_distance(a.vertices)
    return float(max(np.abs(da).max(), np.abs(db).max()))


@pytest.mark.parametrize(("robot_key", "link_name"), TEST_CASES)
def test_xarm_collision_obj_stl_equivalent(robot_key: str, link_name: str) -> None:
    stl_path = _collision_dir(robot_key) / f"{link_name}.stl"
    assert stl_path.exists(), f"missing working-tree STL: {stl_path}"

    obj_mesh = _load_baseline_obj(robot_key, link_name)
    stl_mesh = trimesh.load(stl_path, force="mesh", process=False)

    # STL export duplicates per-face vertices; compare topology + surface distance.
    assert len(obj_mesh.faces) == len(stl_mesh.faces)

    vol_obj = float(abs(obj_mesh.volume))
    vol_stl = float(abs(stl_mesh.volume))
    if vol_obj > 1e-9:
        rel_vol = abs(vol_obj - vol_stl) / vol_obj
        assert rel_vol < VOLUME_REL_TOL, f"{link_name} volume rel diff {rel_vol:.4f}"

    assert _hausdorff_m(obj_mesh, stl_mesh) < HAUSDORFF_TOL_M
