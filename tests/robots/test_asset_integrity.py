"""Lightweight checks for public robot asset consistency."""

from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))
from conftest import PROJECT_ROOT


def _public_files() -> set[Path]:
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=PROJECT_ROOT).split(b"\0")[:-1]
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=PROJECT_ROOT,
    ).split(b"\0")[:-1]
    return {PROJECT_ROOT / p.decode() for p in (*tracked, *untracked) if (PROJECT_ROOT / p.decode()).exists()}


def test_public_urdf_mesh_references_exist() -> None:
    urdfs = sorted(p for p in _public_files() if p.suffix == ".urdf" or p.name.endswith(".glb.urdf"))
    missing: list[str] = []
    for urdf in urdfs:
        root = ET.parse(urdf).getroot()
        for mesh in root.findall(".//mesh"):
            filename = mesh.get("filename")
            if not filename or filename.startswith("package://"):
                continue
            target = (urdf.parent / filename).resolve()
            if not target.exists():
                missing.append(f"{urdf.relative_to(PROJECT_ROOT)} -> {filename}")

    assert missing == []


def test_intermediate_assets_and_dev_scripts_are_not_public_files() -> None:
    forbidden_markers = [
        "visual_glb_raw",
        "visual_glb_src",
        "relocalize_metrics.json",
        "assets/urdf/bio_gripper_g2/meshes/visual/bio_gripper_g2.glb",
        "assets/urdf/bio_gripper_g2/meshes/visual/bio_gripper_g2_visual.glb",
        "assets/urdf/lite6_gripper/meshes/collision/gripper_lite.stl",
        "scripts/relocalize_",
        "scripts/diagnose_",
        "scripts/vendor_robot_assets.py",
    ]
    forbidden_script_shapes = [
        ("scripts/generate_", "_combo_urdf.py"),
        ("scripts/verify_", "_assets.py"),
    ]
    public = [str(p.relative_to(PROJECT_ROOT)) for p in _public_files()]
    leaked = [
        p
        for p in public
        if any(marker in p for marker in forbidden_markers)
        or any(p.startswith(prefix) and p.endswith(suffix) for prefix, suffix in forbidden_script_shapes)
    ]
    assert leaked == []


_DYNAMICS_ROBOT_DIRS = (
    ("xarm5", "xarm5_1305"),
    ("xarm6", "xarm6_1305"),
    ("xarm7", "xarm7_1305"),
    ("lite6", "lite6"),
    ("uf850", "uf850"),
)

# Base / combo dynamics URDFs only — GLB visual combo URDFs may reference accessory visual meshes.
_DYNAMICS_URDF_BASENAMES: dict[str, tuple[str, ...]] = {
    "xarm5": ("xarm5_1305.urdf",),
    "xarm6": ("xarm6_1305.urdf", "xarm6_with_gripper.urdf"),
    "xarm7": ("xarm7_1305.urdf",),
    "lite6": ("lite6.urdf", "lite6_with_gripper.urdf", "lite6_with_vacuum_gripper.urdf"),
    "uf850": ("uf850.urdf",),
}


def test_dynamics_urdf_collision_meshes_use_stl_not_visual() -> None:
    """Arm dynamics URDFs must reference collision/*.stl, never visual/ or *.obj."""
    bad: list[str] = []
    for robot_dir, _mesh_prefix in _DYNAMICS_ROBOT_DIRS:
        urdf_dir = PROJECT_ROOT / "assets" / "urdf" / robot_dir
        for basename in _DYNAMICS_URDF_BASENAMES[robot_dir]:
            urdf = urdf_dir / basename
            if not urdf.exists():
                bad.append(f"missing dynamics URDF: {urdf.relative_to(PROJECT_ROOT)}")
                continue
            root = ET.parse(urdf).getroot()
            for collision in root.findall(".//collision"):
                mesh = collision.find(".//mesh")
                if mesh is None:
                    continue
                filename = mesh.get("filename") or ""
                if "/visual/" in filename:
                    bad.append(f"{basename}: collision points to visual mesh {filename}")
                elif filename.endswith(".obj"):
                    bad.append(f"{basename}: collision still uses OBJ {filename}")
                elif "/collision/" in filename and not filename.endswith(".stl"):
                    bad.append(f"{basename}: collision mesh not STL {filename}")
    assert bad == []


def test_no_collision_obj_meshes_in_public_assets() -> None:
    obj_files = sorted(
        p.relative_to(PROJECT_ROOT)
        for p in _public_files()
        if "collision" in str(p) and p.suffix == ".obj"
    )
    assert obj_files == []
