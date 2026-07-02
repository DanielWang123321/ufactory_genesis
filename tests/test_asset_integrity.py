"""Lightweight checks for public robot asset consistency."""

from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def test_xarm6_1305_collision_meshes_use_collision_objs() -> None:
    bad: list[str] = []
    for urdf in sorted((PROJECT_ROOT / "assets" / "urdf" / "xarm6").glob("*.urdf")):
        if "1305" not in urdf.name and urdf.name != "xarm6_with_gripper.urdf":
            continue
        root = ET.parse(urdf).getroot()
        for link in root.findall("link"):
            for collision in link.findall("collision"):
                mesh = collision.find(".//mesh")
                if mesh is None:
                    continue
                filename = mesh.get("filename") or ""
                if filename.startswith("meshes/xarm6_1305/visual/"):
                    bad.append(f"{urdf.name}: {filename}")

    assert bad == []
