"""Lightweight checks for public robot asset consistency."""

from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

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
        "assets/urdf/lite6_gripper/meshes/collision/gripper_lite.stl",
        "scripts/relocalize_",
        "scripts/diagnose_",
        "scripts/vendor_robot_assets.py",
        "bio_gripper_g2_visual_link5.glb",
        "bio_gripper_g2_visual_link6.glb",
        "bio_gripper_g2_visual_link7.glb",
        "bio_gripper_g2/meshes/visual/visual_glb/link5/",
        "bio_gripper_g2/meshes/visual/visual_glb/link6/",
        "bio_gripper_g2/meshes/visual/visual_glb/link7/",
        "gripper_g2_static_link5.glb",
        "gripper_g2_static_link6.glb",
        "gripper_g2_static_link7.glb",
        "gripper_g2/meshes/visual/visual_glb/link5/",
        "gripper_g2/meshes/visual/visual_glb/link6/",
        "gripper_g2/meshes/visual/visual_glb/link7/",
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


_REGISTERED_ARM_URDFS: dict[str, tuple[int, str, tuple[str, ...]]] = {
    "xarm5": (
        5,
        "xarm5_1305",
        (
            "xarm5_1305.urdf",
            "xarm5_1305_visual.glb.urdf",
            "xarm5_1305_g2_visual.urdf",
            "xarm5_1305_g2_movable_visual.urdf",
            "xarm5_1305_bio_gripper_g2_visual.glb.urdf",
            "xarm5_1305_bio_gripper_g2_movable_visual.glb.urdf",
        ),
    ),
    "xarm6": (
        6,
        "xarm6_1305",
        (
            "xarm6_1305.urdf",
            "xarm6_1305_visual.glb.urdf",
            "xarm6_1305_g2_visual.urdf",
            "xarm6_1305_g2_movable_visual.urdf",
            "xarm6_1305_bio_gripper_g2_visual.glb.urdf",
            "xarm6_1305_bio_gripper_g2_movable_visual.glb.urdf",
            "xarm6_with_gripper.urdf",
        ),
    ),
    "xarm7": (
        7,
        "xarm7_1305",
        (
            "xarm7_1305.urdf",
            "xarm7_1305_visual.glb.urdf",
            "xarm7_1305_g2_visual.urdf",
            "xarm7_1305_g2_movable_visual.urdf",
            "xarm7_1305_bio_gripper_g2_visual.glb.urdf",
            "xarm7_1305_bio_gripper_g2_movable_visual.glb.urdf",
        ),
    ),
    "lite6": (
        6,
        "lite6",
        (
            "lite6.urdf",
            "lite6_visual.glb.urdf",
            "lite6_gripper_visual.glb.urdf",
            "lite6_gripper_movable_visual.glb.urdf",
            "lite6_gripper_reversed_movable_visual.glb.urdf",
            "lite6_vacuum_gripper_visual.glb.urdf",
            "lite6_with_gripper.urdf",
            "lite6_with_vacuum_gripper.urdf",
        ),
    ),
    "uf850": (
        6,
        "uf850",
        (
            "uf850.urdf",
            "uf850_visual.glb.urdf",
            "uf850_g2_visual.urdf",
            "uf850_g2_movable_visual.urdf",
            "uf850_bio_gripper_g2_visual.glb.urdf",
            "uf850_bio_gripper_g2_movable_visual.glb.urdf",
        ),
    ),
}

_BIO_GRIPPER_G2_LINKS = (
    "bio_gripper_g2_base_link",
    "bio_gripper_g2_left_finger",
    "bio_gripper_g2_right_finger",
)

# Canonical Bio G2 visual GLBs are shared across arms; collision origins are unified.
_BIO_GRIPPER_G2_COLLISION_ORIGINS = {
    "bio_gripper_g2_base_link": (-0.022475, -0.00083, -0.004696),
    "bio_gripper_g2_left_finger": (-0.01287, -0.011316, -0.002219),
    "bio_gripper_g2_right_finger": (-0.01379, 0.011745, -0.001039),
}

_BIO_GRIPPER_G2_COLLISION_ORIGIN_URDFS = (
    PROJECT_ROOT / "assets" / "urdf" / "bio_gripper_g2" / "bio_gripper_g2_movable_visual.urdf",
    PROJECT_ROOT / "assets" / "urdf" / "xarm5" / "xarm5_1305_bio_gripper_g2_visual.glb.urdf",
    PROJECT_ROOT / "assets" / "urdf" / "xarm5" / "xarm5_1305_bio_gripper_g2_movable_visual.glb.urdf",
    PROJECT_ROOT / "assets" / "urdf" / "xarm6" / "xarm6_1305_bio_gripper_g2_visual.glb.urdf",
    PROJECT_ROOT / "assets" / "urdf" / "xarm6" / "xarm6_1305_bio_gripper_g2_movable_visual.glb.urdf",
    PROJECT_ROOT / "assets" / "urdf" / "xarm7" / "xarm7_1305_bio_gripper_g2_visual.glb.urdf",
    PROJECT_ROOT / "assets" / "urdf" / "xarm7" / "xarm7_1305_bio_gripper_g2_movable_visual.glb.urdf",
    PROJECT_ROOT / "assets" / "urdf" / "uf850" / "uf850_bio_gripper_g2_visual.glb.urdf",
    PROJECT_ROOT / "assets" / "urdf" / "uf850" / "uf850_bio_gripper_g2_movable_visual.glb.urdf",
)

_BIO_GRIPPER_G2_MOVABLE_COLLISION_ALIGNMENT_URDFS = (
    PROJECT_ROOT / "assets" / "urdf" / "bio_gripper_g2" / "bio_gripper_g2_movable_visual.urdf",
    PROJECT_ROOT / "assets" / "urdf" / "xarm5" / "xarm5_1305_bio_gripper_g2_movable_visual.glb.urdf",
    PROJECT_ROOT / "assets" / "urdf" / "xarm6" / "xarm6_1305_bio_gripper_g2_movable_visual.glb.urdf",
    PROJECT_ROOT / "assets" / "urdf" / "xarm7" / "xarm7_1305_bio_gripper_g2_movable_visual.glb.urdf",
    PROJECT_ROOT / "assets" / "urdf" / "uf850" / "uf850_bio_gripper_g2_movable_visual.glb.urdf",
)


def _arm_link_names(dof: int) -> set[str]:
    return {"link_base", *(f"link{i}" for i in range(1, dof + 1))}


def _xyz_tuple(value: str | None) -> tuple[float, float, float]:
    parts = tuple(float(part) for part in (value or "0 0 0").split())
    if len(parts) != 3:
        raise ValueError(f"expected xyz/rpy triplet, got {value!r}")
    return parts


def _rounded_xyz(value: str | None) -> tuple[float, float, float]:
    return tuple(round(part, 6) for part in _xyz_tuple(value))


def _mesh_vertices(mesh_path: Path, trimesh_module) -> np.ndarray:
    """Load mesh vertices, including Draco-compressed GLB visuals."""
    if mesh_path.suffix.lower() == ".glb":
        try:
            from pygltflib import GLTF2
            import DracoPy
        except ImportError:
            DracoPy = None  # type: ignore[assignment]
            GLTF2 = None  # type: ignore[assignment]
        if GLTF2 is not None and DracoPy is not None:
            glb = GLTF2().load(str(mesh_path))
            if glb.extensionsUsed and "KHR_draco_mesh_compression" in glb.extensionsUsed:
                blob = glb.binary_blob() or b""
                chunks: list[np.ndarray] = []
                for mesh in glb.meshes or []:
                    for prim in mesh.primitives or []:
                        extensions = prim.extensions or {}
                        draco = extensions.get("KHR_draco_mesh_compression")
                        if not draco:
                            continue
                        view = glb.bufferViews[draco["bufferView"]]
                        offset = view.byteOffset or 0
                        decoded = DracoPy.decode(blob[offset : offset + view.byteLength])
                        chunks.append(np.asarray(decoded.points, dtype=np.float64))
                if chunks:
                    return np.vstack(chunks)

    scene = trimesh_module.load(mesh_path, force="scene", process=False)
    vertices: list[np.ndarray] = []
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph.get(node_name)
        geometry = scene.geometry[geometry_name]
        local = np.asarray(geometry.vertices, dtype=np.float64)
        vertices.append(local @ transform[:3, :3].T + transform[:3, 3])
    if vertices:
        return np.vstack(vertices)
    mesh = trimesh_module.load(mesh_path, process=False)
    return np.asarray(mesh.vertices, dtype=np.float64)


def _collision_mesh_filenames(urdf: Path) -> list[str]:
    root = ET.parse(urdf).getroot()
    filenames: list[str] = []
    for collision in root.findall(".//collision"):
        mesh = collision.find(".//mesh")
        if mesh is not None and mesh.get("filename"):
            filenames.append(mesh.get("filename") or "")
    return filenames


def _mesh_filenames(urdf: Path) -> list[str]:
    root = ET.parse(urdf).getroot()
    return [mesh.get("filename") or "" for mesh in root.findall(".//mesh") if mesh.get("filename")]


def _collision_meshes_by_link(urdf: Path) -> dict[str, list[str]]:
    root = ET.parse(urdf).getroot()
    out: dict[str, list[str]] = {}
    for link in root.findall("link"):
        link_name = link.get("name") or ""
        for collision in link.findall("collision"):
            mesh = collision.find(".//mesh")
            if mesh is not None and mesh.get("filename"):
                out.setdefault(link_name, []).append(mesh.get("filename") or "")
    return out


def _collision_links(urdf: Path) -> set[str]:
    root = ET.parse(urdf).getroot()
    return {
        link.get("name") or ""
        for link in root.findall("link")
        if link.get("name") and link.find("collision") is not None
    }


def _collision_boxes_by_link(urdf: Path) -> dict[str, tuple[tuple[str, str], ...]]:
    """Map link name -> ((origin_xyz, box_size), ...) for box collision geoms."""
    root = ET.parse(urdf).getroot()
    out: dict[str, tuple[tuple[str, str], ...]] = {}
    for link in root.findall("link"):
        link_name = link.get("name") or ""
        boxes: list[tuple[str, str]] = []
        for collision in link.findall("collision"):
            box = collision.find("./geometry/box")
            if box is None:
                continue
            origin = collision.find("origin")
            boxes.append((origin.get("xyz") if origin is not None else "0 0 0", box.get("size") or ""))
        if boxes:
            out[link_name] = tuple(boxes)
    return out


def _collision_origin_pose_by_link(
    urdf: Path,
) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
    root = ET.parse(urdf).getroot()
    out: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {}
    for link in root.findall("link"):
        link_name = link.get("name") or ""
        if link_name not in _BIO_GRIPPER_G2_LINKS:
            continue
        origin = link.find("./collision/origin")
        out[link_name] = (
            _rounded_xyz(origin.get("xyz") if origin is not None else None),
            _rounded_xyz(origin.get("rpy") if origin is not None else None),
        )
    return out


def _resolved_collision_meshes_by_link(urdf: Path) -> dict[str, tuple[Path, ...]]:
    out: dict[str, tuple[Path, ...]] = {}
    for link, filenames in _collision_meshes_by_link(urdf).items():
        out[link] = tuple((urdf.parent / filename).resolve() for filename in filenames)
    return out


def _non_fixed_joint_children(urdf: Path) -> set[str]:
    root = ET.parse(urdf).getroot()
    children: set[str] = set()
    for joint in root.findall("joint"):
        if joint.get("type") == "fixed":
            continue
        child = joint.find("child")
        if child is not None and child.get("link"):
            children.add(child.get("link") or "")
    return children


def _non_fixed_joint_names(urdf: Path) -> set[str]:
    root = ET.parse(urdf).getroot()
    return {joint.get("name") or "" for joint in root.findall("joint") if joint.get("type") != "fixed"}


def test_registered_arm_collision_meshes_use_visual_stl() -> None:
    """Registered arm links use visual STL as collision; accessories may keep collision STL."""
    bad: list[str] = []
    for robot_dir, (dof, mesh_prefix, basenames) in _REGISTERED_ARM_URDFS.items():
        urdf_dir = PROJECT_ROOT / "assets" / "urdf" / robot_dir
        arm_links = _arm_link_names(dof)
        for basename in basenames:
            urdf = urdf_dir / basename
            if not urdf.exists():
                bad.append(f"missing registered URDF: {urdf.relative_to(PROJECT_ROOT)}")
                continue
            root = ET.parse(urdf).getroot()
            seen_arm_links: set[str] = set()
            for link in root.findall("link"):
                link_name = link.get("name") or ""
                for collision in link.findall("collision"):
                    mesh = collision.find(".//mesh")
                    if mesh is None:
                        continue
                    filename = mesh.get("filename") or ""
                    if filename.endswith(".obj"):
                        bad.append(f"{basename}: collision still uses OBJ {filename}")
                    elif "/collision/" in filename and not filename.endswith(".stl"):
                        bad.append(f"{basename}: collision mesh not STL {filename}")
                    if link_name not in arm_links:
                        continue
                    seen_arm_links.add(link_name)
                    expected = f"meshes/{mesh_prefix}/visual/{link_name}.stl"
                    if filename != expected:
                        bad.append(f"{basename}: {link_name} collision should be {expected}, got {filename}")
            missing_arm = sorted(arm_links - seen_arm_links)
            if missing_arm:
                bad.append(f"{basename}: missing arm collision links {missing_arm}")
    assert bad == []


def test_bio_gripper_g2_collision_mesh_basenames_are_unique_in_combos() -> None:
    """MuJoCo keys URDF mesh assets by basename, so Bio G2 combo basenames must not collide."""
    bad: list[str] = []
    combo_urdfs = sorted((PROJECT_ROOT / "assets" / "urdf").glob("*/*bio_gripper_g2*.urdf"))
    for urdf in combo_urdfs:
        names = [Path(filename).name for filename in _collision_mesh_filenames(urdf)]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            bad.append(f"{urdf.relative_to(PROJECT_ROOT)}: duplicate collision mesh basename(s) {duplicates}")
    assert bad == []


def test_bio_gripper_g2_urdfs_do_not_reference_legacy_short_stl_names() -> None:
    legacy_refs = {
        "meshes/visual/link_base.stl",
        "meshes/visual/left_finger.stl",
        "meshes/visual/right_finger.stl",
        "../bio_gripper_g2/meshes/visual/link_base.stl",
        "../bio_gripper_g2/meshes/visual/left_finger.stl",
        "../bio_gripper_g2/meshes/visual/right_finger.stl",
    }
    bad: list[str] = []
    bio_urdfs = sorted((PROJECT_ROOT / "assets" / "urdf").glob("*/*bio_gripper_g2*.urdf"))
    for urdf in bio_urdfs:
        for filename in _mesh_filenames(urdf):
            if filename in legacy_refs:
                bad.append(f"{urdf.relative_to(PROJECT_ROOT)}: legacy Bio G2 STL reference {filename}")
    assert bad == []


def test_bio_gripper_g2_collision_origins_align_stl_to_glb_reference() -> None:
    bad: list[str] = []
    for urdf in _BIO_GRIPPER_G2_COLLISION_ORIGIN_URDFS:
        actual = _collision_origin_pose_by_link(urdf)
        for link_name, expected_xyz in _BIO_GRIPPER_G2_COLLISION_ORIGINS.items():
            pose = actual.get(link_name)
            if pose is None:
                bad.append(f"{urdf.relative_to(PROJECT_ROOT)}: missing collision origin for {link_name}")
                continue
            actual_xyz, actual_rpy = pose
            if actual_xyz != expected_xyz:
                bad.append(
                    f"{urdf.relative_to(PROJECT_ROOT)}: {link_name} collision origin xyz "
                    f"should be {expected_xyz}, got {actual_xyz}"
                )
            if actual_rpy != (0.0, 0.0, 0.0):
                bad.append(
                    f"{urdf.relative_to(PROJECT_ROOT)}: {link_name} collision origin rpy "
                    f"should be (0.0, 0.0, 0.0), got {actual_rpy}"
                )
    assert bad == []


def test_bio_gripper_g2_movable_collision_bbox_centers_match_visual_glb() -> None:
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("DracoPy")
    pytest.importorskip("pygltflib")
    bad: list[str] = []
    for urdf in _BIO_GRIPPER_G2_MOVABLE_COLLISION_ALIGNMENT_URDFS:
        root = ET.parse(urdf).getroot()
        for link in root.findall("link"):
            link_name = link.get("name") or ""
            if link_name not in _BIO_GRIPPER_G2_LINKS:
                continue
            visual_mesh = link.find("./visual/geometry/mesh")
            collision = link.find("collision")
            collision_mesh = collision.find("./geometry/mesh") if collision is not None else None
            collision_origin = collision.find("origin") if collision is not None else None
            if visual_mesh is None or collision_mesh is None:
                bad.append(f"{urdf.relative_to(PROJECT_ROOT)}: {link_name} missing visual/collision mesh")
                continue
            visual_vertices = _mesh_vertices((urdf.parent / (visual_mesh.get("filename") or "")).resolve(), trimesh)
            collision_vertices = _mesh_vertices(
                (urdf.parent / (collision_mesh.get("filename") or "")).resolve(),
                trimesh,
            )
            visual_center = (visual_vertices.min(axis=0) + visual_vertices.max(axis=0)) / 2.0
            collision_center = (collision_vertices.min(axis=0) + collision_vertices.max(axis=0)) / 2.0
            collision_center += np.asarray(
                _xyz_tuple(collision_origin.get("xyz") if collision_origin is not None else None)
            )
            delta_mm = float(np.linalg.norm((collision_center - visual_center) * 1000.0))
            if delta_mm > 1.0:
                bad.append(
                    f"{urdf.relative_to(PROJECT_ROOT)}: {link_name} collision bbox center is "
                    f"{delta_mm:.3f} mm from visual GLB center"
                )
    assert bad == []


def test_bio_gripper_g2_visual_glb_budget() -> None:
    """High-poly + native Draco should keep Bio G2 visual GLBs under 1 MiB."""
    package_dir = PROJECT_ROOT / "assets" / "urdf" / "bio_gripper_g2"
    visual_dir = package_dir / "meshes" / "visual"
    glbs = sorted(
        path
        for path in visual_dir.rglob("*.glb")
        if "visual_glb_src" not in path.parts and "visual_glb_raw" not in path.parts
    )
    assert glbs, "expected Bio G2 visual GLBs"
    glb_total = sum(path.stat().st_size for path in glbs)
    # Same tris as pre-slim hipoly; native draco_encoder ~0.65 MB vs DracoPy ~5.5 MB.
    # Collision STL stay at original density; only GLB size is the optimization target.
    assert glb_total < 1_000_000, f"Bio G2 visual GLBs total {glb_total} bytes; expected < 1 MB"
    names = {path.name for path in glbs}
    assert "bio_gripper_g2_visual.glb" in names
    assert "bio_gripper_g2_base.glb" in names
    assert "bio_gripper_g2_left_finger.glb" in names
    assert "bio_gripper_g2_right_finger.glb" in names
    assert not any(name.startswith("bio_gripper_g2_visual_link") for name in names)


def test_gripper_g2_visual_glb_budget() -> None:
    """Canonical Gripper G2 visual GLBs (native Draco) must stay under 1 MiB."""
    package_dir = PROJECT_ROOT / "assets" / "urdf" / "gripper_g2"
    visual_dir = package_dir / "meshes" / "visual"
    glbs = sorted(
        path
        for path in visual_dir.rglob("*.glb")
        if "visual_glb_src" not in path.parts and "visual_glb_raw" not in path.parts
    )
    assert glbs, "expected Gripper G2 visual GLBs"
    glb_total = sum(path.stat().st_size for path in glbs)
    assert glb_total < 1_000_000, f"Gripper G2 visual GLBs total {glb_total} bytes; expected < 1 MB"
    names = {path.name for path in glbs}
    assert "gripper_g2_static.glb" in names
    assert "base.glb" in names
    assert "left_finger.glb" in names
    assert "right_finger.glb" in names
    assert not any(name.startswith("gripper_g2_static_link") for name in names)
    assert not any(part in {"link5", "link6", "link7"} for path in glbs for part in path.parts)


def test_lite6_gripper_visual_glb_budget() -> None:
    """Lite6 gripper visual GLBs (native Draco) must stay under 1 MiB."""
    package_dir = PROJECT_ROOT / "assets" / "urdf" / "lite6_gripper"
    visual_dir = package_dir / "meshes" / "visual"
    glbs = sorted(visual_dir.rglob("*.glb"))
    assert glbs, "expected Lite6 gripper visual GLBs"
    glb_total = sum(path.stat().st_size for path in glbs)
    assert glb_total < 1_000_000, f"Lite6 gripper visual GLBs total {glb_total} bytes; expected < 1 MB"
    names = {path.name for path in glbs}
    assert "lite6_gripper_static_link6.glb" in names
    assert "shell.glb" in names
    assert "finger1.glb" in names
    assert "finger2.glb" in names


def test_lite6_vacuum_gripper_visual_glb_budget() -> None:
    """Lite6 vacuum gripper visual GLB (native Draco) must stay under 1 MiB."""
    package_dir = PROJECT_ROOT / "assets" / "urdf" / "lite6_vacuum_gripper"
    visual_dir = package_dir / "meshes" / "visual"
    glbs = sorted(visual_dir.rglob("*.glb"))
    assert glbs, "expected Lite6 vacuum gripper visual GLBs"
    glb_total = sum(path.stat().st_size for path in glbs)
    assert glb_total < 1_000_000, f"Lite6 vacuum visual GLBs total {glb_total} bytes; expected < 1 MB"
    names = {path.name for path in glbs}
    assert "lite6_vacuum_gripper_visual_link6.glb" in names


@pytest.mark.parametrize(
    ("arm", "visual_glb_dir", "budget", "expected_names"),
    [
        (
            "xarm5",
            "assets/urdf/xarm5/meshes/xarm5_1305/visual_glb",
            1_500_000,
            {"link_base.glb", "link1.glb", "link2.glb", "link3.glb", "link4.glb", "link5.glb"},
        ),
        (
            "xarm6",
            "assets/urdf/xarm6/meshes/xarm6_1305/visual_glb",
            1_500_000,
            {
                "link_base.glb",
                "link1.glb",
                "link2.glb",
                "link3.glb",
                "link4.glb",
                "link5.glb",
                "link6.glb",
            },
        ),
        (
            "xarm7",
            "assets/urdf/xarm7/meshes/xarm7_1305/visual_glb",
            1_000_000,
            {
                "link_base.glb",
                "link1.glb",
                "link2.glb",
                "link3.glb",
                "link4.glb",
                "link5.glb",
                "link6.glb",
                "link7.glb",
            },
        ),
        (
            "uf850",
            "assets/urdf/uf850/meshes/uf850/visual_glb",
            1_000_000,
            {
                "link_base.glb",
                "link1.glb",
                "link2.glb",
                "link3.glb",
                "link4.glb",
                "link5.glb",
                "link6.glb",
            },
        ),
        (
            "lite6",
            "assets/urdf/lite6/meshes/lite6/visual_glb",
            1_000_000,
            {
                "link_base.glb",
                "link1.glb",
                "link2.glb",
                "link3.glb",
                "link4.glb",
                "link5.glb",
                "link6.glb",
            },
        ),
    ],
)
def test_arm_body_visual_glb_budget(
    arm: str,
    visual_glb_dir: str,
    budget: int,
    expected_names: set[str],
) -> None:
    """Arm-body visual GLBs (native Draco, STL untouched) stay within per-arm budgets."""
    visual_dir = PROJECT_ROOT / visual_glb_dir
    glbs = sorted(visual_dir.glob("*.glb"))
    assert glbs, f"expected {arm} visual GLBs under {visual_glb_dir}"
    glb_total = sum(path.stat().st_size for path in glbs)
    assert glb_total < budget, f"{arm} visual GLBs total {glb_total} bytes; expected < {budget}"
    names = {path.name for path in glbs}
    assert names == expected_names, f"{arm} GLB names {names} != {expected_names}"


def test_accessory_collision_meshes_are_stl_files() -> None:
    urdfs = (
        PROJECT_ROOT / "assets" / "urdf" / "gripper_g2" / "gripper_g2.urdf",
        PROJECT_ROOT / "assets" / "urdf" / "gripper_g2" / "gripper_g2_movable_visual.urdf",
        PROJECT_ROOT / "assets" / "urdf" / "bio_gripper_g2" / "bio_gripper_g2.urdf",
        PROJECT_ROOT / "assets" / "urdf" / "bio_gripper_g2" / "bio_gripper_g2_movable_visual.urdf",
        PROJECT_ROOT / "assets" / "urdf" / "lite6_gripper" / "lite6_gripper.urdf",
        PROJECT_ROOT / "assets" / "urdf" / "lite6_gripper" / "lite6_gripper_movable_visual.urdf",
        PROJECT_ROOT / "assets" / "urdf" / "lite6_vacuum_gripper" / "lite6_vacuum_gripper.urdf",
        PROJECT_ROOT / "assets" / "urdf" / "lite6_vacuum_gripper" / "lite6_vacuum_gripper_collision.urdf",
        PROJECT_ROOT / "assets" / "urdf" / "xarm6" / "xarm6_1305_g2_movable_visual.urdf",
        PROJECT_ROOT / "assets" / "urdf" / "xarm6" / "xarm6_1305_bio_gripper_g2_movable_visual.glb.urdf",
        PROJECT_ROOT / "assets" / "urdf" / "lite6" / "lite6_gripper_reversed_movable_visual.glb.urdf",
        PROJECT_ROOT / "assets" / "urdf" / "lite6" / "lite6_vacuum_gripper_visual.glb.urdf",
    )
    bad: list[str] = []
    for urdf in urdfs:
        for filename in _collision_mesh_filenames(urdf):
            if not filename.endswith(".stl"):
                bad.append(f"{urdf.relative_to(PROJECT_ROOT)}: collision mesh not STL {filename}")
                continue
            target = (urdf.parent / filename).resolve()
            if not target.exists():
                bad.append(f"{urdf.relative_to(PROJECT_ROOT)}: missing collision mesh {filename}")
    assert bad == []


def test_standalone_movable_accessory_collision_links_are_movable() -> None:
    expected = {
        PROJECT_ROOT / "assets" / "urdf" / "gripper_g2" / "gripper_g2_movable_visual.urdf": {
            "left_outer_knuckle",
            "left_finger",
            "left_inner_knuckle",
            "right_outer_knuckle",
            "right_finger",
            "right_inner_knuckle",
        },
        PROJECT_ROOT / "assets" / "urdf" / "bio_gripper_g2" / "bio_gripper_g2_movable_visual.urdf": {
            "bio_gripper_g2_left_finger",
            "bio_gripper_g2_right_finger",
        },
        PROJECT_ROOT / "assets" / "urdf" / "lite6_gripper" / "lite6_gripper_movable_visual.urdf": {
            "uflite_finger1",
            "uflite_finger2",
        },
    }
    bad: list[str] = []
    for urdf, moving_links in expected.items():
        collision_mesh_by_link = _collision_meshes_by_link(urdf)
        collision_links = _collision_links(urdf)
        non_fixed_children = _non_fixed_joint_children(urdf)
        for link in sorted(moving_links):
            if link not in collision_links:
                bad.append(f"{urdf.relative_to(PROJECT_ROOT)}: {link} has no collision geometry")
            # Mesh-based accessories still require a collision mesh; Lite6 fingers use boxes.
            if urdf.name != "lite6_gripper_movable_visual.urdf" and link not in collision_mesh_by_link:
                bad.append(f"{urdf.relative_to(PROJECT_ROOT)}: {link} has no collision mesh")
            if link not in non_fixed_children:
                bad.append(f"{urdf.relative_to(PROJECT_ROOT)}: {link} is not child of a movable joint")
    assert bad == []


def test_lite6_vacuum_standalone_collision_is_static_only() -> None:
    urdfs = (
        PROJECT_ROOT / "assets" / "urdf" / "lite6_vacuum_gripper" / "lite6_vacuum_gripper.urdf",
        PROJECT_ROOT / "assets" / "urdf" / "lite6_vacuum_gripper" / "lite6_vacuum_gripper_collision.urdf",
    )
    bad: list[str] = []
    for urdf in urdfs:
        movable_joints = sorted(_non_fixed_joint_names(urdf))
        if movable_joints:
            bad.append(f"{urdf.relative_to(PROJECT_ROOT)}: unexpected movable joints {movable_joints}")
    assert bad == []


def test_arm_movable_accessory_collision_links_match_standalone_models() -> None:
    cases = (
        (
            PROJECT_ROOT / "assets" / "urdf" / "gripper_g2" / "gripper_g2_movable_visual.urdf",
            {
                "left_outer_knuckle",
                "left_finger",
                "left_inner_knuckle",
                "right_outer_knuckle",
                "right_finger",
                "right_inner_knuckle",
            },
            (
                PROJECT_ROOT / "assets" / "urdf" / "xarm5" / "xarm5_1305_g2_movable_visual.urdf",
                PROJECT_ROOT / "assets" / "urdf" / "xarm6" / "xarm6_1305_g2_movable_visual.urdf",
                PROJECT_ROOT / "assets" / "urdf" / "xarm7" / "xarm7_1305_g2_movable_visual.urdf",
                PROJECT_ROOT / "assets" / "urdf" / "uf850" / "uf850_g2_movable_visual.urdf",
            ),
        ),
        (
            PROJECT_ROOT / "assets" / "urdf" / "bio_gripper_g2" / "bio_gripper_g2_movable_visual.urdf",
            {
                "bio_gripper_g2_left_finger",
                "bio_gripper_g2_right_finger",
            },
            (
                PROJECT_ROOT / "assets" / "urdf" / "xarm5" / "xarm5_1305_bio_gripper_g2_movable_visual.glb.urdf",
                PROJECT_ROOT / "assets" / "urdf" / "xarm6" / "xarm6_1305_bio_gripper_g2_movable_visual.glb.urdf",
                PROJECT_ROOT / "assets" / "urdf" / "xarm7" / "xarm7_1305_bio_gripper_g2_movable_visual.glb.urdf",
                PROJECT_ROOT / "assets" / "urdf" / "uf850" / "uf850_bio_gripper_g2_movable_visual.glb.urdf",
            ),
        ),
        (
            PROJECT_ROOT / "assets" / "urdf" / "lite6_gripper" / "lite6_gripper_movable_visual.urdf",
            {
                "uflite_finger1",
                "uflite_finger2",
            },
            (
                PROJECT_ROOT / "assets" / "urdf" / "lite6" / "lite6_gripper_movable_visual.glb.urdf",
                PROJECT_ROOT / "assets" / "urdf" / "lite6" / "lite6_gripper_reversed_movable_visual.glb.urdf",
            ),
        ),
    )
    bad: list[str] = []
    for standalone, moving_links, combos in cases:
        if standalone.name == "lite6_gripper_movable_visual.urdf":
            standalone_boxes = _collision_boxes_by_link(standalone)
            expected = {link: standalone_boxes.get(link, ()) for link in moving_links}
            for combo in combos:
                combo_boxes = _collision_boxes_by_link(combo)
                for link, expected_boxes in expected.items():
                    actual_boxes = combo_boxes.get(link, ())
                    if actual_boxes != expected_boxes:
                        bad.append(
                            f"{combo.relative_to(PROJECT_ROOT)}: {link} collision boxes {actual_boxes} "
                            f"do not match standalone {expected_boxes}"
                        )
            continue
        standalone_meshes = _resolved_collision_meshes_by_link(standalone)
        expected = {link: standalone_meshes.get(link, ()) for link in moving_links}
        for combo in combos:
            combo_meshes = _resolved_collision_meshes_by_link(combo)
            for link, expected_meshes in expected.items():
                actual_meshes = combo_meshes.get(link, ())
                if actual_meshes != expected_meshes:
                    bad.append(
                        f"{combo.relative_to(PROJECT_ROOT)}: {link} collision {actual_meshes} "
                        f"does not match standalone {expected_meshes}"
                    )
    assert bad == []


_LITE6_GRIPPER_REVERSIBLE_URDFS = (
    PROJECT_ROOT / "assets" / "urdf" / "lite6_gripper" / "lite6_gripper_movable_visual.urdf",
    PROJECT_ROOT / "assets" / "urdf" / "lite6" / "lite6_gripper_movable_visual.glb.urdf",
    PROJECT_ROOT / "assets" / "urdf" / "lite6" / "lite6_gripper_reversed_movable_visual.glb.urdf",
    PROJECT_ROOT / "assets" / "urdf" / "lite6" / "lite6_with_gripper.urdf",
)

_LITE6_FINGER_BOXES = {
    "uflite_finger1": (
        ("0 0.00775 0.002", "0.02 0.0155 0.0095"),
        ("0 0.013375 0.016875", "0.02 0.00425 0.01975"),
    ),
    "uflite_finger2": (
        ("0 -0.00775 0.002", "0.02 0.0155 0.0095"),
        ("0 -0.013375 0.016875", "0.02 0.00425 0.01975"),
    ),
}


def _lite6_gripper_visual_origin_by_link(urdf: Path) -> dict[str, np.ndarray]:
    root = ET.parse(urdf).getroot()
    out: dict[str, np.ndarray] = {}
    for link_name in ("uflite_finger1", "uflite_finger2"):
        link = root.find(f".//link[@name='{link_name}']")
        if link is None:
            continue
        origin = link.find("./visual/origin")
        out[link_name] = np.asarray(_xyz_tuple(origin.get("xyz") if origin is not None else None), dtype=np.float64)
    return out


def _lite6_gripper_visual_mesh_by_link(urdf: Path) -> dict[str, str]:
    root = ET.parse(urdf).getroot()
    out: dict[str, str] = {}
    for link_name in ("uflite_finger1", "uflite_finger2"):
        link = root.find(f".//link[@name='{link_name}']")
        mesh = link.find("./visual/geometry/mesh") if link is not None else None
        if mesh is not None:
            out[link_name] = mesh.get("filename") or ""
    return out


def _lite6_gripper_root_inner_gap_mm(
    finger1_vertices: np.ndarray,
    finger2_vertices: np.ndarray,
    *,
    finger1_origin_xyz: tuple[float, float, float],
    finger2_origin_xyz: tuple[float, float, float],
    finger1_axis_xyz: tuple[float, float, float],
    finger2_axis_xyz: tuple[float, float, float],
    finger1_visual_origin_xyz: tuple[float, float, float],
    finger2_visual_origin_xyz: tuple[float, float, float],
    q_m: float,
) -> float:
    origin1 = np.asarray(finger1_origin_xyz, dtype=np.float64) + np.asarray(finger1_axis_xyz, dtype=np.float64) * q_m
    origin2 = np.asarray(finger2_origin_xyz, dtype=np.float64) + np.asarray(finger2_axis_xyz, dtype=np.float64) * q_m
    vis1 = np.asarray(finger1_visual_origin_xyz, dtype=np.float64)
    vis2 = np.asarray(finger2_visual_origin_xyz, dtype=np.float64)
    w1 = finger1_vertices + origin1 + vis1
    w2 = finger2_vertices + origin2 + vis2
    z_root = finger1_origin_xyz[2]
    root1 = w1[(w1[:, 2] >= z_root) & (w1[:, 2] <= z_root + 0.008)]
    root2 = w2[(w2[:, 2] >= z_root) & (w2[:, 2] <= z_root + 0.008)]
    return float((root1[:, 1].min() - root2[:, 1].max()) * 1000.0)


def _lite6_gripper_finger_center_separation_mm(
    finger1_vertices: np.ndarray,
    finger2_vertices: np.ndarray,
    *,
    finger1_origin_xyz: tuple[float, float, float],
    finger2_origin_xyz: tuple[float, float, float],
    finger1_axis_xyz: tuple[float, float, float],
    finger2_axis_xyz: tuple[float, float, float],
    finger1_visual_origin_xyz: tuple[float, float, float],
    finger2_visual_origin_xyz: tuple[float, float, float],
    q_m: float,
) -> float:
    origin1 = np.asarray(finger1_origin_xyz, dtype=np.float64) + np.asarray(finger1_axis_xyz, dtype=np.float64) * q_m
    origin2 = np.asarray(finger2_origin_xyz, dtype=np.float64) + np.asarray(finger2_axis_xyz, dtype=np.float64) * q_m
    vis1 = np.asarray(finger1_visual_origin_xyz, dtype=np.float64)
    vis2 = np.asarray(finger2_visual_origin_xyz, dtype=np.float64)
    w1 = finger1_vertices + origin1 + vis1
    w2 = finger2_vertices + origin2 + vis2
    center1 = (w1.min(axis=0) + w1.max(axis=0)) / 2.0
    center2 = (w2.min(axis=0) + w2.max(axis=0)) / 2.0
    return float(abs(center2[1] - center1[1]) * 1000.0)


def test_lite6_gripper_finger_collision_gap_at_keyframes() -> None:
    trimesh = pytest.importorskip("trimesh")
    mesh_dir = PROJECT_ROOT / "assets" / "urdf" / "lite6_gripper" / "meshes" / "collision"
    finger1_vertices = _mesh_vertices(mesh_dir / "finger1.stl", trimesh)
    finger2_vertices = _mesh_vertices(mesh_dir / "finger2.stl", trimesh)

    for urdf in _LITE6_GRIPPER_REVERSIBLE_URDFS:
        root = ET.parse(urdf).getroot()
        joints = {joint.get("name"): joint for joint in root.findall("joint")}
        j1 = joints["finger_joint1"]
        j2 = joints["finger_joint2"]
        origin1 = _xyz_tuple(j1.find("origin").get("xyz"))
        origin2 = _xyz_tuple(j2.find("origin").get("xyz"))
        axis1 = _xyz_tuple(j1.find("axis").get("xyz"))
        axis2 = _xyz_tuple(j2.find("axis").get("xyz"))
        visual_origins = _lite6_gripper_visual_origin_by_link(urdf)
        vis1 = tuple(visual_origins.get("uflite_finger1", np.zeros(3)).tolist())
        vis2 = tuple(visual_origins.get("uflite_finger2", np.zeros(3)).tolist())
        visual_meshes = _lite6_gripper_visual_mesh_by_link(urdf)
        boxes = _collision_boxes_by_link(urdf)
        assert vis1 == (0.0, 0.0, 0.0), (
            f"{urdf.relative_to(PROJECT_ROOT)}: finger1 visual origin should align with the mesh frame, got {vis1}"
        )
        assert vis2 == (0.0, 0.0, 0.0), (
            f"{urdf.relative_to(PROJECT_ROOT)}: finger2 visual origin should align with the mesh frame, got {vis2}"
        )
        assert visual_meshes == {
            "uflite_finger1": "../lite6_gripper/meshes/collision/finger1.stl",
            "uflite_finger2": "../lite6_gripper/meshes/collision/finger2.stl",
        }
        assert boxes.get("uflite_finger1") == _LITE6_FINGER_BOXES["uflite_finger1"], (
            f"{urdf.relative_to(PROJECT_ROOT)}: finger1 dual-box collision mismatch"
        )
        assert boxes.get("uflite_finger2") == _LITE6_FINGER_BOXES["uflite_finger2"], (
            f"{urdf.relative_to(PROJECT_ROOT)}: finger2 dual-box collision mismatch"
        )

        closed_sep = _lite6_gripper_finger_center_separation_mm(
            finger1_vertices,
            finger2_vertices,
            finger1_origin_xyz=origin1,
            finger2_origin_xyz=origin2,
            finger1_axis_xyz=axis1,
            finger2_axis_xyz=axis2,
            finger1_visual_origin_xyz=vis1,
            finger2_visual_origin_xyz=vis2,
            q_m=0.0,
        )
        open_sep = _lite6_gripper_finger_center_separation_mm(
            finger1_vertices,
            finger2_vertices,
            finger1_origin_xyz=origin1,
            finger2_origin_xyz=origin2,
            finger1_axis_xyz=axis1,
            finger2_axis_xyz=axis2,
            finger1_visual_origin_xyz=vis1,
            finger2_visual_origin_xyz=vis2,
            q_m=0.0089,
        )

        closed_gap = _lite6_gripper_root_inner_gap_mm(
            finger1_vertices,
            finger2_vertices,
            finger1_origin_xyz=origin1,
            finger2_origin_xyz=origin2,
            finger1_axis_xyz=axis1,
            finger2_axis_xyz=axis2,
            finger1_visual_origin_xyz=vis1,
            finger2_visual_origin_xyz=vis2,
            q_m=0.0,
        )
        mid_gap = _lite6_gripper_root_inner_gap_mm(
            finger1_vertices,
            finger2_vertices,
            finger1_origin_xyz=origin1,
            finger2_origin_xyz=origin2,
            finger1_axis_xyz=axis1,
            finger2_axis_xyz=axis2,
            finger1_visual_origin_xyz=vis1,
            finger2_visual_origin_xyz=vis2,
            q_m=0.00445,
        )
        open_gap = _lite6_gripper_root_inner_gap_mm(
            finger1_vertices,
            finger2_vertices,
            finger1_origin_xyz=origin1,
            finger2_origin_xyz=origin2,
            finger1_axis_xyz=axis1,
            finger2_axis_xyz=axis2,
            finger1_visual_origin_xyz=vis1,
            finger2_visual_origin_xyz=vis2,
            q_m=0.0089,
        )

        assert 14.0 <= closed_sep <= 17.0, (
            f"{urdf.relative_to(PROJECT_ROOT)}: closed finger center separation {closed_sep:.2f} mm"
        )
        assert 32.0 <= open_sep <= 35.0, (
            f"{urdf.relative_to(PROJECT_ROOT)}: open finger center separation {open_sep:.2f} mm"
        )
        assert open_sep > closed_sep, (
            f"{urdf.relative_to(PROJECT_ROOT)}: opening should increase collision finger separation"
        )
        assert closed_gap == pytest.approx(0.0, abs=0.5), (
            f"{urdf.relative_to(PROJECT_ROOT)}: closed root-region inner gap should overlap at the finger root"
        )
        assert mid_gap == pytest.approx(8.9, abs=0.5), (
            f"{urdf.relative_to(PROJECT_ROOT)}: mid-stroke root-region inner gap should track prismatic travel"
        )
        assert open_gap == pytest.approx(17.8, abs=0.5), (
            f"{urdf.relative_to(PROJECT_ROOT)}: open root-region inner gap should track prismatic travel"
        )


def test_lite6_vacuum_collision_uses_visual_stl() -> None:
    expected_by_urdf = {
        PROJECT_ROOT / "assets" / "urdf" / "lite6_vacuum_gripper" / "lite6_vacuum_gripper.urdf": {
            "meshes/visual/vacuum_gripper_lite.stl",
        },
        PROJECT_ROOT / "assets" / "urdf" / "lite6_vacuum_gripper" / "lite6_vacuum_gripper_collision.urdf": {
            "meshes/visual/vacuum_gripper_lite.stl",
        },
        PROJECT_ROOT / "assets" / "urdf" / "lite6" / "lite6_with_vacuum_gripper.urdf": {
            "../lite6_vacuum_gripper/meshes/visual/vacuum_gripper_lite.stl",
        },
        PROJECT_ROOT / "assets" / "urdf" / "lite6" / "lite6_vacuum_gripper_visual.glb.urdf": {
            "../lite6_vacuum_gripper/meshes/visual/vacuum_gripper_lite.stl",
        },
    }
    bad: list[str] = []
    for urdf, expected in expected_by_urdf.items():
        actual = {filename for filename in _collision_mesh_filenames(urdf) if "vacuum_gripper_lite.stl" in filename}
        if actual != expected:
            bad.append(f"{urdf.relative_to(PROJECT_ROOT)}: expected {sorted(expected)}, got {sorted(actual)}")
    assert bad == []


def test_no_collision_obj_meshes_in_public_assets() -> None:
    obj_files = sorted(
        p.relative_to(PROJECT_ROOT) for p in _public_files() if "collision" in str(p) and p.suffix == ".obj"
    )
    assert obj_files == []
