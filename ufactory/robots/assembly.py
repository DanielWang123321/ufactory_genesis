"""Runtime URDF assembly: arm URDF + end-effector part + mount → one deterministic URDF.

Flanges, parts and mounts live in ``assets/urdf/assembly.yaml``. Robots sharing a flange share each part's mount. Composed files are written under
``<repo>/.cache/ufactory/urdf/composed/`` with mesh paths relative to that directory, so identical inputs give
identical bytes (and ``sha256``) on every checkout. File names are content-addressed and writes are atomic.

Runtime configurations and :func:`ufactory.robots.paths.robot_urdf` accept an assembly reference in place of a
URDF file name::

    assembly:<end_effector>[:<visual>[:<gripper_visual>]]

``visual`` is ``glb`` (default, GLB arm visuals) or ``stl`` (the arm's dynamics URDF; end-effector GLB visuals are
replaced by their collision geometry). ``gripper_visual`` is ``movable`` (default, per-link visuals) or ``static``
(one overlay mesh, kinematics unchanged).
"""

from __future__ import annotations

import copy
import hashlib
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Protocol

import yaml

from ufactory.robots.registry import PROJECT_ROOT, ROBOT_PROFILES, get_profile_key_for_robot_name, get_robot_profile

ASSEMBLY_PREFIX = "assembly:"
ASSEMBLY_YAML = Path("assets") / "urdf" / "assembly.yaml"
COMPOSED_DIR = Path(".cache") / "ufactory" / "urdf" / "composed"
VISUALS = ("glb", "stl")
GRIPPER_VISUALS = ("movable", "static")
# Bump when the composition rules change so cached files are regenerated.
COMPOSER_VERSION = "1"


class AssemblyError(ValueError):
    """Raised for unknown parts, unsupported robot/end-effector pairs, or malformed references."""


class _AssetStore(Protocol):
    @property
    def root(self) -> Path: ...

    def require(self, relative_path: str | Path) -> Path: ...


@dataclass(frozen=True)
class StaticVisual:
    link: str
    joint: str
    mesh: str
    material: str | None = None


Pose = tuple[tuple[float, float, float], tuple[float, float, float]]
IDENTITY: Pose = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


@dataclass(frozen=True)
class EndEffector:
    name: str
    source: str
    base_link: str
    mount_joint: str
    flange: str
    robots: tuple[str, ...]
    mount: Pose
    static_visual: StaticVisual | None
    glb_visuals: Mapping[str, str]
    stl_visuals: Mapping[str, str]
    stl_materials: Mapping[str, str]


@dataclass(frozen=True)
class AssemblyRef:
    end_effector: str
    visual: str = "glb"
    gripper_visual: str = "movable"

    def __str__(self) -> str:
        return assembly_ref(self.end_effector, visual=self.visual, gripper_visual=self.gripper_visual)


def _triplet(value: Any, *, where: str) -> tuple[float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise AssemblyError(f"{where} must be a list of three numbers")
    return (float(value[0]), float(value[1]), float(value[2]))


def _load_flanges(data: Mapping[str, Any], path: Path) -> dict[str, tuple[str, ...]]:
    flanges: dict[str, tuple[str, ...]] = {}
    owner: dict[str, str] = {}
    for name, spec in (data.get("flanges") or {}).items():
        robots = tuple(get_profile_key_for_robot_name(str(r)) for r in spec["robots"])
        for key in robots:
            if key in owner:
                raise AssemblyError(f"{path}: {key} listed under flanges {owner[key]!r} and {name!r}")
            owner[key] = str(name)
        flanges[str(name)] = robots
    return flanges


@lru_cache(maxsize=8)
def _load_end_effectors(path: Path, mtime_ns: int) -> dict[str, EndEffector]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 2:
        raise AssemblyError(f"{path}: unsupported assembly schema")
    flanges = _load_flanges(data, path)
    out: dict[str, EndEffector] = {}
    for name, spec in (data.get("end_effectors") or {}).items():
        flange = str(spec["flange"])
        if flange not in flanges:
            raise AssemblyError(f"{name}: unknown flange {flange!r}; expected one of {sorted(flanges)}")
        m = spec.get("mount") or {}
        mount = (
            _triplet(m.get("xyz", (0, 0, 0)), where=f"{name}.mount.xyz"),
            _triplet(m.get("rpy", (0, 0, 0)), where=f"{name}.mount.rpy"),
        )
        sv = spec.get("static_visual")
        out[str(name)] = EndEffector(
            name=str(name),
            source=str(spec["source"]),
            base_link=str(spec["base_link"]),
            mount_joint=str(spec["mount_joint"]),
            flange=flange,
            robots=flanges[flange],
            mount=mount,
            static_visual=StaticVisual(str(sv["link"]), str(sv["joint"]), str(sv["mesh"]), sv.get("material"))
            if sv
            else None,
            glb_visuals={str(k): str(v) for k, v in (spec.get("glb_visuals") or {}).items()},
            stl_visuals={str(k): str(v) for k, v in (spec.get("stl_visuals") or {}).items()},
            stl_materials={str(k): str(v) for k, v in (spec.get("stl_materials") or {}).items()},
        )
    return out


def end_effectors(root: Path | None = None) -> dict[str, EndEffector]:
    """End-effector parts declared in ``assets/urdf/assembly.yaml``."""
    path = (root or PROJECT_ROOT) / ASSEMBLY_YAML
    return _load_end_effectors(path, path.stat().st_mtime_ns)


def get_end_effector(name: str, root: Path | None = None) -> EndEffector:
    parts = end_effectors(root)
    if name not in parts:
        raise AssemblyError(f"unknown end effector {name!r}; expected one of {sorted(parts)}")
    return parts[name]


def supported_end_effectors(robot_key: str, root: Path | None = None) -> tuple[str, ...]:
    key = get_profile_key_for_robot_name(robot_key)
    return tuple(name for name, ee in end_effectors(root).items() if key in ee.robots)


def supports_end_effector(robot_key: str, end_effector: str, root: Path | None = None) -> bool:
    return end_effector in supported_end_effectors(robot_key, root)


def assembly_ref(end_effector: str, *, visual: str = "glb", gripper_visual: str = "movable") -> str:
    """Canonical reference string, omitting default fields."""
    parts = [end_effector]
    if gripper_visual != "movable":
        parts += [visual, gripper_visual]
    elif visual != "glb":
        parts.append(visual)
    return ASSEMBLY_PREFIX + ":".join(parts)


def parse_assembly_ref(value: str) -> AssemblyRef | None:
    """Parse ``assembly:<end_effector>[:<visual>[:<gripper_visual>]]``; ``None`` for a plain file name."""
    if not value.startswith(ASSEMBLY_PREFIX):
        return None
    fields = value[len(ASSEMBLY_PREFIX) :].split(":")
    if not fields[0] or len(fields) > 3:
        raise AssemblyError(f"malformed assembly reference {value!r}")
    visual = fields[1] if len(fields) > 1 else "glb"
    gripper_visual = fields[2] if len(fields) > 2 else "movable"
    if visual not in VISUALS or gripper_visual not in GRIPPER_VISUALS:
        raise AssemblyError(f"{value!r}: visual must be one of {VISUALS}, gripper_visual one of {GRIPPER_VISUALS}")
    return AssemblyRef(fields[0], visual, gripper_visual)


def validate_assembly(
    robot_key: str,
    end_effector: str,
    *,
    visual: str = "glb",
    gripper_visual: str = "movable",
    root: Path | None = None,
) -> EndEffector:
    """Check that the assembly can be composed, without writing anything."""
    root = root or PROJECT_ROOT
    key = get_profile_key_for_robot_name(robot_key)
    if visual not in VISUALS or gripper_visual not in GRIPPER_VISUALS:
        raise AssemblyError(f"visual must be one of {VISUALS}, gripper_visual one of {GRIPPER_VISUALS}")
    ee = get_end_effector(end_effector, root)
    if key not in ee.robots:
        raise AssemblyError(f"{end_effector} is not supported on {key}; supported robots: {list(ee.robots)}")
    if gripper_visual == "static" and ee.static_visual is None:
        raise AssemblyError(f"{end_effector} has no static visual")
    sources = [_arm_source(key, visual, root), root / "assets" / "urdf" / ee.source]
    for path in sources:
        if not path.is_file():
            raise AssemblyError(f"missing assembly source {path}")
    return ee


def _arm_source(robot_key: str, visual: str, root: Path) -> Path:
    profile = ROBOT_PROFILES[robot_key]
    name = str(profile.visual_glb_urdf if visual == "glb" else profile.default_urdf)
    return root / "assets" / "urdf" / str(profile.assets_dir.name) / name


def _fmt(values: tuple[float, float, float]) -> str:
    return " ".join(repr(v) if v else "0" for v in values)


def _rebase_meshes(elem: ET.Element, src_dir: Path, out_dir: Path) -> None:
    for mesh in elem.iter("mesh"):
        name = mesh.get("filename")
        if not name or name.startswith(("package://", "file://")):
            continue
        target = Path(os.path.normpath(Path(name) if os.path.isabs(name) else src_dir / name))
        mesh.set("filename", Path(os.path.relpath(target, out_dir)).as_posix())


def _link_ref(joint: ET.Element, tag: str) -> str | None:
    node = joint.find(tag)
    return node.get("link") if node is not None else None


def _tool_elements(ee: EndEffector, source: Path) -> tuple[list[ET.Element], dict[str, ET.Element]]:
    root = ET.parse(source).getroot()
    attach = [j for j in root.findall("joint") if _link_ref(j, "child") == ee.base_link]
    if len(attach) != 1:
        raise AssemblyError(f"{source}: expected exactly one joint into {ee.base_link}")
    stub = _link_ref(attach[0], "parent")
    elems = [
        copy.deepcopy(e)
        for e in root
        if e.tag in ("link", "joint") and e is not attach[0] and not (e.tag == "link" and e.get("name") == stub)
    ]
    materials = {str(m.get("name")): m for m in root.findall("material")}
    return elems, materials


def _apply_visual_mode(
    elems: list[ET.Element], ee: EndEffector, visual: str, gripper_visual: str, urdf_root: Path
) -> None:
    for link in (e for e in elems if e.tag == "link"):
        name = link.get("name") or ""
        if gripper_visual == "static":
            for v in link.findall("visual"):
                link.remove(v)
            continue
        override = (ee.glb_visuals if visual == "glb" else ee.stl_visuals).get(name)
        material = ee.stl_materials.get(name, ee.stl_materials.get("*")) if visual == "stl" else None
        if override is not None:
            for v in link.findall("visual"):
                mesh = v.find("geometry/mesh")
                if mesh is not None:
                    mesh.set("filename", str(urdf_root / override))
                mat = v.find("material")
                if visual == "glb" and mat is not None:
                    v.remove(mat)
                elif material and mat is None:
                    ET.SubElement(v, "material", {"name": material})
            continue
        if visual == "stl":
            for v in link.findall("visual"):
                mesh = v.find("geometry/mesh")
                if mesh is None or not (mesh.get("filename") or "").lower().endswith(".glb"):
                    continue
                index = list(link).index(v)
                link.remove(v)
                for offset, col in enumerate(link.findall("collision")):
                    nv = ET.Element("visual")
                    for child in col:
                        if child.tag in ("origin", "geometry"):
                            nv.append(copy.deepcopy(child))
                    if material:
                        ET.SubElement(nv, "material", {"name": material})
                    link.insert(index + offset, nv)


def build_assembly_tree(
    robot_key: str,
    end_effector: str,
    *,
    visual: str = "glb",
    gripper_visual: str = "movable",
    root: Path | None = None,
    out_dir: Path | None = None,
) -> ET.Element:
    """Return the composed ``<robot>`` element with mesh paths relative to ``out_dir``."""
    root = (root or PROJECT_ROOT).resolve()
    out_dir = (out_dir or root / COMPOSED_DIR).resolve()
    key = get_profile_key_for_robot_name(robot_key)
    ee = validate_assembly(key, end_effector, visual=visual, gripper_visual=gripper_visual, root=root)
    urdf_root = root / "assets" / "urdf"
    arm_file = _arm_source(key, visual, root)
    tool_file = urdf_root / ee.source
    flange = get_robot_profile(key).ee_link

    robot = ET.Element("robot", {"name": f"{key}__{end_effector}__{visual}_{gripper_visual}"})
    robot.append(
        ET.Comment(
            f" Generated by ufactory.robots.assembly v{COMPOSER_VERSION} from "
            f"{arm_file.relative_to(root).as_posix()} + {tool_file.relative_to(root).as_posix()}; do not edit. "
        )
    )
    for e in ET.parse(arm_file).getroot():
        e = copy.deepcopy(e)
        _rebase_meshes(e, arm_file.parent, out_dir)
        robot.append(e)

    elems, tool_materials = _tool_elements(ee, tool_file)
    _apply_visual_mode(elems, ee, visual, gripper_visual, urdf_root)
    for e in elems:
        _rebase_meshes(e, tool_file.parent, out_dir)
    defined = [m.get("name") for m in robot.findall("material")]
    used = {m.get("name") for e in elems for m in e.iter("material")}
    insert_at = max((i for i, e in enumerate(robot) if e.tag == "material"), default=0) + 1
    for name in sorted(n for n in used - set(defined) if n in tool_materials):
        robot.insert(insert_at, copy.deepcopy(tool_materials[name]))
        insert_at += 1

    xyz, rpy = ee.mount
    origin = {"xyz": _fmt(xyz), "rpy": _fmt(rpy)}
    sv = ee.static_visual
    if gripper_visual == "static" and sv is not None:
        link = ET.SubElement(robot, "link", {"name": sv.link})
        vis = ET.SubElement(link, "visual")
        ET.SubElement(vis, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(ET.SubElement(vis, "geometry"), "mesh", {"filename": str(urdf_root / sv.mesh)})
        if sv.material:
            ET.SubElement(vis, "material", {"name": sv.material})
        _rebase_meshes(link, urdf_root, out_dir)
        joint = ET.SubElement(robot, "joint", {"name": sv.joint, "type": "fixed"})
        ET.SubElement(joint, "parent", {"link": flange})
        ET.SubElement(joint, "child", {"link": sv.link})
        ET.SubElement(joint, "origin", dict(origin))
    mount = ET.SubElement(robot, "joint", {"name": ee.mount_joint, "type": "fixed"})
    ET.SubElement(mount, "parent", {"link": flange})
    ET.SubElement(mount, "child", {"link": ee.base_link})
    ET.SubElement(mount, "origin", dict(origin))
    robot.extend(elems)

    names = [e.get("name") for e in robot if e.tag in ("link", "joint")]
    duplicates = sorted({n for n in names if names.count(n) > 1 and n is not None})
    if duplicates:
        raise AssemblyError(f"{key} + {end_effector}: duplicate link/joint names {duplicates}")
    return robot


def serialize_assembly(robot: ET.Element) -> bytes:
    tree = ET.ElementTree(robot)
    ET.indent(tree, space="  ")
    body: bytes = ET.tostring(robot, encoding="utf-8")
    return b'<?xml version="1.0" encoding="utf-8"?>\n' + body + b"\n"


def compose_robot_urdf(
    robot_key: str,
    end_effector: str,
    *,
    visual: str = "glb",
    gripper_visual: str = "movable",
    root: Path | None = None,
    out_dir: Path | None = None,
) -> str:
    """Compose (or reuse) the assembled URDF and return its absolute path."""
    root = (root or PROJECT_ROOT).resolve()
    out_dir = (out_dir or root / COMPOSED_DIR).resolve()
    key = get_profile_key_for_robot_name(robot_key)
    tree = build_assembly_tree(
        key, end_effector, visual=visual, gripper_visual=gripper_visual, root=root, out_dir=out_dir
    )
    data = serialize_assembly(tree)
    digest = hashlib.sha256(data).hexdigest()[:12]
    path = out_dir / f"{key}__{end_effector}__{visual}_{gripper_visual}.{digest}.urdf"
    if not path.is_file() or path.read_bytes() != data:
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
    return str(path)


def resolve_robot_urdf(
    robot_key: str,
    urdf: str,
    *,
    assets_dir: str | Path | None = None,
    store: _AssetStore | None = None,
) -> Path:
    """Resolve a configured URDF value: an assembly reference is composed, a file name is looked up."""
    root = store.root if store is not None else PROJECT_ROOT
    ref = parse_assembly_ref(urdf)
    if ref is not None:
        return Path(
            compose_robot_urdf(
                robot_key, ref.end_effector, visual=ref.visual, gripper_visual=ref.gripper_visual, root=root
            )
        )
    directory = Path(assets_dir) if assets_dir is not None else get_robot_profile(robot_key).assets_dir
    if store is not None and not directory.is_absolute():
        return store.require(directory / urdf)
    path = (directory if directory.is_absolute() else root / directory) / urdf
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def validate_robot_urdf(
    robot_key: str,
    urdf: str,
    *,
    assets_dir: str | Path,
    store: _AssetStore,
) -> None:
    """Validate a configured URDF value without composing (no files are written)."""
    ref = parse_assembly_ref(urdf)
    if ref is None:
        store.require(Path(assets_dir) / urdf)
        return
    validate_assembly(
        robot_key, ref.end_effector, visual=ref.visual, gripper_visual=ref.gripper_visual, root=store.root
    )
