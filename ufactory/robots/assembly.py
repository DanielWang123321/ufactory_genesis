"""Runtime URDF assembly: arm URDF [+ camera stand / camera] [+ end-effector part] → one deterministic URDF.

Flanges, parts and mounts live in ``assets/urdf/assembly.yaml``. Robots sharing a flange share each part's mount. An
accessory is a spacer (``camera_stand``) or a camera on a spacer; the spacer sits between the flange and the end
effector, so the end effector (or the arm's ``link_eef``) moves by the spacer's ``tool_lift_m``. Composed files are
written under
``<repo>/.cache/ufactory/urdf/composed/`` with mesh paths relative to that directory, so identical inputs give
identical bytes (and ``sha256``) on every checkout. File names are content-addressed and writes are atomic.

Runtime configurations and :func:`ufactory.robots.paths.robot_urdf` accept an assembly reference in place of a
URDF file name::

    assembly:<end_effector>[+<accessory>][:<visual>[:<gripper_visual>]]
    assembly:bare+<accessory>[:<visual>[:<gripper_visual>]]

``bare`` means no end effector. ``visual`` is ``glb`` (default, GLB arm visuals) or ``stl`` (the arm's dynamics URDF; end-effector GLB visuals are
replaced by their collision geometry; cameras draw their collision boxes). ``gripper_visual`` is ``movable`` (default, per-link visuals) or ``static``
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


BARE = "bare"


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
class Spacer:
    name: str
    source: str
    base_link: str
    mount_joint: str
    flange: str
    robots: tuple[str, ...]
    mount: Pose
    tool_lift_m: float
    end_effectors: tuple[str, ...]
    glb_visuals: Mapping[str, str]
    stl_visuals: Mapping[str, str]
    stl_materials: Mapping[str, str]


@dataclass(frozen=True)
class Camera:
    name: str
    source: str
    base_link: str
    mount_joint: str
    spacer: str
    mount: Pose
    optical_frame: str
    glb_visuals: Mapping[str, str]
    stl_visuals: Mapping[str, str]
    stl_materials: Mapping[str, str]


class _VisualParts(Protocol):
    @property
    def glb_visuals(self) -> Mapping[str, str]: ...

    @property
    def stl_visuals(self) -> Mapping[str, str]: ...

    @property
    def stl_materials(self) -> Mapping[str, str]: ...


@dataclass(frozen=True)
class _Catalog:
    end_effectors: dict[str, EndEffector]
    spacers: dict[str, Spacer]
    cameras: dict[str, Camera]


@dataclass(frozen=True)
class AssemblyRef:
    end_effector: str | None
    visual: str = "glb"
    gripper_visual: str = "movable"
    accessory: str | None = None

    def __str__(self) -> str:
        return assembly_ref(
            self.end_effector, visual=self.visual, gripper_visual=self.gripper_visual, accessory=self.accessory
        )


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


def _pose(spec: Mapping[str, Any], name: str) -> Pose:
    m = spec.get("mount") or {}
    return (
        _triplet(m.get("xyz", (0, 0, 0)), where=f"{name}.mount.xyz"),
        _triplet(m.get("rpy", (0, 0, 0)), where=f"{name}.mount.rpy"),
    )


def _str_map(value: Any) -> dict[str, str]:
    return {str(k): str(v) for k, v in (value or {}).items()}


@lru_cache(maxsize=8)
def _load_catalog(path: Path, mtime_ns: int) -> _Catalog:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 3:
        raise AssemblyError(f"{path}: unsupported assembly schema")
    flanges = _load_flanges(data, path)

    def flange_of(name: str, spec: Mapping[str, Any]) -> str:
        flange = str(spec["flange"])
        if flange not in flanges:
            raise AssemblyError(f"{name}: unknown flange {flange!r}; expected one of {sorted(flanges)}")
        return flange

    out: dict[str, EndEffector] = {}
    for name, spec in (data.get("end_effectors") or {}).items():
        flange = flange_of(str(name), spec)
        mount = _pose(spec, str(name))
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
            glb_visuals=_str_map(spec.get("glb_visuals")),
            stl_visuals=_str_map(spec.get("stl_visuals")),
            stl_materials=_str_map(spec.get("stl_materials")),
        )
    spacers: dict[str, Spacer] = {}
    for name, spec in (data.get("spacers") or {}).items():
        flange = flange_of(str(name), spec)
        fits = tuple(str(e) for e in spec.get("end_effectors") or ())
        for ee_name in fits:
            if ee_name not in out or out[ee_name].flange != flange:
                raise AssemblyError(f"{name}: end effector {ee_name!r} is not a {flange!r}-flange part")
        spacers[str(name)] = Spacer(
            name=str(name),
            source=str(spec["source"]),
            base_link=str(spec["base_link"]),
            mount_joint=str(spec["mount_joint"]),
            flange=flange,
            robots=flanges[flange],
            mount=_pose(spec, str(name)),
            tool_lift_m=float(spec.get("tool_lift_m", 0.0)),
            end_effectors=fits,
            glb_visuals=_str_map(spec.get("glb_visuals")),
            stl_visuals=_str_map(spec.get("stl_visuals")),
            stl_materials=_str_map(spec.get("stl_materials")),
        )
    cameras: dict[str, Camera] = {}
    for name, spec in (data.get("cameras") or {}).items():
        spacer = str(spec["spacer"])
        if spacer not in spacers:
            raise AssemblyError(f"{name}: unknown spacer {spacer!r}; expected one of {sorted(spacers)}")
        cameras[str(name)] = Camera(
            name=str(name),
            source=str(spec["source"]),
            base_link=str(spec["base_link"]),
            mount_joint=str(spec["mount_joint"]),
            spacer=spacer,
            mount=_pose(spec, str(name)),
            optical_frame=str(spec["optical_frame"]),
            glb_visuals=_str_map(spec.get("glb_visuals")),
            stl_visuals=_str_map(spec.get("stl_visuals")),
            stl_materials=_str_map(spec.get("stl_materials")),
        )
    clash = sorted((set(spacers) | set(cameras)) & (set(out) | {BARE})) + sorted(set(spacers) & set(cameras))
    if clash:
        raise AssemblyError(f"{path}: part names must be unique across sections and not {BARE!r}: {clash}")
    return _Catalog(out, spacers, cameras)


def _catalog(root: Path | None = None) -> _Catalog:
    path = (root or PROJECT_ROOT) / ASSEMBLY_YAML
    return _load_catalog(path, path.stat().st_mtime_ns)


def end_effectors(root: Path | None = None) -> dict[str, EndEffector]:
    """End-effector parts declared in ``assets/urdf/assembly.yaml``."""
    return _catalog(root).end_effectors


def spacers(root: Path | None = None) -> dict[str, Spacer]:
    """Spacers (camera stands) declared in ``assets/urdf/assembly.yaml``."""
    return _catalog(root).spacers


def cameras(root: Path | None = None) -> dict[str, Camera]:
    """Cameras declared in ``assets/urdf/assembly.yaml``."""
    return _catalog(root).cameras


def accessories(root: Path | None = None) -> tuple[str, ...]:
    """Accessory names: every spacer, then every camera."""
    catalog = _catalog(root)
    return (*catalog.spacers, *catalog.cameras)


def resolve_accessory(name: str, root: Path | None = None) -> tuple[Spacer, Camera | None]:
    """The spacer and (for a camera) the camera an accessory name stands for."""
    catalog = _catalog(root)
    if name in catalog.cameras:
        camera = catalog.cameras[name]
        return catalog.spacers[camera.spacer], camera
    if name in catalog.spacers:
        return catalog.spacers[name], None
    raise AssemblyError(f"unknown accessory {name!r}; expected one of {sorted(accessories(root))}")


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


def supported_accessories(robot_key: str, end_effector: str | None = None, root: Path | None = None) -> tuple[str, ...]:
    """Accessories that fit ``robot_key`` (with ``end_effector``; ``None`` = bare arm)."""
    key = get_profile_key_for_robot_name(robot_key)
    out = []
    for name in accessories(root):
        spacer, _ = resolve_accessory(name, root)
        if key in spacer.robots and (end_effector is None or end_effector in spacer.end_effectors):
            out.append(name)
    return tuple(out)


def supports_accessory(
    robot_key: str, accessory: str, end_effector: str | None = None, root: Path | None = None
) -> bool:
    return accessory in supported_accessories(robot_key, end_effector, root)


def assembly_ref(
    end_effector: str | None,
    *,
    visual: str = "glb",
    gripper_visual: str = "movable",
    accessory: str | None = None,
) -> str:
    """Canonical reference string, omitting default fields."""
    if end_effector is None and accessory is None:
        raise AssemblyError("an assembly needs an end effector or an accessory")
    head = end_effector or BARE
    parts = [f"{head}+{accessory}" if accessory else head]
    if gripper_visual != "movable":
        parts += [visual, gripper_visual]
    elif visual != "glb":
        parts.append(visual)
    return ASSEMBLY_PREFIX + ":".join(parts)


def parse_assembly_ref(value: str) -> AssemblyRef | None:
    """Parse ``assembly:<end_effector|bare>[+<accessory>][:<visual>[:<gripper_visual>]]``; ``None`` for a file name."""
    if not value.startswith(ASSEMBLY_PREFIX):
        return None
    fields = value[len(ASSEMBLY_PREFIX) :].split(":")
    if not fields[0] or len(fields) > 3:
        raise AssemblyError(f"malformed assembly reference {value!r}")
    head, plus, accessory = fields[0].partition("+")
    if not head or (plus and not accessory) or "+" in accessory or (head == BARE and not accessory):
        raise AssemblyError(f"malformed assembly reference {value!r}")
    visual = fields[1] if len(fields) > 1 else "glb"
    gripper_visual = fields[2] if len(fields) > 2 else "movable"
    if visual not in VISUALS or gripper_visual not in GRIPPER_VISUALS:
        raise AssemblyError(f"{value!r}: visual must be one of {VISUALS}, gripper_visual one of {GRIPPER_VISUALS}")
    return AssemblyRef(None if head == BARE else head, visual, gripper_visual, accessory or None)


def validate_assembly(
    robot_key: str,
    end_effector: str | None,
    *,
    visual: str = "glb",
    gripper_visual: str = "movable",
    accessory: str | None = None,
    root: Path | None = None,
) -> EndEffector | None:
    """Check that the assembly can be composed, without writing anything."""
    root = root or PROJECT_ROOT
    key = get_profile_key_for_robot_name(robot_key)
    if visual not in VISUALS or gripper_visual not in GRIPPER_VISUALS:
        raise AssemblyError(f"visual must be one of {VISUALS}, gripper_visual one of {GRIPPER_VISUALS}")
    if end_effector is None and accessory is None:
        raise AssemblyError("an assembly needs an end effector or an accessory")
    urdf_root = root / "assets" / "urdf"
    sources = [_arm_source(key, visual, root)]
    ee = None
    if end_effector is not None:
        ee = get_end_effector(end_effector, root)
        if key not in ee.robots:
            raise AssemblyError(f"{end_effector} is not supported on {key}; supported robots: {list(ee.robots)}")
        if gripper_visual == "static" and ee.static_visual is None:
            raise AssemblyError(f"{end_effector} has no static visual")
        sources.append(urdf_root / ee.source)
    elif gripper_visual == "static":
        raise AssemblyError("gripper_visual 'static' needs an end effector")
    if accessory is not None:
        spacer, camera = resolve_accessory(accessory, root)
        if key not in spacer.robots:
            raise AssemblyError(f"{accessory} is not supported on {key}; supported robots: {list(spacer.robots)}")
        if end_effector is not None and end_effector not in spacer.end_effectors:
            raise AssemblyError(
                f"{accessory} does not fit with {end_effector}; supported end effectors: "
                f"{[BARE, *spacer.end_effectors]}"
            )
        sources.append(urdf_root / spacer.source)
        if camera is not None:
            sources.append(urdf_root / camera.source)
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


def _tool_elements(ee: EndEffector | Spacer | Camera, source: Path) -> tuple[list[ET.Element], dict[str, ET.Element]]:
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
    elems: list[ET.Element], ee: _VisualParts, visual: str, gripper_visual: str, urdf_root: Path
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


def _merge_part_materials(robot: ET.Element, elems: list[ET.Element], part_materials: Mapping[str, ET.Element]) -> None:
    """Insert the part materials its elements use that the robot does not define yet."""
    defined = {m.get("name") for m in robot.findall("material")}
    used = {name for e in elems for m in e.iter("material") if (name := m.get("name")) is not None}
    insert_at = max((i for i, e in enumerate(robot) if e.tag == "material"), default=0) + 1
    for name in sorted(n for n in used - defined if n in part_materials):
        robot.insert(insert_at, copy.deepcopy(part_materials[name]))
        insert_at += 1


def _fixed_joint(robot: ET.Element, name: str, parent: str, child: str, origin: Mapping[str, str]) -> None:
    joint = ET.SubElement(robot, "joint", {"name": name, "type": "fixed"})
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    ET.SubElement(joint, "origin", dict(origin))


def _attach_part(
    robot: ET.Element, part: Spacer | Camera, parent: str, visual: str, urdf_root: Path, out_dir: Path
) -> None:
    """Append a spacer or camera part, fixed to ``parent`` at its mount pose."""
    source = urdf_root / part.source
    elems, materials = _tool_elements(part, source)
    _apply_visual_mode(elems, part, visual, "movable", urdf_root)
    for e in elems:
        _rebase_meshes(e, source.parent, out_dir)
    _merge_part_materials(robot, elems, materials)
    xyz, rpy = part.mount
    _fixed_joint(robot, part.mount_joint, parent, part.base_link, {"xyz": _fmt(xyz), "rpy": _fmt(rpy)})
    robot.extend(elems)


def _lift_link_eef(robot: ET.Element, lift_m: float) -> None:
    """Move the arm's ``link_eef`` (when it has one) along its parent's +Z by ``lift_m``."""
    for joint in robot.findall("joint"):
        if _link_ref(joint, "child") != "link_eef":
            continue
        origin = joint.find("origin")
        if origin is None:
            origin = ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        x, y, z = (float(v) for v in (origin.get("xyz") or "0 0 0").split())
        origin.set("xyz", _fmt((x, y, z + lift_m)))


def _assembly_label(end_effector: str | None, accessory: str | None) -> str:
    return f"{end_effector or BARE}__{accessory}" if accessory else str(end_effector)


def build_assembly_tree(
    robot_key: str,
    end_effector: str | None,
    *,
    visual: str = "glb",
    gripper_visual: str = "movable",
    accessory: str | None = None,
    root: Path | None = None,
    out_dir: Path | None = None,
) -> ET.Element:
    """Return the composed ``<robot>`` element with mesh paths relative to ``out_dir``."""
    root = (root or PROJECT_ROOT).resolve()
    out_dir = (out_dir or root / COMPOSED_DIR).resolve()
    key = get_profile_key_for_robot_name(robot_key)
    ee = validate_assembly(
        key, end_effector, visual=visual, gripper_visual=gripper_visual, accessory=accessory, root=root
    )
    urdf_root = root / "assets" / "urdf"
    arm_file = _arm_source(key, visual, root)
    flange = get_robot_profile(key).ee_link
    spacer, camera = resolve_accessory(accessory, root) if accessory is not None else (None, None)

    sources = [arm_file]
    if spacer is not None:
        sources.append(urdf_root / spacer.source)
    if camera is not None:
        sources.append(urdf_root / camera.source)
    if ee is not None:
        sources.append(urdf_root / ee.source)
    label = _assembly_label(end_effector, accessory)
    robot = ET.Element("robot", {"name": f"{key}__{label}__{visual}_{gripper_visual}"})
    robot.append(
        ET.Comment(
            f" Generated by ufactory.robots.assembly v{COMPOSER_VERSION} from "
            f"{' + '.join(p.relative_to(root).as_posix() for p in sources)}; do not edit. "
        )
    )
    for e in ET.parse(arm_file).getroot():
        e = copy.deepcopy(e)
        _rebase_meshes(e, arm_file.parent, out_dir)
        robot.append(e)

    lift = 0.0
    if spacer is not None:
        _attach_part(robot, spacer, flange, visual, urdf_root, out_dir)
        lift = spacer.tool_lift_m
        if camera is not None:
            _attach_part(robot, camera, spacer.base_link, visual, urdf_root, out_dir)

    if ee is not None:
        tool_file = urdf_root / ee.source
        elems, tool_materials = _tool_elements(ee, tool_file)
        _apply_visual_mode(elems, ee, visual, gripper_visual, urdf_root)
        for e in elems:
            _rebase_meshes(e, tool_file.parent, out_dir)
        _merge_part_materials(robot, elems, tool_materials)

        xyz, rpy = ee.mount
        if lift:
            xyz = (xyz[0], xyz[1], xyz[2] + lift)
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
            _fixed_joint(robot, sv.joint, flange, sv.link, origin)
        _fixed_joint(robot, ee.mount_joint, flange, ee.base_link, origin)
        robot.extend(elems)
    elif lift:
        _lift_link_eef(robot, lift)

    names = [e.get("name") for e in robot if e.tag in ("link", "joint")]
    duplicates = sorted({n for n in names if names.count(n) > 1 and n is not None})
    if duplicates:
        raise AssemblyError(f"{key} + {label}: duplicate link/joint names {duplicates}")
    return robot


def serialize_assembly(robot: ET.Element) -> bytes:
    tree = ET.ElementTree(robot)
    ET.indent(tree, space="  ")
    body: bytes = ET.tostring(robot, encoding="utf-8")
    return b'<?xml version="1.0" encoding="utf-8"?>\n' + body + b"\n"


def compose_robot_urdf(
    robot_key: str,
    end_effector: str | None,
    *,
    visual: str = "glb",
    gripper_visual: str = "movable",
    accessory: str | None = None,
    root: Path | None = None,
    out_dir: Path | None = None,
) -> str:
    """Compose (or reuse) the assembled URDF and return its absolute path.

    ``end_effector`` may be ``None`` (bare arm) when ``accessory`` (a spacer or camera name) is given.
    """
    root = (root or PROJECT_ROOT).resolve()
    out_dir = (out_dir or root / COMPOSED_DIR).resolve()
    key = get_profile_key_for_robot_name(robot_key)
    tree = build_assembly_tree(
        key,
        end_effector,
        visual=visual,
        gripper_visual=gripper_visual,
        accessory=accessory,
        root=root,
        out_dir=out_dir,
    )
    data = serialize_assembly(tree)
    digest = hashlib.sha256(data).hexdigest()[:12]
    path = out_dir / f"{key}__{_assembly_label(end_effector, accessory)}__{visual}_{gripper_visual}.{digest}.urdf"
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
                robot_key,
                ref.end_effector,
                visual=ref.visual,
                gripper_visual=ref.gripper_visual,
                accessory=ref.accessory,
                root=root,
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
        robot_key,
        ref.end_effector,
        visual=ref.visual,
        gripper_visual=ref.gripper_visual,
        accessory=ref.accessory,
        root=store.root,
    )
