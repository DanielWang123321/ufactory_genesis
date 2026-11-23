"""Runtime URDF assembly (arm + end effector) contract tests."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from ufactory.config import RepositoryAssetStore, load_runtime_config
from ufactory.robots import assembly
from ufactory.robots.assembly import (
    AssemblyError,
    AssemblyRef,
    assembly_ref,
    compose_robot_urdf,
    end_effectors,
    parse_assembly_ref,
    resolve_robot_urdf,
    supported_end_effectors,
    validate_robot_urdf,
)
from ufactory.robots.paths import robot_urdf, robot_visual_glb_urdf
from ufactory.robots.registry import PROJECT_ROOT, get_robot_profile

_VARIANTS = (("glb", "movable"), ("stl", "movable"), ("glb", "static"))


def _supported() -> list[tuple[str, str, str, str]]:
    out = []
    for name, ee in end_effectors().items():
        for key in ee.robots:
            for visual, gripper_visual in _VARIANTS:
                if gripper_visual == "static" and ee.static_visual is None:
                    continue
                out.append((key, name, visual, gripper_visual))
    return out


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("assembly:gripper_g2", AssemblyRef("gripper_g2", "glb", "movable")),
        ("assembly:lite6_gripper:stl", AssemblyRef("lite6_gripper", "stl", "movable")),
        ("assembly:bio_gripper_g2:glb:static", AssemblyRef("bio_gripper_g2", "glb", "static")),
        ("xarm6_1305.urdf", None),
    ],
)
def test_parse_assembly_ref(value: str, expected: AssemblyRef | None) -> None:
    assert parse_assembly_ref(value) == expected
    if expected is not None:
        assert str(expected) == value


@pytest.mark.parametrize("value", ["assembly:", "assembly:g2:obj", "assembly:g2:glb:moving", "assembly:a:b:c:d"])
def test_parse_assembly_ref_rejects_malformed(value: str) -> None:
    with pytest.raises(AssemblyError):
        parse_assembly_ref(value)


def test_assembly_ref_omits_defaults() -> None:
    assert assembly_ref("gripper_g2") == "assembly:gripper_g2"
    assert assembly_ref("gripper_g2", visual="stl") == "assembly:gripper_g2:stl"
    assert assembly_ref("gripper_g2", gripper_visual="static") == "assembly:gripper_g2:glb:static"


def test_compatibility_matrix() -> None:
    assert supported_end_effectors("xarm6") == ("gripper_g2", "bio_gripper_g2", "vacuum_gripper")
    assert supported_end_effectors("lite6") == ("lite6_gripper", "lite6_vacuum_gripper")
    assert get_robot_profile("uf850").supports_bio_gripper_g2
    assert not get_robot_profile("lite6").supports_gripper_g2
    with pytest.raises(AssemblyError, match="not supported on lite6"):
        compose_robot_urdf("lite6", "gripper_g2")
    with pytest.raises(AssemblyError, match="no static visual"):
        compose_robot_urdf("xarm6", "vacuum_gripper", gripper_visual="static")
    with pytest.raises(AssemblyError, match="unknown end effector"):
        compose_robot_urdf("xarm6", "camera")


def test_composed_urdf_is_deterministic_and_location_independent(tmp_path: Path) -> None:
    first = Path(compose_robot_urdf("xarm6", "gripper_g2"))
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        second = Path(compose_robot_urdf("xarm6_1305", "gripper_g2"))
    finally:
        os.chdir(cwd)
    assert first == second
    assert first.parent == PROJECT_ROOT / assembly.COMPOSED_DIR
    data = first.read_bytes()
    assert data == assembly.serialize_assembly(assembly.build_assembly_tree("xarm6", "gripper_g2"))
    meshes = [m.get("filename") or "" for m in ET.fromstring(data).iter("mesh")]
    assert meshes and not any(os.path.isabs(name) for name in meshes)
    assert all(name.startswith("../../../../assets/urdf/") for name in meshes)


def test_custom_output_directory_keeps_meshes_resolvable(tmp_path: Path) -> None:
    path = Path(compose_robot_urdf("lite6", "lite6_gripper", visual="stl", out_dir=tmp_path))
    assert path.parent == tmp_path
    for mesh in ET.parse(path).getroot().iter("mesh"):
        assert (path.parent / (mesh.get("filename") or "")).resolve().is_file()


def _physics(path: Path, drop_links: set[str]) -> dict[tuple[str, str | None], str]:
    out = {}
    for e in ET.parse(path).getroot():
        if e.tag not in ("link", "joint") or e.get("name") in drop_links:
            continue
        if e.tag == "joint" and e.find("child") is not None and e.find("child").get("link") in drop_links:
            continue
        e = ET.fromstring(ET.tostring(e))
        for v in e.findall("visual"):
            e.remove(v)
        for mesh in e.iter("mesh"):
            mesh.set("filename", str((path.parent / (mesh.get("filename") or "")).resolve()))
        out[(e.tag, e.get("name"))] = ET.tostring(e, encoding="unicode")
    return out


@pytest.mark.parametrize(("robot", "end_effector"), sorted({(r, ee) for r, ee, _, _ in _supported()}))
def test_visual_modes_do_not_change_physics(robot: str, end_effector: str) -> None:
    ee = end_effectors()[end_effector]
    overlay = {ee.static_visual.link} if ee.static_visual else set()
    reference = _physics(Path(compose_robot_urdf(robot, end_effector)), overlay)
    stl = _physics(Path(compose_robot_urdf(robot, end_effector, visual="stl")), overlay)
    # Arm dynamics URDFs differ from GLB visual URDFs only in visuals, except Lite6's fixed-base link_base inertia.
    assert {k for k in reference if reference[k] != stl.get(k)} <= (
        {("link", "link_base")} if robot == "lite6" else set()
    )
    assert reference.keys() == stl.keys()
    if ee.static_visual is not None:
        static = _physics(Path(compose_robot_urdf(robot, end_effector, gripper_visual="static")), overlay)
        assert static == reference


@pytest.mark.parametrize(("robot", "end_effector", "visual", "gripper_visual"), _supported())
def test_every_supported_assembly_mounts_on_the_flange(
    robot: str, end_effector: str, visual: str, gripper_visual: str
) -> None:
    ee = end_effectors()[end_effector]
    root = ET.parse(compose_robot_urdf(robot, end_effector, visual=visual, gripper_visual=gripper_visual)).getroot()
    mount = root.find(f"./joint[@name='{ee.mount_joint}']")
    assert mount is not None and mount.get("type") == "fixed"
    assert mount.find("parent").get("link") == get_robot_profile(robot).ee_link
    assert mount.find("child").get("link") == ee.base_link
    names = [e.get("name") for e in root if e.tag in ("link", "joint")]
    assert len(names) == len(set(names))
    assert root.find("./link[@name='world']") is not None
    assert len([e for e in root if e.tag == "link"]) == len([e for e in root if e.tag == "joint"]) + 1


def test_bio_gripper_g2_mounts_follow_assembly_table() -> None:
    data = yaml.safe_load((PROJECT_ROOT / "assets" / "urdf" / "assembly.yaml").read_text(encoding="utf-8"))
    mounts = data["end_effectors"]["bio_gripper_g2"]["mounts"]
    assert set(mounts) == {"xarm5_1305", "xarm6_1305", "xarm7_1305", "uf850"}
    for robot, mount in mounts.items():
        root = ET.parse(compose_robot_urdf(robot, "bio_gripper_g2", gripper_visual="static")).getroot()
        for joint_name in ("bio_gripper_g2_attach", "bio_gripper_g2_visual_fix"):
            origin = root.find(f"./joint[@name='{joint_name}']/origin")
            assert [float(v) for v in origin.get("xyz").split()] == pytest.approx(mount["xyz"], abs=0.0)
            assert [float(v) for v in origin.get("rpy").split()] == pytest.approx(mount["rpy"], abs=0.0)


def test_runtime_configs_resolve_assembled_models() -> None:
    store = RepositoryAssetStore.discover()
    expected = {
        "xarm5_1305": "assembly:gripper_g2",
        "xarm6_1305": "assembly:gripper_g2",
        "xarm7_1305": "assembly:gripper_g2",
        "uf850": "assembly:gripper_g2",
        "lite6": "assembly:lite6_gripper:stl",
    }
    for key, ref in expected.items():
        config = load_runtime_config(key)
        assert config.robot.urdf == ref
        path = resolve_robot_urdf(key, config.robot.urdf, assets_dir=config.robot.assets_dir, store=store)
        assert path == Path(robot_urdf(key, config.robot.urdf))
        assert path.is_file()
    reach = load_runtime_config("xarm6", task="reach")
    assert reach.robot.urdf == "xarm6_1305.urdf"
    assert resolve_robot_urdf("xarm6", reach.robot.urdf, assets_dir=reach.robot.assets_dir, store=store) == Path(
        robot_urdf("xarm6")
    )


def test_validate_robot_urdf_never_composes(monkeypatch: pytest.MonkeyPatch) -> None:
    store = RepositoryAssetStore.discover()

    def _fail(*args: object, **kwargs: object) -> str:
        raise AssertionError("validation must not compose")

    monkeypatch.setattr(assembly, "compose_robot_urdf", _fail)
    validate_robot_urdf("xarm6", "assembly:gripper_g2", assets_dir="assets/urdf/xarm6", store=store)
    validate_robot_urdf("xarm6", "xarm6_1305.urdf", assets_dir="assets/urdf/xarm6", store=store)
    with pytest.raises(AssemblyError):
        validate_robot_urdf("lite6", "assembly:gripper_g2", assets_dir="assets/urdf/lite6", store=store)


def test_visual_glb_wrapper_maps_flags_to_assemblies() -> None:
    assert Path(robot_visual_glb_urdf("xarm6")).name == "xarm6_1305_visual.glb.urdf"
    cases = {
        ("gripper_g2", True): dict(with_gripper_g2=True, movable=True),
        ("gripper_g2", False): dict(with_gripper_g2=True),
        ("bio_gripper_g2", False): dict(with_bio_gripper_g2=True),
        ("vacuum_gripper", True): dict(with_vacuum_gripper=True),
    }
    for (end_effector, movable), flags in cases.items():
        gripper_visual = "movable" if movable else "static"
        assert robot_visual_glb_urdf("xarm6", **flags) == compose_robot_urdf(
            "xarm6", end_effector, gripper_visual=gripper_visual
        )
