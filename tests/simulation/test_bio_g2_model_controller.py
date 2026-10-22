"""Fast regressions for Bio Gripper G2 units, geometry, virtual work, and controller transitions."""

from __future__ import annotations

import copy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from ufactory.robots.paths import bio_gripper_g2_urdf
from ufactory.simulation.bio_g2_controller import BioGripperG2Controller, GripperController
from ufactory.simulation.bio_g2_model import (
    BASE_INERTIA_SCALE,
    DEFAULT_SPEED_MM_S,
    LEFT_JOINT as LEFT,
    MAX_SPEED_MM_S,
    MIN_SPEED_MM_S,
    RIGHT_JOINT as RIGHT,
    SPEEDS_MM_S,
    BioGripperG2Model,
    bio_g2_physics_summary,
    bio_g2_speed_mm_s_from_percent,
    bio_g2_speed_percent_from_mm_s,
    drive_limit_n,
    finger_force_n,
    gap_to_q,
    q_to_gap,
)
from ufactory.trajectory.bio_g2_replay import (
    COMBO_MODEL_PATH,
    DOWN_QUAT_WXYZ,
    PLACE_XY,
    base_link_xyz_to_grasp_xyz,
    build_xarm6_combo_model,
    default_phases,
    grasp_xyz_to_base_link_xyz,
    parse_args,
)

MODEL_PATH = Path(bio_gripper_g2_urdf())


class FakeAdapter:
    def __init__(self):
        self.q, self.v, self.resets, self.calls = gap_to_q(150), 0.0, 0, []

    def state(self):
        return self.q, self.v, -self.q, -self.v

    def set_force_limit(self, limit):
        self.calls.append(("limit", limit))

    def servo(self, q, v):
        self.calls.append(("servo", q, v))

    def initialize(self, q):
        self.q, self.v = q, 0.0
        self.resets += 1


@pytest.mark.parametrize("gap", [71, 80, 110, 140, 150])
def test_units_and_virtual_work(gap):
    q = gap_to_q(gap)
    assert q_to_gap(q, -q) == pytest.approx(gap)
    assert q_to_gap(q) == pytest.approx(gap)
    for force in (1, 25, 50, 75, 100):
        dq = 0.001
        assert drive_limit_n(force) * dq == pytest.approx(finger_force_n(force) * 2 * dq)
    assert SPEEDS_MM_S == (500 / 12, 1000 / 12, 2000 / 12, 3000 / 12, 4000 / 12)


def test_speed_percent_and_mm_s_conversions():
    assert bio_g2_speed_mm_s_from_percent(0) == 0.0
    assert bio_g2_speed_mm_s_from_percent(10) == pytest.approx(MIN_SPEED_MM_S)
    assert bio_g2_speed_mm_s_from_percent(50) == pytest.approx(DEFAULT_SPEED_MM_S)
    assert bio_g2_speed_mm_s_from_percent(100) == pytest.approx(MAX_SPEED_MM_S)
    assert bio_g2_speed_percent_from_mm_s(0.0) == 0.0
    assert bio_g2_speed_percent_from_mm_s(DEFAULT_SPEED_MM_S) == pytest.approx(50.0)
    assert bio_g2_speed_percent_from_mm_s(MAX_SPEED_MM_S) == pytest.approx(100.0)


def test_mass_inertia_and_mimic():
    model = BioGripperG2Model.load()
    summary = bio_g2_physics_summary()
    assert model.total_mass_kg == pytest.approx(0.79, abs=1e-12)
    assert summary["total_mass_kg"] == pytest.approx(0.79, abs=1e-12)
    robot = ET.parse(MODEL_PATH).getroot()
    masses = [float(n.get("value")) for n in robot.findall("link/inertial/mass")]
    assert sum(masses) == pytest.approx(0.79, abs=1e-12)
    assert all(m > 0 for m in masses)
    assert BASE_INERTIA_SCALE == pytest.approx(1.9719005072482427)
    for node in robot.findall("link/inertial/inertia"):
        a = {k: float(v) for k, v in node.attrib.items()}
        matrix = [
            [a["ixx"], a["ixy"], a["ixz"]],
            [a["ixy"], a["iyy"], a["iyz"]],
            [a["ixz"], a["iyz"], a["izz"]],
        ]
        assert np.linalg.eigvalsh(matrix).min() > 0
    mimic = robot.find(f"joint[@name='{LEFT}']/mimic")
    assert mimic is not None
    assert mimic.get("joint") == RIGHT and float(mimic.get("multiplier")) == -1.0


@pytest.mark.parametrize("gap", [71, 80, 110, 140, 150])
def test_actual_visual_and_collision_surfaces(gap):
    trimesh = pytest.importorskip("trimesh")
    urdf_root = ET.parse(MODEL_PATH).getroot()
    bounds = {}
    for side in ("left", "right"):
        name = f"bio_gripper_g2_{side}_finger"
        link = urdf_root.find(f"link[@name='{name}']")
        vis_origin = np.fromstring(link.find("visual/origin").get("xyz", "0 0 0"), sep=" ")
        q = gap_to_q(gap) * (-1 if side == "left" else 1)
        visual = trimesh.load(MODEL_PATH.parent / f"meshes/visual/{name}.stl", force="mesh")
        visual.apply_translation(vis_origin + np.array([0.059, q, 0.027]))
        collision = trimesh.load(MODEL_PATH.parent / f"meshes/collision/{name}_convex_0.stl", force="mesh")
        collision.apply_translation([0.059, q, 0.027])
        for surface, mesh in (("visual", visual), ("collision", collision)):
            tip = mesh.vertices[mesh.vertices[:, 0] > 0.09]
            bounds[(side, surface)] = tip[:, 1].max() if side == "left" else tip[:, 1].min()
        assert bounds[(side, "visual")] == pytest.approx(bounds[(side, "collision")], abs=1e-8)
        assert collision.is_convex and collision.is_watertight
        if side == "left":
            assert collision.vertices[:, 1].max() <= -gap / 2000 + 1e-8
        else:
            assert collision.vertices[:, 1].min() >= gap / 2000 - 1e-8
        root = trimesh.load(MODEL_PATH.parent / f"meshes/collision/{name}_convex_1.stl", force="mesh")
        assert root.is_convex and root.is_watertight
        assert root.bounds[1, 0] + 0.059 == pytest.approx(0.05, abs=1e-8)
        assert collision.bounds[0, 0] == pytest.approx(0.05, abs=1e-8)
    for surface in ("visual", "collision"):
        assert (bounds[("right", surface)] - bounds[("left", surface)]) * 1000 == pytest.approx(gap, abs=1e-5)


@pytest.mark.parametrize("gap", [71, 80, 110, 140, 150])
def test_gripping_face_ray_intersections(gap):
    trimesh = pytest.importorskip("trimesh")
    urdf_root = ET.parse(MODEL_PATH).getroot()
    for side in ("left", "right"):
        name = f"bio_gripper_g2_{side}_finger"
        link = urdf_root.find(f"link[@name='{name}']")
        vis_origin = np.fromstring(link.find("visual/origin").get("xyz", "0 0 0"), sep=" ")
        shift = gap_to_q(gap) * (-1 if side == "left" else 1)
        expected_y = gap / 2000 * (-1 if side == "left" else 1)
        for rel_path, extra_origin in (
            (f"meshes/visual/{name}.stl", vis_origin),
            (f"meshes/collision/{name}_convex_0.stl", np.zeros(3)),
        ):
            mesh = trimesh.load(MODEL_PATH.parent / rel_path, force="mesh")
            mesh.apply_translation(extra_origin + np.array([0.059, shift, 0.027]))
            tri = mesh.triangles
            a = tri[:, 0][:, [0, 2]]
            u, v = tri[:, 1][:, [0, 2]] - a, tri[:, 2][:, [0, 2]] - a
            det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
            valid = abs(det) > 1e-14
            divisor = np.where(valid, det, 1)
            for x in (0.1155, 0.128, 0.135):
                for z in (0.05,):
                    b = np.array([x, z]) - a
                    c = (b[:, 0] * v[:, 1] - b[:, 1] * v[:, 0]) / divisor
                    d = (u[:, 0] * b[:, 1] - u[:, 1] * b[:, 0]) / divisor
                    hit = valid & (c >= -1e-8) & (d >= -1e-8) & (c + d <= 1 + 1e-8)
                    y = tri[:, 0, 1] + c * (tri[:, 1, 1] - tri[:, 0, 1]) + d * (tri[:, 2, 1] - tri[:, 0, 1])
                    assert hit.any()
                    inner_y = y[hit].max() if side == "left" else y[hit].min()
                    assert inner_y == pytest.approx(expected_y, abs=1e-8)


@pytest.mark.parametrize("field", ["position_mm", "speed_mm_s", "force_percent"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -1, 0, 10000])
def test_invalid_commands_are_atomic(field, value):
    adapter = FakeAdapter()
    controller = GripperController(adapter)
    controller.command(110)
    before = copy.deepcopy({k: v for k, v in vars(controller).items() if k != "adapter"})
    calls = copy.deepcopy(adapter.calls)
    arguments = dict(position_mm=110, speed_mm_s=500 / 3, force_percent=50)
    arguments[field] = value
    with pytest.raises(ValueError):
        controller.command(**arguments)
    assert {k: v for k, v in vars(controller).items() if k != "adapter"} == before
    assert adapter.calls == calls


def test_rate_limit_pause_force_reversal_and_reset():
    adapter = FakeAdapter()
    controller = BioGripperG2Controller(adapter)
    controller.command(71, speed_mm_s=500 / 12, force_percent=100)
    old_q = controller.reference_q_m
    controller.apply(0.02)
    assert old_q - controller.reference_q_m == pytest.approx((500 / 12) * 0.02 / 1000)
    controller.integral_n = -30
    controller.pause()
    reference = controller.reference_q_m
    controller.command(150, force_percent=1)
    assert controller.paused
    assert abs(controller.integral_n) <= 8
    assert adapter.calls[-2] == ("limit", 8.0)
    for _ in range(5):
        controller.apply(0.02)
    assert controller.reference_q_m == reference and adapter.resets == 0
    controller.resume()
    assert controller.integral_n == 0
    controller.apply(0.02)
    assert adapter.resets == 0
    for gap in (71, 110, 150, 110):
        controller.reset(gap)
        assert q_to_gap(adapter.q) == pytest.approx(gap)
        assert controller.integral_n == 0 and not controller.paused
    assert adapter.resets == 4

    # Test speed_percent=0 pauses and speed_percent=50 resumes at 2000 rpm (166.67 mm/s)
    controller.command(71, speed_percent=0, force_percent=50)
    assert controller.paused
    controller.command(71, speed_percent=50, force_percent=50)
    assert not controller.paused
    assert controller.speed_mm_s == pytest.approx(DEFAULT_SPEED_MM_S)


def test_conditional_integral_does_not_wind_up_at_saturation():
    adapter = FakeAdapter()
    controller = GripperController(adapter)
    controller.command(71, force_percent=1)
    for _ in range(200):
        controller.apply(0.02)
    before = controller.integral_n
    for _ in range(100):
        controller.apply(0.02)
    assert controller.integral_n == before
    assert abs(controller.integral_n) <= 8
    assert adapter.resets == 0


@pytest.mark.parametrize("dt", [0, -1, float("nan"), float("inf")])
def test_invalid_dt_is_atomic(dt):
    adapter = FakeAdapter()
    controller = GripperController(adapter)
    before = copy.deepcopy({k: v for k, v in vars(controller).items() if k != "adapter"})
    calls = copy.deepcopy(adapter.calls)
    with pytest.raises(ValueError):
        controller.apply(dt)
    assert {k: v for k, v in vars(controller).items() if k != "adapter"} == before
    assert calls == adapter.calls


def test_xarm6_combo_model_preserves_gripper_mass_and_mimic():
    manifest = build_xarm6_combo_model()
    phases = {p.name: p for p in default_phases()}
    phase_names = [p.name for p in default_phases()]
    assert phase_names[phase_names.index("release") : phase_names.index("return_home") + 1] == [
        "release",
        "retreat_up",
        "return_home",
    ]
    assert phases["retreat_up"].start_grasp_xyz[:2] == PLACE_XY
    assert phases["retreat_up"].end_grasp_xyz[:2] == PLACE_XY
    assert phases["retreat_up"].end_grasp_xyz[2] > phases["retreat_up"].start_grasp_xyz[2]
    assert manifest["visual_format"] == "glb"
    assert manifest["gripper_mass_kg"] == pytest.approx(0.79, abs=1e-12)
    assert manifest["total_mass_kg"] == pytest.approx(manifest["arm_mass_kg"] + 0.79, abs=1e-12)
    root = ET.parse(COMBO_MODEL_PATH).getroot()
    visual_meshes = root.findall("link/visual/geometry/mesh")
    assert len(visual_meshes) == 10
    for mesh_node in visual_meshes:
        filename = mesh_node.get("filename") or ""
        assert filename.endswith(".glb"), filename
        assert (COMBO_MODEL_PATH.parent / filename).resolve().is_file(), filename
    gripper_links = [
        n
        for n in root.findall("link")
        if (n.get("name") or "").startswith("bio_gripper_g2_") and n.find("inertial/mass") is not None
    ]
    assert len(gripper_links) == 3
    gripper_mass = sum(float(n.find("inertial/mass").get("value")) for n in gripper_links)
    assert gripper_mass == pytest.approx(0.79, abs=1e-12)
    mimic = root.find(f"joint[@name='{LEFT}']/mimic")
    assert mimic is not None and mimic.get("joint") == RIGHT and float(mimic.get("multiplier")) == -1.0

    target_grasp = np.array([0.38, -0.12, 0.438], dtype=np.float64)
    base_xyz = grasp_xyz_to_base_link_xyz(target_grasp)
    recovered = base_link_xyz_to_grasp_xyz(base_xyz, DOWN_QUAT_WXYZ)
    assert np.allclose(recovered, target_grasp, atol=1e-12)

    for bad_args in (
        ["--speed-mm-s", "10"],
        ["--force-percent", "0"],
    ):
        with pytest.raises(SystemExit) as exc_info:
            parse_args(bad_args)
        assert exc_info.value.code == 2
