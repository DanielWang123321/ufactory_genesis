"""Independent checks for Bio Gripper G2 gravity compensation, oriented clearances, andSBS 96-well microplate."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from ufactory.simulation import bio_g2_microplate as microplate
from ufactory.simulation.bio_g2_adapter import (
    URDFGravity,
    axis_rotation,
    collision_gripping_planes,
    signed_plate_gaps_mm,
)
from ufactory.simulation.bio_g2_model import LEFT_JOINT as LEFT, RIGHT_JOINT as RIGHT
from ufactory.trajectory import bio_g2_replay as pick_place


@pytest.mark.parametrize("q", [(0, -0.5, 0.2, -0.3, 0.5, 0.2), (0.4, -0.8, -0.2, 0.6, 0.9, -0.4)])
def test_gravity_matches_potential_gradient_including_fixed_gripper(q):
    model = URDFGravity(pick_place.COMBO_MODEL_PATH, pick_place.ARM_JOINT_NAMES)
    positions = dict(zip(pick_place.ARM_JOINT_NAMES, q)) | {RIGHT: 0.025, LEFT: -0.025}
    torque, _ = model.compute(positions)
    for idx, name in enumerate(pick_place.ARM_JOINT_NAMES):
        _, plus = model.compute(positions | {name: positions[name] + 1e-6})
        _, minus = model.compute(positions | {name: positions[name] - 1e-6})
        assert torque[idx] == pytest.approx((plus - minus) / 2e-6, abs=1e-7)
    assert abs(torque[1]) > 1
    assert sum(mass for mass, _ in model.inertials.values()) == pytest.approx(12.84164)
    assert model.inertials["bio_gripper_g2_base_link"][0] > 0.7


def test_gravity_sign_for_analytic_pendulum(tmp_path):
    path = tmp_path / "pendulum.urdf"
    path.write_text(
        """<robot name="test"><link name="base"/><link name="bob"><inertial>
        <origin xyz="1 0 0"/><mass value="2"/></inertial></link>
        <joint name="hinge" type="revolute"><parent link="base"/><child link="bob"/>
        <axis xyz="0 1 0"/></joint></robot>"""
    )
    model = URDFGravity(path, ("hinge",))
    torque, _ = model.compute({"hinge": 0.3})
    assert torque[0] == pytest.approx(-2 * 9.81 * np.cos(0.3))


@pytest.mark.parametrize("offset_mm", [0, 1.445, -2.0, 35.0])
@pytest.mark.parametrize("down", [False, True])
def test_signed_gaps_use_collision_planes_and_world_poses(offset_mm, down):
    pytest.importorskip("trimesh")
    planes = collision_gripping_planes(pick_place.COMBO_MODEL_PATH)
    rotation = axis_rotation([1, 0, 0], np.pi if down else 0)
    quat = [0.0, 1.0, 0.0, 0.0] if down else [1.0, 0.0, 0.0, 0.0]
    translation = np.array([0.38, -0.12, 0.44])
    q = (150 - 71) / 2000
    poses = [(rotation @ np.array([0.059, sign * q, 0.027]) + translation, quat) for sign in (-1, 1)]
    plate_pos = rotation @ np.array([0.128, offset_mm / 1000, 0.050]) + translation
    gaps = signed_plate_gaps_mm(planes, poses, plate_pos, quat, np.asarray(microplate.plate_size_m()))
    expected = (150 - 85.48) / 2 + np.array([offset_mm, -offset_mm])
    np.testing.assert_allclose(gaps, expected, atol=1e-5)


def test_plate_yaw_changes_projected_width_and_detects_penetration():
    planes = [
        (np.array([0.0, -0.05, 0.0]), np.array([0.0, 1.0, 0.0])),
        (np.array([0.0, 0.05, 0.0]), np.array([0.0, -1.0, 0.0])),
    ]
    angle = 0.2
    size = np.asarray(microplate.plate_size_m())
    quat = [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]
    poses = [(np.zeros(3), [1.0, 0.0, 0.0, 0.0])] * 2
    gaps = signed_plate_gaps_mm(planes, poses, np.zeros(3), quat, size)
    expected = (100 - (size[0] * np.sin(angle) + size[1] * np.cos(angle)) * 1000) / 2
    np.testing.assert_allclose(gaps, expected, atol=1e-12)
    assert gaps.max() < 0


@pytest.fixture(scope="module")
def plate():
    return microplate.build_microplate()


def test_plate_dimensions_mass_inertia_and_convex_grip_faces(plate):
    trimesh = pytest.importorskip("trimesh")
    size = np.array([0.12776, 0.08548, 0.01435])
    for filename in ("sbs_96_well.glb", "sbs_96_well_convex.stl"):
        mesh = trimesh.load(microplate.ASSET_DIR / filename, force="mesh")
        np.testing.assert_allclose(mesh.extents, size, atol=2e-8)
        np.testing.assert_allclose(mesh.bounds.mean(axis=0), 0.0, atol=1e-8)
    collision = trimesh.load(microplate.ASSET_DIR / "sbs_96_well_convex.stl", force="mesh")
    assert collision.is_convex and collision.is_watertight
    assert np.count_nonzero(np.abs(collision.face_normals[:, 1]) > 0.99) == 4
    root = ET.parse(microplate.PLATE_URDF).getroot()
    assert float(root.find("link/inertial/mass").get("value")) == 0.1
    inertia = root.find("link/inertial/inertia")
    expected = (
        0.1 / 12 * np.array([size[1] ** 2 + size[2] ** 2, size[0] ** 2 + size[2] ** 2, size[0] ** 2 + size[1] ** 2])
    )
    np.testing.assert_allclose([float(inertia.get(key)) for key in ("ixx", "iyy", "izz")], expected)
    assert plate["assumptions"]["skirt_wall_m"] == 0.00127


def test_visual_has_96_open_wells_on_9mm_pitch(plate):
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.load(microplate.ASSET_DIR / "sbs_96_well.glb", force="mesh")
    triangles = mesh.triangles
    triangles = triangles[np.all(np.abs(triangles[:, :, 2] - 0.01435 / 2) < 1e-8, axis=1), :, :2]

    def covered(point):
        edges = np.roll(triangles, -1, axis=1) - triangles
        delta = np.asarray(point) - triangles
        cross = edges[:, :, 0] * delta[:, :, 1] - edges[:, :, 1] * delta[:, :, 0]
        return np.any(np.all(cross >= -1e-12, axis=1) | np.all(cross <= 1e-12, axis=1))

    for row in range(8):
        for column in range(12):
            center = np.array([-0.0495 + column * 0.009, 0.0315 - row * 0.009])
            assert not covered(center)
            assert covered(center + [0.004, 0])
    assert plate["source"]["license"] == "BSD-3-Clause"
    assert len(plate["source"]["definition_sha256"]) == 64
    assert json.loads((microplate.ASSET_DIR / "SBS_96_well_plate.json").read_text())["number_of_wells"] == 96
