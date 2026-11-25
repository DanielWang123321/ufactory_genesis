"""Parts fixed to the flange never collide with each other over the full gripper stroke, except by design.

Safety preflight skips every pair inside the flange subtree (flange link, camera stand, cameras, end effector;
``PinocchioCollisionBackend(tool_root_link=...)``): their relative poses depend only on the gripper drive, never on
the arm trajectory. This test is the check that replaces it: for every end-effector / accessory combination it sweeps
the single independent drive joint over its full URDF range and allows only the designed contacts.
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("coal")

from ufactory.robots.assembly import accessories, compose_robot_urdf  # noqa: E402
from ufactory.robots.registry import get_robot_profile  # noqa: E402
from ufactory.safety.adapters.pinocchio import PinocchioCollisionBackend, urdf_subtree_links  # noqa: E402

# One robot per tool flange: parts and their mounts are shared by every robot on the same flange.
FLANGE_ROBOTS = ("xarm6_1305", "lite6")
END_EFFECTORS = (
    "gripper_g2",
    "vacuum_gripper",
    "robotiq_2f_85",
    "robotiq_2f_140",
    "lite6_gripper",
    "lite6_vacuum_gripper",
)
# Bio Gripper G2 has no Pinocchio preflight; ufactory/simulation/bio_g2_collision.py audits it separately.
SAMPLES = 21


def _four_bar(prefix: str) -> tuple[frozenset[str], ...]:
    return (
        # Loop closure: the finger pin sits in the inner knuckle at every opening.
        frozenset((f"{prefix}left_finger", f"{prefix}left_inner_knuckle")),
        frozenset((f"{prefix}right_finger", f"{prefix}right_inner_knuckle")),
    )


# Contacts present over the whole stroke (joint pins), and finger contacts allowed only at the closed end.
PIN_CONTACTS: dict[str, tuple[frozenset[str], ...]] = {
    "gripper_g2": _four_bar(""),
    "robotiq_2f_85": _four_bar("robotiq_2f_85_"),
    "robotiq_2f_140": _four_bar("robotiq_2f_140_"),
}
CLOSED_FINGER_CONTACT: dict[str, frozenset[str]] = {
    "gripper_g2": frozenset(("left_finger", "right_finger")),
    "robotiq_2f_85": frozenset(("robotiq_2f_85_left_finger", "robotiq_2f_85_right_finger")),
    "robotiq_2f_140": frozenset(("robotiq_2f_140_left_finger", "robotiq_2f_140_right_finger")),
    "lite6_gripper": frozenset(("uflite_finger1", "uflite_finger2")),
}


def _combinations() -> list[tuple[str, str | None, str | None]]:
    combos = []
    for robot in FLANGE_ROBOTS:
        profile = get_robot_profile(robot)
        for end_effector in (None, *(e for e in END_EFFECTORS if profile.supports(e))):
            for accessory in (None, *accessories()):
                if end_effector is None and accessory is None:
                    continue
                if accessory is not None and not profile.supports_accessory(accessory, end_effector):
                    continue
                combos.append((robot, end_effector, accessory))
    return combos


def _sweep(robot: str, end_effector: str | None, accessory: str | None) -> tuple[dict[frozenset[str], list[int]], int]:
    """Indices of colliding samples per flange-subtree link pair, and the number of such pairs checked."""
    profile = get_robot_profile(robot)
    urdf = Path(compose_robot_urdf(robot, end_effector, accessory=accessory, visual="stl"))
    root = ET.parse(urdf).getroot()
    arm_joints = tuple(f"joint{i}" for i in range(1, profile.dof + 1))
    drives = [
        joint
        for joint in root.findall("joint")
        if joint.get("type") in {"revolute", "prismatic"}
        and joint.find("mimic") is None
        and joint.get("name") not in arm_joints
    ]
    assert len(drives) <= 1, f"{end_effector}: expected at most one independent drive joint"
    passive = {}
    values: list[float | None] = [None]
    if drives:
        limit = drives[0].find("limit")
        assert limit is not None
        lower, upper = float(limit.get("lower")), float(limit.get("upper"))
        passive = {str(drives[0].get("name")): lower}
        values = list(np.linspace(lower, upper, SAMPLES))
    backend = PinocchioCollisionBackend(
        urdf, joint_names=arm_joints, ee_link=profile.ee_link, passive_joint_positions=passive
    )
    tool = urdf_subtree_links(root, profile.ee_link)
    # Keep only pairs inside the flange subtree (URDF parent-child pairs are already removed).
    pin = backend.pin
    outside = [
        (int(pair.first), int(pair.second))
        for pair in backend.geometry_model.collisionPairs
        if not {backend._link_name(pair.first), backend._link_name(pair.second)} <= tool
    ]
    for first, second in outside:
        backend.geometry_model.removeCollisionPair(pin.CollisionPair(first, second))
    backend.geometry_data = backend.geometry_model.createData()
    collisions: dict[frozenset[str], list[int]] = {}
    q_arm = np.zeros(profile.dof)
    for index, drive in enumerate(values):
        for result in backend.check_all(q_arm, stage="sweep", gripper_drive=drive):
            if result.colliding:
                collisions.setdefault(frozenset((result.link_a, result.link_b)), []).append(index)
    return collisions, len(backend.geometry_model.collisionPairs)


@pytest.mark.parametrize(("robot", "end_effector", "accessory"), _combinations())
def test_flange_subtree_parts_only_touch_by_design(robot: str, end_effector: str | None, accessory: str | None) -> None:
    collisions, checked = _sweep(robot, end_effector, accessory)
    pins = set(PIN_CONTACTS.get(end_effector or "", ()))
    for pair in pins:
        assert collisions.get(pair) == list(range(SAMPLES)), f"loop-closure pin {sorted(pair)} should always touch"
    fingers = CLOSED_FINGER_CONTACT.get(end_effector or "")
    if fingers is not None:
        # The fingers touch only at the closed end of the stroke (one endpoint sample).
        assert collisions.get(fingers) in ([0], [SAMPLES - 1]), f"finger contact {collisions.get(fingers)}"
    unexpected = {tuple(sorted(pair)): samples for pair, samples in collisions.items() if pair not in pins | {fingers}}
    assert not unexpected, f"{robot} + {end_effector} + {accessory}: unexpected contacts {unexpected}"
    if end_effector in PIN_CONTACTS or end_effector == "lite6_gripper":
        assert checked > 0


def test_camera_parts_share_the_flange_body_with_link6() -> None:
    # Fixed joints merge the stand and camera into the flange body, so their pairs never reach a collision query.
    collisions, checked = _sweep("xarm6_1305", None, "d435")
    assert (collisions, checked) == ({}, 0)
