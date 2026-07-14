"""Runtime robot parameter profile tests."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from ufactory.robots.paths import robot_urdf
from ufactory.robots.runtime import ABS_ERR_FRACTION, get_robot_runtime_profile
from ufactory.robots.registry import ROBOT_PROFILES


@pytest.mark.parametrize("robot_key", sorted(ROBOT_PROFILES))
def test_runtime_profile_lengths_and_urdf_names(robot_key: str) -> None:
    runtime = get_robot_runtime_profile(robot_key)
    dof = runtime.model.dof
    arm = runtime.arm
    assert len(arm.joint_names) == dof
    assert len(arm.home_qpos) == dof
    assert len(arm.default_qpos) == dof
    assert len(arm.kp) == dof
    assert len(arm.kv) == dof
    assert len(arm.force_lower) == dof
    assert len(arm.force_upper) == dof
    assert len(arm.effort_limits) == dof
    assert len(runtime.dynamics.abs_err_limits) == dof
    for effort, limit in zip(arm.effort_limits, runtime.dynamics.abs_err_limits, strict=True):
        assert limit == pytest.approx(effort * ABS_ERR_FRACTION)

    root = ET.parse(robot_urdf(robot_key)).getroot()
    joint_names = {joint.get("name") for joint in root.findall("joint")}
    link_names = {link.get("name") for link in root.findall("link")}
    assert set(arm.joint_names).issubset(joint_names)
    assert arm.ee_link in link_names


@pytest.mark.parametrize("robot_key", ("lite6", "xarm6_1305", "xarm7_1305"))
def test_abs_err_limits_are_ten_percent_of_effort(robot_key: str) -> None:
    runtime = get_robot_runtime_profile(robot_key)
    expected = tuple(e * ABS_ERR_FRACTION for e in runtime.arm.effort_limits)
    assert runtime.dynamics.abs_err_limits == expected
