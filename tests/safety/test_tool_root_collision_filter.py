"""Safety preflight skips pairs inside the flange subtree and keeps every pair involving the arm or the environment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ufactory.config import RepositoryAssetStore, load_runtime_config
from ufactory.manipulation.commands import build_manipulation_backends
from ufactory.robots.assembly import compose_robot_urdf, resolve_robot_urdf
from ufactory.safety.adapters import EnvironmentObstacle, PinocchioCollisionBackend

pytest.importorskip("pinocchio")
pytest.importorskip("coal")


def _backend(config, urdf: str | Path, **kwargs) -> PinocchioCollisionBackend:
    return PinocchioCollisionBackend(
        urdf,
        joint_names=config.robot.joint_names,
        ee_link=config.robot.ee_link,
        passive_joint_positions={config.gripper.drive_joint: config.gripper.open_drive},
        **kwargs,
    )


def _pairs(backend: PinocchioCollisionBackend) -> dict[tuple[str, str], frozenset[str]]:
    out = {}
    for pair in backend.geometry_model.collisionPairs:
        geometries = (
            str(backend.geometry_model.geometryObjects[pair.first].name),
            str(backend.geometry_model.geometryObjects[pair.second].name),
        )
        out[geometries] = frozenset((backend._link_name(pair.first), backend._link_name(pair.second)))
    return out


def test_tool_root_skips_only_pairs_inside_the_flange_subtree() -> None:
    config = load_runtime_config("xarm6")
    urdf = compose_robot_urdf("xarm6_1305", "gripper_g2", accessory="d435")
    table = (EnvironmentObstacle("table", (1.2, 1.2, 0.05), (0.3, 0.0, -0.025)),)
    plain = _backend(config, urdf, obstacles=table)
    filtered = _backend(config, urdf, obstacles=table, tool_root_link=config.robot.ee_link)

    tool = filtered.tool_links
    assert {"link6", "left_outer_knuckle", "left_finger", "right_inner_knuckle"} <= tool
    assert any(link.startswith("d435") for link in tool)
    assert not tool & {"link_base", "link1", "link5"}

    before, after = _pairs(plain), _pairs(filtered)
    removed = set(before) - set(after)
    assert set(after) <= set(before)
    assert removed == {geometries for geometries, links in before.items() if links <= tool}
    assert {"left_finger", "right_finger"} in [before[g] for g in removed]
    assert all(not links <= tool for links in after.values())
    # Tool-arm and tool-environment pairs remain.
    assert frozenset(("link4", "left_finger")) in after.values()
    assert frozenset(("table", "left_finger")) in after.values()


def test_unknown_tool_root_fails_closed() -> None:
    config = load_runtime_config("xarm6")
    with pytest.raises(ValueError, match="tool root link 'flange' missing"):
        _backend(config, compose_robot_urdf("xarm6_1305", "gripper_g2"), tool_root_link="flange")


@pytest.mark.parametrize("gripper", ["robotiq_2f_85", "robotiq_2f_140"])
def test_preflight_backend_ignores_robotiq_internal_clearance(gripper: str) -> None:
    """Regression: 2F-85 outer/inner knuckles sit 4.06 / 4.34 mm apart when open, below the 5 mm margin."""
    config = load_runtime_config("xarm6", gripper=gripper)
    urdf = resolve_robot_urdf(
        config.robot.key, config.robot.urdf, assets_dir=config.robot.assets_dir, store=RepositoryAssetStore.discover()
    )
    q = np.asarray(config.arm.default_qpos_rad, dtype=np.float64)
    margin = float(config.safety.min_collision_distance_m)
    prefix = f"{gripper}_"

    plain = _backend(config, urdf)
    raw = plain.check_all_within_margin(q, stage="start", margin_m=margin, gripper_drive=config.gripper.open_drive)
    internal = [r for r in raw if r.link_a.startswith(prefix) and r.link_b.startswith(prefix)]
    internal_links = {frozenset((r.link_a, r.link_b)) for r in internal}
    # Four-bar pins: finger and inner knuckle meet by design (the former per-robot adjacent_collision_pairs).
    assert {
        frozenset((f"{prefix}left_finger", f"{prefix}left_inner_knuckle")),
        frozenset((f"{prefix}right_inner_knuckle", f"{prefix}right_finger")),
    } <= internal_links
    if gripper == "robotiq_2f_85":
        knuckle_pairs = {
            frozenset((f"{prefix}left_outer_knuckle", f"{prefix}left_inner_knuckle")),
            frozenset((f"{prefix}right_outer_knuckle", f"{prefix}right_inner_knuckle")),
        }
        assert knuckle_pairs <= internal_links
        knuckles = [r for r in internal if frozenset((r.link_a, r.link_b)) in knuckle_pairs]
        assert min(r.min_distance_m for r in knuckles) == pytest.approx(0.0040596, abs=1e-6)

    _, collision = build_manipulation_backends(
        config,
        urdf,
        obstacles=(EnvironmentObstacle("object", (0.03, 0.03, 0.03), (0.3, 0.0, 0.015)),),
        spawn_center_m=(0.3, 0.0, 0.015),
        place_center_m=(0.3, 0.3, 0.015),
        relocated_stages=(),
        object_width_m=0.03,
        calibrated_g2=True,
    )
    for drive in (config.gripper.open_drive, config.gripper.closed_drive):
        results = collision.check_all_within_margin(q, stage="start", margin_m=margin, gripper_drive=drive)
        assert not [r for r in results if r.link_a.startswith(prefix) and r.link_b.startswith(prefix)]
