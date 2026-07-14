"""GPU regression: xArm5 pose 4 with STL collision meshes and PD-hold check."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("genesis")
pytest.importorskip("pinocchio")

pytestmark = pytest.mark.gpu

from ufactory.dynamics.analysis import evaluate_pd_hold_gate  # noqa: E402
from ufactory.dynamics.probe import (  # noqa: E402
    build_genesis_scene,
    capture_self_contacts,
    genesis_pd_hold_torque_at_q,
    set_pd_gains,
)
from ufactory.robots.paths import robot_urdf
from ufactory.robots.runtime import get_robot_runtime_profile  # noqa: E402

POSE4_DEG = [90.0, -90.0, -60.0, 160.0, -90.0]
LINK3_LINK5 = frozenset({"link3", "link5"})


@pytest.fixture(autouse=True)
def _genesis_teardown():
    yield
    import genesis as gs

    if getattr(gs, "_initialized", False):
        gs.destroy()


def _pose4_target_q() -> np.ndarray:
    return np.deg2rad(POSE4_DEG)


def _link_names(robot) -> dict[int, str]:
    return {link.idx: link.name.split("/")[-1] for link in robot.links}


def _link_pair_force(contacts: list[dict], names: dict[int, str], pair: frozenset[str]) -> float:
    max_force = 0.0
    for c in contacts:
        a = names.get(c["link_a"], str(c["link_a"]))
        b = names.get(c["link_b"], str(c["link_b"]))
        if frozenset((a, b)) == pair:
            max_force = max(max_force, float(c["force_n"]))
    return max_force


def test_xarm5_pose4_stl_collision_pd_hold_check_bypass():
    """Pose 4: STL collision loads; J4 PD may saturate but check allows hardware via pin_G."""
    runtime = get_robot_runtime_profile("xarm5")
    urdf = robot_urdf("xarm5_1305")
    target_q = _pose4_target_q()

    scene, robot, _ee, dof_idx = build_genesis_scene(
        urdf,
        runtime_profile=runtime,
        show_viewer=False,
    )
    sample = genesis_pd_hold_torque_at_q(
        robot,
        scene,
        dof_idx,
        target_q,
        runtime_profile=runtime,
        capture_contacts=True,
    )

    import pinocchio as pin

    model = pin.buildModelFromUrdf(urdf)
    data = model.createData()
    pin_g_target = pin.computeGeneralizedGravity(model, data, target_q)

    # J4 gravity at target should stay ~1 Nm (matches real robot / SDK), not PD-saturation scale.
    assert abs(pin_g_target[3]) < 2.0
    assert abs(sample.pd_hold_tau[3]) <= runtime.arm.effort_limits[3] + 1e-6

    contacts = sample.self_contacts or []
    names = _link_names(robot)
    link35_force = _link_pair_force(contacts, names, LINK3_LINK5)
    assert link35_force > 10.0, f"expected link3↔link5 spurious contact, got {link35_force:.1f} N"

    reference = type("Ref", (), {"gravity": lambda _self, q: pin.computeGeneralizedGravity(model, data, q)})()
    gate = evaluate_pd_hold_gate(sample, target_q, runtime_profile=runtime, reference=reference)

    if not sample.settled:
        assert gate.block_hardware is False
        assert gate.reason == "pd_tracking_saturation"
        assert gate.pin_gravity_theory is not None
        assert abs(gate.pin_gravity_theory[3]) < 2.0
    else:
        assert abs(sample.pd_hold_tau[3]) < 5.0


def test_xarm5_pose4_kinematic_self_contact():
    """Kinematic teleport: link3↔link5 overlap indicates mesh geometry, not PD artifact."""
    runtime = get_robot_runtime_profile("xarm5")
    urdf = robot_urdf("xarm5_1305")
    target_q = _pose4_target_q()

    scene, robot, _ee, dof_idx = build_genesis_scene(
        urdf,
        runtime_profile=runtime,
        show_viewer=False,
    )
    target_f = np.asarray(target_q, dtype=np.float32)
    robot.set_dofs_position(target_f, dof_idx, zero_velocity=True)
    set_pd_gains(robot, dof_idx, runtime)
    robot.set_dofs_kp(np.zeros(len(dof_idx), dtype=np.float32), dof_idx)
    robot.set_dofs_kv(np.zeros(len(dof_idx), dtype=np.float32), dof_idx)
    for _ in range(20):
        scene.step()

    contacts = capture_self_contacts(robot)
    names = _link_names(robot)
    link35_force = _link_pair_force(contacts, names, LINK3_LINK5)
    assert link35_force > 10.0, f"kinematic link3↔link5 force {link35_force:.1f} N"
