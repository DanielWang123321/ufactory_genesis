"""Control-mode regression tests for the shared GLB viewer."""

from __future__ import annotations

from contextlib import nullcontext
import importlib
import sys
import types
from pathlib import Path

_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))
from ufactory.grippers.lite6 import (  # noqa: E402
    LITE6_GRIPPER_SIM_CLOSED_DRIVE,
    LITE6_GRIPPER_SIM_OPEN_DRIVE,
)
from ufactory.robots.registry import get_robot_profile  # noqa: E402


class _FakeOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeURDF:
    def __init__(self, **kwargs):
        self.file = kwargs["file"]
        self.kwargs = kwargs


class _FakeJoint:
    def __init__(self, name: str, dof_idx: int):
        self.name = name
        self.dofs_idx_local = [dof_idx]


class _FakeRobot:
    def __init__(self):
        self.joints = [
            *[_FakeJoint(f"joint{i}", i - 1) for i in range(1, 6)],
            _FakeJoint("drive_joint", 5),
            _FakeJoint("left_finger_joint", 6),
            _FakeJoint("right_finger_joint", 7),
        ]
        self.links = []
        self.calls: list[tuple[str, tuple[int, ...], bool | None]] = []

    def set_dofs_kp(self, values, dof_idx):
        self.calls.append(("kp", tuple(dof_idx), None))

    def set_dofs_kv(self, values, dof_idx):
        self.calls.append(("kv", tuple(dof_idx), None))

    def set_dofs_force_range(self, lower, upper, dof_idx):
        self.calls.append(("force_range", tuple(dof_idx), None))

    def set_dofs_damping(self, values, dof_idx):
        self.calls.append(("damping", tuple(dof_idx), None))

    def set_dofs_frictionloss(self, values, dof_idx):
        self.calls.append(("frictionloss", tuple(dof_idx), None))

    def set_dofs_position(self, values, dof_idx, zero_velocity=None):
        self.calls.append(("set_position", tuple(dof_idx), zero_velocity))

    def control_dofs_position(self, values, dof_idx):
        self.calls.append(("control_position", tuple(dof_idx), None))

    def control_dofs_force(self, values, dof_idx):
        import numpy as np

        assert np.asarray(values).ndim == 1
        self.calls.append(("control_force", tuple(dof_idx), None))


class _FakeScene:
    def __init__(self, robot: _FakeRobot):
        self.robot = robot
        self.step_count = 0
        self.step_calls: list[dict] = []
        self.visualizer = types.SimpleNamespace(viewer=None)

    def add_entity(self, morph, surface=None):
        if getattr(morph, "file", "") == "urdf/plane/plane.urdf":
            return object()
        return self.robot

    def build(self, **kwargs):
        pass

    def step(self, *args, **kwargs):
        self.step_count += 1
        self.step_calls.append(kwargs)


class _FakeVisualizer:
    def __init__(self):
        self.viewer = object()
        self.update_calls: list[dict] = []

    def update(self, **kwargs):
        self.update_calls.append(kwargs)


class _FakeGenesis:
    cpu = "fake-cpu"
    gpu = "fake-gpu"
    constraint_solver = types.SimpleNamespace(Newton="newton")
    friction_cone = types.SimpleNamespace(pyramidal="pyramidal", elliptic="elliptic")
    contact_resolution = types.SimpleNamespace(convex="convex", signorini="signorini")
    options = types.SimpleNamespace(
        ViewerOptions=_FakeOptions,
        SimOptions=_FakeOptions,
        RigidOptions=_FakeOptions,
    )
    morphs = types.SimpleNamespace(URDF=_FakeURDF)

    def __init__(self, scene: _FakeScene):
        self.scene = scene

    def init(self, backend):
        self.backend = backend

    def Scene(self, **kwargs):
        self.scene.kwargs = kwargs
        return self.scene


def _load_robot_viewer(monkeypatch):
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    try:
        import genesis  # noqa: F401
    except ModuleNotFoundError:
        monkeypatch.setitem(sys.modules, "genesis", types.SimpleNamespace())
    return importlib.import_module("ufactory.visualization.robot_viewer")


def test_g2_demo_holds_arm_with_pd_and_only_drives_master(monkeypatch, tmp_path):
    robot_viewer = _load_robot_viewer(monkeypatch)
    robot = _FakeRobot()
    scene = _FakeScene(robot)
    fake_gs = _FakeGenesis(scene)
    arm_pd_calls = []

    monkeypatch.setattr(robot_viewer, "gs", fake_gs)
    monkeypatch.setattr(robot_viewer, "require_genesis_runtime", lambda _gs: None)
    monkeypatch.setattr(robot_viewer, "glb_pbr_surfaces", nullcontext)
    monkeypatch.setattr(robot_viewer, "glb_view_surface", lambda: object())
    monkeypatch.setattr(robot_viewer, "setup_arm_pd", lambda *args: arm_pd_calls.append(args))
    monkeypatch.setattr(robot_viewer, "resolve_robot_link", lambda *_args: object())
    monkeypatch.setattr("ufactory.kinematics.genesis.solve_link6_ik", lambda *_args, **_kwargs: [0.0] * 5)
    model = tmp_path / "fake.urdf"
    model.write_text('<robot><joint name="drive_joint"/></robot>')

    def setup_gripper(robot, drive, all_idx):
        robot.set_dofs_kp([40], drive)
        robot._ufactory_g2_controller = types.SimpleNamespace(gap_mm=lambda: 84)

    monkeypatch.setattr(robot_viewer, "setup_gripper_pd", setup_gripper)
    monkeypatch.setattr(
        robot_viewer,
        "set_gripper_pose",
        lambda robot, drive, all_idx, value: robot.set_dofs_position([value] * len(all_idx), all_idx),
    )
    monkeypatch.setattr(
        robot_viewer,
        "control_gripper_pose",
        lambda robot, drive, all_idx, value: robot.control_dofs_position([value], drive),
    )

    robot_viewer.run_glb_viewer(
        get_robot_profile("xarm5_1305"),
        str(model),
        headless=True,
        pd_demo=False,
        gripper_demo=True,
        headless_steps=2,
    )

    arm_dofs = set(range(5))
    assert len(arm_pd_calls) == 1
    assert [call for call in robot.calls if call[0] == "control_position" and arm_dofs.intersection(call[1])]
    assert any(call == ("set_position", (0, 1, 2, 3, 4), True) for call in robot.calls)
    assert ("kp", (5,), None) in robot.calls
    assert ("control_position", (5,), None) in robot.calls
    assert ("control_position", (5, 6, 7), None) not in robot.calls
    assert len([call for call in robot.calls if call[0] == "set_position" and 5 in call[1]]) == 1
    assert scene.step_count == 5


def test_kinematic_step_reholds_before_visual_update(monkeypatch):
    robot_viewer = _load_robot_viewer(monkeypatch)
    robot = _FakeRobot()
    scene = _FakeScene(robot)
    scene.visualizer = _FakeVisualizer()

    robot_viewer._kinematic_step(
        scene,
        robot,
        arm_kinematic_hold=True,
        idle_gripper_kinematic_hold=False,
        arm_dof_idx=[0, 1, 2, 3, 4],
        home=[0.0, 0.0, 0.0, 0.0, 0.0],
        all_gripper_dof_idx=[5, 6],
        all_lite6_gripper_dof_idx=[],
        all_bio_gripper_g2_dof_idx=[],
    )

    assert scene.step_calls == [{"update_visualizer": False}]
    assert scene.visualizer.update_calls == [{"force": False}]
    assert robot.calls[-1] == ("set_position", (0, 1, 2, 3, 4), True)


def test_lite6_viewer_gripper_demo_uses_reversed_open_close_semantics():
    lite6_demo = importlib.import_module("ufactory.visualization.lite6_gripper_viewer")

    assert lite6_demo.LITE6_GRIPPER_CLOSE == LITE6_GRIPPER_SIM_CLOSED_DRIVE
    assert lite6_demo.LITE6_GRIPPER_OPEN == LITE6_GRIPPER_SIM_OPEN_DRIVE
    assert lite6_demo.lite6_gripper_demo_target(0) == LITE6_GRIPPER_SIM_OPEN_DRIVE
    assert lite6_demo.lite6_gripper_demo_target(lite6_demo.LITE6_GRIPPER_HOLD_STEPS) == LITE6_GRIPPER_SIM_CLOSED_DRIVE

    robot = _FakeRobot()
    lite6_demo.setup_lite6_gripper_pd(robot, [6], [6, 7])
    assert ("kp", (6,), None) in robot.calls
    assert ("kv", (6,), None) in robot.calls
    assert ("force_range", (6,), None) in robot.calls
    assert ("control_force", (7,), None) in robot.calls

    robot.calls.clear()
    lite6_demo.set_lite6_gripper_pose(robot, [6], [6, 7], LITE6_GRIPPER_SIM_OPEN_DRIVE)
    assert ("set_position", (6, 7), None) in robot.calls
    assert ("control_position", (6,), None) in robot.calls
    assert ("control_force", (7,), None) in robot.calls

    robot.calls.clear()
    lite6_demo.control_lite6_gripper_pose(robot, [6], [6, 7], LITE6_GRIPPER_SIM_CLOSED_DRIVE)
    assert robot.calls == [("control_position", (6,), None)]


def test_lite6_gripper_demo_runs_in_headless_viewer(monkeypatch, tmp_path):
    robot_viewer = _load_robot_viewer(monkeypatch)
    robot = _FakeRobot()
    robot.joints = [
        *[_FakeJoint(f"joint{i}", i - 1) for i in range(1, 7)],
        _FakeJoint("finger_joint1", 6),
        _FakeJoint("finger_joint2", 7),
    ]
    scene = _FakeScene(robot)
    fake_gs = _FakeGenesis(scene)

    monkeypatch.setattr(robot_viewer, "gs", fake_gs)
    monkeypatch.setattr(robot_viewer, "require_genesis_runtime", lambda _gs: None)
    monkeypatch.setattr(robot_viewer, "glb_pbr_surfaces", nullcontext)
    monkeypatch.setattr(robot_viewer, "glb_view_surface", lambda: object())
    monkeypatch.setattr(robot_viewer, "configure_lite6_collision_filters", lambda _robot: None)
    model = tmp_path / "fake_lite6_gripper.urdf"
    model.write_text('<robot><joint name="finger_joint1"/></robot>')

    robot_viewer.run_glb_viewer(
        get_robot_profile("lite6"),
        str(model),
        headless=True,
        pd_demo=False,
        gripper_demo=True,
        headless_steps=2,
    )

    assert ("kp", (6,), None) in robot.calls
    assert ("control_force", (7,), None) in robot.calls
    assert ("control_position", (6,), None) in robot.calls
    assert scene.step_count == 5


def test_bio_g2_demo_holds_arm_with_pd_and_drives_only_right_finger(monkeypatch, tmp_path):
    robot_viewer = _load_robot_viewer(monkeypatch)
    robot = _FakeRobot()
    robot.joints = [
        *[_FakeJoint(f"joint{i}", i - 1) for i in range(1, 7)],
        _FakeJoint("bio_gripper_g2_right_finger_joint", 6),
        _FakeJoint("bio_gripper_g2_left_finger_joint", 7),
    ]
    scene = _FakeScene(robot)
    fake_gs = _FakeGenesis(scene)
    arm_pd_calls = []

    monkeypatch.setattr(robot_viewer, "gs", fake_gs)
    monkeypatch.setattr(robot_viewer, "require_genesis_runtime", lambda _gs: None)
    monkeypatch.setattr(robot_viewer, "glb_pbr_surfaces", nullcontext)
    monkeypatch.setattr(robot_viewer, "glb_view_surface", lambda: object())
    monkeypatch.setattr(robot_viewer, "setup_arm_pd", lambda *args: arm_pd_calls.append(args))
    monkeypatch.setattr(robot_viewer, "resolve_robot_link", lambda *_args: object())
    monkeypatch.setattr("ufactory.kinematics.genesis.solve_link6_ik", lambda *_args, **_kwargs: [0.0] * 6)
    model = tmp_path / "fake_bio_g2.urdf"
    model.write_text('<robot><joint name="bio_gripper_g2_right_finger_joint"/></robot>')

    result = robot_viewer.run_glb_viewer(
        get_robot_profile("xarm6_1305"),
        str(model),
        headless=True,
        pd_demo=False,
        gripper_demo=True,
        headless_steps=2,
    )

    assert result is not None and result["ticks"] == 2
    assert len(arm_pd_calls) == 1
    assert ("kp", (6,), None) in robot.calls
    assert ("kp", (7,), None) in robot.calls
    assert ("control_force", (7,), None) in robot.calls
    assert ("control_position", (6,), None) in robot.calls
    assert ("control_position", (6, 7), None) not in robot.calls


def test_g2_demo_cycle_starts_closing_and_waits_half_second_after_arrival():
    from ufactory.visualization.gripper_g2_viewer import (
        GRIPPER_CLOSE,
        GRIPPER_OPEN,
        step_gripper_demo_target,
        get_g2_demo_cycle,
    )

    state = {"gap_mm": 84.0, "ref_mm": 84.0}
    robot = types.SimpleNamespace(
        _ufactory_g2_demo_settings=(0.0, 50.0, 50.0),
        _ufactory_g2_controller=types.SimpleNamespace(
            gap_mm=lambda: state["gap_mm"],
            controller=types.SimpleNamespace(reference_gap_mm=[state["ref_mm"]]),
        ),
    )
    cycle = get_g2_demo_cycle(robot, dt=0.02)
    assert cycle.hold_steps_required == 25

    # Step 0 starts closing immediately without waiting 200 open steps.
    drive, label = step_gripper_demo_target(robot, dt=0.02)
    assert (drive, label) == (GRIPPER_CLOSE, "closed")

    # While moving toward 0 mm, hold_steps stays at 0.
    for gap in (60.0, 30.0, 5.0):
        state["gap_mm"] = gap
        state["ref_mm"] = gap
        robot._ufactory_g2_controller.controller.reference_gap_mm = [gap]
        cycle.on_step_completed(robot)
        drive, label = step_gripper_demo_target(robot, dt=0.02)
        assert (drive, label) == (GRIPPER_CLOSE, "closed")
        assert cycle.hold_steps == 0

    # Arrive at closed target (<= 1.0 mm) and hold for 25 steps (0.5 s at dt=0.02s).
    state["gap_mm"] = 0.2
    robot._ufactory_g2_controller.controller.reference_gap_mm = [0.0]
    for expected_hold in range(1, 25):
        cycle.on_step_completed(robot)
        assert cycle.hold_steps == expected_hold
        drive, label = step_gripper_demo_target(robot, dt=0.02)
        assert (drive, label) == (GRIPPER_CLOSE, "closed")

    # On the 25th arrived step, phase switches to "open".
    cycle.on_step_completed(robot)
    drive, label = step_gripper_demo_target(robot, dt=0.02)
    assert (drive, label) == (GRIPPER_OPEN, "open")
    assert cycle.completed_cycles == 0

    # Arrive at open target (84 mm) and hold for 25 steps -> completes 1 full cycle.
    state["gap_mm"] = 83.8
    robot._ufactory_g2_controller.controller.reference_gap_mm = [84.0]
    for _ in range(24):
        cycle.on_step_completed(robot)
        step_gripper_demo_target(robot, dt=0.02)
    assert cycle.completed_cycles == 0
    cycle.on_step_completed(robot)
    assert cycle.completed_cycles == 1
    drive, label = step_gripper_demo_target(robot, dt=0.02)
    assert (drive, label) == (GRIPPER_CLOSE, "closed")


def test_bio_g2_and_lite6_demo_cycles_wait_half_second_after_arrival():
    from ufactory.grippers.bio_g2 import BioGripperG2
    from ufactory.visualization.lite6_gripper_viewer import (
        LITE6_GRIPPER_CLOSE,
        LITE6_GRIPPER_OPEN,
        get_lite6_gripper_demo_cycle,
        step_lite6_gripper_demo_target,
    )

    # Bio Gripper G2 (dt=0.02s -> 25 hold steps for 0.5s)
    bio_state = {"gap_mm": 150.0, "ref_mm": 150.0}
    bio_robot = _FakeRobot()
    bio_robot.joints = [
        _FakeJoint("bio_gripper_g2_right_finger_joint", 6),
        _FakeJoint("bio_gripper_g2_left_finger_joint", 7),
    ]
    bio_robot._ufactory_bio_g2_demo_settings = (71.0, 50.0, 50.0, None)
    bio_robot._ufactory_bio_g2_controller = types.SimpleNamespace(
        gap_mm=lambda: bio_state["gap_mm"],
        telemetry=lambda: {"reference_gap_mm": bio_state["ref_mm"]},
    )
    bio = BioGripperG2(bio_robot)
    assert bio.step_demo_target() == (BioGripperG2.CLOSE_POS, "closed")
    bio_state["gap_mm"] = 71.2
    bio_state["ref_mm"] = 71.0
    for _ in range(25):
        bio.on_step_completed(bio_robot)
        if bio.demo_phase == "closed":
            bio.step_demo_target()
    assert bio.step_demo_target() == (BioGripperG2.OPEN_POS, "open")

    # Lite6 Gripper (dt=0.01s -> 50 hold steps for 0.5s)
    lite6_pos = [LITE6_GRIPPER_OPEN]
    lite6_robot = types.SimpleNamespace(get_dofs_position=lambda _idx: [lite6_pos[0]])
    lite6_cycle = get_lite6_gripper_demo_cycle(lite6_robot, [6], dt=0.01)
    assert lite6_cycle.hold_steps_required == 50
    assert step_lite6_gripper_demo_target(lite6_robot, [6], dt=0.01) == (LITE6_GRIPPER_CLOSE, "closed")
    lite6_pos[0] = 0.0001
    for _ in range(50):
        lite6_cycle.on_step_completed(lite6_robot)
        if lite6_cycle.phase == "closed":
            step_lite6_gripper_demo_target(lite6_robot, [6], dt=0.01)
    assert step_lite6_gripper_demo_target(lite6_robot, [6], dt=0.01) == (LITE6_GRIPPER_OPEN, "open")
