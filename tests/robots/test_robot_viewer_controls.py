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
    lite6_demo.control_lite6_gripper_pose(robot, [6], [6, 7], LITE6_GRIPPER_SIM_CLOSED_DRIVE)
    assert robot.calls == [("control_position", (6,), None)]


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
