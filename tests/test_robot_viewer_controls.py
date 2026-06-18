"""Control-mode regression tests for the shared GLB viewer."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = PROJECT_ROOT / "examples"
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))
if str(EXAMPLES_ROOT) not in sys.path:
  sys.path.insert(0, str(EXAMPLES_ROOT))

from ufactory.robot_registry import get_robot_profile  # noqa: E402


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

  def build(self):
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
  gpu = "fake-gpu"
  options = types.SimpleNamespace(ViewerOptions=_FakeOptions, SimOptions=_FakeOptions)
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
    import genesis  # noqa: F401
  except ModuleNotFoundError:
    monkeypatch.setitem(sys.modules, "genesis", types.SimpleNamespace())
  return importlib.import_module("_robot_viewer")


def test_gripper_demo_without_pd_holds_arm_kinematically(monkeypatch):
  robot_viewer = _load_robot_viewer(monkeypatch)
  robot = _FakeRobot()
  scene = _FakeScene(robot)
  fake_gs = _FakeGenesis(scene)
  arm_pd_calls = []

  monkeypatch.setattr(robot_viewer, "gs", fake_gs)
  monkeypatch.setattr(robot_viewer, "enable_glb_pbr_surfaces", lambda: None)
  monkeypatch.setattr(robot_viewer, "glb_view_surface", lambda: object())
  monkeypatch.setattr(robot_viewer, "setup_arm_pd", lambda *args: arm_pd_calls.append(args))

  robot_viewer.run_glb_viewer(
    get_robot_profile("xarm5_1305"),
    "fake.urdf",
    headless=True,
    pd_demo=False,
    gripper_demo=True,
  )

  arm_dofs = set(range(5))
  assert arm_pd_calls == []
  assert not [
    call
    for call in robot.calls
    if call[0] == "control_position" and arm_dofs.intersection(call[1])
  ]
  assert any(call == ("set_position", (0, 1, 2, 3, 4), True) for call in robot.calls)
  assert ("kp", (5,), None) in robot.calls
  assert ("control_position", (5, 6, 7), None) in robot.calls
  assert scene.step_count == 3


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
