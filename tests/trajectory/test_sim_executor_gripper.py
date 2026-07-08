"""Simulation executor gripper-control tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from ufactory.robots.runtime import G2_GRIPPER_PARAMS, LITE6_GRIPPER_PARAMS
from ufactory.trajectory.mirror_executor import gap_m_from_drive
from ufactory.trajectory.segments import Program, Segment
from ufactory.trajectory.scene import (
    LITE6_FINGER_CLOSE_DESCENT,
    LITE6_FINGER_PAD_BELOW_FC,
    LITE6_GRASP_LINK6_Z_EXTRA_M,
    LITE6_GRASP_TABLE_CLEARANCE,
    LITE6_OBJ_SIZE,
    default_grasp_gap_m,
    drive_for_gap_m,
    dry_heights,
)
from ufactory.trajectory import sim_executor
from ufactory.trajectory.sim_executor import DEFAULT_GRIPPER_HOLD_BIAS_GAP_M, replay_sim


@pytest.mark.parametrize(
    "gripper,closed_gap,open_gap",
    [
        (G2_GRIPPER_PARAMS, 0.0, 0.084),
        (LITE6_GRIPPER_PARAMS, 0.020, 0.038),
    ],
)
def test_drive_gap_mapping_round_trips_physical_range(gripper, closed_gap, open_gap):
    close_drive = drive_for_gap_m(closed_gap, gripper)
    open_drive = drive_for_gap_m(open_gap, gripper)

    assert close_drive == pytest.approx(gripper.close_pos)
    assert open_drive == pytest.approx(gripper.open_pos)
    assert gap_m_from_drive(close_drive, gripper) == pytest.approx(closed_gap)
    assert gap_m_from_drive(open_drive, gripper) == pytest.approx(open_gap)


def test_default_grasp_gaps_match_contact_preload_targets():
    assert default_grasp_gap_m("xarm5") == pytest.approx(0.022)
    assert default_grasp_gap_m("xarm6") == pytest.approx(0.022)
    assert default_grasp_gap_m("xarm7") == pytest.approx(0.022)
    assert default_grasp_gap_m("uf850") == pytest.approx(0.022)
    assert default_grasp_gap_m("lite6") == pytest.approx(0.020)


def test_lite6_default_grasp_height_targets_flat_finger_pad():
    heights = dry_heights("lite6")
    # link6 height = table clearance + fingertip-plate length + link6->fc offset.
    assert heights.grasp_link6_z == pytest.approx(0.0963)

    # Finger-center (fc) height above the table.
    fc_above_table = (
        LITE6_GRASP_TABLE_CLEARANCE
        + LITE6_FINGER_CLOSE_DESCENT
        + LITE6_FINGER_PAD_BELOW_FC
        + LITE6_GRASP_LINK6_Z_EXTRA_M
    )
    assert fc_above_table == pytest.approx(0.042)

    # Positive clearance keeps the low boss/stop region above the cube top while
    # the large flat inner pad still spans the cube sides.
    cube_top_below_fc = fc_above_table - LITE6_OBJ_SIZE[2]
    assert cube_top_below_fc == pytest.approx(0.012)
    boss_reach_below_fc = 0.0065
    assert cube_top_below_fc > boss_reach_below_fc


class _FakeRobot:
    def __init__(self, *, contact_limited_drive: float) -> None:
        self.contact_limited_drive = contact_limited_drive
        self.drive = 0.0
        self.grip_commands: list[float] = []
        self._solver = _FakeSolver()

    def control_dofs_position(self, target, dofs_idx) -> None:
        if dofs_idx == [1]:
            cmd = float(target.reshape(-1)[0].item())
            self.grip_commands.append(cmd)
            self.drive = min(cmd, self.contact_limited_drive)

    def get_dofs_position(self, _dofs_idx):
        return torch.tensor([[self.drive]], dtype=torch.float32)


class _FakeScene:
    def step(self) -> None:
        pass


class _FakeSolver:
    def __init__(self) -> None:
        self.added: list[tuple[int, int]] = []
        self.deleted: list[tuple[int, int]] = []

    def add_weld_constraint(self, link1_idx, link2_idx) -> None:
        self.added.append((int(link1_idx), int(link2_idx)))

    def delete_weld_constraint(self, link1_idx, link2_idx) -> None:
        self.deleted.append((int(link1_idx), int(link2_idx)))


class _FakeLink:
    def __init__(self, idx: int = 0, pos: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        self.idx = idx
        self._pos = torch.tensor([pos], dtype=torch.float32)

    def get_pos(self):
        return self._pos


class _FakeObj:
    def __init__(self, pos: tuple[float, float, float] = (0.30, 0.30, 0.02), link_idx: int = 2) -> None:
        self._pos = torch.tensor([pos], dtype=torch.float32)
        self.links = [_FakeLink(link_idx, pos)]

    def get_pos(self):
        return self._pos

    def get_quat(self):
        return torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    def set_pos(self, pos, zero_velocity=True) -> None:
        self._pos = pos

    def set_quat(self, quat, zero_velocity=True) -> None:
        return None


class _FakeObjWithContacts(_FakeObj):
    def __init__(
        self,
        *,
        pos: tuple[float, float, float] = (0.0, 0.0, 0.05),
        contacts: tuple[int, ...] = (),
    ) -> None:
        super().__init__(pos=pos, link_idx=20)
        self._contacts = tuple(int(link_idx) for link_idx in contacts)

    def get_contacts(self, with_entity=None, exclude_self_contact=False):
        n = len(self._contacts)
        return {
            "link_a": torch.tensor(self._contacts, dtype=torch.int32),
            "link_b": torch.full((n,), 20, dtype=torch.int32),
            "force_a": torch.ones((n, 3), dtype=torch.float32),
        }


@dataclass
class _FakeCtx:
    robot: _FakeRobot
    scene: _FakeScene
    obj: _FakeObj
    ik_link: _FakeLink
    arm_dof_idx: list[int]
    gripper_dof_idx: list[int]
    home_qpos: np.ndarray
    left_finger: _FakeLink | None = None
    right_finger: _FakeLink | None = None
    obj_size: tuple[float, float, float] = (0.030, 0.030, 0.030)
    place_xy: tuple[float, float] = (0.30, 0.30)
    home_pos_base: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gripper: object = G2_GRIPPER_PARAMS

    def base_to_world(self, pos):
        return list(pos)


def test_replay_sim_holds_contact_limited_gripper_drive_after_close(monkeypatch):
    monkeypatch.setattr(sim_executor.gs, "device", torch.device("cpu"), raising=False)
    monkeypatch.setattr(sim_executor.gs, "tc_float", torch.float32, raising=False)

    robot = _FakeRobot(contact_limited_drive=0.45)
    ctx = _FakeCtx(
        robot=robot,
        scene=_FakeScene(),
        obj=_FakeObj(),
        ik_link=_FakeLink(),
        arm_dof_idx=[0],
        gripper_dof_idx=[1],
        home_qpos=np.zeros(2),
    )
    program = Program(
        rate=50.0,
        segments=[
            Segment(kind="gripper", duration=0.02, v_max=0.0, a_max=0.0, gap_start=0.084, gap_end=0.024),
            Segment(kind="movej", duration=0.02, v_max=1.0, a_max=1.0, q_start=np.zeros(1), q_end=np.zeros(1)),
        ],
    )

    replay_sim(program, ctx)

    gripper = G2_GRIPPER_PARAMS
    overclosed_target = drive_for_gap_m(0.024, gripper)
    # Hold closedness = actual contact-limited closedness + bias, capped at
    # the originally-planned target closedness (see sim_executor.replay_sim).
    actual_closedness = (0.45 - gripper.open_pos) / (gripper.close_pos - gripper.open_pos)
    target_closedness = (overclosed_target - gripper.open_pos) / (gripper.close_pos - gripper.open_pos)
    bias_closedness = DEFAULT_GRIPPER_HOLD_BIAS_GAP_M / (gripper.open_gap_m - gripper.closed_gap_m)
    expected_closedness = min(target_closedness, actual_closedness + bias_closedness)
    expected_hold = gripper.open_pos + expected_closedness * (gripper.close_pos - gripper.open_pos)
    assert robot.grip_commands[0] == pytest.approx(overclosed_target)
    assert robot.grip_commands[-1] == pytest.approx(expected_hold)
    assert robot.grip_commands[-1] < overclosed_target


def test_replay_sim_does_not_weld_by_default_even_with_bilateral_contact(monkeypatch):
    monkeypatch.setattr(sim_executor.gs, "device", torch.device("cpu"), raising=False)
    monkeypatch.setattr(sim_executor.gs, "tc_float", torch.float32, raising=False)

    robot = _FakeRobot(contact_limited_drive=0.45)
    ctx = _FakeCtx(
        robot=robot,
        scene=_FakeScene(),
        obj=_FakeObjWithContacts(contacts=(10, 11)),
        ik_link=_FakeLink(idx=10),
        left_finger=_FakeLink(idx=10, pos=(0.0, -0.02, 0.0)),
        right_finger=_FakeLink(idx=11, pos=(0.0, 0.02, 0.0)),
        arm_dof_idx=[0],
        gripper_dof_idx=[1],
        home_qpos=np.zeros(2),
    )
    program = Program(
        rate=50.0,
        segments=[
            Segment(kind="gripper", duration=0.02, v_max=0.0, a_max=0.0, gap_start=0.084, gap_end=0.024),
            Segment(kind="movej", duration=0.02, v_max=1.0, a_max=1.0, q_start=np.zeros(1), q_end=np.zeros(1)),
            Segment(kind="gripper", duration=0.02, v_max=0.0, a_max=0.0, gap_start=0.024, gap_end=0.084),
        ],
    )

    replay_sim(program, ctx)

    assert robot._solver.added == []
    assert robot._solver.deleted == []


def test_replay_sim_adds_and_deletes_debug_grasp_weld_after_bilateral_contact(monkeypatch):
    monkeypatch.setattr(sim_executor.gs, "device", torch.device("cpu"), raising=False)
    monkeypatch.setattr(sim_executor.gs, "tc_float", torch.float32, raising=False)

    robot = _FakeRobot(contact_limited_drive=0.45)
    ctx = _FakeCtx(
        robot=robot,
        scene=_FakeScene(),
        obj=_FakeObjWithContacts(contacts=(10, 11)),
        ik_link=_FakeLink(idx=30),
        left_finger=_FakeLink(idx=10, pos=(0.0, -0.02, 0.0)),
        right_finger=_FakeLink(idx=11, pos=(0.0, 0.02, 0.0)),
        arm_dof_idx=[0],
        gripper_dof_idx=[1],
        home_qpos=np.zeros(2),
    )
    program = Program(
        rate=50.0,
        segments=[
            Segment(kind="gripper", duration=0.02, v_max=0.0, a_max=0.0, gap_start=0.084, gap_end=0.024),
            Segment(kind="movej", duration=0.02, v_max=1.0, a_max=1.0, q_start=np.zeros(1), q_end=np.zeros(1)),
            Segment(kind="gripper", duration=0.02, v_max=0.0, a_max=0.0, gap_start=0.024, gap_end=0.084),
        ],
    )

    replay_sim(program, ctx, stabilize_grasp_weld=True)

    assert robot._solver.added == [(30, 20)]
    assert robot._solver.deleted == [(30, 20)]


def test_debug_grasp_weld_requires_bilateral_finger_contact():
    robot = _FakeRobot(contact_limited_drive=0.45)
    no_contact = _FakeCtx(
        robot=robot,
        scene=_FakeScene(),
        obj=_FakeObjWithContacts(contacts=()),
        ik_link=_FakeLink(idx=30),
        left_finger=_FakeLink(idx=10, pos=(0.0, -0.02, 0.0)),
        right_finger=_FakeLink(idx=11, pos=(0.0, 0.02, 0.0)),
        arm_dof_idx=[0],
        gripper_dof_idx=[1],
        home_qpos=np.zeros(2),
        gripper=LITE6_GRIPPER_PARAMS,
    )
    near_left_only = _FakeCtx(
        robot=robot,
        scene=_FakeScene(),
        obj=_FakeObjWithContacts(contacts=(10,)),
        ik_link=_FakeLink(idx=30),
        left_finger=_FakeLink(idx=10, pos=(0.0, -0.02, 0.0)),
        right_finger=_FakeLink(idx=11, pos=(0.0, 0.02, 0.0)),
        arm_dof_idx=[0],
        gripper_dof_idx=[1],
        home_qpos=np.zeros(2),
        gripper=LITE6_GRIPPER_PARAMS,
    )
    near_both = _FakeCtx(
        robot=robot,
        scene=_FakeScene(),
        obj=_FakeObjWithContacts(contacts=(10, 11)),
        ik_link=_FakeLink(idx=30),
        left_finger=_FakeLink(idx=10, pos=(0.0, -0.02, 0.0)),
        right_finger=_FakeLink(idx=11, pos=(0.0, 0.02, 0.0)),
        arm_dof_idx=[0],
        gripper_dof_idx=[1],
        home_qpos=np.zeros(2),
        gripper=LITE6_GRIPPER_PARAMS,
    )

    assert sim_executor._should_weld_grasp(no_contact, 0.08) is False
    assert sim_executor._should_weld_grasp(near_left_only, 0.08) is False
    assert sim_executor._should_weld_grasp(near_both, 0.08) is True
