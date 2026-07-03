"""Simulation replay executor for the trajectory pipeline.

Replays a :class:`ufactory.trajectory.segments.Program` in Genesis: each tick
sets absolute joint targets via PD position control (arm), while MoveL segments
solve link6 IK per tick and Gripper segments map the gap (m) to a drive value.
Reports per-segment duration/profile and final place/home error.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

import genesis as gs

from ufactory.grippers.g2 import GRIPPER_G2_OPEN_GAP_M
from ufactory.trajectory.mirror_executor import resolve_tick_arm_q, resolve_tick_grip_drive
from ufactory.trajectory.scene import TrajSceneContext, drive_for_gap_m
from ufactory.trajectory.segments import Program, Segment


@dataclass
class PhaseStatus:
    label: str
    seg_index: int
    ticks: int
    duration: float
    v_max: float
    a_max: float
    eside_arm_mm: float = 0.0
    eside_grip_mm: float = 0.0
    obj_pos_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    link6_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class SimReport:
    phases: list[PhaseStatus] = field(default_factory=list)
    place_error_mm: float = 0.0
    home_drift_mm: float = 0.0
    total_ticks: int = 0
    total_duration: float = 0.0


def replay_sim(
    program: Program,
    ctx: TrajSceneContext,
    *,
    gripper_holds_after_grasp: bool = True,
    on_phase: Callable[[PhaseStatus, Segment], None] | None = None,
) -> SimReport:
    """Replay ``program`` in the Genesis scene and return a SimReport.

    MoveJ segments stream arm joint targets directly; MoveL segments solve
    link6 IK per tick (base->world). Gripper segments stream the drive value.
    The current gripper drive is held constant during arm MoveJ/MoveL segments
    (open before grasp, closed after).
    """
    robot = ctx.robot
    scene = ctx.scene
    ik_link = ctx.ik_link
    arm_idx = ctx.arm_dof_idx
    grip_idx = ctx.gripper_dof_idx
    rate = float(program.rate)
    report = SimReport(total_duration=program.total_duration, total_ticks=program.total_ticks)

    current_grip_drive = drive_for_gap_m(GRIPPER_G2_OPEN_GAP_M)
    last_q_arm = ctx.home_qpos[arm_idx].astype(np.float64).copy() if len(ctx.home_qpos) else None

    for si, seg in enumerate(program.segments):
        samples, n = seg.samples(rate)
        if n == 0:
            continue
        status = PhaseStatus(
            label=seg.label or seg.kind, seg_index=si, ticks=n, duration=seg.duration,
            v_max=seg.v_max, a_max=seg.a_max,
        )
        if seg.kind == "gripper":
            for t in range(n):
                current_grip_drive = resolve_tick_grip_drive(samples, t)
                _step(robot, scene, arm_idx, grip_idx, last_q_arm, grip_drive=current_grip_drive)
        else:
            for t in range(n):
                q_arm = resolve_tick_arm_q(ctx, seg, samples, t)
                last_q_arm = q_arm
                _step(robot, scene, arm_idx, grip_idx, q_arm, grip_drive=current_grip_drive)

        # End-side error vs the segment target (both in world frame).
        if seg.kind == "movel" and seg.pose_end is not None:
            target_world = np.asarray(ctx.base_to_world(seg.pose_end), dtype=np.float64)
            link6_w = ik_link.get_pos()[0]
            final_world = np.array([link6_w[i].item() for i in range(3)])
            status.eside_arm_mm = float(np.linalg.norm(final_world - target_world) * 1000.0)
        # Snapshot obj + link6 (world, m -> mm) for diagnostics.
        op = ctx.obj.get_pos()[0]
        status.obj_pos_mm = (float(op[0].item()) * 1000.0, float(op[1].item()) * 1000.0, float(op[2].item()) * 1000.0)
        lp = ik_link.get_pos()[0]
        status.link6_mm = (float(lp[0].item()) * 1000.0, float(lp[1].item()) * 1000.0, float(lp[2].item()) * 1000.0)
        if on_phase is not None:
            on_phase(status, seg)
        report.phases.append(status)

    # Final metrics: place error + home drift.
    obj_pos = ctx.obj.get_pos()[0]
    obj_half_z = ctx.obj_size[2] / 2
    place_target_world = np.array(ctx.base_to_world([ctx.place_xy[0], ctx.place_xy[1], obj_half_z]))
    report.place_error_mm = float(torch.norm(obj_pos - torch.as_tensor(place_target_world, device=gs.device, dtype=gs.tc_float)).item() * 1000.0)
    final_link6 = np.array([ik_link.get_pos()[0][i].item() for i in range(3)])
    home_world = np.array(ctx.base_to_world(ctx.home_pos_base))
    report.home_drift_mm = float(np.linalg.norm(final_link6 - home_world) * 1000.0)
    return report


def _step(robot, scene, arm_idx, grip_idx, q_arm_target, grip_drive):
    """Send one PD control tick and step the scene."""
    if q_arm_target is not None:
        q_t = torch.as_tensor([q_arm_target.tolist()], device=gs.device, dtype=gs.tc_float)
        robot.control_dofs_position(q_t, arm_idx)
    g_t = torch.as_tensor([[float(grip_drive)]], device=gs.device, dtype=gs.tc_float)
    robot.control_dofs_position(g_t, grip_idx)
    scene.step()
