"""Kinematic Genesis mirror for the real-robot trajectory executor.

Replays the same segment samples as :mod:`sim_executor` using lightweight
teleport stepping (no PD tracking) so a Genesis viewer can stay in sync with
the xArm servo stream without adding measurable tick overrun.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import torch

from ufactory.kinematics.genesis import solve_link6_ik
from ufactory.trajectory.scene import TrajSceneContext, drive_for_gap_m
from ufactory.trajectory.segments import Program, Segment
from ufactory.grippers.lite6 import snap_object_to_lite6_glb_grasp


def gap_m_from_drive(drive: float, gripper) -> float:
    """Map a gripper drive-DOF value back to a physical two-finger gap (m)."""
    open_fraction = (float(drive) - gripper.close_pos) / (gripper.open_pos - gripper.close_pos)
    open_fraction = max(0.0, min(1.0, open_fraction))
    closed_gap = float(getattr(gripper, "closed_gap_m", 0.0))
    return closed_gap + open_fraction * (float(gripper.open_gap_m) - closed_gap)


def resolve_segment_start_arm_q(
    ctx: TrajSceneContext,
    seg: Segment,
) -> np.ndarray:
    """Arm joint vector at the start of an arm segment (matches sim/real preposition)."""
    if seg.kind == "movej":
        if seg.q_start is None:
            raise ValueError(f"MoveJ segment {seg.label!r} has no q_start")
        return np.asarray(seg.q_start, dtype=np.float64).reshape(-1)
    if seg.kind == "movel":
        if seg.pose_start is None:
            raise ValueError(f"MoveL segment {seg.label!r} has no pose_start")
        pose_world = ctx.base_to_world(seg.pose_start)
        return solve_link6_ik(
            ctx.robot,
            ctx.ik_link,
            pose_world,
            arm_dof_idx=ctx.arm_dof_idx,
            quat=ctx.down_quat,
        )
    raise ValueError(f"segment {seg.label!r} is not an arm segment")


def resolve_tick_arm_q(
    ctx: TrajSceneContext,
    seg: Segment,
    samples: np.ndarray,
    tick_idx: int,
) -> np.ndarray:
    """Resolve arm joint targets for one tick (same logic as sim replay)."""
    if seg.kind == "movej":
        return np.asarray(samples[tick_idx], dtype=np.float64)
    if seg.kind == "movel":
        pose_world = ctx.base_to_world(samples[tick_idx])
        return solve_link6_ik(
            ctx.robot,
            ctx.ik_link,
            pose_world,
            arm_dof_idx=ctx.arm_dof_idx,
            quat=ctx.down_quat,
        )
    raise ValueError(f"unexpected arm segment kind: {seg.kind}")


def resolve_tick_grip_drive(samples: np.ndarray, tick_idx: int, gripper) -> float:
    gap_m = float(samples[tick_idx, 0])
    return drive_for_gap_m(gap_m, gripper)


def _disable_pd(robot, dof_idx: list[int]) -> None:
    if not dof_idx:
        return
    n = len(dof_idx)
    zeros = np.zeros(n)
    robot.set_dofs_kp(zeros, dof_idx)
    robot.set_dofs_kv(zeros, dof_idx)
    robot.set_dofs_force_range(zeros, zeros, dof_idx)


def kinematic_step(ctx: TrajSceneContext, q_arm: np.ndarray, grip_drive: float) -> None:
    """Teleport arm/gripper DOFs and refresh the viewer without PD forces.

    The G2 gripper's non-driven joints (fingers/knuckles) are linked to
    ``drive_joint`` via Genesis equality (mimic) constraints that are only
    reliably enforced under real PD dynamics. In a pure kinematic teleport,
    every mimic DOF must be set explicitly (see ``examples/_gripper_demo.py``)
    or the un-driven joints are left to a soft constraint chasing a
    force-less target and diverge.
    """
    robot = ctx.robot
    scene = ctx.scene
    arm_idx = ctx.arm_dof_idx
    all_grip_idx = ctx.all_gripper_dof_idx

    robot.set_dofs_position(q_arm, arm_idx, zero_velocity=True)
    grip_target = np.full(len(all_grip_idx), float(grip_drive))
    robot.set_dofs_position(grip_target, all_grip_idx, zero_velocity=True)
    scene.step(update_visualizer=False)
    visualizer = getattr(scene, "visualizer", None)
    if getattr(visualizer, "viewer", None) is not None:
        visualizer.update(force=False)


class TrajKinematicMirror:
    """Open-loop kinematic Genesis mirror driven by the trajectory program."""

    def __init__(self, ctx: TrajSceneContext, program: Program) -> None:
        self.ctx = ctx
        self.program = program
        self.rate = float(program.rate)
        arm_idx = ctx.arm_dof_idx
        if len(ctx.home_qpos):
            self._last_q_arm = ctx.home_qpos[arm_idx].astype(np.float64).copy()
        else:
            self._last_q_arm = np.zeros(len(arm_idx), dtype=np.float64)
        self._grip_drive = drive_for_gap_m(ctx.gripper.open_gap_m, ctx.gripper)
        _disable_pd(ctx.robot, arm_idx)
        _disable_pd(ctx.robot, ctx.all_gripper_dof_idx)

    @property
    def last_q_arm(self) -> np.ndarray:
        return self._last_q_arm

    @property
    def grip_drive(self) -> float:
        return self._grip_drive

    def prime_to_home(self) -> None:
        arm_idx = self.ctx.arm_dof_idx
        if len(self.ctx.home_qpos):
            self._last_q_arm = self.ctx.home_qpos[arm_idx].astype(np.float64).copy()
        self._grip_drive = drive_for_gap_m(self.ctx.gripper.open_gap_m, self.ctx.gripper)
        kinematic_step(self.ctx, self._last_q_arm, self._grip_drive)

    def prime_to_segment_start(self, seg: Segment) -> None:
        if seg.kind not in ("movej", "movel"):
            raise ValueError(f"cannot prime arm pose from segment kind {seg.kind!r}")
        self._last_q_arm = resolve_segment_start_arm_q(self.ctx, seg)
        kinematic_step(self.ctx, self._last_q_arm, self._grip_drive)

    def prime_to_first_arm_segment_start(self, program: Program) -> None:
        for seg in program.segments:
            if seg.kind in ("movej", "movel"):
                self.prime_to_segment_start(seg)
                return

    def tick(self, seg: Segment, tick_idx: int) -> None:
        samples, n = seg.samples(self.rate)
        if tick_idx < 0 or tick_idx >= n:
            raise IndexError(f"tick {tick_idx} out of range for segment {seg.label!r} (N={n})")
        if seg.kind == "gripper":
            self._grip_drive = resolve_tick_grip_drive(samples, tick_idx, self.ctx.gripper)
            kinematic_step(self.ctx, self._last_q_arm, self._grip_drive)
            return
        self._last_q_arm = resolve_tick_arm_q(self.ctx, seg, samples, tick_idx)
        kinematic_step(self.ctx, self._last_q_arm, self._grip_drive)

    def on_tick(self, seg: Segment, tick_idx: int) -> None:
        """Callback compatible with :func:`replay_real` ``on_tick`` hook."""
        self.tick(seg, tick_idx)

    def hold_step(self) -> None:
        """Re-assert the last commanded pose for one tick (kinematic hold, no PD).

        Without this, letting the viewer idle via plain ``scene.step()`` calls
        after replay finishes drops the arm/gripper under gravity: PD gains are
        zeroed for the whole mirror (see :func:`_disable_pd`), so nothing but
        the per-tick teleport keeps the last pose in place.
        """
        kinematic_step(self.ctx, self._last_q_arm, self._grip_drive)

    def replay_with_pacing(self) -> None:
        """Play the full program at ``program.rate`` (dry-run visual preview)."""
        tick_s = 1.0 / self.rate
        next_deadline = time.monotonic()
        for seg in self.program.segments:
            _samples, n = seg.samples(self.rate)
            for t in range(n):
                self.tick(seg, t)
                next_deadline += tick_s
                sleep_s = next_deadline - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)


def make_on_tick(mirror: TrajKinematicMirror) -> Callable[[Segment, int], None]:
    return mirror.on_tick


class KinematicCarryTracker:
    """Kinematically "attaches" ``ctx.obj`` to the gripper once closed near it.

    Mirrors the ``carry_offset``/``carry_quat`` latch technique already proven
    in ``examples/xarm6/xarm6_g2_showcase.py``: once attached, the object's
    pose is forcibly slaved to the finger-center frame every tick (no contact
    physics), then handed back to the physics engine on release. This avoids
    coupling a kinematically-teleported gripper to real contact dynamics,
    which is the same class of instability fixed for the mimic joints
    themselves (see :func:`kinematic_step`).

    During the grasp-close segment (``grasp_segment_label``, default
    ``"grip"``), the object pose is frozen at the segment-start pose on every
    tick while the kinematic gripper closes, then latched on the **last tick**
    of that segment (matching ``xarm6_g2_showcase.py``). This prevents the
    hybrid "kinematic fingers + physics block" collision push that visibly
    slides the cube on the table before carry engages.

    Before the grasp latch, the spawn-table pose is restored on every tick of
    ``descend`` so coarse collision STLs cannot shove the cube into the table
    while the arm approaches. During ``release``, the object is snapped to the
    table and frozen for the whole open segment so spreading collision fingers
    cannot push it downward after carry detaches.

    ``grasp_gap_m`` still sets ``release_gap_m`` for the release segment; it is
    no longer used as a mid-close attach threshold on the grasp segment.
    """

    _APPROACH_FREEZE_LABELS = frozenset({"descend"})
    _PLACE_FREEZE_LABELS = frozenset({"place-standoff"})
    _RELEASE_LABEL = "release"

    def __init__(
        self,
        mirror: TrajKinematicMirror,
        *,
        grasp_gap_m: float,
        release_gap_m: float | None = None,
        attach_dist_m: float = 0.08,
        grasp_segment_label: str = "grip",
    ) -> None:
        self.mirror = mirror
        self.ctx = mirror.ctx
        self.grasp_gap_m = float(grasp_gap_m)
        # Release as soon as the gripper starts reopening: waiting until fully
        # open would keep re-latching the object to a spreading finger center,
        # which visibly jitters the carried object during the open motion.
        self.release_gap_m = float(release_gap_m) if release_gap_m is not None else self.grasp_gap_m + 0.003
        self.attach_dist_m = float(attach_dist_m)
        self.grasp_segment_label = str(grasp_segment_label)
        self._attached = False
        self._carry_offset: torch.Tensor | None = None
        self._carry_quat: torch.Tensor | None = None
        self._grip_freeze_pos: torch.Tensor | None = None
        self._grip_freeze_quat: torch.Tensor | None = None
        self._spawn_freeze_pos: torch.Tensor | None = None
        self._spawn_freeze_quat: torch.Tensor | None = None
        self._release_freeze_pos: torch.Tensor | None = None
        self._release_freeze_quat: torch.Tensor | None = None
        self._place_table_locked = False
        self._capture_spawn_freeze_pose()

    @property
    def attached(self) -> bool:
        return self._attached

    def _finger_center(self) -> torch.Tensor:
        lf = self.ctx.left_finger.get_pos()[0]
        rf = self.ctx.right_finger.get_pos()[0]
        return (lf + rf) / 2

    def _capture_spawn_freeze_pose(self) -> None:
        self._spawn_freeze_pos = self.ctx.obj.get_pos()[0].clone()
        self._spawn_freeze_quat = self.ctx.obj.get_quat()[0].clone()

    def _restore_spawn_frozen_obj(self) -> None:
        if self._spawn_freeze_pos is None or self._spawn_freeze_quat is None:
            return
        self.ctx.obj.set_pos(self._spawn_freeze_pos.unsqueeze(0), zero_velocity=True)
        self.ctx.obj.set_quat(self._spawn_freeze_quat.unsqueeze(0), zero_velocity=True)

    def _table_object_center_z(self) -> float:
        return float(self.ctx.base_pos_world[2] + self.ctx.obj_size[2] / 2)

    def _snap_object_to_table(self) -> None:
        pos = self.ctx.obj.get_pos()[0].clone()
        pos[2] = self._table_object_center_z()
        self.ctx.obj.set_pos(pos.unsqueeze(0), zero_velocity=True)
        self.ctx.obj.set_quat(self.ctx.obj.get_quat()[0].unsqueeze(0), zero_velocity=True)

    def _capture_release_freeze_pose(self) -> None:
        self._snap_object_to_table()
        self._release_freeze_pos = self.ctx.obj.get_pos()[0].clone()
        self._release_freeze_quat = self.ctx.obj.get_quat()[0].clone()

    def _restore_release_frozen_obj(self) -> None:
        if self._release_freeze_pos is None or self._release_freeze_quat is None:
            return
        self.ctx.obj.set_pos(self._release_freeze_pos.unsqueeze(0), zero_velocity=True)
        self.ctx.obj.set_quat(self._release_freeze_quat.unsqueeze(0), zero_velocity=True)

    def _clear_release_freeze_pose(self) -> None:
        self._release_freeze_pos = None
        self._release_freeze_quat = None

    def _is_release_segment(self, seg: Segment) -> bool:
        return seg.kind == "gripper" and seg.label == self._RELEASE_LABEL

    def _prepare_place_release(self) -> None:
        self._snap_object_to_table()
        self._attached = False
        self._carry_offset = None
        self._carry_quat = None
        self._capture_release_freeze_pose()

    def _is_grasp_close_segment(self, seg: Segment) -> bool:
        return seg.kind == "gripper" and seg.label == self.grasp_segment_label

    def _should_freeze_approach(self, seg: Segment) -> bool:
        if self._attached or seg.kind != "movel" or seg.label not in self._APPROACH_FREEZE_LABELS:
            return False
        if self._spawn_freeze_pos is None:
            return False
        obj_pos = self.ctx.obj.get_pos()[0]
        return float(torch.norm(obj_pos - self._spawn_freeze_pos)) <= self.attach_dist_m

    def _should_freeze_on_table(self, seg: Segment) -> bool:
        return (
            getattr(self.ctx.gripper, "family", None) == "lite6"
            and seg.kind == "movel"
            and seg.label in self._PLACE_FREEZE_LABELS
        )

    def _capture_grip_freeze_pose(self) -> None:
        self._grip_freeze_pos = self.ctx.obj.get_pos()[0].clone()
        self._grip_freeze_quat = self.ctx.obj.get_quat()[0].clone()

    def _restore_grip_frozen_obj(self) -> None:
        if self._grip_freeze_pos is None or self._grip_freeze_quat is None:
            return
        self.ctx.obj.set_pos(self._grip_freeze_pos.unsqueeze(0), zero_velocity=True)
        self.ctx.obj.set_quat(self._grip_freeze_quat.unsqueeze(0), zero_velocity=True)

    def _clear_grip_freeze_pose(self) -> None:
        self._grip_freeze_pos = None
        self._grip_freeze_quat = None

    def _latch_carry(self, fc: torch.Tensor) -> None:
        if getattr(self.ctx.gripper, "family", None) == "lite6":
            snap_object_to_lite6_glb_grasp(self.ctx)
        obj_pos = self.ctx.obj.get_pos()[0]
        self._carry_offset = obj_pos - fc
        self._carry_quat = self.ctx.obj.get_quat()[0].clone()
        self._attached = True

    def _maybe_attach(self, gap_m: float, fc: torch.Tensor) -> None:
        if self._place_table_locked or self._attached or gap_m > self.grasp_gap_m + 1e-3:
            return
        obj = self.ctx.obj
        obj_pos = obj.get_pos()[0]
        if float(torch.norm(obj_pos - fc)) > self.attach_dist_m:
            return
        self._latch_carry(fc)

    def _sync_carry(self, fc: torch.Tensor) -> None:
        if not self._attached:
            return
        self.ctx.obj.set_pos((fc + self._carry_offset).unsqueeze(0), zero_velocity=True)
        self.ctx.obj.set_quat(self._carry_quat.unsqueeze(0), zero_velocity=True)

    def _maybe_release(self, gap_m: float) -> None:
        if self._attached and gap_m >= self.release_gap_m - 1e-3:
            self._attached = False

    def on_tick(self, seg: Segment, tick_idx: int) -> None:
        """Callback compatible with :func:`replay_real` ``on_tick`` hook."""
        _, n = seg.samples(self.mirror.rate)
        is_grasp_close = self._is_grasp_close_segment(seg)
        is_release = self._is_release_segment(seg)

        if is_release and tick_idx == 0:
            self._prepare_place_release()

        if is_grasp_close and tick_idx == 0:
            self._capture_grip_freeze_pose()

        self.mirror.on_tick(seg, tick_idx)

        gap_m = gap_m_from_drive(self.mirror.grip_drive, self.ctx.gripper)
        fc = self._finger_center()

        if (
            getattr(self.ctx.gripper, "family", None) == "lite6"
            and seg.kind == "movel"
            and seg.label == "place-descend"
            and tick_idx == n - 1
        ):
            self._snap_object_to_table()
            self._attached = False
            self._carry_offset = None
            self._carry_quat = None
            self._place_table_locked = True

        if self._should_freeze_approach(seg):
            self._restore_spawn_frozen_obj()
        elif self._should_freeze_on_table(seg):
            self._snap_object_to_table()
        elif is_release:
            self._restore_release_frozen_obj()
        elif is_grasp_close and not self._attached:
            self._restore_grip_frozen_obj()
            if tick_idx == n - 1:
                obj_pos = self.ctx.obj.get_pos()[0]
                if float(torch.norm(obj_pos - fc)) <= self.attach_dist_m:
                    self._latch_carry(fc)
                self._clear_grip_freeze_pose()
        elif not is_grasp_close:
            self._maybe_attach(gap_m, fc)

        if self._attached:
            self._sync_carry(fc)

        if is_release and tick_idx == n - 1:
            self._clear_release_freeze_pose()
        elif not is_release:
            self._maybe_release(gap_m)

    def hold_sync(self) -> None:
        """Hold-phase counterpart to :meth:`on_tick`: keep a carried object glued in place.

        No gap-based state transitions here (the gripper isn't moving during
        hold); this only re-applies the current carry lock so an attached
        object does not fall once the replay loop stops ticking.
        """
        if self._attached:
            self._sync_carry(self._finger_center())

    def hold_step(self) -> None:
        """Composed hold callback: re-teleport the mirror pose and any carried object."""
        self.mirror.hold_step()
        self.hold_sync()
