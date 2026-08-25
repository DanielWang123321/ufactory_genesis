"""Simulation replay executor for the trajectory pipeline.

Replays a :class:`ufactory.trajectory.segments.Program` in Genesis: each tick
sets absolute joint targets via PD position control (arm), while MoveL segments
solve link6 IK per tick and Gripper segments map the gap (m) to a drive value.
Reports per-segment duration/profile and final place/home error.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import time

import numpy as np
import torch

import genesis as gs

from ufactory.simulation import G2ContactHoldController, object_finger_contact_forces_n
from ufactory.trajectory.mirror_executor import gap_m_from_drive, resolve_tick_arm_q, resolve_tick_grip_drive
from ufactory.trajectory.scene import TrajSceneContext, drive_for_gap_m
from ufactory.trajectory.segments import Program, Segment

# 6.9 mm: matches the prior Gripper-G2-only tuning (0.07 drive-radians at the
# G2's 0.85-radian/84mm drive-to-gap ratio), expressed in physical gap units
# and normalized by each gripper's effective open-close travel.
DEFAULT_GRIPPER_HOLD_BIAS_GAP_M = 0.0069
# Lite6: small flat pad + unified cube μ=1.0 / pad μ=1.2 need a slightly larger
# post-contact hold bias than the prior 0.8 mm (tuned for μ=5.0) so friction
# can carry the 17 g cube through lift without raising material friction again.
LITE6_GRIPPER_CONTACT_HOLD_BIAS_GAP_M = 0.0020
DEFAULT_GRASP_WELD_ATTACH_DIST_M = 0.08
# Final metrics are observations, not extra trajectory commands.  Two physics
# ticks let the last already-issued PD target take effect before measuring,
# avoiding a one-tick tracking-lag artifact at the return-home boundary.
DEFAULT_FINAL_METRIC_SETTLE_S = 0.04


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
    ik_duration_ms: float = 0.0
    ik_failures: int = 0
    cpu_gpu_syncs: int = 0


@dataclass
class SimReport:
    phases: list[PhaseStatus] = field(default_factory=list)
    place_error_mm: float = 0.0
    home_drift_mm: float = 0.0
    total_ticks: int = 0
    total_duration: float = 0.0
    metric_settle_ticks: int = 0


def replay_sim(
    program: Program,
    ctx: TrajSceneContext,
    *,
    gripper_holds_after_grasp: bool = True,
    gripper_hold_bias_gap_m: float = DEFAULT_GRIPPER_HOLD_BIAS_GAP_M,
    stabilize_grasp_weld: bool = False,
    grasp_weld_attach_dist_m: float = DEFAULT_GRASP_WELD_ATTACH_DIST_M,
    final_metric_settle_s: float = DEFAULT_FINAL_METRIC_SETTLE_S,
    on_phase: Callable[[PhaseStatus, Segment], None] | None = None,
    on_tick: Callable[[Segment, int], None] | None = None,
) -> SimReport:
    """Replay ``program`` in the Genesis scene and return a SimReport.

    MoveJ segments stream arm joint targets directly; MoveL segments solve
    link6 IK per tick (base->world). Gripper segments stream the drive value.
    During the close segment, the planned target may intentionally be smaller
    than the object width to create contact. Once the close finishes, subsequent
    arm segments hold the actually reached drive value plus a small bias instead
    of continuing to chase the over-closed target. This keeps a clamp load while
    avoiding persistent finger penetration into the cube.  Final metrics are
    sampled after a short physics-only settling window while the final approved
    targets remain latched; this does not append or alter trajectory commands.
    ``on_tick`` runs after each commanded physics step and is intended for
    diagnostics that need to observe intra-segment ordering.
    """
    robot = ctx.robot
    scene = ctx.scene
    ik_link = ctx.ik_link
    arm_idx = ctx.arm_dof_idx
    grip_idx = ctx.gripper_dof_idx
    grip_control_idx = (
        ctx.all_gripper_dof_idx if getattr(ctx, "control_all_gripper_dofs", False) else ctx.gripper_dof_idx
    )
    control_all_on_open = bool(getattr(ctx, "control_all_gripper_dofs_on_open", False))
    configured_open_idx = list(getattr(ctx, "gripper_open_dof_idx", ()))
    if configured_open_idx:
        opening_grip_control_idx = configured_open_idx
    elif control_all_on_open:
        opening_grip_control_idx = ctx.all_gripper_dof_idx
    else:
        opening_grip_control_idx = grip_control_idx
    initial_opening_grip_control_idx = list(getattr(ctx, "gripper_initial_open_dof_idx", ()))
    initial_opening_clearance_m = float(getattr(ctx, "gripper_initial_open_clearance_m", 0.0))
    binary_gripper_commands = bool(getattr(ctx, "gripper_binary_commands", False))
    all_opening_grip_control_idx = list(dict.fromkeys([*initial_opening_grip_control_idx, *opening_grip_control_idx]))
    gripper_buffer_width = max(len(grip_control_idx), len(all_opening_grip_control_idx))
    opening_only_grip_idx = [idx for idx in all_opening_grip_control_idx if idx not in grip_control_idx]
    rate = float(program.rate)
    if not np.isfinite(final_metric_settle_s) or final_metric_settle_s < 0.0:
        raise ValueError("final_metric_settle_s must be finite and non-negative")
    report = SimReport(total_duration=program.total_duration, total_ticks=program.total_ticks)

    gripper = ctx.gripper
    current_grip_drive = drive_for_gap_m(gripper.open_gap_m, gripper)
    g2_contact_hold = (
        G2ContactHoldController(
            1,
            device=gs.device,
            dtype=gs.tc_float,
            initial_gap_m=float(gripper.open_gap_m),
        )
        if getattr(gripper, "family", "") == "g2"
        else None
    )
    last_q_arm = ctx.home_qpos[arm_idx].astype(np.float64).copy() if len(ctx.home_qpos) else None
    grasp_welded = False
    buffers = _TensorBuffers(
        arm=torch.empty((1, len(arm_idx)), device=gs.device, dtype=gs.tc_float),
        gripper=torch.empty((1, gripper_buffer_width), device=gs.device, dtype=gs.tc_float),
        arm_cpu=torch.empty((1, len(arm_idx)), device="cpu", dtype=gs.tc_float),
        gripper_cpu=torch.empty((1, gripper_buffer_width), device="cpu", dtype=gs.tc_float),
    )
    buffers.arm_numpy = buffers.arm_cpu.numpy()
    buffers.gripper_numpy = buffers.gripper_cpu.numpy()

    for si, seg in enumerate(program.segments):
        samples, n = seg.samples(rate)
        if n == 0:
            continue
        status = PhaseStatus(
            label=seg.label or seg.kind,
            seg_index=si,
            ticks=n,
            duration=seg.duration,
            v_max=seg.v_max,
            a_max=seg.a_max,
        )
        if seg.kind == "gripper":
            is_holding_gap = (
                seg.gap_start is not None
                and seg.gap_end is not None
                and abs(float(seg.gap_end) - float(seg.gap_start)) < 1e-12
            )
            is_closing = (
                gripper_holds_after_grasp
                and seg.gap_start is not None
                and seg.gap_end is not None
                and seg.gap_end < seg.gap_start
            )
            is_opening = seg.gap_start is not None and seg.gap_end is not None and seg.gap_end > seg.gap_start
            is_holding_fully_open = (
                is_holding_gap
                and seg.gap_end is not None
                and abs(float(seg.gap_end) - float(gripper.open_gap_m)) < 1e-12
            )
            if is_holding_fully_open and initial_opening_grip_control_idx:
                segment_grip_control_idx = initial_opening_grip_control_idx
            else:
                segment_grip_control_idx = opening_grip_control_idx if is_opening else grip_control_idx
            if is_opening and grasp_welded:
                _delete_grasp_weld(ctx)
                grasp_welded = False
            contact_hold_drive: float | None = None
            final_target_drive = resolve_tick_grip_drive(samples, n - 1, gripper)
            opening_start_targets: dict[int, float] | None = None
            opening_start_finger_span_m: float | None = None
            nominal_opening_start_drive: float | None = None
            if is_opening:
                opening_positions = (
                    robot.get_dofs_position(all_opening_grip_control_idx)[0].detach().cpu().numpy().astype(np.float64)
                )
                opening_start_targets = dict(zip(all_opening_grip_control_idx, opening_positions, strict=True))
                if initial_opening_grip_control_idx and initial_opening_clearance_m > 0.0:
                    opening_start_finger_span_m = _finger_span_m(ctx)
                nominal_opening_start_drive = resolve_tick_grip_drive(samples, 0, gripper)
            for t in range(n):
                planned_grip_drive = resolve_tick_grip_drive(samples, t, gripper)
                grip_targets: np.ndarray | None = None
                if is_opening:
                    assert opening_start_targets is not None
                    assert nominal_opening_start_drive is not None
                    denominator = final_target_drive - nominal_opening_start_drive
                    alpha = (
                        1.0
                        if binary_gripper_commands or abs(denominator) < 1e-12
                        else (planned_grip_drive - nominal_opening_start_drive) / denominator
                    )
                    alpha = float(np.clip(alpha, 0.0, 1.0))
                    if (
                        opening_start_finger_span_m is not None
                        and _finger_span_m(ctx) < opening_start_finger_span_m + initial_opening_clearance_m
                    ):
                        segment_grip_control_idx = initial_opening_grip_control_idx
                    else:
                        segment_grip_control_idx = opening_grip_control_idx
                    grip_targets = np.asarray(
                        [
                            opening_start_targets[idx] + alpha * (final_target_drive - opening_start_targets[idx])
                            for idx in segment_grip_control_idx
                        ],
                        dtype=np.float64,
                    )
                    drive_start = opening_start_targets.get(grip_idx[0], planned_grip_drive)
                    current_grip_drive = float(drive_start + alpha * (final_target_drive - drive_start))
                elif not is_holding_gap:
                    current_grip_drive = contact_hold_drive if contact_hold_drive is not None else planned_grip_drive
                _step(
                    robot,
                    scene,
                    arm_idx,
                    segment_grip_control_idx,
                    last_q_arm,
                    grip_drive=current_grip_drive,
                    grip_targets=grip_targets,
                    buffers=buffers,
                )
                _assert_finite_replay_state(ctx, phase=status.label, tick=t)
                if g2_contact_hold is not None:
                    requested_gap_m = float(samples[t, 0])
                    current_grip_drive = _advance_g2_contact_hold(
                        g2_contact_hold,
                        ctx,
                        requested_gap_m=requested_gap_m,
                        closing=(
                            gripper_holds_after_grasp
                            and not is_opening
                            and requested_gap_m < float(gripper.open_gap_m) - 1e-9
                        ),
                        release=is_opening,
                    )
                    if bool(g2_contact_hold.latched[0].item()):
                        contact_hold_drive = current_grip_drive
                if on_tick is not None:
                    on_tick(seg, t)
                if (
                    is_closing
                    and _latches_contact_during_close(gripper)
                    and contact_hold_drive is None
                    and _has_bilateral_finger_object_contact(ctx)
                ):
                    contact_hold_drive = _biased_gripper_hold_drive(
                        _current_drive(robot, grip_idx),
                        final_target_drive,
                        gripper,
                        gripper_hold_bias_gap_m,
                    )
                    current_grip_drive = contact_hold_drive
            if is_closing:
                # Work in a "closedness" fraction (0=open, 1=closed) rather
                # than the raw drive-DOF: this is monotonic in the same
                # direction for both Gripper G2 (open=0.0 < close=0.85) and
                # Lite6 (close=0.0 < open=0.0089), so the "hold a bit past
                # actual contact, but never past the originally planned
                # closedness" logic generalizes unchanged across grippers.
                if contact_hold_drive is not None:
                    current_grip_drive = contact_hold_drive
                else:
                    current_grip_drive = _biased_gripper_hold_drive(
                        _current_drive(robot, grip_idx),
                        current_grip_drive,
                        gripper,
                        gripper_hold_bias_gap_m,
                    )
                if stabilize_grasp_weld and _should_weld_grasp(ctx, float(grasp_weld_attach_dist_m)):
                    _add_grasp_weld(ctx)
                    grasp_welded = True
        else:
            for t in range(n):
                ik_started = time.perf_counter_ns()
                try:
                    q_arm = resolve_tick_arm_q(ctx, seg, samples, t)
                except Exception:
                    status.ik_failures += 1
                    raise
                finally:
                    if seg.kind == "movel":
                        status.ik_duration_ms += (time.perf_counter_ns() - ik_started) / 1e6
                last_q_arm = q_arm
                _step(
                    robot,
                    scene,
                    arm_idx,
                    grip_control_idx,
                    q_arm,
                    grip_drive=current_grip_drive,
                    buffers=buffers,
                )
                _assert_finite_replay_state(ctx, phase=status.label, tick=t)
                if g2_contact_hold is not None:
                    requested_gap_m = gap_m_from_drive(current_grip_drive, gripper)
                    current_grip_drive = _advance_g2_contact_hold(
                        g2_contact_hold,
                        ctx,
                        requested_gap_m=requested_gap_m,
                        closing=requested_gap_m < float(gripper.open_gap_m) - 1e-9,
                        release=False,
                    )
                if on_tick is not None:
                    on_tick(seg, t)

        # End-side error vs the segment target (both in world frame).
        if seg.kind == "movel" and seg.pose_end is not None:
            target_world = np.asarray(ctx.base_to_world(seg.pose_end), dtype=np.float64)
            link6_w = ik_link.get_pos()[0]
            final_world = np.array([link6_w[i].item() for i in range(3)])
            status.cpu_gpu_syncs += 3
            status.eside_arm_mm = float(np.linalg.norm(final_world - target_world) * 1000.0)
        # Snapshot obj + link6 (world, m -> mm) for diagnostics.
        op = ctx.obj.get_pos()[0]
        status.obj_pos_mm = (float(op[0].item()) * 1000.0, float(op[1].item()) * 1000.0, float(op[2].item()) * 1000.0)
        lp = ik_link.get_pos()[0]
        status.link6_mm = (float(lp[0].item()) * 1000.0, float(lp[1].item()) * 1000.0, float(lp[2].item()) * 1000.0)
        status.cpu_gpu_syncs += 6
        if on_phase is not None:
            on_phase(status, seg)
        report.phases.append(status)

    # Observe after the final already-issued target has had a brief chance to
    # settle.  Bare steps retain Genesis' latched arm/gripper control targets.
    report.metric_settle_ticks = int(np.ceil(float(final_metric_settle_s) * rate))
    for settle_tick in range(report.metric_settle_ticks):
        scene.step()
        _assert_finite_replay_state(ctx, phase="final-settle", tick=settle_tick)

    # Final metrics: place error + home drift, both against the complete
    # three-dimensional positions consumed when the scene was constructed.
    obj_pos = ctx.obj.get_pos()[0]
    place_target_world = np.asarray(ctx.base_to_world(ctx.place_pos_base), dtype=np.float64)
    place_target_t = torch.as_tensor(place_target_world, device=obj_pos.device, dtype=obj_pos.dtype)
    report.place_error_mm = float(torch.norm(obj_pos - place_target_t).item() * 1000.0)
    final_link6 = np.array([ik_link.get_pos()[0][i].item() for i in range(3)])
    home_world = np.array(ctx.base_to_world(ctx.home_pos_base))
    report.home_drift_mm = float(np.linalg.norm(final_link6 - home_world) * 1000.0)
    if opening_only_grip_idx:
        # Opening may temporarily put extra mimic DOFs into position-control
        # mode.  Genesis latches that mode across replay calls, so a later
        # close that intentionally drives only the master joint would otherwise
        # fight the stale open target.  Return those release-only DOFs to
        # passive zero-force mode after all metrics have been sampled.
        robot.control_dofs_force(
            torch.zeros((1, len(opening_only_grip_idx)), device=gs.device, dtype=gs.tc_float),
            opening_only_grip_idx,
        )
    return report


@dataclass
class _TensorBuffers:
    arm: torch.Tensor
    gripper: torch.Tensor
    arm_cpu: torch.Tensor
    gripper_cpu: torch.Tensor
    arm_numpy: np.ndarray | None = None
    gripper_numpy: np.ndarray | None = None


def _advance_g2_contact_hold(
    controller: G2ContactHoldController,
    ctx: TrajSceneContext,
    *,
    requested_gap_m: float,
    closing: bool,
    release: bool,
) -> float:
    """Observe the completed physics tick and resolve the next G2 drive target."""

    left_force, right_force = _object_finger_contact_forces_n(ctx)
    measured_drive = _current_drive(ctx.robot, ctx.gripper_dof_idx)
    measured_gap = gap_m_from_drive(measured_drive, ctx.gripper)
    next_gap = controller.update(
        requested_gap_m=torch.tensor([requested_gap_m], device=gs.device, dtype=gs.tc_float),
        measured_gap_m=torch.tensor([measured_gap], device=gs.device, dtype=gs.tc_float),
        left_force_n=left_force,
        right_force_n=right_force,
        closing=torch.tensor([closing], device=gs.device, dtype=torch.bool),
        release=torch.tensor([release], device=gs.device, dtype=torch.bool),
    )
    return drive_for_gap_m(float(next_gap[0].item()), ctx.gripper)


def _object_finger_contact_forces_n(ctx: TrajSceneContext) -> tuple[torch.Tensor, torch.Tensor]:
    """Return object-only contact loads for the two physical finger links."""

    if ctx.left_finger is None or ctx.right_finger is None or not callable(getattr(ctx.obj, "get_contacts", None)):
        zero = torch.zeros(1, device=gs.device, dtype=gs.tc_float)
        return zero, zero.clone()
    contacts = ctx.obj.get_contacts(with_entity=ctx.robot, exclude_self_contact=True)
    left, right = object_finger_contact_forces_n(
        contacts,
        left_link_idx=int(ctx.left_finger.idx),
        right_link_idx=int(ctx.right_finger.idx),
    )
    return (
        left.to(device=gs.device, dtype=gs.tc_float),
        right.to(device=gs.device, dtype=gs.tc_float),
    )


def _assert_finite_replay_state(ctx: TrajSceneContext, *, phase: str, tick: int) -> None:
    """Fail at the first corrupted trajectory state with phase-local context."""

    all_gripper_dof_idx = list(getattr(ctx, "all_gripper_dof_idx", ctx.gripper_dof_idx))
    dof_idx = list(dict.fromkeys([*ctx.arm_dof_idx, *all_gripper_dof_idx]))
    fields: dict[str, object] = {
        "robot_qpos": ctx.robot.get_dofs_position(dof_idx),
        "object_pos": ctx.obj.get_pos(),
    }
    get_velocity = getattr(ctx.robot, "get_dofs_velocity", None)
    if callable(get_velocity):
        fields["robot_qvel"] = get_velocity(dof_idx)
    get_obj_velocity = getattr(ctx.obj, "get_vel", None)
    if callable(get_obj_velocity):
        fields["object_vel"] = get_obj_velocity()
    for name, value in fields.items():
        tensor = torch.as_tensor(value)
        if not bool(torch.isfinite(tensor).all().item()):
            raise RuntimeError(f"G2 physical replay produced NaN or Inf in {name} at {phase} tick {tick}")


def _step(
    robot,
    scene,
    arm_idx,
    grip_control_idx,
    q_arm_target,
    grip_drive,
    *,
    buffers: _TensorBuffers,
    grip_targets: np.ndarray | None = None,
):
    """Send one PD control tick and step the scene."""
    if q_arm_target is not None:
        assert buffers.arm_numpy is not None
        np.copyto(buffers.arm_numpy[0], np.asarray(q_arm_target, dtype=buffers.arm_numpy.dtype))
        buffers.arm.copy_(buffers.arm_cpu, non_blocking=True)
        robot.control_dofs_position(buffers.arm, arm_idx)
    if grip_targets is None:
        buffers.gripper.fill_(float(grip_drive))
    else:
        assert buffers.gripper_numpy is not None
        width = len(grip_control_idx)
        np.copyto(buffers.gripper_numpy[0, :width], np.asarray(grip_targets, dtype=buffers.gripper_numpy.dtype))
        buffers.gripper.copy_(buffers.gripper_cpu, non_blocking=True)
    robot.control_dofs_position(buffers.gripper[:, : len(grip_control_idx)], grip_control_idx)
    scene.step()


def _current_drive(robot, grip_idx) -> float:
    q = robot.get_dofs_position(grip_idx)
    return float(q.reshape(-1)[0].item())


def _latches_contact_during_close(gripper) -> bool:
    """Return true for grippers whose sim close should stop at first bilateral contact."""
    return getattr(gripper, "family", "") == "lite6"


def _biased_gripper_hold_drive(actual_drive: float, target_drive: float, gripper, bias_gap_m: float) -> float:
    actual_closedness = _closedness_fraction(actual_drive, gripper)
    target_closedness = _closedness_fraction(target_drive, gripper)
    gap_span = float(gripper.open_gap_m) - float(getattr(gripper, "closed_gap_m", 0.0))
    bias_closedness = float(bias_gap_m) / max(gap_span, 1e-9)
    hold_closedness = min(target_closedness, actual_closedness + bias_closedness)
    return gripper.open_pos + hold_closedness * (gripper.close_pos - gripper.open_pos)


def _closedness_fraction(drive: float, gripper) -> float:
    """Map a drive-DOF value to a closedness fraction in [0, 1] (0=open, 1=closed).

    Sign-convention-agnostic: works whether ``open_pos < close_pos`` (Gripper
    G2) or ``close_pos < open_pos`` (Lite6).
    """
    span = gripper.close_pos - gripper.open_pos
    if span == 0:
        return 0.0
    frac = (float(drive) - gripper.open_pos) / span
    return max(0.0, min(1.0, frac))


def _finger_span_m(ctx: TrajSceneContext) -> float:
    """Return the dominant horizontal distance between the two finger links."""
    left = ctx.left_finger.get_pos()[0]
    right = ctx.right_finger.get_pos()[0]
    return max(abs(float(left[0] - right[0])), abs(float(left[1] - right[1])))


def _should_weld_grasp(ctx: TrajSceneContext, attach_dist_m: float) -> bool:
    """Debug contact-conditioned weld: allow only real bilateral finger/object contact.

    ``attach_dist_m`` is kept for API compatibility with older callers; default
    sim no longer uses distance-based attachment.
    """
    required = ("left_finger", "right_finger", "obj")
    if any(getattr(ctx, name, None) is None for name in required):
        return False
    _ = attach_dist_m
    return _has_bilateral_finger_object_contact(ctx)


def _has_bilateral_finger_object_contact(ctx: TrajSceneContext, *, min_force_n: float = 1e-4) -> bool:
    """Return true when the object is physically contacted by both gripper fingers."""
    required = ("left_finger", "right_finger", "obj", "robot")
    if any(getattr(ctx, name, None) is None for name in required):
        return False
    finger_idx = {int(ctx.left_finger.idx), int(ctx.right_finger.idx)}
    contact_fingers = _object_finger_contact_links(ctx, finger_idx, min_force_n=min_force_n)
    return finger_idx.issubset(contact_fingers)


def _object_finger_contact_links(
    ctx: TrajSceneContext,
    finger_idx: set[int],
    *,
    min_force_n: float,
) -> set[int]:
    try:
        contacts = ctx.obj.get_contacts(with_entity=ctx.robot, exclude_self_contact=True)
    except Exception:
        return set()
    link_a = _contact_to_numpy(contacts.get("link_a", np.zeros(0)))
    link_b = _contact_to_numpy(contacts.get("link_b", np.zeros(0)))
    if link_a.size == 0 or link_b.size == 0:
        return set()
    valid = _contact_to_numpy(contacts.get("valid_mask", np.ones(link_a.shape, dtype=bool))).astype(bool).reshape(-1)
    link_a = link_a.reshape(-1)
    link_b = link_b.reshape(-1)
    if valid.size != link_a.size:
        valid = np.ones(link_a.shape, dtype=bool)
    force_a = _contact_force_array(contacts, "force_a", link_a.size)
    force_b = _contact_force_array(contacts, "force_b", link_a.size)

    out: set[int] = set()
    for a_raw, b_raw, force_vec_a, force_vec_b, is_valid in zip(link_a, link_b, force_a, force_b, valid):
        if not bool(is_valid):
            continue
        a = int(a_raw)
        b = int(b_raw)
        touched = a if a in finger_idx else b if b in finger_idx else None
        if touched is None:
            continue
        force_norm = max(
            float(np.linalg.norm(np.asarray(force_vec_a, dtype=np.float64))),
            float(np.linalg.norm(np.asarray(force_vec_b, dtype=np.float64))),
        )
        if force_norm < float(min_force_n):
            continue
        out.add(touched)
    return out


def _contact_force_array(contacts: dict, name: str, rows: int) -> np.ndarray:
    force = _contact_to_numpy(contacts.get(name, np.zeros((rows, 3), dtype=np.float64)))
    if force.size == 0:
        return np.zeros((rows, 3), dtype=np.float64)
    force = force.reshape((-1, force.shape[-1]))
    if force.shape[0] != rows:
        return np.zeros((rows, 3), dtype=np.float64)
    return force


def _contact_to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return np.asarray(value)


def _add_grasp_weld(ctx: TrajSceneContext) -> None:
    solver = getattr(ctx.robot, "_solver", None)
    obj_links = getattr(ctx.obj, "links", None)
    if solver is None or not obj_links:
        return
    solver.add_weld_constraint(ctx.ik_link.idx, obj_links[0].idx)


def _delete_grasp_weld(ctx: TrajSceneContext) -> None:
    solver = getattr(ctx.robot, "_solver", None)
    obj_links = getattr(ctx.obj, "links", None)
    if solver is None or not obj_links:
        return
    solver.delete_weld_constraint(ctx.ik_link.idx, obj_links[0].idx)
