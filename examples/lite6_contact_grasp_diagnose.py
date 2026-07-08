#!/usr/bin/env python3
"""Diagnose Lite6 reversed gripper contact against the default 30 mm cube.

Fixed Lite6 fixture: the arm moves to the trajectory grasp pose once, then stays
fixed while only the parallel gripper closes. Reports GLB pad side clearance,
Genesis bilateral finger/object contact, and object displacement for candidate
grasp gaps.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = PROJECT_ROOT / "examples"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from _grasp_place_traj import _build_program  # noqa: E402
from ufactory.grippers.lite6 import measure_lite6_glb_side_clearance_m  # noqa: E402
from ufactory.trajectory.mirror_executor import resolve_tick_arm_q, resolve_tick_grip_drive  # noqa: E402
from ufactory.trajectory.scene import build_scene, drive_for_gap_m  # noqa: E402
from ufactory.trajectory.sim_executor import (  # noqa: E402
    LITE6_GRIPPER_CONTACT_HOLD_BIAS_GAP_M,
    _biased_gripper_hold_drive,
    _has_bilateral_finger_object_contact,
    _step,
)


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


def _contact_summary(ctx) -> tuple[set[int], np.ndarray, int, float]:
    contacts = ctx.obj.get_contacts(with_entity=ctx.robot, exclude_self_contact=True)
    link_a = _to_numpy(contacts.get("link_a", np.zeros(0))).reshape(-1)
    link_b = _to_numpy(contacts.get("link_b", np.zeros(0))).reshape(-1)
    if link_a.size == 0 or link_b.size == 0:
        return set(), np.zeros(3), 0, 0.0
    valid = _to_numpy(contacts.get("valid_mask", np.ones(link_a.shape, dtype=bool))).astype(bool).reshape(-1)
    if valid.size != link_a.size:
        valid = np.ones(link_a.shape, dtype=bool)
    force_a = _force_array(contacts, "force_a", link_a.size)
    force_b = _force_array(contacts, "force_b", link_a.size)
    finger_idx = {int(ctx.left_finger.idx), int(ctx.right_finger.idx)}
    obj_idx = int(ctx.obj.links[0].idx)

    touched: set[int] = set()
    object_force = np.zeros(3)
    max_contact_norm = 0.0
    rows = 0
    for a_raw, b_raw, fa, fb, is_valid in zip(link_a, link_b, force_a, force_b, valid):
        if not bool(is_valid):
            continue
        a = int(a_raw)
        b = int(b_raw)
        if a not in finger_idx and b not in finger_idx:
            continue
        touched.add(a if a in finger_idx else b)
        if a == obj_idx:
            f_obj = fa
        elif b == obj_idx:
            f_obj = fb
        elif a in finger_idx:
            f_obj = -fa
        else:
            f_obj = -fb
        object_force += f_obj
        max_contact_norm = max(max_contact_norm, float(np.linalg.norm(f_obj)))
        rows += 1
    return touched, object_force, rows, max_contact_norm


def _object_finger_contact_local_z_mm(ctx) -> tuple[float, float, int]:
    contacts = ctx.obj.get_contacts(with_entity=ctx.robot, exclude_self_contact=True)
    link_a = _to_numpy(contacts.get("link_a", np.zeros(0))).reshape(-1)
    link_b = _to_numpy(contacts.get("link_b", np.zeros(0))).reshape(-1)
    position = _to_numpy(contacts.get("position", np.zeros((0, 3)))).reshape((-1, 3))
    if link_a.size == 0 or link_b.size == 0 or position.shape[0] == 0:
        return float("nan"), float("nan"), 0
    valid = _to_numpy(contacts.get("valid_mask", np.ones(link_a.shape, dtype=bool))).astype(bool).reshape(-1)
    if valid.size != link_a.size:
        valid = np.ones(link_a.shape, dtype=bool)

    finger_by_idx = {
        int(ctx.left_finger.idx): ctx.left_finger,
        int(ctx.right_finger.idx): ctx.right_finger,
    }
    obj_idx = int(ctx.obj.links[0].idx)
    z_values: list[float] = []
    for a_raw, b_raw, pos, is_valid in zip(link_a, link_b, position, valid):
        if not bool(is_valid):
            continue
        a = int(a_raw)
        b = int(b_raw)
        if obj_idx not in (a, b):
            continue
        finger = finger_by_idx.get(a) or finger_by_idx.get(b)
        if finger is None:
            continue
        local = _link_local_point(finger, pos)
        z_values.append(float(local[2]) * 1000.0)
    if not z_values:
        return float("nan"), float("nan"), 0
    return min(z_values), max(z_values), len(z_values)


def _link_local_point(link, point: np.ndarray) -> np.ndarray:
    pos = link.get_pos()[0].detach().cpu().numpy()
    quat = link.get_quat()[0].detach().cpu().numpy()
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    return (np.asarray(point, dtype=np.float64) - pos) @ rot


def _force_array(contacts: dict, name: str, rows: int) -> np.ndarray:
    force = _to_numpy(contacts.get(name, np.zeros((rows, 3), dtype=np.float64)))
    if force.size == 0:
        return np.zeros((rows, 3), dtype=np.float64)
    force = force.reshape((-1, force.shape[-1]))
    if force.shape[0] != rows:
        return np.zeros((rows, 3), dtype=np.float64)
    return force


def _summarize(gap_mm: float, phase: str, rows: list[tuple], spawn_z: float) -> None:
    pos = np.stack([r[0] for r in rows])
    gap_neg = np.asarray([r[8] for r in rows], dtype=np.float64)
    gap_pos = np.asarray([r[9] for r in rows], dtype=np.float64)
    bilateral = np.asarray([r[10] for r in rows], dtype=bool)
    contact_z_min = np.asarray([r[11] for r in rows], dtype=np.float64)
    contact_z_max = np.asarray([r[12] for r in rows], dtype=np.float64)
    max_norm = np.asarray([r[6] for r in rows], dtype=np.float64)
    drive = np.asarray([r[7] for r in rows], dtype=np.float64)
    dz_mm = (pos[:, 2] - spawn_z) * 1000.0
    if np.all(np.isnan(contact_z_min)) or np.all(np.isnan(contact_z_max)):
        contact_z_text = "[nan,nan]"
    else:
        contact_z_text = f"[{np.nanmin(contact_z_min):.2f},{np.nanmax(contact_z_max):.2f}]"
    print(
        f"gap={gap_mm:4.1f} {phase:5s} "
        f"gap_neg_mm={gap_neg.mean() * 1000.0:6.2f} gap_pos_mm={gap_pos.mean() * 1000.0:6.2f} "
        f"max_side_mm={max(gap_neg.mean(), gap_pos.mean()) * 1000.0:6.2f} "
        f"bilateral_pct={100.0 * bilateral.mean():5.1f} "
        f"contact_z_mm={contact_z_text} "
        f"obj_dz_mm={dz_mm.mean():6.2f} "
        f"max_contact_mean={max_norm.mean():.4f} "
        f"drive_end={drive[-1]:.4f} "
        f"touched_end={sorted(rows[-1][3])}"
    )


def _print_keyframes(gap_mm: float, phase: str, rows: list[tuple], spawn_z: float) -> None:
    indices = sorted(set([0, len(rows) // 2, len(rows) - 1]))
    for idx in indices:
        (
            pos,
            _fc,
            _quat,
            touched,
            force,
            contact_rows,
            _max_norm,
            drive,
            gap_neg,
            gap_pos,
            bilateral,
            local_z_min,
            local_z_max,
            object_finger_rows,
        ) = rows[idx]
        print(
            f"  key gap={gap_mm:4.1f} {phase:5s} k={idx:3d} "
            f"obj_mm={np.round(pos * 1000.0, 2).tolist()} "
            f"side_mm=[{gap_neg * 1000.0:.2f},{gap_pos * 1000.0:.2f}] "
            f"bilateral={bilateral} drive={drive:.4f} "
            f"contacts={sorted(touched)} rows={contact_rows} "
            f"object_finger_rows={object_finger_rows} "
            f"contact_z_mm=[{local_z_min:.2f},{local_z_max:.2f}] "
            f"force_n={np.round(force, 4).tolist()} "
            f"obj_dz_mm={(pos[2] - spawn_z) * 1000.0:.2f}"
        )


def run_gap(args: argparse.Namespace, gap_mm: float) -> None:
    ctx = build_scene(
        robot_key=args.robot,
        rate=args.rate,
        show_viewer=False,
        substeps=args.substeps,
        visual_model=args.visual_model,
    )
    spawn_z = float(ctx.obj.get_pos()[0][2].item())
    program_args = SimpleNamespace(
        rate=args.rate,
        speed_rad_s=args.speed_rad_s,
        mvacc_rad_s2=args.mvacc_rad_s2,
        z_min_mm=args.z_min_mm,
    )
    program = _build_program(
        args.robot,
        ctx,
        program_args,
        grip_open_m=ctx.gripper.open_gap_m,
        grip_close_m=gap_mm / 1000.0,
    )
    robot = ctx.robot
    scene = ctx.scene
    arm_idx = ctx.arm_dof_idx
    grip_idx = ctx.gripper_dof_idx
    last_q_arm = ctx.home_qpos[arm_idx].astype(np.float64).copy()
    grip_drive = drive_for_gap_m(ctx.gripper.open_gap_m, ctx.gripper)

    for seg in program.segments:
        samples, n_ticks = seg.samples(program.rate)
        if seg.label == "grip":
            break
        for tick in range(n_ticks):
            last_q_arm = resolve_tick_arm_q(ctx, seg, samples, tick)
            _step(robot, scene, arm_idx, grip_idx, last_q_arm, grip_drive=grip_drive)

    grip_seg = next(seg for seg in program.segments if seg.label == "grip")
    samples, n_ticks = grip_seg.samples(program.rate)
    final_target_drive = resolve_tick_grip_drive(samples, n_ticks - 1, ctx.gripper)
    contact_hold_drive: float | None = None
    close_rows: list[tuple] = []
    hold_rows: list[tuple] = []

    for tick in range(n_ticks):
        planned_drive = resolve_tick_grip_drive(samples, tick, ctx.gripper)
        grip_drive = contact_hold_drive if contact_hold_drive is not None else planned_drive
        _step(robot, scene, arm_idx, grip_idx, last_q_arm, grip_drive=grip_drive)
        if contact_hold_drive is None and _has_bilateral_finger_object_contact(ctx):
            actual_drive = float(robot.get_dofs_position(grip_idx).reshape(-1)[0].item())
            contact_hold_drive = _biased_gripper_hold_drive(
                actual_drive,
                final_target_drive,
                ctx.gripper,
                args.hold_bias_gap_mm / 1000.0,
            )
            grip_drive = contact_hold_drive
        close_rows.append(_sample(ctx, robot, grip_idx))

    gripper = ctx.gripper
    if contact_hold_drive is not None:
        hold_drive = contact_hold_drive
    else:
        hold_drive = _biased_gripper_hold_drive(
            close_rows[-1][7],
            grip_drive,
            gripper,
            args.hold_bias_gap_mm / 1000.0,
        )

    for _ in range(max(1, int(round(args.hold_s * args.rate)))):
        _step(robot, scene, arm_idx, grip_idx, last_q_arm, grip_drive=hold_drive)
        hold_rows.append(_sample(ctx, robot, grip_idx))

    _summarize(gap_mm, "close", close_rows, spawn_z)
    _summarize(gap_mm, "hold", hold_rows, spawn_z)
    if args.keyframes:
        _print_keyframes(gap_mm, "close", close_rows, spawn_z)
        _print_keyframes(gap_mm, "hold", hold_rows, spawn_z)

    import genesis as gs

    gs.destroy()


def _sample(ctx, robot, grip_idx) -> tuple:
    pos = ctx.obj.get_pos()[0].detach().cpu().numpy()
    quat = ctx.obj.get_quat()[0].detach().cpu().numpy()
    left = ctx.left_finger.get_pos()[0].detach().cpu().numpy()
    right = ctx.right_finger.get_pos()[0].detach().cpu().numpy()
    touched, force, rows, max_norm = _contact_summary(ctx)
    drive = float(robot.get_dofs_position(grip_idx).reshape(-1)[0].item())
    gap_neg, gap_pos = measure_lite6_glb_side_clearance_m(ctx)
    bilateral = _has_bilateral_finger_object_contact(ctx)
    local_z_min, local_z_max, object_finger_rows = _object_finger_contact_local_z_mm(ctx)
    return (
        pos,
        0.5 * (left + right),
        quat,
        touched,
        force,
        rows,
        max_norm,
        drive,
        gap_neg,
        gap_pos,
        bilateral,
        local_z_min,
        local_z_max,
        object_finger_rows,
    )


def _parse_gaps(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="lite6", help="Robot fixture; default lite6")
    parser.add_argument("--gaps-mm", default="38,30,28,24,22,20")
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--visual-model", choices=("glb", "stl"), default="glb")
    parser.add_argument("--speed-rad-s", type=float, default=0.35)
    parser.add_argument("--mvacc-rad-s2", type=float, default=2.0)
    parser.add_argument("--z-min-mm", type=float, default=0.0)
    parser.add_argument("--hold-s", type=float, default=2.0)
    parser.add_argument("--keyframes", action="store_true", help="Print start/mid/end keyframe metrics.")
    parser.add_argument(
        "--hold-bias-gap-mm",
        type=float,
        default=LITE6_GRIPPER_CONTACT_HOLD_BIAS_GAP_M * 1000.0,
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    for gap_mm in _parse_gaps(args.gaps_mm):
        run_gap(args, gap_mm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
