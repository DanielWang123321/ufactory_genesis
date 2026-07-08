#!/usr/bin/env python3
"""Isolate Lite6 gripper/cube contact without the arm trajectory.

The scene contains only the standalone Lite6 gripper, a 30 mm cube, and a
plane.  It compares Genesis's default processed collision proxy with the raw
STL collision surface used by the trajectory scene fix.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = PROJECT_ROOT / "examples"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from _lite6_gripper_demo import lite6_gripper_dof_indices, setup_lite6_gripper_pd  # noqa: E402
from ufactory.grippers.lite6 import (  # noqa: E402
    LITE6_GRIPPER_SIM_OPEN_DRIVE,
    measure_lite6_glb_side_clearance_m,
)
from ufactory.robots.paths import lite6_gripper_movable_visual_urdf  # noqa: E402
from ufactory.robots.runtime import LITE6_GRIPPER_PARAMS  # noqa: E402
from ufactory.trajectory.scene import (  # noqa: E402
    FINGER_Z_OFFSET_LITE6,
    LITE6_FINGER_CLOSE_DESCENT,
    LITE6_FINGER_PAD_BELOW_FC,
    LITE6_GRASP_LINK6_Z_EXTRA_M,
    LITE6_GRASP_TABLE_CLEARANCE,
    LITE6_OBJ_FRICTION,
    LITE6_OBJ_INERTIAL_MASS_KG,
    OBJ_SIZE,
    RIGID_CONSTRAINT_TIMECONST,
    RIGID_NOSLIP_ITERATIONS,
    RIGID_SOLVER_ITERATIONS,
    drive_for_gap_m,
)
from ufactory.trajectory.sim_executor import (  # noqa: E402
    LITE6_GRIPPER_CONTACT_HOLD_BIAS_GAP_M,
    _biased_gripper_hold_drive,
)

GRIPPER_WORLD_JOINT_Z_M = 0.120


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


def _link_local_point(link, point: np.ndarray) -> np.ndarray:
    pos = link.get_pos()[0].detach().cpu().numpy()
    quat = link.get_quat()[0].detach().cpu().numpy()
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    return (np.asarray(point, dtype=np.float64) - pos) @ rot


def _contact_metrics(ctx) -> dict:
    contacts = ctx.obj.get_contacts(with_entity=ctx.robot, exclude_self_contact=True)
    link_a = _to_numpy(contacts.get("link_a", np.zeros(0))).reshape(-1)
    link_b = _to_numpy(contacts.get("link_b", np.zeros(0))).reshape(-1)
    rows_total = int(link_a.size)
    if rows_total == 0:
        return {
            "touched": set(),
            "rows": 0,
            "force": np.zeros(3),
            "world_z": (float("nan"), float("nan")),
            "local_z": (float("nan"), float("nan")),
        }
    valid = _to_numpy(contacts.get("valid_mask", np.ones(link_a.shape, dtype=bool))).astype(bool).reshape(-1)
    if valid.size != link_a.size:
        valid = np.ones(link_a.shape, dtype=bool)
    position = _to_numpy(contacts.get("position", np.zeros((rows_total, 3)))).reshape((-1, 3))
    force_a = _force_array(contacts, "force_a", rows_total)
    force_b = _force_array(contacts, "force_b", rows_total)

    finger_by_idx = {
        int(ctx.left_finger.idx): ctx.left_finger,
        int(ctx.right_finger.idx): ctx.right_finger,
    }
    finger_idx = set(finger_by_idx)
    obj_idx = int(ctx.obj.links[0].idx)
    touched: set[int] = set()
    force = np.zeros(3)
    world_z: list[float] = []
    local_z: list[float] = []
    rows = 0
    for a_raw, b_raw, pos, fa, fb, is_valid in zip(link_a, link_b, position, force_a, force_b, valid):
        if not bool(is_valid):
            continue
        a = int(a_raw)
        b = int(b_raw)
        if obj_idx not in (a, b):
            continue
        finger = finger_by_idx.get(a) or finger_by_idx.get(b)
        if finger is None:
            continue
        rows += 1
        touched.add(a if a in finger_idx else b)
        if a == obj_idx:
            force += fa
        elif b == obj_idx:
            force += fb
        elif a in finger_idx:
            force -= fa
        else:
            force -= fb
        world_z.append(float(pos[2]) * 1000.0)
        local_z.append(float(_link_local_point(finger, pos)[2]) * 1000.0)

    return {
        "touched": touched,
        "rows": rows,
        "force": force,
        "world_z": (min(world_z), max(world_z)) if world_z else (float("nan"), float("nan")),
        "local_z": (min(local_z), max(local_z)) if local_z else (float("nan"), float("nan")),
    }


def _force_array(contacts: dict, name: str, rows: int) -> np.ndarray:
    force = _to_numpy(contacts.get(name, np.zeros((rows, 3), dtype=np.float64)))
    if force.size == 0:
        return np.zeros((rows, 3), dtype=np.float64)
    force = force.reshape((-1, force.shape[-1]))
    if force.shape[0] != rows:
        return np.zeros((rows, 3), dtype=np.float64)
    return force


def _has_bilateral(ctx) -> bool:
    metrics = _contact_metrics(ctx)
    expected = {int(ctx.left_finger.idx), int(ctx.right_finger.idx)}
    return expected.issubset(metrics["touched"])


def _print_sample(mode: str, phase: str, ctx, robot, drive_idx: list[int]) -> None:
    gap_neg, gap_pos = measure_lite6_glb_side_clearance_m(ctx)
    metrics = _contact_metrics(ctx)
    obj = ctx.obj.get_pos()[0].detach().cpu().numpy()
    drive = float(robot.get_dofs_position(drive_idx).reshape(-1)[0].item())
    touched = sorted(int(v) for v in metrics["touched"])
    print(
        f"mode={mode:9s} phase={phase:8s} "
        f"drive={drive:.5f} side_mm=[{gap_neg * 1000.0:.2f},{gap_pos * 1000.0:.2f}] "
        f"bilateral={_has_bilateral(ctx)} rows={metrics['rows']} touched={touched} "
        f"contact_world_z_mm=[{metrics['world_z'][0]:.2f},{metrics['world_z'][1]:.2f}] "
        f"contact_local_z_mm=[{metrics['local_z'][0]:.2f},{metrics['local_z'][1]:.2f}] "
        f"force_n={np.round(metrics['force'], 5).tolist()} "
        f"obj_mm={np.round(obj * 1000.0, 3).tolist()}"
    )


def _build_scene(args: argparse.Namespace, mode: str):
    import genesis as gs

    dt = 1.0 / float(args.rate)
    substep_dt = dt / int(args.substeps)
    constraint_timeconst = max(float(RIGID_CONSTRAINT_TIMECONST), 2.0 * substep_dt)
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=args.substeps),
        rigid_options=gs.options.RigidOptions(
            dt=dt,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True,
            iterations=RIGID_SOLVER_ITERATIONS,
            noslip_iterations=RIGID_NOSLIP_ITERATIONS,
            constraint_timeconst=float(constraint_timeconst),
        ),
        show_viewer=args.visual,
    )
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    obj_size = tuple(float(v) for v in args.obj_size_m)
    obj = scene.add_entity(gs.morphs.Box(size=obj_size, pos=(0.0, 0.0, obj_size[2] / 2.0), fixed=False))

    finger_center_z = (
        LITE6_GRASP_TABLE_CLEARANCE
        + LITE6_FINGER_CLOSE_DESCENT
        + LITE6_FINGER_PAD_BELOW_FC
        + LITE6_GRASP_LINK6_Z_EXTRA_M
    )
    root_z = finger_center_z + GRIPPER_WORLD_JOINT_Z_M + FINGER_Z_OFFSET_LITE6
    morph_kwargs = {}
    if mode == "raw":
        morph_kwargs.update(convexify=False, decimate=False, watertighten=None)
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=lite6_gripper_movable_visual_urdf(),
            fixed=True,
            pos=(0.0, 0.0, root_z),
            # Genesis morph Euler angles are degrees; 180 deg points the gripper down.
            euler=(180.0, 0.0, 0.0),
            **morph_kwargs,
        )
    )
    scene.build(n_envs=1)

    drive_idx, all_idx = lite6_gripper_dof_indices(robot)
    setup_lite6_gripper_pd(robot, drive_idx, all_idx)
    open_drive = float(LITE6_GRIPPER_SIM_OPEN_DRIVE)
    robot.set_dofs_position(np.full(len(all_idx), open_drive), all_idx, zero_velocity=True)
    robot.control_dofs_position(np.full(len(drive_idx), open_drive), drive_idx)
    obj.set_friction(float(args.obj_friction))
    obj.set_links_inertial_mass(torch.tensor([args.obj_mass_kg], device=gs.device, dtype=gs.tc_float))
    left_finger = robot.get_link("uflite_finger1")
    right_finger = robot.get_link("uflite_finger2")
    left_finger.set_friction(float(args.finger_friction))
    right_finger.set_friction(float(args.finger_friction))
    ctx = SimpleNamespace(
        scene=scene,
        robot=robot,
        obj=obj,
        left_finger=left_finger,
        right_finger=right_finger,
        obj_size=obj_size,
    )
    return ctx, robot, drive_idx


def run_mode(args: argparse.Namespace, mode: str) -> None:
    import genesis as gs

    ctx, robot, drive_idx = _build_scene(args, mode)
    for _ in range(max(1, int(round(args.settle_s * args.rate)))):
        ctx.scene.step()
    _print_sample(mode, "open", ctx, robot, drive_idx)

    target_drive = drive_for_gap_m(args.gap_mm / 1000.0, LITE6_GRIPPER_PARAMS)
    hold_drive: float | None = None
    close_ticks = max(1, int(round(args.close_s * args.rate)))
    for tick in range(close_ticks):
        alpha = (tick + 1) / close_ticks
        planned_drive = LITE6_GRIPPER_SIM_OPEN_DRIVE + alpha * (target_drive - LITE6_GRIPPER_SIM_OPEN_DRIVE)
        drive = hold_drive if hold_drive is not None else planned_drive
        robot.control_dofs_position(np.full(len(drive_idx), drive), drive_idx)
        ctx.scene.step()
        if mode == "raw" and hold_drive is None and _has_bilateral(ctx):
            actual_drive = float(robot.get_dofs_position(drive_idx).reshape(-1)[0].item())
            hold_drive = _biased_gripper_hold_drive(
                actual_drive,
                target_drive,
                LITE6_GRIPPER_PARAMS,
                args.hold_bias_gap_mm / 1000.0,
            )
    _print_sample(mode, "close", ctx, robot, drive_idx)

    if hold_drive is None:
        hold_drive = target_drive
    for _ in range(max(1, int(round(args.hold_s * args.rate)))):
        robot.control_dofs_position(np.full(len(drive_idx), hold_drive), drive_idx)
        ctx.scene.step()
    _print_sample(mode, "hold", ctx, robot, drive_idx)

    if args.visual:
        for _ in range(max(1, int(round(args.visual_hold_s * args.rate)))):
            ctx.scene.step()
    gs.destroy()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collision-mode", choices=("raw", "processed", "both"), default="both")
    parser.add_argument("--gap-mm", type=float, default=20.0)
    parser.add_argument("--hold-bias-gap-mm", type=float, default=LITE6_GRIPPER_CONTACT_HOLD_BIAS_GAP_M * 1000.0)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--settle-s", type=float, default=0.4)
    parser.add_argument("--close-s", type=float, default=2.0)
    parser.add_argument("--hold-s", type=float, default=1.0)
    parser.add_argument("--visual-hold-s", type=float, default=3.0)
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--obj-size-m", type=float, nargs=3, default=OBJ_SIZE)
    parser.add_argument("--obj-mass-kg", type=float, default=LITE6_OBJ_INERTIAL_MASS_KG)
    parser.add_argument("--obj-friction", type=float, default=LITE6_OBJ_FRICTION)
    parser.add_argument("--finger-friction", type=float, default=LITE6_OBJ_FRICTION)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    modes = ("processed", "raw") if args.collision_mode == "both" else (args.collision_mode,)
    for mode in modes:
        run_mode(args, mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
