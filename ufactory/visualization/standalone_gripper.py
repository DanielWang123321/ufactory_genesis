"""Shared Genesis loop for standalone gripper visual demos."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from itertools import count

import genesis as gs

from ufactory.simulation import make_rigid_options
from ufactory.simulation.compat import require_genesis_runtime
from ufactory.simulation.collision import configure_lite6_collision_filters
from ufactory.visualization.glb import glb_pbr_surfaces, glb_view_surface
from ufactory.visualization.viewer import start_deferred_viewer

CAMERA_POS = (0.22, -0.28, 0.18)
CAMERA_LOOKAT = (0.0, 0.0, 0.14)
CAMERA_FOV = 35
# Same as the view_robot.py warm-up for calibrated grippers.
VIEWER_WARMUP_STEPS = 3


def _open_viewer(scene) -> None:
    """Warm up, then open the window through the deferred viewer.

    A window opened at scene construction shows black or undefined frames until the scene renders.
    The deferred viewer only presents complete frames, and the first shows the initial gripper pose.
    The warm-up steps leave the demo unchanged: the gripper holds the PD target set by its setup.
    """
    for _ in range(VIEWER_WARMUP_STEPS):
        scene.step()
    start_deferred_viewer(scene)


def run_standalone_gripper_viewer(
    description: str,
    urdf_path: str,
    controller: Callable[[object], Callable[[int], None]],
    *,
    headless_steps: int = 600,
    calibrated_g2: bool = False,
    calibrated_bio_g2: bool = False,
    base_pos: tuple[float, float, float] | None = None,
    base_quat: tuple[float, float, float, float] | None = None,
) -> dict | None:
    """Run a gripper demo, with the URDF root placed at ``base_pos`` / ``base_quat`` (w, x, y, z) when given."""
    from ufactory.simulation import genesis_backend_constant

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--headless", action="store_true", help="Run without viewer window")
    parser.add_argument(
        "--backend",
        choices=("cpu", "gpu"),
        default="gpu",
        help="Genesis backend (use cpu without a supported GPU)",
    )
    if calibrated_g2:
        parser.add_argument("--position-mm", type=float, default=0)
        parser.add_argument("--speed-percent", type=float, default=50)
        parser.add_argument("--force-percent", type=float, default=50)
    elif calibrated_bio_g2:
        parser.add_argument("--position-mm", type=float, default=71)
        parser.add_argument("--speed-percent", type=float, default=50)
        parser.add_argument("--speed-mm-s", type=float, default=None)
        parser.add_argument("--force-percent", type=float, default=50)
    args = parser.parse_args()
    if calibrated_g2:
        from ufactory.simulation.g2_model import checked

        for name, low, high in (("position_mm", 0, 84), ("speed_percent", 0, 100), ("force_percent", 1, 100)):
            checked(getattr(args, name), low, high, name)
    elif calibrated_bio_g2:
        from ufactory.simulation.bio_g2_model import MAX_SPEED_MM_S, MIN_SPEED_MM_S, checked

        for name, low, high in (("position_mm", 71, 150), ("speed_percent", 0, 100), ("force_percent", 1, 100)):
            checked(getattr(args, name), low, high, name)
        if args.speed_mm_s is not None:
            checked(args.speed_mm_s, MIN_SPEED_MM_S, MAX_SPEED_MM_S, "speed_mm_s")

    calibrated = calibrated_g2 or calibrated_bio_g2
    require_genesis_runtime(gs)
    gs.init(backend=genesis_backend_constant(gs, args.backend), logging_level="error")
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(dt=0.02, substeps=32) if calibrated else gs.options.SimOptions(dt=0.01),
        rigid_options=(
            make_rigid_options(gs, constraint_timeconst=0.00125 if calibrated_g2 else 0.002)
            if calibrated
            else make_rigid_options(gs)
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=CAMERA_POS, camera_lookat=CAMERA_LOOKAT, camera_fov=CAMERA_FOV
        ),
    )
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    pose = {}
    if base_pos is not None:
        pose["pos"] = base_pos
    if base_quat is not None:
        pose["quat"] = base_quat
    with glb_pbr_surfaces():
        robot = scene.add_entity(gs.morphs.URDF(file=urdf_path, fixed=True, **pose), surface=glb_view_surface())
    configure_lite6_collision_filters(robot)
    scene.build()
    if calibrated_g2:
        robot._ufactory_g2_demo_settings = (args.position_mm, args.speed_percent, args.force_percent)
    elif calibrated_bio_g2:
        robot._ufactory_bio_g2_demo_settings = (
            args.position_mm,
            args.speed_percent,
            args.force_percent,
            args.speed_mm_s,
        )

    step_controller = controller(robot)
    if not args.headless:
        _open_viewer(scene)
    gaps = []
    for step in count():
        step_controller(step)
        scene.step()
        if calibrated_g2:
            gaps.append(robot._ufactory_g2_controller.gap_mm())
        elif calibrated_bio_g2:
            gaps.append(robot._ufactory_bio_g2_controller.gap_mm())
        demo_cycle = (
            getattr(robot, "_ufactory_g2_demo_cycle", None)
            or getattr(robot, "_ufactory_bio_g2_demo_cycle", None)
            or getattr(robot, "_ufactory_lite6_demo_cycle", None)
        )
        if demo_cycle is not None:
            demo_cycle.on_step_completed(robot)
        if args.headless and (
            step + 1 >= headless_steps
            or (headless_steps == 600 and demo_cycle is not None and demo_cycle.completed_cycles >= 1)
        ):
            break
    if calibrated:
        return {"min_gap_mm": min(gaps), "max_gap_mm": max(gaps), "final_gap_mm": gaps[-1], "ticks": len(gaps)}
    return None
