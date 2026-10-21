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

CAMERA_POS = (0.22, -0.28, 0.18)
CAMERA_LOOKAT = (0.0, 0.0, 0.14)
CAMERA_FOV = 35


def run_standalone_gripper_viewer(
    description: str,
    urdf_path: str,
    controller: Callable[[object], Callable[[int], None]],
    *,
    headless_steps: int = 600,
    calibrated_g2: bool = False,
) -> None:
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
    args = parser.parse_args()
    if calibrated_g2:
        from ufactory.simulation.g2_model import checked

        for name, low, high in (("position_mm", 0, 84), ("speed_percent", 0, 100), ("force_percent", 1, 100)):
            checked(getattr(args, name), low, high, name)

    require_genesis_runtime(gs)
    gs.init(backend=genesis_backend_constant(gs, args.backend), logging_level="error")
    scene = gs.Scene(
        show_viewer=not args.headless,
        sim_options=gs.options.SimOptions(dt=0.02, substeps=32) if calibrated_g2 else gs.options.SimOptions(dt=0.01),
        rigid_options=make_rigid_options(gs, constraint_timeconst=0.00125) if calibrated_g2 else make_rigid_options(gs),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=CAMERA_POS, camera_lookat=CAMERA_LOOKAT, camera_fov=CAMERA_FOV
        ),
    )
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    with glb_pbr_surfaces():
        robot = scene.add_entity(gs.morphs.URDF(file=urdf_path, fixed=True), surface=glb_view_surface())
    configure_lite6_collision_filters(robot)
    scene.build()
    if calibrated_g2:
        robot._ufactory_g2_demo_settings = (args.position_mm, args.speed_percent, args.force_percent)

    step_controller = controller(robot)
    gaps = []
    for step in count():
        step_controller(step)
        scene.step()
        if calibrated_g2:
            gaps.append(robot._ufactory_g2_controller.gap_mm())
        if args.headless and step + 1 >= headless_steps:
            break
    if calibrated_g2:
        return {"min_gap_mm": min(gaps), "max_gap_mm": max(gaps), "final_gap_mm": gaps[-1], "ticks": len(gaps)}
