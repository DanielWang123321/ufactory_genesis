"""Shared Genesis loop for standalone gripper visual demos."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from itertools import count

import genesis as gs

from ufactory.glb_visual import enable_glb_pbr_surfaces, glb_view_surface

CAMERA_POS = (0.22, -0.28, 0.18)
CAMERA_LOOKAT = (0.0, 0.0, 0.14)
CAMERA_FOV = 35


def run_standalone_gripper_viewer(
    description: str,
    urdf_path: str,
    controller: Callable[[object], Callable[[int], None]],
    *,
    headless_steps: int = 600,
) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--headless", action="store_true", help="Run without viewer window")
    args = parser.parse_args()

    enable_glb_pbr_surfaces()
    gs.init(backend=gs.gpu, logging_level="error")
    scene = gs.Scene(
        show_viewer=not args.headless,
        sim_options=gs.options.SimOptions(dt=0.01),
        viewer_options=gs.options.ViewerOptions(camera_pos=CAMERA_POS, camera_lookat=CAMERA_LOOKAT, camera_fov=CAMERA_FOV),
    )
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    robot = scene.add_entity(gs.morphs.URDF(file=urdf_path, fixed=True), surface=glb_view_surface())
    scene.build()

    step_controller = controller(robot)
    for step in count():
        step_controller(step)
        scene.step()
        if args.headless and step + 1 >= headless_steps:
            break
