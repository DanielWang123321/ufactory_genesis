"""
Standalone Gripper G2 movable visual demo (no robot arm).

Usage:
    export NUMBA_CACHE_DIR=~/.cache/numba
    python examples/gripper_g2/view_gripper_g2_movable.py
    python examples/gripper_g2/view_gripper_g2_movable.py --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import genesis as gs
from ufactory.glb_visual import enable_glb_pbr_surfaces, glb_view_surface
from ufactory.paths import gripper_g2_movable_visual_urdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _gripper_demo import (  # noqa: E402
    control_gripper_pose,
    gripper_demo_target,
    gripper_dof_indices,
    setup_gripper_pd,
)

CAMERA_POS = (0.22, -0.28, 0.18)
CAMERA_LOOKAT = (0.0, 0.0, 0.14)
CAMERA_FOV = 35


def main() -> None:
    parser = argparse.ArgumentParser(description="Gripper G2 open/close demo (standalone)")
    parser.add_argument("--headless", action="store_true", help="Run without viewer window")
    args = parser.parse_args()

    enable_glb_pbr_surfaces()
    gs.init(backend=gs.gpu, logging_level="error")

    scene = gs.Scene(
        show_viewer=not args.headless,
        sim_options=gs.options.SimOptions(dt=0.01),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=CAMERA_POS,
            camera_lookat=CAMERA_LOOKAT,
            camera_fov=CAMERA_FOV,
        ),
    )
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    robot = scene.add_entity(
        gs.morphs.URDF(file=gripper_g2_movable_visual_urdf(), fixed=True),
        surface=glb_view_surface(),
    )
    scene.build()

    drive_idx, all_idx = gripper_dof_indices(robot)
    setup_gripper_pd(robot, drive_idx, all_idx)

    step = 0
    while True:
        q = gripper_demo_target(step)
        control_gripper_pose(robot, drive_idx, all_idx, q)
        scene.step()
        step += 1
        if args.headless and step >= 600:
            break


if __name__ == "__main__":
    main()
