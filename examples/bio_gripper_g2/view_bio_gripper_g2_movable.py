"""
Standalone Bio Gripper G2 movable visual demo (no robot arm).

Usage:
    export NUMBA_CACHE_DIR=~/.cache/numba
    python examples/bio_gripper_g2/view_bio_gripper_g2_movable.py
    python examples/bio_gripper_g2/view_bio_gripper_g2_movable.py --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLES_DIR))

import _bootstrap  # noqa: F401

import genesis as gs
from ufactory.bio_gripper_g2 import BioGripperG2
from ufactory.glb_visual import enable_glb_pbr_surfaces, glb_view_surface
from ufactory.paths import bio_gripper_g2_movable_visual_urdf

CAMERA_POS = (0.22, -0.28, 0.18)
CAMERA_LOOKAT = (0.0, 0.0, 0.14)
CAMERA_FOV = 35


def main() -> None:
    parser = argparse.ArgumentParser(description="Bio Gripper G2 open/close demo (standalone)")
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
        gs.morphs.URDF(file=bio_gripper_g2_movable_visual_urdf(), fixed=True),
        surface=glb_view_surface(),
    )
    scene.build()

    # Reusable controller: the same BioGripperG2 class drives the gripper whether it is
    # standalone (this demo) or mounted on an xArm / UF850 combo URDF.  Joint zero is the
    # closed pose (71 mm gap); the dof opens symmetrically to OPEN_POS (150 mm gap).
    gripper = BioGripperG2(robot)
    gripper.setup_pd()

    step = 0
    while True:
        gripper.control_pose(gripper.demo_target(step))
        scene.step()
        step += 1
        if args.headless and step >= 600:
            break


if __name__ == "__main__":
    main()
