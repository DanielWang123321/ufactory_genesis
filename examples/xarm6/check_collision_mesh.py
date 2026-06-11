"""
Visualize xarm6_with_gripper collision meshes at neutral pose (qpos0).
Helps diagnose the self-collision warning from Genesis collider.

Usage:
    source ~/envs/py312/bin/activate
    python examples/xarm6/check_collision_mesh.py         # print info + viewer
    python examples/xarm6/check_collision_mesh.py --no-vis # print info only
"""

import argparse
import sys

import _bootstrap  # noqa: F401
import genesis as gs
from ufactory.paths import xarm6_urdf

parser = argparse.ArgumentParser()
parser.add_argument("--no-vis", action="store_true", help="Skip viewer, print info only")
args = parser.parse_args()

gs.init(backend=gs.cpu, logging_level="warning")

show = not args.no_vis
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0.8, -0.5, 0.7),
        camera_lookat=(0.0, 0.0, 0.3),
        camera_fov=40,
    ),
    show_viewer=show,
)

robot = scene.add_entity(
    gs.morphs.URDF(file=xarm6_urdf("xarm6_with_gripper.urdf")),
    surface=gs.surfaces.Default(vis_mode="collision"),
)

scene.build()

# Print geom → link mapping
all_geoms = scene.rigid_solver.geoms
print("\n=== Collision Geom -> Link Map ===")
print(f"{'Geom':>5} | {'Link':>30} | {'Type':>10} | Faces")
print("-" * 65)
for i, geom in enumerate(all_geoms):
    n_faces = geom.init_faces.shape[0]
    print(f"{i:5d} | {geom.link.name:>30} | {geom.type.name:>10} | {n_faces}")

# Identify warning pairs
print("\n=== Self-Collision Warning Pairs ===")
warning_pairs = [(8, 9), (9, 12), (10, 11), (11, 13)]
for a, b in warning_pairs:
    if a < len(all_geoms) and b < len(all_geoms):
        print(f"  ({a:2d}, {b:2d}): {all_geoms[a].link.name} <-> {all_geoms[b].link.name}")

sys.stdout.flush()

if show:
    print("\nViewer is open. Inspect gripper collision meshes for overlap.")
    print("Close the viewer window to exit.\n")
    sys.stdout.flush()
    while scene.viewer.is_alive():
        scene.visualizer.update(force=True)
else:
    print("\nDone. Use without --no-vis to open viewer.")
