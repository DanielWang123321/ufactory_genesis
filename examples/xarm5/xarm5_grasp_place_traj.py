"""
xArm5 Grasp-Place — trajectory-planned pipeline.

Same grasp-place sequence, gripper (Gripper G2), and workspace coordinates as
``xarm6/xarm6_grasp_place_traj.py`` (xArm5/6/7 share a 700 mm reach and G2
mount geometry); this is a thin per-robot CLI wrapper around the shared
``examples/_grasp_place_traj.py`` module.

Usage (sim):
    conda activate py313
    python examples/xarm5/xarm5_grasp_place_traj.py --headless --rate 50
    python examples/xarm5/xarm5_grasp_place_traj.py --visual --rate 50

Real path (dry-run digest only by default; physical arm has no end effector):
    python examples/xarm5/xarm5_grasp_place_traj.py \
        --executor servo_cartesian --dry-run --rate 50 --z-min-mm 0
    python examples/xarm5/xarm5_grasp_place_traj.py \
        --executor servo_j --dry-run --rate 50 --z-min-mm 0

To move the real arm, add ``--no-dry-run`` and ``--ip 192.168.1.xx``.
Do not add ``--real-gripper`` unless a physical gripper is installed.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
from _grasp_place_traj import main

if __name__ == "__main__":
    raise SystemExit(main("xarm5", robot_label="xArm5"))
