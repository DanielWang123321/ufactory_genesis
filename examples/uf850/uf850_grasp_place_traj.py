"""
UF850 Grasp-Place — trajectory-planned pipeline.

Same grasp-place sequence, gripper (Gripper G2), and workspace coordinates as
``xarm6/xarm6_grasp_place_traj.py`` (UF850's 850 mm reach comfortably covers
the same workspace; see ``dev/`` sim verification notes for the reachability
check); this is a thin per-robot CLI wrapper around the shared
``examples/_grasp_place_traj.py`` module.

Usage (sim):
    conda activate py313
    python examples/uf850/uf850_grasp_place_traj.py --headless --rate 50
    python examples/uf850/uf850_grasp_place_traj.py --visual --rate 50

Real path (dry-run digest only by default):
    python examples/uf850/uf850_grasp_place_traj.py \
        --executor servo-cartesian --ip 192.168.1.55 --z-min-mm 0

To actually move the real arm, add ``--no-dry-run``. Physical Gripper G2
commands are opt-in via ``--real-gripper`` after the accessory is installed.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
from _grasp_place_traj import main

if __name__ == "__main__":
    raise SystemExit(main("uf850", robot_label="UF850"))
