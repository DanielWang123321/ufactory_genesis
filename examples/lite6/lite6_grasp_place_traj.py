"""
Lite6 Grasp-Place — trajectory-planned pipeline.

Same grasp-place sequence as the other arms, but scaled for Lite6's smaller
440 mm reach and its reversed Lite6 parallel gripper (20-38 mm physical
opening, vs. Gripper G2's 84 mm): the default object is a 30 mm cube and the
workspace coordinates are pulled in closer to the base (see
``ufactory.trajectory.scene``'s ``LITE6_*`` defaults). This is a thin
per-robot CLI wrapper around the shared ``examples/_grasp_place_traj.py``
module.

Real-hardware note: the xArm Python SDK exposes the Lite6 gripper only as two
digital-IO commands (``open_lite6_gripper`` / ``close_lite6_gripper``), not a
continuous position API. The real path here fires whichever endpoint the
planned gap is moving toward at the start of each gripper segment, then paces
the rest of the segment -- there is no closed-loop position feedback on real
hardware for this gripper.

Usage (sim):
    conda activate py313
    python examples/lite6/lite6_grasp_place_traj.py --headless --rate 50
    python examples/lite6/lite6_grasp_place_traj.py --visual --rate 50

Real path (dry-run digest only by default):
    python examples/lite6/lite6_grasp_place_traj.py \
        --executor servo-cartesian --ip 192.168.1.170 --z-min-mm 0

To actually move the real arm, add ``--no-dry-run``. Add ``--real-gripper``
only when the physical gripper should open/close.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
from _grasp_place_traj import main

if __name__ == "__main__":
    raise SystemExit(main("lite6", robot_label="Lite6"))
