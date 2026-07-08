"""
xArm6 Grasp-Place — trajectory-planned pipeline.

A structured pick-and-place sequence with **time-parameterized trajectories**
(LSPB / trapezoidal) replayed identically in Genesis (sim) and on the real xArm
(MODE_SERVO), giving sim-to-real alignment by construction: the same absolute
target stream is sampled at the same rate on both sides.

Default (no ``--executor``) runs the **sim** grasp-place sequence and reports
``place_error_mm`` / ``home_drift_mm`` and per-segment duration/profile/ESE.

Real path (``--executor servo-cartesian``) default ``--dry-run`` prints a
per-segment/per-tick digest and never moves the arm. To move the real arm only:

    python examples/xarm6/xarm6_grasp_place_traj.py \
        --executor servo-cartesian --ip 192.168.1.65 --z-min-mm 0 --no-dry-run

Add ``--real-gripper`` only when the physical gripper is installed and ready.

Real path with Genesis kinematic mirror (same planned trajectory, lightweight viewer):

    python examples/xarm6/xarm6_grasp_place_traj.py \
        --executor servo-cartesian --visual --ip 192.168.1.65 --z-min-mm 0 --no-dry-run

Prerequisite for real motion: pass the existing FK/IK alignment gate
(``xarm6_reach_deploy.py --mode align ...``) first.

SDK simulation validation still connects to the controller, but first switches
``set_simulation_robot(True)`` and streams in simulation mode:

    python examples/xarm6/xarm6_grasp_place_traj.py \
        --executor servo-cartesian --sdk-sim-validate --ip 192.168.1.65 \
        --rate 50 --z-min-mm 0 \
        --sdk-sim-report-csv reports/servo_sim.csv

Usage (sim):
    conda activate py313
    python examples/xarm6/xarm6_grasp_place_traj.py --headless --rate 50
    python examples/xarm6/xarm6_grasp_place_traj.py --visual --rate 50
    python examples/xarm6/xarm6_grasp_place_traj.py --visual --visual-model stl

This is a thin per-robot wrapper: the scene/pipeline logic lives in the shared
``examples/_grasp_place_traj.py`` module (reused by xArm5/7, UF850, and Lite6).
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
from _grasp_place_traj import main

if __name__ == "__main__":
    raise SystemExit(main("xarm6", robot_label="xArm6"))
