# examples/

[中文](README_cn.md)

Example scripts and tutorials.

## Quick Start

| File | Description |
|------|-------------|
| `view_robot_glb.py` | **Unified GLB viewer** — supports all robot profiles and end effectors |

```bash
python examples/view_robot_glb.py --robot xarm6
```

## Robot Visualization

Each robot directory has its own GLB preview wrapper:

| Directory | Preview command |
|-----------|-----------------|
| `xarm5/` | `python examples/xarm5/view_xarm5_glb.py` |
| `xarm6/` | `python examples/xarm6/view_xarm6_glb.py` |
| `xarm7/` | `python examples/xarm7/view_xarm7_glb.py` |
| `lite6/` | `python examples/lite6/view_lite6_glb.py` |
| `uf850/` | `python examples/uf850/view_uf850_glb.py` |

## Gripper Demos

| Directory/File | Description |
|----------------|-------------|
| `gripper_g2/` | Movable Gripper G2 visual demo |
| `bio_gripper_g2/` | Movable Bio Gripper G2 visual demo |
| `lite6_gripper/` | Movable Lite6 parallel gripper visual demo |

## Robot Validation

| File | Description |
|------|-------------|
| `verify_robot.py` | Generic FK/PD smoke test for all supported robots |
| `fk_verify_robot.py` | Generic FK validation, including optional real-robot comparison |
| `ik_verify_robot.py` | Generic IK validation, including optional real-robot comparison |
| `packaging_showcase.py` | Generic showcase entry point; currently supports xArm6 + Gripper G2 |

Dynamics validation is exposed through installed console scripts:

```bash
dynamics-sim-check --robot xarm6 --random-count 5
dynamics-hardware-check --robot xarm6 --ip <ip>
```

## xArm 6 — Reference Implementation

xArm 6 has the broadest example coverage in this repository.

### Kinematics and Dynamics

| File | Description |
|------|-------------|
| `xarm6/verify_xarm6.py` | FK + PD smoke test |
| `xarm6/verify_xarm6_dynamics.py` | Dynamics validation |
| `xarm6/fk_verify.py` | Compatibility wrapper, equivalent to generic FK validation with default `--robot xarm6` |
| `xarm6/ik_verify.py` | Compatibility wrapper, equivalent to generic IK validation with default `--robot xarm6` |

### Reinforcement Learning

| File | Description |
|------|-------------|
| `xarm6/xarm6_reach_env.py` / `_train.py` | Reach task environment and training |
| `xarm6/xarm6_reach_deploy.py` | Reach real-robot deployment: align / smoke / replay / deploy |
| `xarm6/xarm6_grasp_place_env.py` / `_train.py` / `_eval.py` | Grasp-place task |

### Trajectory Planning

| File | Description |
|------|-------------|
| `xarm5/xarm5_grasp_place_traj.py` | Shared waypoint/LSPB pick-and-place wrapper for xArm5 + Gripper G2 scene; sim and dry-run only |
| `xarm6/xarm6_grasp_place_traj.py` | Shared waypoint/LSPB pick-and-place wrapper for xArm6 + Gripper G2 scene |
| `xarm7/xarm7_grasp_place_traj.py` | Shared waypoint/LSPB pick-and-place wrapper for xArm7 + Gripper G2 scene |
| `uf850/uf850_grasp_place_traj.py` | Shared waypoint/LSPB pick-and-place wrapper for UF850 + Gripper G2 scene |
| `lite6/lite6_grasp_place_traj.py` | Shared waypoint/LSPB pick-and-place wrapper for Lite6 reversed gripper scene |

All five wrappers call the shared `examples/_grasp_place_traj.py` pipeline. It builds the mixed waypoint program with `TrajectoryPlannerConfig`, `CartesianWaypoint`, and `plan_mixed_waypoints`, then samples the motion with LSPB profiles. Segment labels are stable: `home->pregrasp`, `descend`, `grip`, `lift`, `transit`, `place-descend`, `release`, `retreat`, `return-home`.

When `--executor` is omitted, the script runs Genesis sim:

```bash
# Headless simulation; prints place_error / home_drift at the end.
python examples/xarm6/xarm6_grasp_place_traj.py --headless --rate 50

# Genesis viewer with GLB visuals and STL collision.
python examples/xarm6/xarm6_grasp_place_traj.py --visual --rate 50

# Legacy STL visuals for geometry alignment checks.
python examples/xarm6/xarm6_grasp_place_traj.py --visual --visual-model stl --rate 50
```

The grasp-place real path is enabled with `--executor servo-cartesian`. It defaults to dry-run mode, which prints segment and per-tick safety summaries and does not move the real robot:

```bash
python examples/xarm6/xarm6_grasp_place_traj.py --executor servo-cartesian --dry-run --rate 50 --z-min-mm 0
```

Real MODE_SERVO execution streams Cartesian targets to the arm. Gripper commands are off by default; add `--real-gripper` only after the physical gripper is installed and ready. Run the FK/IK alignment gate first and confirm that the workspace is safe:

```bash
# Arm-only safety validation.
python examples/xarm6/xarm6_grasp_place_traj.py \
  --executor servo-cartesian --ip 192.168.1.xx --z-min-mm 0 --no-dry-run

# Arm + physical gripper.
python examples/xarm6/xarm6_grasp_place_traj.py \
  --executor servo-cartesian --ip 192.168.1.xx --z-min-mm 0 --no-dry-run --real-gripper
```

`--visual` means different things on each path. In sim, it opens the full Genesis viewer. On the real path, it opens a kinematic mirror of the same planned trajectory so you can inspect the plan and grasp state; it is not contact physics:

```bash
# Real dry-run + kinematic mirror.
python examples/xarm6/xarm6_grasp_place_traj.py \
  --executor servo-cartesian --visual --dry-run --rate 50 --z-min-mm 0

# Real MODE_SERVO + kinematic mirror.
python examples/xarm6/xarm6_grasp_place_traj.py \
  --executor servo-cartesian --visual --ip 192.168.1.xx --z-min-mm 0 --no-dry-run
```

To connect to the controller without moving the physical arm, use SDK simulation validation. It calls `set_simulation_robot(True)` and can write a per-tick report:

```bash
python examples/xarm6/xarm6_grasp_place_traj.py \
  --executor servo-cartesian --sdk-sim-validate --ip 192.168.1.xx \
  --rate 50 --z-min-mm 0 \
  --sdk-sim-report-csv reports/servo_sim.csv
```

### Showcase

| File | Description |
|------|-------------|
| `xarm6/xarm6_g2_showcase.py` | xArm6 + Gripper G2 physical packaging demo |

## Internal Modules

Files with the `_` prefix are shared internal modules used by multiple examples:

| File | Purpose |
|------|---------|
| `_bootstrap.py` | Adds the project root to `sys.path` |
| `_robot_viewer.py` | Shared Genesis GLB viewer core |
| `_gripper_demo.py` | Gripper G2 open/close control |
| `_bio_gripper_g2_demo.py` | Bio Gripper G2 open/close control |
| `_lite6_gripper_demo.py` | Lite6 gripper open/close control |
| `_packaging_scene.py` | Showcase scene builder |
