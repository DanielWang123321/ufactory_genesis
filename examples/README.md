# examples/

[中文](README_cn.md)

Catalog of example scripts. For install, grasp-place modes, and real-robot safety, see the repository root [README.md](../README.md).

## Quick Start

| File | Description |
|------|-------------|
| `view_robot_glb.py` | Unified GLB viewer for all robot profiles and end effectors |

```bash
python examples/view_robot_glb.py --robot xarm6
```

## Robot Visualization

Per-robot wrappers are equivalent to `view_robot_glb.py --robot <key>`:

| Directory | Preview command |
|-----------|-----------------|
| `xarm5/` | `python examples/xarm5/view_xarm5_glb.py` |
| `xarm6/` | `python examples/xarm6/view_xarm6_glb.py` |
| `xarm7/` | `python examples/xarm7/view_xarm7_glb.py` |
| `lite6/` | `python examples/lite6/view_lite6_glb.py` |
| `uf850/` | `python examples/uf850/view_uf850_glb.py` |

## Gripper Demos

| Directory | Description |
|-----------|-------------|
| `gripper_g2/` | Movable Gripper G2 visual demo |
| `bio_gripper_g2/` | Movable Bio Gripper G2 visual demo |
| `lite6_gripper/` | Movable Lite6 parallel gripper visual demo |

## Robot Validation

| File | Description |
|------|-------------|
| `verify_robot.py` | Generic FK/PD smoke test for all supported robots |
| `fk_verify_robot.py` | Generic FK validation, including optional real-robot comparison |
| `ik_verify_robot.py` | Generic IK validation, including optional real-robot comparison |
| `packaging_showcase.py` | YAML-driven five-robot packaging entry |

Dynamics validation uses installed console scripts:

```bash
dynamics-sim-check --robot xarm6 --random-count 5
dynamics-hardware-check --robot xarm6 --ip <ip>
```

## Trajectory Grasp-Place

Thin wrappers around the installed `ufactory-grasp-place` CLI (`ufactory.cli.grasp_place`). Full mode/executor docs live in the root [README.md](../README.md#trajectory-grasp-place).

| File | Description |
|------|-------------|
| `xarm5/xarm5_grasp_place_traj.py` | Wrapper for `--robot xarm5` |
| `xarm6/xarm6_grasp_place_traj.py` | Wrapper for `--robot xarm6` |
| `xarm7/xarm7_grasp_place_traj.py` | Wrapper for `--robot xarm7` |
| `uf850/uf850_grasp_place_traj.py` | Wrapper for `--robot uf850` |
| `lite6/lite6_grasp_place_traj.py` | Wrapper for `--robot lite6` |

```bash
python examples/xarm6/xarm6_grasp_place_traj.py --mode dry-run --executor servo_j
# Equivalent:
# ufactory-grasp-place --robot xarm6 --mode dry-run --executor servo_j

# Real + kinematic mirror (see root README):
# ufactory-grasp-place --robot xarm6 --mode real --executor servo_j \
#   --calibration path/to/exact.yaml --confirm-real --visual
```

## xArm 6 — Reference Implementation

xArm 6 has the broadest example coverage.

### Kinematics

| File | Description |
|------|-------------|
| `xarm6/verify_xarm6.py` | FK + PD smoke test |
| `xarm6/fk_verify.py` | Compatibility wrapper; default `--robot xarm6` |
| `xarm6/ik_verify.py` | Compatibility wrapper; default `--robot xarm6` |

### Reinforcement Learning

| File | Description |
|------|-------------|
| `xarm6/xarm6_reach_env.py` / `_train.py` | Reach task environment and training |
| `xarm6/xarm6_reach_deploy.py` | Reach deploy helper: align / offline preflight remain available; **v0.2.5 hard-disables** online real-policy `deploy` and `smoke-random` modes (see [SECURITY.md](../SECURITY.md)) |
| `xarm6/xarm6_grasp_place_env.py` / `_train.py` / `_eval.py` | Grasp-place RL task |

### Packaging Showcase

| File | Description |
|------|-------------|
| `packaging_showcase.py` | Generic CLI wrapper for xArm5/6/7, UF850, and Lite6 |
| `_packaging_showcase.py` | Shared physical execution implementation (internal) |
| `xarm6/xarm6_g2_showcase.py` | Compatibility wrapper selecting `--robot xarm6` |

The shared YAML defines the cube, table, box, path, timing, contact policy, and success thresholds. Lite6 adds a robot-specific task overlay and its own gripper geometry. All five robots support simulation, dry-run, and SDK simulation with both executors. Full real packaging is enabled only for xArm6 + G2 and Lite6 + Lite6 Gripper; the other profiles reject real mode before connection.

```bash
# Simulation defaults to one cycle; finite repetition is explicit
python examples/packaging_showcase.py --robot lite6 --mode sim --executor servo_j
python examples/packaging_showcase.py --robot lite6 --mode sim --executor servo_j --cycles 3
python examples/packaging_showcase.py --robot xarm7 --mode sim --executor servo_cartesian

python examples/packaging_showcase.py --robot uf850 --mode dry-run --executor servo_j
python examples/packaging_showcase.py --robot xarm5 --mode dry-run --executor servo_cartesian
XARM_IP=<ip> python examples/packaging_showcase.py --robot lite6 --mode real \
  --executor servo_j --calibration path/to/exact.yaml --confirm-real
```

## Internal Modules

Files with a `_` prefix are shared internals, not user entry points:

| File | Purpose |
|------|---------|
| `_bootstrap.py` | Adds the project root to `sys.path` |
| `_robot_viewer.py` | Shared Genesis GLB viewer core |
| `_gripper_demo.py` | Gripper G2 open/close control |
| `_bio_gripper_g2_demo.py` | Bio Gripper G2 open/close control |
| `_lite6_gripper_demo.py` | Lite6 gripper open/close control |
| `_packaging_scene.py` | Showcase scene builder |
| `_grasp_place_traj.py` | Legacy shared module (tests still import); user path is `ufactory-grasp-place` |
| `_standalone_gripper_viewer.py` | Standalone gripper viewer helper |
