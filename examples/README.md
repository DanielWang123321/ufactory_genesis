# Task-oriented examples

[中文](README_cn.md)

v0.2.7 organizes public examples by task. Shared implementations live in `ufactory`; these scripts only parse user-facing arguments and start the requested workflow.

## Prerequisites

1. Clone this repository and run commands from the repository root.
2. Editable install (source-only; wheels/sdists are unsupported):

```bash
pip install -e ".[sim]"

# Optional:
#   pip install -e ".[real]"      # xArm SDK + Pinocchio/Coal for dry-run/sdk-sim/real safety
#   pip install -e ".[showcase]"  # packaging box textures (scipy)
```

3. Set a writable Numba cache directory before Genesis-backed runs:

```bash
export NUMBA_CACHE_DIR=~/.cache/numba
```

4. Supported `--robot` keys: `xarm5`, `xarm6`, `xarm7`, `uf850`, `lite6`.

Public directories in this release:

| Directory | Purpose |
|---|---|
| `visualization/` | GLB robot and gripper viewers |
| `kinematics/` | FK/IK / robot verification wrappers |
| `pick_place/` | Multi-robot pick-place entry + optional overlay |
| `packaging/` | Multi-robot packaging showcase entry + optional overlay |

RL example scripts are not part of the v0.2.7 public tree. Library helpers remain under `ufactory.training` when `.[rl]` is installed.

## Visualization

```bash
# Bare arm
python examples/visualization/view_robot.py --robot xarm6

# Gripper G2 (static combo)
python examples/visualization/view_robot.py --robot xarm6 --gripper-g2

# Movable fingers + open/close demo
python examples/visualization/view_robot.py --robot xarm6 --gripper-g2 --movable --gripper-demo

# Bio Gripper G2 / Lite6 grippers
python examples/visualization/view_robot.py --robot uf850 --bio-gripper-g2
python examples/visualization/view_robot.py --robot lite6 --lite6-gripper --movable --gripper-demo
python examples/visualization/view_robot.py --robot lite6 --lite6-vacuum-gripper

# Standalone gripper viewers
python examples/visualization/view_gripper_g2.py
python examples/visualization/view_bio_gripper_g2.py
python examples/visualization/view_lite6_gripper.py
```

| Flag | Meaning |
|------|---------|
| `--robot` | Required profile key |
| `--gripper-g2` / `--bio-gripper-g2` / `--lite6-gripper` / `--lite6-vacuum-gripper` | End-effector combo (at most one) |
| `--movable` | Per-link movable gripper meshes (requires a supported gripper flag) |
| `--gripper-demo` | Cycle open/close (requires `--movable`) |
| `--pd` | Smooth joint-motion demo (~0.873 rad/s), visual only |
| `--show-tcp` | Red flange TCP marker (hidden by default) |
| `--diagnose` | Headless STL/GLB link-pose diagnostic |
| `--headless` | No viewer window |

## Kinematics

```bash
# Offline robot / asset sanity (no controller)
python examples/kinematics/verify_robot.py --robot xarm6

# Compare Genesis URDF FK/IK against the xArm SDK (requires network + IP)
python examples/kinematics/verify_fk.py --robot xarm6 --ip <ip>
python examples/kinematics/verify_ik.py --robot lite6 --ip <ip>
```

Dynamics validation stays on the console commands:

```bash
dynamics-sim-check --robot xarm6 --random-count 5
dynamics-hardware-check --robot xarm6 --ip <ip> --confirm-real
dynamics-sim-collision-check --robot xarm6 --ip <ip>
```

## Pick-place

`examples/pick_place/run.py` delegates to the stable `ufactory-pick-place` console command. Both forms accept the same flags.

| Flag | Values / notes |
|------|----------------|
| `--robot` | Required: `xarm5` / `xarm6` / `xarm7` / `uf850` / `lite6` |
| `--mode` | Required: `sim` (Genesis), `dry-run` (offline preflight, no controller), `sdk-sim` (controller simulation), `real` |
| `--executor` | Required: `servo_j` or `servo_cartesian` |
| `--config` | Optional strict partial overlay YAML |
| `--print-config` | Print resolved runtime YAML and exit |
| `--ip` | Controller IP (or set `XARM_IP`) for `sdk-sim` / `real` |
| `--calibration` | Exact per-unit kinematics YAML (required for real) |
| `--confirm-real` | Explicit confirmation gate for `--mode real` |
| `--visual` | `sim`: force Genesis viewer; `real`: kinematic mirror (not for dry-run/sdk-sim) |
| `--report` | Optional report output path |

```bash
# Resolve configuration only
python examples/pick_place/run.py \
  --robot xarm6 --mode dry-run --executor servo_j --print-config

# Offline predictive preflight
python examples/pick_place/run.py \
  --robot xarm6 --mode dry-run --executor servo_j

# Genesis simulation with viewer
python examples/pick_place/run.py \
  --robot lite6 --mode sim --executor servo_cartesian --visual

# Optional overlay (only listed fields override assets/configs/runtime)
python examples/pick_place/run.py \
  --robot xarm6 --mode dry-run --executor servo_j \
  --config examples/pick_place/runtime.example.yaml

# Real motion (exact calibration + confirmation required)
XARM_IP=<ip> python examples/pick_place/run.py \
  --robot xarm6 --mode real --executor servo_j \
  --calibration path/to/exact.yaml --confirm-real
```

`runtime.example.yaml` is a strict partial overlay. Omitted robot, geometry, motion, safety, and simulation values continue to resolve from `assets/configs/runtime`.

## Packaging

Generate box textures once before the first packaging run (requires `.[showcase]`):

```bash
python scripts/generate_showcase_textures.py
```

`examples/packaging/run.py` delegates to `ufactory-packaging-showcase`.

| Flag | Values / notes |
|------|----------------|
| `--robot` | Default `xarm6`; all five families supported in sim/dry-run/sdk-sim |
| `--mode` | Default `sim`; same four modes as pick-place |
| `--executor` | Default `servo_j`; also `servo_cartesian` |
| `--cycles N` | Exact simulation cycle count (default 1) |
| `--speed` | Simulation playback multiplier (`>1` is faster) |
| `--table-height` | Simulation display height only; base-frame geometry unchanged |
| `--config` / `--print-config` / `--ip` / `--calibration` / `--confirm-real` / `--visual` / `--report` | Same roles as pick-place |

```bash
# One simulation cycle, then hold the final frame
python examples/packaging/run.py \
  --robot xarm6 --mode sim --executor servo_j

# Three-cycle regression
python examples/packaging/run.py \
  --robot lite6 --mode sim --executor servo_j --cycles 3

# Offline preflight / controller simulation
python examples/packaging/run.py \
  --robot xarm7 --mode dry-run --executor servo_j
python examples/packaging/run.py \
  --robot lite6 --mode sdk-sim --executor servo_cartesian \
  --ip <ip> --calibration path/to/exact.yaml

# Real packaging is enabled only for xArm6 + G2 and Lite6 + Lite6 Gripper
XARM_IP=<ip> python examples/packaging/run.py \
  --robot lite6 --mode real --executor servo_j \
  --calibration path/to/exact.yaml --confirm-real

# Optional overlay
python examples/packaging/run.py \
  --robot xarm6 --mode sim --executor servo_j \
  --config examples/packaging/runtime.example.yaml
```

Real mode never loops. xArm5, xArm7, and UF850 `--mode real` fail before controller connection until a real gripper path is enabled.

## Migration from v0.2.6

| Old category | v0.2.7 path |
|---|---|
| `examples/view_robot_glb.py`, per-robot viewers | `examples/visualization/view_robot.py --robot <key>` |
| Root/per-robot FK and IK wrappers | `examples/kinematics/verify_{robot,fk,ik}.py` |
| Per-robot pick-place wrappers | `examples/pick_place/run.py --robot <key>` |
| Root/xArm6 packaging wrappers | `examples/packaging/run.py --robot <key>` |

Old paths, bootstrap files, and underscore-prefixed example internals were removed without compatibility wrappers. RL example entry points are not published in this release.
