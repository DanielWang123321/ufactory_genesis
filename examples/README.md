# Task-oriented examples

[中文](README_zh.md)

v0.3.2 organizes public examples by task. Shared implementations live in `ufactory`; entry modules parse user-facing arguments and start the requested workflow.

## Prerequisites

Validated with Python 3.12/3.13 and Genesis World 1.4.4; the minimum Genesis version is 1.4.2. Versions 1.4.2 and 1.4.3 warn once; newer versions must pass the hook checks. See the [version policy](../ROADMAP.md).

1. Clone this repository and run commands from the repository root.
2. Editable install (source-only; wheels/sdists are unsupported):

```bash
git clone https://github.com/xArm-Developer/ufactory_genesis.git
cd ufactory_genesis
pip install -e ".[sim]"

# Optional:
#   pip install -e ".[sim,rl]"        # Linux/NVIDIA reach and pick-place
#   pip install -e ".[sim,real]"      # xArm SDK + Pinocchio/Coal for dry-run/sdk-sim/real safety
#   pip install -e ".[sim,dynamics]"  # dynamics reference checks
```

3. Set a writable Numba cache directory before Genesis-backed runs:

```bash
# Linux / macOS
export NUMBA_CACHE_DIR=~/.cache/numba

# Windows PowerShell
# $env:NUMBA_CACHE_DIR="$env:USERPROFILE\.cache\numba"
```

4. Supported `--robot` keys: `xarm5`, `xarm6`, `xarm7`, `uf850`, `lite6`.

5. **Windows / no supported GPU:** install the CPU PyTorch wheel first, then `pip install -e ".[sim]"`. Pass `--backend cpu` on visualization, pick-place, and packaging commands (do not rely on GPU auto-fallback when an unsupported card still enumerates under CUDA). RL requires Linux/NVIDIA GPUs.

6. **Windows, viewer exits with no error:** if `view_robot.py` returns to the prompt with no window after `Building visualizer...`:

```powershell
$env:MKL_THREADING_LAYER = "SEQUENTIAL"
python examples/visualization/view_robot.py --robot xarm6
```

Permanent: `conda env config vars set MKL_THREADING_LAYER=SEQUENTIAL -n <env>`. `--headless` is unaffected.

Public directories in this release:

| Directory | Purpose |
|---|---|
| `visualization/` | GLB robot and gripper viewers |
| `kinematics/` | FK/IK / robot verification wrappers |
| `pick_place/` | Multi-robot pick-place entry + optional overlay |
| `packaging/` | Multi-robot packaging showcase entry + optional overlay |
| `rl/` | Bare-arm Reach → tabletop Pick & Place → box packing (planned); Linux/NVIDIA |

See the [RL index](rl/README.md) for the learning sequence and training/inference guides.

## Supported robots and end effectors

| Robot | Available end effectors |
|---|---|
| `xarm5` | Gripper G2, Bio Gripper G2 |
| `xarm6` | Gripper G2, Bio Gripper G2 |
| `xarm7` | Gripper G2, Bio Gripper G2 |
| `uf850` | Gripper G2, Bio Gripper G2 |
| `lite6` | Lite6 Gripper, Lite6 Vacuum Gripper |

All five families support simulation and offline preflight. Real packaging is enabled for xArm6 + Gripper G2 and Lite6 + Lite6 Gripper; see [Hardware & Real Robot](#hardware--real-robot).

G2 simulations use `g2_calibrated_v1`. The [pick-place RL guide](rl/pick_place/README.md) documents training and the current v3 checkpoint contract.

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
| `--backend` | `cpu` / `gpu` (default `gpu`; use `cpu` without a supported GPU) |

## Kinematics & Dynamics Simulation

```bash
# Offline robot / asset sanity (no controller)
python examples/kinematics/verify_robot.py --robot xarm6

# Dynamics simulation check
dynamics-sim-check --robot xarm6 --random-count 5
```

> For FK/IK comparison against a physical controller or hardware dynamics verification, see [Hardware & Real Robot](#hardware--real-robot).

## Pick-place

`examples/pick_place/run.py` delegates to the stable `ufactory-pick-place` console command. Both forms accept the same flags.

| Flag | Values / notes |
|------|----------------|
| `--robot` | Required: `xarm5` / `xarm6` / `xarm7` / `uf850` / `lite6` |
| `--mode` | Required for simulation: `sim` (Genesis physics), `dry-run` (offline trajectory and collision preflight) |
| `--executor` | Required: `servo_j` or `servo_cartesian` |
| `--backend` | Optional: `cpu` / `gpu`; overrides `simulation.backend` (use `cpu` without a Genesis-supported GPU) |
| `--visual` | Optional: open the Genesis viewer in `sim` mode |
| `--config` | Optional strict partial overlay YAML |
| `--print-config` | Print resolved runtime YAML and exit |
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

# CPU-only / unsupported GPU
python examples/pick_place/run.py \
  --robot lite6 --mode sim --executor servo_cartesian --backend cpu

# Optional overlay (only listed fields override assets/configs/runtime)
python examples/pick_place/run.py \
  --robot xarm6 --mode dry-run --executor servo_j \
  --config examples/pick_place/runtime.example.yaml
```

`runtime.example.yaml` is a strict partial overlay. Omitted robot, geometry, motion, safety, and simulation values continue to resolve from `assets/configs/runtime`.

> For physical robot pick-place, see [Hardware & Real Robot](#hardware--real-robot).

## Packaging Showcase Simulation

`examples/packaging/run.py` delegates to `ufactory-packaging-showcase`.

The cube is released 50 mm above the box rim and falls naturally. GPU regressions
allow rebound and require its final collision bounds to remain inside the box
(1 mm contact tolerance), rather than within 25 mm of the target center.
Grasp, transport, release and return-home checks remain in place.

| Flag | Values / notes |
|------|----------------|
| `--robot` | Default `xarm6`; all five families supported in sim / dry-run |
| `--mode` | Required for simulation: `sim` (default), `dry-run` (offline preflight) |
| `--executor` | Default `servo_j`; also `servo_cartesian` |
| `--backend` | Optional: `cpu` / `gpu`; same meaning as pick-place |
| `--cycles N` | Exact simulation cycle count (default 1) |
| `--speed` | Simulation playback multiplier (`>1` is faster) |
| `--table-height` | Simulation display height only; base-frame geometry unchanged |
| `--visual` | Optional: hold the final viewer frame |
| `--config` / `--print-config` / `--report` | Same roles as pick-place |

```bash
# One simulation cycle
python examples/packaging/run.py \
  --robot xarm6 --mode sim --executor servo_j --visual

# CPU-only / unsupported GPU
python examples/packaging/run.py \
  --robot xarm6 --mode sim --executor servo_j --backend cpu

# Three-cycle regression
python examples/packaging/run.py \
  --robot lite6 --mode sim --executor servo_j --cycles 3

# Offline preflight
python examples/packaging/run.py \
  --robot xarm7 --mode dry-run --executor servo_j

# Optional overlay
python examples/packaging/run.py \
  --robot xarm6 --mode sim --executor servo_j \
  --config examples/packaging/runtime.example.yaml
```

> For physical robot packaging, see [Hardware & Real Robot](#hardware--real-robot).

## Reinforcement Learning

The Linux/NVIDIA simulation examples follow this order:

1. [Reach](rl/reach/README.md): bare-arm xArm 6, with a 5 mm success threshold.
2. [Pick & Place](rl/pick_place/README.md): xArm6 + Gripper G2 transfers a cube to a tabletop target.
3. Box packing: place an object inside a box; the RL example is planned.

See the [RL index](rl/README.md) for training and inference. The packaging showcase above uses trajectory control.

## Hardware & Real Robot

> [!CAUTION]
> **Safety Requirements**:
> 1. Requires the real hardware extra: `pip install -e ".[real]"` (includes xArm SDK and Pinocchio/Coal collision checks).
> 2. Physical motion carries collision and equipment damage risks. Motion commands require exact per-unit kinematics calibration (`--calibration`) and an explicit confirmation gate (`--confirm-real`). Omitting either rejects before connection.
> 3. Real-robot commands are strictly capped at **1 cycle**.

### 1. Hardware Read-only & Inspection Commands (No Motion)

Connects to a physical controller over the network for validation without commanding joint motion:

```bash
# Compare Genesis URDF FK/IK against the xArm SDK (requires network + IP)
python examples/kinematics/verify_fk.py --robot xarm6 --ip <ip>
python examples/kinematics/verify_ik.py --robot lite6 --ip <ip>

# Collision detection check against controller pose (no motion commanded)
dynamics-sim-collision-check --robot xarm6 --ip <ip>

# Controller-internal simulation (sdk-sim: validates controller logic, joints do not move)
python examples/packaging/run.py \
  --robot lite6 --mode sdk-sim --executor servo_cartesian \
  --ip <ip> --calibration path/to/exact.yaml
```

### 2. Physical Motion Commands (Drives Robot & Gripper)

Sends actual trajectory and servo setpoints to physical hardware:

```bash
# Dynamics hardware identification & torque verification (requires confirmation gate)
dynamics-hardware-check --robot xarm6 --ip <ip> --confirm-real

# Real pick-place motion (exact calibration + confirmation gate required)
XARM_IP=<ip> python examples/pick_place/run.py \
  --robot xarm6 --mode real --executor servo_j \
  --calibration path/to/exact.yaml --confirm-real

# Real packaging motion (currently enabled for xArm6 + G2 and Lite6 + Gripper)
XARM_IP=<ip> python examples/packaging/run.py \
  --robot lite6 --mode real --executor servo_j \
  --calibration path/to/exact.yaml --confirm-real
```

> Packaging `--mode real` on xArm5, xArm7, and UF850 aborts before connecting until real gripper support is enabled. Real mode never loops.
