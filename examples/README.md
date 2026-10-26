# Task-oriented examples

[中文](README_zh.md)

## Prerequisites

Validated with Python 3.12/3.13 and Genesis World 1.4.5; the minimum Genesis version is 1.4.2. Versions 1.4.2 to 1.4.4 warn once; 1.4.5 is also the maximum, and newer versions are rejected. See the [version policy](../ROADMAP.md).

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

# Default: PyPI build of torch 2.10.0 (CUDA 12.8). Validated build: 2.10.0+cu128.
# For another build, install the same torch version first (not validated: cu126, cu130, cpu):
#   pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126   # or cu130 / cpu
#   pip install -e ".[sim]"
```

`[sim]` pins `torch==2.10.0`; use a separate virtual environment for packages that require another torch version (for example lerobot).

3. Set a writable Numba cache directory before Genesis-backed runs:

```bash
# Linux / macOS
export NUMBA_CACHE_DIR=~/.cache/numba

# Windows PowerShell
# $env:NUMBA_CACHE_DIR="$env:USERPROFILE\.cache\numba"
```

4. Supported `--robot` keys: `xarm5`, `xarm6`, `xarm7`, `uf850`, `lite6`.

5. **Windows / no supported GPU:** install `torch==2.10.0` from the CPU index (`--index-url https://download.pytorch.org/whl/cpu`) first, then `pip install -e ".[sim]"`. Pass `--backend cpu` on visualization, pick-place, and packaging commands (do not rely on GPU auto-fallback when an unsupported card still enumerates under CUDA). RL requires Linux/NVIDIA GPUs.

6. **Windows, viewer exits with no error:** if `view_robot.py` returns to the prompt with no window after `Building visualizer...`:

```powershell
$env:MKL_THREADING_LAYER = "SEQUENTIAL"
python examples/visualization/view_robot.py --robot xarm6
```

Permanent: `conda env config vars set MKL_THREADING_LAYER=SEQUENTIAL -n <env>`. `--headless` is unaffected.

Public directories in this release:

| Directory | Purpose |
|---|---|
| `visualization/` | GLB robot and gripper viewers; see [Visualization](#visualization) |
| `kinematics/` | FK/IK / robot verification wrappers; see [Kinematics & Dynamics Simulation](#kinematics--dynamics-simulation) |
| `pick_place/` | Multi-robot pick-place entry + optional overlay; see [pick-place guide](pick_place/README.md) |
| `packaging/` | Multi-robot packaging showcase entry + optional overlay; see [packaging guide](packaging/README.md) |
| `rl/` | Bare-arm Reach → tabletop Pick & Place → box packing (planned); Linux/NVIDIA; see [RL index](rl/README.md) |

See the [RL index](rl/README.md) for the learning sequence and training/inference guides.

## Supported robots and end effectors

| Robot | Available end effectors |
|---|---|
| `xarm5` | Gripper G2, Bio Gripper G2, Vacuum Gripper |
| `xarm6` | Gripper G2, Bio Gripper G2, Vacuum Gripper |
| `xarm7` | Gripper G2, Bio Gripper G2, Vacuum Gripper |
| `uf850` | Gripper G2, Bio Gripper G2, Vacuum Gripper |
| `lite6` | Lite6 Gripper, Lite6 Vacuum Gripper |

All five families support simulation and offline preflight. Real packaging is enabled for xArm6 + Gripper G2 and Lite6 + Lite6 Gripper; see [Hardware & Real Robot](#hardware--real-robot).

G2 simulations use `g2_calibrated_v1`. The [pick-place RL guide](rl/pick_place/README.md) documents training and the current v3 checkpoint contract.

## Visualization

```bash
# xArm6 bare arm static view
python examples/visualization/view_robot.py --robot xarm6

# xArm6 + Gripper G2 static view
python examples/visualization/view_robot.py --robot xarm6 --gripper-g2

# xArm6 + Gripper G2 open/close demo
python examples/visualization/view_robot.py --robot xarm6 --gripper-g2 --movable --gripper-demo

# UF850 + Bio Gripper G2 static view
python examples/visualization/view_robot.py --robot uf850 --bio-gripper-g2

# xArm6 + Bio Gripper G2 open/close demo
python examples/visualization/view_robot.py --robot xarm6 --bio-gripper-g2 --movable --gripper-demo

# xArm6 + Vacuum Gripper static view
python examples/visualization/view_robot.py --robot xarm6 --vacuum-gripper

# Lite6 + Lite6 Gripper open/close demo
python examples/visualization/view_robot.py --robot lite6 --lite6-gripper --movable --gripper-demo

# Lite6 + Lite6 Vacuum Gripper static view
python examples/visualization/view_robot.py --robot lite6 --lite6-vacuum-gripper

# Gripper G2 standalone viewer and open/close demo
python examples/visualization/view_gripper_g2.py

# Bio Gripper G2 standalone viewer and open/close demo
python examples/visualization/view_bio_gripper_g2.py

# Lite6 Gripper standalone viewer and open/close demo
python examples/visualization/view_lite6_gripper.py
```

| Flag | Meaning |
|------|---------|
| `--robot` | Required profile key |
| `--gripper-g2` / `--bio-gripper-g2` / `--vacuum-gripper` / `--lite6-gripper` / `--lite6-vacuum-gripper` | End-effector combo (at most one) |
| `--movable` | Per-link movable gripper meshes (requires a supported gripper flag) |
| `--gripper-demo` | Cycle open/close with a 0.5 s hold after reaching each endpoint (requires `--movable`) |
| `--pd` | Smooth joint-motion demo (~0.873 rad/s), visual only |
| `--show-tcp` | Red flange TCP marker (hidden by default) |
| `--diagnose` | Headless STL/GLB link-pose diagnostic |
| `--headless` | No viewer window |
| `--backend` | `cpu` / `gpu` (default `gpu`; use `cpu` without a supported GPU) |

## Kinematics & Dynamics Simulation

```bash
# xArm6 offline URDF, FK, and PD control sanity check (no controller)
python examples/kinematics/verify_robot.py --robot xarm6

# xArm6 dynamics simulation check (default poses + 5 random poses)
dynamics-sim-check --robot xarm6 --random-count 5
```

> For FK/IK comparison against a physical controller or hardware dynamics verification, see [Hardware & Real Robot](#hardware--real-robot).

## Pick-place

Multi-robot pick-place in Genesis physics simulation or offline preflight. Flags, more commands, and the configuration overlay are in the [pick-place guide](pick_place/README.md).

```bash
# xArm6 + Gripper G2: offline trajectory and collision preflight (servo_j)
python examples/pick_place/run.py \
  --robot xarm6 --mode dry-run --executor servo_j

# Lite6 + Lite6 Gripper: pick-place physics simulation with viewer (servo_cartesian)
python examples/pick_place/run.py \
  --robot lite6 --mode sim --executor servo_cartesian --visual
```

## Packaging Showcase Simulation

Multi-robot box packing with trajectory control. Flags and more commands are in the [packaging guide](packaging/README.md).

```bash
# xArm6 + Gripper G2: 1-cycle packaging physics simulation with viewer (servo_j)
python examples/packaging/run.py \
  --robot xarm6 --mode sim --executor servo_j --visual
```

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
# xArm6: compare Genesis URDF FK against physical controller SDK
python examples/kinematics/verify_fk.py --robot xarm6 --ip <ip>

# Lite6: compare Genesis URDF IK against physical controller SDK
python examples/kinematics/verify_ik.py --robot lite6 --ip <ip>

# xArm6: self-collision check across preset test poses in controller simulation mode (joints do not move)
dynamics-sim-collision-check --robot xarm6 --ip <ip>

# Lite6 + Lite6 Gripper: packaging controller-internal simulation (sdk-sim, joints do not move)
python examples/packaging/run.py \
  --robot lite6 --mode sdk-sim --executor servo_cartesian \
  --ip <ip> --calibration path/to/exact.yaml
```

### 2. Physical Motion Commands (Drives Robot & Gripper)

Sends actual trajectory and servo setpoints to physical hardware:

```bash
# xArm6: move physical robot to validation poses and compare joint torques (requires --confirm-real)
dynamics-hardware-check --robot xarm6 --ip <ip> --confirm-real

# xArm6 + Gripper G2: real pick-place motion (requires per-unit calibration + --confirm-real)
XARM_IP=<ip> python examples/pick_place/run.py \
  --robot xarm6 --mode real --executor servo_j \
  --calibration path/to/exact.yaml --confirm-real

# Lite6 + Lite6 Gripper: real packaging motion (enabled for xArm6 + Gripper G2 and Lite6 + Lite6 Gripper; requires per-unit calibration + --confirm-real)
XARM_IP=<ip> python examples/packaging/run.py \
  --robot lite6 --mode real --executor servo_j \
  --calibration path/to/exact.yaml --confirm-real
```

> Packaging `--mode real` on xArm5, xArm7, and UF850 aborts before connecting until real gripper support is enabled. Real mode never loops.
