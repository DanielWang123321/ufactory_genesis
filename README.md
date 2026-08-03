# ufactory_genesis

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12%20%7C%203.13-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/version-0.2.12-orange" alt="Version">
  <img src="https://img.shields.io/badge/genesis-1.3.1-lightgrey" alt="Genesis">
  <a href="https://github.com/DanielWang123321/ufactory_genesis/actions/workflows/ci.yml"><img src="https://github.com/DanielWang123321/ufactory_genesis/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

UFACTORY robot models and Genesis simulation utilities — high-fidelity GLB visualization, kinematic calibration, trajectory examples, and simulation-only RL pick-place workflows.

[中文](README.zh.md) | [Contributing](CONTRIBUTING.md) | [Changelog](CHANGELOG.md) | [Roadmap](ROADMAP.md) | [Security](SECURITY.md)

## Project status

- **0.2.x is Alpha / preview.** Public APIs may break between minor versions; pin a git tag for secondary development. See [ROADMAP.md](ROADMAP.md).
- Hosted today on a **personal GitHub account** (`DanielWang123321/ufactory_genesis`). This is **not** an official UFACTORY organization repository and is **not** an official product support channel.
- **0.3.x is planned** to move into a UFACTORY official GitHub organization and become the first user-facing supported surface. Until then, treat 0.2 docs and APIs as transitional.
- xArm / UF850 / UFACTORY are trademarks of their respective owners; this repository does not claim official SDK endorsement.

## Verification boundary

| Layer | What it proves | Where it runs |
|-------|----------------|---------------|
| **Public CI** (`project-check fast` equivalent) | Lint, typecheck subset, CPU unit tests | GitHub Actions (no GPU, no Genesis install) |
| **Local sim** (`project-check sim`) | In-process GPU / Genesis regression | Maintainer machines with `[sim]` |
| **Maintainer hardware** (`sdk-sim` / `hardware`) | Cabinet SDK sim and listed real robots | Maintainer lab; optional sanitized summary on GitHub Releases |

The **reference (pinned) baseline** is Python 3.13, Genesis World 1.3.1, PyTorch 2.10.0+cu128, and RSL-RL 5.4.2 for RL. Newer Genesis releases may run when compatibility hooks still match; they are **not** treated as a maintainer-verified physics or hardware baseline until the local sim and hardware checks pass. Public CI does **not** replace those checks.

## Table of Contents

- [Project status](#project-status)
- [Verification boundary](#verification-boundary)
- [Quick Start](#quick-start)
- [Windows and CPU-only simulation](#windows-and-cpu-only-simulation)
- [Supported Robots](#supported-robots)
- [GLB Visual Preview](#glb-visual-preview)
- [Trajectory Pick-Place](#trajectory-pick-place)
- [RL pick-place examples](#rl-pick-place-examples)
- [Showcase](#showcase-yaml-driven-packaging)
- [API Quick Reference](#api-quick-reference)
- [Real-Robot Kinematic Calibration](#real-robot-kinematic-calibration-sn-rules)
- [xArm 6 — Reference Robot](#xarm-6)
- [Project Layout](#project-layout)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Quick Start

v0.2.12 is source-only: clone this repository and use an editable install. Wheels, sdists, remote asset downloads, and installation outside a Git checkout are unsupported. Missing repository assets fail with an actionable `AssetLayoutError`. Genesis World 1.3.1 or newer is required. Visual GLB meshes are Draco-compressed (typical `assets/` checkout on the order of tens of MB).

```bash
# From a cloned repository
pip install -e ".[sim]"

# Optional extras:
#   pip install -e ".[real]"      # xArm SDK + Pinocchio/Coal safety backends
#   pip install -e ".[sim,rl]"    # RL pick-place examples + ufactory.training
#   pip install -e ".[showcase]"  # packaging showcase scipy dependency

export NUMBA_CACHE_DIR=~/.cache/numba

# Preview xArm 6 GLB model
python examples/visualization/view_robot.py --robot xarm6

# Local CPU quality check (same class as public CI; does not replace sim/hardware evidence)
project-check fast
```

Since 2024, new xArm shipments use the **XI1305** hardware revision. Short names `xarm5` / `xarm6` / `xarm7` resolve to `xarm5_1305` / `xarm6_1305` / `xarm7_1305`. The explicit `*_1305` keys remain supported. Older model codes (11, 12, 1300–1304) are not bundled — supply your own URDF via `--urdf` or `prepare_robot_model_for_verification(robot_model=...)`.

## Windows and CPU-only simulation

Pick-place and packaging `--mode sim` (plus GLB preview) can run on Windows and on machines without a Genesis-supported GPU. Reinforcement learning is out of scope for this path.

**Install CPU PyTorch first**, then this package (Genesis World ≥ 1.3.1):

```bash
# Linux / Windows — CPU torch (see https://pytorch.org for the current CPU index)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[sim]"
# Packaging textures (optional): pip install -e ".[showcase]"
```

Default `simulation.backend` is `gpu` (prefer GPU on maintainer machines). **Do not rely on automatic GPU→CPU fallback** when a card is present but unsupported by Genesis (for example older GeForce that still enumerates under CUDA). Force CPU:

```bash
python examples/visualization/view_robot.py --robot xarm6 --backend cpu
python examples/pick_place/run.py --robot lite6 --mode sim --executor servo_cartesian --backend cpu
python examples/packaging/run.py --robot xarm6 --mode sim --executor servo_j --backend cpu
```

Or set `simulation.backend: cpu` in a `--config` overlay. Optional: `QD_ENABLE_CUDA=0` to skip CUDA probing. Keep default `GS_ENABLE_NDARRAY=1`. Do not set `PYOPENGL_PLATFORM=osmesa` on Windows.

**Windows (PowerShell) Numba cache:**

```powershell
$env:NUMBA_CACHE_DIR="$env:USERPROFILE\.cache\numba"
# optional on unsupported GPUs:
# $env:QD_ENABLE_CUDA="0"
```

`.[real]` / Pinocchio dry-run backends are not part of this Windows/CPU-only goal.

## Supported Robots

| profile key | alias | Model | DOF | Gripper G2 | Bio Gripper G2 | Lite6 Gripper | Lite6 Vacuum |
|-------------|-------|-------|-----|:----------:|:--------------:|:-------------:|:------------:|
| `xarm5_1305` | `xarm5` | xArm 5 | 5 | ✓ | ✓ | — | — |
| `xarm6_1305` | `xarm6` | xArm 6 | 6 | ✓ | ✓ | — | — |
| `xarm7_1305` | `xarm7` | xArm 7 | 7 | ✓ | ✓ | — | — |
| `uf850` | — | UF850 | 6 | ✓ | ✓ | — | — |
| `lite6` | — | Lite6 | 6 | — | — | ✓ | ✓ |

✓ = combo URDF available (static GLB visual); Gripper G2, Bio Gripper G2, and Lite6 Gripper also support `--movable` open/close animation.

**Gripper G2** and **Bio Gripper G2** are shared xArm/UF850 accessories. **Lite6 Gripper** (parallel jaw) and **Lite6 Vacuum Gripper** are Lite6-only. End-effector flags are mutually exclusive.

## GLB Visual Preview

Unified entry point `examples/visualization/view_robot.py` for all robot profiles and end effectors:

```bash
export NUMBA_CACHE_DIR=~/.cache/numba

# Arm only
python examples/visualization/view_robot.py --robot <profile_key>

# Gripper G2 (static / movable open-close)
python examples/visualization/view_robot.py --robot xarm6 --gripper-g2
python examples/visualization/view_robot.py --robot xarm6 --gripper-g2 --movable --gripper-demo

# Bio Gripper G2 (static)
python examples/visualization/view_robot.py --robot uf850 --bio-gripper-g2

# Lite6 parallel gripper (static / movable open-close)
python examples/visualization/view_robot.py --robot lite6 --lite6-gripper
python examples/visualization/view_robot.py --robot lite6 --lite6-gripper --movable --gripper-demo

# Lite6 vacuum gripper (static)
python examples/visualization/view_robot.py --robot lite6 --lite6-vacuum-gripper
```

Standalone movable gripper viewers are available beside the robot viewer as `view_gripper_g2.py`, `view_bio_gripper_g2.py`, and `view_lite6_gripper.py`.

| Flag | Product | Effect |
|------|---------|--------|
| `--gripper-g2` | Gripper G2 | Load combo URDF |
| `--movable` | Gripper G2 / Lite6 Gripper / Bio Gripper G2 | Per-link GLBs (required for animation) |
| `--gripper-demo` | Gripper G2 / Bio Gripper G2 / Lite6 Gripper | Cycle open ↔ close |
| `--bio-gripper-g2` | Bio Gripper G2 | Static GLB overlay |
| `--lite6-gripper` | Lite6 Gripper | Lite6 parallel gripper combo URDF |
| `--lite6-vacuum-gripper` | Lite6 Vacuum Gripper | Lite6 vacuum static GLB |
| `--pd` | Arm | Joint motion demo (50 deg/s smooth interp, not stiff PD) |
| `--show-tcp` | Arm | Show red TCP debug marker on EE flange (default: hidden) |

## Trajectory Pick-Place

The configuration-driven v0.2.12 entry point supports all five robot families. `dry-run` performs calibrated FK, whole-program timing and Pinocchio/Coal collision checks without connecting to a controller.

| Robot | Command | Notes |
|-------|---------|-------|
| xArm5 | `ufactory-pick-place --robot xarm5 --mode dry-run --executor servo_j` | Both executors |
| xArm6 | `ufactory-pick-place --robot xarm6 --mode dry-run --executor servo_j` | Both executors |
| xArm7 | `ufactory-pick-place --robot xarm7 --mode dry-run --executor servo_j` | Both executors |
| UF850 | `ufactory-pick-place --robot uf850 --mode dry-run --executor servo_j` | Both executors |
| Lite6 | `ufactory-pick-place --robot lite6 --mode dry-run --executor servo_j` | Binary real gripper capability |

```bash
# Resolve configuration only; no Genesis or robot connection.
ufactory-pick-place --robot xarm6 --mode dry-run --executor servo_j --print-config

# Offline predictive preflight.
ufactory-pick-place --robot xarm6 --mode dry-run --executor servo_j

# Real motion requires strict per-unit calibration and explicit confirmation.
# The task program is planned from default_qpos. If the arm is away from that
# start, --confirm-real first prepositions with MODE_POSITION set_servo_angle,
# then switches to servo streaming.
XARM_IP=192.168.1.xx ufactory-pick-place --robot xarm6 --mode real \
  --executor servo_j --calibration path/to/exact.yaml --confirm-real

# Real + kinematic mirror window (open-loop teleport, async, off the servo path).
XARM_IP=192.168.1.xx ufactory-pick-place --robot xarm6 --mode real \
  --executor servo_j --calibration path/to/exact.yaml --confirm-real --visual
```

`--visual`: with `--mode sim`, force the Genesis viewer; with `--mode real`, open the kinematic mirror (no contact physics). The generic pick-place mirror uses capped non-blocking updates; the packaging command runs its full-rate Genesis/GLB viewer in a separate process, consuming all 50 Hz mirror states with the normal 60 Hz repaint rate without sharing the servo sender's Python scheduler. The window remains open until it is closed or Ctrl+C is pressed. Not supported for `dry-run` / `sdk-sim`.

v0.2.7 removed online real-policy deployment, random-action execution, policy sessions, and SDK policy-action adapters from the public package (see [SECURITY.md](SECURITY.md)). The v0.2.12 RL examples remain simulation-only and add no real-policy execution interface.

## RL pick-place examples

The original xArm6 + Gripper G2 fixed `+Y` workflow remains available with its canonical recipe, fixed 512-episode scenario bank, and 2.7 MB `model_199.pt` checkpoint. v0.2.12 also adds an object-randomized, fixed-target workflow under `random_start/`, with separate recipes, scenario banks, evaluation gates, and documentation. Its bundled checkpoint is explicitly hierarchical: it preserves the fixed RL actor and uses a simulator-state scripted guide on non-canonical layouts; it is not presented as learned random-layout PPO generalization. Both workflows are simulation-only.

```bash
pip install -e ".[sim,rl]"
python -m examples.rl.pick_place.evaluate --episodes 1
```

The examples support Linux with an NVIDIA GPU only. Start with the [English RL index](examples/rl/README.md) or [Chinese RL index](examples/rl/README_cn.md) for fixed-layout and random-start guides, formal evaluation, training, metrics, and limitations.

## Showcase (YAML-Driven Packaging)

`ufactory-packaging-showcase` supports xArm5, xArm6, xArm7, UF850, and Lite6 in `sim`, `dry-run`, and `sdk-sim`, with both `servo_j` and `servo_cartesian`. The physical scene, task trajectory, timings, success thresholds, contacts, and gripper geometry resolve from versioned YAML. The shared defaults live in `assets/configs/runtime/tasks/packaging_showcase.yaml`; Lite6 applies `assets/configs/runtime/tasks/robots/lite6_packaging_showcase.yaml`.

| Robot/end effector | Sim / dry-run / SDK sim | Full real packaging |
|--------------------|:-----------------------:|:-------------------:|
| xArm6 + Gripper G2 | ✓ | ✓ |
| Lite6 + Lite6 Gripper | ✓ | ✓ |
| xArm5 / xArm7 / UF850 + configured G2 model | ✓ | Disabled until a real gripper is enabled |

All simulations use contact and friction; the cube is never attached to the gripper. G2 retains its tuned 22→29 mm, 0.2 s preload relaxation and one-tick full-open target. Lite6 disables preload relaxation and issues its binary full-open target at the start of a 0.5 s release/settle segment. Its closer workspace uses cube/home coordinates `[0.200, 0, 0.015]` / `[0.200, 0, 0.200]` and the nearest fully preflighted box center `(0.200, 0.220)`; `(0.200, 0.150)` intersects the Lite6 link4 pickup envelope with the fixed 300 × 200 mm box. Generate box textures once before the first run:

```bash
export NUMBA_CACHE_DIR=~/.cache/numba
python scripts/generate_showcase_textures.py

# One cycle, then hold the final frame; select any supported robot.
ufactory-packaging-showcase --robot xarm6 --mode sim --executor servo_j
ufactory-packaging-showcase --robot lite6 --mode sim --executor servo_cartesian

# Release regression: three physical cycles per robot/executor.
ufactory-packaging-showcase --robot lite6 --mode sim --executor servo_j --cycles 3

# Offline collision preflight and controller simulation.
ufactory-packaging-showcase --robot xarm7 --mode dry-run --executor servo_j
ufactory-packaging-showcase --robot lite6 --mode sdk-sim --executor servo_cartesian \
  --ip <ip> --calibration path/to/exact.yaml

# Real execution is enabled only for xArm6 + G2 and Lite6 + Lite6 Gripper.
XARM_IP=<ip> ufactory-packaging-showcase --robot lite6 --mode real \
  --executor servo_j --calibration path/to/exact.yaml --confirm-real

# Task-oriented example entry: delegates to the same CLI.
python examples/packaging/run.py \
  --robot xarm6 --mode sim --executor servo_j
```

`servo_j` startup first builds the Genesis IK scene, then preflights the complete trajectory. `[ik-compile]` and `[preflight]` messages show the active phase, sample count, and timing; hardware motion remains unauthorized until `preflight=PASS`. Genesis's neutral self-collision filtering message and Quadrants' `ast.keyword(..., ctx=...)` Python 3.15 deprecation message are upstream diagnostics, not failures in the reference Python 3.13 / Genesis 1.3.1 baseline. Collision preflight still checks every trajectory sample and geometry pair, but uses the configured 5 mm security margin to select candidates and computes exact distances only for those candidates; unsupported backends automatically retain the full-distance fallback.

| Flag | Description |
|------|-------------|
| `--robot` | `xarm5`, `xarm6`, `xarm7`, `uf850`, or `lite6` |
| `--table-height` | Simulation display height only; base-frame packaging geometry is unchanged |
| `--speed` | Simulation playback multiplier (>1 is faster) |
| `--cycles N` | Run exactly N simulation cycles (default: 1) |
| `--loop` / `--no-loop` | Explicit infinite loop / compatibility alias for one cycle |

Every repeated simulation cycle restores the cube position, identity orientation, and zero velocity before the next grasp. A failed lift, placement, or home return stops further cycles. Real mode requires exact per-unit calibration and explicit confirmation and never loops; `servo_cartesian` additionally obtains matching same-process SDK-simulation evidence. xArm5, xArm7, and UF850 `--mode real` are rejected before controller connection rather than silently skipping the gripper.

After release, the main packaging scene never resets the cube, clears its velocity, binds it to another body, or scales its rebound. `ufactory.manipulation.packaging.measure_natural_drop()` provides an isolated diagnostic using the configured cube, box floor, and solver parameters; it records impact velocities, peak rebound, and settling time without requiring a visibly large bounce.

### Optional dependency matrix

| Workflow | Install | Pinocchio / Coal |
|---|---|---|
| Visualization and contact simulation | `.[sim]` | Not installed or imported |
| Fixed-layout RL example and `ufactory.training` | `.[sim,rl]` | Not installed or imported |
| Real-robot predictive safety | `.[real]` | Required; missing backends fail before motion |
| Independent dynamics reference | `.[dynamics]` | Required; missing backends report an actionable error |

The public RL tree is intentionally limited to `examples/rl/pick_place`; private experiment archives remain outside the tracked example.

## API Quick Reference

The project uses functional subpackages. The root namespace is intentionally small:

```python
import ufactory
```

| Root API | Description |
|----------|-------------|
| `ufactory.ROBOT_PROFILES` | Supported robot profile registry |
| `ufactory.get_robot_profile(key)` | Resolve a robot profile by key or short name |
| `ufactory.robot_cli_choices()` | Sorted `--robot` choices |
| `ufactory.robot_urdf(key, name=None)` | Absolute path to a default or named URDF |
| `ufactory.robot_visual_glb_urdf(key, with_*=..., movable=...)` | Absolute path to GLB visual URDFs |
| `ufactory.robot_assets(name)` | Robot asset directory |
| `ufactory.kinematics_user_dir(robot_name)` | Per-unit calibration YAML directory |
| `ufactory.RepositoryAssetStore` | Validate source-tree asset layout and manifest |

Advanced APIs are imported from their owning modules:

| Domain | Canonical import |
|--------|------------------|
| Robot registry and source assets | `ufactory.robots.registry`, `ufactory.robots.paths` |
| Versioned runtime configuration | `ufactory.config` |
| Kinematics calibration and FK/IK validation | `ufactory.kinematics.calibration`, `ufactory.kinematics.validation` |
| Dynamics reports and validation service | `ufactory.dynamics` |
| Real robot SDK/session helpers | `ufactory.hardware.xarm`, `ufactory.hardware.session`, `ufactory.hardware.observe` |
| Gripper command conversions/controllers | `ufactory.grippers.g2`, `ufactory.grippers.bio_g2` |
| Approved planning/preflight/execution | `ufactory.trajectory`, `ufactory.safety` |
| GLB visualization helpers | `ufactory.visualization.glb` |
| Packaging task geometry/scene/diagnostics | `ufactory.manipulation.packaging` |
| RL configuration and safe artifacts | `ufactory.training` |

```python
from ufactory.config import load_runtime_config
from ufactory.safety import SafetyGate, ApprovedProgram
from ufactory.trajectory import preflight_program, execute_real
```

Old root modules such as `ufactory.paths`, `ufactory.robot_params`,
`ufactory.kinematics_validation`, `ufactory.real_robot_session`, and
`ufactory.dynamics_validation` were removed in v0.2.0.

## Real-Robot Kinematic Calibration (SN Rules)

Per-unit firmware calibration eligibility (SN positions 3–6, four-digit model code):

| Model | SN code | Compensation |
|-------|---------|--------------|
| xArm 5/6/7 | `< 1304` | **None** — use nominal URDF only |
| xArm 5/6/7 | `≥ 1304` (e.g. 1305) | Extract YAML from this unit |
| Lite6 | `< 1006` | **None** |
| Lite6 | `≥ 1006` | Extract YAML from this unit |
| UF850 | any | **Always** |

Example SN: `XI130506XXXXXX` → model code `1305` (xArm6, calibration required).

```bash
python scripts/gen_kinematics_params.py <ip>   # suffix defaults to last 6 SN chars; skips old SNs
python examples/kinematics/verify_fk.py --robot xarm6 --ip <ip>
python examples/kinematics/verify_ik.py --robot lite6 --ip <ip>
dynamics-sim-check --robot xarm6 --random-count 5
dynamics-hardware-check --robot xarm6 --ip <ip> --confirm-real
dynamics-sim-collision-check --robot xarm6 --ip <ip>   # simulation-mode chained self-collision pre-check
# Other robots: swap --robot (xarm5 / xarm7 / uf850 / lite6).
```

Default dynamics validation poses for UF850 / Lite6 / xArm5 / xArm7 come from
[`assets/configs/dynamics_validation_pose.yaml`](assets/configs/dynamics_validation_pose.yaml)
(20 uniformly interpolated points per robot). Read and extend them via
`ufactory.dynamics.poses_config`.

## xArm 6

xArm 6 remains the reference robot, but v0.2.12 does not publish per-robot wrapper directories. Select it through the task-oriented entries, for example `examples/visualization/view_robot.py --robot xarm6` or `examples/packaging/run.py --robot xarm6 ...`. See [examples/README.md](examples/README.md) for the complete v0.2.6 path migration table.

## Project Layout

```
ufactory/config/          # Versioned runtime YAML loading and hashes
ufactory/safety/          # SafetyGate, ApprovedProgram, collision/timing ports
ufactory/cli/             # Console entry points (pick-place, packaging)
ufactory/quality/         # Local project-check reports
ufactory/robots/          # Registry, asset paths, runtime profiles
ufactory/kinematics/      # Calibration and FK/IK validation helpers
ufactory/dynamics/        # Dynamics simulation, validation, reports, CLIs
ufactory/hardware/        # xArm SDK/session and hold-current observation
ufactory/grippers/        # Gripper conversions and controllers
ufactory/trajectory/      # Trajectory profiles, segments, sim/real executors
ufactory/manipulation/    # Task frame and reusable packaging helpers
ufactory/simulation/      # Shared Genesis runtime ownership
ufactory/training/        # Scenarios, acceptance profiles, safe checkpoint YAML/manifest helpers
ufactory/visualization/   # GLB/PBR visualization helpers
assets/                   # URDF, mesh, config, and scene assets
examples/                 # Visualization, kinematics, pick-place, packaging, and rl/pick_place entries
scripts/                  # User helpers + maintainer tools (see scripts/README.md)
tests/                    # Contributor pytest suite
```

Private experiment archives stay under the ignored root `/rl/`; only `examples/rl/` is public.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding style, asset pipeline, and pull request process.

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md) code of conduct.

## License

- **Code** in this repository is MIT — see [LICENSE](LICENSE).
- **Robot URDF / mesh assets** include material derived from upstream UFACTORY / xArm ROS packages; see [NOTICE](NOTICE) for attribution and license notes. Do not treat the entire `assets/` tree as original MIT-only content.

## Citation

If you use genesis-ufactory in your research, please cite:

```bibtex
@misc{genesis-ufactory,
  author = {Wang, Daniel},
  title = {genesis-ufactory: UFACTORY Robot Models for Genesis Simulation},
  year = {2026},
  note = {Preview 0.2.x on personal GitHub; planned 0.3.x move to a UFACTORY organization repository},
  publisher = {GitHub},
  url = {https://github.com/DanielWang123321/ufactory_genesis}
}
```
