# ufactory_genesis

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12%20%7C%203.13-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/version-0.2.6-orange" alt="Version">
  <img src="https://img.shields.io/badge/genesis-1.2.2-lightgrey" alt="Genesis">
</p>

UFACTORY robot models and Genesis simulation utilities — high-fidelity GLB visualization, kinematic calibration, trajectory grasp-place examples, and RL environments.

[中文](README.zh.md) | [Contributing](CONTRIBUTING.md) | [Changelog](CHANGELOG.md) | [Security](SECURITY.md)

## Table of Contents

- [Quick Start](#quick-start)
- [Supported Robots](#supported-robots)
- [GLB Visual Preview](#glb-visual-preview)
- [Trajectory Grasp-Place](#trajectory-grasp-place)
- [Showcase](#showcase-yaml-driven-packaging)
- [API Quick Reference](#api-quick-reference)
- [Real-Robot Kinematic Calibration](#real-robot-kinematic-calibration-sn-rules)
- [xArm 6 — Reference Robot](#xarm-6)
- [Project Layout](#project-layout)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Quick Start

v0.2.6 is source-only: clone this repository and use an editable install. Wheels, sdists, remote asset downloads, and installation outside a Git checkout are unsupported. Missing repository assets fail with an actionable `AssetLayoutError`. Genesis World 1.2.2 or newer is required; the validated stack is Python 3.13, Genesis World 1.2.2, and PyTorch 2.10.0+cu128. Newer Genesis releases are allowed only when the runtime compatibility hooks remain available and are not physics- or hardware-validated until the full project matrix passes.

```bash
# From a cloned repository
pip install -e ".[sim]"

# Optional extras:
#   pip install -e ".[real]"      # xArm SDK + Pinocchio/Coal safety backends
#   pip install -e ".[rl]"        # RL training/evaluation examples
#   pip install -e ".[showcase]"  # packaging showcase scipy dependency

export NUMBA_CACHE_DIR=~/.cache/numba

# Preview xArm 6 GLB model
python examples/view_robot_glb.py --robot xarm6

# Local quality report (does not replace pytest / release evidence)
project-check
```

Since 2024, new xArm shipments use the **XI1305** hardware revision. Short names `xarm5` / `xarm6` / `xarm7` resolve to `xarm5_1305` / `xarm6_1305` / `xarm7_1305`. The explicit `*_1305` keys remain supported. Older model codes (11, 12, 1300–1304) are not bundled — supply your own URDF via `--urdf` or `prepare_robot_model_for_verification(robot_model=...)`.

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

Unified entry point `examples/view_robot_glb.py` for all robot profiles and end effectors:

```bash
export NUMBA_CACHE_DIR=~/.cache/numba

# Arm only
python examples/view_robot_glb.py --robot <profile_key>

# Gripper G2 (static / movable open-close)
python examples/view_robot_glb.py --robot xarm6 --gripper-g2
python examples/view_robot_glb.py --robot xarm6 --gripper-g2 --movable --gripper-demo

# Bio Gripper G2 (static)
python examples/view_robot_glb.py --robot uf850 --bio-gripper-g2

# Lite6 parallel gripper (static / movable open-close)
python examples/view_robot_glb.py --robot lite6 --lite6-gripper
python examples/view_robot_glb.py --robot lite6 --lite6-gripper --movable --gripper-demo

# Lite6 vacuum gripper (static)
python examples/view_robot_glb.py --robot lite6 --lite6-vacuum-gripper
```

Per-model `view_*_glb.py` scripts (e.g. `examples/xarm6/view_xarm6_glb.py`) are thin wrappers around `view_robot_glb.py --robot <key>`; the xArm6 script also adds `--diagnose`.

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

## Trajectory Grasp-Place

The configuration-driven v0.2.6 entry point supports all five robot families. `dry-run` performs calibrated FK, whole-program timing and Pinocchio/Coal collision checks without connecting to a controller.

| Robot | Command | Notes |
|-------|---------|-------|
| xArm5 | `ufactory-grasp-place --robot xarm5 --mode dry-run --executor servo_j` | Both executors |
| xArm6 | `ufactory-grasp-place --robot xarm6 --mode dry-run --executor servo_j` | Both executors |
| xArm7 | `ufactory-grasp-place --robot xarm7 --mode dry-run --executor servo_j` | Both executors |
| UF850 | `ufactory-grasp-place --robot uf850 --mode dry-run --executor servo_j` | Both executors |
| Lite6 | `ufactory-grasp-place --robot lite6 --mode dry-run --executor servo_j` | Binary real gripper capability |

```bash
# Resolve configuration only; no Genesis or robot connection.
ufactory-grasp-place --robot xarm6 --mode dry-run --executor servo_j --print-config

# Offline predictive preflight.
ufactory-grasp-place --robot xarm6 --mode dry-run --executor servo_j

# Real motion requires strict per-unit calibration and explicit confirmation.
# The task program is planned from default_qpos. If the arm is away from that
# start, --confirm-real first prepositions with MODE_POSITION set_servo_angle,
# then switches to servo streaming.
XARM_IP=192.168.1.xx ufactory-grasp-place --robot xarm6 --mode real \
  --executor servo_j --calibration path/to/exact.yaml --confirm-real

# Real + kinematic mirror window (open-loop teleport, async, off the servo path).
XARM_IP=192.168.1.xx ufactory-grasp-place --robot xarm6 --mode real \
  --executor servo_j --calibration path/to/exact.yaml --confirm-real --visual
```

`--visual`: with `--mode sim`, force the Genesis viewer; with `--mode real`, open the kinematic mirror (no contact physics). The generic grasp-place mirror uses capped non-blocking updates; the packaging command runs its full-rate Genesis/GLB viewer in a separate process, consuming all 50 Hz mirror states with the normal 60 Hz repaint rate without sharing the servo sender's Python scheduler. The window remains open until it is closed or Ctrl+C is pressed. Not supported for `dry-run` / `sdk-sim`.

Online real-policy deployment and random-action real modes have been hard-disabled since v0.2.5 (see [SECURITY.md](SECURITY.md)). Training, simulation evaluation, static FK alignment, and offline action preflight remain available.

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

# Compatibility entry: delegates to the generic simulator with --robot xarm6.
python examples/xarm6/xarm6_g2_showcase.py
```

`servo_j` startup first builds the Genesis IK scene, then preflights the complete trajectory. `[ik-compile]` and `[preflight]` messages show the active phase, sample count, and timing; hardware motion remains unauthorized until `preflight=PASS`. Genesis's neutral self-collision filtering message and Quadrants' `ast.keyword(..., ctx=...)` Python 3.15 deprecation message are upstream diagnostics, not failures in the validated Python 3.13 / Genesis 1.2.2 stack. Collision preflight still checks every trajectory sample and geometry pair, but uses the configured 5 mm security margin to select candidates and computes exact distances only for those candidates; unsupported backends automatically retain the full-distance fallback.

| Flag | Description |
|------|-------------|
| `--robot` | `xarm5`, `xarm6`, `xarm7`, `uf850`, or `lite6` |
| `--table-height` | Simulation display height only; base-frame packaging geometry is unchanged |
| `--speed` | Simulation playback multiplier (>1 is faster) |
| `--cycles N` | Run exactly N simulation cycles (default: 1) |
| `--loop` / `--no-loop` | Explicit infinite loop / compatibility alias for one cycle |

Every repeated simulation cycle restores the cube position, identity orientation, and zero velocity before the next grasp. A failed lift, placement, or home return stops further cycles. Real mode requires exact per-unit calibration and explicit confirmation and never loops; `servo_cartesian` additionally obtains matching same-process SDK-simulation evidence. xArm5, xArm7, and UF850 `--mode real` are rejected before controller connection rather than silently skipping the gripper.

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
| Policy deployment helpers | `ufactory.deploy` (online real policy hard-disabled since v0.2.5) |

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
python examples/fk_verify_robot.py --robot xarm6 --ip <ip>
python examples/ik_verify_robot.py --robot lite6 --ip <ip>
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

xArm 6 is the reference robot in this repo, with compatibility wrappers under `examples/xarm6/`. New generic entry points prefer `--robot`, for example `examples/view_robot_glb.py --robot xarm6 --diagnose` and `examples/packaging_showcase.py --robot xarm6 --gripper-g2`. Online real-policy and random-action modes in `examples/xarm6/xarm6_reach_deploy.py` have been hard-disabled since v0.2.5; align and offline preflight paths remain available.

## Project Layout

```
ufactory/config/          # Versioned runtime YAML loading and hashes
ufactory/safety/          # SafetyGate, ApprovedProgram, collision/timing ports
ufactory/cli/             # Console entry points (grasp-place, packaging)
ufactory/quality/         # Local project-check reports
ufactory/robots/          # Registry, asset paths, runtime profiles
ufactory/kinematics/      # Calibration and FK/IK validation helpers
ufactory/dynamics/        # Dynamics simulation, validation, reports, CLIs
ufactory/hardware/        # xArm SDK/session and hold-current observation
ufactory/grippers/        # Gripper conversions and controllers
ufactory/trajectory/      # Trajectory profiles, segments, sim/real executors
ufactory/manipulation/    # Task frame helpers
ufactory/simulation/      # Shared Genesis runtime ownership
ufactory/training/        # Safe checkpoint YAML/manifest helpers
ufactory/visualization/   # GLB/PBR visualization helpers
ufactory/deploy/          # Policy helpers (online real policy hard-disabled)
assets/                   # URDF, mesh, config, and scene assets
examples/                 # Usage examples (viewer, FK/IK, RL)
scripts/                  # User-facing helper scripts
tests/                    # Contributor pytest suite
```

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding style, asset pipeline, and pull request process.

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md) code of conduct.

## License

MIT — see [LICENSE](LICENSE) for details.

## Citation

If you use genesis-ufactory in your research, please cite:

```bibtex
@misc{genesis-ufactory,
  author = {UFACTORY},
  title = {genesis-ufactory: UFACTORY Robot Models for Genesis Simulation},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/DanielWang123321/ufactory_genesis}
}
```
