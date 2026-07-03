# ufactory_genesis

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12%20%7C%203.13-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/version-0.2.0-orange" alt="Version">
  <img src="https://img.shields.io/badge/genesis-1.2.0%2B-lightgrey" alt="Genesis">
</p>

UFACTORY robot models and Genesis simulation utilities — high-fidelity GLB visualization, kinematic calibration, and RL environments.

[中文](README.zh.md) | [Contributing](CONTRIBUTING.md) | [Changelog](CHANGELOG.md)

## Table of Contents

- [Quick Start](#quick-start)
- [Supported Robots](#supported-robots)
- [GLB Visual Preview](#glb-visual-preview)
- [API Quick Reference](#api-quick-reference)
- [Real-Robot Kinematic Calibration](#real-robot-kinematic-calibration-sn-rules)
- [xArm 6 — Reference Robot](#xarm-6)
- [Project Layout](#project-layout)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Quick Start

Tested with Python 3.13, Genesis 1.2.0, PyTorch 2.10.0+cu128.

```bash
# 1. Install Genesis (platform-specific: CPU / CUDA / macOS / AMD)
#    Follow the official guide: https://genesis-world.readthedocs.io/
pip install "genesis-world>=1.2.0"

# 2. Install ufactory_genesis
pip install -r requirements.txt
pip install -e .

# Optional extras:
#   pip install -e ".[real]"      # xArm SDK / real robot commands
#   pip install -e ".[rl]"        # RL training/evaluation examples
#   pip install -e ".[showcase]"  # packaging showcase scipy dependency

export NUMBA_CACHE_DIR=~/.cache/numba

# Preview xArm 6 GLB model
python examples/view_robot_glb.py --robot xarm6
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

High-fidelity GLB rendering with PBR material preservation; collision and physics still use STL meshes. Single entry point:

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

## API Quick Reference

v0.2.0 uses functional subpackages. The root namespace is intentionally small:

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
| `ufactory.get_robot_runtime_profile(key)` | Typed runtime profile |

Advanced APIs are imported from their owning modules:

| Domain | Canonical import |
|--------|------------------|
| Robot registry, paths, runtime profiles | `ufactory.robots.registry`, `ufactory.robots.paths`, `ufactory.robots.runtime` |
| Kinematics calibration and FK/IK validation | `ufactory.kinematics.calibration`, `ufactory.kinematics.validation` |
| Dynamics simulation and hardware validation | `ufactory.dynamics` |
| Real robot SDK/session helpers | `ufactory.hardware.xarm`, `ufactory.hardware.session`, `ufactory.hardware.observe` |
| Gripper command conversions/controllers | `ufactory.grippers.g2`, `ufactory.grippers.bio_g2` |
| Trajectory and manipulation helpers | `ufactory.trajectory`, `ufactory.manipulation.frames` |
| GLB visualization helpers | `ufactory.visualization.glb` |
| Policy deployment helpers | `ufactory.deploy` |

```python
from ufactory.kinematics.calibration import build_calibrated_urdf
from ufactory.dynamics import dynamics_default_configs
from ufactory.grippers.g2 import gripper_g2_gap_m_to_sdk_pos_mm
from ufactory.visualization.glb import enable_glb_pbr_surfaces
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

Example SN: `XI130506D43A0A` → model code `1305` (xArm6, calibration required).

```bash
python scripts/gen_kinematics_params.py <ip>   # suffix defaults to last 6 SN chars; skips old SNs
python examples/fk_verify_robot.py --robot xarm6 --ip <ip>
python examples/ik_verify_robot.py --robot lite6 --ip <ip>
dynamics-sim-check --robot xarm6 --random-count 5
dynamics-hardware-check --robot xarm6 --ip <ip>
dynamics-sim-collision-check --ip <ip>   # simulation-mode chained self-collision pre-check
```

## xArm 6

xArm 6 is the reference robot in this repo, with compatibility wrappers under `examples/xarm6/`. New generic entry points prefer `--robot`, for example `examples/view_robot_glb.py --robot xarm6 --diagnose` and `examples/packaging_showcase.py --robot xarm6 --gripper-g2`.

## Project Layout

```
ufactory/robots/          # Registry, asset paths, runtime profiles
ufactory/kinematics/      # Calibration and FK/IK validation helpers
ufactory/dynamics/        # Dynamics simulation, validation, reports, CLIs
ufactory/hardware/        # xArm SDK/session and hold-current observation
ufactory/grippers/        # Gripper conversions and controllers
ufactory/trajectory/      # Trajectory profiles, segments, sim/real executors
ufactory/manipulation/    # Task frame helpers
ufactory/visualization/   # GLB/PBR visualization helpers
ufactory/deploy/          # Policy deployment helpers
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
