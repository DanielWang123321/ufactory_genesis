# ufactory_genesis

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12%20%7C%203.13-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/version-0.1.3-orange" alt="Version">
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

Tested with Python 3.13, Genesis ≥1.2.0, PyTorch 2.12.

```bash
# 1. Install Genesis (platform-specific: CPU / CUDA / macOS / AMD)
#    Follow the official guide: https://genesis-world.readthedocs.io/
pip install "genesis-world>=1.2.0"

# 2. Install ufactory_genesis
pip install -r requirements.txt
pip install -e .

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

```python
import ufactory
```

### Robot Registry

| Function / Object | Description |
|-------------------|-------------|
| `ufactory.ROBOT_PROFILES` | Dict of all supported robot `RobotModelSpec` entries |
| `ufactory.get_robot_profile(key)` | Get `RobotModelSpec` by profile key or short name (`xarm6`) |
| `ufactory.get_profile_key_for_robot_name(name)` | Resolve robot name to profile key (`xarm6` → `xarm6_1305`) |
| `ufactory.robot_cli_choices()` | Sorted `--robot` choices (keys + short aliases) |
| `ufactory.arm_link_names(profile)` | Link name tuple for a robot profile |
| `ufactory.joint_names(profile)` | Joint name tuple for a robot profile |

### Paths

#### Core (generic APIs)

| Function | Description |
|----------|-------------|
| `ufactory.robot_urdf(key, name=None)` | Absolute path to default URDF by profile key or short name |
| `ufactory.robot_visual_glb_urdf(key, with_*=..., movable=...)` | URDF with GLB visuals; end-effector flags are mutually exclusive — see source for `movable` constraints |
| `ufactory.robot_assets(name)` | `Path` to robot asset directory |
| `ufactory.kinematics_user_dir(robot_name)` | Per-unit calibration YAML directory: `assets/urdf/<robot>/kinematics/user/` |

```python
# Recommended: generic entry points (same as examples with --robot xarm6)
ufactory.robot_urdf("xarm6")
ufactory.robot_visual_glb_urdf("xarm6", with_gripper_g2=True, movable=True)
```

#### Arm convenience (xArm 5/6/7)

| Function | Description |
|----------|-------------|
| `ufactory.xarm5_urdf(name="xarm5_1305.urdf")` | Default xArm 5 URDF |
| `ufactory.xarm6_urdf(name="xarm6_1305.urdf")` | Default xArm 6 URDF |
| `ufactory.xarm7_urdf(name="xarm7_1305.urdf")` | Default xArm 7 URDF |
| `ufactory.xarm5_1305_urdf()` | Thin wrapper around `robot_urdf("xarm5_1305")` |
| `ufactory.xarm6_1305_urdf()` | Same as `xarm6_urdf()` |
| `ufactory.xarm7_1305_urdf()` | Thin wrapper around `robot_urdf("xarm7_1305")` |
| `ufactory.xarm5_1305_visual_glb_urdf(with_bio_gripper_g2=False)` | xArm 5 GLB visual URDF |
| `ufactory.xarm6_1305_visual_glb_urdf(with_bio_gripper_g2, with_gripper_g2, movable)` | xArm 6 GLB visual URDF (Gripper G2 / Bio G2 / movable) |
| `ufactory.xarm7_1305_visual_glb_urdf(with_bio_gripper_g2=False)` | xArm 7 GLB visual URDF |

#### Lite6 convenience

| Function | Description |
|----------|-------------|
| `ufactory.lite6_urdf()` | Default Lite6 URDF |
| `ufactory.lite6_visual_glb_urdf(with_lite6_gripper, with_lite6_vacuum_gripper, movable)` | Lite6 GLB visual URDF with gripper options |
| `ufactory.lite6_with_gripper_urdf()` | Physics combo URDF with parallel gripper |
| `ufactory.lite6_with_vacuum_gripper_urdf()` | Physics combo URDF with vacuum gripper |
| `ufactory.lite6_gripper_movable_visual_urdf()` | Standalone gripper-only movable visual URDF (no arm) |

#### UF850 convenience

| Function | Description |
|----------|-------------|
| `ufactory.uf850_urdf()` | Default UF850 URDF |
| `ufactory.uf850_visual_glb_urdf(with_bio_gripper_g2=False)` | UF850 GLB visual URDF |

#### Standalone gripper assets

| Function | Description |
|----------|-------------|
| `ufactory.gripper_g2_movable_visual_urdf()` | Gripper G2 standalone movable visual URDF |
| `ufactory.gripper_g2_static_glb(ee_link="link6")` | Gripper G2 static assembly GLB |
| `ufactory.gripper_g2_base_glb(ee_link="link6")` | Gripper G2 base GLB (movable mode) |
| `ufactory.gripper_g2_shared_glb(name)` | Gripper G2 shared link GLB |
| `ufactory.bio_gripper_g2_movable_visual_urdf()` | Bio Gripper G2 standalone movable visual URDF |
| `ufactory.bio_gripper_g2_glb(ee_link="link6")` | Bio Gripper G2 static GLB |

> Full signatures and parameter constraints: [`ufactory/paths.py`](ufactory/paths.py). Package-level exports: [`ufactory/__init__.py`](ufactory/__init__.py).

### Runtime Profiles

| Function | Description |
|----------|-------------|
| `ufactory.get_robot_runtime_profile(key)` | Typed runtime profile: arm PD, torque limits, validation poses, task capabilities |
| `ufactory.dynamics_default_configs(key)` | Profile-specific dynamics validation poses |

### Kinematic Calibration

| Function | Description |
|----------|-------------|
| `ufactory.load_kinematics_yaml(path)` | Load joint offsets from kinematics YAML |
| `ufactory.build_calibrated_urdf(base, kinematics)` | Generate URDF with calibrated joint origins |
| `ufactory.parse_sn_model_code(sn)` | Extract 4-digit model code from serial number |
| `ufactory.has_per_unit_kinematics_calibration(sn, name)` | Check if SN-based calibration applies |

### GLB PBR Visuals

| Function | Description |
|----------|-------------|
| `ufactory.enable_glb_pbr_surfaces()` | Monkey-patch Genesis to preserve PBR materials from GLB |
| `ufactory.glb_view_surface()` | Default double-sided surface for non-GLB geometries |

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
python scripts/gen_kinematics_params.py <ip> <suffix>   # skips old SNs automatically
python examples/fk_verify_robot.py --robot xarm6 --ip <ip> --kinematics-suffix <suffix>
python examples/ik_verify_robot.py --robot lite6 --ip <ip> --kinematics-suffix <suffix>
dynamics-sim-check --robot xarm6 --random-count 5
dynamics-hardware-check --robot xarm6 --ip <ip> --kinematics-suffix <suffix>
dynamics-sim-collision-check --ip <ip>   # simulation-mode chained self-collision pre-check
```

## xArm 6

xArm 6 is the reference robot in this repo, with compatibility wrappers under `examples/xarm6/`. New generic entry points prefer `--robot`, for example `examples/view_robot_glb.py --robot xarm6 --diagnose` and `examples/packaging_showcase.py --robot xarm6 --gripper-g2`.

## Project Layout

```
ufactory/             # Core Python package (robot registry, paths, kinematics, GLB)
assets/urdf/          # Robot URDFs, STL collision, GLB visual meshes
assets/scenes/        # Simulation scene assets (textures, props)
examples/             # Usage examples (viewer, FK/IK, RL)
scripts/              # Asset generation and maintenance scripts
tests/                # Pytest test suite
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
