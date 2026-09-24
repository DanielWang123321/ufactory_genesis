# ufactory_genesis

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12%20%7C%203.13-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/version-0.3.0-orange" alt="Version">
  <img src="https://img.shields.io/badge/genesis-1.4.2-lightgrey" alt="Genesis">
  <a href="https://github.com/xArm-Developer/ufactory_genesis/actions/workflows/ci.yml"><img src="https://github.com/xArm-Developer/ufactory_genesis/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

UFACTORY robot models and Genesis simulation utilities for visualization, kinematics, trajectory execution, packaging, and simulation-only reinforcement learning (RL).

[中文](README.zh.md) | [Examples](examples/README.md) | [Changelog](CHANGELOG.md) | [Roadmap](ROADMAP.md) | [Security](SECURITY.md)

> v0.3.0 is the official repository under [xArm-Developer](https://github.com/xArm-Developer/ufactory_genesis). This project is still 0.x: entry points may be renamed or removed before 1.0.0. Genesis World is validated at 1.4.2 only.

## Supported scope

- Python 3.12/3.13. Genesis World 1.4.2 is the validated baseline. Newer Genesis releases are not guaranteed; hook checks fail closed when they do not match.
- Source checkout with editable install only; wheels, sdists, and remote asset downloads are unsupported.
- Visualization, calibrated FK/IK, contact simulation, dry-run, controller simulation, and guarded real execution.
- Real packaging is enabled only for xArm6 + Gripper G2 and Lite6 + Lite6 Gripper.

| Robot | Available end effectors |
|--------|-------------------------|
| `xarm5` | Gripper G2, Bio Gripper G2 |
| `xarm6` | Gripper G2, Bio Gripper G2 |
| `xarm7` | Gripper G2, Bio Gripper G2 |
| `uf850` | Gripper G2, Bio Gripper G2 |
| `lite6` | Lite6 Gripper, Lite6 Vacuum Gripper |

## Install

```bash
git clone https://github.com/xArm-Developer/ufactory_genesis.git
cd ufactory_genesis
pip install -e ".[sim]"
export NUMBA_CACHE_DIR=~/.cache/numba
python examples/visualization/view_robot.py --robot xarm6
```

| Goal | Editable install |
|------|------------------|
| Simulation and visualization | `pip install -e ".[sim]"` |
| Fixed-layout RL | `pip install -e ".[sim,rl]"` |
| Real-robot safety backends | `pip install -e ".[sim,real]"` |
| Packaging texture generation | `pip install -e ".[sim,showcase]"` |
| Dynamics reference checks | `pip install -e ".[sim,dynamics]"` |

For Windows or CPU-only simulation, install CPU PyTorch first and pass `--backend cpu`. RL remains Linux/NVIDIA-only; see the [examples guide](examples/README.md).

If the `view_robot.py` viewer returns to the prompt with no window after `Building visualizer...`, set `MKL_THREADING_LAYER=SEQUENTIAL` and rerun (`$env:MKL_THREADING_LAYER = "SEQUENTIAL"` in PowerShell); `--headless` is unaffected.

## Main workflows

```bash
# GLB visualization
python examples/visualization/view_robot.py --robot xarm6 --gripper-g2 --movable

# Pick-place configuration check and simulation
ufactory-pick-place --robot xarm6 --mode dry-run --executor servo_j --print-config
ufactory-pick-place --robot xarm6 --mode sim --executor servo_j --visual

# YAML-driven packaging simulation
python scripts/generate_showcase_textures.py
ufactory-packaging-showcase --robot lite6 --mode sim --executor servo_cartesian --visual
```

Full visualization, kinematics, CPU/Windows, configuration, and real-robot examples are in [examples/README.md](examples/README.md).

## Current entry points

These are what v0.3.0 runs against Genesis World 1.4.2. They are not a promise that later 0.x releases or later Genesis versions will keep them.

Commands: `ufactory-pick-place`, `ufactory-packaging-showcase`, `dynamics-sim-check`, `dynamics-sim-collision-check`, `dynamics-hardware-check`. `project-check` and `observe-hold-current` are maintainer tools.

Examples: `examples/visualization`, `examples/kinematics`, `examples/pick_place`, `examples/packaging`, `examples/rl/pick_place`, including the `runtime.example.yaml` files beside the pick-place and packaging entries.

Library exports from `ufactory`: `get_robot_profile`, `robot_urdf`, `robot_visual_glb_urdf`, `robot_assets`, `RepositoryAssetStore`.

## Fixed-layout RL in v0.3.0

The public xArm6 + Gripper G2 task uses one fixed `+Y` layout: cube `[0.300, 0, 0.015]` m and target `[0.300, 0.300, 0.015]` m. Random starts are not included in this release.

The bundled seed-7 `model_299_g2stable.pt` was retrained under the `g2_stable_v1_3_3` contact-physics profile and passed every release check: all nine evaluation seed/batch combinations (219/219 episodes), the independent 64-episode fixed bank (64/64), and 512/512 episodes with 0.02 action noise. It is not evidence of random-layout generalization and has no real-robot execution interface.

```bash
python -m examples.rl.pick_place.evaluate --expert --headless -B 1 --episodes 1
python -m examples.rl.pick_place.evaluate --headless -B 8 --episodes 8
```

See the [RL guide](examples/rl/pick_place/README.md) for training, release metrics, artifact validation, and limitations.

## Real-robot safety

Real motion requires exact per-unit calibration, complete predictive preflight, and explicit `--confirm-real`. Software checks do not replace a trained operator, an isolated workspace, risk assessment, or a physical emergency stop. Read [SECURITY.md](SECURITY.md) before connecting hardware.

## Documentation

- [Task-oriented examples](examples/README.md)
- [Contributing and project checks](CONTRIBUTING.md)
- [Release history](CHANGELOG.md)
- [Project roadmap](ROADMAP.md)
- [Security and trust boundaries](SECURITY.md)

## License and citation

Code is MIT licensed; robot URDF and mesh assets include upstream material with separate attribution in [NOTICE](NOTICE). xArm, UF850, and UFACTORY are trademarks of their respective owners.

For research use, cite “genesis-ufactory: UFACTORY Robot Models for Genesis Simulation,” Daniel Wang, 2026, `https://github.com/xArm-Developer/ufactory_genesis`.
