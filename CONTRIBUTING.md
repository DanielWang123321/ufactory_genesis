# Contributing to genesis-ufactory

Thanks for your interest in contributing! This guide explains how to set up your development environment and contribute changes.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/DanielWang123321/ufactory_genesis.git
cd ufactory_genesis

# Create a virtual environment (Python >= 3.12 required)
python -m venv .venv
source .venv/bin/activate

# Recommended contributor install (validated Genesis baseline + test tools)
pip install -e ".[sim,dev]"

# Optional: Pinocchio for dynamics unit tests in the fast tier
pip install -e ".[dynamics,dev]"

# Optional: real robot / RL library APIs / showcase workflows
pip install -e ".[real]"
pip install -e ".[sim,rl]"   # ufactory.training (no public examples/rl tree in v0.2.7)
pip install -e ".[showcase]"
```

The `[sim]` extra requires `genesis-world>=1.2.2` (see also `requirements.txt`), while the tracked lock file and validated simulation baseline use 1.2.2. Newer releases must pass runtime hook checks and the full simulation/hardware matrix before they are considered validated. Install Genesis from PyPI or follow platform notes at https://genesis-world.readthedocs.io/.

The `tests/` suite is for **contributors and maintainers**, not end-user onboarding. Library users should start from `examples/` and the dynamics CLI entry points.

Local quality reports (not a substitute for hardware evidence):

```bash
project-check fast
```

## Running Tests

Three tiers (orthogonal pytest markers):

| Tier | Command | When |
|------|---------|------|
| **fast** (default for PRs) | `pytest -m "not hardware and not gpu and not integration and not display"` | Daily development; completes in seconds |
| **sim** | `pytest -m "not hardware"` | Pre-release; requires GPU + Genesis; may take tens of minutes and write `logs/xarm6-*` |
| **hardware** | `XARM_IP=<robot-ip> pytest -m hardware` | Real robot acceptance |

```bash
# Fast tier (default PR checks)
pytest -m "not hardware and not gpu and not integration and not display"

# Full simulation regression (no real robot)
pytest -m "not hardware"

# Hardware acceptance (requires xArm SDK + robot on the network)
XARM_IP=192.168.1.xx pytest -m hardware
```

Online real-policy deployment, random-action real modes, and the `ufactory.deploy` package were removed in v0.2.7; see [SECURITY.md](SECURITY.md).

### Markers

| Marker | Meaning |
|--------|---------|
| `hardware` | Real UFACTORY robot + xArm SDK (`XARM_IP`) |
| `gpu` | In-process Genesis GPU simulation |
| `integration` | Subprocess smoke tests (`examples/` public entries) |
| `display` | Visual subprocess dry-run (`DISPLAY` required) |

### Layout

Tests are grouped under `tests/` by `ufactory` module:

| Subdirectory | Module | Examples |
|--------------|--------|----------|
| `tests/robots/` | `ufactory.robots`, kinematics SN, assets | `test_robot_registry.py`, `test_asset_integrity.py` |
| `tests/config/` | `ufactory.config` | `test_runtime_config.py` |
| `tests/safety/` | `ufactory.safety` | `test_gate_and_approval.py`, `test_models_v025.py` |
| `tests/kinematics/` | `ufactory.kinematics` | `test_pinocchio_pose_ik.py`, `test_strict_calibration.py` |
| `tests/dynamics/` | `ufactory.dynamics` | `test_dynamics_validation.py`, `test_dynamics_sim_regression.py` |
| `tests/hardware/` | `ufactory.hardware` | `test_real_robot_session.py`, `test_xarm6_smoke.py` |
| `tests/trajectory/` | `ufactory.trajectory` | `test_trajectory_profile.py` |
| `tests/training/` | `ufactory.training` | `test_artifacts.py` |
| `tests/visualization/` | `ufactory.visualization` | `test_pbr_scope.py` |
| `tests/manipulation/` | `ufactory.manipulation` | `test_pick_place_contract.py` |

- Unmarked files — unit / mock tests (fast tier)
- `test_*_smoke.py` — integration smoke tests
- `tests/dynamics/test_dynamics_*.py` — dynamics validation helpers and regressions

## Code Style

- Python >= 3.12 with `from __future__ import annotations`
- Use type hints on public functions
- Keep docstrings concise but informative
- Follow the existing patterns in the codebase

## Asset Policy

The public release contains final URDF, STL/OBJ collision meshes, and GLB visual meshes needed by users and tests. Raw/source GLBs, relocalize metrics, and vendor/combo generation scripts are internal maintenance material and are ignored from the public source tree.

Before release, run the fast tier plus the asset integrity tests. They verify every public URDF mesh reference resolves and prevent raw/source GLB intermediates from returning to the public file set.

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `ufactory/config/` | Versioned runtime YAML loading, validation, and hashes |
| `ufactory/safety/` | SafetyGate, ApprovedProgram, collision/timing ports |
| `ufactory/cli/` | Console entry points (`ufactory-pick-place`, packaging) |
| `ufactory/quality/` | Local `project-check` reports |
| `ufactory/robots/` | Robot registry, asset paths, and runtime profiles |
| `ufactory/kinematics/` | Calibration and Genesis FK/IK validation helpers |
| `ufactory/dynamics/` | Dynamics simulation, validation, reports, and CLIs |
| `ufactory/hardware/` | xArm SDK/session helpers and hold-current observation |
| `ufactory/grippers/` | Gripper command conversions and controllers |
| `ufactory/trajectory/`, `ufactory/manipulation/` | Trajectory planning/preflight/execution and reusable task helpers |
| `ufactory/simulation/` | Shared Genesis runtime ownership |
| `ufactory/training/` | RL task configuration, action scaling, and safe checkpoint artifacts |
| `ufactory/visualization/` | GLB/PBR visualization helpers |
| `assets/urdf/` | Robot and gripper URDFs + mesh files |
| `assets/configs/runtime/` | Versioned robot/task/safety/motion YAML |
| `examples/` | Task-oriented visualization, kinematics, pick_place, and packaging entries |
| `scripts/` | User-facing helper scripts |
| `tests/` | Contributor pytest suite (not required for library use) |

## Pull Request Process

1. Fork the repository and create a feature branch
2. Make your changes, following the code style
3. Run the **fast** tier to verify no regressions:
   `pytest -m "not hardware and not gpu and not integration and not display"`
4. Update documentation if needed
5. Submit a PR with a clear description of the change

## Questions?

Open an issue on GitHub.
