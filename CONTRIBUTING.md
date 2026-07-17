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

# Recommended contributor install (reference Genesis baseline + test tools)
pip install -e ".[sim,dev]"

# For public-CI parity (no Genesis / no GPU):
#   pip install -e ".[dev]"
# Optional: Pinocchio for dynamics unit tests in the fast tier
pip install -e ".[dynamics,dev]"

# Optional: real robot / RL library APIs / showcase workflows
pip install -e ".[real]"
pip install -e ".[sim,rl]"   # ufactory.training (no public examples/rl tree in v0.2.8)
pip install -e ".[showcase]"
```

The `[sim]` extra requires `genesis-world>=1.2.2` (see also `requirements.txt`). The tracked lock file pins the **reference baseline** Genesis World 1.2.2. Newer releases must pass runtime hook checks and the maintainer sim/hardware checks before they are treated as a verified baseline. Install Genesis from PyPI or follow platform notes at https://genesis-world.readthedocs.io/.

0.2.x is Alpha: public APIs may break between minor versions. The first supported freeze is planned for 0.3.x after the move to a UFACTORY organization repository — see [ROADMAP.md](ROADMAP.md).

The `tests/` suite is for **contributors and maintainers**, not end-user onboarding. Library users should start from `examples/` and the dynamics CLI entry points.

Local CPU quality reports (same class as public CI; not a substitute for sim/hardware evidence):

```bash
project-check fast
```

## Running Tests

Use **`project-check`** as the single quality entry. Pytest markers remain orthogonal; do not treat raw `pytest -m "not hardware"` as the default pre-release check (it pulls in long `integration` / `slow` matrices that overlap `sdk-sim`).

| Tier | Command | When |
|------|---------|------|
| **fast** (default for PRs / daily) | `project-check fast` | Seconds to ~1 min; no GPU |
| **sim** (pre-release GPU) | `project-check sim` | `fast` + `-m "gpu and not slow"`; minutes |
| **sdk-sim** | `project-check sdk-sim --inventory <yaml>` | Five-robot cabinet evidence |
| **hardware** | `project-check hardware --inventory <yaml> --confirm-real` | Real robot acceptance |
| **release** | `project-check release --version X.Y.Z` | Snapshot/reuse fast + evidence + lock `pip-audit` (`--no-deps`, parallel with snapshot) |
| **deep** (optional maintainer) | `project-check deep` | Heavy markers only: `gpu or integration or display or slow` (run `fast` first). Target ≤15–20 min |

```bash
# Daily / before push
project-check fast

# Pre-release GPU (excludes @pytest.mark.slow)
project-check sim

# Optional deep regression (not the default release check; run fast first)
project-check deep
# equivalent: pytest -m "gpu or integration or display or slow"

# Hardware via inventory (preferred) or ad-hoc marker
# project-check hardware --inventory dev/hardware/hardware_inventory.yaml --confirm-real
XARM_IP=192.168.1.xx pytest -m hardware
```

**Parallelism boundary:** `sdk-sim` / `hardware` inventory runs **one worker per robot IP** (cabinet evidence). In-process `@gpu` multi-robot matrices stay **serial** on a single GPU (Genesis one owner per process; 8GB VRAM). Five-robot CLI smoke is **not** re-run in `deep`; use `sdk-sim`.

Online real-policy deployment, random-action real modes, and the `ufactory.deploy` package were removed in v0.2.7; see [SECURITY.md](SECURITY.md).

### Markers

| Marker | Meaning |
|--------|---------|
| `hardware` | Real UFACTORY robot + xArm SDK (`XARM_IP`) |
| `gpu` | In-process Genesis GPU simulation |
| `integration` | Subprocess smoke tests (`examples/` public entries); xArm6 representative + cheap `--print-config` |
| `display` | Visual subprocess dry-run (`DISPLAY` required) |
| `slow` | Extra multi-robot GPU samples for `deep` (e.g. five-robot packaging one cycle); excluded from `project-check sim` |

### Layout

Tests are grouped under `tests/` by `ufactory` module:

| Subdirectory | Module | Examples |
|--------------|--------|----------|
| `tests/robots/` | `ufactory.robots`, kinematics SN, assets | `test_robot_registry.py`, `test_asset_integrity.py` |
| `tests/config/` | `ufactory.config` | `test_runtime_config.py` |
| `tests/safety/` | `ufactory.safety` | `test_gate_and_approval.py`, `test_models.py` |
| `tests/kinematics/` | `ufactory.kinematics` | `test_pinocchio_pose_ik.py`, `test_strict_calibration.py` |
| `tests/dynamics/` | `ufactory.dynamics` | `test_dynamics_validation.py`, `test_dynamics_sim_regression.py` |
| `tests/hardware/` | `ufactory.hardware` | `test_real_robot_session.py`, `test_xarm6_real.py` |
| `tests/trajectory/` | `ufactory.trajectory` | `test_trajectory_profile.py` |
| `tests/training/` | `ufactory.training` | `test_artifacts.py` |
| `tests/visualization/` | `ufactory.visualization` | `test_pbr_scope.py` |
| `tests/manipulation/` | `ufactory.manipulation` | `test_pick_place_contract.py` |

- Unmarked files — unit / mock tests (fast tier)
- `tests/robots/test_representative_example_smoke.py` — xArm6 representative integration smoke (examples CLI)
- `tests/dynamics/test_dynamics_*.py` — dynamics validation helpers and regressions

## Code Style

- Python >= 3.12 with `from __future__ import annotations`
- Use type hints on public functions
- Keep docstrings concise but informative
- Follow the existing patterns in the codebase

## Asset Policy

The public release contains final URDF, STL meshes, and GLB visual meshes needed by users and tests. Registered arm URDFs use `meshes/<variant>/visual/*.stl` for link collision (not OBJ; public `collision/**/*.obj` is forbidden). Per-arm `collision/*.stl` shells are historical/vendor comparison assets. Raw/source GLBs, relocalize metrics, and vendor/combo generation scripts are internal maintenance material and are ignored from the public source tree.

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
| `scripts/` | User helpers and maintainer tools (see `scripts/README.md`) |
| `tests/` | Contributor pytest suite (not required for library use) |

## Pull Request Process

1. Fork the repository and create a feature branch
2. Make your changes, following the code style
3. Run the **fast** tier to verify no regressions (must match public CI markers):
   `pytest -m "not slow and not hardware and not gpu and not integration and not display"`
   or simply `project-check fast` when `[sim,dev]` (or at least `[dev]`) is installed
4. Update documentation if needed
5. Submit a PR with a clear description of the change

## Questions?

Open an issue on GitHub.
