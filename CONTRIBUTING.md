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

# Install Genesis (platform-specific: https://genesis-world.readthedocs.io/)
pip install "genesis-world>=1.2.0"

# Install in editable mode with contributor test tools
pip install -e ".[dev]"

# Optional: Pinocchio for dynamics unit tests in the fast tier
pip install -e ".[dynamics,dev]"

# Optional: real robot / RL / showcase workflows
pip install -e ".[real]"
pip install -e ".[rl]"
pip install -e ".[showcase]"
```

The `tests/` suite is for **contributors and maintainers**, not end-user onboarding. Library users should start from `examples/` and the dynamics CLI entry points.

## Running Tests

Three tiers (orthogonal pytest markers):

| Tier | Command | When |
|------|---------|------|
| **fast** (default for PRs) | `pytest -m "not hardware and not gpu and not integration and not display"` | Daily development; completes in seconds |
| **sim** | `pytest -m "not hardware"` | Pre-release; requires GPU + Genesis; may take tens of minutes and write `logs/xarm6-*` |
| **hardware** | `XARM_IP=<robot-ip> pytest -m hardware` | Real robot acceptance |

```bash
# Fast tier (default PR gate)
pytest -m "not hardware and not gpu and not integration and not display"

# Full simulation regression (no real robot)
pytest -m "not hardware"

# Hardware acceptance (requires xArm SDK + robot on the network)
XARM_IP=192.168.1.65 pytest -m hardware
```

### Markers

| Marker | Meaning |
|--------|---------|
| `hardware` | Real UFACTORY robot + xArm SDK (`XARM_IP`) |
| `gpu` | In-process Genesis GPU simulation |
| `integration` | Subprocess smoke tests (`examples/`, short RL runs) |
| `display` | Visual subprocess dry-run (`DISPLAY` required) |

### Layout

- Unmarked files — unit / mock tests (fast tier)
- `test_*_smoke.py` — integration smoke tests
- `test_dynamics_*.py` — dynamics validation helpers and regressions

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
| `ufactory/` | Core Python package (robot registry, paths, kinematics, GLB visuals) |
| `assets/urdf/` | Robot and gripper URDFs + mesh files |
| `examples/` | Usage examples (viewer, FK/IK verification, RL) |
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
