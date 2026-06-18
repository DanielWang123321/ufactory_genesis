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
pip install genesis-world==1.1.2

# Install in editable mode
pip install -e .
```

## Running Tests

```bash
# Unit and smoke tests (no hardware required)
pytest -m "not hardware"

# All tests including hardware tests (requires real robot + xArm SDK)
pytest
```

Tests are organized by:
- `test_*_smoke.py` — End-to-end simulation smoke tests (headless, no hardware)
- `test_kinematics_sn.py` — Unit tests for kinematics calibration logic
- `test_robot_viewer_controls.py` — Unit tests for viewer control logic (mocked)
- Tests marked `@pytest.mark.hardware` — Require a real UFACTORY robot on the network

## Code Style

- Python >= 3.12 with `from __future__ import annotations`
- Use type hints on public functions
- Keep docstrings concise but informative
- Follow the existing patterns in the codebase

## Asset Pipeline

Robot URDF assets are generated via a multi-step pipeline:

1. **Vendor** — `scripts/vendor_robot_assets.py` clones `xarm_ros2` and generates base URDFs + STL meshes
2. **Relocalize** — `scripts/relocalize_*_glb.py` align CAD GLB meshes to URDF reference frames
3. **Generate Combo** — `scripts/generate_*_combo_urdf.py` create arm+gripper combined URDFs

> **Note:** Template URDFs in `assets/urdf/gripper_g2/gripper_g2.urdf` and similar files are xacro-generated artifacts — their mesh paths are replaced during combo generation. Do not load them directly.

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `ufactory/` | Core Python package (robot registry, paths, kinematics, GLB visuals) |
| `assets/urdf/` | Robot and gripper URDFs + mesh files |
| `examples/` | Usage examples (viewer, FK/IK verification, RL) |
| `scripts/` | Asset generation and maintenance scripts |
| `tests/` | Pytest test suite |
| `docs/` | Additional documentation |

## Pull Request Process

1. Fork the repository and create a feature branch
2. Make your changes, following the code style
3. Run `pytest -m "not hardware"` to verify no regressions
4. Update documentation if needed
5. Submit a PR with a clear description of the change

## Questions?

Open an issue on GitHub.
