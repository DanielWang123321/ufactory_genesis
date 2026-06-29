# Changelog

All notable changes to genesis-ufactory will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.5] — 2026-06-29

### Added

- **`ufactory.robot_params`**: typed runtime profiles (joint PD, torque limits, dynamics poses, task capabilities)
- **`ufactory.dynamics_validation`**: Genesis PD hold vs real-robot static torque validation; CLIs `dynamics-sim-check`, `dynamics-hardware-check`, `dynamics-sim-collision-check`, `dynamics-report-compare`
- **`ufactory.real_robot_session`** and **`ufactory.xarm_control`**: real-robot connection, rad/rad/s motion, hold sampling
- **`ufactory.dynamics_pose_selection`** and **`scripts/select_dynamics_calib_poses.py`**: EE y hemisphere stratified calibration pose selection
- **`ufactory.kinematics_validation`**: shared FK/IK verification CLI for generic robot examples
- Generic examples: `packaging_showcase.py`, `fk_verify_robot.py`, `ik_verify_robot.py`, `arm_reach_env.py`, `grasp_place_env.py`; expanded test suite

### Changed

- **xArm6 dynamics poses**: `home` + 20 `calib_*` points (10 y+ / 10 y−) from absolute-accuracy calibration file, replacing legacy `config_*` presets
- **Real-robot motion units**: unified **rad / rad/s**; default speed ≈0.698 rad/s (40°/s equivalent)
- **Default move strategy**: `direct` (axis-sequential caused self-collision on several calibration poses)
- Multi-robot examples slimmed to profile-driven thin wrappers

### Removed

- Public **`docs/`** directory (content moved to local `dev/` workspace)
- Two misnamed xarm6 cached calibration URDFs under `assets/urdf/xarm6/`

### Fixed

- **`speed=40` with `is_radian=True`** was interpreted as 40 rad/s and clamped to π rad/s (~180°/s); default now correctly uses `math.radians(40.0)`

## [0.1.3] — 2026-06-25

### Changed

- **Minimum Genesis World version** raised to 1.2.0 (`ViewerOptions.max_FPS` → `refresh_rate`)

### Fixed

- **Bio Gripper G2 on xArm7 (link7)**: reject mirrored pin-hole solution that sank the static GLB into the flange in Genesis preview
- **Regenerated** link5/6/7 Bio G2 visual GLBs and `relocalize_metrics.json`; updated uf850 movable attach origin

## [0.1.2] — 2026-06-22

### Added

- **`BioGripperG2` controller module** (`ufactory/bio_gripper_g2.py`) for reusable open/close control across all supported arms
- **Per-robot `bio_gripper_g2_attach` origins** computed during relocalize and written into combo URDFs
- **`robot_cli_choices()`** and short-name aliases: `xarm5` / `xarm6` / `xarm7` resolve to `*_1305` profiles
- **`tests/test_robot_registry.py`** for profile resolution and default URDF paths

### Changed

- **Bio Gripper G2 movable mount** uses canonical EE pin-hole alignment (fixes UF850 flange inversion and finger +X orientation)
- **Regenerated** Bio Gripper G2 combo URDFs, per-link GLBs, and `relocalize_metrics.json`
- **Default xArm URDF** paths now point to `*_1305.urdf` (`xarm6_urdf()` and friends); CLI/docs recommend short names `xarm6` etc.

### Fixed

- UF850 Bio Gripper G2 movable attach (`ring_gap_mm=0`, fingers toward base +X)
- `verify_bio_gripper_g2_assets.py` world-frame finger-direction checks for movable combos

## [0.1.1] — 2026-06-18

### Changed

- **Rename `bio_gripper` → `bio_gripper_g2`** across all assets, URDFs, examples, and scripts
- **Fix Bio Gripper G2 flange orientation** when mounted on robot arm (link5/link6/link7)

### Fixed

- `xarm6_1305_visual_glb_urdf()` now accepts `with_bio_gripper_g2` parameter (consistent with xarm5/xarm7/uf850)
- README `--gripper-demo` flag table now includes Bio Gripper G2

### Removed

- Diagnostic and keyframe-capture scripts moved from `scripts/` and `examples/xarm6/` to `dev/diagnostics/` (gitignored)
- Lerobot experimental code moved to `dev/lerobot/` (gitignored)

## [0.1.0] — 2026-06-18

### Added

- **Robot profiles:** xArm 5/6/7 (1305 variant), UF850, Lite6 with `RobotModelSpec` registry
- **End-effector support:** Gripper G2, Bio Gripper G2 (xArm/UF850), Lite6 Gripper, Lite6 Vacuum
- **GLB visual rendering** with PBR material preservation (metallic/roughness) via Genesis monkey-patching
- **Unified GLB viewer** (`examples/view_robot_glb.py`) supporting all robots and accessories
- **xArm6 reference verification:** FK/IK comparison with real robot, dynamics validation
- **Kinematic calibration:** Per-unit URDF patching from firmware YAML (SN-based eligibility)
- **RL environments:** Reach and grasp-place tasks for xArm6 (rsl-rl-lib)
- **Showcase scene:** Physical pick-place demo (xArm6 + Gripper G2 + cardboard box)
- **Multi-robot smoke tests** (headless, no hardware required)
