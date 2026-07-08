# Changelog

All notable changes to genesis-ufactory will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.2] — 2026-07-08

### Added

- Contact-friction grasp diagnostics for Gripper G2 (`g2_contact_grasp_diagnose.py`), Lite6 reversed gripper (`lite6_contact_grasp_diagnose.py`), and standalone Lite6 gripper+cube collision (`lite6_gripper_cube_diagnose.py`).
- Lite6 `place-settle` segment: a 0.18 s closed-gripper hold before release so the block opens at table height.
- `gap_lspb_samples` support for holding an identical gripper gap over a requested duration (used by `place-settle`).

### Changed

- Grasp-place simulation now defaults to rigid-body contact and friction (`sim_grasp_weld=False`); distance weld, geometric snap, block freeze, and forced block motion are removed from the default path.
- Gripper G2 default grasp gap is now 22 mm for contact preload on the 30 mm cube.
- Lite6 grasp-place uses raw STL finger collision (no Genesis convex proxy), contact-latched close after bilateral finger contact, and a 0.8 mm default sim hold bias.
- Lite6 finger visual/collision origins are reset to zero so the flat pad grips the cube instead of the finger root/limit area.

### Fixed

- Lite6 grasping on the finger limit area instead of the fingertip pad.
- Lite6 premature top-surface contact from processed convex collision proxies.
- Lite6 block sliding during lift/transit and pressing into the table during place.
- Lite6 place bounce and mid-air release after removing `place-standoff`.
- Gripper G2 over-tight preload causing visible carry jitter; default gap tuned to 22 mm.
- Lite6 headless real dry-run now includes `place-settle` because `dry_heights()` exposes gripper metadata.

## [0.2.1] — 2026-07-08

### Added

- Robot-aware LSPB trajectory planning APIs for joint, Cartesian, and mixed waypoint programs.
- Shared five-robot grasp-place trajectory examples for xArm5, xArm6, xArm7, UF850, and Lite6, with Genesis sim, dry-run, real `servo-cartesian`, and kinematic mirror paths.
- Lite6 gripper command conversion and tests for its 20-38 mm physical opening range.

### Changed

- Grasp-place scenes now use 30 mm default cubes, calibrated grasp gaps, and shared trajectory scene defaults across supported robots.
- Registered arm links for xArm5/6/7 1305, Lite6, and UF850 now use `visual/*.stl` meshes for collision to better match the physical curved link surfaces.
- Collision visualization tooling can now load arm+accessory combo URDFs with raw STL collision meshes, including Gripper G2, Bio Gripper G2, Lite6 Gripper, and Lite6 Vacuum.
- Gripper G2 and Lite6 Gripper keep their packaged STL paths because they match upstream xarm_ros2 visual STL assets; Bio Gripper G2 already uses visual STL collision, and Lite6 Vacuum now uses upstream visual STL collision.
- Asset integrity tests now assert registered arm collision references point to visual STL, validate accessory STL collision references, and still reject OBJ collision meshes.

### Fixed

- Grasp-place viewers now start from the planned home/pregrasp pose instead of briefly showing zero-position arms with grippers inside the table.
- Lite6 grasp-place now keeps finger collision/visual gaps aligned with the 30 mm cube and freezes release/place transitions to avoid pushing the cube into the table.
- Simulation final place-error reporting now handles mixed CPU/CUDA tensor devices in test doubles and Genesis contexts.

## [0.2.0] — 2026-07-03

### Added

- **Five-robot dynamics validation**: xArm5, xArm6, xArm7, Lite6, and UF850 hardware-verified against Genesis PD hold (20 calibration poses each).
- **xArm5/6/7 collision meshes**: vendor simplified collision hulls converted from OBJ to STL (geometry-equivalent; aligns with firmware STL collision format).
- **Tests reorganized by module**: `tests/robots/`, `tests/dynamics/`, `tests/hardware/`, `tests/trajectory/`, `tests/deploy/`, `tests/manipulation/`.
- **`test_public_api_layout.py`**: asserts canonical subpackage imports and that legacy root modules raise `ModuleNotFoundError`.
- **`test_xarm5_collision_mesh_equivalence.py`**: git OBJ vs working-tree STL Hausdorff/volume equivalence.
- **`test_xarm5_pose4_stl_collision.py`**: documents link3↔link5 Genesis self-contact and `pd_tracking_saturation` gate bypass.
- **`assets/urdf/xarm{5,6,7}/README.md`**: collision/visual mesh layout notes.

### Changed

- **Breaking package layout**: public Python modules are now grouped by function domain: `ufactory.robots`, `ufactory.kinematics`, `ufactory.dynamics`, `ufactory.hardware`, `ufactory.grippers`, `ufactory.trajectory`, `ufactory.manipulation`, `ufactory.visualization`, and `ufactory.deploy`.
- The root `ufactory` namespace now exposes only core robot convenience APIs: profile lookup, generic URDF/asset paths, and runtime profile lookup.
- Five-robot arm dynamics URDFs now reference `collision/*.stl` instead of `collision/*.obj`.
- `observe-hold-current` console entry point targets `ufactory.hardware.observe`; dynamics CLI entry points remain under `ufactory.dynamics.cli`.
- Dynamics report schema v3: explicit per-joint torque units (`*_nm`), `status_reason`, `worst_joint`, `worst_abs_err_nm`; reports archived under `reports/dyn_ver_<SN>/`.

### Removed

- Legacy root modules were removed with no compatibility wrappers: `ufactory.paths`, `ufactory.robot_registry`, `ufactory.robot_params`, `ufactory.kinematics_validation`, `ufactory.real_robot_session`, `ufactory.xarm_control`, `ufactory.gripper_g2`, `ufactory.bio_gripper_g2`, `ufactory.glb_visual`, `ufactory.dynamics_validation`, `ufactory.dynamics_static_analysis`, and `ufactory.dynamics_verify`.
- xArm5/6/7 `collision/*.obj` meshes (replaced by equivalent STL).

### Known Issues

- **xArm5 pose 4 (Genesis)**: link3↔link5 self-contact persists in kinematic and PD-hold modes (mesh geometry overlap in simulation). Real robot and SDK collision checks pass; `pd_tracking_saturation` gate allows hardware validation when Pinocchio gravity at the target is nominal.

## [0.1.6] — 2026-07-02

### Added

- **Asset integrity tests**: public URDF mesh references must resolve, dynamics URDF collision meshes must use `collision/*.stl` (not visual or OBJ), and raw/source GLB intermediates must stay out of the public file set.
- Optional dependency extras: `real` for xArm SDK, `rl` for rsl-rl training/evaluation, and `showcase` for the packaging showcase scipy dependency.

### Changed

- Public source tree slimmed for GitHub release: final GLB visual assets remain bundled, while raw/source GLBs and relocalize metrics are no longer part of the public surface.
- Default `requirements.txt` now contains only baseline simulation/viewer dependencies; advanced real-robot and RL dependencies are opt-in extras.
- xArm6 1305 URDF collision entries now reference `meshes/xarm6_1305/collision/*.obj` instead of reusing visual STL meshes.
- Standalone Gripper G2 and Bio Gripper G2 URDFs now point to repository-local mesh paths so all public URDF mesh references resolve.

### Removed

- Public asset-regeneration scripts (`vendor_*`, `relocalize_*`, `generate_*_combo_urdf.py`, `verify_*_assets.py`) from `scripts/`; generated final URDF/GLB assets remain available.
- `ufactory.dynamics_pose_selection`, `scripts/select_dynamics_calib_poses.py`, and their tests; default validation poses now live in `assets/configs/dynamics_validation_pose.yaml` and are loaded through `ufactory.dynamics.poses_config`.

## [0.1.5] — 2026-06-29

### Added

- **Runtime profiles**: typed runtime profiles (joint PD, torque limits, dynamics poses, task capabilities)
- **Dynamics validation**: Genesis PD hold vs real-robot static torque validation; CLIs `dynamics-sim-check`, `dynamics-hardware-check`, `dynamics-sim-collision-check`, `dynamics-report-compare`
- **Real-robot helpers**: real-robot connection, rad/rad/s motion, hold sampling
- **`ufactory.dynamics_pose_selection`** and **`scripts/select_dynamics_calib_poses.py`**: EE y hemisphere stratified calibration pose selection
- **`ufactory.kinematics.validation`**: shared FK/IK verification CLI for generic robot examples
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

- **`BioGripperG2` controller module** for reusable open/close control across all supported arms
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
