# Changelog

All notable changes to genesis-ufactory will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.7] — 2026-07-15

### Added

- Task-oriented public examples under `visualization/`, `kinematics/`, `pick_place/`, and `packaging/`, with strict runtime overlays beside their entry points. The local `examples/rl/` tree is intentionally gitignored and is not part of the v0.2.7 public checkout.
- `ufactory.manipulation.packaging` as the reusable packaging geometry, planning, scene, simulation, and natural-drop diagnostics package.
- Shared `ufactory.visualization` viewer implementations, package-level RL configuration builders in `ufactory.training`, reach simulation evaluation helpers, overwrite-safe training directories, and complete checkpoint/config inventories.
- Architecture tests that reject production imports from `examples`, public `sys.path` mutation, legacy bootstrap paths, and Pinocchio/Coal leakage into the simulation/RL extras.
- Runtime compatibility checks for the Genesis version, GLB PBR hooks, deferred Viewer internals, and FK/IK scratch allocation. Versions newer than the validated 1.2.2 baseline emit a one-time warning and fail early when required hooks have changed.
- YAML-driven packaging profiles for xArm5, xArm6, xArm7, UF850, and Lite6, including strict box/table/target/gripper validation, robot-specific gripper geometry, effective scene hashing, and a Lite6 task overlay.
- Full real-packaging capability for Lite6 with its binary open/close gripper adapter; xArm6 + Gripper G2 remains enabled.

### Changed

- **Breaking:** public examples are organized by task rather than robot; the old per-robot wrappers and root underscore modules are removed without compatibility shims. See the English and Chinese example indexes for migration commands.
- **Breaking:** the shared trajectory task identity, CLI, and example paths are renamed from `grasp_place` to `pick_place` (`ufactory-pick-place`, `task.name: pick_place`, `examples/pick_place/`) with no compatibility aliases. Packaging remains `packaging_showcase` / `examples/packaging/`. Public RL example paths are deferred; library APIs stay under `ufactory.training`.
- Reach retains the current multi-robot environment while pick-place RL now explicitly accepts only xArm6 + Gripper G2. Both bind `env.runtime_config_sha256`; recipes contain only RL environment/reward/PPO/run settings and physical values resolve from `ResolvedRuntimeConfig`.
- Packaging release and post-release motion remains entirely Genesis contact-driven. The isolated drop report records impact, post-impact velocity, rebound peak, and settle time without imposing a minimum visible bounce.
- Pinocchio 4/Coal remain optional public backends: absent from `.[sim,rl]`, present in `.[real]` for safety preflight and `.[dynamics]` for reference dynamics.
- The simulation extra now requires `genesis-world>=1.2.2` and `packaging>=24`; the reproducible lock remains on the validated Genesis World 1.2.2 and PyTorch 2.10.0 stack.
- Existing manipulation physics settings remain unchanged while inheriting Genesis 1.2.2 contact-pruning, non-convex collision, thin-shell, and no-slip solver fixes.
- Packaging scene construction and physical execution are robot-generic and now exposed through the task-oriented packaging example and unchanged console command.
- Packaging CLI, report names, calibrated URDF selection, SDK simulation, and isolated real-time mirrors now preserve the selected robot key. xArm5, xArm7, and UF850 real packaging fails before controller connection while their simulation and SDK-simulation paths remain available.
- `box_floor_top_z_m` is consumed as the actual floor top, placement uses `fixed_target_position_m`, and grasp/release link heights derive from configured gripper geometry instead of G2-only constants.
- Packaging showcase `simulation_substeps` is restored from 32 to 8 to cut per-step physics cost during the motion phase (about 4× less work per control step on the measured packaging scene). The GPU three-cycle place-success gate is softened to 100 mm to match that tradeoff.
- Lite6 gripper finger collision uses dual convex boxes (inner pad + outer boss) that preserve the L-shaped concavity. Trajectory and packaging scenes no longer disable Genesis `convexify`/`decimate`/`watertighten` for Lite6 whole-robot URDFs. Safety exemptions for opposing-finger pairs are bound to both the nominal and SN-calibrated Lite6 URDF hashes; the Lite6 packaging box center is moved to Y=0.280 so calibrated preflight clears the near wall.

### Removed

- The disabled `ufactory.deploy` package, xArm6 reach deployment entry point, online policy/session/action adapters, legacy `SafetyGuard`, random-action execution, and their tests.
- Duplicate reach/grasp RL defaults from `TaskProfile`, the legacy `_grasp_place_traj.py`, all `_bootstrap.py` files, and per-robot public example wrappers.

### Fixed

- Removed the obsolete exact-1.2.1 runtime gate that rejected the validated Genesis 1.2.2 environment before scene initialization.
- Lite6 packaging no longer fails the two prior inverse-kinematics points or reports the gripper-stage collision set; its collision exemptions are limited to the two verified settle stages and bound to the current URDF hash.
- `project-check release` snapshot now uses a detached git worktree so asset tests that call `git ls-files`/`git show` still work, and `pip-audit --strict` skips the editable local install.

### Security

- The public package no longer ships online real-policy execution code. Training-only action scaling lives in `ufactory.training.actions`; real robot motion continues to require predictive safety checks, exact calibration binding, and explicit confirmation.

## [0.2.5] — 2026-07-14

### Added

- Versioned immutable YAML configuration with strict validation, deterministic source precedence, canonical hashes, `--print-config`, and source-tree asset integrity checks.
- `SafetyGate`, structured violations/reports, hash-bound `ApprovedProgram`, calibrated Pinocchio 4/Coal full-sample collision prediction, and injected kinematics/collision/clock/transport ports.
- Unified `ufactory-grasp-place` entry point, monotonic fail-closed real executor, same-run SDK simulation evidence for `servo_cartesian`, strict calibration manifests, and capability-driven G2/Lite6 adapters.
- Safe checkpoint YAML/manifests, scoped Genesis 1.2.1 PBR integration, `project-check` local quality reports, and a reproducible `uv.lock`.

### Changed

- **Breaking:** v0.2.5 supports cloned source repositories installed with `pip install -e .` only; wheel/sdist installation is unsupported.
- **Breaking:** configuration, trajectory and safety APIs are replaced by `ResolvedRuntimeConfig`, `preflight_program()`, `ApprovedProgram`, `execute_sim()` and `execute_real()`; no compatibility shim is promised.
- Maintainer technical notes live under the local gitignored `dev/docs/` workspace and are not published with the repository; breaking API changes are recorded in this CHANGELOG.
- Whole-program motion statistics now use signed velocity, vector acceleration, wrapped orientation differences and static endpoint velocities. Genesis initialization has a single runtime owner.
- Dynamics reports write schema v4 while readers retain v1-v3 support. Genesis and Torch are pinned to the validated 1.2.1 / 2.10.0 environment.
- Grasp-place trajectory, training/evaluation, and packaging examples now resolve one runtime-configured 30 mm, 17 g reference block; its table-resting base-frame center is consistently 15 mm in Genesis and predictive collision scenes.
- Pinocchio Cartesian shadow IK now enforces the shared gripper-down full pose with nominal `1e-6` damping and bounded near-singularity regularization. Offline Cartesian simulation uses a simulation-only approval while hardware approval still requires same-run, hash-bound SDK evidence no older than five minutes.

### Fixed

- Fixed Lite6 grasp-place preflight failures caused by a 5 mm object-height mismatch and non-existent collision-exemption stage names. Failure output now groups the complete violation set instead of silently showing only the first 20 samples.
- Simulation final metrics now hold the last approved target for two physics ticks before observation, eliminating end-of-tick tracking-lag false failures without changing the grasp-place command stream.

### Security

- Online real policy deployment and random-action real modes are hard-disabled. Runtime pickle loading is removed; checkpoints use `weights_only=True` and integrity manifests.
- Real execution never automatically clears errors, resets, resumes or catches up missed servo deadlines. Real mode requires exact serial/calibration binding and `--confirm-real`.

## [0.2.4] — 2026-07-09

### Changed

- Moved maintainer-only diagnose and legacy dynamics/demo scripts from `examples/` into the local gitignored workspace `dev/diagnostics/` (`manipulation/`, `dynamics/`); public docs and CI now point at `dynamics-sim-check` / `dynamics-hardware-check` instead of the old example wrappers.
- Gripper G2 grasp-place examples now insert a 0.5 s closed-gap `grip-settle` segment before lift so xArm5/6/7 and UF850 wait for the grasp to settle before raising the arm.
- Lite6 grasp-place `grip` and `release` segments now use 0.5 s durations, making the Lite6 gripper open/close four times faster while keeping the 0.18 s `place-settle`.
- Real gripper execution treats same-gap gripper hold segments as paced no-ops, so `grip-settle` and `place-settle` do not resend SDK gripper commands.
- Grasp-place MoveL source timing is now controlled by `--speed-mm-s` / `--mvacc-mm-s2` for sim, dry-run, `servo_cartesian`, and `servo_j`; defaults are 150 mm/s and 800 mm/s².
- Host-side IK `servo_j` now preserves the source Cartesian tick count and duration instead of silently retiming, so unsafe custom speeds fail through the real executor's finite-difference safety checks.
- UF850 host-side IK uses higher damping during `servo_j` compilation to keep a calibrated unit's wrist solution continuous near singular poses.
- Root README (EN/ZH): simplified GLB preview and trajectory grasp-place sections; added English Showcase section; aligned dynamics CLI examples.

## [0.2.3] — 2026-07-09

### Added

- Host-side IK `servo_j` path for shared grasp-place trajectory examples: MoveL samples are compiled with Genesis IK into explicit `set_servo_angle_j` joint streams.
- `--kinematics-yaml`, `--kinematics-yaml-dir`, and `--force-kinematics` options on grasp-place real paths; `--ip` still auto-resolves the SN-derived kinematics suffix.
- `compile_cartesian_program_to_joint_stream()` trajectory API and explicit `q_samples` MoveJ segments.

### Changed

- Grasp-place `--executor` values renamed to underscores: `servo_j` and `servo_cartesian` (hyphen forms removed).
- Grasp-place `--executor servo_j` is now supported for dry-run, SDK simulation validation, real streaming, mirror mode, and optional physical gripper commands.
- Host-side IK joint streams are plateau-collapsed and LSPB-retimed after Genesis IK so 100 Hz `servo_j` finite-difference acceleration stays within servo limits.
- Real-path mirror/carry tracking now handles compiled `movej` arm segments by label, preserving existing grasp/release behavior.
- Shared grasp-place cube physics unified across robots: 30 mm painted wood block at 17 g with friction 1.0; silicone fingertip pads at friction 1.2; contact stiffness left at Genesis rigid defaults (no per-object sol_params).
- Lite6 default sim hold bias raised from 0.8 mm to 2.0 mm so contact friction still carries the unified 17 g cube after lowering material μ.

### Fixed

- `servo_j` SDK feedback comparison now respects the program DOF instead of assuming six joints.

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
- Shared five-robot grasp-place trajectory examples for xArm5, xArm6, xArm7, UF850, and Lite6, with Genesis sim, dry-run, real `servo_cartesian`, and kinematic mirror paths.
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
- **`test_xarm5_pose4_stl_collision.py`**: documents link3↔link5 Genesis self-contact and `pd_tracking_saturation` check bypass.
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

- **xArm5 pose 4 (Genesis)**: link3↔link5 self-contact persists in kinematic and PD-hold modes (mesh geometry overlap in simulation). Real robot and SDK collision checks pass; `pd_tracking_saturation` check allows hardware validation when Pinocchio gravity at the target is nominal.

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
