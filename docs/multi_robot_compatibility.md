# Multi-Robot Compatibility Guide

## G2 Accessories (xArm / UF850)

| Name | Description | Assets |
|------|-------------|--------|
| **Gripper G2** | UFACTORY parallel-jaw G2 gripper; static + movable (`drive_joint`) visual tracks | `assets/urdf/gripper_g2/` |
| **Bio Gripper G2** | Bio Gripper G2; static monolithic GLB + movable (`bio_gripper_g2_right_finger_joint` prismatic) | `assets/urdf/bio_gripper_g2/` |

CLI: `--gripper-g2` vs `--bio-gripper-g2` (mutually exclusive). Not supported on Lite6.

## Lite6 Accessories (Lite6 only)

| Name | Description | Assets |
|------|-------------|--------|
| **Lite6 Gripper** | UFACTORY Lite6 parallel gripper; static + movable (`finger_joint1` prismatic) | `assets/urdf/lite6_gripper/` |
| **Lite6 Vacuum Gripper** | Lite6 vacuum end-effector; static monolithic GLB | `assets/urdf/lite6_vacuum_gripper/` |

CLI: `--lite6-gripper` vs `--lite6-vacuum-gripper` (mutually exclusive with each other and with G2 accessories).

Physics combo URDFs: `lite6_with_gripper.urdf`, `lite6_with_vacuum_gripper.urdf`.

## Supported Models

| Profile key | DOF | EE link | Gripper G2 | Bio G2 | Lite6 Gripper | Lite6 Vacuum | Base URDF | GLB visual URDF |
|-------------|-----|---------|------------|--------|---------------|--------------|-----------|-----------------|
| `lite6` | 6 | link6 | No | No | Yes | Yes | `lite6.urdf` | `lite6_visual.glb.urdf` |
| `uf850` | 6 | link6 | Yes | Yes | No | No | `uf850.urdf` | `uf850_visual.glb.urdf` |
| `xarm5_1305` | 5 | link5 | Yes | Yes | No | No | `xarm5_1305.urdf` | `xarm5_1305_visual.glb.urdf` |
| `xarm7_1305` | 7 | link7 | Yes | Yes | No | No | `xarm7_1305.urdf` | `xarm7_1305_visual.glb.urdf` |
| `xarm6_1305` | 6 | link6 | Yes | Yes | No | No | `xarm6_1305.urdf` | `xarm6_1305_visual.glb.urdf` |

## Asset Pipeline

**When to run:** A normal `git clone` uses pre-built GLB/URDF assets — no relocalize step is required. Run this pipeline only when you replace source GLBs under `visual_glb_src/`, update vendor URDF/STL, or change relocalize scripts.

1. **Vendor URDF + STL** from [xarm_ros2](https://github.com/xArm-Developer/xarm_ros2).

```bash
python scripts/vendor_robot_assets.py
```

2. **Relocalize arm GLB**:

```bash
export NUMBA_CACHE_DIR=~/.cache/numba
python scripts/relocalize_arm_glb.py --robot lite6
```

3. **Gripper G2** (shared `assets/urdf/gripper_g2/`):

```bash
python scripts/relocalize_gripper_glb.py
python scripts/generate_gripper_g2_combo_urdf.py
```

4. **Bio Gripper G2 combo URDF**:

```bash
python scripts/relocalize_bio_gripper_g2_glb.py
python scripts/generate_bio_gripper_g2_combo_urdf.py
```

5. **Lite6 Gripper** (`assets/urdf/lite6_gripper/`):

```bash
python scripts/relocalize_lite6_gripper_glb.py
python scripts/generate_lite6_gripper_combo_urdf.py
python scripts/generate_lite6_physics_combo_urdf.py
```

6. **Lite6 Vacuum Gripper** (`assets/urdf/lite6_vacuum_gripper/`):

```bash
python scripts/relocalize_lite6_vacuum_gripper_glb.py
python scripts/generate_lite6_vacuum_gripper_combo_urdf.py
python scripts/generate_lite6_physics_combo_urdf.py
```

Source CAD GLBs are copied by `vendor_robot_assets.py --sim-root <path>` (provide the path to your sim repository containing `lite6_gripper/` and `lite6_vacuum_gripper/` GLB files).

### Maintainer checks

After Gripper G2 asset changes, verify combo URDFs and relocalize metrics:

```bash
export NUMBA_CACHE_DIR=~/.cache/numba
PYTHONPATH=. python scripts/verify_gripper_g2_assets.py
PYTHONPATH=. python scripts/verify_lite6_gripper_assets.py
```

See `assets/urdf/gripper_g2/meshes/visual/relocalize_metrics.json` for per-link alignment scores.

## Preview

```bash
python examples/view_robot_glb.py --robot lite6
python examples/view_robot_glb.py --robot lite6 --lite6-gripper --movable --gripper-demo
python examples/view_robot_glb.py --robot lite6 --lite6-vacuum-gripper
python examples/view_robot_glb.py --robot uf850 --gripper-g2
python examples/view_robot_glb.py --robot xarm5_1305 --gripper-g2 --movable --gripper-demo
python examples/view_robot_glb.py --robot xarm7_1305 --bio-gripper-g2
python examples/view_robot_glb.py --robot xarm6_1305 --bio-gripper-g2 --movable --gripper-demo
python examples/bio_gripper_g2/view_bio_gripper_g2_movable.py
```

## Verification

```bash
python examples/verify_robot.py --robot lite6
python scripts/verify_gripper_g2_assets.py
python scripts/verify_bio_gripper_g2_assets.py
PYTHONPATH=. python scripts/verify_lite6_gripper_assets.py
```

### Per-unit kinematics (SN rules)

| Family | No compensation (nominal URDF only) | May need `--kinematics-suffix` |
|--------|-----------------------------------|--------------------------------|
| xArm 5/6/7 | SN code **< 1304** | SN code **≥ 1304** |
| Lite6 | SN code **< 1006** | SN code **≥ 1006** |
| UF850 | — | **all units** |

## Known Limits

- GLB relocalize quality depends on STL/EE flange alignment; see `gripper_g2/meshes/visual/relocalize_metrics.json`.
- G2 static preview uses per-EE high-res GLB; movable mode uses shared finger/knuckle GLBs + per-EE `base.glb`.
- Bio Gripper G2 static preview uses per-EE monolithic GLB; movable mode uses `bio_gripper_g2_left_finger.glb` / `bio_gripper_g2_right_finger.glb` + per-EE `bio_gripper_g2_base.glb` (`bio_gripper_g2_right_finger_joint` 0–40 mm, mimic `bio_gripper_g2_left_finger_joint`).
- Lite6 Gripper movable mode uses `finger_joint1` (prismatic 0–8.9 mm) + mimic `finger_joint2`; demo via `--lite6-gripper --movable --gripper-demo`.
- Lite6 Vacuum Gripper preview is static (monolithic GLB); no animation.
- Per-unit kinematics YAML files are gitignored under `assets/urdf/*/kinematics/user/`.
