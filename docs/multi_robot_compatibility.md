# Multi-Robot Compatibility Guide

## G2 Accessories (distinct products)

| Name | Description | Assets |
|------|-------------|--------|
| **Gripper G2** | UFACTORY parallel-jaw G2 gripper; static + movable (`drive_joint`) visual tracks | `assets/urdf/gripper_g2/` |
| **Bio Gripper G2** | Bio gripper G2; static monolithic GLB overlay | `assets/urdf/bio_gripper/` |

CLI: `--gripper-g2` vs `--bio-gripper-g2` (mutually exclusive). Lite6 supports neither.

## Supported Models

| Profile key | DOF | EE link | Gripper G2 | Bio Gripper G2 | Base URDF | GLB visual URDF |
|-------------|-----|---------|------------|------------------|-----------|-----------------|
| `lite6` | 6 | link6 | No | No | `lite6.urdf` | `lite6_visual.glb.urdf` |
| `uf850` | 6 | link6 | Yes | Yes | `uf850.urdf` | `uf850_visual.glb.urdf` |
| `xarm5_1305` | 5 | link5 | Yes | Yes | `xarm5_1305.urdf` | `xarm5_1305_visual.glb.urdf` |
| `xarm7_1305` | 7 | link7 | Yes | Yes | `xarm7_1305.urdf` | `xarm7_1305_visual.glb.urdf` |
| `xarm6_1305` | 6 | link6 | Yes | Yes | `xarm6_1305.urdf` | `xarm6_1305_visual.glb.urdf` |

## Asset Pipeline

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
python scripts/relocalize_bio_gripper_glb.py
python scripts/generate_bio_gripper_combo_urdf.py
```

## Preview

```bash
python examples/view_robot_glb.py --robot lite6
python examples/view_robot_glb.py --robot uf850 --gripper-g2
python examples/view_robot_glb.py --robot xarm5_1305 --gripper-g2 --movable --gripper-demo
python examples/view_robot_glb.py --robot xarm7_1305 --bio-gripper-g2
```

## Verification

```bash
python examples/verify_robot.py --robot lite6
python scripts/verify_gripper_g2_assets.py
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
- Bio Gripper G2 preview is static (monolithic GLB); no finger animation yet.
- Per-unit kinematics YAML files are gitignored under `assets/urdf/*/kinematics/user/`.
