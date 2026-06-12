# xArm 6 Genesis Simulation - Verification Guide

## Environment

```bash
conda activate py313

# PyTorch 2.12 Stable (CUDA 13.0), driver >=580.65
pip install torch==2.12.0+cu130 torchvision==0.27.0+cu130 \
  --index-url https://download.pytorch.org/whl/cu130

pip install -r requirements.txt
pip install -e .
export NUMBA_CACHE_DIR=~/.cache/numba
```

- Linux Ubuntu 24.04, x86-64
- NVIDIA RTX 4060Ti 8GB, driver >=580.65
- xArm 6 real robot IP (optional): `192.168.1.60`
- RL framework: `rsl-rl-lib==2.2.4`

---

## Step 1 - Forward Kinematics (FK)

Genesis URDF FK vs xarm-python-sdk FK, SDK simulation mode as ground truth.

**Pure Genesis verification (no real robot):**

```bash
python examples/xarm6/verify_xarm6.py -v        # with viewer
python examples/xarm6/verify_xarm6.py            # headless
```

**Genesis FK vs SDK FK comparison (requires robot network):**

XI1305 (SN model code ≥ 1304) and other newer xArms have per-unit kinematic calibration in firmware.
Units with SN code **< 1304** have **no** compensation — use nominal `xarm6_1305.urdf` without `--kinematics-*`.

Extract YAML from the control box (when SN allows), then verify:

```bash
# One-time per robot (saves to assets/urdf/xarm6/kinematics/user/, gitignored)
python scripts/gen_kinematics_params.py 192.168.1.60 xi1305

python examples/xarm6/fk_verify.py --ip 192.168.1.60 --kinematics-suffix xi1305
```

- PASS criteria: position error < 1.0 mm, rotation error < 0.5 deg
- Default base URDF: `xarm6_1305.urdf` (calibration applied via `--kinematics-*`)

---

## Step 2 - FK with Real Robot Comparison

```bash
python examples/xarm6/verify_xarm6.py --real-ip 192.168.1.60 -v
```

Kinematic calibration URDF generation:

```bash
python examples/xarm6/run_fk_alignment_cycle.py --real-ip 192.168.1.60
```

---

## Step 3 - Inverse Kinematics (IK)

```bash
python examples/xarm6/ik_verify.py --ip 192.168.1.60 --kinematics-suffix xi1305
```

- PASS criteria: position error < 1.0 mm, rotation error < 0.5 deg
- Test scale: 10 manual configs + 40 random = 50 total

---

## Step 4 - Dynamics

```bash
python examples/xarm6/verify_xarm6_dynamics.py
python examples/xarm6/verify_xarm6_dynamics.py -v
```

---

## Step 5 - Reach RL Task

**Smoke test:**

```bash
python examples/xarm6/xarm6_reach_train.py -B 1 --max_iterations 10 -v
```

- Observation: 18-dim
- Action: 6-dim delta joint positions
- URDF: `assets/urdf/xarm6/xarm6.urdf`

**4060Ti 8GB**: use `-B 1` for smoke; `-B 64~256` for small-scale training.

---

## Step 6 - Grasp-Place RL Task

**Smoke test:**

```bash
python examples/xarm6/xarm6_grasp_place_train.py -B 1 --max_iterations 5 -v
```

- URDF: `assets/urdf/xarm6/xarm6_with_gripper.urdf`
- **4060Ti 8GB**: avoid `-B 4096`; use `-B 1` smoke or `-B 64` max locally.

---

## Automated Tests

```bash
pytest tests/test_xarm6_smoke.py -v
pytest tests/test_xarm6_smoke.py -v -m hardware   # SDK FK/IK (needs XARM_IP)
```

---

## Asset Paths

xArm URDF files live in `assets/urdf/xarm6/`. Scripts load them via `ufactory.paths.xarm6_urdf()`.
Genesis built-in assets (e.g. ground plane) use relative paths from the pip package.

### Visual vs Simulation URDF

| URDF | Visual | Collision / Joints | Use case |
|------|--------|-------------------|----------|
| `xarm6_1305.urdf` | STL | STL/OBJ, 6 DOF | Simulation baseline |
| `xarm6_with_gripper.urdf` | STL | STL/OBJ, 12 DOF | RL / grasp-place (default) |
| `xarm6_1305_visual.glb.urdf` | GLB (7 links) | STL/OBJ, 6 DOF | High-fidelity arm preview |
| `xarm6_1305_g2_visual.urdf` | GLB arm + Gripper G2 static | STL/OBJ + gripper joints, 12 DOF | High-res Gripper G2 preview (fixed) |
| `xarm6_1305_g2_movable_visual.urdf` | GLB arm + Gripper G2 per-link | STL/OBJ + gripper joints, 12 DOF | Gripper G2 open/close animation |

Gripper G2 uses a **dual-track** visual setup:

- **Static** (`visual_glb_src/gripper_g2_movable.glb` → `gripper_g2_static_{ee_link}.glb`): high-res CAD assembly on a fixed link; does not move with `drive_joint`.
- **Movable** (`visual_glb_src/gripper_g2.glb` → `visual_glb/*.glb` + `visual_glb/{ee_link}/base.glb`): semantic parts split across gripper links; visuals follow `drive_joint` + mimic joints.

Assets live under `assets/urdf/gripper_g2/` (shared across xArm5/6/7 and UF850). Regenerate with `python scripts/relocalize_gripper_glb.py` and `python scripts/generate_gripper_g2_combo_urdf.py`.

**GLB preview:**

```bash
python examples/view_robot_glb.py --robot xarm6_1305              # arm GLB
python examples/view_robot_glb.py --robot xarm6_1305 --gripper-g2         # arm + Gripper G2 static
python examples/view_robot_glb.py --robot xarm6_1305 --gripper-g2 --movable --gripper-demo
python examples/view_robot_glb.py --robot xarm6_1305 --gripper-g2 --movable --pd --gripper-demo

# xArm6-specific wrapper (also supports --diagnose)
python examples/xarm6/view_xarm6_glb.py --gripper-g2 --movable --gripper-demo
```

**Verify GLB URDF (FK/IK/PD unchanged):**

```bash
python examples/xarm6/verify_xarm6.py --robot-model assets/urdf/xarm6/xarm6_1305_visual.glb.urdf
python examples/xarm6/verify_xarm6.py --robot-model assets/urdf/xarm6/xarm6_1305_g2_visual.urdf
python examples/xarm6/verify_xarm6.py --robot-model assets/urdf/xarm6/xarm6_1305_g2_movable_visual.urdf
```

Helper: `ufactory.paths.robot_visual_glb_urdf("xarm6_1305", with_gripper_g2=True|False, movable=True|False)` (alias: `xarm6_1305_visual_glb_urdf`).

Other robots: `python examples/view_robot_glb.py --robot uf850 --gripper-g2` (see [multi_robot_compatibility.md](multi_robot_compatibility.md)).
