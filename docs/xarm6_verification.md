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

```bash
python examples/xarm6/fk_verify.py --ip 192.168.1.60
python examples/xarm6/fk_verify.py --ip 192.168.1.60 --urdf assets/urdf/xarm6/xarm6_1305.urdf
```

- PASS criteria: position error < 1.0 mm, rotation error < 0.5 deg
- Default URDF: `xarm6_xarm6_kinematics_calib1_calib.urdf`

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
python examples/xarm6/ik_verify.py --ip 192.168.1.60
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
| `xarm6_1305_g2_visual.urdf` | GLB arm + G2 | STL/OBJ + gripper joints, 12 DOF | Preview with G2 look |

G2 is **visual-only**: the single `xarm_gripper_g2.glb` does not animate with `drive_joint`; physics still uses the original finger collision meshes.

**GLB preview:**

```bash
python examples/xarm6/view_xarm6_glb.py              # arm GLB
python examples/xarm6/view_xarm6_glb.py --g2         # arm + G2
python examples/xarm6/view_xarm6_glb.py --g2 --pd    # with PD motion
```

**Verify GLB URDF (FK/IK/PD unchanged):**

```bash
python examples/xarm6/verify_xarm6.py --robot-model assets/urdf/xarm6/xarm6_1305_visual.glb.urdf
python examples/xarm6/verify_xarm6.py --robot-model assets/urdf/xarm6/xarm6_1305_g2_visual.urdf
```

Helper: `ufactory.paths.xarm6_1305_visual_glb_urdf(with_g2=True|False)`.
