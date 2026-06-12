# ufactory_genesis

UFACTORY robot models and Genesis simulation tests.

[中文文档](README.zh.md)

## Setup

Tested with Python 3.13, Genesis 1.1.1, PyTorch 2.12.0+cu130.

```bash
conda activate py313

pip install torch==2.12.0+cu130 torchvision==0.27.0+cu130 \
  --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install -e .

export NUMBA_CACHE_DIR=~/.cache/numba
python -c "import genesis, torch; print('OK', torch.__version__, 'cuda:', torch.cuda.is_available())"
```

## Supported Robots

| profile key | Model | Gripper G2 | Bio Gripper G2 |
|-------------|-------|:----------:|:--------------:|
| `xarm5_1305` | xArm 5 | ✓ | ✓ |
| `xarm6_1305` | xArm 6 | ✓ | ✓ |
| `xarm7_1305` | xArm 7 | ✓ | ✓ |
| `uf850` | UF850 | ✓ | ✓ |
| `lite6` | Lite6 | — | — |

Two distinct G2 accessories: **Gripper G2** (parallel jaw, `assets/urdf/gripper_g2/`) and **Bio Gripper G2** (bio gripper, `assets/urdf/bio_gripper/`). Mutually exclusive at load time; Lite6 supports neither.

See [docs/multi_robot_compatibility.md](docs/multi_robot_compatibility.md).

## GLB Visual Preview

High-fidelity GLB rendering; collision and physics still use STL meshes. Single entry point:

```bash
export NUMBA_CACHE_DIR=~/.cache/numba

# Arm only
python examples/view_robot_glb.py --robot <profile_key>

# Gripper G2 (static / movable open-close)
python examples/view_robot_glb.py --robot xarm6_1305 --gripper-g2
python examples/view_robot_glb.py --robot xarm6_1305 --gripper-g2 --movable --gripper-demo

# Bio Gripper G2 (static)
python examples/view_robot_glb.py --robot uf850 --bio-gripper-g2
```

Per-model `view_*_glb.py` scripts (e.g. `examples/xarm6/view_xarm6_glb.py`) are thin wrappers around `view_robot_glb.py --robot <key>`; the xArm6 script also adds `--diagnose`.

| Flag | Product | Effect |
|------|---------|--------|
| `--gripper-g2` | Gripper G2 | Load combo URDF |
| `--movable` | Gripper G2 | Per-link GLBs (required for animation) |
| `--gripper-demo` | Gripper G2 | Cycle `drive_joint` open ↔ close |
| `--bio-gripper-g2` | Bio Gripper G2 | Static GLB overlay |
| `--pd` | Arm | Joint pose demo |
| `--no-show-tcp` | Arm | Hide red TCP marker on EE flange |

After updating source GLBs:

```bash
python scripts/relocalize_gripper_glb.py           # Gripper G2
python scripts/generate_gripper_g2_combo_urdf.py
python scripts/relocalize_bio_gripper_glb.py       # Bio Gripper G2
python scripts/generate_bio_gripper_combo_urdf.py
python scripts/relocalize_arm_glb.py --robot <profile_key>
```

Load in code: `ufactory.paths.robot_visual_glb_urdf(robot_key, with_gripper_g2=..., with_bio_gripper_g2=..., movable=...)`.

Verify: `python examples/verify_robot.py --robot <key>`, `PYTHONPATH=. python scripts/verify_gripper_g2_assets.py`.

## Kinematic calibration (SN rules)

Per-unit firmware calibration eligibility (SN positions 3–6, four-digit model code):

| Model | SN code | Compensation |
|-------|---------|--------------|
| xArm 5/6/7 | `< 1304` | **None** — use nominal URDF only |
| xArm 5/6/7 | `≥ 1304` (e.g. 1305) | Extract YAML from this unit |
| Lite6 | `< 1006` | **None** |
| Lite6 | `≥ 1006` | Extract YAML from this unit |
| UF850 | any | **Always** |

Example SN: `XI130506D43A0A` → model code `1305` (xArm6, calibration required).

```bash
python scripts/gen_kinematics_params.py <ip> <suffix>   # skips old SNs automatically
python examples/fk_verify_robot.py --robot xarm6_1305 --ip <ip> --kinematics-suffix <suffix>
python examples/ik_verify_robot.py --robot lite6 --ip <ip> --kinematics-suffix <suffix>
```

## xArm 6 Verification

```bash
export NUMBA_CACHE_DIR=~/.cache/numba

python examples/xarm6/verify_xarm6.py
python examples/xarm6/verify_xarm6_dynamics.py

# Optional: FK/IK vs real robot (SN ≥ 1304 needs per-unit calibration)
python scripts/gen_kinematics_params.py 192.168.1.60 xi1305
python examples/xarm6/fk_verify.py --ip 192.168.1.60 --kinematics-suffix xi1305
python examples/xarm6/ik_verify.py --ip 192.168.1.60 --kinematics-suffix xi1305

python examples/xarm6/xarm6_reach_train.py -B 1 --max_iterations 10
python examples/xarm6/xarm6_grasp_place_train.py -B 1 --max_iterations 5

pytest tests/test_xarm6_smoke.py -v
```

Details: [docs/xarm6_verification.md](docs/xarm6_verification.md).

Simulation URDFs: `xarm6_1305.urdf` (6 DOF), `xarm6_with_gripper.urdf` (12 DOF, default for RL).

## Roadmap

- [ ] **Multi-robot kinematics verification** — Extend xArm6 FK/IK real-robot comparison and SN calibration to Lite6, UF850, and xArm5/7
- [ ] **Multi-robot dynamics verification** — Generalize xArm6 dynamics checks to all models (URDF inertias and Genesis physics)
- [ ] **RL environment verification** — Formalize reach / grasp-place obs, rewards, and collision checks in pytest
- [ ] **RL training examples** — Reproducible training configs and eval demos (rsl-rl-lib)
- [ ] **LeRobot integration** — Bridge sim policies to real-robot data collection and deployment

> xArm6 is the reference implementation today; verification depth for other models is still expanding.

## Project Layout

```
assets/urdf/
  xarm6/ xarm5/ xarm7/ lite6/ uf850/ gripper_g2/ bio_gripper/
ufactory/                   # paths, robot_registry, kinematics, GLB PBR
examples/xarm6/             # xArm6 verify, RL, viewer
examples/{lite6,uf850,xarm5,xarm7}/
examples/view_robot_glb.py  # generic GLB preview
scripts/                    # vendor, relocalize, bio combo
tests/
```
