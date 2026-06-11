# ufactory_genesis

UFACTORY robot models and Genesis simulation tests.

[中文文档](README.zh.md)

## Setup

```bash
conda activate py313

# PyTorch 2.12 Stable (CUDA 13.0) — requires NVIDIA driver >=580.65
pip install torch==2.12.0+cu130 torchvision==0.27.0+cu130 \
  --index-url https://download.pytorch.org/whl/cu130

pip install -r requirements.txt
pip install -e .
```

Verify environment:

```bash
export NUMBA_CACHE_DIR=~/.cache/numba
python -c "import genesis, torch; print('genesis OK, torch', torch.__version__, 'cuda:', torch.cuda.is_available())"
```

## xArm 6 Verification Pipeline

Run from project root:

```bash
export NUMBA_CACHE_DIR=~/.cache/numba

# 1. Pure Genesis verification (headless)
python examples/xarm6/verify_xarm6.py

# 2. Dynamics tests
python examples/xarm6/verify_xarm6_dynamics.py

# 3. SDK FK/IK comparison (optional, needs xArm network)
python examples/xarm6/fk_verify.py --ip 192.168.1.60
python examples/xarm6/ik_verify.py --ip 192.168.1.60

# 4. RL smoke tests
python examples/xarm6/xarm6_reach_train.py -B 1 --max_iterations 10
python examples/xarm6/xarm6_grasp_place_train.py -B 1 --max_iterations 5

# 5. Automated pytest suite
pytest tests/test_xarm6_smoke.py -v

# 6. GLB visual preview (1305 arm + optional Gripper G2)
python examples/xarm6/view_xarm6_glb.py              # arm only; red dot = DH TCP (link6)
python examples/xarm6/view_xarm6_glb.py --g2         # arm + G2 visual
python examples/xarm6/view_xarm6_glb.py --g2 --pd    # with joint motion demo
python examples/xarm6/view_xarm6_glb.py --no-show-tcp # hide TCP marker
python examples/xarm6/view_xarm6_glb.py --diagnose   # headless alignment check
```

See [docs/xarm6_verification.md](docs/xarm6_verification.md) for full details.

## GLB Visual Models

High-fidelity GLB meshes for rendering; collision/physics unchanged (STL/OBJ):

| URDF | Purpose |
|------|---------|
| `xarm6_1305.urdf` | Simulation baseline (STL visual) |
| `xarm6_with_gripper.urdf` | RL baseline (STL visual + gripper joints) |
| `xarm6_1305_visual.glb.urdf` | Arm GLB visual only |
| `xarm6_1305_g2_visual.urdf` | Arm GLB + G2 gripper visual (physics from STL) |

GLB assets: `assets/urdf/xarm6/meshes/xarm6_1305/visual_glb/` and `meshes/gripper_g2/visual/`.

CAD-exported GLBs are relocalized to URDF link frames via `python scripts/relocalize_arm_glb.py` (originals kept in `visual_glb_raw/`). The viewer preserves GLB PBR (metallic/roughness per part) and holds the arm at the zero pose under gravity. By default a **red sphere** marks the theoretical DH TCP (`link6` flange origin, no gripper); use `--no-show-tcp` to hide it.

Load via `ufactory.paths.xarm6_1305_visual_glb_urdf(with_g2=False|True)`.

## Project Layout

```
assets/urdf/xarm6/     # xArm 6 URDF + meshes (local, not in pip genesis)
  meshes/xarm6_1305/visual_glb/   # GLB arm visuals
  meshes/gripper_g2/visual/       # G2 gripper GLB
ufactory/paths.py      # URDF path helpers
ufactory/glb_visual.py # GLB PBR surface helpers
examples/xarm6/        # Verification and RL scripts
scripts/               # Asset tooling (GLB relocalization)
tests/                 # pytest smoke tests
```

## Hardware Notes

- RTX 4060Ti 8GB: RL smoke tests use `-B 1`; avoid 4096 parallel envs.
- PyTorch: `2.12.0+cu130` (Stable). cu132 requires driver >=595 and is experimental.
- Set `NUMBA_CACHE_DIR=~/.cache/numba` if genesis import fails on numba cache.
- Set `XARM_IP=192.168.1.60` for hardware-marked pytest tests.
