# ufactory_genesis

UFACTORY 机器人模型与 Genesis 仿真测试。

[English](README.md)

## 环境安装

```bash
conda activate py313

# PyTorch 2.12 Stable (CUDA 13.0)，需 NVIDIA 驱动 >=580.65
pip install torch==2.12.0+cu130 torchvision==0.27.0+cu130 \
  --index-url https://download.pytorch.org/whl/cu130

pip install -r requirements.txt
pip install -e .
```

验证环境：

```bash
export NUMBA_CACHE_DIR=~/.cache/numba
python -c "import genesis, torch; print('genesis OK, torch', torch.__version__, 'cuda:', torch.cuda.is_available())"
```

## xArm 6 验证流程

在项目根目录运行：

```bash
export NUMBA_CACHE_DIR=~/.cache/numba

# 1. 纯 Genesis 验证（无界面）
python examples/xarm6/verify_xarm6.py

# 2. 动力学测试
python examples/xarm6/verify_xarm6_dynamics.py

# 3. SDK FK/IK 对比（可选，需连接 xArm 网络）
python examples/xarm6/fk_verify.py --ip 192.168.1.60
python examples/xarm6/ik_verify.py --ip 192.168.1.60

# 4. 强化学习冒烟测试
python examples/xarm6/xarm6_reach_train.py -B 1 --max_iterations 10
python examples/xarm6/xarm6_grasp_place_train.py -B 1 --max_iterations 5

# 5. 自动化 pytest 套件
pytest tests/test_xarm6_smoke.py -v

# 6. GLB 视觉预览（1305 机械臂 + 可选 G2 夹爪）
python examples/xarm6/view_xarm6_glb.py              # 仅机械臂；红点 = DH TCP（link6）
python examples/xarm6/view_xarm6_glb.py --g2         # 机械臂 + G2 视觉模型
python examples/xarm6/view_xarm6_glb.py --g2 --pd    # 带关节运动演示
python examples/xarm6/view_xarm6_glb.py --no-show-tcp # 隐藏 TCP 红点
python examples/xarm6/view_xarm6_glb.py --diagnose   # 无界面对齐诊断
```

完整说明见 [docs/xarm6_verification.md](docs/xarm6_verification.md)。

## GLB 视觉模型

高精度 GLB 网格用于渲染；碰撞/物理仍使用 STL/OBJ，不受影响：

| URDF | 用途 |
|------|------|
| `xarm6_1305.urdf` | 仿真基线（STL 视觉） |
| `xarm6_with_gripper.urdf` | RL 基线（STL 视觉 + 夹爪关节） |
| `xarm6_1305_visual.glb.urdf` | 仅机械臂 GLB 视觉 |
| `xarm6_1305_g2_visual.urdf` | 机械臂 GLB + G2 夹爪视觉（物理仍用 STL） |

GLB 资产路径：`assets/urdf/xarm6/meshes/xarm6_1305/visual_glb/` 与 `meshes/gripper_g2/visual/`。

CAD 导出的 GLB 通过 `python scripts/relocalize_arm_glb.py` 重定位到 URDF link 坐标系（原始文件保留在 `visual_glb_raw/`）。查看器保留 GLB PBR 材质（各部件 metallic/roughness），并在重力下以 PD 控制锁定零位。默认显示**红色小球**标定理论 DH 的 TCP（`link6` 法兰原点，不含夹爪）；可用 `--no-show-tcp` 关闭。

加载方式：`ufactory.paths.xarm6_1305_visual_glb_urdf(with_g2=False|True)`。

### 如何解读 TCP 红点

- **红点 = 算法 TCP**（URDF/DH 定义的 `link6` 帧，IK/FK 使用的参考点）
- **GLB 法兰外观** = 网格几何；若红点与法兰中心略有偏差，属于视觉网格与运动学帧的差异（link6 GLB 相对 STL 表面偏差约 0.3 mm 级），**不代表 IK/FK 算错**
- 若红点随关节运动始终贴在法兰关节处，则算法与 DH 链一致

## 项目结构

```
assets/urdf/xarm6/     # xArm 6 URDF + 网格（本地资产，不在 pip genesis 包内）
  meshes/xarm6_1305/visual_glb/   # GLB 机械臂视觉
  meshes/gripper_g2/visual/       # G2 夹爪 GLB
ufactory/paths.py      # URDF 路径辅助
ufactory/glb_visual.py # GLB PBR 表面辅助
examples/xarm6/        # 验证与 RL 脚本
scripts/               # 资产工具（GLB 重定位）
tests/                 # pytest 冒烟测试
```

## 硬件说明

- RTX 4060Ti 8GB：RL 冒烟测试使用 `-B 1`；避免 4096 并行环境。
- PyTorch：`2.12.0+cu130`（Stable）。cu132 需驱动 >=595，仍为实验性。
- 若 genesis 导入因 numba 缓存失败，设置 `NUMBA_CACHE_DIR=~/.cache/numba`。
- 硬件相关 pytest 测试设置 `XARM_IP=192.168.1.60`。
