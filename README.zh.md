# ufactory_genesis

UFACTORY 机器人模型与 Genesis 仿真测试。

[English](README.md)

## 环境安装

已在 Python 3.13、Genesis 1.1.1、PyTorch 2.12.0+cu130 下验证。

```bash
conda activate py313

pip install torch==2.12.0+cu130 torchvision==0.27.0+cu130 \
  --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install -e .

export NUMBA_CACHE_DIR=~/.cache/numba
python -c "import genesis, torch; print('OK', torch.__version__, 'cuda:', torch.cuda.is_available())"
```

## 支持机型

| profile key | 机型 | Gripper G2 | Bio Gripper G2 |
|-------------|------|:----------:|:--------------:|
| `xarm5_1305` | xArm 5 | ✓ | ✓ |
| `xarm6_1305` | xArm 6 | ✓ | ✓ |
| `xarm7_1305` | xArm 7 | ✓ | ✓ |
| `uf850` | UF850 | ✓ | ✓ |
| `lite6` | Lite6 | — | — |

两种 G2 配件为不同产品：**Gripper G2**（原厂平行夹爪，资产 `assets/urdf/gripper_g2/`）与 **Bio Gripper G2**（生物夹爪，资产 `assets/urdf/bio_gripper/`）。加载时互斥；Lite6 均不支持。

多机型资产管线与限制见 [docs/multi_robot_compatibility.md](docs/multi_robot_compatibility.md)。

## GLB 视觉预览

GLB 用于高精度渲染；碰撞与物理仍走 STL 网格。统一入口：

```bash
export NUMBA_CACHE_DIR=~/.cache/numba

# 仅机械臂
python examples/view_robot_glb.py --robot <profile_key>

# Gripper G2（静态 / 可动开合）
python examples/view_robot_glb.py --robot xarm6_1305 --gripper-g2
python examples/view_robot_glb.py --robot xarm6_1305 --gripper-g2 --movable --gripper-demo

# Bio Gripper G2（静态）
python examples/view_robot_glb.py --robot uf850 --bio-gripper-g2
```

各目录下 `view_*_glb.py`（如 `examples/xarm6/view_xarm6_glb.py`）等价于 `view_robot_glb.py --robot <key>`；xArm6 专用脚本额外提供 `--diagnose`。

| 参数 | 产品 | 说明 |
|------|------|------|
| `--gripper-g2` | Gripper G2 | 加载 combo URDF |
| `--movable` | Gripper G2 | 分 link GLB（开合必需） |
| `--gripper-demo` | Gripper G2 | `drive_joint` 循环演示 |
| `--bio-gripper-g2` | Bio Gripper G2 | 静态 GLB 叠加 |
| `--pd` | 机械臂 | 关节演示 |
| `--no-show-tcp` | 机械臂 | 隐藏 EE 法兰红色 TCP 标记 |

更换源 GLB 后重定位与生成 combo URDF：

```bash
python scripts/relocalize_gripper_glb.py           # Gripper G2
python scripts/generate_gripper_g2_combo_urdf.py
python scripts/relocalize_bio_gripper_glb.py       # Bio Gripper G2
python scripts/generate_bio_gripper_combo_urdf.py
python scripts/relocalize_arm_glb.py --robot <profile_key>
```

代码加载：`ufactory.paths.robot_visual_glb_urdf(robot_key, with_gripper_g2=..., with_bio_gripper_g2=..., movable=...)`。

校验：`python examples/verify_robot.py --robot <key>`、`PYTHONPATH=. python scripts/verify_gripper_g2_assets.py`。

## 运动学补偿（按 SN 判断）

控制柜内**逐台运动学补偿**是否可用，可由 SN 第 3–6 位（四位型号码）判断：

| 机型 | SN 型号码 | 是否有补偿 |
|------|-----------|------------|
| xArm 5/6/7 | `< 1304` | **一定没有** — 使用标称 URDF，勿传 `--kinematics-*` |
| xArm 5/6/7 | `≥ 1304`（如 1305） | 可能有 — 需从本机提取 YAML |
| Lite6 | `< 1006` | **一定没有** |
| Lite6 | `≥ 1006` | 可能有 |
| UF850 | 任意 | **一定有** |

示例 SN：`XI130506D43A0A` → 型号码 `1305`（xArm6，需标定）。

```bash
# 仅当 SN 规则允许时才会导出；旧款 xArm 会提示跳过
python scripts/gen_kinematics_params.py <ip> <suffix>

# 通用 FK/IK 验证（--robot 见上表「支持机型」）
python examples/fk_verify_robot.py --robot xarm6_1305 --ip <ip> --kinematics-suffix <suffix>
python examples/ik_verify_robot.py --robot lite6 --ip <ip> --kinematics-suffix <suffix>
```

## xArm 6 验证

```bash
export NUMBA_CACHE_DIR=~/.cache/numba

python examples/xarm6/verify_xarm6.py
python examples/xarm6/verify_xarm6_dynamics.py

# 可选：与真机对比（XI1305 等 SN≥1304 需先提取本机运动学标定）
python scripts/gen_kinematics_params.py 192.168.1.60 xi1305
python examples/xarm6/fk_verify.py --ip 192.168.1.60 --kinematics-suffix xi1305
python examples/xarm6/ik_verify.py --ip 192.168.1.60 --kinematics-suffix xi1305

python examples/xarm6/xarm6_reach_train.py -B 1 --max_iterations 10
python examples/xarm6/xarm6_grasp_place_train.py -B 1 --max_iterations 5

pytest tests/test_xarm6_smoke.py -v
```

完整说明：[docs/xarm6_verification.md](docs/xarm6_verification.md)。

仿真 URDF：`xarm6_1305.urdf`（6 自由度）、`xarm6_with_gripper.urdf`（12 自由度，RL 默认）。

## 后续计划

- [ ] **多机型运动学验证** — 在 xArm6 已验证基线上，为 Lite6 / UF850 / xArm5/7 补齐 FK/IK 真机对比与 SN 标定流程
- [ ] **多机型动力学验证** — 抽象 xArm6 动力学验证，覆盖各机型 URDF 惯量与 Genesis 物理行为
- [ ] **强化学习环境验证** — 规范化 reach / grasp-place 环境的观测、奖励与碰撞检查，纳入 pytest
- [ ] **强化学习实例** — 提供可复现的训练配置与 eval demo（基于 rsl-rl-lib）
- [ ] **LeRobot 集成** — 仿真策略与真机数据采集/部署的桥接适配

> 当前 xArm6 为参考实现；其他机型的验证深度仍在扩展中。

## 项目结构

```
assets/urdf/
  xarm6/ xarm5/ xarm7/ lite6/ uf850/ gripper_g2/ bio_gripper/
ufactory/                   # paths, robot_registry, kinematics, GLB PBR
examples/xarm6/             # xArm6 验证、RL、查看器
examples/{lite6,uf850,xarm5,xarm7}/
examples/view_robot_glb.py  # 通用 GLB 预览
scripts/                    # vendor, relocalize, bio combo
tests/
```
