# examples/

[English](README.md)

示例脚本目录索引。安装、抓放模式与真机安全流程见仓库根目录 [README.zh.md](../README.zh.md)。

## 快速开始

| 文件 | 说明 |
|------|------|
| `view_robot_glb.py` | 统一 GLB 查看器，支持全部机型与末端执行器 |

```bash
python examples/view_robot_glb.py --robot xarm6
```

## 机器人可视化

各机型目录下的包装脚本等价于 `view_robot_glb.py --robot <key>`：

| 目录 | 预览命令 |
|------|----------|
| `xarm5/` | `python examples/xarm5/view_xarm5_glb.py` |
| `xarm6/` | `python examples/xarm6/view_xarm6_glb.py` |
| `xarm7/` | `python examples/xarm7/view_xarm7_glb.py` |
| `lite6/` | `python examples/lite6/view_lite6_glb.py` |
| `uf850/` | `python examples/uf850/view_uf850_glb.py` |

## 夹爪演示

| 目录 | 说明 |
|------|------|
| `gripper_g2/` | Gripper G2 可动视觉演示 |
| `bio_gripper_g2/` | Bio Gripper G2 可动视觉演示 |
| `lite6_gripper/` | Lite6 平行夹爪可动视觉演示 |

## 机械臂验证

| 文件 | 说明 |
|------|------|
| `verify_robot.py` | 通用正运动学/位置控制冒烟测试（所有机型） |
| `fk_verify_robot.py` | 通用正运动学验证（可与真机对比） |
| `ik_verify_robot.py` | 通用逆运动学验证（可与真机对比） |
| `packaging_showcase.py` | YAML 驱动的五机型装箱入口 |

动力学验证使用安装后的控制台命令：

```bash
dynamics-sim-check --robot xarm6 --random-count 5
dynamics-hardware-check --robot xarm6 --ip <ip>
```

## 轨迹抓放

五个脚本是已安装命令 `ufactory-grasp-place`（`ufactory.cli.grasp_place`）的薄包装。完整的 `--mode` / `--executor` 说明见根目录 [README.zh.md](../README.zh.md#轨迹抓放)。

| 文件 | 说明 |
|------|------|
| `xarm5/xarm5_grasp_place_traj.py` | `--robot xarm5` 包装 |
| `xarm6/xarm6_grasp_place_traj.py` | `--robot xarm6` 包装 |
| `xarm7/xarm7_grasp_place_traj.py` | `--robot xarm7` 包装 |
| `uf850/uf850_grasp_place_traj.py` | `--robot uf850` 包装 |
| `lite6/lite6_grasp_place_traj.py` | `--robot lite6` 包装 |

```bash
python examples/xarm6/xarm6_grasp_place_traj.py --mode dry-run --executor servo_j
# 等价：
# ufactory-grasp-place --robot xarm6 --mode dry-run --executor servo_j

# 真机 + 运动学镜像窗口（详见根 README）：
# ufactory-grasp-place --robot xarm6 --mode real --executor servo_j \
#   --calibration path/to/exact.yaml --confirm-real --visual
```

## xArm 6 — 参考实现

xArm 6 拥有最完整的示例覆盖。

### 运动学

| 文件 | 说明 |
|------|------|
| `xarm6/verify_xarm6.py` | 正运动学 + 位置控制冒烟测试 |
| `xarm6/fk_verify.py` | 兼容入口，默认 `--robot xarm6` |
| `xarm6/ik_verify.py` | 兼容入口，默认 `--robot xarm6` |

### 强化学习

| 文件 | 说明 |
|------|------|
| `xarm6/xarm6_reach_env.py` / `_train.py` | 到达任务环境与训练 |
| `xarm6/xarm6_reach_deploy.py` | 到达任务部署辅助：对齐 / 离线预检仍可用；**v0.2.5 已硬禁用**在线真机策略 `deploy` 与 `smoke-random`（见 [SECURITY.md](../SECURITY.md)） |
| `xarm6/xarm6_grasp_place_env.py` / `_train.py` / `_eval.py` | 抓放强化学习任务 |

### 装箱展示场景

| 文件 | 说明 |
|------|------|
| `packaging_showcase.py` | xArm5/6/7、UF850、Lite6 通用 CLI 包装 |
| `_packaging_showcase.py` | 共享物理执行实现（内部模块） |
| `xarm6/xarm6_g2_showcase.py` | 固定选择 `--robot xarm6` 的兼容入口 |

共享 YAML 定义方块、桌面、箱体、路径、时序、接触策略和成功阈值；Lite6 叠加机型任务配置及自身夹爪几何。五机型的两种执行器均支持仿真、dry-run 和 SDK 仿真。完整真机装箱仅启用 xArm6 + G2 与 Lite6 + Lite6 夹子；其他 profile 会在连接控制器前拒绝 real 模式。

```bash
# 仿真默认单次；有限多轮和无限循环需显式指定
python examples/packaging_showcase.py --robot lite6 --mode sim --executor servo_j
python examples/packaging_showcase.py --robot lite6 --mode sim --executor servo_j --cycles 3
python examples/packaging_showcase.py --robot xarm7 --mode sim --executor servo_cartesian

python examples/packaging_showcase.py --robot uf850 --mode dry-run --executor servo_j
python examples/packaging_showcase.py --robot xarm5 --mode dry-run --executor servo_cartesian
XARM_IP=<ip> python examples/packaging_showcase.py --robot lite6 --mode real \
  --executor servo_j --calibration path/to/exact.yaml --confirm-real
```

## 内部模块

以 `_` 前缀的文件是内部共享模块，不是用户入口：

| 文件 | 用途 |
|------|------|
| `_bootstrap.py` | 将项目根目录加入 `sys.path` |
| `_robot_viewer.py` | 共享 Genesis GLB 查看器核心 |
| `_gripper_demo.py` | Gripper G2 开合控制 |
| `_bio_gripper_g2_demo.py` | Bio Gripper G2 开合控制 |
| `_lite6_gripper_demo.py` | Lite6 夹爪开合控制 |
| `_packaging_scene.py` | 展示场景构建器 |
| `_grasp_place_traj.py` | 遗留共享模块（测试仍引用）；用户路径为 `ufactory-grasp-place` |
| `_standalone_gripper_viewer.py` | 独立夹爪查看器辅助 |
