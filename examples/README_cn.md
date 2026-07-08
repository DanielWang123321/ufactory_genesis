# examples/

[English](README.md)

使用示例和教程。

## 快速开始

| 文件 | 说明 |
|------|------|
| `view_robot_glb.py` | **统一 GLB 查看器** — 支持全部机型和末端执行器 |

```bash
python examples/view_robot_glb.py --robot xarm6
```

## 机器人可视化

每个机器人目录包含独立的 GLB 预览脚本：

| 目录 | 预览命令 |
|------|----------|
| `xarm5/` | `python examples/xarm5/view_xarm5_glb.py` |
| `xarm6/` | `python examples/xarm6/view_xarm6_glb.py` |
| `xarm7/` | `python examples/xarm7/view_xarm7_glb.py` |
| `lite6/` | `python examples/lite6/view_lite6_glb.py` |
| `uf850/` | `python examples/uf850/view_uf850_glb.py` |

## 夹爪演示

| 目录/文件 | 说明 |
|-----------|------|
| `gripper_g2/` | Gripper G2 可动视觉演示 |
| `bio_gripper_g2/` | Bio Gripper G2 可动视觉演示 |
| `lite6_gripper/` | Lite6 平行夹爪可动视觉演示 |

## 机械臂验证

| 文件 | 说明 |
|------|------|
| `verify_robot.py` | 通用正运动学/位置控制冒烟测试（所有机型） |
| `fk_verify_robot.py` | 通用正运动学验证（可与真机对比） |
| `ik_verify_robot.py` | 通用逆运动学验证（可与真机对比） |
| `packaging_showcase.py` | 通用展示场景入口（当前支持 xArm6 + Gripper G2） |

动力学验证使用安装后的控制台命令：

```bash
dynamics-sim-check --robot xarm6 --random-count 5
dynamics-hardware-check --robot xarm6 --ip <ip>
```

## xArm 6 — 参考实现

xArm 6 拥有最完整的示例覆盖：

### 运动学与动力学
| 文件 | 说明 |
|------|------|
| `xarm6/verify_xarm6.py` | 正运动学 + 位置控制冒烟测试 |
| `xarm6/verify_xarm6_dynamics.py` | 动力学验证 |
| `xarm6/fk_verify.py` | 兼容入口，等价于通用正运动学验证默认 `--robot xarm6` |
| `xarm6/ik_verify.py` | 兼容入口，等价于通用逆运动学验证默认 `--robot xarm6` |

### 强化学习
| 文件 | 说明 |
|------|------|
| `xarm6/xarm6_reach_env.py` / `_train.py` | 到达任务环境与训练 |
| `xarm6/xarm6_reach_deploy.py` | 到达任务真机部署（对齐 / 冒烟 / 回放 / 部署） |
| `xarm6/xarm6_grasp_place_env.py` / `_train.py` / `_eval.py` | 抓放任务 |

### 轨迹规划
| 文件 | 说明 |
|------|------|
| `xarm5/xarm5_grasp_place_traj.py` | 共享路点/LSPB 抓放入口，xArm5 + Gripper G2 场景；仅仿真和 dry-run |
| `xarm6/xarm6_grasp_place_traj.py` | 共享路点/LSPB 抓放入口，xArm6 + Gripper G2 场景 |
| `xarm7/xarm7_grasp_place_traj.py` | 共享路点/LSPB 抓放入口，xArm7 + Gripper G2 场景 |
| `uf850/uf850_grasp_place_traj.py` | 共享路点/LSPB 抓放入口，UF850 + Gripper G2 场景 |
| `lite6/lite6_grasp_place_traj.py` | 共享路点/LSPB 抓放入口，Lite6 反装夹爪场景 |

五个入口都调用共享的 `examples/_grasp_place_traj.py`。该模块生成混合路点程序，按 LSPB 采样后执行抓放序列（`home->pregrasp` → 抓取 → 搬运 → 放置 → 回零）。

**1. Genesis 仿真**

仅在 Genesis 中运行完整物理仿真（`--visual` 会打开三维窗口），不连接机械臂。用于验证轨迹与抓取逻辑。

```bash
python examples/xarm6/xarm6_grasp_place_traj.py --visual --rate 50
```

**2. 真机 dry-run 安全验证**

连接真机前可先 dry-run，检查同一条轨迹的段数、速度、加速度和安全高度；默认不运动机械臂，也不发送夹爪命令。

```bash
python examples/xarm6/xarm6_grasp_place_traj.py \
  --executor servo-cartesian --dry-run --rate 50 --z-min-mm 0
```

**3. Genesis 镜像 + 真机执行**

连接真机并按同一条规划轨迹运动；Genesis 三维窗口同步显示轨迹与夹持状态。上机前请确认机械臂与夹爪均已就绪：

> **上机前请确认**：机械臂已上电、使能，周围无人员与障碍物；工作空间满足安全高度限制；已完成正/逆运动学对齐检查（`xarm6_reach_deploy.py --mode align`）；急停按钮随手可及。物理夹爪命令默认关闭，仅在夹爪已安装且确认可正常开合后添加 `--real-gripper`。

```bash
# 只运动机械臂，不发送物理夹爪命令。
python examples/xarm6/xarm6_grasp_place_traj.py \
  --executor servo-cartesian --visual --ip 192.168.1.xx --z-min-mm 0 --no-dry-run

# 机械臂 + 物理夹爪。
python examples/xarm6/xarm6_grasp_place_traj.py \
  --executor servo-cartesian --visual --ip 192.168.1.xx --z-min-mm 0 --no-dry-run --real-gripper
```

将 `192.168.1.xx` 换成你的机械臂 IP。

### 展示场景
| 文件 | 说明 |
|------|------|
| `xarm6/xarm6_g2_showcase.py` | xArm6 + Gripper G2 物理装箱演示 |

## 内部模块

以 `_` 前缀的文件是内部共享模块，被多个示例引用：

| 文件 | 用途 |
|------|------|
| `_bootstrap.py` | 将项目根目录加入 `sys.path` |
| `_robot_viewer.py` | 共享 Genesis GLB 查看器核心 |
| `_gripper_demo.py` | Gripper G2 开合控制 |
| `_bio_gripper_g2_demo.py` | Bio Gripper G2 开合控制 |
| `_lite6_gripper_demo.py` | Lite6 夹爪开合控制 |
| `_packaging_scene.py` | 展示场景构建器 |
