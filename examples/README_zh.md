# 按任务组织的示例

[English](README.md)

## 前置条件

已验证 Python 3.12/3.13、Genesis World 1.4.4，Genesis 最低版本为 1.4.2。1.4.2 和 1.4.3 会警告一次；更高版本须通过接口检查。详见[版本策略](../ROADMAP.md)。

1. 克隆本仓库，并在**仓库根目录**运行命令。
2. 使用 editable install（仅支持源码树；不支持 wheel/sdist）：

```bash
git clone https://github.com/xArm-Developer/ufactory_genesis.git
cd ufactory_genesis
pip install -e ".[sim]"

# 可选：
#   pip install -e ".[sim,rl]"        # Linux/NVIDIA 到达与抓放
#   pip install -e ".[sim,real]"      # xArm SDK + Pinocchio/Coal（dry-run / sdk-sim / real 安全预检）
#   pip install -e ".[sim,dynamics]"  # 动力学参考检查
```

3. 运行依赖 Genesis 的流程前，设置可写的 Numba 缓存目录：

```bash
# Linux / macOS
export NUMBA_CACHE_DIR=~/.cache/numba

# Windows PowerShell
# $env:NUMBA_CACHE_DIR="$env:USERPROFILE\.cache\numba"
```

4. 支持的 `--robot`：`xarm5`、`xarm6`、`xarm7`、`uf850`、`lite6`。
5. **Windows / 无可用 GPU：** 先装 CPU 版 PyTorch，再 `pip install -e ".[sim]"`。可视化、抓放、装箱命令加 `--backend cpu`（有显卡但不被 Genesis 支持时，不要依赖 GPU 自动回退）。强化学习需要 Linux/NVIDIA GPU。
6. **Windows 下查看器无报错退出：** 若 `view_robot.py` 在 `Building visualizer...` 之后无窗口直接回到提示符：

```powershell
$env:MKL_THREADING_LAYER = "SEQUENTIAL"
python examples/visualization/view_robot.py --robot xarm6
```

长期生效：`conda env config vars set MKL_THREADING_LAYER=SEQUENTIAL -n <env>`。`--headless` 不受影响。

本版公开目录：


| 目录 | 用途 |
|---|---|
| `visualization/` | GLB 机型与夹爪查看器，见[可视化](#可视化) |
| `kinematics/` | FK/IK / 机型校验包装，见[运动学与动力学仿真](#运动学与动力学仿真) |
| `pick_place/` | 多机型抓放入口与可选覆盖配置，见[抓放指南](pick_place/README_zh.md) |
| `packaging/` | 多机型装箱展示入口与可选覆盖配置，见[装箱指南](packaging/README_zh.md) |
| `rl/` | 裸臂 Reach → 桌面 Pick & Place → 装箱（待实现）；Linux/NVIDIA，见 [RL 索引](rl/README_zh.md) |

学习顺序及训练、推理指南见 [RL 索引](rl/README_zh.md)。

## 支持机型与末端

| 机械臂 | 可用末端 |
|---|---|
| `xarm5` | Gripper G2、Bio Gripper G2、Vacuum Gripper |
| `xarm6` | Gripper G2、Bio Gripper G2、Vacuum Gripper |
| `xarm7` | Gripper G2、Bio Gripper G2、Vacuum Gripper |
| `uf850` | Gripper G2、Bio Gripper G2、Vacuum Gripper |
| `lite6` | Lite6 Gripper、Lite6 Vacuum Gripper |

五类机型均支持仿真与离线预检。真机装箱启用 xArm6 + Gripper G2、Lite6 + Lite6 Gripper，详见[真机操作与硬件校验](#真机操作与硬件校验)。

G2 仿真使用 `g2_calibrated_v1`。策略训练与当前 v3 检查点契约见 [抓放 RL 指南](rl/pick_place/README_zh.md)。

## 可视化

```bash
# xArm6 裸臂静态查看
python examples/visualization/view_robot.py --robot xarm6

# xArm6 + Gripper G2 静态查看
python examples/visualization/view_robot.py --robot xarm6 --gripper-g2

# xArm6 + Gripper G2 开合演示
python examples/visualization/view_robot.py --robot xarm6 --gripper-g2 --movable --gripper-demo

# UF850 + Bio Gripper G2 静态查看
python examples/visualization/view_robot.py --robot uf850 --bio-gripper-g2

# xArm6 + Bio Gripper G2 开合演示
python examples/visualization/view_robot.py --robot xarm6 --bio-gripper-g2 --movable --gripper-demo

# xArm6 + Vacuum Gripper 静态查看
python examples/visualization/view_robot.py --robot xarm6 --vacuum-gripper

# Lite6 + Lite6 Gripper 开合演示
python examples/visualization/view_robot.py --robot lite6 --lite6-gripper --movable --gripper-demo

# Lite6 + Lite6 Vacuum Gripper 静态查看
python examples/visualization/view_robot.py --robot lite6 --lite6-vacuum-gripper

# Gripper G2 独立查看与开合演示
python examples/visualization/view_gripper_g2.py

# Bio Gripper G2 独立查看与开合演示
python examples/visualization/view_bio_gripper_g2.py

# Lite6 Gripper 独立查看与开合演示
python examples/visualization/view_lite6_gripper.py
```


| 参数                                                                                 | 说明                                    |
| ---------------------------------------------------------------------------------- | ------------------------------------- |
| `--robot`                                                                          | 必选机型 key                              |
| `--gripper-g2` / `--bio-gripper-g2` / `--vacuum-gripper` / `--lite6-gripper` / `--lite6-vacuum-gripper` | 末端组合（最多选一个）                           |
| `--movable`                                                                        | 按连杆可动夹爪网格（需配合受支持的夹爪标志）                |
| `--gripper-demo`                                                                   | 循环开合，到位后保持 0.5 s 切换（需要 `--movable`）   |
| `--pd`                                                                             | 平滑关节运动演示（约 0.873 rad/s），仅视觉           |
| `--show-tcp`                                                                       | 法兰 TCP 红色标记（默认隐藏）                     |
| `--diagnose`                                                                       | 无窗口 STL/GLB 连杆位姿诊断                    |
| `--headless`                                                                       | 不打开查看器窗口                              |
| `--backend`                                                                        | `cpu` / `gpu`（默认 `gpu`；无支持显卡时用 `cpu`） |




## 运动学与动力学仿真

```bash
# xArm6 离线 URDF 模型、FK 与 PD 控制自检（不连控制器）
python examples/kinematics/verify_robot.py --robot xarm6

# xArm6 动力学仿真校验（默认姿态 + 5 组随机姿态）
dynamics-sim-check --robot xarm6 --random-count 5
```

> 如需与真实控制器的 FK/IK 对比或硬件动力学检查，请参见 [真机操作与硬件校验](#真机操作与硬件校验)。



## 抓放仿真

多机型抓放的 Genesis 物理仿真与离线预检。参数、更多指令及覆盖配置见[抓放指南](pick_place/README_zh.md)。

```bash
# xArm6 + Gripper G2：抓放离线轨迹与碰撞预检（servo_j）
python examples/pick_place/run.py \
  --robot xarm6 --mode dry-run --executor servo_j

# Lite6 + Lite6 Gripper：抓放物理仿真并打开查看器（servo_cartesian）
python examples/pick_place/run.py \
  --robot lite6 --mode sim --executor servo_cartesian --visual
```

## 装箱展示仿真

多机型轨迹控制装箱。参数及更多指令见[装箱指南](packaging/README_zh.md)。

```bash
# xArm6 + Gripper G2：单轮装箱物理仿真并打开查看器（servo_j）
python examples/packaging/run.py \
  --robot xarm6 --mode sim --executor servo_j --visual
```

## 强化学习

Linux/NVIDIA 上的仿真示例按以下顺序运行：

1. [Reach](rl/reach/README_zh.md)：xArm 6 裸臂到达，成功阈值为 5 mm。
2. [Pick & Place](rl/pick_place/README_zh.md)：xArm6 + Gripper G2 将方块搬运到桌面目标位置。
3. 装箱：将物体放入箱内，RL 示例待实现。

训练、推理指令见 [RL 索引](rl/README_zh.md)。上面的装箱展示采用轨迹控制。

## 真机操作与硬件校验

> [!CAUTION]
> **真机安全规范**：
>
> 1. 执行前需安装真机扩展依赖：`pip install -e ".[real]"`（包含 xArm SDK 与 Pinocchio/Coal 碰撞检测）。
> 2. 真实机械臂物理运动具有碰撞与损坏风险。驱动真机运动必须提供实机逐台标定文件（`--calibration`）并显式传入安全确认门（`--confirm-real`）。缺少任一项均会在连接前直接拒绝执行。
> 3. 真机模式单次运行强制限制为 **1 轮（cycle）**。



### 1. 硬件只读与状态对比指令（不驱动运动）

通过网络读取控制器数据或进行计算对比，真实关节不会产生位移：

```bash
# xArm6：Genesis URDF 与真实控制器 SDK 正运动学（FK）对比
python examples/kinematics/verify_fk.py --robot xarm6 --ip <ip>

# Lite6：Genesis URDF 与真实控制器 SDK 逆运动学（IK）对比
python examples/kinematics/verify_ik.py --robot lite6 --ip <ip>

# xArm6：在控制器虚拟仿真模式下，对预设测试点位进行自碰撞检查（真实关节不动作）
dynamics-sim-collision-check --robot xarm6 --ip <ip>

# Lite6 + Lite6 Gripper：装箱控制器内部虚拟仿真（sdk-sim，真实关节不动作）
python examples/packaging/run.py \
  --robot lite6 --mode sdk-sim --executor servo_cartesian \
  --ip <ip> --calibration path/to/exact.yaml
```



### 2. 真机物理运动指令（实际驱动机械臂与夹爪）

实际下发目标位姿与轨迹控制指令至真实机械臂：

```bash
# xArm6：驱动真机运动至验证姿态并对比关节力矩（需 --confirm-real）
dynamics-hardware-check --robot xarm6 --ip <ip> --confirm-real

# xArm6 + Gripper G2：抓放真机物理运动（需实机标定文件与 --confirm-real）
XARM_IP=<ip> python examples/pick_place/run.py \
  --robot xarm6 --mode real --executor servo_j \
  --calibration path/to/exact.yaml --confirm-real

# Lite6 + Lite6 Gripper：装箱真机物理运动（当前仅支持 xArm6 + Gripper G2、Lite6 + Lite6 Gripper；需实机标定文件与 --confirm-real）
XARM_IP=<ip> python examples/packaging/run.py \
  --robot lite6 --mode real --executor servo_j \
  --calibration path/to/exact.yaml --confirm-real
```

> 注：xArm5、xArm7、UF850 的装箱 `--mode real` 会在连接控制器前自动拦截退出，直到对应机型启用真机夹爪支持。

