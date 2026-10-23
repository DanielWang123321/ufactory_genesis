# 装箱展示仿真

[English](README.md) · [示例指南](../README_zh.md)

多机型轨迹控制装箱的 Genesis 物理仿真与离线预检。安装方式及运行目录见[前置条件](../README_zh.md#前置条件)。

`examples/packaging/run.py` 委托给 `ufactory-packaging-showcase`。

方块在箱口上方 50 mm 释放后自然下落。GPU 回归允许弹跳，以最终碰撞包围盒位于箱内为准（接触容差 1 mm），不再要求落点距目标中心小于 25 mm；保留抓取、搬运、释放与回零检查。

## 参数

| 参数                                         | 取值 / 说明                            |
| ------------------------------------------ | ---------------------------------- |
| `--robot`                                  | 默认 `xarm6`；五机型均支持 sim / dry-run    |
| `--mode`                                   | 仿真必选：`sim`（默认）、`dry-run`（离线预检）     |
| `--executor`                               | 默认 `servo_j`；也可选 `servo_cartesian` |
| `--backend`                                | 可选：`cpu` / `gpu`；含义与[抓放](../pick_place/README_zh.md#参数)相同 |
| `--cycles N`                               | 精确仿真轮数（默认 1）                       |
| `--speed`                                  | 仿真播放倍率（`>1` 更快）                    |
| `--table-height`                           | 仅覆盖仿真展示高度，不改变基座系几何                 |
| `--visual`                                 | 可选：保持最终查看器画面                       |
| `--config` / `--print-config` / `--report` | 含义与[抓放](../pick_place/README_zh.md#参数)相同 |

## 运行指令

```bash
# xArm6 + Gripper G2：单轮装箱物理仿真并打开查看器（servo_j）
python examples/packaging/run.py \
  --robot xarm6 --mode sim --executor servo_j --visual

# xArm6 + Gripper G2：使用 CPU 后端运行单轮装箱物理仿真
python examples/packaging/run.py \
  --robot xarm6 --mode sim --executor servo_j --backend cpu

# Lite6 + Lite6 Gripper：连续运行 3 轮装箱物理仿真
python examples/packaging/run.py \
  --robot lite6 --mode sim --executor servo_j --cycles 3

# xArm7 + Gripper G2：装箱离线轨迹与碰撞预检
python examples/packaging/run.py \
  --robot xarm7 --mode dry-run --executor servo_j

# xArm6 + Gripper G2：使用局部配置覆盖文件运行装箱物理仿真
python examples/packaging/run.py \
  --robot xarm6 --mode sim --executor servo_j \
  --config examples/packaging/runtime.example.yaml
```

## 真机

`sdk-sim` 与 `real` 模式见[真机操作与硬件校验](../README_zh.md#真机操作与硬件校验)。真机装箱当前启用 xArm6 + Gripper G2、Lite6 + Lite6 Gripper。
