# Gripper G2 Assets

标准 xArm/UF850 平行夹爪。

## 零件 URDF

`gripper_g2_movable_visual.urdf` 是唯一零件 URDF：standalone 查看直接加载（`python examples/visualization/view_gripper_g2.py`，`world` 预览连杆抬高 0.12 m），搭配机械臂时由 `ufactory.robots.assembly` 去掉 `world` 后挂到法兰（见 [`assembly.yaml`](../assembly.yaml)）。厂商模板 `gripper_g2.urdf` 已删除，历史版本见 Git。

## 目录结构

```
gripper_g2/
├── gripper_g2_movable_visual.urdf    # 零件 URDF（组合源 + standalone 预览）
├── meshes/
│   ├── collision/                    # STL 碰撞网格（保持原密度，不做抽面）
│   │   ├── base_link.stl
│   │   ├── left_finger.stl / right_finger.stl
│   │   ├── left_inner_knuckle.stl / right_inner_knuckle.stl
│   │   └── left_outer_knuckle.stl / right_outer_knuckle.stl
│   └── visual/                       # GLB 视觉网格（法兰中心 canonical + 原生 Draco）
│       ├── gripper_g2_static.glb     # 合并静态高模（全机型共用）
│       └── visual_glb/               # 可动分体 GLB（base + fingers/knuckles）
```

视觉面数保持不变，用原生 `draco_encoder` 压 **视觉 GLB**（合计目标 **< 1 MiB**）。所有视觉 GLB 与碰撞 STL 均以官方 SolidWorks 模型（`Gripper G2 AG1200 20250710.STEP`）法兰安装面 `Z = 0`、法兰回转轴 `(0, 0, 0)` 为基准（定位凸台底面 `Z = -5.0 mm`，底座顶面 `Z = 104.5 mm`，外指节销孔 `Z = 0.0606 m`，内指节销孔 `Z = 0.075602 m`，`joint_tcp` `Z = 0.1735 m`）；零件及全部组合输出的 `<visual><origin>` 与 `<collision><origin>` 均为 `0 0 0`。

## 加载方式

机械臂场景运行时组合（输出到 `.cache/ufactory/urdf/composed/`）：

```python
from ufactory.robots.assembly import compose_robot_urdf

urdf = compose_robot_urdf("xarm6_1305", "gripper_g2")                          # 可动 GLB
urdf = compose_robot_urdf("xarm6_1305", "gripper_g2", gripper_visual="static")  # 静态 GLB
urdf = compose_robot_urdf("xarm6_1305", "gripper_g2", visual="stl")             # STL 视觉
```

运行配置中写作 `urdf: "assembly:gripper_g2"`。

## Source / License

- 模板与网格源自上游 xArm ROS / xarm_ros2 家族（见仓库根 [NOTICE](../../../NOTICE)）。
- 本仓库维护零件 URDF、共享 `visual_glb/`（法兰安装面基准）与 Draco 压缩 GLB。
