# xArm / UF850 Vacuum Gripper Assets

xArm 系列（`xArm5`、`xArm6`、`xArm7`）与 `UF850` 通用五吸杯真空吸头静态配件，无开合活动关节。

## 参数规格

- **质量**：`0.610 kg`（实测值）
- **质心（相对法兰安装面）**：`[0, 0, 0.053] m`（实测值）
- **惯性张量（在质心处，与法兰轴向一致，单位 $\text{kg}\cdot\text{m}^2$）**：
  - `ixx = 0.00047106`, `ixy = 3.9292e-7`, `ixz = 2.6537e-6`
  - `iyy = 0.00033072`, `iyz = -1.0975e-5`, `izz = 0.00025642`
- **TCP 偏移（相对法兰安装面）**：`[0, 0, 0.126] m`，方向与法兰一致（关节 `vacuum_joint_tcp` -> 连杆 `vacuum_tcp`）
- **最大吸附负载能力**：`5.0 kg`（能力参数，不计入工具自身质量）

## 目录结构与碰撞网格

```text
vacuum_gripper/
├── vacuum_gripper.urdf                    # 独立吸头 URDF（GLB 视觉 + 33 个水密凸包碰撞）
├── vacuum_gripper_stl.urdf                # 独立吸头 STL 视觉变体 URDF
├── meshes/
│   ├── collision/
│   │   └── convex/hull_000_*.stl .. hull_032_*.stl  # 33 个分区水密凸包（11,358 面）
│   └── visual/
│       ├── vacuum_gripper.glb             # Draco 压缩高模视觉 GLB（209,604 面、5 组材质，~540 KB）
│       └── vacuum_gripper_visual.stl      # 官方原始视觉 STL（20,544 面）
```

碰撞几何按法兰、主体、上盖及五组支柱与吸杯分区裁剪官方视觉 STL 生成 **33 个独立水密凸包**（`hull_000_flange.stl` 至 `hull_032_cup_05.stl`），相对官方视觉 STL 外轮廓双向偏差 P95 `0.384 mm`、最大值 `0.797 mm`，完整保留五组吸杯间空隙与波纹轮廓。

## 加载与预览方式

```python
from ufactory.robots.paths import (
    robot_visual_glb_urdf,
    vacuum_gripper_urdf,
)

# 1) 独立真空吸头
urdf = vacuum_gripper_urdf()

# 2) 搭配 xArm5 / xArm6 / xArm7 / UF850
urdf = robot_visual_glb_urdf("xarm6_1305", with_vacuum_gripper=True)
```

命令行可视化预览：

```bash
python examples/visualization/view_robot.py --robot xarm6 --vacuum-gripper
python examples/visualization/view_robot.py --robot uf850 --vacuum-gripper
```

## Source / License

- 原始 URDF 参数与 STL 源自上游 [xArm-Developer/xarm_ros2](https://github.com/xArm-Developer/xarm_ros2)（BSD-3-Clause，见仓库根 [NOTICE](../../../NOTICE)）。
- 本仓库维护 Draco 压缩 GLB、33 个分区水密凸包碰撞 STL 及多机型组合 URDF。
