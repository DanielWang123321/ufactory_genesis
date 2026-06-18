# Lite6 Gripper Assets

Lite6 平行夹爪。

## 模板 URDF

`lite6_gripper.urdf` 来自 xarm_ros2，mesh 路径已被展平但未被 Lite6 组合 URDF 直接引用。组合 URDF 生成脚本会处理路径替换。

## 目录结构

```
lite6_gripper/
├── lite6_gripper.urdf                   # 模板 URDF
├── lite6_gripper_movable_visual.urdf    # 仅夹爪可动视觉 URDF（调试用）
├── meshes/
│   ├── collision/                        # STL 碰撞网格
│   └── visual/                           # GLB 视觉网格
```
