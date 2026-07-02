# Gripper G2 Assets

标准 xArm/UF850 平行夹爪。

## 模板 URDF

`gripper_g2.urdf` 是从 xarm_ros2 的 xacro 生成的 standalone 模板文件；v0.1.6 中 mesh 路径已改为本仓库内的 `meshes/collision/*.stl`，可用于基础结构解析。

实际机械臂搭配仍应优先使用 `robot_visual_glb_urdf(..., with_gripper_g2=True)` 返回的组合 URDF。

## 目录结构

```
gripper_g2/
├── gripper_g2.urdf                   # standalone 模板 URDF
├── gripper_g2_movable_visual.urdf    # 仅夹爪可动视觉 URDF（调试用）
├── meshes/
│   ├── collision/                    # STL 碰撞网格
│   │   ├── base_link.stl
│   │   ├── left_finger.stl
│   │   ├── left_inner_knuckle.stl
│   │   ├── left_outer_knuckle.stl
│   │   ├── right_finger.stl
│   │   ├── right_inner_knuckle.stl
│   │   └── right_outer_knuckle.stl
│   └── visual/                       # GLB 视觉网格
│       ├── gripper_g2_static_*.glb   # 合并静态 GLB（分 link5/link6/link7）
│       └── visual_glb/               # 可动分体 GLB
```

## 加载方式

机械臂场景使用组合 URDF：

```python
from ufactory.paths import robot_visual_glb_urdf

# 静态 GLB
urdf = robot_visual_glb_urdf("xarm6_1305", with_gripper_g2=True)

# 可动 GLB
urdf = robot_visual_glb_urdf("xarm6_1305", with_gripper_g2=True, movable=True)
```
