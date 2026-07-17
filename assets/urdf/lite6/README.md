# Lite6 Assets

动力学 URDF：`lite6.urdf`（及 Lite6 Gripper / Vacuum / GLB 组合变体）。

## 碰撞与视觉

```
meshes/lite6/
├── collision/*.stl
├── visual/*.stl
└── visual_glb/       # Draco 压缩视觉 GLB（单臂合计约 0.7–1.0 MB）
```

末端配件见同级目录 `lite6_gripper/`、`lite6_vacuum_gripper/`。

## Source / License

- URDF / STL 源自上游 xArm ROS / xarm_ros2 家族（见仓库根 [NOTICE](../../../NOTICE)）。
- 本仓库维护组合 URDF、GLB 视觉与压缩流水线；代码 MIT，资产归属见 NOTICE。
