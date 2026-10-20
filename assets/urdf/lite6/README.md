# Lite6 Assets

动力学 URDF：`lite6.urdf`（及 Lite6 Gripper / Vacuum / GLB 组合变体）。

## 碰撞与视觉

```
meshes/lite6/
├── visual/*.stl
└── visual_glb/       # Draco 压缩视觉 GLB（单臂合计约 0.7–1.0 MB）
```

末端配件见同级目录 `lite6_gripper/`、`lite6_vacuum_gripper/`。

机械臂连杆碰撞引用 `visual/*.stl`。Genesis 1.4.4 默认将每个机器人碰撞网格取单凸包后再简化，碰撞查看器显示加载后的几何。单凸包会填平源网格的凹处；源 STL 的准确外形不能直接保证仿真碰撞外形准确。历史与加载处理对照见 [xArm5 碰撞说明](../xarm5/README.md#碰撞网格)。

未引用的历史碰撞简化网格已删除，可从 Git 历史获取。

## Source / License

- URDF / STL 源自上游 xArm ROS / xarm_ros2 家族（见仓库根 [NOTICE](../../../NOTICE)）。
- 本仓库维护组合 URDF、GLB 视觉与压缩流水线；代码 MIT，资产归属见 NOTICE。
