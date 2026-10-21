# Lite6 Assets

动力学 URDF：`lite6.urdf`（及 Lite6 Gripper / Vacuum / GLB 组合变体）。

## 碰撞与视觉

```
meshes/lite6/
├── collision/*.stl   # 注册 URDF 的机械臂选择性多凸包碰撞 STL（17 个凸体）
├── visual/*.stl      # STL 视觉与 CAD 几何参考资源
└── visual_glb/       # Draco 压缩视觉 GLB（单臂合计约 0.7–1.0 MB）
```

末端配件见同级目录 `lite6_gripper/`、`lite6_vacuum_gripper/`。

机械臂连杆 `<collision>` 使用 `collision/lite6_link*_col_*.stl`（共 17 个水密凸包，`link_base..link6` 分别为 `3/2/4/2/4/1/1` 个凸体，单凸体 `<= 128` 面），消除零位 `qpos0` 下底座与腕部的单凸包伪自碰撞及紧凑折叠姿态力矩饱和。详见 [xArm5 碰撞说明](../xarm5/README.md#碰撞网格)。

未引用的历史碰撞简化网格已删除，可从 Git 历史获取。

## Source / License

- URDF / STL 源自上游 xArm ROS / xarm_ros2 家族（见仓库根 [NOTICE](../../../NOTICE)）。
- 本仓库维护组合 URDF、GLB 视觉与压缩流水线；代码 MIT，资产归属见 NOTICE。
