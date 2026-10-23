# xArm7 1305 Assets

动力学 URDF：`xarm7_1305.urdf`（及 G2 / bio_gripper / vacuum_gripper / GLB 组合变体）。

## 碰撞网格

```
meshes/xarm7_1305/
├── collision/*.stl   # 注册 URDF 的机械臂选择性多凸包碰撞 STL（26 个凸体）
├── visual/*.stl      # STL 视觉与 CAD 几何参考资源
└── visual_glb/       # Draco 压缩视觉 GLB
```

注册机型 URDF 的机械臂连杆 `<collision>` 使用 `collision/xarm7_link*_col_*.stl`（共 26 个水密凸包，`link_base..link7` 分别为 `2/2/5/4/5/5/2/1` 个凸体，单凸体 `<= 128` 面）；`visual/*.stl` 保留供 STL `<visual>` 与 CAD 几何参考使用，末端配件按各自资源保留独立 collision STL。详见 [xArm5 碰撞说明](../xarm5/README.md#碰撞网格)。

视觉 GLB（`meshes/xarm7_1305/visual_glb/`）经原生 Draco 压缩，单臂合计约 0.7–1.0 MB 量级。

未引用的历史碰撞简化网格已删除，可从 Git 历史获取。

## Source / License

- URDF / STL 源自上游 xArm ROS 描述（见仓库根 [NOTICE](../../../NOTICE)；典型 BSD-3-Clause）。
- 本仓库维护组合 URDF、GLB 视觉与压缩流水线；代码 MIT，资产归属见 NOTICE。
