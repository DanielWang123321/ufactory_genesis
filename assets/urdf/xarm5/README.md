# xArm5 1305 Assets

动力学 URDF：`xarm5_1305.urdf`（及 G2 / bio_gripper / vacuum_gripper / GLB 组合变体）。

## 碰撞网格

```
meshes/xarm5_1305/
├── collision/*.stl   # 注册 URDF 的机械臂选择性多凸包碰撞 STL（19 个凸体）
├── visual/*.stl      # STL 视觉与 CAD 几何参考资源
└── visual_glb/       # Draco 压缩视觉 GLB
```

注册机型 URDF 的机械臂连杆 `<collision>` 使用 `collision/xarm5_link*_col_*.stl`（共 19 个水密凸包，`link_base..link5` 分别为 `2/2/5/7/2/1` 个凸体，单凸体 `<= 128` 面）；`visual/*.stl` 保留供 STL `<visual>` 与 CAD 几何参考使用，末端配件按各自资源保留独立 collision STL。

Genesis 1.4.4 默认启用 `convexify=True` 且 `decompose_robot_error_threshold=inf`，会对单个碰撞网格直接取单凸包并填平弯曲连杆凹面（例如单凸包下 `link2`/`link3` 外扩 P95 达 `40.1/45.3 mm`，在姿态 `[90, -90, -60, 160, -90]°` 下引发 `link3↔link5` 伪自碰撞与 J4 力矩饱和）。通过在 URDF 中将凹连杆预分解为紧贴表面的多个水密凸包，外扩 P95 降至 `6.4/7.2 mm`，消除该姿态伪碰撞并保持纯凸 SupportField+MPR 快速路径。

视觉 GLB（`meshes/xarm5_1305/visual_glb/`）经原生 Draco 压缩，单臂合计约 0.7–1.0 MB 量级。

未引用的历史碰撞简化网格已删除，可从 Git 历史获取。

## Source / License

- URDF / STL 源自上游 xArm ROS 描述（见仓库根 [NOTICE](../../../NOTICE)；典型 BSD-3-Clause）。
- 本仓库维护组合 URDF、GLB 视觉与压缩流水线；代码 MIT，资产归属见 NOTICE。
