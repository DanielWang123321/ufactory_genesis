# xArm6 1305 Assets

公开树仅维护 **1305** 变体：动力学 URDF `xarm6_1305.urdf`（及 G2 / bio_gripper / GLB 组合变体）。另保留 `xarm6_with_gripper.urdf` 供轨迹场景 `visual_model='stl'` 几何对齐检查。

## 碰撞网格

```
meshes/xarm6_1305/
├── collision/*.stl   # 注册 URDF 的机械臂选择性多凸包碰撞 STL（22 个凸体）
├── visual/*.stl      # STL 视觉与 CAD 几何参考资源
└── visual_glb/       # Draco 压缩视觉 GLB
```

注册机型 URDF 的机械臂连杆 `<collision>` 使用 `collision/xarm6_link*_col_*.stl`（共 22 个水密凸包，`link_base..link6` 分别为 `2/2/5/5/5/2/1` 个凸体，单凸体 `<= 128` 面）；`visual/*.stl` 保留供 STL `<visual>` 与 CAD 几何参考使用，末端配件按各自资源保留独立 collision STL。

通过选择性结构多凸包分解，`link2`/`link3`/`link4` 的包络外扩 P95 由单凸包的 `40.3/30.0/35.5 mm` 降至 `6.5/5.8/4.6 mm`，消除了 J5 在 `165°～180°`（如 `[90, -90, -60, 0, 175, -90]°`）下的 `link4↔link6` 伪自碰撞与力矩饱和。详见 [xArm5 碰撞说明](../xarm5/README.md#碰撞网格)。

视觉 GLB（`meshes/xarm6_1305/visual_glb/`）经原生 Draco 压缩；完整性测试预算 `< 1.5 MB`（典型约 0.7–1.1 MB）。

未引用的历史碰撞简化网格已删除，可从 Git 历史获取。

## Source / License

- URDF / STL 源自上游 xArm ROS 描述（见仓库根 [NOTICE](../../../NOTICE)；典型 BSD-3-Clause）。
- 本仓库维护组合 URDF、GLB 视觉与压缩流水线；代码 MIT，资产归属见 NOTICE。
