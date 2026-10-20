# xArm5 1305 Assets

动力学 URDF：`xarm5_1305.urdf`（及 G2 / bio_gripper / GLB 组合变体）。

## 碰撞网格

```
meshes/xarm5_1305/
├── visual/*.stl      # 注册 URDF 的 arm collision 与 STL 视觉资源
└── visual_glb/       # Draco 压缩视觉 GLB
```

注册机型 URDF 的机械臂连杆 `<collision>` 使用 `visual/*.stl`；末端配件按各自资源保留独立 collision STL。

这里的“视觉 STL”指 `visual/link*.stl` 外观网格，保留连杆曲面与凹处；碰撞查看器显示的是加载后生成的凸包。精细材质外观使用独立的 `visual_glb/link*.glb`。例如 link3 的视觉 STL 为 11,602 面，历史简化 STL 为 4,120 面，Genesis 1.4.4 默认加载后的碰撞网格约 539 面。

该引用自 v0.2.1（`56ece26`，2026-07-08）替换厂家简化 STL，原意是贴合连杆曲面。Genesis 1.4.4 默认启用 `convexify=True`，且 `decompose_robot_error_threshold=inf`，每个机器人碰撞网格取单凸包后再简化。碰撞查看器显示加载后的几何；单凸包会填平弯曲连杆的凹处，源 STL 的准确外形不能直接保证仿真碰撞外形准确。

xArm5 姿态 `[90, -90, -60, 160, -90]°` 下，link3/link5 原始视觉 STL 间距为 35.658 mm，默认加载后的碰撞几何却穿透约 8.581 mm。历史简化 STL 的原始间距为 35.408 mm，默认加载后仍穿透约 10.186 mm。因此仅恢复历史 STL 不能消除伪碰撞，需要保留凹形或使用多个凸体表示同一连杆。

视觉 GLB（`meshes/xarm5_1305/visual_glb/`）经原生 Draco 压缩，单臂合计约 0.7–1.0 MB 量级。

未引用的历史碰撞简化网格已删除，可从 Git 历史获取。

## Source / License

- URDF / STL 源自上游 xArm ROS 描述（见仓库根 [NOTICE](../../../NOTICE)；典型 BSD-3-Clause）。
- 本仓库维护组合 URDF、GLB 视觉与压缩流水线；代码 MIT，资产归属见 NOTICE。
