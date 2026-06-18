# scripts/

维护与开发辅助脚本。

## 资产流水线

生成和维护机器人 URDF 资产：

| 脚本 | 用途 |
|------|------|
| `vendor_robot_assets.py` | 从 xarm_ros2 克隆并生成基础 URDF + STL，迁移源 CAD GLB |
| `relocalize_arm_glb.py` | 将 arm CAD GLB 对齐到 STL 参考坐标系 |
| `relocalize_gripper_glb.py` | 将 Gripper G2 GLB 对齐 |
| `relocalize_bio_gripper_glb.py` | 将 Bio Gripper G2 GLB 对齐 |
| `relocalize_lite6_gripper_glb.py` | 将 Lite6 Gripper GLB 对齐 |
| `relocalize_lite6_vacuum_gripper_glb.py` | 将 Lite6 Vacuum GLB 对齐 |
| `generate_gripper_g2_combo_urdf.py` | 生成臂体 + Gripper G2 组合 URDF |
| `generate_bio_gripper_combo_urdf.py` | 生成臂体 + Bio Gripper G2 组合 URDF |
| `generate_lite6_gripper_combo_urdf.py` | 生成 Lite6 + 平行夹爪组合 URDF |
| `generate_lite6_vacuum_gripper_combo_urdf.py` | 生成 Lite6 + 真空吸盘组合 URDF |
| `generate_lite6_physics_combo_urdf.py` | 生成 Lite6 物理仿真组合 URDF |
| `generate_showcase_textures.py` | 生成展示场景贴图 |

## 验证

| 脚本 | 用途 |
|------|------|
| `verify_gripper_g2_assets.py` | 验证 Gripper G2 组合 URDF 和重定位指标 |
| `verify_lite6_gripper_assets.py` | 验证 Lite6 Gripper 资产 |

## 运动学校准

| 脚本 | 用途 |
|------|------|
| `gen_kinematics_params.py` | 从机器人控制柜提取逐台运动学 YAML |

## 关键帧采集

| 脚本 | 用途 |
|------|------|
| `capture_gripper_keyframes.py` | 无头截取夹爪开合关键帧 |
| `capture_bio_gripper_keyframes.py` | 无头截取生物夹爪关键帧 |
| `capture_showcase_keyframes.py` | 无头截取展示场景关键帧 |
| `capture_startup_motion_keyframes.py` | 截取启动运动关键帧 |

## 诊断工具

> 这些是开发调试工具，普通用户无需使用。

| 脚本 | 用途 |
|------|------|
| `diagnose_gripper_*.py` | 夹爪 mesh 诊断（框架、方向、部件、壳体、指节） |
| `diagnose_knuckle_*.py` | 指节 Genesis 调试 |
| `analyze_gripper_assembly.py` | 夹爪装配分析 |
