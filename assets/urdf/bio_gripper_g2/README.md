# Bio Gripper G2 Assets

柔性仿生夹爪（xArm/UF850 共用）。

## 模板 URDF

`bio_gripper_g2.urdf` 是从 xarm_ros2 的 xacro 生成的**模板文件**。它的 mesh 路径（`meshes/gripper/bio/*.stl`）来自 ROS 目录结构，**不可直接加载**。

组合 URDF 生成脚本 `scripts/generate_bio_gripper_g2_combo_urdf.py` 会读取此模板提取关节/惯性数据，并生成正确的组合 URDF。

## 目录结构

```
bio_gripper_g2/
├── bio_gripper_g2.urdf                      # 模板 URDF（勿直接加载）
├── bio_gripper_g2_movable_visual.urdf       # 仅夹爪可动视觉 URDF（调试用）
├── meshes/
│   └── visual/                           # GLB 视觉网格
│       ├── bio_gripper_g2_visual_*.glb   # 合并静态 GLB（分 link5/link6/link7）
│       ├── visual_glb/*/bio_gripper_g2_*.glb  # 可动分体 GLB（分 link5/link6/link7）
│       ├── left_finger.stl / right_finger.stl / link_base.stl  # 参考 STL
│       ├── relocalize_metrics.json        # 重定位指标
│       └── visual_glb_src/bio_gripper_g2.glb  # 源 CAD GLB
```

## 加载方式

不要在代码中直接加载 `bio_gripper_g2.urdf`。使用组合 URDF：

```python
from ufactory.paths import robot_visual_glb_urdf

urdf = robot_visual_glb_urdf("xarm6_1305", with_bio_gripper_g2=True)
```
