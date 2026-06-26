# Bio Gripper G2 Assets

柔性仿生夹爪（公共可动模块，可独立使用或搭配 xArm / UF850）。

## 行程（重要）

两指夹持面间距的真实机械范围：**闭合 71 mm → 张开 150 mm**。

- 驱动关节为右指 prismatic dof，左指 `mimic ×-1` 对称跟随。
- **关节零位 = 闭合（71 mm）**，dof 增大到 `0.0395 m` 时张开到 150 mm（单指行程 39.5 mm）。
- 间距与关节值近似线性：`gap ≈ 0.071 + 2 × right_joint_value`（米）。
- 这些数值集中在 `ufactory/bio_gripper_g2.py`（`CLOSE_POS=0`、`OPEN_POS=0.0395`、`STROKE=0.0395`、`CLOSED_GAP=0.071`、`OPEN_GAP=0.150`）以及模板 URDF 的关节 `limit` 中。改行程后需重跑下方两个脚本。

## 模板 URDF

`bio_gripper_g2.urdf` 是从 xarm_ros2 的 xacro 生成的**模板文件**。它的 mesh 路径（`meshes/gripper/bio/*.stl`）来自 ROS 目录结构，**不可直接加载**。它也是关节 `limit`（行程）的唯一真源。

组合 URDF 生成脚本 `scripts/generate_bio_gripper_g2_combo_urdf.py` 会读取此模板提取关节/惯性数据，并生成正确的组合 URDF 与独立可动 URDF。

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

不要在代码中直接加载 `bio_gripper_g2.urdf`（模板）。同一份可动夹爪定义有两种加载入口：

```python
from ufactory.paths import (
    bio_gripper_g2_movable_visual_urdf,  # 独立（无机械臂）
    robot_visual_glb_urdf,               # 搭配机械臂
)

# 1) 独立可动夹爪
urdf = bio_gripper_g2_movable_visual_urdf()

# 2) 搭配 xArm5/6/7 或 UF850（movable=True 取可动版）
urdf = robot_visual_glb_urdf("xarm6_1305", with_bio_gripper_g2=True, movable=True)
# 静态高模（不可动）：去掉 movable=True
```

## 控制（公共模块）

两种加载方式都用同一个控制类 `ufactory.bio_gripper_g2.BioGripperG2`，它会自动发现夹爪关节并镜像左指：

```python
from ufactory.bio_gripper_g2 import BioGripperG2

gripper = BioGripperG2(robot)        # robot 为加载上述任一 URDF 得到的 entity
if gripper.found:
    gripper.setup_pd()
    gripper.open()                   # 张开到 150 mm（OPEN_POS）
    gripper.close()                  # 闭合到 71 mm（CLOSE_POS）

# 开合循环：
for step in range(1000):
    gripper.control_pose(gripper.demo_target(step))
    scene.step()
```

预览命令：

```bash
# 独立
python examples/bio_gripper_g2/view_bio_gripper_g2_movable.py
# 搭配机械臂（开合演示）
python examples/view_robot_glb.py --robot xarm6_1305 --bio-gripper-g2 --movable --gripper-demo
```

静态与可动组合 URDF 的 `bio_gripper_g2_attach` 安装位姿都由 `scripts/relocalize_bio_gripper_g2_glb.py` 写入 `meshes/visual/relocalize_metrics.json`（按 `robot_key` 索引，同 EE link 不同法兰如 uf850 单独计算），再由 `scripts/generate_bio_gripper_g2_combo_urdf.py` 生成到各机械臂组合 URDF。静态 monolithic GLB 与可动 base GLB 都保持在 canonical `bio_gripper_g2_base_link` frame；改安装或网格后需先 relocalize 再 generate。
