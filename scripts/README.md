# scripts/

面向用户的辅助脚本。v0.1.6 源码发布包只保留可直接使用的脚本，资产 vendor / relocalize / combo 生成流水线属于内部维护材料，不随公开面发布。

## 可用脚本

| 脚本 | 用途 |
|------|------|
| `gen_kinematics_params.py` | 从机器人控制柜提取逐台运动学 YAML |
| `generate_showcase_textures.py` | 生成展示场景贴图与 UV box mesh |
| `observe_hold_current.py` | 观察真机 hold 电流/力矩时间序列 |

诊断、关键帧采集与资产再生成脚本保留在本地 `dev/` 工作区或 Git 历史中，供维护者调试参考。
