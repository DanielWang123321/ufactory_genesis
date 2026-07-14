# scripts/

面向用户的辅助脚本。v0.2.5 公开源码包只保留可直接使用的脚本，资产 vendor / relocalize / combo 生成流水线属于内部维护材料，不随公开面发布。

## 可用脚本

| 脚本 | 模块 | 用途 |
|------|------|------|
| `gen_kinematics_params.py` | `ufactory.kinematics` | 从机器人控制柜提取逐台运动学 YAML（严格 v0.2.5 校准 schema） |
| `observe_hold_current.py` | `ufactory.hardware` | 观察真机 hold 电流/力矩时间序列 |
| `generate_showcase_textures.py` | `ufactory.visualization` / showcase | 生成展示场景贴图与 UV box mesh |
| `migrate_legacy_checkpoint.py` | `ufactory.training` | 将可信旧版 `cfgs.pkl` 迁移为 v0.2.5 安全 YAML/清单；仅在 `--trusted-input` 真实适用时使用 |

`observe-hold-current` console entry point 由 `ufactory.hardware.observe` 提供。

诊断、关键帧采集与资产再生成脚本保留在本地 `dev/diagnostics/` 工作区或 Git 历史中，供维护者调试参考。
