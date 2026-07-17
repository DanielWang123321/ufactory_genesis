# scripts/

用户可用的辅助脚本，以及少量维护者发版工具。资产 vendor / relocalize / combo 生成流水线属于内部维护材料，不随公开面发布。

诊断、关键帧采集与资产再生成脚本保留在本地 `dev/diagnostics/` 工作区或 Git 历史中，供维护者调试参考。

本目录下若还有 `rl_*` 等实验编排脚本，属于本地 RL 验证用途，**暂未整理为公开用户面**，请勿按用户文档依赖。

## 用户脚本

| 脚本 | 模块 | 用途 |
|------|------|------|
| `gen_kinematics_params.py` | `ufactory.kinematics` | 从机器人控制柜提取逐台运动学 YAML（严格 v0.2.5 校准 schema） |
| `observe_hold_current.py` | `ufactory.hardware` | 观察真机 hold 电流/力矩时间序列 |
| `generate_showcase_textures.py` | `ufactory.visualization` / showcase | 生成展示场景贴图与 UV box mesh |

`observe-hold-current` console entry point 由 `ufactory.hardware.observe` 提供（与脚本等效；安装后可直接调用）。

## 维护者脚本

| 脚本 | 模块 | 用途 |
|------|------|------|
| `export_release_evidence_summary.py` | `ufactory.quality.evidence_summary` | 从本地 `project-check` JSON 报告生成脱敏 Release 附件摘要 |
