# 裸臂 xArm 6 到达任务

只在仿真里训练。机械臂不装夹爪，每回合把末端伸向一个随机点。仓库不附带训练好的模型，也不能拿这个策略去动真机。

[English](README.md)

目标点在 x 0.15–0.55 m、y -0.30–0.30 m、z 0.05–0.50 m 里均匀抽取。末端离目标小于 3 cm 时本回合结束，否则最长 5 秒。6 维动作 `[-1, 1]` 线性映射为每步 `±0.01 rad`，累积位置指令受 URDF 关节限位约束。

观测为 24 维：关节位置、速度、末端位置、目标相对位置，以及位置指令与实际关节角之差。奖励为 `dt × [exp(-10d) − 0.001‖q̇‖²] + 2 × success`，成功奖励每回合仅一次。PPO 使用 Beta 动作分布、固定学习率 `3e-4` 和观测归一化。环境版本为 2，旧的 18 维检查点需重新训练。

参考环境与抓放示例相同：Linux、NVIDIA GPU、Genesis World 1.4.4、RSL-RL 5.4.2。

```bash
pip install -e ".[sim,rl]"
export NUMBA_CACHE_DIR="$HOME/.cache/numba"

python -m examples.rl.reach.train -e reach
python -m examples.rl.reach.evaluate \
  --checkpoint outputs/rl/reach/reach/model_299.pt \
  --episodes 512 -B 128 --seed 17000 \
  --min-success-rate 0.95 --max-mean-distance-m 0.035 \
  --summary-json outputs/rl/reach/reach/evaluation-17000.json
python -m examples.rl.reach.evaluate \
  --checkpoint outputs/rl/reach/reach/model_299.pt --vis
```

打开查看器时，抵达目标后保持当前姿态 1 秒，再重置并进入下一目标。无窗口评估不停留。

先完成训练，再把 `--checkpoint` 换成该次运行写出的 `model_*.pt`。默认 300 轮，每 25 轮保存检查点，最后为 `model_299.pt`；检查点、同目录 `config.yaml` 和校验清单必须一起保留。

默认 16384 个并行环境，是本机 RTX 4060 Ti（8GB）上这套裸臂场景测到的较快档。显存或时间不够时用 `-B` 改小。续训时 `-B` 要和原运行一致。产物写在 `outputs/rl/reach/`，不提交。

评估使用独立 CPU 随机数生成器抽取固定目标，每个目标仅执行一次，等待全部目标成功或超时。`--seed` 默认 17002，目标与并行环境数无关。JSON 保留全部成功与失败回合；任一验收指标不满足时退出码为 1。开发集可用种子 17002 的 128 目标筛选检查点，验收分别使用种子 17000、17001 的 512 目标。

训练目录保留源码副本、配置与源码/URDF 哈希、逐轮统计、停止原因和检查点。统计按完成回合加权，包含成功率、终止距离、奖励分量、动作饱和率和指令跟踪误差。续训恢复关节位置、速度、位置指令和回合状态。
