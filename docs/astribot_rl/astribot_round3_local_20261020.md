# Astribot 第三轮：本机单 RTX 5090

冻结 step 8000 基座，重新初始化 SAC，不继承第二轮 actor。第二轮仍在远端验证，尚无最终结果；本轮是降低残差扰动的独立实验，不能预判收益。

| 参数 | 第二轮 | 第三轮 |
|---|---:|---:|
| transitions | 50,000 | 100,000 |
| actor LR | 3e-5 | 1e-5 |
| 固定 alpha | 0.02 | 0.005 |
| pre-tanh 均值正则 | 0.1 | 1.0 |
| 右臂残差上限 rad | 0.02 | 0.01 |
| 夹爪残差上限 rad | 0.05 | 0.025 |

降低残差幅度与熵系数、加强均值正则，以减少对冻结轨迹的扰动；降低 actor 学习率并增加采样预算。warmup=5000、batch=256、gamma=0.995、shaping=1.0、40 s 回合保持不变。seed=20261023，train 划分；完成后七路相机、GPU 物理、val-40 确定性评估，与 23/40 基线配对。

本机基座推理和仿真共享 GPU 0，SAC 在 CPU 更新。数据、模型和输出均存于项目目录所在磁盘。入口 scripts/run_astribot_round3_job.py；输出 outputs/astribot_online_rl/residual_100k_conservative_local_20261020_seed20261023/。

## 自动进度

<!-- ROUND2_LIVE_START -->
更新时间：2026-10-21T07:52:53+08:00（北京时间）

- 阶段：`complete`；监督进程 PID：947351；工作进程 PID：987828。
- 训练 transitions：0 / 100,000；更新：100,000；回合：992。
- 近 20 回合训练成功率：0.7（含探索，仅用于监控）。
- 从启动起平均吞吐：0.00 transition/s；近回合采样吞吐：3.15 transition/s。
- 最新指标：alpha=0.005；动作范数=4.417769908905029；幅度惩罚=0.0002028046437771991；critic loss=0.0023123850114643574。
- val 评估：25 / 40；目标 40 场景。

GPU 采样（index, utilization %, memory MiB, power W）：

```text
0, 0, 13466, 96.82
```

完整状态：`outputs/astribot_online_rl/residual_100k_conservative_local_20261020_seed20261023/progress.json`；日志、checkpoint、replay 和逐回合证据位于同目录。
<!-- ROUND2_LIVE_END -->
