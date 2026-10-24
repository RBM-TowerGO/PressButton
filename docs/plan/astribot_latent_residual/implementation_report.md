# 原始视觉语言残差观测：实现与验收报告

日期：2026-10-24。实现基线：`59281a7`，分支 `astribot/residual-rl-adaptation`。

## 实现结果

已接入 `--observation-mode state_latent_raw`：实测 q32 + qd32 + 原始 token 均值 2048 维，
obs 共 2112 维，Actor 与 Critic 总输入均为 2168 维。特征与基础动作来自同一次冻结 VLA 推理。
无输入 LayerNorm、无投影、无第二次视觉前向。默认保留 `state_only`。

新增特征契约、冻结策略适配、RPC 能力声明、末帧后继采集、checkpoint/replay 语义恢复检查及验收脚本。
特征契约为 v2、`normalization=none`，拒绝 main 的归一化 `state_latent` 及同维度错误契约。
模型运行时快照和 `learner.py` 未改动；SAC 初始化、奖励、网络隐藏层和控制频率保持原样。
两种模式实测 `q_lr=3e-4`、`policy_lr=1e-4`，未引入 main 的 critic 初始化/学习率调整。

## CPU 与真实 GPU 验证

- 观测/runner/learner：71 项通过；控制、任务作业、原始协议/推理回归：37 项通过，合计 **108 项**。
- 覆盖原始特征保留均值与尺度、Actor/Critic 实际输入、错误特征、跨模式和同维度错误契约、
  replay/checkpoint 恢复、成功/失败/超时/动作拒绝的后继、相机故障、推理期间物理变化及不完整数据拒绝提交。
- GPU：RTX 5090，冻结 step 8000；实际 token `[1,32,2048]`，动作头 `torch.float32`。
- 同一真实相机输入、同 seed 下，旧包装层、新包装层和重复调用的 `action8`、`action32`、`normalized_actions` **逐元素完全一致**。
- 每请求仅一次模型调用，特征与原始 float32 均值一致；所有基座参数冻结。GPU 校验含加载耗时约 52.84 秒，峰值模型分配约 10.34 GB。
- 最初误用仿真 Python 环境运行模型校验，因缺少 accelerate 退出；改用项目现有 `.venv-astribot-inference` 后通过，没有安装或修改依赖。

## Isaac 实测

每模式只采集一个 2 秒模拟回合，随后做一个 2 秒确定性 val 回合；不进行长训练。
原 runner 按回合结束后更新，训练各采集 9 条 transition，容量为 8 的环形 replay 保留 8 条，各完成 **8 次学习更新**。
损失、Actor/Critic/target 参数均有限；Critic 梯度范数在此次短冒烟中均非零。

| 项目 | state_only | state_latent_raw |
|---|---:|---:|
| 训练子进程耗时（含 Isaac 启动） | 19.23 s | 17.57 s |
| 评估子进程耗时（含 Isaac 启动） | 21.04 s | 20.73 s |
| 训练控制请求数 | 9 | 9 |
| 额外末帧请求数 | 0 | 1 |
| 额外末帧推理耗时 | 0 | 66.79 ms |
| 训练进程峰值 RSS | 6.26 GB | 6.22 GB |
| 保存 learner + replay 耗时 | 9.90 ms | 23.77 ms |
| replay 单独恢复耗时 | 6.93 ms | 2.82 ms |

raw replay 中每条 next_obs 与 next_base 的 hash 均匹配实际请求；至少一条匹配 `successor_only` 末帧。
末帧来自 reset 前，不执行额外动作，两个训练回合均为 60 控制步、241 物理 tick。
评估从 checkpoint 恢复，参数和更新数不变，learner/replay/status/protocol 四个训练文件 hash 不变。
这次成功/物理失败分支由 CPU 注入测试覆盖；真实短回合均 timeout，不声称覆盖真实按钮按下事件。

额外特征缓存的确定性成本：100k transitions 当前与后继共 **1,638,400,000 bytes（1.53 GiB）**。
表中单回合耗时受加载、缓存等影响，不能用来推断 raw 模式更快或实际内存更少。

## 证据与使用

- [真实模型校验报告](../../../outputs/astribot_online_rl/raw_latent_engineering_20261024/policy_gpu/report.json)
- [Isaac 冒烟结果](../../../outputs/astribot_online_rl/raw_latent_engineering_20261024/smoke/smoke_results.json)
- [源码身份与检查汇总](../../../outputs/astribot_online_rl/raw_latent_engineering_20261024/implementation_identity.json)
- 同目录保留日志、命令、数值证据、请求 hash、checkpoint 和 replay；未覆盖历史实验产物。本次启动的 8799 推理服务已关闭。

训练/评估入口显式增加 `--observation-mode state_latent_raw`，使用新的输出目录；恢复时仍需指定同模式及原训练配置。
复现命令见 [validation.md](validation.md)，字段和兼容规则见 [data_contract.md](data_contract.md)。

**工程接入与短冒烟通过；长期学习稳定性、成功率提升和真机效果未评估。**
