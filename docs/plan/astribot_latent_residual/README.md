# Astribot 第一阶段：latent 条件化残差强化学习

状态：**critic 修复后分阶段训练已启动；当前进度以运行记录为准**。创建/更新日期：2026-10-22。

用户已授权冒烟 → A/B各10k → 通过后三种子100k及val-40，详见 [本轮协议与实时产物](training_campaign.md)。以下早期验收记录保留为历史证据，不代表当前训练已完成。

本目录包含原始设计与本次工程实现证据。用户将本次范围限定为代码接入、CPU 契约测试和真实 GPU 冒烟；已完成 101 项 CPU 测试、真实接口比对及两模式短冒烟。latent critic 出现饱和，未开展 10k 或正式训练，不声明性能提升。详见 [实施与验收记录](implementation_report.md)。

## 目标

在当前 Astribot S1 / VIB-820 主线上，让残差 SAC 同时读取冻结 VLA-JEPA 的视觉语言 latent、实测关节状态和基座动作，学习“根据当前场景如何修正”。基座固定为 step 8000，RL 只更新残差 actor 和 critic。

第一阶段自身是完整可训练、可评估的方案。**只接入 latent，保留当前训练设置**；奖励塑形、Q 值范围和终止语义问题单列诊断，不在本阶段顺便修正。后续优化全部列为未来待定，不自动启动。

## 阅读顺序

| 文档 | 内容 |
|---|---|
| [implementation_report.md](implementation_report.md) | 已实现内容、C01–C12 证据、GPU 结果、风险和复现入口 |
| [plan.md](plan.md) | 背景、固定设计、实施任务及工程验收 |
| [data_contract.md](data_contract.md) | 特征来源、RPC、观测布局、时序、replay 和恢复契约 |
| [validation.md](validation.md) | 契约测试、GPU 冒烟、对照实验、风险诊断和结果模板 |
| [future_work.md](future_work.md) | 后续方向、启动条件与范围边界 |

## 证据与参考

- [可行性评估](../../research/vlarl_feasibility_20261022.md)：论文与当前实现差距、最新结果和风险分析。
- [现有 Astribot 在线 RL](../../astribot_rl/astribot_online_rl.md)：基础执行与训练流程；其中早期状态不代表本计划已完成。
- [第三轮保守残差实验](../../astribot_rl/astribot_round3_local_20261020.md)：本阶段训练设置的参考。
- [VLaRL 原文](https://arxiv.org/html/2609.30868v1)：方法来源。本阶段是借鉴 latent 条件化思路，不是完整 sim-to-real 复现。

历史验证基座为 23/40，第三轮保守残差为 25/40，配对改善 3、退步 1，精确 McNemar p=0.625；这些结果不证明 latent 有效，也不替代本计划重新运行的无 latent 对照。PiPER 的历史结果不作为 Astribot 验收目标。

## 完成的含义

工程完成与算法有效分开记录：接口和数据链路可以全部通过，而效果仍可能是“证据不足”或“出现退化”。原计划中的“初始状态”保留为历史基线；当前实测状态以实施与验收记录为准。10k、正式对照和完整 val-40 均未执行。不得用计划参数填充实测结果。
