# Astribot 第一阶段：latent 条件化残差强化学习

## 最新诊断结果（2026-10-24 更新）

后续方案见 [latent 残差快速效果迭代计划](../astribot_latent_fast_iteration/README.md)：按基线诊断、均值正则、共享残差、偏移辅助学习依次筛选，以确定性成功率改善为首要目标。当前仅完成计划文档，新增实验均待执行。

[latent 偏移实验报告](../../results/astribot_latent_offset_probe_20261023.md)与[复现说明](offset_probe.md)已完成：100 个受控场景、7,500 条观测，以及 40 个原始 VLA 仿真闭环场景；本轮没有重新训练 A/B 策略。

受控测试中加入 latent 后，水平 MAE 从 2.894 降至 0.829 mm；垂直 MAE 从 0.448 增至 0.628 mm。闭环接触前仅使用 latent 的探针达到水平 1.234、垂直 0.914 mm，但融合探针因状态分布偏移而失稳，尚不能确认闭环中相对 A 输入的增量收益，更不能据此宣称策略已学会纠错。相关测试 36 项通过；独立重置的轨迹等价性仍未通过。

以下训练状态是此前阶段记录，不代表本次诊断运行状态；A/B 作业的具体进度以对应运行记录和产物为准。

状态：**B正式100k运行中；已授权按B训练→B val-40→A训练→A val-40串行完成。** 创建/更新日期：2026-10-22。

本轮按工程项目执行，仅使用种子 **20261024**：A/B 各 10k 稳定性检查，通过后各从头训练一次 100k，再各评估 val-40。合计两次正式训练、200000 条正式训练经验和 80 个评估回合；不要求多种子复现或统计显著性。实际阶段和进度见 [本轮运行记录](training_campaign.md)，已完成的 [A/B 10k对比](trial_10k_comparison.md) 可直接查阅。

## 目标与边界

在 Astribot S1 / VIB-820 上，让残差 SAC 利用冻结 VLA-JEPA 的视觉语言 latent、实测关节状态和基座动作，学习适合当前场景的小幅动作修正。基座固定为 step 8000，只更新残差 actor 和 critic。

A 为 `state_only`，B 为 `state_latent`。两组统一显式使用 `--critic-init fan_in --q-lr 3e-5`，其余训练配置一致。修复了旧 critic 初始化与新增输入规模不匹配的问题；奖励、Q 有界输出和 bootstrap 语义保留。mapper、特权 critic、示范正则、TD3、低维残差和真机实验均不自动启动。

## 阅读顺序

| 文档 | 内容 |
|---|---|
| [plan.md](plan.md) | 当前工程目标、固定设计、实施任务及阶段门槛 |
| [validation.md](validation.md) | 单种子 A/B 参数、稳定性检查、工程比较指标和结果模板 |
| [trial_10k_comparison.md](trial_10k_comparison.md) | 完整训练回合、学习稳定性、耗时、资源及结论边界 |
| [training_campaign.md](training_campaign.md) | 本轮协议、进度快照、监督进程、日志和实时产物 |
| [data_contract.md](data_contract.md) | 特征身份、观测布局、时序、精确预算、replay 与恢复契约 |
| [critic_diagnosis.md](critic_diagnosis.md) | 旧版饱和诊断、已实施修复及后续验证边界 |
| [implementation_report.md](implementation_report.md) | 初始接入版本的历史验收与饱和证据，不代表当前配置状态 |
| [taskmemory.md](taskmemory.md) | 当前授权、约束及接续工作要点 |
| [future_work.md](future_work.md) | 不属于当前交付范围的后续方向 |

## 验证状态与完成标准

特征链路已有真实接口比对和 GPU 证据：tokens 为 `[1,32,2048]` float32，新旧动作逐元素一致，池化与重复请求通过。旧配置曾出现 critic 饱和；修复后的两组短 GPU 冒烟各完成 8 次更新，均无零 critic 梯度，保存恢复及七相机确定性评估通过。随后两组10k及恢复均通过，无零critic梯度或精确边界饱和。正式A保持停止；用户随后启动B组从头100k，并进一步授权B完成后先做val-40，再接续A正式训练和val-40；当前A尚未启动接续。尚无最终策略val-40，不能从训练成功率宣布latent收益。

工程验收关注可持续学习、数据与恢复正确、完整评估，以及成功率、耗时、异常接触是否满足本项目使用需要。报告实际结果，不预设任意成功率门槛，也不将单种子 A/B 解释为跨种子算法结论。任何失败门槛触发后都停止并保留证据。

## 历史参考

- [可行性评估](../../research/vlarl_feasibility_20261022.md)：方法背景、实现差距与风险。
- [现有 Astribot 在线 RL](../../astribot_rl/astribot_online_rl.md)：基础执行流程。
- [第三轮保守残差实验](../../astribot_rl/astribot_round3_local_20261020.md)：历史配置及结果。
- [VLaRL 原文](https://arxiv.org/html/2609.30868v1)：方法来源；本阶段不是完整 sim-to-real 复现。

历史基座 23/40、第三轮残差 25/40 只作背景，不替代本轮 A/B。所有历史证据保留，计划预算、运行状态和实测结果分别记录。
