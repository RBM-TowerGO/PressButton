# Latent critic 饱和定位（2026-10-22）

## 结论

此次短冒烟可以离线复现。直接触发点是第一个 Adam step 将 critic logits 从约 [-0.13, 0.98] 推至 [-7.22, -5.02]，随后 tanh 下界饱和。初始化尺度、2048 维高度相关输入、3e-4 学习率和有界输出共同构成已复现的失稳路径；尚不能声称解释所有在线训练风险。

生产 learner、奖励、Q 规则及历史产物均未修改。本次新增独立离线诊断脚本和本报告，没有运行仿真或长训练。

## 代码证据

`learner.py::_EnsembleLinear` 的 weight 为 (ensemble,in,out)，forward 使用 x @ weight；但 kaiming_uniform_(weight[i], a=sqrt(5)) 把第二维 out 当作 fan_in。本地 torch.nn.init._calculate_fan_in_and_fan_out 源码确认该行为。现有注释明确这是保留上游初始化约定，因此不是 latent 接入时新写错的代码，而是原有约定在新输入规模下的适配问题。

- 首层 2168→256：实际均匀界限 1/sqrt(256)=0.0625，标准 fan-in 为 1/sqrt(2168)≈0.02148，权重尺度约大 2.91 倍。
- 最后一层 256→1：实际界限 1，标准 fan-in 为 0.0625，权重及 bias 的界限大 16 倍。
- 256→256 中间层没有该差异。
- LayerNorm 令 latent 每坐标方差约 1，总 L2 范数仍约 sqrt(2048)=45.25。真实 8 条数据的 latent 两两余弦相似度为 0.99889–0.99981，更新容易沿共同方向叠加；这不表示 latent 无信息。
- 原首步梯度范数 16.06 小于裁剪阈值 50，裁剪没有触发。Adam 的参数步长不等于 LR 乘原始梯度；只降低裁剪阈值不能可靠限制 logits 的移动。

## 离线对照

输入：最终 state_latent 的真实 replay（8 条）及 checkpoint 的配置；重新初始化网络，不从已饱和权重续训。6 种设置 × 5 个种子（20261024、42–45）× 100 次更新，batch=4，保留 actor 更新及原 Bellman target。所有变体使用相同种子的采样序列。初始化变体通过缩放同一随机权重及 bias 来匹配正确 fan-in 分布，并同步初始 target critic；没有修改 Q 输出函数。

原种子前 8 次的 Q、target、critic 梯度与原日志最大绝对差约 2.9e-5（本脚本固定 CPU 单线程），复现原始坍塌。

| 设置 | 100 次内出现整体 critic 零梯度的种子数 | 原种子首步后 logits 范围 |
|---|---:|---:|
| 原配置，LR=3e-4 | 5/5 | -7.22 ~ -5.02 |
| 仅 LR=3e-5 | 0/5 | -0.84 ~ 0.37 |
| 仅 LR=1e-5 | 0/5 | -0.37 ~ 0.78 |
| 仅首层正确 fan-in，LR=3e-4 | 4/5 | -4.19 ~ -3.03 |
| 仅末层正确 fan-in，LR=3e-4 | 0/5 | -0.46 ~ -0.32 |
| 所有层正确 fan-in，LR=3e-4 | 0/5 | -0.27 ~ -0.19 |

因此，不能将问题仅归因于首层输入维数；末层过大初始化是重要放大因素。降低 LR 或纠正初始化均能在本次数据上阻止整体零梯度。这里统计的是整体零梯度，不能替代逐 critic、逐样本饱和监控。

局限：只有一个短回合，数据高度相关；五个种子只是网络/采样种子，不是五份独立环境数据。不同初始化会改变初始 bootstrap targets，因此这些结果验证数值稳定性，不是同一监督目标下的精度比较。原配置后期 loss 降低也可能来自 target network 随 Polyak 更新坍塌，不能解释为学习成功。

## 建议实施顺序

1. 建立显式初始化配置（legacy / fan_in），保留旧默认和 checkpoint 恢复兼容；新 latent 协议选择正确 fan-in：对 weight[i].T 初始化，bias 用 1/sqrt(in_dim)。同一新协议的 state-only 对照也使用相同初始化规则，避免把初始化收益误算为 latent 收益。
2. 新实验从头初始化 actor/critic/target/optimizer。原已饱和 checkpoint 只作诊断证据，不直接降低 LR 续训。先采用正确 fan-in + critic LR=3e-5 的保守候选，再单独比较 3e-4；本次尚未测试这个组合的在线效果。
3. 添加逐 critic 的更新前后 logits 分位数、输出精确 0/1 比例、tanh 导数、参数更新范数及 Q 对动作的梯度。actor 单独记录价值项梯度，不能用包含熵/正则的总梯度替代。对有显著预测误差且连续零梯度的情况中止训练并保存诊断。
4. 先补固定非零 target 的拟合与动作梯度检查，防止动态 target 坍塌掩盖失败；再使用更多独立回合、计划中的正式 batch/warmup 做短在线 A/B。通过之后才进入 10k 稳定性试验。
5. 暂不替换 tanh、修改奖励或裁剪 target。sigmoid(2*logits) 在数学上等价，不能消除饱和机制；线性 Q 则改变既定协议。持续监控 target 超出 [0,1] 的比例；只有发现目标范围冲突才另行讨论 Q/奖励设计。

## 复现与产物

```bash
.conda/envs/pressb/bin/python scripts/diagnose_astribot_latent_critic.py \
  --source outputs/astribot_online_rl/latent_engineering_20261022/final/state_latent \
  --output outputs/astribot_online_rl/latent_critic_diagnosis_20261022/ablations.json
```

逐步数值：`outputs/astribot_online_rl/latent_critic_diagnosis_20261022/ablations.json`。
简表：`outputs/astribot_online_rl/latent_critic_diagnosis_20261022/summary.txt`。

## 已实施修复（2026-10-22）

- `LearnerConfig.critic_init` 支持 `legacy` / `fan_in`；fan_in 按实际输入维度初始化 critic 各层权重与 bias，target critic 仍从在线 critic 完整复制。
- Astribot runner 新 latent 训练默认 `fan_in`、`q_lr=3e-5`；state-only 默认保持 `legacy`、`3e-4`。可用 `--critic-init` 和 `--q-lr` 显式覆盖。进行方法对照时两组应显式使用 `--critic-init fan_in --q-lr 3e-5`。
- 初始化策略与 critic LR 写入训练/恢复契约及状态；旧 learner checkpoint 缺少初始化字段时使用 legacy，加载权重和优化器不会重置。runner 原有源码 hash 严格检查仍有效，历史实验不能冒充新协议续训；请在新的输出目录从头训练。
- 每次更新增加逐 critic 的更新前 logits 最小/最大值及精确边界比例，方便发现梯度消失。Q 输出、奖励、Bellman target 及动作合成保持原规则。
- 新增五种子的固定非零目标拟合测试，验证每步梯度非零、输出无精确边界、动作梯度存在，以及新配置保存恢复后逐步指标一致；另验证缺少新字段的旧 checkpoint 加载。
- 使用生产实现重新运行真实 replay：5 个种子 × 100 次更新，全部没有零 critic 梯度，逐 critic 精确边界比例全部为 0。原种子首步后 logits 约 [-0.042, 0.012]，100 步后 Q 均值约 0.404。证据：`outputs/astribot_online_rl/latent_critic_diagnosis_20261022/fixed.json`。

复现修复配置：在上面的诊断命令追加 `--variants fixed`，并使用新的 `--output` 路径。上述验证属于 CPU 单元测试及真实数据离线重放，尚未进行修复后的新在线仿真或 10k 训练。
