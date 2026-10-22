# VLaRL 在 PressButton 的可行性评估

评估日期：2026-10-22。基于论文全文、当前源码、模型运行时快照和本地实验产物；未运行新的 RL 或真机实验。本文区分已有证据、代码推断与建议实验，不把设计预期当作实测收益。

结论：适合先在 **Astribot S1 / VIB-820 当前主线**实现 latent-conditioned residual RL，再单独评估 sim-to-real。工程入口已具备，效果尚待验证。它训练冻结 VLA 外面的修正策略，不直接强化更新 VLA 权重。历史 PiPER 的 119/120 不能用来预测 S1 收益。

## 论文中值得迁移的部分

原文：[VLaRL, arXiv:2609.30868v1](https://arxiv.org/html/2609.30868v1)，重点为 III-C–E、IV-B、V-B、VI-C。

论文冻结 VLA，以其视觉语言 latent、基座动作、本体状态及腕部力为残差输入；仿真用 TD3、特权 critic、示范残差正则训练。Mapper 在 token 池化前用带轨迹进度约束的 Sinkhorn 对齐 sim→real，**同时供给动作头和残差策略**；真机直接用真实 latent，移除 mapper。Flower 按键真机结果为 67.5%→100%，每条件 40 次，去掉 latent 后为 70%。这些结果不是对 S1 的性能承诺。Mapper 需要近似对应的真实/仿真轨迹，也不能解决接触动力学差异。

## 项目证据与差距

| 项目 | 当前证据 | 对本方案的意义 |
|---|---|---|
| 任务与数据 | 400 条仿真专家示范；冻结 train/val/test=320/40/40；头部和右腕 RGB | 可做仿真残差训练；不能充当真实 latent 对齐数据 |
| 基座 | VLA-JEPA，Qwen3-VL-2B + DiT-B，step 8000 | 继续冻结，不需先换 Flower/GR00T 或仿真器 |
| 动作 | 每次预测 7×8 绝对关节目标；右臂 7 关节和夹爪 | 残差应先沿用关节坐标，不照搬论文归一化笛卡尔尺度 |
| actor | 64 维 q/qd，加 learner 内部拼接的 56 维基座 chunk | 已有动作条件，但没有直接视觉语言特征 |
| critic | 与 actor 使用同类观测；评价合成后的动作；Q 限制在 (0,1) | 尚无独立特权状态接口；塑形回报需要重新审计数值范围 |
| 物理反馈 | runner 用按钮行程、接触力、灯和异常接触评分 | 这些仿真信息不应自动进入部署 actor；尤其不能把目标接触报告当作腕部力传感器 |
| 最新结果 | 第三轮 val 25/40，基线 23/40；改善 3、退步 1，McNemar p=0.625 | 收益尚不显著，值得验证信息不足而非只加训练步数 |

代码依据：

- `src/pressb/online_rl/astribot.py:57`：q/qd observation；`:68`：joint8 残差合成。
- `src/pressb/online_rl/learner.py`：`SACLearner.__init__`、`_actor_input` 说明 actor 实际输入还包含基座 chunk；`_Critics.forward` 为有界 Q。
- `scripts/run_astribot_online_rl.py`：`Recorder.observe`、`distance`、`run_episode`、`build_learner`。
- `models/astribot_head_right_joint8_ft_20261018/runtime/starVLA/model/framework/VLA_JEPA.py:323`：推理及 latent 返回。
- 同一 runtime 下 `starVLA/inference/astribot_policy.py:49`：包装层只提取 `normalized_actions`，丢弃 latent。
- 最新结果：`outputs/astribot_online_rl/residual_100k_conservative_local_20261020_seed20261023/run/eval/summary.json` 与 `paired_comparison.json`。后者字段仍叫 `round2_successes`，但文件属于第三轮目录。

README 中第三轮前的状态不能代表最新结果。第三轮监督页显示 transitions=0 是报告字段问题：`describe` 在 eval 的 replay=None 时写 0；应结合训练 metrics/checkpoint 查计数，不能据此说没有训练。验证文件明确 completed=40、successes=25、异常接触合计为 0。

## 最小可行方案：先补 latent，不做 mapper

当前基座在仿真示范上训练，近期目标也可先限定为仿真闭环提升；没有真实数据时，先用恒等映射。应称为“借鉴 VLaRL 的 latent-conditioned residual SAC”，不是完整复现。

1. **复用已有特征。** `VLA_JEPA.predict_action` 已返回 `embodied_action_tokens`，而且这些 token 正是动作头的输入。配置为 32 tokens、2048 hidden width，预期形状 `[B,32,2048]`，需接口冒烟实测确认。包装层保留返回字典，沿 token 维 mean-pool，随 action8 一起通过现有 NumPy RPC 返回。无需第二次 VLM 前向，也无需启动 JEPA 世界模型编码器。这是动作条件表示，与论文使用的 token 类型并不完全相同，须做消融验证。
2. **输入扩展。** 第一版用 `obs = concat(q32, qd32, pooled_latent2048)`，obs_dim=2112；learner 自动再接 56 维基座动作，actor 总输入为 2168。对 q、qd、latent 分别处理尺度；latent 可用 LayerNorm 或训练集拟合并冻结的统计量。首轮不压缩特征，后续再比较 2048→256 投影。
3. **保留动作契约。** `A_exec = A_VLA + scale * tanh(residual)`，先继续输出 7×8。首轮沿用第三轮的小尺度与固定低熵设置，降低同时改变多个因素造成的混淆。论文的残差尺度不能直接换成 rad。
4. **扩展 replay 和版本。** 缓存当前及下一次真实请求的 pooled latent，与 q、图像、基座动作使用相同观测时刻；不能用未来帧或重采样的基座 chunk 代替。obs_dim 改变后重新初始化 actor/critic/replay，不直接恢复旧残差 checkpoint。保存特征类型、模型 hash、归一化、动作单位和协议版本。
5. **保留基座与历史快照。** 新建适配层/实验 runtime，不原地改写已经用于结果复现的模型源码快照。验收“加特征接口后同图像、状态、seed 的 action8 不变”，以及零残差确实恢复冻结策略。

100k transitions 同时存储当前/下一状态的 2048 维 float32 latent，额外约 1.64 GB（1.53 GiB），不含其他字段。没有必要在线 replay 缓存全部 32×2048 tokens；完整 tokens 留给未来 mapper 离线数据。若投影随训练更新，应缓存原始 pooled latent 并在采样时投影，避免存储过期投影结果。

## 在增加训练预算前必须核对的细节

**奖励和 critic 的取值范围。** 当前残差 critic 用 `0.5*(tanh(logits)+1)` 限制在 (0,1)，与原有稀疏成功回报较匹配；加入势函数塑形后不保证仍匹配。应记录折扣 return、Bellman target 的分位数、超出 [0,1] 的比例及 Q 饱和比例，再决定改无界 critic 或重新设计/缩放奖励。不能用未折扣的 shaping_total 直接证明 Q 超界。

**终止势函数。** 当前成功、失败和 timeout 都把实测末端距离继续传给 shaping，而 timeout 默认不 bootstrap。有限回合下，“势函数塑形保持最优策略不变”需要相应终止约定，不能只引用该公式就认定成立。若 timeout 是任务真正结束，应统一吸收态势函数为 0；若只是采样截断，则需真实 next obs/base/latent 和一致 bootstrap，不能保留全零 next_base 再打开 bootstrap。

**距离代理。** `Recorder.distance()` 用的是 finger body 原点到按钮原点的距离，未计算接触表面距离。第三轮成功样本的该距离约 8.5 cm，说明它不宜直接解释为“离按钮还差多少厘米”。建议用已知夹爪接触点标定、按钮平面法向距离和横向误差设计奖励/critic 特征，仅在训练端使用。

**折扣的时间单位。** 当前 gamma 按 30 Hz 控制步使用，chunk discount 为 gamma^实际步数。gamma=0.995 时，完整 7 步为约 0.9655；40 秒的 1200 步为约 0.00244。因此不能直接把论文每环境步的 gamma 与这里同值比较。先明确想优化成功速度还是给慢速成功足够信用，再固定折扣协议；不把它与 latent 消融混在一起调整。

**接触反馈频率。** 7 步对应约 233 ms，新的视觉条件约每 4.3 Hz 更新；30 Hz 是目标执行频率。需要更快接触反馈时，可保持 VLA chunk 频率、提高残差反馈频率，或缩短执行 chunk，但这属于后续独立实验，涉及 replay、折扣、RPC 成本与时序调整。

以上是代码风险/待验证假设，不是已经证明的首轮 RL 退化原因。

## 第二阶段：增加训练辅助，而不是让 actor 读取答案

- **非对称 critic：** actor 仍只读取可部署的 latent、基座动作与本体状态；critic 单独增加按钮相对位置、平面法向、接触点误差、行程/速度等仿真信息。实现独立 `critic_obs/next_critic_obs`，不要把特权值直接追加进共用 obs。对于未来随机化的物理参数，可在证明必要后加入 critic。
- **示范残差正则：** 在 train 的同一帧运行冻结基座，构造 `(expert_action - base_action) / scale`；必须使用相同单位、未来 7 步对齐和 padding mask。先看超出 [-1,1] 的比例；大量不可达目标不能粗暴裁剪后声称是有效监督。可用 Q-filter 或衰减权重，减少最终策略被脚本专家限制。
- **降维动作：** 可比较 7 维右臂残差跨 chunk 共享、夹爪残差恒零，或低秩 chunk 残差。它们改变探索难度和表达能力，应单独消融。不要直接将 PiPER 的 XYZ 残差代码套在 S1 的关节动作上；笛卡尔残差需要 FK/IK、关节限制和奇异性处理。
- **TD3：** 先保留 SAC 来隔离 latent 的效果；后续增加 TD3 对照以更接近论文。无需把更换算法作为第一步。

## 真机阶段的额外条件

当前仓库可证实的 400 条示范来自仿真；尚未从本次检查得到可用于本方案的真实配对轨迹证据，因此完整 sim-to-real 的前提未满足。

需要真实 S1 相机标定、时间同步、关节/控制契约、可测反馈和真实按钮接触特性；先评估基座在真机能否接近目标。若基座系统性找不到按钮，小幅残差未必能补救，应先补真实示范/SFT，再冻结新基座、重新提取全部 latent。

Mapper 若按论文路径实现，必须位于完整 tokens 与动作头之间：仿真 tokens→mapper→冻结 action head，同时 mean-pool(mapped tokens)→actor；真机 raw tokens 同时供 action head 和 actor。只对 RPC 输出的 pooled latent 做 mapper 是另一种变体，并未修正仿真基座动作分布。对齐数据按任务阶段/轨迹进度构建，留出真实验证轨迹，避免仅分布距离下降但动作语义被破坏。

不确定真机是否有同等腕部力反馈时，第一版无力 actor 更符合当前项目部署约束。仿真对象定向接触力不能直接当作真实传感器读数。后续若接入真实可用的腕部力或估计器，应在仿真模拟其噪声、坐标系和延迟。

视觉对齐之外仍要标定/随机化按钮刚度、摩擦、控制增益、时延与相机外参，并在实机处理推理异步化。现有“推理期间物理暂停”的评估不能代表真机实时闭环。

## 建议实验顺序与验收

| 阶段 | 对照与变量 | 通过条件 |
|---|---|---|
| 0：接口与契约 | 固定图像/状态/seed，添加 latent 返回；奖励与终止边界测试 | 基座动作不变，零残差等价，特征同帧，部署 actor 无特权输入 |
| 1：latent 主效应 | 同协议的 state+base SAC vs latent+state+base SAC | 固定验证场景配对、多训练种子；报告改善/退步与区间，不能只看训练成功率 |
| 2：训练辅助 | 在阶段 1 上逐项加入特权 critic、示范正则、动作降维 | 用独立消融区分收益来源；必要时做组合版本 |
| 3：算法/时序 | TD3、反馈频率等各自与固定基线比较 | 相同交互预算、实际执行步数与计时协议；报告 wall time |
| 4：真实迁移 | 真机基座、无 mapper 残差、有 mapper 残差 | 独立真实场景，报告成功、异常接触、力与耗时；不能用仿真结果代替 |

建议先做 5k–10k 的工程/学习稳定性试验，再对主要两组使用相同 50k 或 100k chunk 预算、至少 3 个训练种子。小预算不用于最终否定/确认方法。实验改过奖励或 critic 时必须同步重跑无 latent 对照，不能只与历史 23/40 比较。

val-40 用于开发选择；test-40 仅在方案固定后做最终比较，不用于调参。历史 test 已报告，严格的新泛化结论宜再冻结一批未参与开发的场景。额外报告成功耗时、原本成功场景退化数、异常接触、动作平滑度，以及接近/接触失败分组。基础设施失败单列，不混入物理失败。验证采用相同 GPU 物理与相机协议。

按既有约 2–3 chunk transitions/s 粗估，100k 单次采样约 9–14 小时，未计初始化、评估、故障、特征传输和新增优化开销；这是规划量级，不是新方案实测。先测 latent 版真实吞吐再排实验，不直接沿用 PiPER 并行吞吐。

优先级：**特征接口与训练契约 → 配对 latent 消融 → 特权 critic/示范正则 → 真机数据与 mapper**。已有结构足以开始第一阶段，尚不足以承诺真机迁移成功率。
