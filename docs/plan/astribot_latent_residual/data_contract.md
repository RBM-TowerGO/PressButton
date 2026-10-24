# latent 残差数据与接口契约

状态：**特征链路已接入并验证；修复后GPU短冒烟通过，10k检查运行中。** 创建/更新日期：2026-10-22。特征契约仍为版本1；本轮训练使用单种子工程协议，详见 [本轮记录](training_campaign.md)。历史critic饱和及修复见 [诊断记录](critic_diagnosis.md)。

## 1. 特征身份与模式

定义两种观测模式：`state_only` 和 `state_latent`。前者为现有默认兼容模式，后者为第一阶段实验模式。两者使用同一冻结 step 8000 基座与动作契约。

特征固定来自动作头使用的 `embodied_action_tokens`，不是 flow 噪声、JEPA 预测或纯视觉编码器输出。预期 tokens 为 `[B,32,2048]`；在线单样本请求 B=1，真实形状必须在 GPU 接口验收中确认，不符合则停止并修订契约，不静默 reshape。

特征契约初始版本记为 `1`。语义名称为 `embodied_action_tokens_mean`。类型、池化或归一化发生变化时必须提升版本并建立新实验，不能仅保持同维数继续恢复。

## 2. RPC 契约

请求继续使用现有头部图像、右腕图像、state 和 seed，不改变图像顺序、预处理、任务指令或采样随机数语义。在线推理扩展返回：

| 返回字段 | 类型/形状 | 语义 |
|---|---|---|
| `action8` | 保持现有 `(7,8)` | 反归一化后的右臂/夹爪绝对关节目标 |
| `action32` | 保持现有 `(7,32)` | 按固定关节映射展开的目标 |
| `normalized_actions` | 保持现有字段 | 含义和数值不变 |
| `vlm_latent` | float32 `(2048,)` | float32 tokens 沿 token 维均值，尚未做 LayerNorm |

保留现有时延字段。在线不传输完整 32×2048 tokens。健康信息至少声明是否支持 latent、特征类型、维数和契约版本，启动时与基座模型身份一起核验。

`state_latent` 模式中，能力不匹配、字段缺失、维数/类型错误或非有限值均为基础设施错误：记录请求上下文，停止当前流程，不提交无效 transition，不以零向量或上一帧特征替代。`state_only` 可忽略新增字段，并继续兼容没有该能力的旧服务。

## 3. 特征变换、观测与动作

设单样本 token 矩阵为 Z，先转 float32：

```text
z[j]      = mean_i Z[i,j]                  # token 维求均值
mu        = mean_j z[j]
variance  = mean_j (z[j] - mu)^2           # 总体方差，非无偏估计
latent[j] = (z[j] - mu) / sqrt(variance + 1e-5)
```

无可学习 scale/bias；服务返回 z，runner 仅处理一次得到 latent。该规则逐样本执行，不使用数据集统计量，不影响动作头收到的原始 tokens。

| 模式 | observation | actor 输入 | 动作输出 |
|---|---|---|---|
| `state_only` | q32＋qd32，共 64 维 | 64＋56＝120 维 | 56 维归一化残差 |
| `state_latent` | q32＋qd32＋latent2048，共 2112 维 | 2112＋56＝2168 维 | 56 维归一化残差 |

```text
obs[0:32]     = 实测 q32
obs[32:64]    = 实测 qd32
obs[64:2112]  = latent                    # 仅 state_latent
actor_input  = concat(obs, flatten(base_action_7x8))
executed     = base_action + scale * normalized_residual
```

q 的单位为 rad，qd 为 rad/s，均沿用既有关节顺序和处理。基座 chunk 按时间优先展开，每个时刻保留原 8 维顺序。残差按现有 tanh 输出及尺度合成，展开为 action32 后仍执行原限制检查。

critic 读取相同模式的 observation 和合成动作，不增加按钮真值。actor 不读取场景编号、目标坐标、行程、评分或物体定向接触报告。请求诊断元数据不进入学习观测。

## 4. 时序与终止契约

### 4.1 普通 transition

1. 在仿真时刻 t 冻结物理，取得同步图像与 q/qd。
2. 同一次 VLA 请求产生基座动作 A_t 和 z_t，组装 obs_t。
3. 若上一 chunk 处于 pending，用本次 obs_t 和 A_t 作为其后继并提交。
4. actor 读取 obs_t、A_t，执行修正后的 chunk，记录实际执行步数。
5. 未终止则暂存本条，等下一次实际请求提供后继。

禁止使用旧 latent、未来图像或额外重采样的基座动作替换当前/后继内容。观测在 reset 前归属于原 episode；pending 不能跨 episode。图像、state、seed、episode/request 序号和仿真时刻应能在诊断记录中关联，但不作为 actor 输入。

### 4.2 末帧

成功、物理失败和 timeout 的最后一帧若图像与状态有效，在物理冻结下执行一次只用于获取后继特征的请求。使用末帧 q/qd、该请求 latent 和基座动作形成真实后继；不调用残差执行，不向机器人发送末帧请求返回的动作，不增加控制步数。

末帧请求使用该 episode 下一请求序号对应的既有 seed 规则，标记用途为后继特征采集；不能改变下一回合的 seed 规则。额外推理耗时单独记录。

保持既有 terminated/truncated 标志及 `bootstrap_truncated=False`。记录真实后继是数据完整性要求，不表示本阶段改变了 Bellman bootstrap。无 latent 对照沿用原末帧处理；报告两种模式的实际耗时差异。

若状态无效、图像捕获或末帧请求失败，保存已观察到的物理结果及基础设施错误，当前不完整 transition 不入 replay，已合法提交的数据保留。不能把特征采集失败伪装成物理 timeout，也不能重复提交末条。

### 4.3 训练预算边界

每执行一个chunk形成一条transition，最多7个控制步；回合40秒最多1200个控制步，约172条。10k/100k是跨回合累计transition数，不是回合数、控制步数或更新次数。

训练在最后一条pending通过下一真实请求补齐并提交后检查预算；达到容量就停止执行新chunk。最后一条若并未物理终止，保留正常后继及原terminated/truncated标志；回合报告标为`budget_exhausted`，不伪装成物理timeout。A/B采用相同的精确预算规则，实际提交数与replay留存数必须一致。

## 5. Replay 与资源

沿用当前/下一 observation、归一化残差 action、reward、实际步数 discount、terminated/truncated、当前/下一 base action 字段。缓存经固定 LayerNorm 处理的 latent，包含在 obs/next_obs 中；不存完整 token 序列，不存随训练变化的投影。

100k transitions 的新增特征存储为：`100000 × 2048 × 4 × 2 = 1,638,400,000 bytes`，约 1.64 GB / 1.53 GiB，不含旧字段、网络、采样副本及序列化峰值。执行前检查内存余量、磁盘空间、保存/恢复耗时；不根据压缩后的文件大小估计运行内存。

`ChunkJournal`、runner 初始化、replay 采样和恢复都按同一模式选择 obs_dim。不得只扩大网络输入而遗漏 journal 的 64 维检查。

## 6. Checkpoint、恢复与兼容

协议/checkpoint 至少保存：观测模式和维数、特征类型和版本、token/池化维度、float32 与 LayerNorm 参数、基座 checkpoint hash、适配代码版本、动作尺度、划分身份、训练配置（包括critic初始化策略、critic LR）和原有恢复状态。replay元数据同时保存观测与训练身份，不能只凭相同维数恢复。

恢复验证语义和维数均一致。旧 state-only 权重只能在对应模式恢复；不得零填充/截断权重或 replay 来转换模式。新的 latent actor、critic、优化器和 replay 从头开始。正式实验与 10k 试验分别初始化、分别保存。

评估加载与训练相同的观测契约，使用确定性残差，不更新网络或 replay。仅 actor 评估时不需要加载训练 replay，但不得跳过模式/特征身份检查。

本轮A/B显式使用`critic_init=fan_in`、`q_lr=3e-5`，种子均为20261024；旧配置与新配置不是同一训练协议。原已饱和checkpoint只作为历史诊断证据。`state_only`旧checkpoint仍按对应legacy配置加载，不能冒充新协议。

只调整正式种子数量的监督编排不改变正在运行的learner/replay契约；本次交接保留了原A组采样进程和数据，旧监督协议另行归档。10k结束后的正式A/B仍须从头初始化。
