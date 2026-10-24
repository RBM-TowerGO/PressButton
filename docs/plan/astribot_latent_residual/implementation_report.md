# Astribot latent 初始接入版本验收（历史记录）

日期：2026-10-22（北京时间）。**本文仅记录修复前初始接入版本，原实测数值和失败证据保留。** 当时范围为工程接入、CPU契约和GPU短冒烟。其后critic已修复，新协议GPU冒烟通过并进入10k；当前仅授权单种子A/B各一次100k＋val-40，见 [本轮记录](training_campaign.md) 和 [当前验收要求](validation.md)。

**结论：工程数据链路验收通过；学习稳定性未通过，发现 latent critic 饱和。** 该历史版本没有运行10k稳定性试验、正式训练或完整val-40，也没有算法收益结论。

## 实现

- 新增项目外部推理适配层，复用历史包装的初始化、权重校验、图像处理、seed 与动作后处理；`models/` 运行时快照没有修改。每请求仅调用一次模型，保留动作头 tokens，并返回 float32 均值池化 latent。
- 服务健康信息包含 latent 能力、语义版本及适配源码 hash。runner 默认 `state_only`，可用 `--observation-mode state_latent` 切换到 2112 维 observation；actor 拼接基座后为 2168 维。LayerNorm 无可学习参数，epsilon=1e-5，使用总体方差。
- journal、replay、learner 与恢复按模式检查维数；协议、checkpoint 和 replay 均记录语义身份。latent 缺失或非有限立即报基础设施错误；相同维数但不同特征版本也拒绝恢复。旧 state-only 数据保留对应模式的兼容路径。
- 普通后继使用下一真实请求；latent 末帧额外请求仅采集后继，不执行动作。普通请求已因动作安全校验被拒绝时，复用该请求作为 pending 的真实后继，不再次采样。未完成 transition 不提交；特征采集失败分别记录基础设施状态与实际物理结果。
- 修正初始 settling tick 后图像同步：首次请求前重新渲染，确保图像与 q/qd 对齐。记录请求序号、seed、物理时间、state 与图像 hash，诊断信息不进入 actor。
- 增加 Q/target 分位数、Q 边界比例、残差饱和率、折扣 return、末帧势函数、分阶段耗时和资源记录。保留原奖励、critic 输出、动作合成及 `bootstrap_truncated=False`。
- 确定性评估检查参数与 updates 不变，且不重写训练 checkpoint/replay/status/protocol。恢复检查源码与训练配置，因此评估命令需保持训练契约参数；验收脚本保存了完整命令。

## 契约验收

CPU：`101 passed`，完整输出见 [cpu_tests.log](../../../outputs/astribot_online_rl/latent_engineering_20261022/cpu_tests.log)。

| ID | 结果与实际覆盖 |
|---|---|
| C01 | 真实头部/腕部图像、state、seed：旧包装与新适配的 action8/action32/normalized_actions 逐元素完全一致 |
| C02 | 实测 tokens `[1,32,2048]` float32；与 float32 均值及 float64 独立求和参考比对通过 |
| C03 | 同环境、同 seed 重复真实推理，动作及 latent 完全一致；三个请求对应三次模型调用 |
| C04 | CPU：普通/常量向量、总体方差、epsilon，以及 dtype/形状/NaN 拒绝 |
| C05 | CPU：64/2112 维布局、2168 维 actor 输入、journal 模式检查；两模式真实 GPU rollout |
| C06 | CPU：零残差保持完整展开目标，现有限位和展开检查回归通过 |
| C07 | CPU：连续 chunk 的 q/qd、latent、基座请求及 seed 对齐；GPU：真实同步请求与 replay 写入 |
| C08 | CPU：成功、物理失败、timeout、reset、动作拒绝；GPU 实测 timeout 末帧请求不执行额外动作 |
| C09 | CPU：缺失/错误 dtype/形状/NaN，首帧、中途和末帧失败；不提交不完整条目、不把中途错误计为物理 timeout |
| C10 | CPU：checkpoint 优化器/RNG 续训一致性、latent replay 往返及语义版本拒绝；真实产物 replay 恢复与 learner 加载评估 |
| C11 | 观测构造和输入审计：只有 q、qd、可选 latent 与基座动作；场景编号和任务真值只在诊断/环境中 |
| C12 | 真实七相机确定性评估；参数和 updates 不变，训练产物 SHA256 前后一致，评估未加载 replay |

真实接口证据：[report.json](../../../outputs/astribot_online_rl/latent_engineering_20261022/real_interface/report.json)、[evidence.npz](../../../outputs/astribot_online_rl/latent_engineering_20261022/real_interface/evidence.npz)。GPU 未实测的成功/失败末帧分支仅声明 CPU 时序覆盖。

## 最终 GPU 冒烟

本机单 RTX 5090，推理与 GPU 物理共享 GPU 0，SAC 在 CPU。冻结模型 SHA256：`db98b7c5488c666c3e10fa721dc29b25e069c41237bf8cb42b638611863eaaa8`，加载 1617 个 tensors，真实接口检查全部参数冻结。

使用 seed=20261024；每模式训练一个 2 秒回合、小 batch=4、warmup=2、replay capacity=8；保存恢复后，各评估一个 2 秒 val 场景。训练双相机，评估原生七相机。actor LR=1e-5、critic LR=3e-4、固定 alpha=.005、均值正则=1、shaping=1、gamma=.995，右臂尺度 .01 rad、夹爪 .025 rad，沿用原网络。

| 模式 | 实际提交 / replay 留存 | updates | 控制步 | 训练请求数 | 末帧额外推理 | 采样整卡显存峰值 |
|---|---|---|---|---|---|---|
| `state_only` | 9 / 8 | 8 | 60 | 9 | 0.0000 s | 21,903 MiB |
| `state_latent` | 9 / 8 | 8 | 60 | 10 | 0.0765 s | 22,064 MiB |

两模式均完成 actor 和 critic 更新调用、真实 replay 保存恢复，以及确定性 learner 加载评估。每模式 val 仅完成 1 个场景，均为 2 秒 timeout；这只验证工程链路，不是完整 val-40 或成功率结论。每个回合 60 个控制步、241 个物理 tick；latent 比 state-only 多 1 次只采集后继的请求，没有增加控制步。

当时版本的runner在回合末检查预算，故本次实际提交 9 条、容量 8 的 replay 留存 8 条，未将留存条数冒充实际交互数。这次小容量烟测不用于证明正式预算控制或长期内存性能。100k 的新增 latent 理论存储仍为约 1.53 GiB；本次未实际分配或序列化 100k replay。

两模式资源、保存/恢复耗时和实际文件大小见 [engineering_summary.json](../../../outputs/astribot_online_rl/latent_engineering_20261022/engineering_summary.json)；原始资源采样、所有命令、协议、逐回合记录及 checkpoint 见 [final/](../../../outputs/astribot_online_rl/latent_engineering_20261022/final/)。采样显存为整卡占用，包含原有其他进程；不是本任务独占显存或精确瞬时峰值。

## 学习风险、偏差与未执行项

**latent critic 第 2 次更新起 Q 边界比例为 1，第 4–8 次更新 Q 预测为 0、critic 梯度为 0。** 同次 state-only 冒烟未出现零 critic 梯度。latent actor 更新有有限梯度，残差饱和率为 0；target 没有超出 [0,1]。证据见 `final/state_latent/metrics.jsonl`，不能据此声称 latent 已稳定学习。

按原方案只记录此问题，没有改 Q 输出、奖励、bootstrap、网络结构或学习率，也没有继续长训练。后续 10k 或正式对照需要先独立诊断并决定是否建立新协议。短冒烟无法区分小 batch/重复更新效应与长期饱和风险，也不证明 latent 方法本身无效。

首次预验证曾因回合目录提前创建而失败，日志及 `state_only/infrastructure_error.json` 原样保留；修复后在独立 `verified/` 路径通过首轮验证。随后补充 replay 语义元数据及中途错误物理状态区分，最终源码在独立 `final/` 路径重跑通过。`source_manifest.json` 和最终协议记录当前代码身份，旧产物没有覆盖。

该历史版本未执行：A/B 10k、正式训练、完整val-40、test-40、真实机器人及未来优化。其学习稳定性未通过。后续修复版本的状态以本轮记录为准，不能把历史失败覆盖或当作当前配置结果。

## 原始验收入口与复现边界

以下是原始入口形式。当前脚本已使用修复配置和精确预算，重新运行不会复现本文旧版的9条提交/8条留存或旧初始化；复现旧结果须使用对应归档源码/协议。新验收使用当前脚本与新目录，当前训练期间不要另启同卡实验。

```bash
.venv-astribot-inference/bin/python scripts/astribot_policy_server.py \
  --run models/astribot_head_right_joint8_ft_20261018 --device cuda:0 --port 8797

.conda/envs/pressb/bin/python scripts/validate_astribot_latent_engineering.py \
  --output outputs/astribot_online_rl/latent_engineering_NEW --endpoint http://127.0.0.1:8797
```

真实接口独立验证使用 `scripts/verify_astribot_latent_policy.py`，输入为冒烟保存的 `first_request.npz`。测试、验证脚本及新增适配源码的 hash 见 [source_manifest.json](../../../outputs/astribot_online_rl/latent_engineering_20261022/source_manifest.json)。
