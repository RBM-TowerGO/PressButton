# 修复后分阶段训练（2026-10-22）

状态：**修复后 A/B GPU 冒烟已通过，已进入 A组10k稳定性试验；后续阶段仍待门槛通过。**

2026-10-22 16:18（北京时间）：两组各8次更新，均无零critic梯度，末次Q均值约0.473/0.477；恢复和确定性评估通过。全流程尚未完成，不作算法收益结论。

本页取代 taskmemory 中早期“只授权工程接入、不运行10k/正式训练”的范围限制；历史验收结果仍保留，不将旧版本稳定性失败改写为通过。

## 本轮协议

协议：`astribot_latent_fan_in_q3e-5_exact_budget_v2`。A=`state_only`、B=`state_latent` 均显式使用 `--critic-init fan_in --q-lr 3e-5`；保持 actor LR=1e-5、alpha=.005（固定）、均值正则=1、shaping=1、gamma=.995，右臂残差 .01 rad/夹爪 .025 rad，基座冻结 step 8000。奖励、Q 输出和 bootstrap 语义不变。

单 GPU 0 串行运行推理和仿真，SAC 在 CPU。新网络/target/优化器/replay 全部从头初始化，正式运行不会读取10k权重。没有覆盖历史实验。

- 阶段1：两组各一回合 2秒短冒烟，batch=4、warmup=2、容量8，保存/恢复 replay、恢复 learner，使用原生七相机确定性评估。必须无零 critic 梯度、无学习/基础设施失败，并验证评估不改变训练产物。
- 阶段2：两组各精确10000 transitions，seed=20261024，warmup=5000、batch=256、回合40秒。训练后用真实 runner 恢复网络、优化器及 replay，确认预算和指标一致；不要求成功率达到任意门槛。
- 阶段3：阶段2两组通过后，各从头运行种子20261024/20261025/20261026，每组每种子精确100000 transitions，其他设置同阶段2。每次最终 checkpoint 完成原生七相机确定性 val-40；test-40不使用。汇总逐种子配对改善/退步、精确 McNemar p、成功耗时、异常接触及跨种子差值均值/范围。

## 本轮启动前修正

- 修正冒烟入口的 A/B critic 配置差异；不再依赖两个模式不同的默认值。
- runner 在 budget 耗尽后用下一真实请求补齐最后一条后继，不额外执行 chunk，也不把预算边界计为物理 timeout。最后一条非终止 transition 保留原 bootstrap 规则；实际提交数与 replay 留存数必须精确等于预算。
- 连续3次 minibatch 出现非零误差下整体 critic 零梯度，或针对内部目标整 critic 精确边界饱和时停止，并写入 `learning_error.json`。学习异常和基础设施错误分别保存。该停止规则不修改任何梯度或优化目标。
- CPU相关测试：113 passed，包含预算不超采样、真实后继对齐、持续饱和拦截和配对统计。

## 运行与结果路径

输出根目录：[latent_fixed_campaign_20261022](../../../outputs/astribot_online_rl/latent_fixed_campaign_20261022/)。

- 当前阶段、PID、最近进度：[run/progress.json](../../../outputs/astribot_online_rl/latent_fixed_campaign_20261022/run/progress.json)
- 冻结协议及源码 hash：[run/protocol.json](../../../outputs/astribot_online_rl/latent_fixed_campaign_20261022/run/protocol.json)，原始源码副本在 `run/source/`。
- 测试：[cpu_tests.log](../../../outputs/astribot_online_rl/latent_fixed_campaign_20261022/cpu_tests.log)
- 监督进程日志：[supervisor.log](../../../outputs/astribot_online_rl/latent_fixed_campaign_20261022/supervisor.log)
- `run/smoke/`：短冒烟结果；`run/trials/`：10k；`run/formal/`：正式训练和val。
- 所有子进程命令、阶段耗时和定期资源采样保存在 `run/`；每个训练目录记录训练 gate、逐回合、指标、checkpoint、replay和恢复耗时。

各阶段产物仅在实际执行后存在。`trial_gate.json` 仅在两组10k及恢复通过后生成；`paired_results.json` 仅在六个正式训练及全部val-40完整通过后生成。`progress.json.state=complete` 才代表整轮完成；`stopped_with_error` 表示后续阶段未启动，应查看 `campaign_error.json`。没有自动改配置、跳过失败门槛或静默重启。

启动命令已保存于 `launch.json`，监督脚本为 `scripts/run_astribot_latent_campaign.py`。本轮监督PID见进度文件，不依赖聊天保持连接。
