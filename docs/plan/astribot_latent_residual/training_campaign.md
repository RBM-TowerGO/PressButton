# 单种子工程训练与运行记录

状态：**用户已授权完成B训练及val-40，然后继续A训练及val-40。B正式100k持续运行，A等待B完整评估通过。** 更新：2026-10-22。

当前顺序为`B_100k → B_val40 → A_100k → A_val40`。B原worker1081413与策略服务1081370无损保留；新监督1089229已接管。交接时B为20591条/20591次更新。实时状态见`run/progress.json`，两组10k对比见 [比较报告](trial_10k_comparison.md)。

## 当前授权与配置

此前授权的预算按工程项目执行，正式仅使用种子 **20261024**：A/B各10k稳定性检查，通过后各从头训练一次100k，再各做val-40。共两次正式训练、200000条正式经验和80个最终评估回合；10k两组另外合计20000条。旧多种子方案已取消，不要求多种子复现或显著性检验作为验收门槛。用户先停止A、查看10k对比、单独启动B，随后明确授权B训练和val-40全部通过后再继续A训练和val-40。当前按该最新顺序推进，不再受先前“仅B、不自动评估”的范围限制。

协议ID：`astribot_latent_fan_in_q3e-5_exact_budget_v2`，当前`protocol.json.configuration.formal_seeds=[20261024]`。A=`state_only`、B=`state_latent`，均显式使用`--critic-init fan_in --q-lr 3e-5`，actor LR=1e-5，固定alpha=.005、均值正则=1、shaping=1、gamma=.995，右臂尺度.01 rad/夹爪.025 rad，冻结基座step8000。奖励、Q有界输出和bootstrap规则不变。

单GPU0串行运行推理/物理，SAC在CPU。训练使用头部＋右腕两相机，最终评估原生七相机。各组、各阶段重新初始化actor、critic、target、优化器和replay；正式不继承10k权重。当前运行详情以协议和源码hash为准，不依赖CLI的不同模式默认值。

## 阶段与已核验证据

| 阶段 | 执行方式 | 当前状态 |
|---|---|---|
| 修复后GPU短冒烟 | A/B各8条transition、8次更新，batch=4、warmup=2；保存恢复及一回合2秒七相机确定性评估 | 两组通过 |
| A组10k | seed=20261024、warmup=5000、batch=256、回合40秒；完成后恢复训练状态 | 10000条/10000次更新，训练及恢复通过 |
| B组10k | 与A相同配置，额外latent输入 | 10000条/10000次更新，训练及恢复通过 |
| B正式训练 | 从头100000条，seed=20261024 | 正在运行，已保留全部当前进度 |
| B最终评估 | 完整val-40，确定性、原生七相机 | B训练门槛通过后自动执行 |
| A正式训练 | 接续原正式checkpoint至总计100000条 | 等待B val-40通过；原checkpoint488条/0更新已验证可恢复 |
| A最终评估及工程比较 | A训练通过后val-40，再汇总A/B相同40场景 | 等待上述阶段；合计80个最终评估回合 |

短冒烟两组均无零critic梯度，末次Q均值约0.473/0.477；replay保存恢复、learner恢复、确定性评估通过，网络参数、updates及训练产物hash不变。CPU相关回归启动前113项通过；单种子范围调整另有3项编排测试通过。这些证据不替代10k稳定性检查。

runner按回合采集、随后从replay抽batch集中更新。warmup前只采集；达到5000条后补齐计划更新次数。每条transition对应最多7个控制步的chunk，不是一整个回合。精确预算停止会先获取最后一条真实后继，再停止执行新chunk，不把预算边界记为物理timeout，也不超采样后覆盖replay。

严重学习异常连续3batch触发时停止并写`learning_error.json`；基础设施错误单列。具体有限值、饱和和预算门槛见 [validation.md](validation.md)。任一阶段失败后，不改配置、不跳过门槛、不自动重启。两组10k通过后生成`trial_gate.json`；两个正式训练及全部val-40通过后生成`paired_results.json`。`progress.json.state=complete`才代表全轮完成。

## 产物与实时入口

输出根目录：[latent_fixed_campaign_20261022](../../../outputs/astribot_online_rl/latent_fixed_campaign_20261022/)。

- [当前进度与PID](../../../outputs/astribot_online_rl/latent_fixed_campaign_20261022/run/progress.json)
- [当前协议与源码hash](../../../outputs/astribot_online_rl/latent_fixed_campaign_20261022/run/protocol.json)
- [当前恢复监督日志](../../../outputs/astribot_online_rl/latent_fixed_campaign_20261022/recovery_supervisor.log)
- [当前恢复启动命令](../../../outputs/astribot_online_rl/latent_fixed_campaign_20261022/recovery_launch.json)
- [启动前CPU测试](../../../outputs/astribot_online_rl/latent_fixed_campaign_20261022/cpu_tests.log)
- [修复后短冒烟汇总](../../../outputs/astribot_online_rl/latent_fixed_campaign_20261022/run/smoke/smoke_results.json)

`run/trials/`存放10k，`run/formal/`在实际启动后才存在。各训练目录保存协议、逐回合报告、metrics、checkpoint、replay及恢复/保存耗时。`run/`保存子进程命令、阶段日志和定期资源采样。当前版本源码副本在`run/revisions/*/source/`，`run/source/`保留交接前版本。恢复训练前仍需核验源码/训练身份，不能将后续修改无声混入原协议。

18:27停止时，原监督1072000、正式A worker1080395及策略服务1067507均已退出。随后用户授权B独立作业，现用新的监督与策略服务，PID见实时进度。`user_stop.json`记录主动停止，不能当作学习失败或基础设施失败；中断回合不计完整物理结果。只有停止标记实际存在时才报告失败，不能将待运行项写成成功。

## 范围调整与耗时边界

用户取消多种子后，只修改后续编排，监督无损接管原A组worker和策略服务；交接时已有4862条经验保留，没有重启仿真、清空经验池或重采。旧监督已移除，旧协议/进度保存于`run/revisions/`，旧日志`supervisor.log`只供历史审计。

按本机已观察到的约3条transition/秒，单纯采集10k约56分钟，100k约9.3小时；还需加网络更新、保存恢复和评估。两次正式100k及val-40暂估20–30小时，依实际吞吐和异常调整，不是完成时间承诺。当前10k的实测更新速度与后续资源报告将用于修正估计。

## 最新停止与对比产物

2026-10-22 18:26用户要求停止A正式训练并先对比10k。先冻结监督以阻止进入后续阶段，再向worker发送SIGINT；确认worker退出后移除监督并关闭策略服务。保留最后可恢复488条/0更新checkpoint，活动中断回合不计完整结果。18:27更新状态为stopped_by_user，所有相关进程已核实退出。

两组10k：A完整回合51/84＝60.7%，B为51/83＝61.4%；共同83个完整训练场景A成功50、B成功51（改善6、退步5）。两组均无零critic梯度/精确边界饱和，B训练跨度约64.39分钟、A约58.96分钟。含探索且策略在训练中变化，不能作为最终策略验证结果。详细统计和口径见trial_10k_comparison.md。

## B组正式100k单独启动（用户后续指令）

用户明确“先启动B组正式100k”。已核验B的10k训练/恢复门槛通过、生产训练源码与已验证版本一致、B正式输出目录不存在。

- 从头初始化actor、critic、target、优化器和100k replay；不传`--resume`，不读取10k权重或A组权重。
- 参数保持seed20261024、state_latent、fan_in、critic LR=3e-5、actor LR=1e-5、warmup5000、batch256、40秒回合及原奖励/Q规则。
- B正式输出：`run/formal/state_latent_seed20261024/`；独立监督产物：`b_formal_job_20261022/`。
- 启动记录：`b_formal_launch.json`；监督日志：`b_formal_supervisor.log`；独立监督脚本：`scripts/run_astribot_latent_b_job.py`。
- 当前监督PID1081368，策略服务PID1081370，worker启动后PID写入进度。旧停止记录仍保留，B启动前的原流程状态归档在独立作业的`prior_campaign_progress.json`。
- 新作业将当前状态同步到原`run/progress.json`，并单独保存协议、源码、资源、命令、日志和训练门槛。完成仅表示B的100k训练完成，不代表A/B比较或val-40已完成。
- 不自动恢复A，不自动评估；发现学习/基础设施异常停止并保留证据。

## 最新授权：B训练及val-40后继续A及val-40

用户要求“把B训练和val40都做完，完成后再训练A和val40”。2026-10-22 20:47已将监督无损交接至1089229，B worker1081413和policy1081370未停止或重置；交接时20591条/20591更新。原B-only监督已移除。

当前脚本`scripts/run_astribot_b_then_a_job.py`，监督产物`b_then_a_job_20261022/`，日志`b_then_a_supervisor.log`，启动记录`b_then_a_launch.json`；当前状态继续同步`run/progress.json`。原B-only进度和本次授权保存在新作业的`prior_B_progress.json`及`authorization.json`，当前源码/协议位于新作业`source/`和`protocol.json`。

B达到精确100k并通过训练门槛后，使用最终checkpoint确定性评估完整val-40。必须40个冻结场景ID全部完成、参数和updates不变、训练文件hash不变，才启动A。

A接续此前已经从头初始化的正式checkpoint（488条、0更新），不是继承A/B10k权重。已实际验证learner、replay和训练契约可以恢复，至本轮正式总预算100000条；原中断回合不计完整结果。A完成后同样val-40，最后在`run/paired_results.json`汇总单种子A/B结果。

新增10项编排/评估门槛测试通过，包含B→B评估→A→A评估顺序、B评估失败阻断A、验证场景完整性及训练文件不变。任何阶段出现错误均停止后续流程并保留证据，不跳过评估、不自动更改训练配置。当前“完整完成”指两组训练和两个val-40都完成。

## 21:29基础设施中断与已审核恢复（2026-10-22）

B正式训练在21:29发生`Connection reset by peer`，推理服务没有留下Python错误堆栈；其确切退出原因未确认。故障前资源采样可用内存不低于37.7 GiB，未获取到内核故障记录，不能宣称已证明OOM或算法错误。原监督按门槛停止，因此后续val-40和A未启动。

已实际加载校验checkpoint、优化器和replay：27133条经验、27050次更新，源码/训练语义一致；253个完整回合之外，中断回合保存了83条合法transition，未完成的pending没有提交。原始基础设施报告保留、不计成物理失败，恢复后从新回合开始，不跨episode续接pending。

用户此前已授权完整B→B val-40→A→A val-40，因此在核验后按原配置进行一次审核恢复，不重置网络或经验池。checkpoint/replay/status/protocol及原错误原样备份到B目录recovery/，带文件hash；活动错误标记的副本与旧监督失败日志均保留。只接受hash已核验的这一份中断报告及83条已保存数据；再发生错误仍停止，不循环盲目重试，不把新基础设施异常忽略。

当前恢复监督产物：`b_then_a_recovery_20261022/`；日志`recovery_supervisor.log`；启动记录`recovery_launch.json`；复核说明`recovery_review.json`位于恢复监督目录。脚本`scripts/resume_astribot_b_then_a_job.py`，恢复监督PID1094894，策略服务PID1094895，worker启动后以实时进度为准。`run/progress.json`继续记录当前恢复状态，旧失败进度归档为`prior_failed_progress.json`。

新增中断数据核算测试及原B→验证→A→验证门槛测试共8项通过。训练代码、奖励、Q规则及学习率未改，仅监督恢复和证据核算新增；最终报告必须保留这次中断和恢复记录。

恢复确认：2026-10-22 23:19，B已继续至27225条、27225次更新，恢复后首个回合和补齐更新成功；当前worker1094939，监督1094894、policy1094895运行正常，无新的活动错误标记。原有失败记录仍在归档中。

## A中断后的用户授权恢复（2026-10-23）

B的100k训练和val-40已完成，验证24/40。A于16:17遇到推理连接重置，保存84851条经验、84794次更新；新增中断回合57条合法经验已计入replay，不计完整物理回合。底层连接中断原因尚未确认。

用户明确要求“继续跑”。恢复前实际加载learner/优化器及replay，核对训练源码hash、checkpoint/status契约、逐回合经验总数和B已完成门槛，全部通过。原checkpoint/replay/status/protocol/error备份于A目录recovery/并记录hash；仅归档移除已核验的活动错误标记。未修改训练参数或训练源码，未重跑B。

恢复入口`scripts/resume_astribot_a_job.py`；监督产物`outputs/astribot_online_rl/latent_fixed_campaign_20261022/a_recovery_20261023/`；启动记录`a_recovery_launch.json`，日志`a_recovery_supervisor.log`。监督1158814、policy1158862、worker1158909。16:57已确认A增长至85022条/85022次更新，833个完整回合，无新活动错误。相关恢复核算及B→A编排测试8项通过。

后续自动完成A至100000条、通过训练门槛、执行原生七相机确定性val-40、核验参数/updates/训练文件不变，最后生成run/paired_results.json。再遇新错误停止并保留证据，不循环自动重试。预计恢复后约2～2.5小时完成，实时以run/progress.json为准。
