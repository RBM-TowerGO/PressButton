# Astribot latent 工程任务记忆

## 最新交接：偏移诊断已完成（2026-10-24）

- 用户授权的受控采集、闭环验证和离线预测实验已完成，不需要重复采集或重新训练 A/B。入口及参数见 [offset_probe.md](offset_probe.md)，结论见[实验报告](../../results/astribot_latent_offset_probe_20261023.md)。
- 受控 100 场景／7,500 条，按场景 60/20/20 划分；正式闭环 40 场景／5,275 条，40 秒上限。31 个场景进入分析区域（77.5%），含 954 条接触前、108 条接触后观测。25 秒试采与合成工程样本已隔离，不参与正式统计。
- 已确认受控条件下 latent 对水平偏移有增量预测价值，latent-only 可迁移到仿真闭环接触前；融合探针遭遇夹爪位置和速度的分布偏移，闭环条件增量尚未可靠确认。离线探针标准化失稳不能直接解释原 B 策略的近零残差。
- 36 项相关测试通过，数据质量与预测复算通过。同一冻结观测的 61 次配对请求动作／latent 逐元素一致；独立重置存在非确定性，不能写作完整轨迹等价验收通过。
- 产物位于 `outputs/astribot_online_rl/latent_offset_probe_20261023/`，含数据、指标、图、模型、质量检查和源码快照；该目录未纳入 Git。自有采集进程和模型服务已关闭。
- 后续优先补足动态状态覆盖并使用新的留出闭环数据验证，同时继续诊断确定性残差为何近零。预测信息不等于有效纠错；新增训练不自动启动。

以下为此前 A/B 训练交接历史，其中 PID 和“当前／最新”均属于当时记录。

更新：2026-10-22（北京时间）。

**最新指令优先：完成当前B的100k及val-40，通过后继续A正式100k及val-40。顺序B→B验证→A→A验证，自动串行推进；此前仅B/禁止自动评估和主动停止的记录均为历史。** 两组10k及恢复均通过；正式A保留488条/0更新checkpoint，B正式未启动；对比见trial_10k_comparison.md及run/trial_10k_comparison.json。以下早期运行快照仅作历史。当前授权和待完成工作以本文件、plan.md、validation.md及training_campaign.md为准；历史报告不覆盖最新指令。

## 当前授权与范围

- 用户已授权：修复后GPU冒烟 → A/B各10000条稳定性检查 → 两组通过后各从头100000条正式训练及各自val-40。
- 用户明确这是工程项目，正式只保留种子20261024；共两次正式训练、200000条正式经验、80个最终评估回合。多种子方案取消，不自动扩大预算。
- A=state_only，B=state_latent；两组显式fan_in、critic LR=3e-5，actor LR=1e-5，alpha固定.005，均值正则1、shaping1、gamma.995，右臂尺度.01/夹爪.025，warmup5000、batch256、回合40秒。
- 基座step8000冻结；不改models历史运行时、奖励、Q有界输出、bootstrap或SAC结构。未来方向不自动启动，不使用test-40。
- 默认state_only是旧接口兼容路径，不能依赖它的legacy默认值运行本轮A组。禁止跨模式、特征版本或训练身份恢复。

## 已完成与正在进行

- 特征链路、RPC、观测、journal/replay/恢复、末帧后继及确定性评估已实现；tokens实测[1,32,2048] float32，C01–C03真实模型通过。
- 旧版接入曾critic饱和；随后修正fan_in初始化及critic LR。真实replay离线五种子各100次更新无零梯度，该离线验证不算正式训练预算。
- 修复后GPU A/B各精确8条、8次更新，无零critic梯度，末次Q约0.473/0.477；保存恢复及七相机确定性评估通过，参数/updates/训练文件hash不变。
- 启动前113项CPU回归通过，缩减范围的编排测试3项通过，不把两次测试数量直接相加。
- 进度快照：`2026-10-22T16:53:38+08:00`（北京时间），阶段`trial_state_only_train`，A组已收集6055条、更新6055次、完成47个回合。此快照不会自动刷新，实时状态以progress.json为准。
- B组10k、两组正式100k及完整val-40尚未执行。读取实时产物再更新状态，不把已启动记为完成。

## 运行入口与交接

- 输出：outputs/astribot_online_rl/latent_fixed_campaign_20261022/；实时状态run/progress.json，协议run/protocol.json，当前formal_seeds=[20261024]。
- 监督脚本scripts/run_astribot_latent_campaign.py；当前监督1072000、A组worker1068953、policy1067507。监督独立后台运行，PID以进度文件为准。
- 单种子范围修改时无损接管，原worker/policy继续运行，保留当时4862条经验。旧监督已移除，旧协议进度归档run/revisions/。
- 当前启动记录engineering_scope_launch.json，日志supervisor_engineering_scope.log。run/source/是原始快照，最新监督源码副本在run/revisions/*/source/，hash见当前协议。运行中不要编辑冻结的训练或监督源码。
- run/smoke/为已通过的新冒烟；run/trials/为10k；run/formal/只在正式实际启动后存在。失败查看campaign_error.json及相关learning_error.json/infrastructure_error.json。

## 阶段门槛与计数

- 每条transition对应一个最多7步的chunk。按回合采集，再随机抽batch更新；warmup前不更新。阶段预算单位为transition，不是episode、控制步或updates。
- 达到预算前补齐最后真实后继，再停止执行新chunk，report标budget_exhausted，不记物理timeout。实际提交数与replay留存数必须都等于10000/100000。
- 连续3batch出现显著loss下整体critic零梯度，或内部target下整critic精确饱和，停止并保留证据；具体阈值见validation.md。学习异常与基础设施错误分开。
- 10k后恢复learner、优化器、replay并验收；两组都通过才进入正式。正式A/B各重新初始化actor/critic/target/优化器/replay，不继承10k或已饱和权重。
- 最终各完成相同val-40，确定性、无参数/replay更新；比较成功率、耗时、异常接触及逐场景改善/退步，不作跨种子稳健性声明。

## 历史证据

旧接入产物在outputs/astribot_online_rl/latent_engineering_20261022/，101项CPU及初版9条提交/8条留存、critic饱和等结果只属于旧版本。implementation_report.md保留原数据；critic_diagnosis.md记录修复。当前精确预算和已修复短冒烟不能反向改写历史记录。

## 18:27主动停止后的当前状态

- 用户停止的是正式A；已关闭后台自动推进，worker1080395、监督1072000、policy1067507均退出，GPU已释放。
- 原10k产物完整保留：A51/84完整回合成功、B51/83；两组各10000条/10000更新，恢复通过。末条预算边界回合均从成功率分母排除。
- 当前比较仅训练轨迹，不是val-40；未擅自额外评估。先向用户呈现对比，后续操作以其新指令为准。
- 停止过程及最后可恢复checkpoint记录在run/user_stop.json，原进度在progress_before_user_stop.json；主动中断不计成物理失败、稳定性失败或完整回合。

## 最新：B组正式100k独立启动

用户“先启动B组正式100k”。已核验B的10k训练/恢复通过、训练源码身份一致、新输出目录不存在。当前独立监督1081368，策略服务1081370；运行状态同步run/progress.json，实际PID和状态以该文件为准。
B输出run/formal/state_latent_seed20261024/；监督产物b_formal_job_20261022/，日志b_formal_supervisor.log，启动记录b_formal_launch.json，脚本scripts/run_astribot_latent_b_job.py。
仅B从头100000条：不传resume，不继承10k/正式A权重或replay。参数与B10k一致（warmup5000、batch256、fan_in、q_lr3e-5）。A停止记录和488条/0更新checkpoint保留；不自动恢复A或评估。独立作业完成后关闭其策略服务，保留检查点和门槛证据。

## 最新20:47授权与监督交接

用户要求完成B训练及val-40，再A训练及val-40。已无损接管B worker1081413和policy1081370，当前监督1089229，交接时B20591条/20591更新，原B-only监督已移除。
脚本scripts/run_astribot_b_then_a_job.py；产物b_then_a_job_20261022/，日志b_then_a_supervisor.log，启动b_then_a_launch.json，状态run/progress.json。当前自动A/自动评估均已授权。
B精确100k及训练gate通过后先做完整七相机确定性val-40，检查40个冻结场景ID、参数/updates及训练文件hash不变；失败阻断A。A从已验证可恢复的原正式488条/0更新checkpoint接续至总100k，绝不继承10k试验权重。A通过后val-40，然后生成单种子paired_results.json。
编排顺序与评估门槛10项测试通过，A真实learner/replay恢复校验通过。当前源码已再次冻结，不编辑训练、旧/新监督源码以免身份门槛失败。任何阶段异常停止，不能把单个B训练完成写成整轮完成。

## 最新故障与审核恢复

21:29推理连接reset导致监督停止。B已保存27133条/27050更新；253完整回合外，infra中断回合保存83条合法数据，pending未提交。最后critic梯度非零、精确饱和为0，不能把基础设施故障记作critic失败。
已加载验证learner、优化器、replay及源码/配置一致，已备份B目录recovery/及hash，保留原报告和失败日志。按既有完整流程授权做一次已审核恢复，源码训练配置不改，fresh episode恢复，不跨episode连接。恢复脚本scripts/resume_astribot_b_then_a_job.py，恢复监督1094894、policy1094895，实际worker和状态看run/progress.json。
恢复日志recovery_supervisor.log、启动recovery_launch.json、监督目录b_then_a_recovery_20261022/，review和prior_failed_progress在该目录。只接受hash已确认的历史infra报告，合法83条计入预算；不把该回合算物理失败或完整成功率分母。新错误仍停止，不能自动忽略。内存采样余量足够，无kernel记录或服务Pythontrace，退出根因未确认。恢复后仍B100k→B val-40→A100k→A val-40。
