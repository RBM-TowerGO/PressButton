# Astribot latent 接入任务记忆

更新：2026-10-22（北京时间）。

## 授权范围与约束

- 用户批准工程接入、CPU 契约测试和真实 GPU 冒烟；不运行 10k、正式三种子 100k 或完整 val-40。
- 不修改 models 历史运行时，不改奖励、Q 有界输出、动作尺度、bootstrap 或 SAC 结构；未来方向不自动启动。
- 默认 state_only，state_latent 必须显式选择；禁止跨模式/特征版本恢复，不补零或复用旧 latent。

## 已完成与证据

- 外部模型适配、RPC latent/健康声明、2112 维观测、journal/replay/恢复身份检查、真实末帧后继、确定性评估及诊断已实现。
- CPU：101 passed；真实 C01–C03：新旧动作逐元素相同，tokens [1,32,2048] float32，池化及重复请求通过。
- 最终 GPU：final/ 下两模式各一回合 2 s 训练和一个 2 s val 评估；各实际产生 9 条 transition，replay 留存 8 条，updates=8，控制步=60。latent 请求=10，其中末帧只采特征；state_only 请求=9。
- replay 实际产物恢复通过，learner 恢复后七相机确定性评估通过；参数、updates 和训练文件哈希不变。
- C08 成功/物理失败、特征失败等分支在 CPU 真实 runner 仿真替身中验证；GPU 本次只实测 timeout 分支，不能写成 GPU 已覆盖所有终止情况。
- 全部结果：outputs/astribot_online_rl/latent_engineering_20261022/；正式本次版本在 final/，real_interface/ 为真实接口证据，cpu_tests.log 为测试输出。
- implementation_report.md 为完整验收记录，engineering_summary.json/source_manifest.json 保存机器可读结果和源码身份。

## 发现的问题与后续边界

- latent critic 从第 2 次更新起边界比例=1，第 4–8 次更新 Q=0 且梯度=0。工程数据链路通过，但学习稳定性未通过；不宣称算法有效，不继续长训练。
- 原奖励/Q/终止规则保留；若修正必须先设计新协议并重跑 A/B，不能混入本次实验。
- 原 runner 在回合末检查预算，因此烟测产生9条、容量8；结果已分别记录实际产生与留存数量，不能把留存数冒充采样预算。
- 早期目录创建冲突失败证据在 state_only/；首轮通过在 verified/；补强 replay 语义和中途错误区分后，最终源码已在 final/ 重跑通过。历史产物未覆盖。
- 评估 CLI 必须匹配训练契约参数；完整复现命令保存在 final/*_command.json。恢复同时校验源码 hash，源码变更后不能把旧 checkpoint 无声当作同协议恢复。

## 最新授权与运行（后续用户指令覆盖早期范围）

用户明确授权：修复后 GPU 冒烟 → A/B各10k稳定性 → 通过后六次100k及val-40。
本轮已启动监督流程，完整协议和实际状态见 training_campaign.md 和 outputs/astribot_online_rl/latent_fixed_campaign_20261022/run/progress.json。
A/B统一fan_in+q_lr3e-5；正式从头初始化；新增精确预算停止及连续3batch严重学习异常停止。113项CPU测试通过。源码hash与副本已冻结，运行过程中不要改训练源码，否则下一阶段会因版本不一致停止。
旧的“尚未运行”结果对应历史版本；新阶段必须读取当前产物，不能把已启动记为已完成。

### 修复后在线冒烟已实测通过

本轮 run/smoke/：A/B 各精确8 transitions、8 updates，无零critic梯度；最后Q均值分别约0.473/0.477，保存恢复和七相机确定性评估通过。2026-10-22 16:18（北京时间）已自动进入 trial_state_only_train。B组10k和正式六次100k仍待前置gate通过，不能写为完成。
监督进程1067506后台独立运行，启动记录launch.json；不要结束仍在工作的策略服务。
