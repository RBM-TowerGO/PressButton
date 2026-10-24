# 上一条实际控制指令：实现与验收

日期：2026-10-24。在已有原始视觉语言观测上新增 `state_latent_raw_command`。

## 实现

观测布局为 q32、qd32、raw latent2048、previous_command8，共 2120 维；Actor/Critic 总输入为 2176 维。
指令来自 `Recorder.command`，按冻结 joint8 规则提取右臂七关节和夹爪目标。
首次观测使用 reset 后 settling tick 的指令；后继和末帧只使用实际下发的目标。
新增模式能力判断覆盖服务校验、普通请求及末帧推理；推理期间检查指令不变。

原两个模式的观测契约和默认 state_only 保持不变，视觉特征 RPC 没有改动。
新模式保存控制指令来源、布局、单位、版本和初始化语义，恢复时检查维数与语义；旧 2112 维网络/replay 不转换。
源码身份仍遵循原有严格恢复规则，不绕过源码 hash 检查。
SAC、奖励、动作合成、控制频率保持原样，没有加入时间、任务 one-hot 或多步历史。

## CPU 验证

84 项通过，包括现有两个模式回归和新模式检查：

- joint8 提取、2120 维布局、缺失/非有限/错误形状指令拒绝。
- Actor 与 Critic 实际接收到指令特征，学习更新数值有限。
- replay/checkpoint 保存恢复、确定性动作一致，以及旧维度和同维度错误语义拒绝。
- 使用真实四步插值函数的 runner 测试：首次观测、普通后继、成功、超时、插值中途失败、动作拒绝与回合重置。
- 插值第二个物理步失败时保留中间目标，没有泄漏未来终点；推理期间指令变化明确失败。
- 三种模式的 learner 配置除 obs_dim 外一致。

## 真实 Isaac 短冒烟

使用 RTX 5090、冻结 VLA step 8000，仅运行新模式：一个 2 秒模拟训练回合、少量更新和一个 2 秒确定性 val 回合。
训练采集 9 条 transition，容量 8 的环形 replay 保留 8 条，完成 8 次学习更新。
控制步数 60、物理 tick 241；末帧额外推理约 68.52 ms，没有额外控制执行。
保存 learner/replay 耗时约 23.46 ms。
损失和网络参数有限，q_lr=3e-4、policy_lr=1e-4，沿用原设置。

请求日志与 replay 当前/后继观测的指令逐元素一致，末帧观测及基础动作 hash 对应真实 successor_only 请求。
每 100k transitions 当前与后继指令额外缓存为 6,400,000 bytes，约 6.4 MB。
成功/物理失败分支由 CPU 测试覆盖；短仿真回合不用于评估按按钮成功率。

确定性评估恢复通过，参数和更新数不变，learner/replay/status/protocol 四个训练文件 hash 不变。
本次启动的 8799 推理服务已关闭；历史实验产物未修改。

## 使用与证据

新训练显式指定 `--observation-mode state_latent_raw_command`，使用新的输出目录和新网络/replay。
恢复与评估指定同模式及原训练配置。短冒烟入口：

```bash
.conda/envs/pressb/bin/python scripts/validate_astribot_latent_engineering.py \
  --modes state_latent_raw_command --endpoint http://127.0.0.1:8799 --output <new_directory>
```

运行前需启动对应推理服务，完整说明见 [validation.md](validation.md)。

- [CPU 检查日志](../../../outputs/astribot_online_rl/command_engineering_20261024/cpu_tests.log)
- [真实冒烟结果](../../../outputs/astribot_online_rl/command_engineering_20261024/smoke/smoke_results.json)
- [源码身份](../../../outputs/astribot_online_rl/command_engineering_20261024/implementation_identity.json)

本次仅工程验收，没有长训练，不声明长期稳定性或成功率提升。
