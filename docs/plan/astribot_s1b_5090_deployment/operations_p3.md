# P3 影子调度操作手册

本阶段在机器人本地以 30 Hz 运行独立的影子消费线程。它选择目标并记录模拟发送，不创建 SDK 控制客户端，不申请控制权、不发布运动命令。真实 `--mode execute` 仍由统一配置检查拒绝；`enable_fake_execution` 仅允许精确类型的内存假执行器，用于故障测试。

## 运行

相机、ROS 环境、工作站模型服务和 loopback 反向 SSH 隧道按 [P2 手册](operations_p2.md) 启动，使用已核验的独立双相机副本。P3 首次在普通前台进程中复测该副本，不需要 GDB 作为运行依赖。

将更新的 `s1b_robot_bridge.py` 和 `src/pressb/s1b_deployment/{scheduler,shadow}.py` 同步到机器人独立目录，配置也同步为本次版本。从机器人独立目录执行：

```bash
python3 scripts/s1b_robot_bridge.py --mode shadow --duration 610 \
  --output outputs/p3_step002000_final
```

`--schedule` 默认启用；`--no-schedule` 可只跑 P2 采集和请求。开始前核对工作站服务启动参数 `--checkpoint 2000`。切换模型必须退出桥接器、停止原模型服务，启动 `--checkpoint 8000` 服务，再启动新的机器人进程和输出目录，例如 `outputs/p3_step008000_final`。每次生成独立 session，禁止在旧 session 内接受模型热切换。

配置 `network.tcp_nodelay: true` 使 HTTP 服务端关闭 Nagle 缓冲。短测证明原服务端的延迟会造成队列消费缺口；此项改变传输行为，NPZ 格式、协议版本、输入、模型精度、推理步数、deadline 和 30 Hz 目标节拍均不改变。不要以延长 TTL 或降低消费频率替代链路优化。

## 调度语义

- 第 k 个目标 deadline = 原机器人 head 接收 monotonic + round((k+1)×1e9/30)，不从响应时刻重新计时。
- 响应到达时 deadline ≤接收时刻的点不入队；整包过期则明确记录为 `scheduler_accepted=false`，旧预测队列也清空，不重放首点/末点。
- 每个 tick 先丢弃 deadline <实际 tick 时刻的点，再选择最早且位于当前一个策略周期内的目标；整数纳秒周期上界为 33,333,334 ns。已提前排队的点若恰好位于 tick 时刻可消费一次。
- 新观测替换旧队列尚未消费的部分，已记录的模拟发送不回滚，不混合 chunk、不插值、不裁剪模型原始预测。
- 真实网络线程最多一个 in-flight。30 Hz 线程只访问调度锁，不调用 HTTP、ROS 或模型；时钟使用机器人本地 monotonic。线程若错过多个 tick，不补发一串历史目标，而记录跳过的 tick 数。
- SHADOW 中队列空缺会记录 uncovered / would-fault 并继续收集证据。假 EXECUTING 路径会进入锁存 FAULT，清空队列并调用假 stop。两种行为分别用于统计和验证故障停止，不能混为执行授权。
- 网络或响应错误立即清队列并进入 FAULT；重连成功必须创建新 session，只能回到 SHADOW。模型/契约身份变化要求退出，不能自动启用执行。

生产入口的调度器不连接任何执行器；日志中的 `simulated_sdk_target10` 只是夹爪 0/100 解码后的假目标，实际 `sdk_target` 与 `actual_send_ns` 始终为 null。头部原始预测保留，任何偏离当前实测姿态都只记录。

## 证据与分析

除了 P2 的完整来源/请求日志及回放片段，还增加 `ticks.jsonl`。逐 tick 记录计划/实际时间、队列长度、session、观测编号、选择步号、原始目标、模拟 SDK 目标、deadline 与等待时间；无点可用会写明原因。不得仅报告请求 RTT。

```bash
.venv-astribot-inference/bin/python scripts/analyze_s1b_p3.py \
  --run outputs/s1b_deployment/p3_20261026/step002000 \
  --output outputs/s1b_deployment/p3_20261026/analysis_2000.json
```

分析程序检查 deadline 未重新起算、目标与原始请求对应、无过期/修改后的模拟发送。队列覆盖率分母包括正式统计窗口内所有应有 tick 槽，以及线程跳过的 tick；发现阶段和首个可用预测之前的冷启动另行保留，不计入正式持续窗口。正式窗口从首个调度接受时刻到最后有效响应，需至少 600 秒。

报告分开列出：相机 header→接收的记录时间代理年龄、接收→发送、RTT、服务端预处理/推理、RTT 减服务耗时、响应→tick、tick→deadline、可用窗口，以及 p50/p95/p99/最坏值。RTT 减服务耗时仍含序列化、传输与客户端处理，不能宣称为纯网络单程延迟。原始曝光时刻仍未标定。

还会统计逐维原始目标变化、模拟消费目标变化、夹爪切换、头部目标与观测偏差、仍有未来点的响应比例、选择步号和实际队列覆盖率。

## 测试与门槛

```bash
.venv-astribot-inference/bin/python -m pytest \
  tests/test_s1b_deployment.py tests/test_s1b_observation.py \
  tests/test_s1b_scheduler.py tests/test_s1b_network_faults.py \
  tests/test_s1b_p3_analysis.py tests/test_s1b_p3_fault_analysis.py -q
```

自动化测试不依赖 ROS/GPU，包含可控时间、假服务器与假执行器。真实网络故障试验独立于两轮性能长测保存，以免把人为故障混入正常场景的门槛统计；故障日志必须证明持续 tick、故障后清空旧队列、恢复到新 session 的 SHADOW，以及模型变化拒绝。

P3 工程门槛：有效响应比例和可消费队列覆盖率均 ≥99%，原观测接收→响应 p99 <200 ms，无未识别的记录时间域和静默消费过期目标。即使通过，也只允许进入 P4 的 SDK/约束及受控验证准备，不代表允许发送模型运动。P4 所需限值、停止顺序和人工启用仍须单独完成。

真实故障日志的独立复核命令：

```bash
.venv-astribot-inference/bin/python scripts/analyze_s1b_p3_faults.py \
  --run outputs/s1b_deployment/p3_20261026/fault_injection \
  --markers outputs/s1b_deployment/p3_20261026/fault_injection_events.json \
  --output outputs/s1b_deployment/p3_20261026/fault_analysis.json
```

本次三类实际注入依次为：对本次启动的模型进程发 SIGSTOP/SIGCONT，终止并重建本次 SSH 隧道，终止模型进程并启动另一份权重。编排脚本及明确的进程编号保留在原始证据根目录 `fault_orchestration_source.py`，只操作本次创建的进程。原始副本用于审计；复现时须先建立新服务并核对本次 PID，替换历史编号。marker 使用 SSH 采集的机器人本地 monotonic；恢复后的 session、FAULT 区段和逐 tick 目标再与 marker 关联。故障期间重连尝试数与成功恢复数分别报告。
