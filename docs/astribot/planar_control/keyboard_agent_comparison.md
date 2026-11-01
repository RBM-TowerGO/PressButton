# 键盘平面运动与 Agent 接入对比（2026-10-30）

## 对比结论

坐标符号和单位换算一致，但 Agent 在后续修复中改变了 SDK 调用路径和躯干目标类型。两者不是等价接入。当前恢复了键盘普通平移的连续 Cartesian 发送方式；此前只复用发送方法、但保留等待阶段的关节保持，不等价于键盘接入。

比较基准为 `scripts/s1b_keyboard_direct.py` 中 `send_targets(..., gripper_target=None)` 分支；旧键盘程序按 C/O 后会切到 mixed 分支，不能把那一分支也称为明确关闭远程 WBC 的路径。

| 项目 | 原键盘普通平移 | 修复前 Agent | 当前 Agent |
|---|---|---|---|
| 位置读取 | chassis 坐标系 | 同左 | 同左 |
| 方向 | 右 −Y，上 +Z，前 +X | 同左 | 复用 `target_for_key` |
| SDK 运动方法 | `set_cartesian_pose` | `set_different_type_command`，后续还做实例私有适配 | 直接复用键盘 `send_targets(..., None)` |
| 右臂目标 | 7 维位置＋四元数 | 7 维 Cartesian | 同键盘 |
| 躯干运动约束 | 7 维 Cartesian 目标 | 4 维 joints；适配剔除 Cartesian 分支中的躯干键 | 同键盘，保留完整 7 维目标 |
| WBC 参数 | use_wbc=False，remote_wbc_option=False | mixed API 没有公开 remote_wbc_option；适配选择本地 | 公开 API 显式传入两个 False |
| 默认躯干补入 | add_default_torso=False | mixed API 不提供该参数 | 同键盘 |
| 轨迹 | 五次平滑插值，朝向不变 | 五次插值，但采用另一条 SDK 路径 | 共用键盘 interpolate_pose；从上一命令插值，到位保持完整目标 |
| 错过发送周期 | 跳过，不补发 | 新增超过 100 ms 即退出 | 同键盘跳过、不补发；不再增加独立 watchdog |

右夹爪在启动 Cartesian 流之前通过 `close_effector` 单独闭合。之后等待、运动和到位保持始终调用 `send_targets(..., None)`，不切换右臂关节保持。状态读取和记录在独立采样线程进行。2026-10-31 临时加入的 3 mm 反馈领先限制已撤销：现场记录显示它会使目标停在尚未到达最终位置的中间点。

## 已确认的现场证据与不足

`outputs/agent_button_004` 中前两轮移动没有执行；初次与确认后观测的 SDK 末端位置变化约 10.1 mm，触发建议失效。它不是模型用小步反复移动。

`outputs/agent_button_005` 中前伸以外的一次平移请求为 right=-30 mm、up=-10 mm；任务失败状态中的底层错误是 `servo_deadline_missed`，外层只报告 stopped。旧失败记录没有保存运动期间的完整实际轨迹，无法据此精确重建全部法向偏差。因此这里确认的是代码接入差异及已记录的错误，不把某一物理偏差的全部成因说成已经实测证明。

当前增加了：

- 当前按最小接入要求移除了额外运动误差保护。键盘和 Agent 共用 `run_cartesian_loop`；Agent 只提交目标，插值发送结束即完成，不以实测位置/速度作为继续发送的条件。
- 失败任务保留原始 backend 错误、命令位姿、实测位姿和近期轨迹；后台故障退出同样保存 fault.json。
- API 进程退出前等待 stop 响应写完，避免已停止却收到连接重置。
- 相机 capture 从头部历史帧中选择与右腕深度最接近的一帧，避免将延迟处理的右腕帧与最新头部帧硬拼；仍保留 80 ms 匹配要求。

## 验证

88 项 CPU 回归通过，包括公开 SDK 方法和参数与键盘路径一致、两路目标均为 7 维、禁止 mixed 方法、平移目标 X 恒定、朝向恒定、前伸正常返回、等待阶段持续 Cartesian 保持、相机历史帧匹配与停止响应。

这些测试验证接入和目标路径，不代替实际末端路径验收。本次未自动驱动机器人试动；需要在现场以新日志核对实际平面偏差。末端状态仍为 SDK 定义的参考点，未声称 150 mm 长度已经构成完整 TCP 标定。
