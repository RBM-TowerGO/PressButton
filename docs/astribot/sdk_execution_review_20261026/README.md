# 官方 SDK 控制接口核查：2026-10-26

来源：S1B `/home/astribot/astribot_sdk_aarch64`，commit `2644cdfba47f3dd311d923b172ac6c24909f6fda`。原文件副本及 SHA-256 见 `manifest.json`。仅阅读和复制文件，没有运行示例、申请控制权或下发运动。

## 核查结论

检查到的 SDK examples 没有 VLA 模型推理循环；它们提供状态获取、实时目标控制、定时轨迹与停止接口。模型产生目标后如何转换为控制轨迹，需要部署端实现，不能把模型预测频率直接视为官方控制循环频率。

| 官方文件/接口 | 实际写法 | 当前部署的差异 |
|---|---|---|
| `104-set_joint_position.py` | `Astribot(freq=250)`，单独创建 `Rate(250)`，每轮调用 `set_joints_position()` 后 `rate.sleep()`；默认 `control_way="filter"` | 当前执行 runner 继承 30 Hz shadow runner，后端强制 `direct` |
| `105-set_joint_position_direct.py` | 同样采用 250 Hz 应用循环；注释明确 `direct,use_wbc=False` 将目标不经处理直接发送给关节；示例使用 `0.3*sin(t)`、`t += 1/250` 的连续轨迹 | 当前离散模型目标经硬件最大速度限幅后以约 30 Hz 调用 direct；不等同于官方连续轨迹示例 |
| `103-move_to_joint_position.py` | 查询 SDK 关节范围，使用 `move_joints_position(..., duration=3.0)` 完成有持续时间的起始运动 | 当前只短暂重复旧保持指令，没有规划到模型首目标的起始过渡 |
| `206-move_joints_waypoints.py` | 传入 waypoint 和累计到达时间，说明不必给 time=0，当前姿态即默认起点 | 当前自行按目标期限选择单个点，没有走 SDK 定时 waypoint 路径 |
| `208-traj_replay.py` | 读取示范 command 与时间戳，首点到达时间设为 2 秒，再用记录时间差累计到达时间，调用 `move_joints_waypoints()` | 当前首个模型目标没有类似数秒级起始过渡 |
| `999-stop_robot.py` | 使用 SDK `stop_robot()` / `restart_robot()` | SDK stop 调用本身不能证明反馈已稳定；当前日志里反馈稳定验证仍失败 |

构造函数的 `freq=250` 会传给原生 `AstribotFunction`，但不会把我们的 Python 30 Hz 循环改成 250 Hz。原生实现以二进制库提供，未见完整调度/滤波实现源码，不能仅根据构造参数断言底层发布或执行频率。

## 在线接口细节

`astribot_client.py:704`：

```python
set_joints_position(names, position,
                    control_way="filter",
                    use_wbc=False,
                    add_default_torso=True)
```

Python interface 将以上字段传给原生实现。`set_filter_parameters(filter_scale, gripper_filter_scale)` 文档只说明数值更低时更平滑，示例是 `(0.1, 0.5)`；这不是当前机器人已验证的滤波参数，不能把示例数字当部署标定值。

当前后端另外显式设置 `add_default_torso=False`。官方默认值为 True。这个差异需作为控制范围选择明确记录，不能把 True 直接照搬并意外添加躯干命令。示例中的 `move_to_home()` 同样不应在本任务中自动执行。

## 与最新真实日志对照

运行 `live_1793004721715852340`：6 次保持、4 次模型发送。首个模型目标将右臂第 6 关节（零基索引 5）从上一指令约 `0.115000` 改到 `-0.142957`，变化约 `0.257957 rad`，距上一发送约 `33.741 ms`，指令差分速度约 `7.65 rad/s`。这不是实测关节速度，但清楚显示了目标变化幅度。

官方 direct 示例的正弦增量约最多 `0.3/250 = 0.0012 rad` 每轮，并非把接近硬件最大速度的模型跳变直接当连续轨迹。当前速度限幅没有限制加速度、jerk，也没有保证首目标平滑衔接。把 URDF/配置里的最大速度当作正常执行过渡速度，是当前部署设计中的不足。

驱动随后报告 `13243026`。现有 SDK 错误表没有这个代码的定义，因此不能据此证明具体报错是速度、加速度还是其他保护；上面的接口差异也不能独立证明该代码的根因。

## 审查时提出的部署方向

1. 将模型预测/30 Hz 动作时间线与机器人目标发送循环分离。按官方示例建立独立的 250 Hz 应用控制循环，避免网络请求阻塞发送。
2. 在线接口优先对齐官方 `filter` 路径，并核实其滤波参数、原生执行时序。若使用 direct，则部署端必须提供连续且经过速度/加速度等约束的轨迹，不能依赖原生隐式平滑。
3. 为当前保持姿态到首个模型目标规划明确起始过渡；不要自动执行示例的全身 home，也不要偷偷延长旧预测的有效期限。起始过渡属于独立状态，必须重新获取观测和预测后进入闭环。
4. 保留模型原始目标、过渡目标、实际 SDK 目标、驱动错误和反馈。停止、断网和过期策略需按分离后的两个循环重新验证。

本次是接口审查与证据整理，不是新版本真机闭环验收，也未修改运行时控制模式。

后续实现：已按本次审查将执行路径改为独立 250 Hz 应用线程与 SDK filter，并加入实时新预测的 2 秒启动过渡。上面的差异表记录的是审查时旧实现；最新行为和验证边界见 [执行说明](../../plan/astribot_s1b_5090_deployment/operations_execute.md)。
