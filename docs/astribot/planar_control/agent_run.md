# Thor 模型与 S1B 人工确认闭环

本功能已实现。默认 20 mm/s；模型只接收右腕视角，固定点测距按 `forward_mm = depth_mm - 150` 估算前伸距离。前伸插值指令发送完后立即发送回程，不设置额外停留。每次机器人运动需要在本地终端输入 `y`，英文模型提示词不提人工确认。

这是独立 `/v2` 服务，与原标定 API 使用相同默认端口但不同时运行。键盘直控也不能同时占用同一机器人控制权。服务不要求离线 verified 文件；它保持 SDK 控制权、实测状态和停止机制。

## 启动：两个终端

在工作站仓库根目录，终端一运行：

```bash
python scripts/s1b_agent_service.py --host S1B
```

脚本自动打包部署到机器人临时目录、启动相机网页、建立 6210 / 8765 隧道。按 SDK 原有提示决定是否接管控制权。取得控制权后，先用公开 `close_effector` 完成右夹爪闭合，再读取右臂和躯干初始 Cartesian 位姿并启动发送循环。与键盘普通平移一致，启动、等待、运动、到位保持都使用 `send_targets(..., None)`，每周期一条 `set_cartesian_pose`，两路目标均为 7 维，显式 `remote_wbc_option=False`。会话中不向右臂切换发送关节保持，也不调用 mixed API。夹爪保留启动时的闭合目标。状态读取和轨迹记录在独立采样线程进行，不占用发送循环。等待“Agent API 已就绪”。网页地址为 <http://127.0.0.1:8765>，可调整并保存右腕测距点。

终端二运行：

```bash
.conda/envs/pressb/bin/python scripts/s1b_agent.py \
  --goal '按下数字3按钮' \
  --output outputs/agent_button_001
```

此命令自动建立到 Thor 的隧道 `127.0.0.1:18000 → Thor:127.0.0.1:8000`，使用模型 `qwen38`。本地环境需要项目已有依赖 NumPy、Pillow；服务端复用机器人 SDK、ROS 和相机依赖。输出目录必须尚不存在，避免覆盖记录。

每轮显示模型建议及右腕图像路径：

- `y`：确认本次动作。
- `r`：不执行，重新观察和请求模型。
- 其他输入：退出并请求停止；空输入不会执行。
- Ctrl-C：请求停止；模型网络请求不会阻塞本地信号处理。

模型报告完成时，需要人工确认可见触发证据。默认最多 20 轮。程序退出会停止机器人服务并关闭自己创建的 Thor 隧道；再次运行任务需重新启动终端一的服务。不会停止 Thor 模型或共享相机驱动。

## 参数

| 入口 | 参数 | 默认 / 含义 |
|---|---|---|
| 服务 | `--host` | 指定 S1B 时自动部署；省略则在已加载 SDK 环境的机器人上运行 |
| 服务 | `--port` / `--view-port` | 6210 / 8765 |
| 服务 | `--control-hz` | 250 Hz，与 20 mm/s 移动速度独立 |
| 服务 | `--max-move-mm` | 单次平移向量长度上限 20 mm |
| 服务 | `--max-forward-mm` | 单次前伸上限 100 mm |
| 服务 | `--output` | 机器人端 `~/.local/share/pressb/agent_sessions` |
| 客户端 | `--goal` | 必填文字目标 |
| 客户端 | `--output` | 必填本地新输出目录 |
| 客户端 | `--robot-base` | `http://127.0.0.1:6210` |
| 客户端 | `--thor-host` / `--thor-port` | Thor / 18000 |
| 客户端 | `--model-base` | 已有 `/v1` 地址；指定后不自动建立 Thor 隧道 |
| 客户端 | `--max-rounds` | 20 |
| 客户端 | `--prompt` | `docs/astribot/planar_control/agent_system_prompt.txt` |
| 客户端 | `--replay` | 从已保存观测目录调用模型，绝不连接机器人 |
| 客户端 | `--session` | 回放时可指定保存的 session.json；默认限值 20 / 100 mm |

上述行程上限是软件限制，不是已经验收的工作范围。API 五次插值轨迹速度上限为 20 mm/s，不由模型修改。

## 每轮处理

1. 采集并保存新鲜的头部 / 右腕图像、原始右腕深度及对应机器人状态，下载到工作站。
2. 请求模型时只发送带绿色测距十字的右腕 JPEG、文字目标、固定点深度、实测状态、动作限值与最近 5 轮结果。头部图像和原始深度保存在本地，不作为模型图片输入。
3. 模型返回一个工具调用：平移、前伸往返、重新观察、完成或无法继续。未知工具、多工具、额外参数或非法数值不会被执行；最多进行一次输出格式修正请求。
4. 人工确认后再采集一次观测。测距点版本改变，或前伸估算变化超过 2 mm 时重新推理和确认。小于等于 2 mm 的深度波动允许沿用已确认距离，不暗中改写动作量。
5. 平移以上一次命令位姿为起点，正 right 为机器人右侧、正 up 为向上。前伸固定平面位置及朝向，前伸指令结束即回程。末端位置使用 SDK 参考点，未声称已标定 TCP。
6. 运动完成后采集新观测、原子保存，再返回成功。前伸顶点另由独立采样线程尽量保留最近画面和深度，300 ms 内无匹配帧时标记缺失，不等待它才撤回。

相机 RGB / 深度 / 头部来源时间差最多 80 ms，接收和来源新鲜度均为 1 秒。实测右臂状态与右腕图像来源时间差最多 80 ms。所有图像必须晚于本次运动完成或观察请求开始。无效测距像素可以保存为 null；缺少新鲜原始帧会使采集失败。

前伸建议优先取 5×5 中值，无中值时才用有效单像素深度，减去 150 mm。模型输出与决策时公式误差须不超过 0.1 mm；执行前重新观测允许上述 2 mm 变化。结果不大于 0、超行程或无深度时不前伸。此减法是用户采用的估算规则，不写入 TCP 标定，也不证明按钮触发。

## `/v2` HTTP 接口

| 方法 | 路径 | 请求 / 返回 |
|---|---|---|
| GET | `/v2/session` | 会话 ID、20 mm/s、行程上限、坐标和误差容限 |
| POST | `/v2/observations` | `{session_id, command_id}`；202 任务回执 |
| POST | `/v2/actions` | move 或 press；202 任务回执 |
| GET | `/v2/tasks/{task_id}` | 阶段、运动结果、采集 / 保存结果、观测 URL |
| GET | `/v2/state` | 当前阶段、故障、实测状态、最新观测 ID |
| GET | `/v2/observations/{observation_id}` | 不可变观测清单 |
| GET | `/v2/artifacts/{artifact_id}` | 清单登记的二进制文件 |
| POST | `/v2/stop` | `{}`；优先停止并锁定会话 |

move 请求字段：`session_id, command_id, observation_id, type="move", right_mm, up_mm`。

press 请求字段：`session_id, command_id, observation_id, type="press", forward_mm`。

同一 command_id 与原参数重复提交返回原任务；同 ID 不同内容拒绝。动作执行中拒绝新的动作或观察。HTTP 202 不表示运动完成；轮询任务直到 completed / failed / interrupted。参数或状态拒绝返回 400，未知资源 404，内部 / 暂时不可用错误 503，错误体包含 `error`。这是当前实现的状态码，区别于早期设计稿中更细的状态码划分。

任务在执行前写入机器人端日志。重启服务创建新 session，旧的未完成日志标记 interrupted，不重放动作。同一数据根目录只允许一个服务持有写锁。

保存 / 采集失败但运动已完成时，结果仍保留 `motion.action_completed=true`，服务进入 NEEDS_OBSERVATION，只允许补充观察或停止，不能重试原运动来补数据。机器人异常时 FAULT / STOPPED，不自动撤回。客户端意外网络断开后，已接受的有限动作在机器人端完成并保存；重连应查询原任务，不生成新动作 ID。

## 文件与回放

机器人目录为 `<output>/<session_id>/`，包含 session.json、tasks、observations。每个观测包含：

- `manifest.json`、`state.json`，包括实测位姿、原始关节状态 / 力矩和时间戳。
- `head.jpg`、`right.jpg`，原始图像不绘制十字。
- `right_depth_raw.npy`、`depth_metadata.json`。
- 前伸顶点可用时的 `apex_head.jpg`、`apex_right.jpg`、`apex_depth_raw.npy`、`apex_metadata.json`。
- 文件 SHA-256、字节数、设备参数、固定点版本、估算距离及顶点采样时间差。

本地目录另存 `task.json`、每轮请求 / 原始响应 / 决策 / 确认 / 执行观测对应关系、运动请求和任务结果。下载验证 SHA-256；下载重试不会重新运动。模型使用的带十字图像以 data URL 保存在该轮 request.json 中。

只调用模型验证已有观测，不连接机器人：

```bash
.conda/envs/pressb/bin/python scripts/s1b_agent.py \
  --replay outputs/agent_button_001/observations/OBSERVATION_ID \
  --session outputs/agent_button_001/session.json \
  --goal '按下数字3按钮' --output outputs/agent_replay_001
```

## 验证边界

```bash
.conda/envs/pressb/bin/python -m pytest -q \
  tests/test_agent_loop.py tests/test_s1b_camera_view.py \
  tests/test_s1b_keyboard_direct.py tests/test_planar_control.py
```

包含假 SDK 插值与往返、停止、重复请求、旧观测、保存失败、图像 / 深度导出、模型格式、单张右腕输入、未确认不执行、人工确认完整循环与重启不重放测试。已使用真实 Thor 完成合成右腕图像的客户端请求与工具参数解析；未为本次验证执行真实机械臂运动。真机到位误差、停止响应和接触结果仍需现场逐轮确认验证。

连接排查：客户端启动前会先检查机器人 API。若 6210 连接被拒绝，请查看终端一是否已经退出；仅启动 Thor 隧道无法提供机器人 API。`torso_drift` 会报告具体 mm / 度误差，并尽可能将初始、目标和实测躯干数据写入机器人会话目录的 `fault.json`。

步长与复核：默认单次平移向量上限为 20 mm；需要更大粗调时，可在服务启动命令加 `--max-move-mm 50`，模型会读到该上限。若提示“本轮动作未执行”，该建议没有下发；终端和本轮 `invalidated.json` 会记录位姿变化量、选点版本和前伸估算，不能把反复重新推理理解为已经完成微移。位姿漂移不再作为运动门槛。

运动接入、实测偏差监测及历史故障对比见[键盘与 Agent 接入对比](keyboard_agent_comparison.md)。任务异常现在保留底层原因和失败前轨迹，停止响应在服务退出前发送完毕。


## 2026-10-31 接线复核

第一次跟踪故障的修复只加了反馈限速，没有恢复完整的键盘发送方式。随后会话 `f8f4c1539ece44d2b1f4f4ea59748471` 中，上移 15 mm 的最终目标 Z 约 1.0636 m，但发送目标停在约 1.0590 m，反馈约 1.0560 m；新增的 3 mm 领先限制使目标未走完就卡住。这层限速已撤销。

对照原始键盘代码，当前恢复了以下接入：

- 从启动到到位保持，全程连续发送 Cartesian，不在等待/运动之间切换关节控制。
- 键盘与 Agent 共用 `interpolate_pose` 和 `send_targets`，插值从上一命令位姿开始，到位后保留完整目标。
- 夹爪在启动 Cartesian 流之前单独闭合，运动期间没有竞争的关节发送。
- 状态读取/FK 和历史记录移至独立线程，反馈仅用于观测记录，不参与发送循环的运动判定。

按最小接入要求，已移除额外的跟踪误差、平面/姿态偏差、无进展/到位超时、状态速度和位姿复核门槛以及独立 watchdog。键盘和 Agent 共用 `run_cartesian_loop`，保留原控制权检查、SDK 异常退出和人工停止。故障后 API 保持查询能力，拒绝新动作；客户端读取失败任务后通过 `/v2/stop` 退出。

测试验证连续调用链、完整 15 mm 目标、慢状态读取不阻塞发送、往返、故障查询和停止；不代表真机运动验收。重启终端一会自动部署当前代码，终端二使用新的输出目录。


任务中的 `completion_basis=command_interpolation` 表示插值指令已经完整发送，不声称实测到位。按压结果的 `command_returned_to_start=true` 只表示回程命令完成，实测 `returned_to_start` 为 null。按钮触发仍由后续图像及人工确认判断。新增回归覆盖反馈完全不动时仍发送完整 20 mm 目标，以及连续两次动作按命令位置累加。
