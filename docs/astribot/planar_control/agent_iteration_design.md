# Agent 迭代定位与往返按压接口设计

状态：原始设计记录。日期：2026-10-30。闭环代码已实现，当前运行方式、接口细节与验证边界以 [agent_run.md](agent_run.md) 为准；本稿保留设计背景。

## 交互目标

Agent 每轮只决定平面位移或一次法向往返距离。服务完成运动后采集实测位姿、头部 / 右腕图像、右腕深度，先保存为不可变的数据包，再返回给调用者。Agent 依据新数据决定下一轮，直到观察到按钮触发。

默认轨迹峰值速度为 **20 mm/s**，用于平移、前伸和收回。速度属于会话配置，首版动作接口不提供速度参数。服务不进行模型推理，也不主动连接某个 Agent 平台。

```text
初始观测并保存 → Agent 决策 → 平面移动 → 到位后观测并保存
                    ↑                         ↓
                    └──── 返回数据包 ─────────┘
对准后 → 前伸 Z mm → 到位后立即收回 → 观测并保存 → 判断触发结果
```

## 坐标和动作语义

接口使用相对位移，单位 mm，参数命名体现含义，避免和底盘 XYZ 混淆：

| 用户表达 | API 参数 | 正方向 | 负方向 |
|---|---|---|---|
| 左 / 右平移 X mm | `right_mm` | 机器人右侧 | 机器人左侧 |
| 上 / 下平移 Y mm | `up_mm` | 向上 | 向下 |
| 向前伸 Z mm 后收回 | `forward_mm` | 向前，必须大于 0 | 不支持 |

底盘约定仍为 +X 前、+Y 左、+Z 上，因此接口 right 对应底盘 −Y，up 对应 +Z，forward 对应 +X。图像中的左右不必与机器人左右一致。

会话启动时记录末端实测位姿、固定朝向、躯干姿态和初始平面。定义：

`p(u,v,d) = p0 + u*right + v*up + d*forward`

- 平移固定 d=0，同时给出 right/up 时走平面内直线，不分两段移动。
- 位移以本次受理时的稳定实测位姿为起点；该位姿必须属于最新返回的观测版本，并在允许的平面与姿态误差内。保存请求目标和实际终点，避免把未达到的目标当作下一轮起点。
- 往返动作固定 u/v 和朝向，记录出发位姿，前伸到 `d=forward_mm` 后返回该次出发位姿。多次调用不累计法向距离。
- 前伸实测到位后立即开始返回，**不设置按压停留时间，不等待拍照、保存或网络请求**。为平滑反向，轨迹在前端点减速到零后再回程，“立即”指无额外停留，而非瞬时反转速度。
- 20 mm/s 是轨迹峰值上限，不是全程恒速；加减速、短距离和实际跟踪会影响耗时。两轴同时移动时按合成速度限制，不是每轴分别 20 mm/s。
- 正常往返完成才自动收回；停止、控制权丢失或执行异常时中断动作，不再无条件撤回。

已知右手夹爪末端长度 **150 mm**。按用户指定，前伸距离使用估算规则 `forward_mm = fixed_point_depth_mm - 150`，优先取固定点 5×5 有效深度中值。该规则用于动作建议，不代表已经获得相机到 TCP 的标定变换；位姿仍标记为 `sdk_end_effector`，未标定的 `tcp_pose` 返回 null。深度无效或过期时重新观察；计算结果小于等于 0 时不前伸，超出允许行程时明确报告，不裁剪或拆分按压。

## 模型请求约定

固定提示词采用[英文系统提示词](agent_system_prompt.txt)，不包含“建议必须经过人工确认才会执行”这句话。模型请求只发送一张右腕视角图像，以及文字目标、固定测距点深度、机器人状态和近期动作结果；不发送头部图像。头部图像仍按原数据包约定保存。程序中的逐轮人工确认保持不变，不需要在模型提示词中说明。

对准后，模型依据固定点深度减 150 mm 提出前伸往返建议，不额外增加压入行程。客户端后续实现时应记录使用的深度值、来源与估算结果，并核对模型给出的距离是否符合公式。

## 建议接口

新增版本化 `/v2` 路由，保留现有 `/observe`、`/move`、`/press` 语义；不把现有自动测距按压接口悄悄改成定距往返。服务继续绑定 loopback，通过 SSH 隧道访问。

| 方法与路径 | Python 包装（拟定） | 用途 |
|---|---|---|
| GET `/v2/session` | `get_session()` | 会话 ID、坐标约定、20 mm/s 默认速度、有效限值与配置版本 |
| POST `/v2/observations` | `observe(command_id)` | 无运动，采集并保存初始或补充观测 |
| POST `/v2/actions`，type=move | `move(right_mm, up_mm, observation_id, command_id)` | 平移，到位后采集、保存、返回结果 |
| POST `/v2/actions`，type=press | `press(forward_mm, observation_id, command_id)` | 定距前伸并立即返回，再采集、保存、返回结果 |
| GET `/v2/tasks/{task_id}` | `get_task(task_id)` | 获取进度或最终结果 |
| GET `/v2/state` | `get_state()` | 实测状态、忙闲、故障和最新观测 ID |
| GET `/v2/observations/{observation_id}` | `get_observation(observation_id)` | 读取已经保存的数据包清单 |
| GET `/v2/artifacts/{artifact_id}` | `download_artifact(artifact_id)` | 下载该会话已登记的数据文件，支持重试 |
| POST `/v2/stop` | `stop()` | 优先请求本地停止并锁定运动 |

相机网页的选点与保存接口继续独立存在；每个观测数据包复制采样时的选点位置与 revision，历史包不随之后选点而改变。

### 请求示例

右移 10 mm、上移 5 mm：

```json
{
  "session_id": "session-001",
  "command_id": "align-001",
  "observation_id": "obs-001",
  "type": "move",
  "right_mm": 10,
  "up_mm": 5
}
```

在新观测对应的位置向前伸 30 mm 并立即收回：

```json
{
  "session_id": "session-001",
  "command_id": "press-001",
  "observation_id": "obs-002",
  "type": "press",
  "forward_mm": 30
}
```

左移传负 right_mm，下移传负 up_mm。move 两个分量都必须提供且不能同时为零；press 不接受平面位移字段。参数必须为有限数，不接受 NaN、Infinity、字符串或多余字段。

初始 / 补充观测请求包含 `session_id`、`command_id`；stop 请求体为空对象。session_id 防止重启后旧请求落入新会话。

### 异步任务与 Python 同步包装

动作与采集请求先返回 HTTP 202：

```json
{"session_id":"session-001","task_id":"task-001","command_id":"align-001","accepted":true}
```

服务串行执行，任务包括运动、稳定、采集和落盘阶段。任务结束前不接收下一动作，避免位姿和图像跨动作混用。GET task 返回 200；受理后规划失败也通过任务结果报告。

Python 包装提供 `wait=True` 默认行为：内部等待任务结束、下载清单及双路图像和原始深度、验证文件哈希，然后返回同一个结果对象和本地文件路径。HTTP 202 不等于 Python 调用已完成；`wait=False` 可只取任务回执。下载等待不会阻塞机器人周期控制线程。

超时只结束客户端等待，不表示机器人动作取消。使用同一 command_id 和完全相同的请求重试，返回同一个任务，不能重复移动或按压。请求摘要和任务回执应在执行前持久化；进程重启后未完成任务记为 interrupted，禁止自动重放。相同 ID 不同内容返回 409。

## 每次完成后的数据包

终态结果明确区分机械运动、数据采集和存储：

```json
{
  "session_id": "session-001",
  "task_id": "task-001",
  "command_id": "align-001",
  "status": "completed",
  "motion": {
    "action_completed": true,
    "returned_to_start": null,
    "speed_mm_s": 20,
    "contact_evidence": null,
    "button_trigger_evidence": null
  },
  "capture": {"status":"complete","observation_id":"obs-002"},
  "storage": {"status":"committed"},
  "observation_url": "/v2/observations/obs-002",
  "next_action_allowed": true,
  "error": null
}
```

`returned_to_start` 在 move 中为 null，在 press 中表示是否实测返回出发位姿。任务 completed 表示动作完成且必需数据成功保存，**不表示按钮已经触发**。若没有独立反馈，接触和触发证据始终为 null。

观测清单至少包含：

| 数据 | 记录内容 |
|---|---|
| 标识 | schema、session/task/command/observation ID、配置版本与哈希 |
| 时间 | 动作开始、前伸到位、返回到位、采样、保存时间；UTC 与本地单调时钟分开命名 |
| 请求和结果 | 原始动作、目标位姿、实际位移、位置 / 姿态误差、速度、各阶段耗时与错误 |
| 实测末端位姿 | frame_id、reference_point、position_mm、orientation_xyzw；经过验证时另含 tcp_pose |
| 实测机器人状态 | 右臂关节角 rad、速度 rad/s、SDK 原始力矩及字段语义、夹爪开度、躯干状态 |
| 双视角图像 | 头部 JPEG、右腕原始 JPEG；可另存带测距十字的预览，不覆盖原图 |
| 右腕深度 | 原始 uint16 NPY、scale_m_per_unit、宽高、光学坐标系；可另存对齐到 RGB 的深度及有效掩码 |
| 深度参数 | 深度 / RGB 内参、畸变、两者外参、设备 ID、来源时间戳 |
| 测距点 | 归一化位置、像素位置、selection_revision、单像素深度、5×5 中值、有效像素数 |
| 质量 | 各来源 age_ms、RGB / 深度 skew_ms、是否过期、缺失项、测距不可用原因 |
| 文件清单 | artifact_id、相对下载 URL、媒体类型、字节数、SHA-256 |

深度数值使用明确单位；相机 optical Z 深度与末端前方距离分字段表达。未有手眼变换时 `tcp_surface_gap_mm` 为 null；另记录 `estimated_forward_mm = fixed_point_depth_mm - 150`，作为用户接受的前伸估算量，避免将其标记为已标定的实测间隙。

### 采样时机

平移完成后，先确认实测位置和速度稳定，再获取该完成时刻之后的新图像、深度与状态。往返动作以**返回完成后**的观测作为下一轮输入。按时间戳匹配各路数据，记录实际时间差，不声称硬件同步；新鲜度和最大时间差采用服务配置并通过 session 接口公开。

运动期间持续写入带阶段标签的状态和相机环形缓冲区。前伸顶点只记录到位事件，并异步保留最接近该时刻的图像 / 深度；不为了采到顶点照片延迟撤回。没有时间匹配帧时明确缺失。这样能保留瞬时按钮变化，而不只依赖撤回后的画面。

### 保存与传输

建议目录：

```text
outputs/agent_sessions/<session_id>/
  session.json
  commands.jsonl
  tasks/<task_id>/result.json
  observations/<observation_id>/
    manifest.json
    state.json
    head.jpg
    right.jpg
    right_depth_raw.npy
    depth_metadata.json
    right_preview.jpg
```

数据先写临时目录，文件写完并提交清单后原子发布，完成结果才引用该 observation。返回 JSON 元数据和文件下载地址，Python 包装默认自动下载，避免调用者还要自行拼装图像与深度。二进制文件不在轮询响应中反复 base64 传输。

网络下载失败可重试，不重做运动、不重拍替换历史数据。客户端下载完成不作为机器人端“运动完成”的条件，但下一命令必须引用已提交的最新观测。默认不主动向外部服务器推送；后续 Agent 通过包装获取本地数据或文件 bytes。数据保留期限由会话配置声明，活动会话数据不在下载期间自动清理。

## 状态与失败处理

运行阶段：`READY → MOVING → SETTLING → CAPTURING → SAVING → READY`；往返用 `EXTENDING → RETRACTING → SETTLING` 替代 MOVING。每个阶段都响应 stop。

任务终态为 completed / rejected / interrupted / failed。采集或保存失败时，即使 motion.action_completed=true，也不能报整个任务 completed，且不能重试运动来补数据。保存可得的状态与错误，进入 NEEDS_OBSERVATION；只允许重新 observe 补齐数据、读取状态或停止。原动作结果保持不变，新观测关联原任务。

无有效测距像素不等于原始深度流缺失：原始帧有效但选点是空洞时可成功保存，距离为 null。缺少新鲜的必需图像 / 原始深度则 capture 不完整，不把旧数据当作本轮成功观测。

停止或执行异常进入 STOPPED / FAULT，拒绝后续运动，尽可能保存已有证据，不为了补拍或撤回继续运动。普通网络断开时已接收的有限动作由本地完成并保存。

请求层错误建议统一为：400 参数错误；409 忙碌、ID 冲突或旧观测；422 超出本地允许的步长 / 工作区 / 前伸距离；503 控制或传感器服务不可用。错误体包含 code、message、retryable、task_id（如有）。

本设计保留 SDK 控制权、原生限位和停止机制；不新增以离线 verified 文件为启动前提的流程。具体步长、法向最大行程和跟踪容差由本地配置显式设置并通过 session 返回，不能因“默认 20 mm/s”就推断任意 Z 都可执行。SDK 位姿模式与已标定 TCP 模式必须在返回值中明确区分。

## 后续实现范围

1. 从直接 SDK 入口提取可复用的相对平移 / 有限前伸往返执行器，保持控制线程与图像、磁盘、网络分离。
2. 复用网页深度解码、设备参数和选点配置，增加带时间戳的观测打包与文件存储。
3. 实现 `/v2` 服务、持久化命令去重、任务结果和 Python 等待 / 下载包装。
4. 使用假 SDK 验证方向、20 mm/s 合成速度、固定朝向、前伸不累计、到位立即返回、停止优先、重复请求、掉线与保存失败。
5. 真机记录实际往返和停止响应，再验证“平移—新观测—平移—往返—新观测”完整闭环；接入 Agent 放在这些接口稳定之后。

现有标定 API 与键盘入口保持原有行为；新增独立服务实现 `/v2`，不能与原服务同时占用默认端口。
