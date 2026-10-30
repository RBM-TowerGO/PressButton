# S1B 右手平面控制

分支：`feat/right-arm-planar-control`。本模块提供独立 HTTP 工具，不依赖 VLA 推理服务。Python 入口为 `pressb.planar_control.api.PlanarTools`，机器人入口为 `scripts/s1b_planar_control.py`。

实现与 CPU 测试已完成；真机仅做了只读传感器核查。**尚未完成 TCP/手眼/工作范围标定、SDK FK/IK 在线一致性验证、空中运动及接触按压验收。示例配置不能执行运动。** 所有数值是待验证的工程占位值，不是 S1B 已验收参数。

## 几何和执行

`p = p0 + u*right + v*up + d*forward`，内部单位为米、弧度。示例采用底盘 x 向前、y 向左、z 向上，必须结合真机重力方向确认。平面移动保持 d=0；按压保持 u/v，推进后返回原平面。TCP 是闭合夹爪的真实接触点。

每个动作先在笛卡尔空间采样五次平滑轨迹，再逐点求解 SDK IK，复核 FK、固定躯干、关节范围、相邻解跳变、速度及 SDK 自碰撞距离。SDK 无种子参数，因此不连续解直接拒绝；不裁剪、不隐式改换肘部构型。SDK 返回契约固定为 `(ik_flag, named_joint_dictionary)`。SDK 返回坐标需通过标定和启动时实测位姿交叉检查，不能套用仿真 S1 模型。

本地周期循环持续保持或执行目标，复用 `GuardedExecutor` 的 probe profile、控制权与状态检查、独立超时停止。禁止会裁剪关节轨迹的 model_hardware profile。规划与工具请求不在周期循环里执行；SDK 调用耗时过长仍可能触发周期超时或 watchdog，必须在真机上测定吞吐，不能据 CPU 测试声称实时性。默认 100 Hz 是待验证值。

非右臂部件必须已处于标定姿态，启动不自动归位。躯干显式保持，其他固定部件持续检查漂移。启动不抢占已有控制权。SDK 自碰撞检测不包含环境；`workspace_verified` 的证据必须覆盖整个手臂扫掠体和按压/空中测试通道的环境净空。TCP 在平面内不代表整条手臂都不会碰撞。

按压分为自由空间接近、低速接触推进、停留、撤回。异常时锁定停止，不自动撤回。`action_completed=true` 仅表示轨迹完成并到位；`contact_evidence` 和 `button_trigger_evidence` 当前均为 null，供调用者通过独立反馈判断，不能理解为按键成功。

## 只读核查

机器人端加载现有 ROS/SDK 的 **Python 依赖环境**，确认 `ROS_DOMAIN_ID=25`。不要为了探测而初始化 Astribot 客户端；此脚本默认只创建 ROS 订阅者，不启动相机驱动。

```bash
python scripts/s1b_planar_control.py \
  --config configs/s1b_planar_v1.json \
  --output outputs/planar_probe_001
```

输出包含配置快照、状态日志与 `probe.json`，可查看七关节力矩、深度有效比例及话题图。最大只读采样时间 60 秒；默认 5 秒。深度话题和 CameraInfo 话题需要按实时发现结果填入，未配置时只查能力与 RGB。

2026-10-29 实机核查结果见 [readonly_probe_20261029.json](readonly_probe_20261029.json)。确认右腕 D405 配置、七关节 torque 实际发布；本次没有图像/深度发布者。`/ft_data` 出现在话题图，但没有发布者或消息，不能据此宣称有可用腕部六维力传感器。未更改相机配置，未申请控制权，未下发运动。

## 标定配置

复制 `configs/s1b_planar_v1.json` 为现场配置，保留 schema，完成以下测量并把结果、日期、机器人/工具身份和日志路径写入 `calibration.evidence`：

1. 核实底盘轴与重力、FK/IK 根坐标、四元数 xyzw、关节顺序和单位。校准 TCP 相对手腕变换。闭合夹爪及初始朝向必须与标定相同。
2. 在 `fixed.positions` 中记录 `astribot_chassis`、`astribot_torso`、`astribot_arm_left`、`astribot_gripper_left`、`astribot_gripper_right`、`astribot_head` 的具名数组。启动前由已有人工流程定位到平面 `initial_uv_m`；运行服务不负责接近初始平面。
3. 记录无碰撞工作范围、允许法向通道、TCP 跟踪容差、关节上下限与速度。另复制现有 guard 配置，更新对应现场限值；规划器限值必须是 guard 和实机限值的子集，不能直接使用旧任务的狭窄关节范围。
4. 深度启用前确认话题、编码尺度、整流内参、时间戳、深度光学坐标系和手眼外参。输入是深度相机自身的整流深度，不需要假定已对齐 RGB；非零畸变会拒绝。`16UC1` 尺度默认 null，必须实测。配置工具遮挡像素区域，验证正前方法线射线附近的深度代表待按表面而非邻接墙面或夹爪。局部平面拟合检查支持点数、内点比例、平面倾角和残差。
5. 用不同工作姿态下的实际数据标定 torque 语义、基线与异常阈值。当前使用每关节相对固定基线的偏差阈值，只适用于经验证的小工作区；不是外力估计或恒力控制。原始 torque 包含重力/摩擦，不自动从一次静止读数产生按压力阈值。
6. 单独标定 `press` 的行程、速度、停留和误差余量，填入证据并设置 verified。接触按压还要求 torque verified。没有可信深度时，可显式标定 `manual_range` 的间隙及有效二维区域；服务不会在深度失败时自动切换为盲探。

手眼/深度无效不阻止已标定的平面微移。按压要求新的可信测距，并在规划完成后重新核对距离。深度只确定接近距离，不能用于证明毫米级按钮行程。若面板不平行于固定平面，应调整站位，不能让 Agent 修改法向。

## 运行与 API

现场完成标定后，显式开启执行：

```bash
python scripts/s1b_planar_control.py --execute --operator-present --estop-ready \
  --config /path/to/calibrated_planar.json \
  --guard-config /path/to/calibrated_guard.json \
  --output outputs/planar_run_001
```

`--operator-present` 和 `--estop-ready` 沿用已有执行器的本地运行条件。服务仅监听 `127.0.0.1:6210`，远程 Agent 通过 SSH 隧道访问；不提供公网监听开关。

```bash
ssh -N -L 6210:127.0.0.1:6210 S1B
```

```python
import time
from pressb.planar_control.api import PlanarTools

tools = PlanarTools()
observation = tools.observe()
receipt = tools.move_in_plane(
    du_mm=2, dv_mm=0,
    observation_id=observation['observation_id'], command_id='align-001',
)
while tools.get_state()['phase'] not in ('READY', 'FAULT', 'STOPPED'):
    time.sleep(0.1)
# 检查 result.status；完成后取新图像，再决定下一次移动或 press。
# tools.press(new_observation['observation_id'], command_id='press-001')
# tools.stop()
```

| HTTP | Python 工具 | 返回 |
|---|---|---|
| POST `/observe`，`{}` | `observe()` | observation_id、头部/右腕 JPEG base64、来源/接收时间、TCP、测距、机器人右/上方向提示 |
| POST `/move` | `move_in_plane(du_mm,dv_mm,observation_id,command_id)` | HTTP 202，task_id、command_id、accepted |
| POST `/press` | `press(observation_id,command_id)` | HTTP 202，同上；Agent 不传深度/速度/力 |
| GET `/state` | `get_state()` | 阶段、当前任务、结果、实测 u/v/d、TCP 误差、故障 |
| POST `/stop`，`{}` | `stop()` | 锁定停止后的状态 |

方向提示是机器人坐标定义，不能解释为图像像素方向。当前不提供未经标定的画面箭头或“图像中心即 TCP”标记；Agent 必须依据双视角及现场标定判断对准。

动作串行，执行中拒绝新动作。相同 command_id 和参数始终返回同一个 receipt，不重复执行；同 ID 不同参数拒绝。动作/规划拒绝后必须取新观测，旧图像和旧 observation_id 不可复用。HTTP 超时不意味着动作未被接受：重试须使用原 command_id。普通 Agent 连接断开时，已接收的有限动作仍由机器人本地完成并保持；机器人反馈/控制权丢失和周期超时则停止。故障后须关闭服务、查明原因，再创建新会话。

日志保存配置、工具观测双视角图像、深度 NPZ base64（含内参/外参/排除掩码）、实测/命令关节、TCP、torque、动作阶段和结果。日志写入失败或超预算会停止，默认总预算 1 GiB。双视角以默认约 10 Hz 持续保存 JPEG 帧，保留原始时间戳；安装 ffmpeg 后可导出两路按来源时间对齐的录像：

```bash
python scripts/export_planar_video.py outputs/planar_run_001/events.jsonl \
  --output outputs/planar_run_001/video
```

输出 `head.mp4`、`right.mp4` 与 `timestamps.json`。两路使用共同时间原点，较晚开始的一路以首帧填充前段，填充时长记录在清单；日志中的来源时间戳是时序核对依据。

## 验收

```bash
.conda/envs/pressb/bin/python -m pytest -q \
  tests/test_planar_control.py tests/test_s1b_executor.py \
  tests/test_s1b_async_log.py tests/test_s1b_cameras.py
```

先在确认空旷的工作区做小网格二维微移。法向往返使用仅本地 CLI 暴露的有限测试，不开放给 Agent，也不启用接触按压：

```bash
python scripts/s1b_planar_control.py --execute --operator-present --estop-ready \
  --air-probe-mm 3 --config /path/to/calibrated_planar.json \
  --guard-config /path/to/calibrated_guard.json --output outputs/planar_air_001
```

此测试需要通道净空验证，正常完成后停止退出。检查逐点平面误差、末端姿态、关节跳变、固定部件漂移、周期耗时和停止反馈。之后才启用 press，在多个 u/v 位置测定实际触发、接触异常与撤回效果。不得只修改 verified 开关跳过测量；空中测试通过也不能替代接触验收。

## 直接 SDK 键盘入口（无需离线标定文件）

按用户选择，另提供独立的 `scripts/s1b_keyboard_direct.py`。它不加载上述平面服务配置，也不读取 verified 标记；启动时读取 SDK 当前右臂末端位姿，以当前 X 建立竖直 YZ 平面，保留末端朝向，未按夹爪键前保持原开度，躯干使用启动位姿。Cartesian IK 由 SDK 完成。此入口不提供前述离线轨迹复核、深度测距或力矩接触监测，不能视为已经完成标定验收的按键控制器。

在工作站仓库目录的交互终端运行：

```bash
python scripts/s1b_keyboard_direct.py --host S1B
```

命令把独立脚本复制到 S1B 的临时路径，加载既有 ROS/SDK 环境，并通过 SSH TTY 运行；退出清理临时脚本。无需部署整个仓库，也不启动相机。SDK 控制权、原生限位仍然有效。控制权被占用时，SDK 会在终端询问是否接管；输入 `yes` 会接管并停止原控制器运动，直接回车则不接管。脚本不再自动替用户回答该提示；终端断开仍按拒绝处理。

W/S 上下、A/D 左右，每次 2 mm；F 前伸至初始平面前 100 mm 并保持（不累计）；R 退回初始平面；C 闭合右夹爪，O 打开右夹爪（移动期间也可操作）；空格/Q/Ctrl-C 请求停止并退出。夹爪使用 SDK 原生 0–100 闭合量；操作夹爪后使用 set_different_type_command，将机械臂、躯干和夹爪合为每周期一次 filter 指令，避免同周期连续两个 set 调用。前伸后先 R 再平移。默认目标轨迹峰值速度 20 mm/s，100 mm 前伸约 9.4 秒，SDK filter 可能增加实际到位时间。可使用 `--step-mm` 和 `--speed-mm-s` 设置步长与速度。位置方向采用底盘 +X 前、+Y 左、+Z 上；这一路没有对工具 TCP 偏置进行独立标定。

本次仅执行 CPU 测试，未启动该入口驱动真机。

直接键盘入口的退出处理：显式停止、释放控制权并关闭 ROS 后，独立机器人进程使用 `os._exit` 退出，避免 SDK 在解释器销毁阶段重复释放原生对象。已用假 SDK 验证 yes/回车/EOF 与初始化异常路径；原生 core dump 修复仍需真机复测。

频率与日志：默认 SDK 和发送循环均为 250 Hz，可用 `--control-hz 100` 同时调整两者；该参数与移动速度独立。重复的 calling frequency 警告只显示首次，退出时统计折叠条数，其他日志正常保留（包含原生 C++ 标准输出/错误）。单周期调用修复已用假 SDK 验证，实际调度频率仍需真机观察。提高速度示例：`python scripts/s1b_keyboard_direct.py --host S1B --speed-mm-s 50 --step-mm 5`。每段轨迹最短 0.2 秒，极小步长下仅调高速度参数可能不缩短时间。
