# P2 有线只读采集操作手册

本阶段只运行传感器订阅和 HTTP 预测。`s1b_robot_bridge.py` 默认 `--mode shadow`，`--mode execute` 在导入 ROS 之前通过统一配置校验拒绝启动，不包含 SDK 控制客户端、运动 publisher 或动作执行器。

## 已核验连接与环境

- 工作站：`192.168.0.100`，接口 `enp11s0`；机器人 Orin：`192.168.0.11`，接口 `eno1`，SSH 别名 `S1B`。
- 工作站以 `.venv-astribot-inference` 加载模型；机器人以系统 Python 3.10、NumPy 1.24.0、OpenCV 4.11.0、ROS Humble / domain 25 采集。
- SDK 源码 commit：`2644cdfba47f3dd311d923b172ac6c24909f6fda`。采集程序只使用 ROS 消息定义，不导入 SDK 控制库。

非交互 SSH 使用以下仅加载库和 ROS 通信变量的环境片段。不要直接 source 顶层 `robot_env.sh` / `orin_robot_env.sh`：它们额外修改 DDS 配置或 sysctl，并可能调用 sudo。

```bash
source /opt/astribot_ros/middle_ware/local_setup.bash
source /opt/astribot_ros/software/local_setup.bash
source /opt/astribot_ros/third_pkg/local_setup.bash
source /opt/astribot_ros/robot_system_ctrl/ros_env.sh
```

在机器人检查相机话题有持续数据，而不仅检查话题存在。相机专用服务 `/astribot_camera_control` 的 `SetBool(data=true)` 可开启传感器，不能用构造 `Astribot` 控制客户端替代。此次发现服务回复 ON 但采集主进程退出；应核对 `astribot_camera_main` 和 `astribot_camera_sub` 两个进程，保留日志并确认像素流恢复后再验收。

## 本次验收使用的独立双相机配置

原始全相机配置在长测中发生 JPEG 错误和采集主进程段错误，本次不将该配置判为可用。复制原始相机程序和配置到独立目录，仅在副本的 `camera_config_list` 中保留 id 6（right_wrist_rgbd_color）和 id 9（head_rgbd），其余系统配置不改。

```bash
python3 scripts/prepare_s1b_cameras.py \
  --output /home/astribot/pressb_readonly_p2/camera_minimal
```

输出目录须不存在。准备器保存原始文件和副本的完整 SHA-256 manifest。必须保留 head 的 `CameraType: 3` 以及原始通信配置：当前安装驱动的 RGB-only / 禁用深度 publisher 组合未通过测试，分别出现 `bad_optional_access` 或进程退出，不使用这些组合。

加载前述 ROS 环境并设置相机环境后，在两个独立终端运行：

```bash
export ASTRIBOT_ROS_INSTALL_PATH=/opt/astribot_ros/software
export ASTRIBOT_CAMERA_INSTALL_PATH=/opt/astribot_ros/software
export ASTRIBOT_CAMERA_INFO_PATH=/etc/config/camera_info.json
/home/astribot/pressb_readonly_p2/camera_minimal/lib/astribot_camera/astribot_camera_main camera
# 第二终端使用相同环境，待采集初始化成功后启动压缩端：
/home/astribot/pressb_readonly_p2/camera_minimal/lib/astribot_camera/astribot_camera_sub compress_camera
```

启动之前确认不存在其他相机 main/sub 实例，以免争用设备或共享内存。本次 main 在 `gdb -batch -ex run -ex "thread apply all bt 5" --args ...` 下运行以捕获崩溃，实测报告保留这一运行条件；不将未复测的全机相机组合视为已经修复。

## 启动工作站模型与反向隧道

在工作站分别运行，保持终端/进程存活：

```bash
.venv-astribot-inference/bin/python scripts/s1b_policy_server.py \
  --config configs/s1b_deployment_v1.yaml --checkpoint 2000

ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=10 -o ServerAliveCountMax=3 \
  -R 127.0.0.1:8796:127.0.0.1:8796 S1B
```

服务与机器人端隧道均仅监听 `127.0.0.1:8796`。不要将已有同端口服务当作正确模型；机器人桥接器会核对 `/health` 中协议、step、维度、horizon、模型 hash 和数据契约 hash。模型切换需退出桥接器、换模型，再以新进程、新 session 启动。隧道重连不涉及执行启用。

## 部署最小采集目录

将以下文件复制到机器人独立目录，例如本次使用的 `/home/astribot/pressb_readonly_p2/`，保持相对布局：

```text
scripts/s1b_robot_bridge.py
src/pressb/s1b_deployment/__init__.py
src/pressb/s1b_deployment/contract.py
src/pressb/s1b_deployment/protocol.py
src/pressb/s1b_deployment/observation.py
configs/s1b_deployment_v1.yaml
configs/s1b_deployment_assets_v1.json
configs/s1b_observation_v1.json
```

无需复制模型权重或安装训练依赖。机器人已有 PyYAML、NumPy、OpenCV、rclpy、sensor_msgs 和 astribot_msgs。复制不会修改 SDK 或机器人系统目录。

在上述 ROS 环境下，从独立目录执行：

```bash
python3 scripts/s1b_robot_bridge.py --mode shadow --duration 610 \
  --output outputs/p2_final_610
```

输出目录必须不存在，以免覆盖前次证据。610 秒给启动发现留出余量；正式报告还应检查有效请求首末跨度至少 600 秒。配置中的最大有效请求间隔初值为 1 秒（与通信超时初值一致）；未达到持续时间或超过该间隔的运行不会标记 complete。

## 观测、拒绝与日志

映射固定在 `configs/s1b_observation_v1.json`：head_rgbd、right_wrist_rgbd 两路原始相机 JPEG topic，右臂/右夹爪/头部三个 `joint_space_states` topic。实际消息 `name=[]`，因此使用有版本和证据的部件内映射，检查长度 7/1/2；不使用无名全身状态索引。

相机 `CameraImage.image.data` 实测为空，不能直接当作 RGB。订阅既有 JPEG，用 `cv2.imdecode` 得到 BGR，仅在采集边界 `BGR2RGB` 一次，发送 head `(720,1280,3)`、right `(360,640,3)` 的 RGB uint8，不再次 JPEG 编码。state10 是 float32，夹爪保留连续 SDK closure（可略超端点），不二值化。

每个源消息保留完整 header、接收顺序、机器人 receipt monotonic 和 wall time；用一次校准的机器人 wall→monotonic offset 映射 epoch header。每个回调检测 offset 跳变和非法时间域；单流 header/receipt 不递增会明确拒绝，时钟跳变结束本次运行。原始 header 是驱动记录时间，未证明为物理曝光时刻。

使用最新 head 作为锚，在已收到的有限缓冲中按映射时间选最近的 right 和三个部件状态，不等待未来帧。工程初值为 state/head ≤10 ms、right/head ≤40 ms、head 接收至发送年龄 ≤80 ms；编码后再次检查年龄。值均在配置中，不视为运动许可。有效观测编号取 head 接收序号；模型最多一个 in-flight，未发送的中间 head 帧自然被最新帧替换，不形成网络请求队列。

每次输出包括：

- `sources.jsonl`：每条来源的 header、接收/映射时间、状态或图像尺寸/格式，以及拒绝原因。
- `requests.jsonl`：请求关联、三个部件各自时间、state10、原始预测、夹爪类别、RTT/服务耗时、缓冲统计；`sdk_target=null`。
- `events.jsonl`：握手、编码后过期拒绝、重连失败等事件。
- `resolved_config.json`、`mapping.json`、`summary.json`：配置身份与统计。
- `observation_*.npz`：约每 10 秒一份原始 RGB/状态请求，允许离线回放；图片预算 512 MiB，日志预算 512 MiB。图片超过预算记入 image_drops，日志写入失败会报错终止。

缓冲 overwritten 表示容量满后替换最旧历史样本，不等于传感器丢帧。JPEG topic 没有帧序号，丢帧只能按源 header 相邻间隔与名义 30 Hz 估算；报告必须保留“估算”，不将主动跳过的未发送 head 帧计为传感器丢帧。

采集节点会有 ROS 自带 `/rosout`、`/parameter_events` publisher 和参数服务；桥接器不创建应用/运动 publisher 或任何服务客户端。summary 中 `application_publishers_created=0` 指桥接代码显式创建的 publisher 数，不包括 ROS 内建发布项；以运行时 `ros2 node info` 证据为准。

## 分析与结束

将输出目录复制回工作站后运行：

```bash
.venv-astribot-inference/bin/python scripts/analyze_s1b_p2.py \
  --run outputs/s1b_deployment/p2_20261026/p2_final_610 \
  --output outputs/s1b_deployment/p2_20261026/analysis.json
```

分析包含各流帧间隔/记录时间代理年龄、源时间单调性、有效请求持续时间、RTT、原观测接收至响应、原 header 至响应、未来点数量、丢帧估算和产物哈希。未来点比例只是离线诊断，不能替代 P3 的调度与故障注入验收。

运行按 duration 自动结束。完成后结束本次推理服务、SSH 隧道以及为此次任务新启的相机进程；若相机原本关闭，通过相机专用服务恢复关闭。保留机器人独立目录与日志。不要停止其他机器人进程，不修改网口路由，不发送运动或占用控制权。
