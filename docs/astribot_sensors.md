# Astribot 仿真传感器

在现有 Astribot + 按钮场景中接入机器人传感器。左右腕相机外参已按官方 ManiSkill
示例修正，见[安装位置核对](astribot_wrist_camera_audit.md)。头部采用官方 `head_rgbd`
的单路 RGB 视角；其他传感器仍采用项目仿真配置。公开仿真定义不等同于特定真机的硬件标定。

**覆盖范围：当前仅接入头部和双腕三路机器人相机，尚缺官方论文中的头部 RGB 双目和胸部 RGB-D。**
此前七路视频包含额外四路场景视角，不能代表实机全部相机。见[完整配置核对](astribot_camera_inventory.md)。

## 启动和查看

```bash
cd /home/js/robot/VLA/PressB
source scripts/activate_local.sh

# 机器人、按钮面板以及全部传感器，保持窗口打开。
python scripts/preview_astribot_panel.py --sensors --lidar-points

# 从头部相机观察按钮；可换成 wrist_left / wrist_right。
python scripts/preview_astribot_panel.py --view head

# 无窗口运行完整传感器验收；输出目录必须不存在。
python scripts/verify_astribot_sensors.py --headless \
  --output outputs/astribot_sensor_validation
python scripts/plot_astribot_sensors.py outputs/astribot_sensor_validation
```

GUI 的相机选择菜单也可切换到机器人链接下的 `RGBD` 相机。`--lidar-points` 显示雷达
回波点；传感器本身没有额外的外壳网格。预览入口会将当前真实读数保存到当次输出的
`sensors/`，包含三路 RGB、米制深度、雷达点云、内外参、时间戳、IMU、接触和关节数据。
不加 `--sensors` 时保留原来的轻量预览。自动按压 RGB 录制见下方 `--all-cameras` 入口。

## 接入范围

| 数据 | 数量 / 频率 | 输出与挂载 |
| --- | --- | --- |
| 相机 | 3 路，默认 10 Hz；头部 1280×720，双腕 640×480 | 头部单路 RGB 视角、左腕、右腕；完整传感器接口可附加理想光轴深度 |
| 激光雷达 | 1 路，360×16 束，记录 10 Hz | 底座前侧；360° 水平、30° 垂直，0.05–20 m |
| IMU | 5 路，120 Hz | 底座、躯干、头部、双腕；加速度、角速度、姿态 |
| 关节状态 | 32 自由度，120 Hz | 名称、位置、速度、实测关节力矩 |
| 关节反力 | 33×6，120 Hz | 各链接入关节的三轴力和三轴力矩 |
| 接触 | 17 路，120 Hz | 已有碰撞几何的刚体；合力、接触对象、位置、法线、冲量 |

雷达采用 Isaac Sim 5.0 内置 [PhysX raycast lidar](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/sensors/isaacsim_sensors_physx_lidar.html)，
每次生成完整扫描，`rotation_rate=0`；它基于碰撞几何，不模拟材质透射、反射率、旋转畸变或硬件噪声。
点云中的未命中项必须用 `hit` 掩码剔除。配置见
[`configs/astribot_sensors.json`](../configs/astribot_sensors.json)。

上游 16 个链接没有碰撞几何，未人为补造触觉数据；具体列表在 `calibration.coverage`。
17 路接触接口已产生有效读数，验证中对底座执行了真实落块碰撞；没有逐一碰撞所有部位。
关节反力含重力与运动载荷，不能当作已做重力补偿的腕部六维力传感器。
当前接入层是本地 Python / NumPy 与文件输出，尚未发布 ROS/ROS 2 话题。

## 验收证据（2026-10-13）

正式传感器验收：[`outputs/astribot_sensors_v4/validation.json`](../outputs/astribot_sensors_v4/validation.json)，
进程退出码 0，29 项检查全部通过，540 个显式物理步，包含一次重置。

- [三路 RGB 与深度总览](../outputs/astribot_sensors_v4/camera_overview.png)
- [雷达点云](../outputs/astribot_sensors_v4/lidar_points.png)
- [IMU、接触力与关节响应](../outputs/astribot_sensors_v4/sensor_response.png)

三路相机有效深度比例为 97.9%–100%，中心深度与独立 PhysX 射线测量相差约 5 mm
（可视面板与碰撞面存在小偏移）。雷达在已知目标处测得 1.000000 m；目标移远 0.3 m
后测得 1.300000 m。头部和双腕各转动约 0.15 rad，相机外参、画面及陀螺仪读数相应变化。
五路静止 IMU 测得重力模长约 9.81 m/s²；落块触发底座接触，峰值约 18.63 N。

重置前保存 420 帧物理状态、33 帧相机时间戳和完整雷达扫描。
物理/IMU 间隔实测约 0.008333 s，相机间隔约 0.100000 s，相机时间与采集时物理时间无测得差值。
重置后再验证图像、深度、雷达、IMU 和归零关节恢复。较早 v1–v3 为调试记录，以 v4 为准。

按钮入口联调也已通过：`outputs/astribot_panel_sensors_v1/` 保存同场景的预览、传感器快照
及带全部传感器定义的 `scene.usda`。重新打开场景后确认 3 个机器人相机、1 个雷达、
5 个 IMU、17 个接触传感器和 12 个按钮关节均存在。独立数组与时间检查见
[`independent_audit.json`](../outputs/astribot_sensors_v4/independent_audit.json)；
原场景和相机相关的 35 项回归测试通过。当前运行环境无桌面显示，GUI 操作需在你的桌面会话中查看；
本次实际验证使用 headless 渲染。

`initial/moved/reset_*` 保存静止、运动后、重置后的 RGB、深度及点云。
`timeseries.npz` 保存连续关节、反力、IMU、接触力、雷达与各路时间戳。
深度可视化 PNG 仅用于查看，定量计算使用 NPZ 内的浮点米制数组。

## Python 接口与坐标

### 头部单目视角（2026-10-14）

按用户选择，`head` 使用头部 RGB-D 相机的单个彩色视点，对应官方 ManiSkill
`head_rgbd`；没有拆分或合成双目左右画面。录制入口只写入单路 `head.mp4`。

- 挂载：`astribot_head_link_2`。
- 局部位置：`[0.0603, -0.2139, -0.0277]` m。
- 官方 SAPIEN 姿态 wxyz：`[0.7071, 0.7071, 0, 0]`，已转换为 USD 相机轴。
- 分辨率 1280×720，垂直视场 `0.8974046681441843` rad（约 51.42°），近/远裁剪面取官方示例。
- 30 FPS 按压录像；完整传感器接口仍可为该视点附加理想深度，不是双目重建。

来源：[官方单路头部定义](https://github.com/Astribot-Dev/astribot_simulation/blob/1403f9623efaea2709731c74629f395869849e89/src/astribot_envs/astribot_maniskill_env.py#L138-L147)。
这是官方仿真针孔视角的复现，不是实机 Femto Bolt 的逐台标定或硬件深度噪声复现。
每路相机可单独指定分辨率、垂直视场和裁剪面，双腕继续使用已有配置。

验证时使用 `scripts/audit_astribot_camera_mounts.py --include-head RECORDING` 检查实际导出的头部
局部/世界位姿、图像分辨率、视场与裁剪面。新版输出在 `outputs/vib820_head_mono_20261014/`。

本次完整录制为 839 帧、30 FPS，约 27.97 秒；实际头部与双腕安装位姿检查、
23 项物理动作检查通过。[头部视频](../outputs/vib820_head_mono_20261014/head.mp4)、
[七路同屏视频](../outputs/vib820_head_mono_20261014/all_cameras.mp4)及
[录像索引与验证报告](../outputs/vib820_head_mono_20261014/WATCH.md)。
本任务头部关节保持固定，接触点位于头部画面下方之外；可在腕部与面板视角查看按压接触。

### 按按钮全过程的同步视频

在按压入口增加 `--all-cameras`，可同时录制机器人头部、左腕、右腕，
以及场景全景、面板特写；VIB-820 场景还包括面板正面和电梯厅，共七路。
机器人相机复用 `configs/astribot_sensors.json` 的安装位置和光学参数，
这是仿真相机配置，不代表实机外参标定。此选项录制 RGB。

```bash
source scripts/activate_local.sh
python scripts/run_astribot_press.py --headless --video --all-cameras --gpu 0 \
  --config configs/astribot_vib820_ready.json --buttons 0 --repeats 1 \
  --output outputs/vib820_ready_all_cameras_20261014
python scripts/audit_astribot_press.py outputs/vib820_ready_all_cameras_20261014
python scripts/compose_astribot_cameras.py outputs/vib820_ready_all_cameras_20261014
```

录制输出目录必须不存在。每路 MP4 和关键阶段 PNG 单独保存；
`all_cameras.mp4` 为带中文视角名称的同步拼接视频，按仿真时间原速播放。
视频包含预备姿态、夹爪闭合、接近、按压、撤回、返回预备姿态及末尾的 reset 检查。
`camera_timing.json` 保存每帧的物理时间、reset epoch 和每路相机原生时间戳。
合成脚本先检查帧数、帧率、时间戳连续性及跨相机同步，再检查合成文件；
结果写入 `camera_video_audit.json`，重复合成时不会覆盖已有视频。

2026-10-14 七路演示：[同屏视频](../outputs/vib820_ready_all_cameras_20261014/all_cameras.mp4)
与[单路视频索引](../outputs/vib820_ready_all_cameras_20261014/WATCH.md)。
本次每路 839 帧、30 FPS，时长约 27.97 秒；按压触发在视频约 19.03 秒。
这份历史录像中的右腕相机在按压姿态下朝向天花板，保留其实际视野供对照。
后续已按[官方腕部安装定义](astribot_wrist_camera_audit.md)修正配置，
[新版七路视频](../outputs/vib820_ready_official_wrist_20261014_v4/all_cameras.mp4)
和[左右腕对照图](../outputs/vib820_ready_official_wrist_20261014_v4/wrist_before_after.jpg)可用于核对。

### 完整传感器接口

实现为 [`AstribotSensors`](../src/pressb/astribot_sensors.py)，必须在 `SimulationApp` 启动后导入：

```python
rig = AstribotSensors(world, robot, adapter_metadata)
world.reset()
rig.initialize()  # 每次 reset 后重新调用，epoch 自增
for _ in range(12):
    world.render()  # 相机渲染预热，不推进物理
for step in range(120):
    world.step(render=False)  # 1/120 秒
    physics = rig.physics_frame()
    if (step + 1) % 12 == 0:
        world.render()
        observation = rig.capture()
rig.save_snapshot(new_output_directory)
```

`world` 的 `physics_dt=1/120`、`rendering_dt=1/10`。不能将每 12 步一次的
`world.step(render=True)` 当作仅渲染：这个接口会额外推进多个物理子步，导致采样频率失真。
预览与验收入口已改为显式物理步后调用 `world.render()`。

相机局部轴为 USD 的 +X 向右、+Y 向上、-Z 向前；深度是正的光轴 Z 距离。
`world_from_sensor` 是采用列向量的 4×4 变换矩阵，`p_world = T @ p_local`。
雷达点云在雷达局部 XYZ 下；IMU 加速度和角速度为传感器局部坐标，姿态四元数为世界姿态 wxyz。
内参来自实际相机 API；每个相机保留原始 `rendering_time` 与 Fabric 有理数参考时间，
物理读数保留自己的时间，重置由 `epoch` 区分。数据不假装是真机或 ROS 时间。
