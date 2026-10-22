# Astribot S1 相机配置核对

核对日期：2026-10-14。**当前项目的七路录像不等于实机全部相机：其中只有三路机器人相机，另外四路是场景观察视角。头部 RGB 双目和胸部 RGB-D 尚未接入。**

同日补充：扩大检索至官方 `aos` 分支后，找到 `torso_rgbd.xml` 及头部双目节点声明；该分支明确不支持传感器采集，躯干相机的零旋转定义在直立姿态下朝向下方，尚不能作为实机标定使用。下文“未找到”限定为此前核查的 `main` 分支。详见[腰部 / 躯干相机定义检索](../vib820_experiment_spec/torso_camera_definition_20261014.md)。

## 官方硬件与当前实现

Astribot 团队论文《Towards Human-level Intelligence via Human-like Whole-Body Manipulation》
第 2.1 节和图 2 描述的 S1 配置如下。这是论文中的公开硬件版本，不能据此断言每台交付设备的具体配置和标定相同。

| 部位 | 官方论文中的硬件 | 当前项目 |
| --- | --- | --- |
| 头部双目 | Stereo RGB camera，双目彩色系统 | 缺失，没有独立左、右视点和双目基线 |
| 头部 RGB-D | Orbbec Femto Bolt | 用户选用其单路 RGB 视角；`head` 已按官方 ManiSkill 仿真示例设置位姿、1280×720 和视场，但不是逐台硬件标定 |
| 胸部 RGB-D | Orbbec Gemini 335 | 缺失，无对应相机节点和图像流 |
| 左腕 RGB-D | Intel RealSense D401 | 有 `wrist_left`，外参按官方 ManiSkill 示例修正，内参仍为项目配置 |
| 右腕 RGB-D | Intel RealSense D401 | 有 `wrist_right`，同上 |

官方论文还列出底盘 Livox MID-360 激光雷达，RGB-D 相机工作频率为 30 Hz。
若按对外彩色视点计，双目拆为左右两路，加上四台 RGB-D 的彩色视点，共六路；这不包括 RGB-D 内部用于深度测量的 IR 通道，也不包含外部观察相机。

来源：[官方论文第 2.1 节](https://arxiv.org/html/2507.17141v1#S2.SS1)、[官方图 2](https://arxiv.org/html/2507.17141v1/astribot_hardware.png)。

## 为什么公开仿真示例只有三路

核对的官方仿真仓库提交为 `1403f9623efaea2709731c74629f395869849e89`，模型仓库提交为 `8c651a1abbe2821824a692aa3a3aa4cbf88e4122`，核对时均为各自 main。

- [ManiSkill 配置](https://github.com/Astribot-Dev/astribot_simulation/blob/1403f9623efaea2709731c74629f395869849e89/config/astribot_s1/simulation_maniskill_param_chassis_fixed.yaml)：只启用 `head_rgbd`、`left_wrist_rgbd`、`right_wrist_rgbd`。
- [ManiSkill 相机定义](https://github.com/Astribot-Dev/astribot_simulation/blob/1403f9623efaea2709731c74629f395869849e89/src/astribot_envs/astribot_maniskill_env.py#L116-L148)：也是上述三路，不能用来证明实机只有三路相机。
- [MuJoCo 带相机配置](https://github.com/Astribot-Dev/astribot_simulation/blob/1403f9623efaea2709731c74629f395869849e89/config/astribot_s1/simulation_mujoco_param_with_camera.yaml)：上述三路另加 `astribot_global_camera` 外部视角。

已检查的公开配置、MJCF 和当前使用的 URDF 中，未找到可直接接入的胸部外参、独立双目左右外参或双目基线。论文图中的大致安装位置和视场示意不能替代这些参数。

## 当前场景的实际节点

检查对象：`outputs/vib820_ready_official_wrist_20261014_v4/scene.usda`。遍历其 `UsdGeom.Camera` 实体，确认共七个：

| 类型 | 录像名称 | 数量 |
| --- | --- | ---: |
| 机器人相机 | `head`、`wrist_left`、`wrist_right` | 3 |
| 场景观察相机 | `overview`、`panel`、`front`、`lobby` | 4 |

`configs/astribot_sensors.json` 中也只定义三路机器人相机。`AstribotSensors` 可以附加理想光轴深度，但七路按压录制入口只写 RGB 视频；这不构成真实双目成像、IR 或硬件噪声模型。

上游本地展开版 `astribot_s1.usda` 和项目生成的 `astribot_s1.usdc` 均未自带 `UsdGeom.Camera` 节点。上游 `.usd` 包装层另有无法解析的开发者绝对引用，所以不以该包装层的空遍历结果作结论。

核对原始记录位于 `.cache/astribot_camera_inventory/`，包括官方仓库树及版本、论文图和来源校验值、实际场景节点清单。当前配置未添加推测位姿的胸部或双目相机。

补齐到硬件视点覆盖需要：头部左右双目相对挂载链接的位姿和基线、胸部相机的挂载位姿，以及对应内参。若仅用于近似场景可视化，可另行采用有明确“估计值”标记的配置，不能称为官方标定复现。
