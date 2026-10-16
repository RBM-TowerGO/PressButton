# 腰部 / 躯干相机官方定义检索

日期：2026-10-14。结论：**官方 `aos` 分支确有 `torso_rgbd` 相机声明；此前只核查 `main` 分支得到的“没有公开定义”不完整。但已找到的声明尚不能作为真实腰部相机标定直接使用。**

## 找到的官方声明

来源为 Astribot-Dev/astribot_descriptions 的 `aos` 分支，固定提交 `5ca7235afa19180f7ee5d87bfb2dd11392853143`：

- [torso_rgbd.xml](https://github.com/Astribot-Dev/astribot_descriptions/blob/5ca7235afa19180f7ee5d87bfb2dd11392853143/astribot_s1/model/mjcf/robot/sensors/torso_rgbd.xml)
- [kinematic.xml：挂载位置](https://github.com/Astribot-Dev/astribot_descriptions/blob/5ca7235afa19180f7ee5d87bfb2dd11392853143/astribot_s1/model/mjcf/robot/kinematic.xml#L440)

| 项目 | 文件中的定义 |
| --- | --- |
| 相机名称 | `torso_rgbd` |
| 父链接 | `astribot_torso_link_4` |
| 相机 body 相对父链接平移 | `[0,0,0.1]` m |
| body 旋转 | 未指定，使用默认单位旋转 |
| Camera 相对 body 的平移 | `[0,0,0]` m |
| Camera Euler 旋转 | `[0,0,0]` |
| 垂直视场 | `fovy=60`，即 60° |
| 该相机的分辨率、逐台内参、畸变、标定版本 | 该声明中没有提供 |

同一 `aos` 模型还包含 `head_stereo_left.xml` 和 `head_stereo_right.xml`。这些节点的存在不等于该分支已经提供了可用的完整传感器采集链或实机标定。

## 为什么不能直接称为真实腰部视角

1. 官方 `astribot_simulation/aos` 的 [S1 配置](https://github.com/Astribot-Dev/astribot_simulation/blob/c04acaf58b0b8155fc0b605731cda3b939617289/config/astribot_s1/sim.yaml#L244) 明确说明该版本不支持相机等传感器，`camera_names=[]`，仅提供控制相关数据。其 README 提到可选 `sensor_calib.json`，但所检查提交的文件树里没有该标定文件。
2. [MuJoCo 相机定义](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera) 的光轴为局部 -Z。按官方 MJCF 的四级躯干父变换计算，零关节姿态下 `torso_rgbd` 光轴约为 `[0,0,-1]`，即朝下，不是朝向双手前方。
3. 对应到本项目原始站位 `(-0.22,0.28,0.102)` 与零躯干关节，该声明的相机中心约为 `(-0.2200,0.2800,1.0920)` m。这里只计算声明的几何含义，没有改变相机朝向或声称该点位于真实相机镜头处。

基于以上事实，判断它可能是尚未完成的占位或未经校验的传感器定义；这是本次检查的推断，不是官方对该 XML 的直接说明。可以按字面值渲染朝下的诊断画面，但不能把这种录像描述为已复现真实胸部/腰部相机。

## SDK 与其他线索

- 官方 [astribot_msgs README](https://github.com/Astribot-Dev/astribot_msgs/blob/7f918075c7e79d3f5bb9aff927b78ad1f6ed85d0/README.md) 链接到 `Astribot-Dev/astribot_sdk/tree/ros1`。本次对仓库 API、`ros1/README.md` 与 `main/README.md` 的公开访问均返回 HTTP 404。只能确认当前无法公开读取，不能据此断言 SDK 不存在。
- 官方 [论文第 2.1 节](https://arxiv.org/html/2507.17141v1#S2.SS1) 确认胸部安装 Orbbec Gemini 335 RGB-D，相机工作频率 30 Hz；论文未提供其相对机器人链接的完整标定变换。
- 非官方实机客户端 [lerobot_grpc_inference](https://github.com/zz0320/lerobot_grpc_inference/blob/936879ab8079e43e58272b391f2c6a6b61a37f8b/README.md) 使用 `/astribot_camera/torso_rgbd/color_compress/compressed`，标注 1280×720。其源代码确实订阅该话题，但未提供相机安装变换；话题名和图像尺寸不能替代外参。

## 检查范围与后续录制状态

已检查官方三个公开仓库的 `main` 分支，以及 `astribot_simulation` 的 `aos/ros1`、`astribot_descriptions` 的 `aos`、`astribot_msgs` 的 `ros1/feat-update` 分支。共归档 273 个所选文本源文件（主分支 125，其他分支 148），另保存仓库树、版本、下载哈希及访问记录。

证据目录：`.cache/astribot_waist_camera_research_20261014/`；几何检查记录为其中的 `torso_geometry_check.json`。

用户要求先查 SDK 和官方仿真定义，因此未用估计参数增加相机，也没有启动新的录像。若要录制可代表实际安装的视角，还缺可访问的 SDK/部署相机配置、有效的挂载变换及成像内参；取得这些数据后再接入。本项目当前七路相机与规格 v1.1 保持不变。
