# 左右腕相机安装核对（2026-10-14）

原七路视频中的左右腕相机使用项目自定义的安装位姿，与官方仿真定义不一致。
已修正为与当前 `astribot_whole_body_maniskill.urdf` 配套的官方 ManiSkill 相机外参。
头部相机及机器人运动参数不变；保留原七路录像供对照。

## 官方来源与适用范围

- [Astribot 官方 ManiSkill 示例](https://github.com/Astribot-Dev/astribot_simulation/blob/1403f9623efaea2709731c74629f395869849e89/src/astribot_envs/astribot_maniskill_env.py#L116-L137)：左右相机分别挂载于 `astribot_gripper_left_base`、`astribot_gripper_right_base`，局部平移同为 `[0, 0.04, 0.03]` m，SAPIEN 四元数 wxyz 同为 `[0.5, -0.5, -0.5, -0.5]`。
- [官方 URDF](https://github.com/Astribot-Dev/astribot_descriptions/blob/8c651a1abbe2821824a692aa3a3aa4cbf88e4122/urdf/astribot_s1_urdf/astribot_whole_body_maniskill.urdf)：夹爪基座到腕部 link_7 的固定关节平移为 `[0, -0.045, 0]` m，rpy 为 `[1.5708, 0, 0]` rad。
- [ManiSkill 坐标转换实现](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/utils/sapien_utils.py)：SAPIEN 相机使用 +X 前、+Y 左、+Z 上；USD 相机使用 -Z 前、+X 右、+Y 上。不能将两者四元数直接照搬。

核对时官方 simulation 与 descriptions 的 main 提交分别仍为上述固定版本。
这里复现的是公开**仿真外参**，不据此声称完成某台真机的相机标定。
本次仅修正外参；原有 640×480、焦距和裁剪距离保持不变。官方 ManiSkill 示例的腕部图像为 640×360、垂直视场 57°，因此本项目并非完整复现其成像内参。

官方还提供了 [MuJoCo 相机定义](https://github.com/Astribot-Dev/astribot_descriptions/blob/8c651a1abbe2821824a692aa3a3aa4cbf88e4122/mjcf/astribot_s1_mjcf/model_with_effector/astribot_s1_model_with_gripper.xml)。
它以位于 link_7 下 `[0, -0.15, 0]` m 的工具相机节点为父级；加上各自局部平移后，左右相机在 link_7 中的位置分别约为 `[0.011421, -0.091118, 0.044260]` 和 `[0.007720, -0.094080, 0.043190]` m。
这与 ManiSkill 的简化定义不是同一组数值。本次选择与当前 URDF 配套的 ManiSkill 定义，未混用不同引擎的参数。

## 错误与修正

将新旧配置都换算到各自 `astribot_arm_{left|right}_link_7` 坐标系：

| 配置 | 相机位置 XYZ（m） | 光轴方向 |
| --- | --- | --- |
| 原左腕 | `[0.100, 0.037588, -0.013681]` | `[0.8021, 0.4928, 0.3373]` |
| 原右腕 | `[-0.100, 0.037588, -0.013681]` | `[-0.8021, 0.4928, 0.3373]` |
| 官方 ManiSkill，经固定关节换算，两侧相同 | 约 `[0, -0.075, 0.040]` | 约 `[0, -1, 0]` |

两侧旧位置与官方定义都相差约 **159.87 mm**，光轴夹角约 **119.53°**。
相机原先朝向夹爪后侧和侧方，随手腕姿态变化出现天花板视角；不是视频左右标签交换。

转换到现有 USD LookAt 配置后，两侧在各自夹爪基座中采用：

```json
{"eye": [0, 0.04, 0.03], "target": [0, 0.04, 1.03], "up": [0, 1, 0]}
```

实际 USD 导入结果中，夹爪基座为腕部 link_7 下的固定子节点，并非机器人根下的平级刚体。
挂载解析现在按链接名称在机器人子树中唯一匹配，避免漏掉这段固定旋转和平移；缺失或重名都会报错。

## 验证与视频

- `tests/test_astribot_camera_mounts.py`：核对左右挂载、完整相机坐标轴、固定关节变换，以及嵌套链接解析和重名拒绝。
- `scripts/audit_astribot_camera_mounts.py`：读取实际导出的 USD，检查左右相机的父链接、本地及世界矩阵与官方外参是否一致。
- `scripts/compose_astribot_cameras.py`：检查每路视频的帧数、帧率、原生时间戳与重置后的连续性。

运行：

```bash
source scripts/activate_local.sh
python scripts/run_astribot_press.py --headless --video --all-cameras --gpu 0 \
  --config configs/astribot_vib820_ready.json --buttons 0 --repeats 1 \
  --output outputs/vib820_ready_official_wrist_20261014_v4
PYTHONPATH=.cache/usd-inspect:src python scripts/audit_astribot_camera_mounts.py \
  outputs/vib820_ready_official_wrist_20261014_v4
python scripts/audit_astribot_press.py outputs/vib820_ready_official_wrist_20261014_v4
python scripts/compose_astribot_cameras.py outputs/vib820_ready_official_wrist_20261014_v4
```

输出目录必须不存在。首次修正试跑因导入后的夹爪基座为嵌套路径而在相机创建阶段报错，未执行按压；记录保留在 `outputs/vib820_ready_official_wrist_20261014/`。第二次试跑被头部视觉网格的同名 Xform 触发歧义检查，也未执行按压；记录保留在 `_v2`。解析时排除 visuals/collisions 子树并通过真实 USD 资产测试后，`_v3` 的安装位置和物理按压检查通过，但严格视频检查发现 3 次旧帧缓存读取，未发布拼接视频。采集器增加原生时间戳就绪等待后，正式录像使用 `_v4` 目录。

正式运行验证（v4）：31 项相关测试通过；导出 USD 中两侧相机的父节点、局部平移、旋转和世界矩阵全部通过检查，局部和世界矩阵均在设定容差内。物理按压独立检查 23 项全部通过。右腕按压画面包含夹爪和按钮；左臂保持预备姿态，其相机朝向电梯门。

- [左右腕新旧视角对照](../../outputs/vib820_ready_official_wrist_20261014_v4/wrist_before_after.jpg)
- [修正后的七路视频](../../outputs/vib820_ready_official_wrist_20261014_v4/all_cameras.mp4)
- [单路视频与验证索引](../../outputs/vib820_ready_official_wrist_20261014_v4/WATCH.md)

正式 v4 每路 839 帧、30 FPS；所有相机原生时间戳与采样物理时间的最大差值为 0，跨相机时间差为 0。采集时如相机回调仍持有旧帧，会在同一物理时刻重渲染等待；没有插帧或重复写帧。
