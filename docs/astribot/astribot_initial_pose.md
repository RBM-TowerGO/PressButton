# Astribot S1 双臂预备姿态检索

核对日期：2026-10-13。结论：找到符合“双臂抬起、屈肘待命”描述的公开 **Home state** 定义，来自 ActiveArena 的 Astribot S1 适配。它是第三方项目明确配置的初始关节姿态，不是已确认的厂家标准动作，也不是一段现成的抬臂动画。

## 来源与定义

| 来源 | 查到的定义 |
| --- | --- |
| 原 PressB 默认任务 | `run_astribot_press.py` 未配置初始手臂角度时，对全部关节设置零位；随后右臂单独移向任务工作位姿，左臂保持零位 |
| Astribot 官方 IsaacLab 示例 | `InitialStateCfg.joint_pos={".*": 0.0}` |
| Astribot 官方 ManiSkill 示例 | `standing` keyframe 的 qpos 为全零 |
| 已下载的官方 USD | 32 个关节驱动 targetPosition 为零；未查到带时间采样的抬臂动画 |
| 官方 MJCF XML | 所检索的 30 个 XML 中未查到非零 qpos keyframe；MuJoCo 的 reset_from_keyframe 实际恢复启动状态副本 |
| ActiveArena Astribot 资产 | `config.yml` 的 `homestate` 明确配置左右臂非零关节角，`robot.py::move_to_homestate()` 设置驱动目标并同步实际关节状态 |

官方仿真仓库核对提交：`1403f9623efaea2709731c74629f395869849e89`；查询时 main 与此提交相同。官方描述资产提交：`8c651a1abbe2821824a692aa3a3aa4cbf88e4122`。未找到公开定义不代表真机 SDK 或厂家内部动作库不存在。

- [官方 IsaacLab 初始配置](https://github.com/Astribot-Dev/astribot_simulation/blob/1403f9623efaea2709731c74629f395869849e89/src/astribot_envs/astribot_isaaclab_env.py#L258)
- [官方 ManiSkill standing 定义](https://github.com/Astribot-Dev/astribot_simulation/blob/1403f9623efaea2709731c74629f395869849e89/src/astribot_envs/astribot_maniskill_env.py#L41)
- [ActiveArena 论文附录 B.1、表 10](https://arxiv.org/html/2609.24124v2#A2.SS1)
- [ActiveArena move_to_homestate](https://github.com/leeibo/ActiveArena/blob/1bf2c77b8d3ab978fc5b4599a69d017f593edbe8/envs/robot/robot.py#L900)
- [ActiveArena 资产下载定义](https://github.com/leeibo/ActiveArena/blob/1bf2c77b8d3ab978fc5b4599a69d017f593edbe8/script/download_activearena_assets.py#L17)

## 双臂角度

按各自 `astribot_arm_{left|right}_joint_1` 到 `joint_7` 顺序，单位为 **rad**：

| 手臂 | J1 | J2 | J3 | J4 | J5 | J6 | J7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 左臂 | -0.4 | -0.7 | -1.0 | 1.4 | -0.9 | 0 | 0 |
| 右臂 | 0.4 | -0.7 | 1.0 | 1.4 | 0.9 | 0 | 0 |

原始资产 `astribot_descriptions_texture/config.yml` 第 36 行起给出关节名顺序，第 101 行为生效的 `homestate`；第 102–103 行是注释掉的其他候选值，不使用。

同一资产还定义 `head_homestate=[0.0,1.0]`、`torso_homestate=[0.0]`，并使用自己的夹爪映射、机器人世界变换、驱动参数和碰撞过滤。本文对照预览只迁移两条手臂的角度，头、躯干和夹爪保持当前零位，以明确展示手臂姿态的差异。

## 与当前资产的核对

已下载 ActiveArena 对应压缩包，校验 SHA-256 与项目下载器一致：

```text
dataset: leeibo/ActiveArena-Assets
revision: 819632dfd545b569657e195cc47aae3af7fce5f1
archive: astribot_descriptions_texture_20260630_174911.zip
sha256: 49be39c500e763d6d0194d4449751967bff16df5a2c044f1f53f7e9b7adffcb7
```

解出的配置、URDF、CuRobo 双臂配置及碰撞配置位于 `.cache/astribot_pose_research/ActiveArenaAssets/astribot_descriptions_texture/`。完整来源清单、官方源代码和 XML 检索副本位于 `.cache/astribot_pose_research/`；没有执行或安装下载项目的程序。

与当前 `astribot_whole_body_maniskill.urdf` 对比，14 个手臂关节的名称、origin 和 axis 一致，候选角度全部位于当前 URDF 限位内。两份资产的关节限位、力矩和速度参数存在差异，因此只取关节角度作为候选，不能整体覆盖当前动力学和碰撞模型。

按当前 URDF 正向运动学计算，腕部 link_7 在 `astribot_torso_base` 坐标系的位置为：

| 姿态 | 左腕 XYZ（m） | 右腕 XYZ（m） |
| --- | --- | --- |
| 全零位 | (0, 0.3692, 0.7455) | (0, -0.3692, 0.7455) |
| ActiveArena 双臂 Home | (0.3694, 0.3588, 1.1323) | (0.3694, -0.3588, 1.1322) |

可见双腕抬高约 38.7 cm，并向前移动约 36.9 cm。这是当前模型的运动学计算，不是实测真机姿态或碰撞验收。

## 预览与接入含义

静态预览输出目录：`outputs/astribot_initial_pose_research/`。预览直接设置命名关节状态，用 Isaac Sim 渲染外形；没有执行从零位抬臂的运动，也没有进行按压或碰撞安全验证。

- [双臂抬起预览](../../outputs/astribot_initial_pose_research/activearena_arm_home_oblique.png)、[正面预览](../../outputs/astribot_initial_pose_research/activearena_arm_home_front.png)
- [原零位对照](../../outputs/astribot_initial_pose_research/zero_oblique.png)
- [按关节名保存的候选角度与来源](../../outputs/astribot_initial_pose_research/candidate_pose.json)
- [设置值与实际关节读回值](../../outputs/astribot_initial_pose_research/report.json)

预览已完成，14 个候选角度通过当前 URDF 限位核对，两种姿态的实际关节读回与设置值一致。此处“通过”仅指角度映射与静态显示核对。

当前 [随机站位结果](../plan/vib820_random_base_validation/results.md) 的 5 次失败都发生在从零位移向右臂工作位姿阶段。使用双臂预备姿态作为回合起点值得单独验证，但不能据此断言已经解决这 5 次碰撞。

接入需要统一处理的事项（下节记录单次演示的实施结果）：

1. 按关节名设置双臂默认状态、实际位置、零速度及驱动目标；初始化与 reset 使用同一组值。
2. 让左臂保持其非零预备角度，右臂由预备姿态规划接近、按压、撤回。明确“预备姿态”和面板前“工作位姿”的区别。
3. 当前 runner 与 audit 对非活动关节使用“接近零”的条件；需改为“接近该回合预备目标”，保留误差门槛。
4. 在电梯场景内先验证预备姿态本身的重叠、保持稳定性及右臂路径；保留原碰撞规则，不能套用 ActiveArena 的 head-vs-arms 屏蔽设置。
5. 若任务定义是从已经抬臂的状态开始，单独记录这个初始条件；若要模拟从下垂到抬臂，则抬臂路径也必须接受碰撞验证。
6. 建立新批次重放原 30 个站位，保持历史零位起点批次及统计不变。

以上为资产/定义检索、限位与运动学核对及静态预览记录；原随机站位批次仍采用零位起点。

## 双臂预备姿态单次按压

新增配置 `configs/astribot_vib820_ready.json`，通过 `initial_arm_positions` 按名称指定上述 14 个角度，并启用 `return_to_initial_pose`。runner 设置实际状态、默认状态、零速度和驱动目标；左臂、头与躯干的保持条件改为相对于初始目标检查，误差门槛仍为 0.02 rad。独立 audit 同步检查初始角度、非活动关节保持、右臂返回和 reset 后姿态。

回合从已经抬起双臂的状态开始：保持 1.5 s → 闭合右夹爪 → 移向工作位姿 → 接近并物理按压 → 撤回工作位姿 → 返回双臂预备姿态 → 保持 → reset 检查。左臂全程保持预备角度；右夹爪返回时保持闭合。此演示不包含从双臂下垂到抬起的运动。

首次试跑 `outputs/vib820_ready_press_video_v1/` 沿用底盘 X=-0.12 m、Y=0.28 m，静态初始重叠检查通过，但在闭合右夹爪的第 307 步接触面板表面及右边框，触发碰撞停止，未进入按压阶段。接触力分别约 0.303 N 和 0.375 N；完整失败轨迹、视频与原配置均已保留。静态不重叠不能替代闭合路径检查。

因此演示配置将底盘后移 100 mm 至 X=-0.22 m、Y=0.28 m，保持双臂角度、环境、驱动、按钮物理和碰撞规则不变。这是新演示站位，不是对原 30 个随机站位的重放或重新统计。

运行命令（输出目录须不存在）：

```bash
source scripts/activate_local.sh
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_astribot_press.py \
  --headless --video --gpu 1 --config configs/astribot_vib820_ready.json \
  --buttons 0 --repeats 1 --output outputs/vib820_ready_press_video_v2
python scripts/audit_astribot_press.py outputs/vib820_ready_press_video_v2
```

相关运动学/按压及审计单元测试 22 项通过；旧零位起点演示 `outputs/vib820_press_v5_video/` 经更新后的 audit 复核仍为 19/19 项通过。
