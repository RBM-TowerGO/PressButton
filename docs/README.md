# 文档索引

当前主线为 Astribot S1 / VIB-820。交接入口为 [HANDOVER.md](HANDOVER.md)；仓库中原文件名为 HANDOVER.md，沿用该名称。

建议先读交接文档，再按任务查看场景、评估或强化学习说明。带日期的报告记录当时的状态，最新运行状态应结合实际产物确认。

## Astribot S1 场景、资产与传感器

- [Astribot S1 资产加载与适配](astribot/astribot_asset.md)
- [Astribot S1 相机配置核对](astribot/astribot_camera_inventory.md)
- [Astribot S1 双臂预备姿态检索](astribot/astribot_initial_pose.md)
- [S1 右臂闭合夹爪按键](astribot/astribot_press.md)
- [Astribot 仿真传感器](astribot/astribot_sensors.md)
- [左右腕相机安装核对（2026-10-14）](astribot/astribot_wrist_camera_audit.md)
- [VIB-820 单按钮外呼面板与 S1 按压](astribot/vib820.md)

## Astribot S1 闭环评估

- [Astribot VLA 闭环评估（2026-10-18）](astribot_eval/astribot_closedloop_20261018.md)
- [Astribot 8,000／17,000 步闭环配对对照（2026-10-19）](astribot_eval/astribot_paired_val_20261019.md)

## Astribot S1 在线残差强化学习

- [Astribot S1 在线残差强化学习（VIB-820 单按钮）](astribot_rl/astribot_online_rl.md)
- [Astribot S1 第二轮稳定化训练（本机双 RTX 5090）](astribot_rl/astribot_round2_20261020.md)
- [Astribot S1 第二轮稳定化：独立种子 20261022](astribot_rl/astribot_round2_20261020_seed20261022.md)
- [Astribot 第三轮：本机单 RTX 5090](astribot_rl/astribot_round3_local_20261020.md)

## 历史 PiPER 主线

- [Piper 电梯按键：10 Hz 历史批次与 LeRobot 读取](piper/dataset.md)
- [边缘灯带场景：30 Hz LeRobot 数据集](piper/dataset_30hz.md)
- [面板左右、前后随机偏移](piper/dataset_panel_randomization.md)
- [面板位置分层采集：12 × 100 条](piper/dataset_panel_stratified.md)
- [30 Hz：初始姿态到首次成功按键的数据集](piper/dataset_press_only.md)
- [VLA-JEPA 在线残差强化学习](piper/online_rl.md)
- [在线 RL 高吞吐采样](piper/online_rl_fast.md)
- [PressB online RL wire contract (v1)](piper/online_rl_protocol.md)
- [Live W&B monitoring](piper/online_rl_wandb.md)
- [PiPER VLA-JEPA 仿真闭环评估](piper/policy_eval.md)
- [Eval 执行平滑](piper/policy_smoothing.md)
- [从 LeRobot 末端 action 回放实际按键](piper/replay.md)
- [固定初始噪声策略，重新训练动作残差](piper/residual_on_frozen_noise.md)
- [场景结构、相机与历史验证记录](piper/scene_details.md)

## 部署与环境

- [Pro6000 环境配置记录](setup/PRO6000_ENV.md)
- [本机部署](setup/local_deployment.md)
- [RTX 5090 工作站运行说明](setup/rtx5090_setup.md)

## 研究与可行性评估

- [VLaRL 在 PressButton 的可行性评估](research/vlarl_feasibility_20261022.md)

## 原有资料目录

以下目录保持原有内容与结构：

- [plan/](plan/)：采集与验证计划。
- [vib820_experiment_spec/](vib820_experiment_spec/)：实验规格与相机检查。
- [results/](results/)：结果汇总。
- [media/](media/)：图片与视频。
