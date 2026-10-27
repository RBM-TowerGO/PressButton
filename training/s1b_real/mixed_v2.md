# S1B 两批审核数据混合训练 v2

用户于 2026-10-26 指定上传 `data/astribot_reviewed_2`，与第一批混合后重新训练 **8000 个优化器步**。

统一任务文本：

> Use the right gripper to press the circular button on the elevator panel.

## 数据

| 来源 | 审核轨迹 | 30 Hz 导出帧 | 有效七步窗口 |
|---|---:|---:|---:|
| `astribot_reviewed_v1` / elevator1 | 96 | 22754 | 22178 |
| `astribot_reviewed_2` / elevator2 | 116 | 23332 | 22622 |
| 合计 | 212 | 46086 | 44800 |

第一批原有 80/16 条训练/验证划分保持不变。第二批按原采集编号排序，每八条一个连续块，seed 42 留出三个块，共 92/24 条训练/验证；不随机拆帧。合计训练 172 条、37533 个有效窗口；验证 40 条、7267 个有效窗口。两批按有效窗口自然混合，不人为重复某一批。

`scripts/prepare_s1b_mixed_review.py` 校验审核导出的 SHA-256、结构检查结果、有效窗口和重复文件，生成 `data/astribot_reviewed_mixed_v2/`。其中 HDF5 是原审核导出的硬链接，只读用于转换；不要直接编辑这些文件。来源、原编号、裁剪区间与原文件哈希保留在映射中。

`scripts/export_s1b_real_lerobot.py` 用新任务文本转换为 `data/astribot_s1b_real_joint10_mixed_v2/`。沿用 30 Hz、未来七步、右臂七关节+二值夹爪+头部两关节的时间/数值语义；头部目标保留真实记录：第一批固定，第二批随手运动，不覆盖为常量。归一化统计只从混合训练划分重新计算。

转换后以官方读取器回读全部数值，并在每条首/中/末帧核对两路 RGB 图像，共 1272 次图像比较。`check_mixed_dataset.py` 再通过实际训练 reader 检查全体轨迹的首/中/末有效窗口、任务、目标对齐、图像、mask，以及 train-only 归一化。

## 训练

- 独立运行名：`s1b_real_joint10_mixed_8000_v2`。
- 初始化：与第一轮相同的仿真 joint8 step 8000 权重；迁移兼容权重，重新初始化五个状态/动作接口参数，新的 AdamW，不恢复旧真机模型优化器。
- 8 张 A800，单卡 batch 2，梯度累积 4，总 batch 64。
- 每 epoch 587 步，8000 步约 13.6 epoch；预算按优化器步数执行。
- 学习率、warmup 300、cosine、dropout 等保持既有配置。
- 每 500 步验证，每条验证轨迹取 12 个均匀分布的有效 anchor（不足则全部），通常 480 个；并不等于穷举全部验证窗口。
- 验证保留整体及 `by_source_batch` 指标，分别观察固定头部和跟随头部数据。选择标准仍为首步十维归一化 MAE；物理单位误差和状态保持基线另列。
- 保存 `best` 与 `last` 完整 checkpoint；未被这两个别名引用的旧 checkpoint 会按既有训练逻辑清理。

## HPC2 路径与作业

项目根：`/hpc2hdd/home/jluo628/robot/VLA/PressButton`。

- 第二批审核原件：`data/astribot_reviewed_2/`。
- 混合训练集：`data/astribot_s1b_real_joint10_mixed_v2/`。
- 新任务独立代码包：`training/s1b_mixed_v2/`，来自本地 `training/s1b_real/` 快照。
- 配置：`configs/vlajepa_s1b_real_joint10_mixed_hpc2.yaml`。
- 提交：`/opt/slurm/bin/sbatch --parsable training/s1b_mixed_v2/submit_mixed_hpc2.sh`。
- 日志：`logs/s1b-mixed-<jobid>.log`。
- 作业内不可变快照：`outputs/s1b_mixed_jobs/<jobid>/`。
- 正式输出：`runs/s1b_real_joint10_mixed_8000_v2/`。

作业先核对整个数据集的哈希并运行检查，再执行独立两步 GPU 冒烟、恢复到第三步、验收权重更新和优化器恢复。通过后才自动启动从零计数的正式 8000 步任务。排队/提交成功不等于已经完成 GPU 验证或开始正式训练。

本地运行证据保存在 `outputs/s1b_mixed_20261026/`。本次不更改旧模型的部署 instruction 或机器人控制设置；部署新模型时须使用新数据契约、归一化和上述任务文本。
