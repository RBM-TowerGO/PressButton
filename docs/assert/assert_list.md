# Astribot 模型与数据资产列表

更新日期：2026-10-26。路径均相对项目根目录 `/home/js/robot/VLA/PressButton`。本表区分真机 joint10 与历史仿真 joint8，二者的接口及归一化不可混用。

## 本次真机微调模型

共同目录：[models/s1b_real_joint10_8000_v1](../../models/s1b_real_joint10_8000_v1/)。来源为 HPC 作业 `13333936`，运行目录 `/hpc2hdd/home/jluo628/robot/VLA/PressButton/runs/s1b_real_joint10_8000_v1`，训练代码提交 `e545d3d79062d7b7b2ca6a7a7da6844213a6a2bb`。

| 资产 | 本地权重 | 用途／状态 |
|---|---|---|
| 真机 joint10，第 2,000 步 | [step_002000/model.pt](../../models/s1b_real_joint10_8000_v1/checkpoints/step_002000/model.pt) | 固定选模指标下的最佳 checkpoint；已下载并通过 SHA-256 验收 |
| 真机 joint10，第 8,000 步 | [step_008000/model.pt](../../models/s1b_real_joint10_8000_v1/checkpoints/step_008000/model.pt) | 8,000 步训练完成时的最终 checkpoint；已下载并通过 SHA-256 验收 |

- 输入：头部 RGB、右腕 RGB，以及右臂 7＋夹爪 1＋头部 2 的实测状态。
- 输出：未来 7 步、每步 10 维绝对目标；右臂与头部单位 rad，夹爪动作 0/1（SDK 下发映射为 0/100）。夹爪状态保持连续反馈。
- Instruction：`Press the down button while maintaining the demonstrated head posture.`
- 监督时序：30 Hz 记录时间代理；存储 action 已对应下一网格目标，不再整体后移。仅支持本次固定示范头部目标，不代表主动转头或任意姿态恢复能力。
- 归一化使用本运行 `data_contract.json` 中的 `normalization`，不使用旧 joint8 或全数据默认统计量。

### 离线验证结果

使用 16 条验证轨迹、每条 3 个固定 anchor，共 48 个位置。手臂误差与夹爪准确率统计覆盖未来 7 步。选模分数使用第一步各维度归一化 MAE 的均值，不能解释为混合物理单位误差。

| 指标 | 第 2,000 步 | 第 8,000 步 |
|---|---:|---:|
| 归一化选模分数（越低越好） | **0.0825298** | 0.1027910 |
| 手臂平均误差（rad） | **0.0156068** | 0.0187911 |
| 夹爪二值准确率 | 93.15% | **97.02%** |

第 2,000 步的选模分数较优，第 8,000 步的夹爪准确率较高。两份均保留用于后续比较；离线结果不等于真机成功率。完整报告位于共同目录的 `validation_002000.json` 和 `validation_008000.json`。

### 配套文件与使用边界

共同目录保存：

- `processor/`：本次运行保存的 tokenizer、图像及视频处理配置。
- `config.yaml`：原始 HPC 配置，保留远端路径；`config.local.yaml`：本机路径版本，引用已有 `models/astribot_backbones/`。
- `data_contract.json`、`action_schema.json`、`training_contract.json`、`split.json`：训练数据身份、10 维语义、归一化及划分。
- `runtime/`：实际作业快照中的运行时代码，包含 `train_s1b_real.py` 和 `s1b_real.py`。
- `pretrained_loading.json`、`completed.json`、`train.jsonl`、全部验证报告及各 checkpoint 的原始 `manifest.json`：训练与权重来源记录。
- `download_verification.json`：本次本地权重完整性及 10 维接口形状核验结果。

**本次下载 `model.pt`，不下载 `training.pt`。** 原始 manifest 保留 `training_sha256` 供远端追溯，但不表示本地包含优化器、调度器和 RNG 续训状态。若要完整断点续训，需要另从 HPC 取得对应 `training.pt`。`best` 应指向 `step_002000`，`last` 应指向 `step_008000`。

本机路径配置和模型归档不等于完成真机推理部署验收；旧 joint8 推理入口不能直接用于这两个模型。

### 权重 SHA-256

下列期望值来自 HPC checkpoint manifest；本地实测结果由 `download_verification.json` 记录。

| 权重 | SHA-256 |
|---|---|
| step_002000/model.pt | `522b6425d3c71b43d1b41fb72e5468c9dbe5619828211501c3b12136e38ce99d` |
| step_008000/model.pt | `3c56bf3aeb692eebb74a5fbf2753b0d2fc30ebeed2efb6ad26ad7553cc885765` |

下载状态：**两份均已完成，SHA-256 与 HPC manifest 一致**（2026-10-26 11:48，Asia/Shanghai）。每份权重 10,268,130,722 字节，约 9.56 GiB；总计约 19.13 GiB。两份均经 `torch.load(weights_only=True, mmap=True)` 读取，确认状态输入／动作输入／动作输出对应维度为 10；尚未进行本机前向推理或真机执行测试。核验报告：[download_verification.json](../../models/s1b_real_joint10_8000_v1/download_verification.json)。

## 关联真机数据资产

| 资产 | 路径 | 说明 |
|---|---|---|
| 真机原始备份 | [data/astribot_s1b_20261025](../../data/astribot_s1b_20261025/) | 原始 HDF5／MCAP 与备份核验材料；主要原始 HDF5 集 102 条 |
| 审核后示范 | [data/astribot_reviewed_v1](../../data/astribot_reviewed_v1/) | 96 条 HDF5、23,039 帧，保留审核快照及原始／新编号映射 |
| 真机 joint10 LeRobot v3.0 | [data/astribot_s1b_real_joint10_v1](../../data/astribot_s1b_real_joint10_v1/) | 96 条、22,754 帧；train/validation 80/16 条，有效 7 步窗口 18,541/3,637 |

数据集身份 SHA-256（训练契约中的全文件清单指纹）：`8bfd18e4dab199216c113c4400d152c8377f1f3a0a65078e444ba4f8acbe3cf3`。

## 已有仿真模型与共享基础模型

以下为本机已有资产，本次仅登记路径，不表示进行了重新下载或哈希复验。

| 资产 | 路径 | 接口／用途 |
|---|---|---|
| 仿真 joint8，第 8,000 步 | [astribot_head_right_joint8_ft_20261018](../../models/astribot_head_right_joint8_ft_20261018/) | 右臂 7＋仿真夹爪角度 1；本次真机微调的迁移来源 |
| 仿真保守续训，第 17,000 步 | [astribot_head_right_joint8_conservative_20261018](../../models/astribot_head_right_joint8_conservative_20261018/) | 历史 joint8 对照模型 |
| Qwen3-VL-2B-Instruct | [astribot_backbones/Qwen3-VL-2B-Instruct](../../models/astribot_backbones/Qwen3-VL-2B-Instruct/) | 共享基础模型目录，本机配置依赖 |
| V-JEPA2 ViT-L | [astribot_backbones/vjepa2-vitl-fpc64-256](../../models/astribot_backbones/vjepa2-vitl-fpc64-256/) | 共享视觉编码器目录，本机配置依赖 |

相关说明：[真机训练入口](../../training/s1b_real/README.md)、[微调计划](../plan/astribot_s1b_real_finetune/README.md)、[数据转换与读取](../plan/astribot_s1b_real_finetune/lerobot_conversion.md)。
