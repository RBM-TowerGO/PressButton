# 真机 LeRobot 数据转换与读取

转换脚本：[export_s1b_real_lerobot.py](../../../scripts/export_s1b_real_lerobot.py)。使用独立 `.conda/envs/lerobot` Python 3.12 环境，LeRobot 0.6.1、h5py 3.16.0；依赖记录在 `requirements-dataset.txt`。不向仿真环境安装这些依赖。

## 本次完成结果（2026-10-25）

正式目录：`data/astribot_s1b_real_joint10_v1/`。96 条、22,754 帧，约 368 MiB；训练／验证为 80／16 条，有效 7 步窗口分别为 18,541／3,637，总计 22,178。源文件哈希复核通过，未修改审核数据。

全部 22,754 行数值回读一致，576 次图像抽查通过；详细结果见 [转换验收摘要](conversion_result.json) 和数据目录的 `meta/reader_audit.json`。转换与审核相关测试共 9 项通过。

LeRobot 会将单值特征存为标量列，验收时恢复 `(N,1)` 比较；epoch 时间字段通过无格式转换的数据表读取，避免 NumPy formatter 默认 float32 导致时间精度丢失。视频无需为此重新编码。

转换完成不等于模型已适配：下一步是接入有效窗口与专用统计，完成 VLA-JEPA 的 10 维训练接口和冒烟测试。

## 转换命令

```bash
cd /home/js/robot/VLA/PressButton
.conda/envs/lerobot/bin/python scripts/export_s1b_real_lerobot.py
```

默认输入 `data/astribot_reviewed_v1`，默认输出 `data/astribot_s1b_real_joint10_v1`。可用 `--source`／`--output` 指定独立目录。目标或 `.inprogress` 目录存在时拒绝覆盖。先写临时目录，通过官方读取器验收后才原子改名发布。

如果全部视频和元数据已经写完、仅验收阶段中断，可用 `--finish-staging` 重新验证并发布；它不恢复半途编码，不跳过源哈希、数值及图像检查。编码中断的目录须保留后改用新输出目录。

## 数据格式与训练约束

- LeRobot v3.0：Parquet 数值、MP4 两路 RGB；head 1280×720、right 640×360。H.264、CRF 18、30 Hz；视频为有损编码，数值逐行精确校验。
- `observation.state`／`action` 均 10 维。夹爪状态保留连续 SDK 刻度，动作是 0/1，手臂和头部是 rad。
- `action[k]` 已是下一网格时刻目标，7 步动作取 `action[k:k+7]`，不要再额外向后移位。默认 LeRobot 单帧读取不会自动构造正确的训练窗口。
- `meta/training_anchors.json` 给出训练和验证的有效全局 anchor；必须从对应清单采样，不能直接遍历全数据作为 7 步样本，不能允许跨轨迹或尾部补齐。
- `meta/splits.json` 固定 80 条训练、16 条验证。按照审核后采集顺序每 8 条组成一个连续块，seed=42 选 2 块验证。缺少场景标签，因此该划分只提供时间分组，不代表独立场景测试集。
- **训练必须使用 `meta/training_normalization.json`**，仅由训练轨迹计算。官方默认 `meta/stats.json` 为全数据描述统计，不能用作本次训练归一化。夹爪动作 mean=0/std=1；常量头部有独立标记并处理零方差。
- `source_observation_row`、`source_action_row`、对应源时间、网格时间、原始夹爪 command 和 original id 保存数值来源；结合 `source_mapping.json` 回溯源文件与哈希。
- `meta/data_contract.json` 保存转换契约。时间是记录时间代理，仍不代表实际曝光或执行时间。

## 官方读取示例

在上述独立环境中运行：

```python
import json
from pathlib import Path
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = Path('data/astribot_s1b_real_joint10_v1').resolve()
ds = LeRobotDataset('local/astribot_s1b_real_joint10_v1', root=root, video_backend='pyav')
anchors = json.loads((root / 'meta/training_anchors.json').read_text())
i = anchors['train'][0]
observation = ds[i]
# 只访问数值表构造动作，避免重复解码未来帧图像。
actions = np.asarray(ds.hf_dataset.select(range(i, i + 7)).with_format(None)['action'], dtype=np.float32)
assert actions.shape == (7, 10)
print(observation['observation.state'].shape, actions.shape)
```

这仅为读取示例；完整训练 dataset adapter、模型 10 维接口及训练 checkpoint 仍是后续任务。

## 验收覆盖

- 源文件哈希；全量数值有限性与夹爪范围。
- 全部数值记录经官方读取器的数据表与重新生成的源数组逐项比较；episode、frame 与时间索引核查。
- 每条首／中／末帧的两路图像经官方读取器解码，对照对应源 JPEG 的尺寸、有限性及平均绝对像素误差（要求 ≤15／255）。该抽查不等于逐帧视频内容全检。
- 有效 7 步 anchor 不跨轨迹；训练／验证清单隔离；训练集归一化往返检查。
- 单元测试覆盖夹爪阈值／非法值、下一步目标、尾部、长间隔、时间异常和常量维度。

实际导出统计以数据目录的 `meta/reader_audit.json` 为准；版本与来源见 `meta/export_provenance.json`。
