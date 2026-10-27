# Qwen3.5-4B 接线与 HPC 训练

2026-10-27；分支 `feat/qwen3-5-4b-migration`，起点 `986dd0e`。
正式训练在 HPC，本机负责代码接线和验证。

## 模型与接口

已下载模型位于 `models/astribot_backbones/Qwen3.5-4B`，下载记录绑定 revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。检查了全部文件大小、非权重文件的
SHA-256，以及两个 safetensors 分片中 738 个索引条目的存在性；本次未重新计算全部权重哈希。

| 项目 | 现有 Qwen3-VL-2B | 新 Qwen3.5-4B |
|---|---|---|
| 加载类 | `Qwen3VLForConditionalGeneration` | `Qwen3_5ForConditionalGeneration` |
| 文本隐藏维度 | 2048 | 2560 |
| 词表配置大小 | 151936 | 248320 |
| 文本层数 | 28 | 32 |
| Transformers | 4.57.0 | 单独环境固定 5.3.0 |

数值来自本地两份 `config.json`，可对照
[官方配置](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json)。
Qwen3.5 包含视觉编码器，使用多模态加载类，详见
[Transformers 接口文档](https://huggingface.co/docs/transformers/v5.3.0/model_doc/qwen3_5)。

新增 `model/modules/vlm/QWen3_5.py`，由 `qwenvl.model_type: qwen3_5` 显式选择。
输入继续使用双路图像、任务文本、24 个世界模型 action token 与 32 个 embodied token。
token ID 通过新 tokenizer 注册并查询；不复用 Qwen3-VL 的 ID 区间。
模板显式传 `enable_thinking=False`；这里使用一次前向的隐藏状态，不生成思维链。
仅取最后一层特征，不分配完整词表 logits，关闭 cache，保留训练梯度。

VLA-JEPA 已从 backbone 动态推导动作头 cross-attention 和 JEPA predictor 输入宽度；
采用原生 2560 维，输出仍为 `batch × 7 × 10`，数据坐标与归一化不变。
FAST 离散动作标签不在本次接线范围内，传入 `solutions` 会明确报错。

## 初始化与恢复

新配置的 `pretrained_initialization: qwen3_5_from_qwen3_vl_joint8` 表示：

1. 新运行从下载的 Qwen3.5 预训练权重初始化，添加当前任务所需 action token。
2. 从既有仿真 joint8 step 8000 checkpoint 加载兼容的 V-JEPA2、predictor 与动作头权重。
3. 重置原有五个 joint8→joint10 坐标接口参数，以及输入宽度发生变化的 DiT K/V 投影、
   predictor action/state/extrinsics 编码权重。未列入规则的键或尺寸变化直接失败。
4. 排除旧 checkpoint 的整个 Qwen 分支，不安装旧 processor，不继承其 AdamW 状态。

选择 joint8 父模型是为了沿用当前真机训练的既有初始化基线。**这不是从 joint10 的
8000 步继续训练**；若要复用已训练 joint10 的坐标接口，需要单独验证其数据契约并新增迁移模式，
本模式遇到 joint10 父 checkpoint 会拒绝。

报告写入 `pretrained_loading.json`：复用数量、重置参数、排除的旧 VLM 键、新 VLM 来源。
本机实际父 checkpoint 检查：复用 967 个张量，重置 24 个接口张量，跳过旧 VLM 的 626 个键，
保留新 VLM 的 724 个 state_dict 键（含 tied embedding 别名）。
保存配置保持 `init_from_config: true`：只有新迁移运行的构建阶段临时加载 Qwen3.5 预训练权重；
resume/inference 从配置构建后严格载入完整新 checkpoint。保存新的 processor。
分片设置与迁移模式都纳入训练指纹，恢复时不能静默改变。

## HPC 环境与配置

使用独立环境，不原地升级旧训练环境：

```bash
conda create -n vlajepa-qwen35 python=3.11 -y
conda activate vlajepa-qwen35
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r training/s1b_real/requirements-qwen3_5.txt
```

GPU 节点须支持选定的 CUDA 12.8 PyTorch wheel；本机验证不能代替 HPC 驱动兼容检查。
模型目录需完整同步到配置里的 HPC 路径，包含 tokenizer、模板、配置和两个权重分片。
V-JEPA2 路径与 joint8 父 checkpoint 路径沿用当前 HPC 布局；提交前核对。

- `configs/vlajepa_s1b_real_joint10_qwen3_5_hpc2.yaml`：真机 v1 数据。
- `configs/vlajepa_s1b_real_joint10_mixed_qwen3_5_hpc2.yaml`：mixed_v2 数据。
- 两者都为 8 卡、每卡 batch 1、累积 8，有效 batch 64，初始预算 8000 个优化器步。
- 启用 `optimizer_sharding: true`，使用 PyTorch `ZeroRedundancyOptimizer` 分摊 AdamW 状态。
  可训练参数仍为 FP32，算子 BF16 autocast，保留 gradient checkpointing。
- 保存前所有 rank 汇总优化器状态到 rank 0；恢复读取完整状态后分片。模型权重保持普通完整
  `state_dict`，推理不依赖分片优化器。

SDPA 只解决 attention 路径，不代表 DeltaNet 已使用快速内核。`causal-conv1d` 与
`flash-linear-attention` 可在 HPC 独立环境中按 torch/CUDA 版本构建并验证；未安装时使用
Transformers 的 PyTorch fallback。先用冒烟数据确定显存、速度，再决定是否安装优化内核，
变更后重跑冒烟并保留 `environment.txt`。

## 本机检查

```bash
export PYTHONPATH=training/s1b_real/runtime
python -m pytest -q training/s1b_real/tests
python training/s1b_real/check_qwen3_5.py
```

`check_qwen3_5.py` 默认使用本地已下载资产，验证新 VLM 未被父 checkpoint 覆盖、token 数量、
2560 维隐藏状态、有限的 `1×7×10` 输出，以及动作／世界模型两项 loss 的前向计算，
写入 `outputs/qwen3_5_wiring.json`。
输入是合成双视图与零状态，只验证接线，不代表新策略已经训练或具有按键能力。
单元测试包含缩小版混合注意力模型的多模态前向／反向，以及两个 CPU rank 的 AdamW 分片保存恢复。

2026-10-27 本机验证：RTX 5090、PyTorch 2.7.0+cu128、Transformers 5.3.0；实际 4B 模型
输出 `[1,223,2560]`，动作 `[1,7,10]`，两项合成 loss 有限，前向检查峰值分配显存约
10.97 GiB。此数值不包含完整训练反向、FP32 trainable weights 或 AdamW 状态。
新环境运行 `training/s1b_real/tests`：14 项通过；旧 Transformers 4.57 环境：12 项通过，
新 Qwen3.5 专用测试模块按模型支持情况跳过。环境明细保存于 `outputs/qwen3_5_environment.txt`。
Transformers 5.3.0 会对 `enable_thinking` 打印 processor 参数告警；检查渲染后的模板已确认
该参数被聊天模板采用，assistant 前缀包含闭合的空 `<think>…</think>`。

## HPC 冒烟和正式运行

```bash
mkdir -p logs outputs
# 默认只跑两步训练，再独立进程恢复到第三步并验收。
sbatch training/s1b_real/submit_qwen3_5_hpc2.sh

# mixed_v2：用环境变量选择独立配置。
S1B_QWEN35_CONFIG=configs/vlajepa_s1b_real_joint10_mixed_qwen3_5_hpc2.yaml \
  sbatch training/s1b_real/submit_qwen3_5_hpc2.sh

# 确认冒烟结果后：再次冒烟，通过后启动全新正式运行。
S1B_SMOKE_ONLY=0 sbatch training/s1b_real/submit_qwen3_5_hpc2.sh
```

脚本将代码、配置、数据契约和环境列表写入 `outputs/s1b_qwen35_jobs/<jobid>/`。
`S1B_QWEN35_ENV` 可指定独立环境路径，`S1B_QWEN35_CONFIG` 可指定已核对路径的配置。
冒烟 run 带 job ID，正式 run 独立命名；已存在的正式 run 不会被覆盖。
恢复正式训练须使用该作业快照的 config/runtime 与相同 world size：

```bash
export PYTHONPATH="$SNAPSHOT/code/runtime"
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m starVLA.training.train_s1b_real --config "$SNAPSHOT/config.yaml" \
  --resume "$PROJECT/runs/s1b_real_joint10_qwen3_5_4b_v1/checkpoints/last"
```

HPC 实际多卡训练、完整验证集评估、吞吐及显存测量仍由 GPU 作业验收。
训练完成后部署要使用新 run 保存的 processor、配置、checkpoint 和本分支 runtime，重新生成
部署资产锁；旧 `s1b_deployment_assets_*.json` 绑定旧模型与代码，不适用于新模型。
当前机器人运行资产不因本次接线自动切换。
