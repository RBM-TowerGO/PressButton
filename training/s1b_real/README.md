# S1B 真机 joint10 训练

Qwen3.5-4B 使用独立的 [接线与 HPC 训练说明](../../docs/plan/qwen3_5_4b_migration/README.md)、
`requirements-qwen3_5.txt` 与 `submit_qwen3_5_hpc2.sh`。以下说明对应原 Qwen3-VL 训练配置。

## Qwen3.5 原图输入

两份 Qwen3.5 普通／混合数据配置现在均使用 `datasets.vla_data.image_mode: native`，
运行名含 `native`。Qwen 在训练、验证和部署时接收解码后的头部 1280×720、右腕 640×360 RGB 原图，
不额外缩放或补成正方形；Qwen 处理器仍执行必要的 patch 尺寸对齐。
`resolution_size: 224` 继续控制 JEPA 视频输入，视频像素、时间顺序及 mask 与旧路径一致。
原图解码缓存限制为每个 reader 32 帧；batch、梯度累积和学习率沿用当前配置。

未配置 `image_mode` 时默认 `resize`，保留旧输入与训练指纹。
部署从模型保存的训练配置读取模式，并与训练契约核对；不能只改旧模型的部署配置切换原图。
新模式进入训练指纹，旧 224 checkpoint 不能作为该原图运行的断点直接恢复。
`native` 与额外的 `image_size` 缩放配置冲突时会报错。

本地检查（不提交 HPC 作业、不控制机器人）：

```bash
PYTHONPATH=training/s1b_real/runtime:src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv-qwen3-5/bin/python -m pytest -q training/s1b_real/tests tests/test_s1b_deployment.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv-qwen3-5/bin/python \
  training/s1b_real/check_qwen3_5.py --output outputs/qwen35_native_images/wiring.json
```

测试覆盖真实帧原图保真、JEPA 不变、双视角多样本处理器一致性、旧配置兼容及小模型反向传播。
检查脚本使用合成原尺寸图像，报告视觉 token、训练前向损失、动作输出、显存和单次推理耗时；
它不包含完整训练优化器，不能用于估计训练峰值显存或证明闭环按压效果。

以仿真微调 step 8000 权重迁移到真实右臂7＋二值夹爪1＋头部2。保留兼容模型权重，明确重置五个坐标接口参数（状态第一层权重／偏置、动作第一层权重、动作输出层权重／偏置），不继承仿真 AdamW 状态。加载父运行保存的 processor。

运行时来源为 `models/astribot_head_right_joint8_ft_20261018/runtime/starVLA` 的现有部署快照；原文件保留，在 `dataloader/s1b_real.py` 和 `training/train_s1b_real.py` 新增真机接口。原有文件的版权头及许可证声明保持不变。数据接口独立于旧 Joint8 仿真 reader。

## 固定设置

- 数据：`data/astribot_s1b_real_joint10_v1`；80／16 条，18,541／3,637 个有效窗口。
- 8 张 A800，单卡 batch 2，梯度累积 4，总 batch 64；**8,000 个优化器步**（不是样本数、微批数或从 8000 接着计数）。不按 epoch 覆盖步数。
- 学习率：Qwen `5e-6`、动作模型 `1e-4`、世界模型 predictor `5e-5`；warmup 300，后续 cosine。冻结视觉教师和未参与损失的分支，训练 Qwen、动作模型和 predictor。
- 每 500 步评估，每个验证 episode 3 个有效 anchor，共 48；保留 best 与 last。选模依据为第一步 10 维**归一化** MAE 均值，分别报告手臂／头部弧度误差、夹爪二值准确率和逐 horizon 误差，不混合物理单位求均值。
- 夹爪动作 0/1 不标准化；state 连续；统计只来自训练集。头部常量显式处理。
- 只采样核验后的窗口，直接取已偏移的 `action[k:k+7]`，禁止再后移或跨 episode。最后有效 anchor 可能缺少第八张导出图像：只对辅助 video 尾帧 padding 并 mask 对应 tubelet，7 步动作不 padding；预测时只读取当前双视图和 state。

## 数据核验及测试

训练环境已有 torch、transformers、pyarrow、av 等依赖。无需把 LeRobot 安装到旧训练环境；用已验证 Parquet/MP4 元数据及 reader 接口读取 v3。`requirements-runtime.txt` 保存既有环境参考。

```bash
export PYTHONPATH=training/s1b_real/runtime
python training/s1b_real/prepare.py --dataset data/astribot_s1b_real_joint10_v1 --output training/s1b_real/dataset_contract.json
python -m pytest -q training/s1b_real/tests
```

prepare 对整个发布数据集含视频计算 SHA-256，已有契约不同时拒绝覆盖。GPU rank 校验非视频哈希和全部文件大小；提交前及作业内 prepare 校验完整视频哈希。

## HPC 提交与观察

在 `/hpc2hdd/home/jluo628/robot/VLA/PressButton`，核对 Git 提交及数据契约后：

```bash
mkdir -p logs outputs
/opt/slurm/bin/sbatch --parsable training/s1b_real/submit_hpc2.sh
/opt/slurm/bin/squeue -u jluo628
```

脚本先进行实际 GPU 两步训练、保存，再独立进程恢复到第三步，检查权重更新、48 个验证 anchor 的 7×10 输出、优化器恢复和有限 loss。通过后启动全新正式运行 `runs/s1b_real_joint10_8000_v1`，从零计到 8000；冒烟运行独立命名，不占正式步数。失败会终止，已有 run 不自动删除或覆盖。

每次作业快照保存到 `outputs/s1b_real_jobs/<jobid>/`；日志 `logs/s1b-real-<jobid>.log`。实际结果以 `smoke_acceptance.json`、正式 `train.jsonl` 和最终 `completed.json` 为准，提交成功不代表训练完成。预计资源使用参考旧实验约 3.6 秒／步，实际速度、排队和 checkpoint I/O 会影响总时长，24 小时为作业上限。

训练中断后的正式恢复命令（必须使用该作业快照的 config 和 runtime，并保持 8 卡）：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m starVLA.training.train_s1b_real --config "$SNAPSHOT/config.yaml" \
  --resume runs/s1b_real_joint10_8000_v1/checkpoints/last
```

恢复也需在 GPU 调度作业中运行，不在登录节点训练。执行端控制头部和在线延迟测量不属于此次离线训练任务。
