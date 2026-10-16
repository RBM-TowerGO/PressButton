# VIB-820 采集与验收操作

所有命令从 `/home/js/robot/VLA/PressB` 执行。预检与生产输出严格分开；没有通过生产 gate 时不能启动正式采集。当前进度见 [progress.md](progress.md)。

## 环境与配置

```bash
source scripts/activate_local.sh
export CUDA_VISIBLE_DEVICES=0,1 OMNICLIENT_HUB_MODE=disabled
export TMPDIR="$(mktemp -d "$PWD/.cache/sim_tmp/a4_XXXXXX")"
```

仿真使用 `.conda/envs/pressb`；导出使用 `.conda/envs/lerobot`（LeRobot 0.6.1）。每进程显式传物理 GPU 编号 0 或 1，不能另行缩减可见设备后仍传原编号。CPU 路径通过物理对照；GPU Dynamics 关闭 suppressReadback 后也已通过单样本完整审计，实际为 GPU 求解 + CPU 状态/接触回读。最终后端须根据同任务预检选择，不能仅凭单样本就用于生产。

随机候选已冻结于 `outputs/vib820_gpu_collection_400/manifest.json`。不要重复执行 `--freeze-only` 覆盖该文件。新的独立目录才使用：

```bash
python scripts/collect_astribot_dataset.py --output NEW_RAW_DIRECTORY --freeze-only
```

## 预检

作业 JSON 是有序列表，历史对照格式为 `{"id":"wave0_reference024","config":"绝对配置路径"}`。正式作业由派发器从 manifest 生成，不手工改位置。

```bash
python scripts/collect_astribot_dataset.py \
  --output PREFLIGHT_DIRECTORY --jobs FROZEN_JOBS_JSON \
  --gpu 0 --envs-per-gpu 4 --backend cpu \
  --render-layout homogeneous --reset-probe 20 --audit-commit
```

每卡 1/2/4/8 分别至少一轮预热、两轮实测。`events.jsonl` 保存区间，`resources.jsonl` 保存每秒资源，`worker_result.json` 保存状态和阶段时间。完整视频审计使用 `.inprogress` 临时目录，通过后重命名且生成哈希清单。`physics_only_preflight` 不属于完整录像性能证据。各轮共享重置，早结束的成功环境保持连续记录到重置屏障。

完成 sweep 后用 `summarize_astribot_benchmarks.py` 选择并发；再以该并发在两卡真正同时测量两轮。旧七路基线包括完整解码的总时长为 463.699282 s / 4 条；原始证据在 `outputs/vib820_gpu_collection_400_preflight/baseline`。仅当历史对照、20 次重置、全环境亮度、故障注入、CPU/GPU 对照、完整 sweep、双卡吞吐及真实官方导出都通过时，运行 `seal_astribot_preflight.py` 生成 gate。其 `--help` 列出全部必填证据路径；缺项、源码变化或不足两倍吞吐会直接拒绝。

## 正式采集与恢复

用 gate 中的真实已验证 `envs_per_gpu` 和 `render_layout` 替换以下占位符。不可为了启动而编辑 gate 的状态或阈值。

```bash
python scripts/collect_astribot_dataset.py \
  --output outputs/vib820_gpu_collection_400 \
  --target-successes 400 --seed 20261014 --candidate-limit 1024 \
  --gpus 0 1 --envs-per-gpu VERIFIED_COUNT --backend VERIFIED_BACKEND \
  --render-layout VERIFIED_LAYOUT --preflight-report VERIFIED_GATE_JSON
```

恢复使用同一命令附加 `--resume`。仅验证过的原子提交可跳过；中断 attempt 从初始状态重跑，物理失败不重试。`selection.json` 按候选 ID 顺序选前 400 条，`summary.json` 的 `collected` 只表示原始采集完成，尚不表示最终发布。需要平稳停止时在每个活动 `workers/gpu*/` 目录建立 `STOP` 文件，当前整轮关闭视频并审计后退出。磁盘低于 20 GiB 同样停止新轮次；不删除历史产物。

## 导出、回放与最终检查

```bash
.conda/envs/lerobot/bin/python scripts/export_astribot_lerobot.py \
  --selection outputs/vib820_gpu_collection_400/selection.json \
  --output outputs/vib820_lerobot_400 --gpu 1

.conda/envs/pressb/bin/python scripts/finalize_astribot_collection.py \
  --raw outputs/vib820_gpu_collection_400 \
  --output outputs/vib820_gpu_collection_400_replays --prepare-replays
```

导出 25 条一个分片，官方写入器使用显式 VBR/CQ18 配置；`meta/astribot_encoder_config.json` 保存实际参数和源码哈希。官方读取器逐条检查首/中/末帧。完成的分片验证后复用；中断的临时目录改名保存，再从该分片开头重做。`meta/astribot_export_timing.json` 记录导出阶段耗时。

回放目录含两卡各 10 条作业。按已验证并发执行两次 worker，分别指定 GPU 0/1 及对应 jobs 文件：

```bash
python scripts/collect_astribot_dataset.py \
  --output outputs/vib820_gpu_collection_400_replays/gpu0 \
  --jobs outputs/vib820_gpu_collection_400_replays/jobs_gpu0.json \
  --gpu 0 --envs-per-gpu VERIFIED_COUNT --backend VERIFIED_BACKEND \
  --render-layout VERIFIED_LAYOUT --physics-only-preflight
```

回放明确执行训练的下一采样绝对目标，在四个物理步间线性插值，之后使用原始安全返回段。其数据不加入训练集。所有 20 条须在训练片段内真实按压成功，并通过完整返回/复位物理审计。

```bash
.conda/envs/lerobot/bin/python scripts/finalize_astribot_collection.py \
  --raw outputs/vib820_gpu_collection_400 \
  --export outputs/vib820_lerobot_400 \
  --replay-runs outputs/vib820_gpu_collection_400_replays/gpu0 outputs/vib820_gpu_collection_400_replays/gpu1 \
  --output outputs/vib820_gpu_collection_400_release --gpu 0
```

最终检查再次核对原始提交及导出来源、400 条官方读取、全部训练视频完整解码、20 条分层回放，生成低/中/高 yaw 三视角预览及失败/成功分布报告。只有所有检查通过才生成 `release_report.json` 的 `status=complete`。原始已提交目录不添加预览或报告，保持原证据不可变。

## 自动执行与数据盘

完整流水线为 `scripts/run_astribot_collection_pipeline.py`。当前自动推进已停止：v1 在派发前被替代，v2 因多环境相机可见邻居场景而停止。正式 attempt 为零。修复并验证隔离后，使用新的输出目录启动，避免复用配置不匹配的证据：

```bash
python scripts/run_astribot_collection_pipeline.py \
  --output /data/shared/js/PressB/vib820_gpu_collection_400_20261015/NEW_PREFLIGHT_DIRECTORY \
  --official-export "$PWD/outputs/vib820_gpu_collection_400_preflight/real_pilot_lerobot_final" \
  --render-layout VERIFIED_LAYOUT \
  --execute-production
```

启动前须等待诊断关闭；可传入 `--wait-runs` 等待有运行配置记录的旧 worker，防止污染最终测量。状态在输出目录 `pipeline_state.json`。没有通过两倍吞吐、图像隔离等 gate 时不会执行正式数据。

数据根目录为 `/data/shared/js/PressB/vib820_gpu_collection_400_20261015`，包含 `raw`、`lerobot`、`replays`、`release`（后几项会随执行产生）。原项目 `outputs/vib820_gpu_collection_400` 是 `raw` 的软链接。保留 20 GiB 磁盘下限，同时主机可用 RAM 低于 32 GiB 或已用显存超过 80% 时停止派发并排空当前轮。

当前没有可以填入上述命令的已验证多环境生产配置。`isolated` 是实验路径；重复切换环境的图像检查仍失败，不能根据单张稳定截图或数值复位 passed 启动生产。待确认的多常驻单环境进程提案见 [architecture_decision.md](architecture_decision.md)，该架构尚未执行，也尚未通过吞吐门槛。

## 已授权的常驻单环境进程流水线（2026-10-15）

用户明确授权架构提案，当前执行 `scripts/run_astribot_resident_pipeline.py`。与旧多环境流水线分开保存证据：

```bash
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0,1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMNI_KIT_ACCEPT_EULA=YES \
  .conda/envs/pressb/bin/python scripts/run_astribot_resident_pipeline.py \
  --output /data/shared/js/PressB/vib820_gpu_collection_400_20261015/preflight_resident_v1 \
  --official-export "$PWD/outputs/vib820_gpu_collection_400_preflight/real_pilot_lerobot_final" \
  --render-layout homogeneous --execute-production
```

每卡实测 1/2/4 常驻进程，每进程一环境。GPU 编号和 worker slot 独立；成功候选仍按 ID 顺序选择。源代码、门槛文件及每卡进程数冻结。全部独立录像进程的真实重叠区间计算吞吐，不把单进程吞吐乘以并发数。资源超限停止新派发并保留当前轮证据；未通过原至少两倍吞吐门槛不启动生产。正式入口增加 `--processes-per-gpu VERIFIED_COUNT --envs-per-gpu 1`，必须匹配真实 gate。

当前源冻结后的回归为 86 passed、1 skipped（NVENC 硬件检查另行 1 passed）。流水线状态及各 worker 的日志、资源、事件、复位审计均在上述新预检目录。旧失败证据不可替代新 gate。
