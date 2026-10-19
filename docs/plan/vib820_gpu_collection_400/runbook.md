# VIB-820 采集与验收操作

所有命令从 `/home/js/robot/VLA/PressButton/PressB` 执行。预检与生产输出严格分开；没有通过生产 gate 时不能启动正式采集。当前进度见 [progress.md](progress.md)。

> **状态（2026-10-19）**：本文保留为复现手册，正式采集已经结束。400/400 条于 2026-10-17 完成，官方导出、全量视频解码、22 条回放与发布验收均通过。正式采集实际走的是「用户显式授权 + 每卡 2 个常驻单环境进程」路径（`--production-authorization`），**没有**走下文的多环境 `--preflight-report` gate，也没有完成两倍吞吐门槛；相关段落已标注为历史要求。下游训练与闭环见 [配对对照](../../astribot_paired_val_20261019.md)。

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

已授权路径把 `--preflight-report` 换成 `--production-authorization outputs/vib820_gpu_collection_400_pro6000/production_authorization.json`；两者互斥，授权文件必须匹配源码、配置和八项证据哈希（2026-10-16 起的正式采集用的就是这条路径）。

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

完整流水线为 `scripts/run_astribot_collection_pipeline.py`。以下自动推进段落是 2026-10-15 的历史状态：v1 在派发前被替代，v2 因多环境相机可见邻居场景而停止，当时正式 attempt 为零。后续正式采集没有复用这条多环境流水线，而是改走常驻单环境进程路径（见文末两节），于 2026-10-17 完成 400/400。若要在新机器上复现，仍应使用新的输出目录，避免复用配置不匹配的证据：

```bash
python scripts/run_astribot_collection_pipeline.py \
  --output /data/shared/js/PressB/vib820_gpu_collection_400_20261015/NEW_PREFLIGHT_DIRECTORY \
  --official-export "$PWD/outputs/vib820_gpu_collection_400_preflight/real_pilot_lerobot_final" \
  --render-layout VERIFIED_LAYOUT \
  --execute-production
```

启动前须等待诊断关闭；可传入 `--wait-runs` 等待有运行配置记录的旧 worker，防止污染最终测量。状态在输出目录 `pipeline_state.json`。按原设计，没有通过两倍吞吐、图像隔离等 gate 时不会执行正式数据；实际执行改为用户显式授权路径，这一门槛条件没有被满足（见 [plan.md](plan.md) 第 6.3 节）。

数据根目录为 `/data/shared/js/PressB/vib820_gpu_collection_400_20261015`，包含 `raw`、`lerobot`、`replays`、`release`（后几项会随执行产生）。原项目 `outputs/vib820_gpu_collection_400` 是 `raw` 的软链接。保留 20 GiB 磁盘下限，同时主机可用 RAM 低于 32 GiB 或已用显存超过 80% 时停止派发并排空当前轮。

当时没有可以填入上述命令的已验证多环境生产配置。`isolated` 是实验路径；重复切换环境的图像检查失败，不能根据单张稳定截图或数值复位 passed 启动生产。该阻塞通过改用「每卡多个常驻单环境进程」解决（见下两节），多环境路径至今仍未通过验收，也没有用于任何正式数据。

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

每卡实测 1/2/4 常驻进程，每进程一环境。GPU 编号和 worker slot 独立；成功候选仍按 ID 顺序选择。源代码、门槛文件及每卡进程数冻结。全部独立录像进程的真实重叠区间计算吞吐，不把单进程吞吐乘以并发数。资源超限停止新派发并保留当前轮证据；按原设计未通过至少两倍吞吐门槛不启动生产，实际由用户显式授权改用 p2（每卡 2 个常驻进程）启动。正式入口增加 `--processes-per-gpu VERIFIED_COUNT --envs-per-gpu 1`，必须匹配真实 gate 或授权文件。

当前源冻结后的回归为 86 passed、1 skipped（NVENC 硬件检查另行 1 passed）。流水线状态及各 worker 的日志、资源、事件、复位审计均在上述新预检目录。旧失败证据不可替代新 gate。

## 实际生产运行记录（2026-10-16 → 2026-10-17）

- 授权：`outputs/vib820_gpu_collection_400_pro6000/production_authorization.json`（`status=user_authorized`，policy `practical_collection_with_per_episode_audit`，绑定源码哈希、配置与八项证据）。源码覆盖前一份冻结保存在 `raw/source_versions/before_pre_motion_recovery`。
- 配置：双卡、`processes_per_gpu=2`、`envs_per_gpu=1`、`backend=cpu`、`render_layout=individual`、30 Hz 采样、七路原生相机。
- 结果：400/400 条通过逐条审计；日志 3,266 行（2,738 排队 + 528 执行）；`physical_failure` 30、`infrastructure_error` 22；147/378 经全量审计移出并由 515/516 替换。
- 导出与发布：LeRobot v3.0 400 episodes / 245,983 帧 / 13 GiB；官方读取、全帧解码、22 条回放通过；`release_report.json` `status=passed`。
- 未完成：每卡 1/2/4/8 sweep、双卡重叠吞吐测量、≥2 倍加速结论。此类结论不得用本轮数据补记。
