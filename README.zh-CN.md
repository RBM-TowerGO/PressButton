# PressB — 基于 VLA-JEPA 与在线残差强化学习的电梯按键仿真

[English](README.md) | **简体中文**

**2026-10-20 RL 更新：** 第一轮 20k 残差 SAC 的 val-40 为 11/40，低于冻结基线 23/40；第二轮两个种子各 50k，在相机位姿检查中断后已于约 17:56 恢复，主种子超过 20k 并更新，复核种子暖身中。[中断记录、配置与实时进度](docs/astribot_rl/astribot_round2_20261020.md)。下方旧状态中的“尚未正式长跑”已由此更新取代。

**状态：2026-10-19。** 同一套 Isaac Sim 代码里并存两条任务主线。当前主线是 **Astribot S1 人形按 VIB-820 电梯厅呼叫按钮**：400 条专家数据已采集并导出为 LeRobot v3，VLA-JEPA 微调到 **step 8000**，冻结的 test 划分闭环为 **17/40（42.5%）**；在 validation 划分与保守续训的 **step 17,000** 做逐场景配对，结果为 **23/40 对 19/40**（McNemar p = 0.3438，不显著）。这两个权重之间，离线动作误差改善了，闭环成功率没有。在线残差 SAC 已接到该任务上，20 项 CPU 契约测试与本机 GPU/Isaac 冒烟均通过，但尚未执行正式长跑。上一代 **PiPER 12 按键主线**已完成（最佳 119/120 = 99.17%），保留在下文作为历史记录。

[当前结果](#当前结果) · [快速开始](#快速开始) · [复现当前主线](#复现当前主线) · [在线残差强化学习](#在线残差强化学习) · [上一代 PiPER 主线](#上一代-piper-主线) · [工作交接](docs/HANDOVER.md)

已验收环境为 **Isaac Sim 5.0 / Isaac Lab 2.2 / Python 3.11**，在 RTX 5090 与 RTX PRO 6000 Blackwell 上均实际运行过。仓库包含场景构建、脚本专家采集、逐条物理审计、LeRobot 导出、冻结 VLA 推理服务、同步闭环评估与残差强化学习。Isaac Sim 4.5 / Isaac Lab 2.0.2 的历史入口仍保留，但不再是主线。

## 两条任务主线

| | Astribot S1 / VIB-820（当前） | PiPER / 12 按键墙面面板（历史） |
|---|---|---|
| 机器人 | Astribot S1 人形；底盘、躯干、头部、左臂固定，右臂与原装闭合夹爪动作 | AgileX PiPER 机械臂，夹 8 mm 持杆 |
| 任务 | 单个向下外呼按钮；指令 `Press the down hall-call button.` | 12 个楼层按钮；`Press 24 floor.` … `Press 35 floor.` |
| 状态／动作 | 记录 32 关节；学习右臂 7 关节 + 右夹爪（8 维），展开为 32 维 | `base_link` 下 8 维 TCP 位姿 + 夹爪开度 |
| 相机 | 录制 7 路；送模型的为头部 1280×720 + 右腕 640×480 | 双 D435，640×480 |
| 专家数据 | 400 条、245,983 帧、30 Hz | 1,200 条、306,830 帧、30 Hz |
| 策略 | VLA-JEPA DiT-B，7 步预测，step 8000 | VLA-JEPA step 10600 + 残差 SAC |
| 状态 | 40 场景闭环已评估；在线 RL 已接入 | 已完成；最佳 119/120（99.17%） |

「上一代 PiPER 主线」一节及其后的内容属于历史记录，其中的数字不能与 S1 结果混用。

## 当前结果

当前任务是 VIB-820 场景里的 `0` 号按钮：400 × 95 mm 拉丝金属面板上唯一的向下外呼按钮，中心离地 1,200 mm，仿真行程 2 mm。S1 的底盘、躯干、头部与左臂保持固定，用原装闭合右夹爪按压。模型输入只有头部 RGB、右腕 RGB、实测 32 维关节角与固定指令文本；场景坐标、按钮几何和回合编号都不会进入策略。

| 阶段 | 结果 |
|---|---|
| 专家采集 | **400/400** 条验收通过（2026-10-16 → 10-17），来自 528 次实际执行；30 条 `physical_failure`、22 条 `infrastructure_error` 作为证据保留；7 路相机、30 Hz 采样、120 Hz 物理 |
| LeRobot v3 导出 | 400 条 / 245,983 帧 / 13 GiB / `robot_type=astribot`；官方读取与全量视频解码通过；`release_report.json` 为 `status=passed` |
| 微调 | 冻结 320/40/40 训练／验证／测试划分；best 为 **step 8000**，10,268,108,194 字节，SHA-256 `db98b7c5…`；契约、归一化统计、划分与训练源码随权重保存 |
| test 划分闭环 | **17/40 = 42.5%**；23 次超时（16 次移动但未接触、7 次在初始姿态附近停滞）；0 次非预期接触 |
| validation 划分配对 | step 8000 **23/40 = 57.5%**，step 17,000 **19/40 = 47.5%**；精确 McNemar **p = 0.3438** |
| 在线残差 SAC | 20 项 CPU 契约测试 + GPU/Isaac 冒烟通过；单环境约 2.1–2.4 transition/s，20,000 条约 2.5 小时；尚未正式长跑 |

test 与 validation 是**两批不同场景**，42.5% 与 57.5% 不能互相比较。

### 配对对照：离线误差改善，闭环没有改善

step 17,000 由 step 8000 保守续训得到（学习率约为首轮 1/3，按离线 selection score 选取），两个权重在 validation 的 40 个场景上各跑一次，逐场景、逐请求随机种子相同，配对初始条件已核验。

| 权重 | 成功 | 配对结果 |
|---|---:|---|
| step 8000 | 23/40（57.5%） | 均成功 16、均失败 14 |
| step 17,000 | 19/40（47.5%） | 改善 3、退步 7 |

| 离线指标（同一组 120 个验证锚点） | step 8000 | step 17,000 | 方向 |
|---|---:|---:|---|
| arm MAE（rad） | 0.002696 | 0.002620 | 改善 |
| gripper MAE（rad） | 0.003259 | 0.001995 | 改善 |
| selection score（rad） | 0.003190 | 0.003053 | 改善 |
| flow loss | 0.004782 | 0.006078 | 变差 |

用于选 checkpoint 的离线指标与闭环目标存在错位：离线 MAE 更好，成功率反而更低。失败集中在「接近」与「接触」两个阶段，而不是单步动作误差本身，因此更值得改进的是接触阶段的行为质量、终止/超时策略与接近轨迹，而不是继续压低离线 MAE。

| 失败分类（按顺序互斥） | step 8000 | step 17,000 |
|---|---:|---:|
| 移动但未进入按钮近区（≤ 20 mm） | 4 | 9 |
| 初始姿态附近停滞 | 6 | 4 |
| 进入 20 mm 内但未有效接触 | 5 | 4 |
| 超时时仍在接近 | 2 | 1 |
| 接近后偏离 | 0 | 2 |
| 非预期接触 `unexpected_contact` | 0 | 1 |
| **成功** | **23** | **19** |

作为参照，脚本 IK 专家（不含学习）在 40 cm × 40 cm 底盘位移网格上完成 25/30 = 83.3%，30 个关键位姿全部可解，出现 5 次非预期接触。这是另一套协议，只用于说明学习策略还剩多少差距。

## 快速开始

需求：Linux x86_64、Conda、支持 Isaac Sim 的 NVIDIA RTX GPU 与驱动（Blackwell 上已验证驱动 580.178.04）、Vulkan、无窗口运行时仍需 GPU 渲染，以及系统 `git`、`ffmpeg`／`ffprobe`（含 `libx264`）与 DejaVu 字体。

```bash
git clone https://github.com/RBM-TowerGO/PressButton.git
cd PressButton

sudo apt-get install ffmpeg fonts-dejavu-core          # 若尚未安装
bash scripts/setup_env_5090.sh                        # Python 3.11 / Isaac Sim 5.0 / Isaac Lab 2.2
bash scripts/setup_dataset_env.sh                     # Python 3.12 / LeRobot 0.6.1 / CPU PyTorch

# 策略服务使用的推理依赖覆盖环境，不修改 Isaac 环境
.conda/envs/pressb/bin/python -m venv --system-site-packages .venv-astribot-inference
```

三个环境必须分开，且不要把 `requirements-dataset.txt` 装进 Isaac 环境：

| 环境 | 用途 |
|---|---|
| `.conda/envs/pressb` | 仿真、采集、闭环评估、在线 RL |
| `.conda/envs/lerobot` | LeRobot 转换、官方读取、数据审计 |
| `.venv-astribot-inference` | 仅供 VLA-JEPA 策略服务 |

下载固定版本资产并构建当前场景：

```bash
source scripts/activate_local.sh                      # Conda + CUDA_VISIBLE_DEVICES=0
python scripts/fetch_astribot_assets.py               # S1 USD/URDF/网格，约 502 MB，按 SHA-256 校验
PYTHONPATH=.cache/usd-inspect python scripts/prepare_astribot_asset.py
python scripts/build_vib820.py --output outputs/vib820_model
```

`scripts/activate_local.sh` 与 `scripts/*_astribot_*.sh` 中写死了本工作区的绝对路径和 Conda 位置；在全新克隆中直接调用底层的 `scripts/*.py` 入口。`activate_local.sh` 默认只暴露 GPU 0；使用第二张卡时 `export CUDA_VISIBLE_DEVICES=0,1` 并传 `--gpu 1`。

在采集或评估之前，先验证脚本专家。输出目录必须不存在：

```bash
python scripts/run_astribot_press.py --headless --video \
  --config configs/astribot_vib820.json --buttons 0 --repeats 3 \
  --output outputs/vib820_press
python scripts/audit_astribot_press.py outputs/vib820_press
```

运行测试（仅 CPU；截至 2026-10-19 为 959 passed、14 skipped，共收集 969 项）：

```bash
.conda/envs/pressb/bin/python -m pytest -q
```

## 复现当前主线

### 1. 专家数据采集

400 条数据在两张卡上以「每卡 2 个常驻单环境进程」采集（CPU 物理、GPU 渲染、NVENC 编码），依据的是用户显式授权文件，而不是原计划的多环境吞吐 gate；该配置不声明任何加速比。候选位置先冻结（`seed=20261014`，底盘 X ∈ [-0.30, -0.14] m、Y ∈ [0, 0.40] m、yaw ∈ [0°, 10°]），按候选 ID 顺序取前 400 条成功；全量审计后移出 147/378 条，由单独采集的 515/516 号替换。

入口为 `scripts/run_astribot_resident_pipeline.py` 与 `scripts/collect_astribot_dataset.py`。完整的生产命令、授权文件、续采规则与替换证据见[采集操作手册](docs/plan/vib820_gpu_collection_400/runbook.md)，冻结参数与验收门槛见[采集计划](docs/plan/vib820_gpu_collection_400/plan.md)与[实验规格](docs/vib820_experiment_spec/README.md)。始终使用新的输出目录，已提交的证据不可改写。

### 2. 导出为 LeRobot v3

```bash
.conda/envs/lerobot/bin/python scripts/export_astribot_lerobot.py \
  --selection outputs/vib820_gpu_collection_400/selection.json \
  --output outputs/vib820_lerobot_400 --gpu 1
```

`observation.state` 为 32 维实测关节角（rad），`action` 为下一个 30 Hz 采样时刻的 32 维绝对关节目标；末帧重复前一帧目标，而不是撤回动作。七路 `observation.images.<camera>` 保留原分辨率、内参与外参。官方读取器检查首／中／末帧与字段形状，最终检查还会重新解码全部训练视频并执行分层动作回放：

```bash
.conda/envs/pressb/bin/python scripts/finalize_astribot_collection.py \
  --raw outputs/vib820_gpu_collection_400 \
  --export outputs/vib820_lerobot_400 \
  --replay-runs outputs/vib820_gpu_collection_400_replays/gpu0 \
                outputs/vib820_gpu_collection_400_replays/gpu1 \
  --output outputs/vib820_gpu_collection_400_release --gpu 0
```

微调本身**不在本机复现**：它在独立的 HPC 集群上用 VLA-JEPA 与 `astribot_lerobot` 适配器完成。本仓库保存了产出的权重、动作／训练契约、冻结划分、归一化统计、`config.yaml` 以及训练源码快照，位于 `models/astribot_head_right_joint8_ft_20261018/`；策略服务加载的就是这份快照与重定位后的配置。

### 3. 策略服务与闭环评估

服务启动时校验 checkpoint SHA-256 与全部数据／动作／训练契约，严格加载 1617 个张量，只监听 `127.0.0.1:8794`：

```bash
bash scripts/start_astribot_policy_server.sh            # GPU 1

# 接口冒烟（每场景 2 秒），随后跑冻结的 test 划分
bash scripts/run_astribot_closedloop.sh smoke outputs/astribot_closedloop/NEW_smoke
bash scripts/run_astribot_closedloop.sh test  outputs/astribot_closedloop/NEW_test40
```

每次运行写出 `protocol.json`、`summary.json`，以及每场景一个目录，含 `report.json`、`trajectory.npz`、`policy_requests.json`、相机时间与标定、四路视频。只有 `complete=true` 才表示全部预定场景完成。包装脚本写死本工作区路径，全新克隆请直接用 `scripts/astribot_policy_server.py` 与 `scripts/eval_astribot_closedloop.py`。

### 4. 配对权重对照

```bash
.conda/envs/pressb/bin/python scripts/run_astribot_pair.py
```

`run_astribot_pair.py` 固定写入 `outputs/astribot_paired_val_20261019/`，校验两个权重与源码哈希，起两个策略服务，调度 4 个仿真实例（每卡 2 个），并把先前的串行场景与并发分派合并、不重复计数；目录中已有 `status.json` 时直接拒绝。只重新汇总而不重跑仿真时使用 `scripts/astribot_pair_analysis.py`。

### 5. 在线残差强化学习

```bash
# 冒烟：每回合 6 秒、8 条 transition、batch 4
.conda/envs/pressb/bin/python scripts/run_astribot_online_rl.py \
  --model-run models/astribot_head_right_joint8_ft_20261018 --checkpoint-step 8000 \
  --output outputs/astribot_online_rl/NEW_smoke --smoke
```

正式配方（20,000 条 transition）、续跑与评估命令见[在线残差强化学习](docs/astribot_rl/astribot_online_rl.md)。

## 在线残差强化学习

冻结的 VLA 始终不更新，由一个小型 SAC actor 修正它的 8 维关节目标，因此这套配方与真机只能拿到关节状态的部署条件一致。

| 项目 | 取值 |
|---|---|
| 基座策略 | `models/astribot_head_right_joint8_ft_20261018`，step 8000，服务端校验 SHA-256 |
| 基座动作 | 7 步 × 8 维绝对关节目标（右臂 7 关节 + 右夹爪） |
| 残差 | 7 × 8，归一化到 [-1, 1]，按每关节 `0.02 rad`、夹爪 `0.05` 缩放 |
| 观测 | 64 维：32 个实测关节角 + 32 个关节速度；不含场景坐标、按钮几何与任务编号 |
| 奖励 | 稀疏物理成功（行程 ≥ 1 mm、右指尖力 > 0.02 N、灯锁存、无异常接触），按实际执行的 chunk 步数折扣 |
| 安全终止 | 关节非有限或越界、异常接触、底盘漂移 > 1 mm、非活动关节漂移 > 0.02 rad、残差使目标越界 |
| 数据划分 | 训练主循环只能加载冻结的 `train`（320 场景）；`--eval-only` 评估使用 `val`（默认）或 `test` |
| 时序 | 30 Hz 决策、120 Hz 物理，每个目标间线性插值 4 个物理时刻；推理期间物理暂停 |
| 回合预算 | 训练 20 秒仿真时间，评估 40 秒 |

S1 任务只接入了动作残差这一种方法：`AstribotPolicy.predict` 只暴露 `action8`／`action32`，没有接收 flow 初始噪声的接口。冒烟已验证真实相机、冻结 RPC、物理成功判定、稀疏奖励与按步折扣、replay 写入、SAC 更新、原子保存以及 `--resume --eval-only` 在验证场景上的评估，8 个回合中 0 次非预期接触。冒烟里 actor 只更新了 8 次，残差接近 0，所以其成功率基本等同于冻结策略，不能当作 RL 效果。

## 这些数字的边界

- 两轮闭环都是**同步仿真**：模型计算时物理暂停。它们衡量策略能力，不代表真机实时控制延迟，也不构成真机结论。
- 每个场景、每个权重只跑**一次**，没有跨种子稳定性估计，也没有独立泛化测试集。
- 400 条训练数据是**成功筛选后**的集合，不能作为成功率的分母；失败尝试保留为证据，但不进入训练集。
- 正式采集走的是用户显式授权，而不是原计划的「≥ 2 倍吞吐」门槛；不声明加速比，单卡单环境的诊断数字不能当作生产吞吐。
- 失败分类是描述性诊断。到按钮的距离阈值只是几何代理，不是精确碰撞表面距离，也不能用来证明失败原因。
- 场景坐标只用于初始化物理世界和评分，从不作为策略输入或控制目标。

## 上一代 PiPER 主线

黑色 PiPER 放在桌沿，夹爪以固定 8 mm 开度夹住按压杆。2 × 6 面板左列自下而上为 24–29 层，右列为 30–35 层；接触力推动按钮，边缘橙灯在按下后锁存。腕部与桌面各一台 D435，记录 640×480 RGB。采集覆盖面板左右 ±25 mm、前后 ±10 mm 偏移，每层 10 × 10 分层格、各 100 条，共 1,200 条；训练只保留到首次成功亮灯的前缀，撤回与归位留在原始诊断记录中。

原始 VLA-JEPA 在这 1,200 条上微调（step 10600），在线阶段完全冻结。两种方法分别学习：动作输出残差与 flow 初始噪声；随后冻结已训练的噪声策略，从头训练新的动作残差。下表是同一固定条件协议：12 按键 × 5 个面板位置（中心与四角）× 2 次 = 120 次评估，seed 20260930，全程不更新参数。

| 方法 | 在线训练 | 成功 | 成功率 |
|---|---|---:|---:|
| 原始 VLA-JEPA | 无 RL | 25/120 | 20.83% |
| 独立动作残差 | 1M | 81/120 | 67.50% |
| 独立初始噪声策略 | 1M | 81/120 | 67.50% |
| 叠加两份独立训练的策略 | 两份 1M 权重，无联合重训 | 96/120 | 80.00% |
| 固定噪声，重训 9D 残差，γ=0.99 | 400k | 114/120 | 95.00% |
| 固定噪声，重训 9D 残差，γ=0.995 | 400k | 118/120 | 98.33% |
| 上一设置，残差 scale 减半 | 400k | 87/120 | 72.50% |
| **固定噪声，重训 XYZ 残差，γ=0.995** | **400k** | **119/120** | **99.17%** |

最新 XYZ actor 只输出 7 × 3 = 21 维残差，位置 scale 为 0.02 m，旋转与夹爪沿用冻结模型输出。该轮 400k 训练约 7 小时 25 分钟，完成 398k 次更新；119/120 是单次固定条件结果，不代表所有随机种子下都有 99.17% 成功率。平滑实验（1,320 次评估）未通过预设门槛：XYZ 指数平滑 α=0.8 使实际关节 jerk 中位数下降约 14%、末端位置 jerk 下降约 30%，筛选为 119/120，但独立种子复核为 117/120，没有候选满足全部条件，因此默认仍是**不做额外后处理**。细节与复现见[固定噪声后的残差训练](docs/piper/residual_on_frozen_noise.md)、[结果 JSON](docs/results/online_rl_20261009.json)、[结果 CSV](docs/results/online_rl_20261009.csv)、[平滑汇总](docs/results/smoothing_20261009.csv)。

演示视频覆盖全部 12 个按键，保持 1× 仿真时间，上为全局视角、下为腕部视角；任务结束后显示带标记的终止停帧。点击预览图播放或下载。这些演示不是包含面板偏移的 120 次评估。

| 原始 VLA → 固定噪声＋9D 残差 | 9D 残差 → XYZ 残差 | XYZ 平滑实验（未采用） |
|---|---|---|
| [![强化学习前后](docs/media/before-vs-residual.jpg)](docs/media/before-vs-residual.mp4) | [![9D 与 XYZ 残差](docs/media/pose9-vs-xyz.jpg)](docs/media/pose9-vs-xyz.mp4) | [![XYZ 平滑实验](docs/media/xyz-smoothing-experimental.jpg)](docs/media/xyz-smoothing-experimental.mp4) |
| 左 4/12；右 12/12（γ=0.995、400k） | 左 12/12；右 11/12，其中一次 24→25 误按 | 左右均 12/12；完整同期评估为 116/120 与 117/120 |

原视频来源、压缩参数、实际帧数与审核记录见[媒体说明](docs/media/README.md)。

PiPER 主线建立了三服务架构：Isaac 仿真节点、冻结 VLA-JEPA 推理节点与 SAC 训练节点通过 HTTP 通信，采用 64 个并行环境、批量推理与独立 episode 重置。1M transition 的实测训练用时约 19.81 h（动作残差）与 15.16 h（初始噪声）；最新 XYZ 400k 约 7.43 h。一个 transition 是一个环境内最多 7 步的动作 chunk，不是一条 episode。S1 的 runner 是单进程版本，因为多环境相机隔离尚未通过验收；见[在线 RL](docs/piper/online_rl.md)、[高吞吐版本](docs/piper/online_rl_fast.md)与[协议](docs/piper/online_rl_protocol.md)。视频平滑度来自实测轨迹回放，未用播放加速或插帧制造差异。

## 目录结构

| 目录 | 内容 |
|---|---|
| `src/pressb/` | PiPER 场景、运动学、规划与相机；VIB-820 场景、材质与 S1 按键控制；数据语义 |
| `src/pressb/astribot_collection/` | 常驻/流水线采集、逐条审计、NVENC 与渲染控制 |
| `src/pressb/online_rl/` | SAC learner、三节点协议、批量仿真／推理、固定噪声残差与 S1 残差 runner |
| `scripts/` | 安装、资产准备、采集、导出、回放、评估、在线 RL 与审计入口 |
| `configs/` | VIB-820／Astribot 传感器配置、PiPER 面板布局、分层采集与在线 RL 配置 |
| `tests/` | CPU 单元测试与可选 USD 场景集成测试（65 个测试文件，收集 969 项） |
| `assets/*.json` | 固定资产来源、SHA-256 校验和与安装配准元数据 |
| `docs/` | 当前流程说明、实验记录与工作交接 |
| `docs/results/`、`docs/media/` | 随仓库发布的小型结果摘要与精选 PiPER 压缩演示 |
| `models/` | 本地 VLA 权重、契约、数据划分与源码快照（Git 忽略） |
| `datasets/`、`outputs/`、`logs/`、`.conda/`、`.cache/`、`vendor/` | 本地数据、证据、环境、缓存与下载（Git 忽略） |
| `src/VLA-JEPA/`、`src/ZPRL/` | 单独克隆的依赖／参考仓库（Git 忽略） |

## 测试

```bash
.conda/envs/pressb/bin/python -m pytest -q

# USD 集成测试需要一个显式的 PiPER 场景快照
PRESSB_TEST_SNAPSHOT=outputs/edge_feedback/scene.usda \
  PYTHONPATH=.cache/usd-inspect:src \
  .conda/envs/pressb/bin/python -m pytest -q tests/test_dataset_scene.py

# 数据／视频审计在独立的 LeRobot 环境运行
.conda/envs/lerobot/bin/python -m pip install pytest
.conda/envs/lerobot/bin/python -m pytest -q tests/test_press_dataset_audit.py tests/test_export_fps.py
```

没有场景快照的克隆会跳过依赖快照的 USD 集成测试；显式指定不存在的快照会报错。依赖下载素材的运动学测试必须在素材准备之后运行。Isaac 环境缺少 PyAV 时，相应视频测试会明确跳过，而不是静默通过。

## 文档索引

当前主线：[VIB-820 场景](docs/astribot/vib820.md) · [S1 资产](docs/astribot/astribot_asset.md) · [S1 按键控制](docs/astribot/astribot_press.md) · [传感器](docs/astribot/astribot_sensors.md) · [初始姿态](docs/astribot/astribot_initial_pose.md) · [腕部相机核对](docs/astribot/astribot_wrist_camera_audit.md) · [相机清单](docs/astribot/astribot_camera_inventory.md) · [闭环评估](docs/astribot_eval/astribot_closedloop_20261018.md) · [配对对照](docs/astribot_eval/astribot_paired_val_20261019.md) · [在线残差 RL](docs/astribot_rl/astribot_online_rl.md) · [采集计划](docs/plan/vib820_gpu_collection_400/plan.md) · [采集手册](docs/plan/vib820_gpu_collection_400/runbook.md) · [实验规格](docs/vib820_experiment_spec/README.md)。

历史 PiPER 主线：[场景细节](docs/piper/scene_details.md) · [数据集](docs/piper/dataset.md) · [30 Hz 数据](docs/piper/dataset_30hz.md) · [首次亮灯裁剪](docs/piper/dataset_press_only.md) · [面板随机化](docs/piper/dataset_panel_randomization.md) · [分层覆盖](docs/piper/dataset_panel_stratified.md) · [回放](docs/piper/replay.md) · [策略评估](docs/piper/policy_eval.md) · [动作平滑](docs/piper/policy_smoothing.md) · [在线 RL](docs/piper/online_rl.md) · [高吞吐 RL](docs/piper/online_rl_fast.md) · [固定噪声后的残差训练](docs/piper/residual_on_frozen_noise.md)。

部署与运维：[RTX 5090 说明](docs/setup/rtx5090_setup.md) · [PRO 6000 环境](docs/setup/PRO6000_ENV.md) · [本机部署](docs/setup/local_deployment.md) · [工作交接](docs/HANDOVER.md)。

## 数据、权重与许可

Git 只保存源码、配置、资产清单、结果摘要与精选 PiPER 演示。数据集、VLA/RL 权重、场景快照、原始录像与 Conda 环境都在本地并被忽略：400 条导出数据在本地数据盘（文档记录的根目录为 `/data/shared/js/PressB/vib820_gpu_collection_400_20261015`），原始采集在 `outputs/vib820_gpu_collection_400_pro6000/raw`，微调权重在 `models/`。S1 描述与 PiPER 资产按固定提交下载并通过 SHA-256 校验（`assets/sources.lock.json`、`assets/astribot_sources.lock.json`）；VIB-820 材质为 ambientCG／Poly Haven 的 CC0 衍生贴图，来源记录在 `assets/materials/vib820/`。NVIDIA 软件与资产仍受其自身许可约束，见 [THIRD_PARTY.md](THIRD_PARTY.md)。本仓库尚未为自有代码声明开源许可证；公开可见不等于授权。
