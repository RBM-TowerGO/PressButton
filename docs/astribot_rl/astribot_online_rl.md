# Astribot S1 在线残差强化学习（VIB-820 单按钮）

2026-10-19 新增。目标是把 PressB 里已经验证过的「冻结 VLA + 在线残差 SAC」接到 Astribot S1 的 VIB-820 按键任务上：不改模型权重、不改采集数据，用一个小的残差策略修正冻结策略输出的 8 维关节目标。

**当前状态（2026-10-20）：第一轮 20k 训练已完成，val-40 为 11/40，低于冻结基线 23/40。第二轮两个独立种子各 50k；相机位姿检查中断后，已于约 17:56 在本机双 RTX 5090 上从保存点恢复，主种子已超过 20k 并更新，复核种子仍在暖身采样。** 第一轮诊断见第 9 节，第二轮配方见第 10 节；中断证据、修复、效率实测与自动进度见 [2026-10-20 运行记录](astribot_round2_20261020.md)。

**2026-10-24 补充：** 原始视觉特征 + 上一条指令的 100k 实验已按用户要求停止；最后报告 41,674 条 transition / 更新、390 回合、200 次成功（51.28%），可恢复 learner/replay 保存点为 41,041 条，未执行 val-40。配置、停止证据及恢复限制见 [100k 实验记录](raw_command_100k_20261024.md)。上方 2026-10-20 状态为历史记录。

## 1. 为什么只做动作残差

旧 PiPER 线有两条方法：动作残差（`action_residual`）和 flow 初始噪声（`initial_noise`）。S1 的推理服务 `AstribotPolicy.predict` 只暴露 `action8` / `action32`，没有接收初始噪声的接口，因此本轮只接动作残差；要复现噪声方法需要改推理契约，属于另一件事。

## 2. 契约

| 项目 | 取值 |
|---|---|
| 基座策略 | `models/astribot_head_right_joint8_ft_20261018`，step 8000，SHA256 由服务 `/health` 校验 |
| 基座动作 | 7 步 × 8 维绝对关节目标（右臂 7 关节 + 右夹爪开口） |
| 残差 | 7 步 × 8 维，归一化到 [-1,1]，乘 `--scale`（默认 `0.02×7 + 0.05` rad） |
| 执行目标 | `joint8 = base8 + residual × scale`，再展开成 32 维（其余关节用 `fixed_action32`，夹爪按 ±1 符号映射到 6 个连杆） |
| 观测 | 64 维：32 个实测关节角 + 32 个关节速度。**不含**场景坐标、按钮几何、任务编号，因此与真机可用输入一致 |
| 控制 | 30 Hz 决策、120 Hz 物理、每个目标之间线性插值 4 个物理时刻；推理时物理暂停（同步仿真） |
| 奖励 | 稀疏物理成功：同一物理时刻按钮行程 ≥1 mm 且右指尖力 >0.02 N，灯锁存、无异常接触；成功按 γ^(k-1) 折时（k 为 chunk 内第几个控制步） |
| 折扣 | 每个 chunk 用实际执行步数 `γ^n`，n≤7；成功/失败为 terminated，超时为 truncated |
| 安全终止 | 非有限或越界关节、异常接触、底盘漂移 >1 mm、非活动关节漂移 >0.02 rad、残差使目标越界（>0.02 rad 容差）都立即终止该回合并记 0 奖励 |

回合结束条件：成功、安全失败、或到达 `--max-seconds`（默认 20 s 仿真时间，比评估用的 40 s 短，纯粹为了降低训练成本）。

## 3. 数据划分与防泄漏

训练只允许冻结划分里的 `train`（320 条）场景；`val`/`test` 的候选 id 在训练主循环里不可达（`load_contract` 对训练强制 `split=train`）。评估必须显式 `--eval-episodes` + `--resume`，用 `val`（默认）或 `test` 的 40 个场景各跑一次，结果写 `eval/summary.json`。

评估沿用闭环评估的物理判据（同样的行程/力/灯/异常接触规则、同样的 40 秒上限），由本 runner 的 `evaluate()` 执行，不另立标准。

## 4. 运行

前置：策略服务已在 GPU 1 上启动（见 [闭环评估](../astribot_eval/astribot_closedloop_20261018.md)）：

```bash
bash scripts/start_astribot_policy_server.sh
```

冒烟（6 秒仿真、8 条 transition、batch 4，用于确认相机、奖励、保存、恢复都能跑）：

```bash
.conda/envs/pressb/bin/python scripts/run_astribot_online_rl.py \
  --model-run models/astribot_head_right_joint8_ft_20261018 \
  --checkpoint-step 8000 \
  --output outputs/astribot_online_rl/smoke_20261020 --smoke
```

正式训练（20,000 条 transition，默认配置）：

```bash
.conda/envs/pressb/bin/python scripts/run_astribot_online_rl.py \
  --model-run models/astribot_head_right_joint8_ft_20261018 \
  --checkpoint-step 8000 \
  --output outputs/astribot_online_rl/residual_20k_20261020 \
  --max-transitions 20000 --warmup-transitions 2000 --batch-size 256 \
  --max-seconds 20 --gpu 0
```

中断后继续（回合级恢复：learner/replay 恢复，仿真从新回合开始），并在训练结束后用 40 个验证场景、40 秒上限做确定性评估（与闭环评估的上限一致）：

```bash
.conda/envs/pressb/bin/python scripts/run_astribot_online_rl.py \
  --model-run models/astribot_head_right_joint8_ft_20261018 \
  --checkpoint-step 8000 \
  --output outputs/astribot_online_rl/residual_20k_20261020 \
  --resume --eval-episodes 40 --eval-split val --eval-max-seconds 40
```

只做评估（跳过训练循环）：

```bash
.conda/envs/pressb/bin/python scripts/run_astribot_online_rl.py \
  --model-run models/astribot_head_right_joint8_ft_20261018 \
  --checkpoint-step 8000 \
  --output outputs/astribot_online_rl/residual_20k_20261020 \
  --resume --eval-only --eval-episodes 40 --eval-split val
```

## 5. 产物

| 路径 | 内容 |
|---|---|
| `learner.pt` | SAC actor/critic/optimizer/RNG 与状态快照（原子写） |
| `replay.npz` | 完整 replay（含环形位置、base 宽度、RNG） |
| `status.json` | transitions / episodes / updates / 近 20 回合成功率 |
| `metrics.jsonl` | 每次梯度更新的 loss、Q、alpha、熵 |
| `protocol.json` | 模型身份、源码哈希、scale、gamma、预算 |
| `episodes/episode_*/report.json` | 每回合原因、控制步、行程、力、灯 |
| `eval/summary.json` | 残差策略在 val/test 划分上的成功率 |

## 6. 计算成本（2026-10-19 冒烟实测）

本接入是**单进程单环境**，因为多环境相机隔离仍未通过验收。冒烟实测训练吞吐 **2.14–2.43 transition/s**（3 个成功/超时回合，见第 8 节），因此：

- 20,000 条 transition ≈ **2.5 小时**（原按 1.2 s/chunk 的估算 7 小时偏保守）；
- 50,000 条 ≈ **6.3 小时**；
- 回合时长取决于策略：成功回合约 20–26 s 仿真（620–781 个控制步），超时回合 40 s（1200 步）。

要提速只能回到多进程 + 单一 learner 的 actor-learner 拆分（本仓库 PiPER 线的三节点结构），那是第二阶段的工作，本轮不做。

## 7. 验证状态与限制

- 已通过：`tests/test_astribot_online_rl.py` 20 项 CPU 检查，覆盖观测、残差合成、32 维展开、越界检测、奖励与折扣、终止/截断互斥、`ChunkJournal` 的 running/success/failure/timeout 四种路径、评估不写 replay、8 维 base 的 SAC 前向与更新、replay 往返、默认 scale、训练划分隔离与 CLI 参数守卫。
- 后续已完成 GPU/Isaac 冒烟、第一轮正式训练与残差评估，见第 8、9 节。本节最初的 CPU-only 状态已被后续记录取代。
- 限制：同步仿真，不代表真机实时；训练成功率含探索偏差，不能当评估结果；残差幅度受 `--scale` 与关节限位约束，不能改变 `fixed_action32`；每场景单次评估，不给跨种子稳定性结论。
- `--resume` 时 `--max-transitions` 必须与 checkpoint 的 replay 容量一致（脚本会直接拒绝），否则环形缓冲区语义会变。

## 8. GPU/Isaac 冒烟记录（2026-10-19）

环境：两张 RTX PRO 6000 Blackwell（驱动 580.178.04），Isaac Sim 5.0 / Kit 107，策略服务在 cuda:1，仿真在 GPU 0。

| 运行 | 场景 | 结果 | 关键数字 |
|---|---|---|---|
| 训练冒烟 | train 231 | timeout | 180 控制步 / 721 物理步 / 26 transitions / 11.0 s |
| 评估（val 1,3,9） | val 1 timeout；val 3、9 success | 2/3 成功 | 3：620 步、20.7 s、行程 1.103 mm、力 0.332 N；9：763 步、25.4 s、1.205 mm、0.369 N |
| 训练（seed 3） | train 95 | success | 735 步、24.5 s、1.062 mm、0.305 N、105 transitions |
| 训练（seed 5） | train 304 | success | 781 步、26.0 s、1.370 mm、0.410 N、112 transitions |
| 训练（seed 7） | train 2 | timeout | 1200 步、40 s、172 transitions、最大力 0.0156 N（未达 0.02 N） |

逐项确认：

- **真实相机**：每次 `prepare_wave` 都写出 7 路初始化图像（`shared/initial_cpu_env0_*.png`），观测走 head + 右腕两路。
- **奖励与终止**：成功回合在 replay 中写入 `reward=0.9704`（chunk 第 7 步按到，γ^6）与 `reward=0.9851`（第 4 步，γ^3），对应 `terminated=1`；折扣分别为 γ^7=0.9655 与 γ^4=0.9801，说明按实际执行步数折扣生效。超时回合写入 `truncated=1`。
- **保存/恢复**：冒烟保存 `learner.pt` + `replay.npz`（容量 8），`--resume --eval-only --eval-episodes 3` 正确加载并完成 3 个 val 场景评估，写 `eval/summary.json`。
- **稳定性**：6 次成功运行共 8 个回合、0 次非预期接触；运行结束后 GPU 显存释放（策略服务 10.7 GiB 归零）。

冒烟发现并修复的两个问题：

1. `summarize_episode` 返回的 `control_steps` 与回合字典的同名键冲突，导致回合收尾 `TypeError`；已改名为 `physics_records`。
2. `--eval-only` 仍走训练预算校验（warmup 2000 > max-transitions 8）且强制加载 replay；已改为评估只需要 `learner.pt`，不再校验训练预算与 replay 容量。

仍需注意：

- 启动阶段 `omni.warp` 报 `cuDeviceGetUuid` 不可用（CUDA error 36），随后 Warp 1.7.1 正常枚举两张卡、内核在 cuda:0 加载，6 次成功运行均未受影响；正式长跑前建议复核驱动/CUDA 组合。
- Isaac 关闭时会以 0 退出码结束，异常必须看输出目录里的 `infrastructure_error.json`，不能只看退出码。
- 冒烟用的 actor 只更新了 8 次（残差接近 0），因此成功回合基本复现冻结策略；成功率不能当作 RL 效果。

## 9. 20k 正式训练与 val-40 评估（2026-10-19）

按上一节的命令完成首次正式训练，并做确定性评估。结论是**负结果**：残差策略显著劣于冻结基线，且原因可定位到 actor 饱和。

### 9.1 训练

| 项目 | 值 |
|---|---|
| 时间 | 2026-10-19 17:25 → 20:15（约 170 分钟，单环境） |
| 预算 | 20,000 transitions（warmup 2,000，batch 256，UTD≈1） |
| 回合 | 169 个，8 个 train 场景循环采样 |
| 成功 | **67/169（39.6%）**；原始 `status.json` 记为 70，其中 3 条被下面的判定 bug 误记为成功 |
| 失败 | timeout 60，`unexpected_contact` 42，0 次基础设施错误 |
| 吞吐 | ≈2.0 transition/s（与冒烟一致） |
| checkpoint | 每 1,000 次更新落盘，最终 `learner.pt` 7.2 MB、`replay.npz` 11 MB |

训练内部趋势：前 20 回合成功率 70%，最后 20 回合 20%。`metrics.jsonl` 显示归一化动作范数在约 500 次更新内从 **0.14 跳到 4.4**（56 维上限 7.48）后一直饱和；alpha 从 0.010 衰减到 0.0037，熵稳定在约 38 nats。诊断与证据保存在 `outputs/astribot_online_rl/residual_20k_20261020/diagnosis.json`。

**判定 bug（已修）**：如果同一 chunk 内先按下、随后又发生非预期接触，旧代码把回合记为 `status=success` 而 `reason=unexpected_contact`。按评估契约这必须是失败。受影响的 3 条训练回合是候选 63/213/418；训练结束后已把 `episode_outcome()` 改为失败优先，并补了单测，随后用修复后的代码重新评估。

### 9.2 评估（修复后重跑）

val 划分 40 个冻结场景，每个一次，确定性残差，40 s 上限：

| 指标 | 冻结 step 8000 基线 | 20k 残差策略 |
|---|---:|---:|
| 成功 | 23/40（57.5%） | **11/40（27.5%）** |
| timeout | — | 23 |
| `unexpected_contact` | — | 6 |
| 误分类成功（status=success 且 reason≠press） | — | **0** |

逐场景配对：均成功 6、均失败 12、残差改善 5、残差退步 17，精确 McNemar 双侧 **p=0.0169**，即显著变差（不是采样噪声）。

- 改善的场景：1、180、221、293、349；
- 退步的场景：3、51、52、57、72、94、166、249、264、268、309、311、319、325、385、421、429。

证据：`outputs/astribot_online_rl/residual_20k_20261020/eval/summary.json` 与同目录 `paired_comparison.json`（逐场景 baseline/rl 布尔值）。

### 9.3 解释与限制

这是「策略退化」而不是「评估口径」的问题：残差 actor 把 tanh 输出推到饱和（平均残差约 0.012 rad/关节，但每步都在所有关节上同向偏移），critic 在早期接近零残差的数据之外乐观外推，于是 SAC 一路把动作推向边界；结果是冻结策略被推离原本成功的轨迹，出现「原地不动」和「蹭到面板」两类失败。

同时也要承认样本量限制：20k transitions 对 56 维连续动作、稀疏奖励的 SAC 明显不足（PiPER 线用的是 1M 级），所以不能据此断言「残差 RL 对 S1 无效」，只能断言**当前配置（20k、无塑形、无幅度正则、自动 alpha）会退化**。

### 9.4 下一轮方案（按优先级）

1. **势函数奖励塑形（训练期）**：用按钮距离构造 Φ，训练用 `F = γΦ(s') − Φ(s)`，评估仍只用稀疏物理判据。势函数塑形不改变最优策略，能直接解决「进了近区但不接触」的探索难题。
2. **动作幅度正则**：在 pre-tanh 均值上惩罚 `λ_a·mean(mean²)`（不是 tanh 后，因为饱和处梯度会消失），把残差压回小幅修正。
3. **训练稳定性**：warmup 提到 5,000，actor 学习率降到 3e-5，固定 `alpha=0.02`（关掉自动 alpha，避免它一路衰减）。
4. **预算**：至少 50k–100k transitions，并保留 40 s 回合上限。
5. 评估协议保持不变（val 40、每场景一次、同一成功判据），这样下一轮能与本轮和 23/40 基线直接配对比较。

## 10. 第二轮：稳定化配置（本机双种子运行中）

针对第 9 节的退化，第二轮只改「训练信号与稳定性」，不改任务、判据、观测、动作契约和评估协议，便于与 23/40 基线做同场景配对比较。

### 10.1 改动

| 改动 | 说明 |
|---|---|
| 势函数塑形（仅训练） | Φ = −指尖到按钮帽的最小距离，`F = γ^n·Φ(s') − Φ(s) = d − γ^n·d'`。potential-based，不改变最优策略；评估时系数强制为 0，成功判据仍是稀疏物理规则 |
| 动作幅度惩罚 | `lambda_a` 作用在 **pre-tanh 均值** `mean(mean²)` 上（tanh 饱和处梯度为 0，惩罚饱和后的动作无效） |
| 训练稳定性 | warmup 5,000；actor LR 3e-5（q 仍 3e-4）；固定 `alpha=0.02`（`--no-auto-alpha`） |
| 预算 | 50,000 transitions（约 7 小时），checkpoint 每 2,000 次更新 |
| 不变 | 残差 scale 0.02 rad×7 + 0.05、γ=0.995、40 s 回合、train 划分、val-40 评估协议 |

### 10.2 命令

```bash
bash scripts/start_astribot_policy_server.sh
.conda/envs/pressb/bin/python scripts/run_astribot_online_rl.py \
  --model-run models/astribot_head_right_joint8_ft_20261018 --checkpoint-step 8000 \
  --output outputs/astribot_online_rl/residual_50k_shaped_20261019 \
  --max-transitions 50000 --warmup-transitions 5000 --batch-size 256 \
  --max-seconds 40 --training-frequency 1 --checkpoint-every 2000 \
  --shaping 1.0 --action-penalty 0.1 \
  --policy-lr 3e-5 --init-alpha 0.02 --no-auto-alpha \
  --seed 20261021
```

训练结束后用同一协议评估：

```bash
.conda/envs/pressb/bin/python scripts/run_astribot_online_rl.py \
  --model-run models/astribot_head_right_joint8_ft_20261018 --checkpoint-step 8000 \
  --output outputs/astribot_online_rl/residual_50k_shaped_20261019 \
  --resume --eval-only --eval-episodes 40 --eval-split val
```

### 10.3 启动前提与记录

- 2026-10-19 21:10 检查时两张卡被其他用户（jchen572）的 LeRobot 训练占满（GPU0 54.9 GB / GPU1 42.5 GB，均 ~99% 利用率），因此未启动；等空闲后再开。
- 启动前会再次确认：无我们的遗留进程、8794 端口空闲、显存与磁盘余量充足。
- 失败模式：任何基础设施错误会写入输出目录 `infrastructure_error.json`，不会把中断记为成功。
