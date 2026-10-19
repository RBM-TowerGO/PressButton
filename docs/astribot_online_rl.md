# Astribot S1 在线残差强化学习（VIB-820 单按钮）

2026-10-19 新增。目标是把 PressB 里已经验证过的「冻结 VLA + 在线残差 SAC」接到 Astribot S1 的 VIB-820 按键任务上：不改模型权重、不改采集数据，用一个小的残差策略修正冻结策略输出的 8 维关节目标。

**当前状态：CPU 契约测试通过（20 项），GPU/Isaac 冒烟与正式训练尚未在本机执行。** 先跑通 `--smoke`，再开正式训练。

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

前置：策略服务已在 GPU 1 上启动（见 [闭环评估](astribot_closedloop_20261018.md)）：

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

## 6. 计算成本（估算，非实测）

配对评估的实测吞吐约每 chunk 1.2 s 墙钟（4 个独立仿真实例、2 路相机、含推理）。本接入是**单进程单环境**，因为多环境相机隔离仍未通过验收：

- 20 s 上限约 60 个 chunk，成功或安全失败会提前结束；按平均 50 个 chunk 估算；
- 20,000 条 transition ≈ 400 回合 ≈ 7 小时；
- 50,000 条 ≈ 17 小时。

要提速只能回到多进程 + 单一 learner 的 actor-learner 拆分（本仓库 PiPER 线的三节点结构），那是第二阶段的工作，本轮不做。

## 7. 验证状态与限制

- 已通过：`tests/test_astribot_online_rl.py` 20 项 CPU 检查，覆盖观测、残差合成、32 维展开、越界检测、奖励与折扣、终止/截断互斥、`ChunkJournal` 的 running/success/failure/timeout 四种路径、评估不写 replay、8 维 base 的 SAC 前向与更新、replay 往返、默认 scale、训练划分隔离与 CLI 参数守卫。
- 未执行：GPU/Isaac 冒烟、正式训练、残差评估。本会话的 shell 看不到 `/dev/nvidia*`，无法验证。
- 限制：同步仿真，不代表真机实时；训练成功率含探索偏差，不能当评估结果；残差幅度受 `--scale` 与关节限位约束，不能改变 `fixed_action32`；每场景单次评估，不给跨种子稳定性结论。
- `--resume` 时 `--max-transitions` 必须与 checkpoint 的 replay 容量一致（脚本会直接拒绝），否则环形缓冲区语义会变。
