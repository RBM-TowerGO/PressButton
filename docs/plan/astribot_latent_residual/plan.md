# 第一阶段实施计划

状态：**计划，尚未实现/验证**。创建日期：2026-10-22。本次只交付文档；下述任务均为后续实现任务。

## 1. 背景、目标与边界

当前 Astribot 残差 actor 使用 64 维关节位置/速度，并由 learner 拼接 56 维基座动作 chunk，没有直接读取视觉语言特征。假设是：基座动作相似的场景可能需要不同修正，增加视觉条件可以帮助残差区分这些情形。该假设需要对照实验验证，不能视为已证明的性能瓶颈。

模型底层 `VLA_JEPA.predict_action()` 已返回 `embodied_action_tokens`，也是动作头使用的条件；Astribot 推理包装层目前只提取动作。第一阶段复用同一次推理的 tokens，不增加第二次视觉前向。

- **工程成功：** 特征正确进入 actor、critic、replay、续训和评估，接口与时序验证通过。
- **算法有效：** 同协议、多训练种子的对照支持可靠收益；接入成功不等于成功率提高。
- **范围：** Astribot S1 / VIB-820，冻结 VLA-JEPA step 8000，现有残差 SAC。
- **不纳入：** mapper、投影网络、特权 critic、示范正则、TD3、低维动作、反馈频率改变、真机实验。奖励/Q/终止问题只记录诊断。

## 2. 固定设计

| 项目 | 第一阶段决定 |
|---|---|
| 特征来源 | 动作头使用的 `embodied_action_tokens` |
| token 形状 | 配置预期 `[B,32,2048]`，以真实推理验收确认 |
| 池化 | 转为 float32，沿 token 维求均值 |
| 特征处理 | 每样本 2048 维无可学习参数 LayerNorm，epsilon=`1e-5` |
| 本体状态 | 原有 q32、qd32，顺序和处理不变 |
| RL observation | 2112 维；无 latent 模式仍为 64 维 |
| actor 输入 | observation 拼接 7×8 基座 chunk，latent 模式共 2168 维 |
| critic | 同样读取扩展 observation，继续评价合成动作，不增加特权信息 |
| 残差输出 | 7×8 归一化关节残差，保持原动作合成、限位与展开 |
| 网络与算法 | SAC 隐藏层、更新逻辑不变，仅输入宽度随模式变化 |
| 基座 | 参数全部冻结，沿用原图像处理、动作反归一化与随机种子契约 |
| 时序 | GPU 物理、120 Hz 物理/30 Hz 目标、最多 7 步 chunk、推理暂停物理 |
| 额外模块 | 无 mapper、无投影、无新视觉编码器 |

LayerNorm 只处理新增 latent，不使用训练/验证集统计量，不训练编码器。其精确定义与字段见 [数据契约](data_contract.md)。此处 latent 不是 flow 初始噪声，也不是 JEPA 预测特征。

```text
同一时刻的头部/右腕图像＋指令＋关节状态
                         ↓
                  冻结 VLA-JEPA
                    ↙        ↘
      embodied-action tokens  基座动作 7×8
                 ↓               │
       float32 mean-pool           │
                 ↓               │
         无参数 LayerNorm         │
                 ↓               ↓
     q32＋qd32＋latent ─────→ 残差 SAC
                                 ↓
                     基座动作＋有界关节残差
```

## 3. 实施任务及产出

### A. 独立实验适配层

复用冻结权重和运行时，在项目实验适配层保留完整推理返回值；不原地改写 `models/` 中用于历史复现的运行时源码快照。不改变基座参数名、加载逻辑与权重校验。保留无 latent 模式作为默认兼容路径和正式对照。

产出：可追溯的适配代码版本、基座 hash、观测模式配置；同输入/seed 的旧接口与新接口动作一致证据。

### B. 推理接口

从同一次 `predict_action()` 中取得动作和 tokens，池化后随现有 NumPy RPC 返回 `vlm_latent`。健康信息声明特征能力、类型、维数及契约版本。保持动作反归一化、32 维展开和限位检查；不使用额外视觉前向或 JEPA 世界模型。

产出：真实推理形状报告、池化验证、缺失/损坏字段的失败处理。latent 模式不得静默退回无特征运行。

### C. 观测、learner 与 replay

q/qd 后追加固定规则处理的 latent。将 `ChunkJournal` 的固定 64 维检查及 runner 的初始化/恢复检查改为观测模式驱动，默认兼容旧模式。同步改变 replay、actor、critic 输入维数；合成动作逻辑不变。

latent actor、critic、优化器与 replay 从头初始化，不能继承旧残差权重或用零填充转换旧 replay。普通 transition 延迟到下一实际请求提交；末帧处理遵循 [时序契约](data_contract.md#4-时序与终止契约)。

产出：正确的当前/后继观测、可恢复的 replay/checkpoint、模式不匹配时明确拒绝的检查。

### D. 同一入口的验证与对照

无 latent 与 latent 模式使用同一 runner、相同训练和评估协议。观测模式写入协议、checkpoint 和报告。接口、时序与恢复验收全部通过后，再进入 10k 学习稳定性试验；通过后从头开始正式对照。

产出：完整验证证据、每组每种子的结果和失败诊断。计划参数、实际参数和实测结果分开保存。

## 4. 已确认的改动入口

以下是后续实施定位，不表示本次已经修改这些文件：

| 子系统 | 当前入口与需要处理的约束 |
|---|---|
| 模型运行时 | `models/astribot_head_right_joint8_ft_20261018/runtime/starVLA/model/framework/VLA_JEPA.py` 返回 tokens；`starVLA/inference/astribot_policy.py` 丢弃 tokens；在外部适配而非覆盖快照 |
| RPC | `scripts/astribot_policy_server.py` 返回 NumPy 字段；扩展特征和能力声明 |
| 观测与 journal | `src/pressb/online_rl/astribot.py` 的 `observation` 与 `ChunkJournal.start` 当前固定 64 维 |
| runner | `scripts/run_astribot_online_rl.py` 的推理、观测、末帧、初始化/恢复与模式记录 |
| learner | `src/pressb/online_rl/learner.py` 已支持配置 obs_dim；保留基座动作拼接和合成动作 critic 逻辑 |

## 5. 顺序与验收

1. 冻结对照配置、记录源码和基座身份，建立独立适配层。
2. 完成同一次推理的 latent 输出与一致性检查。
3. 完成观测/journal/replay/恢复链路与 CPU 契约检查。
4. 完成真实相机、GPU 推理、物理执行、更新和恢复冒烟。
5. A/B 各执行 10k 学习稳定性试验。
6. A/B 各从头运行三种子 100k 正式对照并完成 val-40。

详细配置、测试和报告模板以 [validation.md](validation.md) 为准。工程验收全部通过才可声明接入完成；算法结论独立给出。test-40 不参与本阶段默认验收。

## 6. 风险与版本边界

Q 的 `(0,1)` 限制、塑形回报范围、终止势函数和几何距离代理均保留现状，仅增加诊断。若诊断说明需要修正，建立新协议并重新运行 A/B，不能将修改混入当前对照。旧实验与新实验分目录保存，不覆盖历史证据。

本阶段工程规模不包含后续优化。所有未来方向见 [future_work.md](future_work.md)，它们不是完成本阶段的前提。
