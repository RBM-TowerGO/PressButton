# 原始视觉语言特征残差：数据契约

更新：2026-10-24。实现模式 `state_latent_raw`；默认 `state_only` 保持兼容。
本契约取代本分支此前未实现的 LayerNorm 计划。main 的 `state_latent` 是另一种观测，不可混用。

## 特征身份与接口

冻结 VLA 最后一层的 embodied action tokens 也是动作头的条件来源。预期形状严格为 `[1,32,2048]`。
适配层复用一次 `predict_action()` 的返回值，先转换到动作头 dtype，再转 float32 沿 token 轴求均值；与原始 PiPER 的提取顺序一致。
没有 LayerNorm、数据集标准化、投影或额外视觉前向。动作头实际 dtype 由真实 GPU 验收报告记录。

`/predict` 在现有动作字段旁增加 `vlm_latent: float32[2048]`，保留图像顺序、预处理、任务、seed、动作反归一化与展开。
`/health.latent` 声明 `feature_type=embodied_action_tokens_mean`、版本 2、来源、token 形状、池化轴、
`token_cast=action_head_dtype_then_float32`、`dtype=float32`、`normalization=none`；另记录适配源码 hash。
能力不匹配、特征缺失、类型/形状错误和非有限值均明确失败，不用零向量、旧帧或 state-only 替代。
state-only 可继续使用没有 latent 能力的旧服务。

## 观测与网络

```text
z = mean(float32(action_head_dtype(tokens)), axis=token)
obs[0:32] = q32                         # rad，既有关节顺序
obs[32:64] = qd32                       # rad/s
obs[64:2112] = z                        # 原始值，未经归一化
actor_input = concat(obs, base7x8)       # 2168 维
critic_input = concat(obs, composed7x8)  # 2168 维
```

`state_only` 的 obs/actor 输入仍为 64/120 维。残差仍输出 56 维归一化关节修正，维持原尺度、展开与限位。
奖励、SAC 初始化、学习率、网络隐藏层、控制频率和 bootstrap 规则不变。
没有新增 PiPER 的 TCP、IK 历史、时间、楼层 one-hot 或任何场景真值。

## 同步与终止

初始化的 settling tick 后刷新图像，再同步读取 q/qd；推理期间校验物理时间、q、qd 都未变化。
同一次请求返回基础动作和 latent，普通 transition 延迟到下一实际控制请求得到后继后再提交。
请求日志记录 seed、回合/请求序号、仿真时间、图像 hash、q/qd、观测与基础动作 hash；这些诊断不进入学习输入。

成功、物理失败和 timeout 在 reset 前刷新图像，以一次 `successor_only` 请求取得真实末帧特征和基础动作。
不执行末帧动作，不增加控制步数；沿用下一请求序号的 seed，并单独统计耗时。
如果控制请求已经产生有效后继但动作校验失败，复用这个请求，不能重新采样。
保持 `bootstrap_truncated=False`；真实末帧用于数据完整性，不改变终止目标。

捕获/推理失败时保留已提交数据和已知物理结果，报告基础设施错误；不完整 transition 不入 replay。
禁止 pending 跨回合、使用 reset 后观测、重复提交或用上一帧特征填充。

## Replay、恢复与评估

replay 缓存原始 latent，直接包含在 obs/next_obs 中，不存完整 tokens。
100k transitions 额外占 `100000 × 2048 × 4 × 2 = 1,638,400,000` bytes（1.53 GiB）。
启动时按未压缩 replay 和保存/恢复副本检查内存与磁盘余量。

checkpoint、replay、协议保存观测契约、冻结模型 hash、适配/runner/learner 源码身份、划分身份与训练设置。
恢复必须验证语义与维数；拒绝同为 2112 维的 LayerNorm 版本。旧 state-only 元数据缺失允许兼容，旧权重和 replay 不转换为 raw 模式。
raw 模式网络、优化器和 replay 从头初始化。评估使用同契约的确定性残差，不加载 replay、不修改训练文件，并核验参数和更新数不变。

## 上一条实际指令模式（2026-10-24）

`state_latent_raw_command` 在原始 latent 观测末尾追加 `previous_command8`：

```text
obs[0:2112] = 原 state_latent_raw 观测
obs[2112:2120] = 上一完成物理步实际下发的关节目标
actor/critic 输入宽度 = 2120 + 56 = 2176
```

指令取自 `Recorder.command`，包含残差合成与插值，七个右臂关节按 `[5,8,11,13,15,17,19]` 提取；
夹爪取索引 `[24,25,26,27,30,31]` 的目标乘 `[-1,-1,1,1,1,-1]` 后求均值，沿用冻结 joint8 压缩规则。
float32、rad、无归一化。它不是当前 VLA 基础动作，也不是尚未执行的 chunk 末目标。

每回合从 reset 后 runtime.targets 初始化，首个观测使用 settling tick 的实际指令。
推理期间检查指令不变；中途终止保留最后完成物理步的指令，动作拒绝不写入历史，末帧推理不更新历史。
日志记录 previous_command8，replay 将其存入当前/后继观测。新增控制指令契约版本 1 记录来源、布局、单位和初始化语义；视觉特征 RPC 契约不变。

显式启用新模式，新建网络/优化器/replay，不从 2112 维权重或缓存补零迁移。
旧模式观测字段及默认 state_only 保持不变；源码身份仍按既有严格恢复规则检查。
没有加入时间、任务 one-hot、多步历史，未改 SAC、奖励或控制频率。
100k transitions 的当前/后继指令额外占 6,400,000 bytes（约 6.4 MB）。
