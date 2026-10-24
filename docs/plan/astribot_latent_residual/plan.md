# 原版视觉语言观测对齐实施说明

更新：2026-10-24；以 `59281a7` 为实现基线。已接入代码，验证结果见 [implementation_report.md](implementation_report.md)。

本次实现替代此前本目录的 LayerNorm 计划，范围限于接入与工程验收，不启动 10k 或正式训练。

## 固定设计

- 增加 `state_latent_raw`：q32、qd32、2048 维原始 VLA token 均值，共 2112 维。
- 特征取自同一次冻结推理的 embodied action tokens，经动作头 dtype 转换，再 float32 均值池化，无输入 LayerNorm。
- Actor 拼接原 7×8 基础动作，Critic 读取扩展观测和合成动作；保留 Astribot 关节动作接口。
- 默认 `state_only` 继续兼容。raw 模式显式启用，新建网络/replay，不迁移旧模型。
- SAC 初始化、学习率、奖励、隐藏层及控制频率沿用当前分支，不引入 main 的 critic 改动。

## 实现路径

独立策略适配层保留底层返回的 tokens，RPC 增加特征与能力声明，不修改模型运行时快照。
runner、journal、replay、checkpoint 和评估按模式选择维数，并验证语义契约。
普通后继复用下一控制请求，终止后继来自 reset 前的末帧请求，禁止执行末帧动作。
错误输入拒绝提交，不以旧特征或零特征补齐；记录已知物理结果与基础设施错误。

完整字段、时序和恢复规则见 [data_contract.md](data_contract.md)。

## 工程验收

1. CPU：原始特征、Actor/Critic 输入、replay、恢复、确定性评估、末帧及错误路径。
2. GPU：新旧包装层同输入同 seed 动作完全一致，tokens 形状和池化正确，每次请求仅一次模型推理。
3. Isaac：每模式短采集、有限数值学习更新、保存/恢复、一回合确定性评估，检查训练文件不变及末帧 hash。
4. 记录耗时、内存开销与未验证项；不由工程接入推断成功率提高。

后续学习稳定性与多种子效果对照不属于本次执行范围。

## 控制指令扩展

已增加 `state_latent_raw_command`，在原始 latent 后追加上一完成物理步的实际 joint8 目标，
观测/网络输入为 2120/2176 维。旧模式保留，显式选择新模式并从头初始化。
该扩展不加入时间或任务 one-hot，详细时序见数据契约，实测见 [command_implementation_report.md](command_implementation_report.md)。
