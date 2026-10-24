# latent 偏移信息诊断

本实验只采集仿真真值并训练离线预测器，不训练残差策略，不改变 actor 输入、奖励或物理成功规则。

## 数据与几何

- 面向面板时，右为世界 −Y，上为 +Z，朝机器人为 −X。按钮旋转时三个轴跟随按钮刚体旋转。
- 标签为实测腕部姿态乘以 `tcp_in_wrist` 得到的几何 TCP 相对按钮表面中心的投影，单位 mm。
- 表面中心使用当前按钮刚体位姿及本场景圆柱帽的局部 `[-0.003,0,0]` 偏移，不使用帽刚体中心。
- 每次观测同时保存指尖碰撞网格最前端的法向距离。夹爪张开时 TCP 是几何参考点，不是接触点。
- 真实接触位置来自 PhysX `position` 字段；无接触时位置列表为空，尚未确认接口能力时标记 null，不能将空列表解释为接触中心。
- 几何、q/qd、图像、VLA 动作和 latent 属于同一次物理冻结观测。图像由既有 LiveRuntime 检验相机位姿和渲染时钟。
- 保存的是 B 使用的逐样本 LayerNorm 后 2048 维 latent，动作是完整 7×8 块。

受控采集固定种子 20261023，从冻结训练集合的 320 个场景中打乱选前 100 个，每个场景 5×5×3 个目标位置。实际标签不强行等于 IK 目标。场景按 60/20/20 划分，跨场景测试是探针的泛化测试，不代表冻结 VLA 从未见过这些场景。

闭环采集使用上述随机序列第 101–200 个场景，原始 VLA 动作、不加残差，正式实验显式使用 `--max-seconds 40`，与既有 A/B 正式评估一致（通用诊断入口默认 25 秒）。先做 40 个场景，有效近按钮场景不足 20 时继续到最多 100。近按钮区域为三个标签绝对值均不超过 30 mm。

每个 scene 目录含 `samples.npz`、`observations.jsonl` 和 `summary.json`，顶层 `protocol.json` 记录场景划分、模型身份、源码哈希。存在完整摘要的场景可以跳过；中断场景移到 `interrupted_*` 后重采，不进入统计。不同协议不能复用同一输出目录。

## 复现

在仓库根目录执行。推理和仿真使用已有独立环境；先确认没有占用同一 GPU 的训练任务。

```bash
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMNI_KIT_ACCEPT_EULA=YES
unset PYTHONPATH PYTHONHOME
.venv-astribot-inference/bin/python scripts/astribot_policy_server.py \
  --run models/astribot_head_right_joint8_ft_20261018 --device cuda:0 --port 8801
```

模型服务保持运行，在另一个终端先设置同样的环境变量，再执行以下命令，`PROBE_OUT` 使用新的实验目录：

```bash
export PROBE_OUT=outputs/astribot_online_rl/latent_offset_probe_new
.conda/envs/pressb/bin/python scripts/collect_astribot_offset_probe.py \
  --mode controlled --scenes 5 --grid 0 --output "$PROBE_OUT/smoke"
.conda/envs/pressb/bin/python scripts/collect_astribot_offset_probe.py \
  --mode audit --output "$PROBE_OUT/audit"
.conda/envs/pressb/bin/python scripts/collect_astribot_offset_probe.py \
  --mode controlled --output "$PROBE_OUT/controlled"
.conda/envs/pressb/bin/python scripts/collect_astribot_offset_probe.py \
  --mode closedloop --max-seconds 40 --output "$PROBE_OUT/closedloop"
.conda/envs/pressb/bin/python scripts/analyze_astribot_offset_probe.py \
  --controlled "$PROBE_OUT/controlled" --closedloop "$PROBE_OUT/closedloop" \
  --output "$PROBE_OUT/analysis" --audit "$PROBE_OUT/audit/summary.json"
.conda/envs/pressb/bin/python scripts/diagnose_astribot_offset_shift.py \
  --controlled "$PROBE_OUT/controlled" --closedloop "$PROBE_OUT/closedloop" \
  --analysis "$PROBE_OUT/analysis" --output "$PROBE_OUT/shift_diagnosis"
.conda/envs/pressb/bin/python -m pytest tests/test_offset_probe.py -q
```

## 统计与验收

预测器为状态＋动作、latent、二者组合。岭回归 alpha 在验证集选取；MLP 两层各 64 单元，三个种子，两档 weight decay、验证早停。标准化只拟合训练集。保存拟合模型、所有测试预测、指标 JSON、中文报告与 PNG 图。

MAE 报告样本平均和场景平均，增量改善置信区间使用场景等权的 2,000 次配对 bootstrap。方向分类以 ±2 mm 为中间区并报告 ±1、±4 mm；平衡准确率按该测试子集中实际存在的类别计算，类别计数同时报告。

打乱 latent 只是敏感性诊断，输入组合可能分布外，不单独作为因果证据。真实闭环仅作冻结测试，按首次目标接触前后分组；任何接触后信息都不能支持接触前纠偏结论。

`diagnose_astribot_offset_shift.py` 不重新训练模型；它复现保存的预测，将指定标准化状态坐标替换为训练均值以定位外推失稳。该诊断不替代正式比较，也不用于选择新的超参数。主判定要求加 latent 后的 MAE 胜过训练均值常数基线，避免把失效预测器之间的改善误判为可用的闭环增益。

本次完整结果见 [2026-10-23 实验报告](../../results/astribot_latent_offset_probe_20261023.md)。

只读验收同时记录独立重置 off/off/on 对照和同一冻结观测的双请求对照。独立重置存在渲染／仿真非确定性时不能声称逐步等价；冻结观测对照要求开关记录产生逐元素相同的动作和 latent，且物理时间、关节位置及速度不变化。独立重置的不一致结果必须保留和报告。
