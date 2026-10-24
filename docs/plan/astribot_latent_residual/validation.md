# 原始 latent 工程验收

更新：2026-10-24。仅工程验收，不启动长训练。实测结果见 [implementation_report.md](implementation_report.md)。

## CPU

```bash
.conda/envs/pressb/bin/python -m pytest tests/test_astribot_latent.py tests/test_astribot_online_rl.py tests/test_online_rl_learner.py -q
```

覆盖原始池化（保留均值和尺度）、坏特征、旧模式兼容、Actor/Critic 实际输入、replay/网络恢复、
同维度错误契约、终止与普通后继、动作拒绝复用、相机/物理同步错误、不完整数据拒绝提交和 SAC 配置不变。

## 真实模型

`verify_astribot_latent_policy.py` 使用 `.venv-astribot-inference/bin/python`，输入为真实同步相机捕获的 NPZ。
参数：`--run models/astribot_head_right_joint8_ft_20261018 --input <first_request.npz> --output <new_directory> --device cuda:0`。
它校验旧包装层、新包装层及同 seed 重复请求的三个动作字段完全一致；校验单次模型调用、冻结权重、
实际 token 形状、动作头 dtype 转换和原始均值池化，并保存数值证据。

## Isaac 短冒烟

先使用推理环境启动 `scripts/astribot_policy_server.py --run models/astribot_head_right_joint8_ft_20261018 --device cuda:0 --port 8799`。
再运行：

```bash
.conda/envs/pressb/bin/python scripts/validate_astribot_latent_engineering.py \
  --endpoint http://127.0.0.1:8799 --output <new_directory>
```

脚本显式覆盖 `state_only` 与 `state_latent_raw`，沿用当前 runner 的 SAC/奖励默认值。
每模式训练采集上限 2 秒模拟时间、replay 容量 8，执行少量更新；随后加载 checkpoint 做一回合 2 秒确定性 val 评估。
检查损失/参数有限、replay 恢复逐字段一致、末帧观测和基础动作 hash 对应真实请求，以及评估不修改训练文件。
短冒烟不检验长期学习稳定性、成功率或真机效果；缺少 GPU/Isaac 时必须报告未验证。

## 控制指令模式验收

CPU 包含 `tests/test_astribot_command_observation.py`：验证指令压缩、输入布局、网络输入、恢复契约、
实际四步插值中途失败、首次指令与回合重置、拒绝动作以及推理期间指令变化。

仅运行新模式真实短冒烟（仍需先启动推理服务）：

```bash
.conda/envs/pressb/bin/python scripts/validate_astribot_latent_engineering.py \
  --modes state_latent_raw_command --endpoint http://127.0.0.1:8799 --output <new_directory>
```

脚本 `--modes` 默认覆盖三种模式，可显式选择。恢复时按观测契约确定维数，核对当前与后继指令与请求日志一致，
并验证末帧请求、有限损失/参数及评估不改训练文件。
