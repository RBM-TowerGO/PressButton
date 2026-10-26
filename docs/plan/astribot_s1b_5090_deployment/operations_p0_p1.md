# P0 / P1 本机操作手册

在仓库根目录执行。无需安装新环境，使用 `.venv-astribot-inference`。以下入口不导入机器人 SDK、不申请控制权。部署模式目前仅支持离线与影子推理服务；P2～P5 不属于此次实现。

## 资产与配置核验

```bash
.venv-astribot-inference/bin/python scripts/check_s1b_deployment.py \
  --config configs/s1b_deployment_v1.yaml \
  --output outputs/s1b_deployment/p0_audit.json
```

核验两份完整权重的 SHA-256、manifest、运行目录 processor、数据/动作/训练契约、split、配置及 runtime 的固定身份。`configs/s1b_deployment_assets_v1.json` 为版本控制中的信任基线，不能在校验失败时自动更新以绕过检查。迁移路径时需显式审核并更新路径配置和身份文件。

`--mode execute` 会列出全部未确认的 SDK、观测及约束字段并失败。即使手动填满，当前版本也会明确拒绝执行：执行器须在 P4 单独实现和验收。已有 SDK commit 仅是 2026-10-25 只读快照，不代表当前机器人安装版本已重新验证。

## 完整离线评估

```bash
.venv-astribot-inference/bin/python scripts/eval_s1b_deployment.py \
  --mode offline --checkpoint 2000 \
  --output outputs/s1b_deployment/p1_step002000

.venv-astribot-inference/bin/python scripts/eval_s1b_deployment.py \
  --mode offline --checkpoint 8000 \
  --output outputs/s1b_deployment/p1_step008000
```

顺序运行以避免两个模型互相影响显存和计时。默认每个模型使用原验证集 48 个固定 anchor，检查输入缩放与训练 reader 逐像素一致、输出及夹爪反馈边界，并对真实尺寸 RGB 输入比较直接调用与本机 HTTP 的输出，要求最大绝对差为 0。随后执行额外 20 次 warmup 和 200 次 GPU 同步计时。计时覆盖模型内部 processor 与预测，训练 reader 的解码、224 缩放和网络传输不计入这 200 次统计；HTTP RTT 单独记录。

报告包含 `resolved_config.json`、`environment.json`、`identity.json`、`anchors.json`、`timings.json`、`summary.json`。逐样本保留 episode、frame、seed、预测、目标与误差；汇总同时给出逐维和 7×10 误差，不混合关节 rad 和夹爪类别作为物理误差。源码文件哈希记录在环境快照。缩短 warmup 或 iterations 的运行只可用于调试，不标记完整验收。

首个 anchor 用于测冷启动推理；HTTP 一致性测试在 anchor 评估之后执行，因此其“首次 HTTP 请求”是已加载并已运行过的模型。模型加载计时包含资产 SHA 校验和模型导入、构造、严格载入、CUDA 同步。冷启动不能用稳定推理延迟替代。

## 本机 HTTP 服务

```bash
.venv-astribot-inference/bin/python scripts/s1b_policy_server.py \
  --config configs/s1b_deployment_v1.yaml --checkpoint 2000 \
  --device cuda:0 --port 8796
curl http://127.0.0.1:8796/health
```

仅绑定 `127.0.0.1`。`--checkpoint` 必填，只接受 2000 / 8000。用 Ctrl-C 停止服务；切换权重须停服并重新启动，新客户端使用新 session。每次启动在 `outputs/s1b_deployment/server_*` 创建独立配置、身份和 JSONL 预测日志。日志中的 `sdk_target=null` 表示未发送任何 SDK 目标；`gripper_binary` 是原始预测按 0.5 阈值解码的类别。

POST `/predict` 使用不含对象数组的 NPZ，最大 8 MiB（同时检查解压大小）。必填字段与精确 dtype/shape 以 `src/pressb/s1b_deployment/protocol.py` 为准；字符串为标量 Unicode/bytes 数组，原始时间/映射时间数组顺序为 head/right/state。本机离线测试的时间字段是合成值，不代表已标定机器人时钟。HTTP 错误只返回错误信息，不提供替代动作。服务最多处理一个模型请求，重复/乱序 observation_id、身份不符和并发请求均失败。

## 自动化验证

```bash
.venv-astribot-inference/bin/python -m pytest tests/test_s1b_deployment.py -q
```

P1 通过仅说明可进入 P2 的只读采集开发。本机推理统计不能替代有线链路、真机相机来源、时钟映射、执行窗口、关节限位或停止能力验收。
