# S1B：官方 OpenPI π0.5 PyTorch 接线

本入口保留头部＋右腕 RGB、右臂七关节＋右夹爪＋头部两关节（joint10），以及现有 S1B 的 HTTP/NPZ → 同步七步 → 250 Hz SDK 链路。训练和推理调用官方模型；本项目只提供数据、变换、产物与部署适配。

上游固定为 `Physical-Intelligence/openpi@215abfb217dbac7d5f1273282331b9b1866c0479`，位置与完整版本见 `repos.lock.json`。不要直接修改 `.deps/openpi`；更换上游或适配代码后，需要重新生成准备包并训练/验收，产物校验会拒绝版本不一致。

本次验收记录：[validation.json](validation.json)。

## 本轮边界与离线检查

没有训练、下载模型权重、连接 S1B、启动机器人服务或下发动作。`check-deps` 只导入依赖并构建配置，不构建模型或下载 tokenizer。`check_openpi_s1b_offline.py` 只使用本机随机端口及明确的测试替身；正式入口不支持其 `test_only` 后端。

从项目根运行：

```bash
# 独立 Python 3.11 环境；不会更改现有 Qwen/ROS 环境。
python3 scripts/setup_openpi_s1b.py --python /path/to/python3.11
.venv-openpi/bin/python scripts/openpi_s1b.py check-deps
.venv-openpi/bin/python scripts/openpi_s1b.py audit \
  --dataset data/astribot_s1b_real_joint10_mixed_v2
.venv-openpi/bin/python scripts/check_openpi_s1b_offline.py
```

安装脚本使用官方 frozen uv lock，`UV_LINK_MODE=copy` 后应用官方 transformers 补丁，避免污染共享缓存。即使仅使用 PyTorch，官方模块也依赖 JAX/Flax；这是独立环境中的正常依赖。首次安装会下载较大的 CUDA/PyTorch 软件包，但不会下载机器人模型权重。若默认源较慢，可给安装脚本增加 `--index-url https://pypi.tuna.tsinghua.edu.cn/simple`；该路径从同一 frozen lock 导出精确版本与 wheel 哈希，再通过镜像安装，不重新解析版本。

离线验收覆盖：数据窗口和划分、双相机解码、共享变换、身份校验、HTTP/NPZ 往返及 shadow 七步调度。**这些结果不证明真实模型加载、训练收敛、GPU 吞吐或真机成功率。**

## 后续数据准备

输入为本地 LeRobot v3。必须提供 `meta/splits.json`、`meta/training_anchors.json`、`meta/data_contract.json`；可用 `--splits`、`--anchors`、`--contract` 显式指定同语义文件。不会读取 Hub 默认 split，也不会将 `info.json` 的 `0:212` 当作训练划分。未来原始 HDF5/MCAP 数据应先通过项目现有导出流程转换。

```bash
.venv-openpi/bin/python scripts/openpi_s1b.py prepare \
  --dataset /path/to/new_s1b_lerobot_v3 \
  --output runs/openpi/prepared_v1
```

准备目录必须不存在。原数据不修改。准备包包含模型配置、统计、划分、有效窗口、源时间契约、数值与视频哈希、上游版本及适配代码哈希。

数据约定：

- 状态/动作顺序见现有 joint10 契约：右臂 7、夹爪、头部 2。关节单位 rad；夹爪状态保留连续 SDK 刻度，动作是 0/1。
- `action[k]` 已对应下一网格目标，窗口为 `action[k:k+7]`。拒绝缺口、跨 episode、不足七步及时间错位，不自动补齐。
- 状态统计取训练有效 anchor；动作统计取训练窗口覆盖的去重行。连续维采用 q01/q99 的中心与半宽；最小半宽 `1e-6`，二值夹爪动作 center=0、scale=1。归一化不裁剪，输出可逆；不与官方标准统计再叠加。
- `head → base_0_rgb`，`right → right_wrist_0_rgb`；左腕补零并 mask。官方 `ResizeImages(224,224)` 同用于训练与推理。
- 模型内部状态/动作补齐到 32 维，部署只取前 10 维；使用绝对关节位置，无 delta transform。头部依照示范记录参与学习，不自动固定。
- 示例任务文本为 `Use the right gripper to press the circular button on the elevator panel.`。旧 v1 按钮任务也可读取，但部署 YAML 的 task 必须与导出产物一致。

## 后续转换和训练（本轮未执行）

先自行下载官方 `pi05_base` JAX checkpoint 至本地。下面的转换命令要求本地目录存在，不会自动下载基础权重。按官方转换器要求，传入 checkpoint 根目录（包含 `params/`），绝对路径中必须含 `pi05`：该固定版本官方转换器用路径字符串选择 adaptive norm 分支，入口会提前检查。

```bash
.venv-openpi/bin/python scripts/openpi_s1b.py convert \
  --checkpoint /path/to/pi05_base \
  --output runs/openpi/pi05_base_pytorch

# GPU 数量、batch、steps 根据后续训练资源调整。
.venv-openpi/bin/python scripts/openpi_s1b.py train \
  --dataset /path/to/new_s1b_lerobot_v3 \
  --bundle runs/openpi/prepared_v1 \
  --base-weights runs/openpi/pi05_base_pytorch \
  --exp-name press_button_v1 --batch-size 8 --steps 30000
```

此入口使用官方 `scripts/train_pytorch.py` 的训练循环，替换数据工厂为本项目的 v3 reader。默认全参数微调、bfloat16、关闭 W&B、无 EMA、不开启 torch.compile；没有宣称单张 5090 可以完成该默认全量训练。官方配置默认学习率/优化器保持不变。多 GPU 使用独立环境的 `torchrun` 启动同一个脚本，`--batch-size` 是全局 batch，必须能被进程数整除：

```bash
.venv-openpi/bin/torchrun --standalone --nproc_per_node=8 \
  scripts/openpi_s1b.py train \
  --dataset /path/to/new_s1b_lerobot_v3 \
  --bundle runs/openpi/prepared_v1 \
  --base-weights runs/openpi/pi05_base_pytorch \
  --exp-name press_button_v1 --batch-size 32 --steps 30000
```

训练输出为 `runs/openpi/pi05_s1b_joint10/press_button_v1/<step>/`。使用官方实际保存的 step（该固定版本会保存 `steps-1`，整除保存间隔时也会保存 `steps`），不要套用旧 Qwen 的 2000/4500/8000 白名单。官方每次完成保存后写入 `s1b/` sidecar，绑定准备包、训练设置及权重哈希；未完成 sidecar 的 checkpoint 不能导出。继续同一训练使用 `--resume`；准备包必须一致。训练读取器目前只训练 train 划分，**不会自动做验证集模型评估或选择 best checkpoint**，后续训练实验需另行评估留出轨迹。

## 后续导出与部署（本轮未执行）

```bash
.venv-openpi/bin/python scripts/openpi_s1b.py export \
  --checkpoint runs/openpi/pi05_s1b_joint10/press_button_v1/30000 \
  --output runs/openpi/s1b_export
```

导出目录必须不存在。导出复制部署权重与 S1B 资产，不携带 optimizer；生成 `identity.json` 和绑定全部运行资产的 `data_contract.json`。更换权重、归一化、配置、适配代码或上游版本都会导致校验失败。模型服务启动会完整校验权重哈希，耗时取决于磁盘速度。

离线加载真实微调模型，可先仅启动工作站上的策略服务：

```bash
.venv-openpi/bin/python scripts/s1b_policy_server.py \
  --config configs/s1b_deployment_openpi_pi05.yaml \
  --checkpoint 30000 --warmup
```

该服务只监听本地 8796，无机器人依赖。还需要使用保存的真实观测核验预测；合成预热不证明控制质量。`inference_seconds` 对此后端包含官方预处理、模型计算和后处理，`preprocess_seconds=0`，不应将其当成单独计时的 GPU forward。

以下命令会连接机器人，仅供后续部署阶段执行。外层启动器使用现有 S1B 环境；模型进程按 YAML 的 `model.python` 使用 `.venv-openpi/bin/python`，也可用 `--policy-python` 指定绝对路径。

```bash
# 真机只读：不获取 SDK 控制权，但会 SSH 启动远端观测/shadow 服务。
.venv-astribot-inference/bin/python scripts/start_s1b_inference.py \
  --config configs/s1b_deployment_openpi_pi05.yaml \
  --checkpoint 30000 --mode shadow --auto-start --duration 60

# shadow 与实际延迟核验完成后，现场人工操作执行。
.venv-astribot-inference/bin/python scripts/start_s1b_inference.py \
  --config configs/s1b_deployment_openpi_pi05.yaml \
  --checkpoint 30000 --mode execute \
  --operator-present --estop-ready --request-control-handover
```

默认保持同步整段执行：新观测→推理→七个 30 Hz 目标→下一轮观测；推理等待预算和 250 Hz 执行端规则保持原样。动作在官方输出变换中反归一化，夹爪进入原有执行层后按 0.5 阈值转为 SDK 0/100。启动及结果日志保留模型和数据契约身份。

物理接线沿用：相机连接 S1 原机载采集系统；工作站通过已有开发网络连接机载计算机。SSH 隧道转发 55555/55557 图像、55558 状态、6200 控制；策略服务 8796 留在工作站。本轮不改变网口、电机线路、ROS、SDK、急停或端口暴露方式。

## 本机 GPU 依赖边界

本次实际安装的官方冻结依赖为 `torch 2.7.1+cu126`，其编译架构最高为 `sm_90`；当前工作站是 RTX 5090（compute capability 12.0）。**导入验收通过不代表该 wheel 可以运行本机模型。** 真正使用 5090 训练或推理前，应在独立 `.venv-openpi` 中换成支持 Blackwell 的 CUDA 12.8+ PyTorch 构建，并重新进行依赖与模型冒烟验收；不要改动现有 Qwen 环境。依赖来源参考 [PyTorch 官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/)。当前训练和策略入口会在分配模型权重之前拒绝不支持 Blackwell 的构建。本轮未进行此 GPU 环境变体迁移或模型运行。
