# Pro6000 环境配置记录

配置日期：2026-10-15（Asia/Shanghai）
项目：`/home/js/robot/VLA/PressButton/PressB`

## 已完成

- 使用项目缓存的 Miniforge3 26.7.2-0 安装 Conda：`/home/js/miniforge3`。官方在线下载本次失败，安装器自身校验及安装成功。
- 初始化 Bash，默认不激活 base；注册项目环境，可用 `conda activate pressb` / `conda activate lerobot`。
- 原复制环境保存在 `.conda/backups-pro6000/`，重建到 `.conda/envs/`，修复旧目录引用，重新安装 PressB 和 Isaac Lab 的 editable 包。
- pressb：Python 3.11.17、Isaac Sim 5.0.0、Isaac Lab v2.2.0、PyTorch 2.7.0+cu128、torchvision 0.22.0+cu128。保留项目约束，修正 packaging=23.0、wheel=0.45.1、setuptools=78.1.1。
- lerobot：独立 Python 3.12、LeRobot 0.6.1、CPU PyTorch 2.10.0；安装 FFmpeg 5.1（含 libx264），pressb 的 ffmpeg/ffprobe 链接到该环境。
- 两套环境 pip check 通过；两张 RTX PRO 6000 的 PyTorch CUDA、矩阵乘法和 Warp kernel 均通过。
- 项目测试：927 passed、17 skipped；LeRobot 数据审计/导出测试：21 passed。
- 35 个固定版本素材全部 SHA-256 校验通过，腕部相机 USD 生成且几何数组逐位一致。
- OpenUSD 24.11 保持在 `.cache/usd-inspect` 隔离目录。

## 使用

```bash
source /home/js/miniforge3/etc/profile.d/conda.sh
cd /home/js/robot/VLA/PressButton/PressB
conda activate pressb
python -m pip check
# 数据处理使用：conda activate lerobot
```

也可在项目目录执行 `source scripts/activate_local.sh`。

## 驱动问题已解决（2026-10-15）

经用户授权，已使用系统仓库将 NVIDIA 595.99.02 切换为 `nvidia-driver-580-open` 580.178.04，并完成重启。

- 安装退出码 0，DKMS 对当前内核 6.8.0-40-generic 构建并安装成功，启动镜像更新完成，dpkg --audit 无异常。
- 重启后两张 RTX PRO 6000 均加载 NVIDIA Open Kernel Module 580.178.04。
- 两张 GPU 的 PyTorch CUDA、矩阵乘法和 Warp kernel 复测通过。
- Isaac Lab 无窗口启动、场景 reset、10 步物理及渲染验证通过，进程退出码 0。此前 RTX 渲染模块崩溃已解决。
- 启动中仍有旧版内置 Warp 的 cuDeviceGetUuid 警告，但本次启动和物理步进成功。

原始故障日志保留，系统驱动安装日志为 `/var/tmp/pro6000-driver-580-install.log`。

尚未运行完整 12 按钮轨迹、训练或批量采集，未配置额外 VLA 模型权重。默认场景快照 `outputs/edge_feedback/scene.usda` 尚未生成，因此此前相应集成测试跳过。

复测命令：

```bash
cd /home/js/robot/VLA/PressButton/PressB
source scripts/activate_local.sh
python scripts/verify_local_isaaclab.py
```

## 证据

`logs/pro6000-gpu-verification.log`、`logs/pro6000-pytest.log`、`logs/pro6000-dataset-tests.log`、`logs/pro6000-assets-final.log`、`logs/pro6000-wrist-prepare.log`、`logs/pro6000-isaaclab*.log`、`logs/pro6000-driver-580-plan.txt`。
依赖快照：`logs/pro6000-pressb-requirements.txt`、`logs/pro6000-lerobot-requirements.txt`。

重启后证据：`logs/pro6000-gpu-after-reboot.log`、`logs/pro6000-isaaclab-after-reboot.log`。

默认单卡激活方式也通过同样的验证，退出码 0：logs/pro6000-isaaclab-default-activation.log。
