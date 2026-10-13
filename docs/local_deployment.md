# 本机部署

项目目录：`/home/js/robot/VLA/PressB`。Conda 安装位于
`/home/js/miniforge3`，无需更改全局 Python 或自动激活 base。

仿真环境采用项目的 Blackwell 安装配置：Python 3.11、Isaac Sim 5.0.0、
Isaac Lab v2.2.0、PyTorch 2.7.0 + CUDA 12.8。独立的数据处理环境采用
Python 3.12、CPU PyTorch 2.10.0 和 LeRobot 0.6.1。

## 使用

Astribot S1 的独立资产下载、加载验证与关节控制适配见
[Astribot 资产说明](astribot_asset.md)。该入口使用同一仿真环境。

已验收的视频可直接点击打开 VS Code 视频预览：

- [场景演示：12 个按钮完整按压](../outputs/local_deployment_20261012_v4/episode.mp4)
- [腕部相机视频](../outputs/local_deployment_20261012_v4/wrist_camera/rgb.mp4)
- [全局相机视频](../outputs/local_deployment_20261012_v4/global_camera/rgb.mp4)

在 Markdown 源码编辑器中按住 Ctrl 点击链接；在 Markdown 预览中直接点击。
打开后点击播放器的播放按钮。视频均为本机文件，无需启动仿真或下载插件。

```bash
cd /home/js/robot/VLA/PressB
source scripts/activate_local.sh
python -m pip check

# 无窗口完整仿真；每次使用新的输出目录。
bash scripts/run_local.sh --headless --video --output outputs/my_run
python scripts/audit_episode.py outputs/my_run
python scripts/audit_global_camera.py outputs/my_run
python scripts/check_results.py outputs/my_run

# 桌面会话中打开图形界面。
bash scripts/run_local.sh --hold --output outputs/my_gui_run
```

`run_local.sh` 使用 GPU 0，并清理继承的 `PYTHONPATH` 和
`PYTHONHOME`。本机入口设置 `CUDA_VISIBLE_DEVICES=0`，并将
OMP/OpenBLAS/MKL 线程数设为 1，避免小矩阵 IK 运算的线程开销；
可用 `PRESSB_CPU_THREADS` 调整线程数。GPU 1 在部署时有其他计算任务。
部署到其他 GPU 时使用原 `scripts/run.sh`，先检查 Vulkan/CUDA 枚举与实际占用。

无窗口启动时让 `World` 创建场景，跳过启动器的 UI viewport 等待。
首次 RTX 管线编译约需数分钟，可在 Kit 详细日志中观察
`Waiting for RtPso async group async compilation`；后续使用本机缓存。

本机为 Ubuntu 22.04.5，NVIDIA 驱动 590.48.01，GPU 为 RTX PRO 6000
Blackwell。Warp 1.7.1 会打印 `cuDeviceGetUuid` 接口提示；GPU 0 上的
实际 PyTorch 与 Warp kernel 检查通过，证据为
`logs/verification-local-gpu.json`。因此不能仅凭启动日志提示判断 CUDA
计算是否可用，应检查实际运算和相机验收结果。

数据处理使用独立解释器：

```bash
.conda/envs/lerobot/bin/python scripts/export_press_dataset.py --help
```

不要将 `requirements-dataset.txt` 安装进仿真环境，也不要在当前环境运行
旧的 `scripts/setup_env.sh`。

## 安装记录

- 已验收运行：`outputs/local_deployment_20261012_v4/`。12/12 按钮依次真实
  按压、回弹并归位，23,206 个物理样本，两路相机各 1,934 帧 RGB-D。
  仿真和四项独立审计均退出 0。
- 视频：[场景演示](../outputs/local_deployment_20261012_v4/episode.mp4)、
  [腕部相机](../outputs/local_deployment_20261012_v4/wrist_camera/rgb.mp4)、
  [全局相机](../outputs/local_deployment_20261012_v4/global_camera/rgb.mp4)。
  后续采集可使用同目录的 `scene.usda` 和双相机内参。
- 仿真日志：`logs/verification-local-simulation-v4.log`。
- 物理审计：`outputs/local_deployment_20261012_v4/audit_summary.json`。
- 相机审计：`outputs/local_deployment_20261012_v4/global_camera_audit.json`。
- 官方支架审计：`outputs/local_deployment_20261012_v4/mount_asset_audit.json`。
- 最终汇总：`logs/verification-local-audit-2.log`。
- 环境和验收摘要：`logs/local-deployment-summary.json`。
- 最终依赖快照：`logs/local-pressb-requirements.txt`、
  `logs/local-lerobot-requirements.txt`。
- Isaac Lab GPU 0 reset、10 步物理与渲染、正常关闭检查通过，退出码 0：
  `logs/verification-local-isaaclab-complete.log`。
  可在激活环境后运行 `python scripts/verify_local_isaaclab.py` 复核。
  验证入口通过本地 AppLauncher 子类传递延后创建场景的参数，显式创建
  USD 场景，并在关闭前注销 SimulationContext 停止回调；上游源码保持不变。
- 仿真安装日志：`logs/install-5090.log`。
- 仿真依赖快照：`logs/install-5090-requirements.txt`。
- 数据环境版本：`logs/install-dataset-versions.json`。
- 数据环境测试：`logs/verification-local-dataset-tests.log`。
- 项目测试：`logs/verification-local-pytest-final.log`，854 passed，1 skipped。
- 场景快照测试：`logs/verification-local-scene-tests.log`，16 passed，补充执行
  前述缺少快照时跳过的场景集成检查。
- 固定资产检查：`logs/fetch-assets.log`，35 个文件通过 SHA-256 校验。

VLA-JEPA 源码位于 `src/VLA-JEPA`，固定运行提交为
`9c9dc71c8e199c7428b25d103ab666395832c001`。按本次部署范围，暂不启动
模型推理或 RL 训练；需要另行提供 step_010600 的 `model.pt`、manifest、
原训练配置和 processor。完整清单见
`src/VLA-JEPA/docs/piper_rtx5090_inference.md`。

## VS Code / Codex 视频链接

本机 Codex 扩展 `26.1007.21434` 的聊天文件链接原先统一使用文本文件
打开方式；MP4 打开失败后异常被忽略，表现为点击没有反应。
已备份并修正本机扩展：MP4/WebM 链接使用 VS Code 内置视频预览，
普通文本文件和文件定位行为保持原样。

聊天中的视频链接修复需要重载一次窗口：按 `Ctrl+Shift+P`，执行
`Developer: Reload Window`（开发人员：重新加载窗口），再点击视频链接。
本文开头的相对链接可从 Markdown 预览直接点击，或在源码中 Ctrl 点击。

原扩展文件及修复记录保存在
`/home/js/.codex/backups/vscode-video-link-fix-26.1007.21434/`。
这是本机扩展修复，扩展升级可能覆盖；本文的视频链接与上游扩展文件独立。

图片链接也已补充相同处理：PNG/JPEG/GIF/WebP/SVG 等 VS Code 内置支持的格式
使用 `imagePreview.previewEditor` 打开。修改前备份、校验和及 16 项链接分流检查
保存在 `/home/js/.codex/backups/vscode-image-link-fix-26.1007.21434/`。
需要再次执行 `Developer: Reload Window` 才能加载本次图片修复。

重载后可点击 [Astribot 与按钮面板全景](../outputs/astribot_panel_preview_v2/overview.png)
或 [面板近景](../outputs/astribot_panel_preview_v2/panel.png)。
在本文的 Markdown 预览中也可直接点击图片链接。

Astribot 全身传感器入口与实际验收数据见 [传感器接入说明](astribot_sensors.md)。
启动 `python scripts/preview_astribot_panel.py --sensors --lidar-points` 可在按钮场景中
启用头部/双腕 RGB-D、底座雷达、IMU、关节及接触读数；`--view head` 可直接切到头部相机。
