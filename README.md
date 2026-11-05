# PressB 部署分支：S1B 按钮 Agent

本分支只保留真机运行所需部分：S1B 控制服务、Agent 闭环，以及模型请求（Thor 本地 Qwen 或外部 API）。仿真、数据采集、训练、VLA 推理和历史报告已移除，需要时回到 `feat/external-model-api` 分支查看。

## 运行拓扑

```
工作站（本仓库）                      S1B 机器人                    模型
s1b_agent_run.py ──SSH 打包上传──▶ s1b_agent_service.py (6210)
   │                              s1b_camera_view.py   (8765)
   ├─ 网页面板 8766                    ▲  SSH 端口转发
   └─ s1b_agent.py ─────────────────────┘
          └─ 模型请求 ─▶ Thor：SSH 隧道 127.0.0.1:18000 → Thor:8000（vLLM qwen38）
                       或 外部 API：deepseek / openai_compatible（HTTPS）
```

`s1b_agent_service.py --host S1B` 每次运行时把 `src/pressb` 和所需脚本打包传到 S1B 的 `/tmp/pressb_agent_*`，结束后删除，机器人上不需要预装本仓库。

## 依赖

- 工作站：Python 3.10/3.11，`pip install -e .`（numpy、Pillow）；能免密 SSH 到 `S1B`，用 Thor 时还需要能 SSH 到 `Thor`（主机别名写在 `~/.ssh/config`）。
- S1B：Astribot SDK（默认 `/home/astribot/astribot_sdk_aarch64`，`--sdk-root` 可改）、`/opt/astribot_ros` 下的 ROS 环境、python3 的 numpy 和 OpenCV。
- Thor：在 8000 端口提供 OpenAI 兼容 `/v1` 接口的 vLLM，模型名 `qwen38`。

## 启动

在交互终端运行（`q` 回车、Ctrl-C 或网页“停止执行”都可以停止）：

```bash
# Thor 本地模型（默认）
python scripts/s1b_agent_run.py \
  --fingertip-calibration docs/astribot/planar_control/fingertip_calibration/right_wrist_fingertip_point.json \
  --fingertip-state open

# DeepSeek 外部 API：图像和提示词会发送到 api.deepseek.com
export DEEPSEEK_API_KEY=...
python scripts/s1b_agent_run.py --model-provider deepseek \
  --fingertip-calibration docs/astribot/planar_control/fingertip_calibration/right_wrist_fingertip_point.json

# 其他 OpenAI 兼容的视觉接口
export MODEL_API_KEY=...
python scripts/s1b_agent_run.py --model-provider openai_compatible \
  --model-base https://<host>/v1 --model-name <model> \
  --fingertip-calibration docs/astribot/planar_control/fingertip_calibration/right_wrist_fingertip_point.json
```

API key 只从环境变量读取，只附加到发往模型地址的请求，不写入记录。

常用参数：`--goal`、`--max-rounds`（默认 40）、`--max-press-attempts`、`--press-standoff-mm`（默认 10）、`--press-overpress-mm`（默认 0）、`--press-drop-mm`（默认 0，即关闭）、`--speed-mm-s`、`--head-speed-deg-s`、`--demo <键盘会话目录>`。启动前确认 6210、8765、8766、18000 端口空闲。

网页：<http://127.0.0.1:8766>。每次运行记录在 `outputs/agent_web/<时间_ID>/`。

## 在 S1B 上直接运行

仓库位于 `/home/astribot/TowerGO/manipulation`，使用 Conda 环境 `towergo`：

```bash
cd /home/astribot/TowerGO/manipulation
source scripts/activate_s1b.sh        # towergo + ROS Domain 25 + Astribot SDK
export DEEPSEEK_API_KEY=...
python scripts/s1b_agent_run.py --host '' --model-provider deepseek \
  --fingertip-calibration docs/astribot/planar_control/fingertip_calibration/right_wrist_fingertip_point.json
```

`--host ''` 让机器人服务在本机启动，不再 SSH 打包。不要用 `/home/astribot/TowerGO/activate.sh`：它切到离线隔离的 Domain 125，连不上真机。S1B 上没有 `Thor` 主机别名，用 Thor 需先在 `~/.ssh/config` 配好，否则用外部 API。网页端口 8766/8765 只监听本机，从工作站查看需 `ssh -L 8766:127.0.0.1:8766 -L 8765:127.0.0.1:8765 S1B`。

## 其他入口

| 用途 | 命令 |
|---|---|
| 单独启动机器人服务 | `python scripts/s1b_agent_service.py --host S1B [--auto-confirm]` |
| 单独启动 Agent（服务已就绪时） | `python scripts/s1b_agent.py --goal ... --output outputs/<新目录> [--model-provider ...]` |
| 只读查看已有运行 | `python scripts/s1b_agent_dashboard.py --run-dir outputs/agent_web/<运行目录> --port 8767` |
| 键盘遥操作 / 录制示范 | `python scripts/s1b_keyboard_direct.py --host S1B --view-cameras [--record]` |

## 文档

- [Agent 运行说明](docs/astribot/planar_control/agent_run.md)
- [指端标定与按压流程](docs/astribot/planar_control/fingertip_calibration/README.md)
- [HTTP 接口](docs/astribot/planar_control/interfaces.md)
- 提示词：`docs/astribot/planar_control/agent_system_prompt_*.txt`

部分文档里的链接指向原分支的 `docs/report/`，在本分支中不存在。

## 测试

```bash
python -m pytest -q
```

测试不连接机器人或模型。
