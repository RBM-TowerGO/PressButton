# 右腕指端固定像素标定

日期：2026-11-03。人工在相机网页（`scripts/s1b_camera_view.py`，指端点模式）中标记，保存为 `right_wrist_fingertip_point.json`；闭合、张开各一个点，同目录 `*_closed.jpg` / `*_open.jpg` 是保存时的右腕原图（核对用）。

| 状态 | 像素 (640×360) | 归一化 |
|---|---|---|
| 闭合 | (343, 248) | [0.5368, 0.6908] |
| 张开 | (465, 286) | [0.7277, 0.7967] |

标记依据：右腕相机刚性安装，夹爪状态固定时物理指端在腕图中位置不变。张开点落在带白箭头的右指前端；该点与此前助手目视估计的指端约 (470,280) 一致。标定为人工标记，不是三维接触标定；夹爪或相机被碰动后必须重新标记。

## 使用

```bash
.conda/envs/pressb/bin/python scripts/s1b_agent_run.py \
  --fingertip-calibration docs/astribot/planar_control/fingertip_calibration/right_wrist_fingertip_point.json \
  --fingertip-state open
```

启用后：腕图的接触端由程序使用该固定点提供，模型只定位按钮；`dx/dy`、腕图正 X 重叠检查均用此点。头图的指端仍由模型定位。执行 `c/o` 后程序按夹爪命令切换使用的标定点；启动时的夹爪状态由 `--fingertip-state` 指定（默认 open），程序无法从实测值自动判断。标定来源、路径与 sha256 记录在 `agent/task.json` 的 `fingertip_calibration`，每轮定位记录含 `tip_source`。不加该参数时行为与之前完全一致。
