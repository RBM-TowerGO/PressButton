# 闭合夹爪视角：Qwen3.8按钮识别

测试日期：2026-11-03。找到10月30日、10月31日和11月1日双指合拢状态的真实右腕原图，选取8张覆盖大小、位置、遮挡、亮环和不可见按钮的样例，逐张调用同一个Qwen3.8服务。此次只画按钮框，标签没有数字序号。

查看 [识别总览](annotated_contact_sheet.jpg)、[原图总览](original_contact_sheet.jpg)。下表有每张原图／预测的并排对照。**所有框和描述均保留模型原答，未手动修正。**

## 结果

完整重跑收到8个可解析响应，共输出5个按钮框。01/02/03完整按钮、04局部遮挡按钮和07边缘裁切按钮都检出；05/06只露部分圆环的重遮挡按钮均未检出；08没有可辨按钮，正确不给框。01虽找到按钮，却将向下箭头读成other。部分检测框仍有偏移或边缘漏覆盖。

这组样例说明：**闭合夹爪的照片中，完整或轻遮挡按钮通常可找到；当夹爪遮住按钮中央、只剩少量圆环时，本次两张都漏检。检测到按钮也不保证读对其符号。**

| 图 | 场景 | 模型框数 | 模型符号／可见性 | 助手目视复核 | 对照 |
|---|---|---|---|---|---|
| 01 | 小按钮上方完整可见 | 1 | other／full | 检出完整圆按钮；框偏上、下缘略漏。向下箭头误识别为other，并称“非箭头符号”。 | [查看](cases/01_small_upper/comparison.png) |
| 02 | 大按钮偏左完整可见 | 1 | down_arrow／full | 完整按钮、向下箭头及框均与目视相符；亮灯uncertain是保守判断。 | [查看](cases/02_large_left/comparison.png) |
| 03 | 中部完整按钮 | 1 | down_arrow／full | 完整按钮及向下箭头判断合理；框略紧，下方圆环有少量漏覆盖。 | [查看](cases/03_middle_complete/comparison.png) |
| 04 | 右下按钮局部遮挡 | 1 | down_arrow／partial | 按钮局部遮挡、向下箭头判断合理；框略偏上。文字称下半部分遮挡偏夸大，实际主要左下缘被挡。 | [查看](cases/04_low_right/comparison.png) |
| 05 | 近距重遮挡，仅露圆环 | 0 | not_visible | 模型称not_visible，未给框；原图能看到部分圆环，为强遮挡按钮漏检。中央箭头确实不可读。 | [查看](cases/05_occluded_unlit/comparison.png) |
| 06 | 近距重遮挡，外环明亮 | 0 | not_visible | 模型称not_visible，未给框；原图能看到明亮的部分外环，为强遮挡按钮漏检。中央箭头确实不可读。 | [查看](cases/06_occluded_lit/comparison.png) |
| 07 | 上边界裁切按钮 | 1 | down_arrow／partial | 正确检出顶部裁切按钮，partial和向下箭头合理；框左侧稍松。 | [查看](cases/07_top_clipped/comparison.png) |
| 08 | 按钮不可见 | 0 | not_visible | 原图没有可辨按钮，模型正确不画框。不能由此断言场景中物理上没有按钮。 | [查看](cases/08_button_not_visible/comparison.png) |

本轮输入 8,696 token、输出 992 token，逐请求耗时之和 95.4 秒。8/8是格式合格情况，不是识别准确率。完整模型字段与像素框见 [summary.json](summary.json) 和 [results.csv](results.csv)。

上述复核为助手目视判断，没有人工标注真值，不计算正式准确率、mAP或IoU。这是定性选取样例，含同一运行相近帧；本轮只检测按钮，与上一轮四类物体加指尖的提示不同，不能将两轮差异全部归因于夹爪开闭。

## 原图与闭合依据

全部为640×360真实右腕图，没有仿真图或关闭过程中的模糊帧。01/02有同步实测夹爪反馈，6张键盘会话图只有清晰的目视合拢证据；未把close命令当作到位证明。SDK夹爪刻度0为打开、100为闭合，不是毫米开口量，见[接口约定](../../astribot/s1b_contract_20261025/README.md)。

| 图 | 拍摄时间（北京时间） | 闭合依据 | 归档原图 |
|---|---|---|---|
| 01 | 2026-10-31T13:25:44.304+08:00 | 实测97.687737＋目视合拢 | [原图](cases/01_small_upper/original.jpg) |
| 02 | 2026-10-30T17:30:47.773+08:00 | 实测97.656053＋目视合拢 | [原图](cases/02_large_left/original.jpg) |
| 03 | 2026-11-01T16:43:54.712+08:00 | 目视合拢，无同步夹爪实测 | [原图](cases/03_middle_complete/original.jpg) |
| 04 | 2026-11-01T16:53:52.429+08:00 | 目视合拢，无同步夹爪实测 | [原图](cases/04_low_right/original.jpg) |
| 05 | 2026-11-01T16:44:37.856+08:00 | 目视合拢，无同步夹爪实测 | [原图](cases/05_occluded_unlit/original.jpg) |
| 06 | 2026-11-01T16:44:54.794+08:00 | 目视合拢，无同步夹爪实测 | [原图](cases/06_occluded_lit/original.jpg) |
| 07 | 2026-11-01T16:52:26.507+08:00 | 目视合拢，无同步夹爪实测 | [原图](cases/07_top_clipped/original.jpg) |
| 08 | 2026-11-01T16:43:25.138+08:00 | 目视合拢，无同步夹爪实测 | [原图](cases/08_button_not_visible/original.jpg) |

01/02的状态相对图像时间差分别约+10.27ms、−4.30ms，图片和状态哈希均与来源manifest匹配。完整来源、时间和证据路径见 [closure_evidence.json](closure_evidence.json)。状态或相机元数据已按原字节复制到每帧的 `source_*.json`。

## 方法与传输记录

- 当前 `/v1/models` 记录为 `qwen38` → `Inferact/Qwen3.8-27B-NVFP4`，见 [models.json](models.json)。本次未重验权重revision。
- 单张原始JPEG独立输入，temperature=0，thinking关闭，max_tokens=1024；没有示范、其他图片、历史动作、文件名、选图说明、实测夹爪状态或助手判读。原图没有缩放、裁剪或加框。
- 只检测真实圆形按钮及其外圈；要求框包围可见部分，不推测被夹爪完全挡住的部分。没有可辨按钮应返回空列表，同时允许uncertain。分别返回full/partial、图标与亮灯外观；看不到图标时应为unreadable。
- 坐标为0–1000归一化xyxy，左上为原点，x向右、y向下；按图像边缘约定换算 `x_px=x*640/1000`、`y_px=y*360/1000`。不裁剪、不重排、不挪动模型框。
- 图片清单与哈希在 [plan.json](plan.json)；[推理前目视参考](visual_reference_before_inference.json) 在任何图像请求前记录，未提供给模型；[推理后复核](visual_review.json) 单独保留。
- **第一次连接中断，只收到01响应；02远端关闭连接，03–08隧道端口拒绝连接。它们不是模型漏检。** [中断记录](../20261103_qwen38_closed_gripper_buttons_transport_interrupted/README.md) 完整保留。重建连接后按相同图片、提示和参数完整重跑8张，本报告统一使用新一轮全部首答，没有在两轮中择优或做答案纠正。见 [传输恢复说明](transport_recovery.json)。
- 本次只访问模型，未连接或控制机器人。

## 归档和复现

每帧保存 `original.jpg`、可读实际请求 `request.readable.json`、HTTP原始正文 `response.body`、HTTP状态 `response.http.json`、完整模型响应 `response.json`、解析及像素框 `result.json`、`annotated.png` 和 `comparison.png`。将可读请求的 `original.jpg` 替换为该文件原始字节的 `data:image/jpeg;base64,...` 即可恢复图像输入。

[准备检查](prepare_checks.json) 与 [完成检查](postrun_checks.json) 核对输入字节、原始响应、坐标和脚本快照。源码快照为 `script_used.py`，依赖为 `support_snapshot.py`。

在一个终端建立模型隧道并保持终端运行：

```bash
ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:18538:127.0.0.1:8000 Thor
```

另一个终端在仓库根目录运行，输出目录必须未存在：

```bash
/home/js/robot/VLA/PressButton/.conda/envs/pressb/bin/python \
  scripts/evaluate_qwen38_closed_gripper_buttons.py \
  --manifest docs/report/20261103_qwen38_closed_gripper_buttons/samples.json \
  --output outputs/qwen38_closed_button_repeat \
  --model-base http://127.0.0.1:18538/v1
```

可先 `--prepare-only` 再 `--execute-prepared`。单独使用归档脚本时，需把 `support_snapshot.py` 复制为同目录的 `evaluate_qwen38_wrist_boxes.py` 以满足原模块名；运行需要Pillow。
