# Qwen3.8 右腕／夹爪视角识别与画框

此目录为最初含贴纸／红标类别的探索记录。用户随后明确只要手指、物理指尖点、按钮和面板；**当前结果请看 [新版评估](../20261103_qwen38_wrist_fingertips/README.md)**。

日期：2026-11-03。使用已保存的真实右腕相机 JPEG，逐张调用当前模型服务；没有采集新帧或执行机器人动作。

查看 [模型画框总览](annotated_contact_sheet.jpg) 和 [原图总览](original_contact_sheet.jpg)。每帧原图和标注并排图位于下表。框与文字都是模型预测，未用人工坐标修正。

## 相机与样本

相机话题为 `/astribot_camera/right_wrist_rgbd/color_compress/compressed`，对应 `right.jpg`。映射见 [相机代码](../../../scripts/s1b_camera_view.py) 与 [接口说明](../../astribot/planar_control/interfaces.md)。所选 8 张原图均为 640×360，覆盖亮灯遮挡、小目标、红标、白箭头、上边界截断和按钮缺失。样本由助手按场景变化选择，非随机抽样；部分来自同一运行。

| 样本 | 选择理由 | 对照图 | 原始来源 |
|---|---|---|---|
| 01_lit_occluded | 近距亮灯按钮，被右侧夹指部分遮挡 | [原图／预测](cases/01_lit_occluded/comparison.png) | [来源](../../../docs/report/20261101_panel_shift_visual_action_mismatch/images/success_step10_right.jpg) |
| 02_small_white_arrow | 小而完整的按钮、白箭头贴纸 | [原图／预测](cases/02_small_white_arrow/comparison.png) | [来源](../../../docs/report/20261102_arrow_progress_validation/images/white_arrow_right.jpg) |
| 03_middle_red_marker | 按钮位于中部，红标记可见 | [原图／预测](cases/03_middle_red_marker/comparison.png) | [来源](../../../docs/report/20261102_arrow_progress_validation/images/round_10_right.jpg) |
| 04_low_red_marker | 按钮接近指端高度，红标记可见 | [原图／预测](cases/04_low_red_marker/comparison.png) | [来源](../../../docs/report/20261102_arrow_progress_validation/images/round_19_right.jpg) |
| 05_large_unmarked | 较大的完整按钮，没有白箭头/红标记 | [原图／预测](cases/05_large_unmarked/comparison.png) | [来源](../../../docs/report/20261102_live_alignment_progress/premature_press/round10_before_right.jpg) |
| 06_top_clipped | 按钮在上边界仅部分可见，红标记可见 | [原图／预测](cases/06_top_clipped/comparison.png) | [来源](../../../docs/report/20261102_red_marker_direct_descent/before_first_lower_right.jpg) |
| 07_button_absent | 面板和按钮均出画，检查误检 | [原图／预测](cases/07_button_absent/comparison.png) | [来源](../../../docs/report/20261102_red_marker_direct_descent/after_125mm_lower_right.jpg) |
| 08_middle_white_arrow | 白箭头场景，按钮比02更低 | [原图／预测](cases/08_middle_white_arrow/comparison.png) | [来源](../../../docs/report/20261102_view_recovery/model_replay/round_08/right.jpg) |

## 请求方式

- 当前 `/v1/models` 返回 `qwen38` → `Inferact/Qwen3.8-27B-NVFP4`；见 [本次服务身份](models.json)。本次未重新核验权重 revision。
- 每张图独立首答，`temperature=0`、`enable_thinking=false`、`max_tokens=1536`。没有示范、历史动作或其他相机图片。
- 类别为 `panel`（面板）、`button`（含外圈的圆形按钮）、`image_left_finger` / `image_right_finger`（图像左右夹指）、`white_arrow_sticker`（白箭头贴纸）、`red_marker`（红标记）。这是指定类别检测，不是无约束的开放类别识别。
- 不存在或无法辨识的类别应省略；框只包围可见范围。输出还包含可见性、场景描述和按钮图标／亮灯外观。
- 输入保持 JPEG 原始字节，未缩放、裁剪或加框；选图说明、样本名称和人工复核未传入模型。
- 坐标为 `xyxy`、0–1000，采用图像边缘约定：`x_px=x*640/1000`、`y_px=y*360/1000`。不交换、不裁剪、不校正模型框；像素坐标保存在每帧 `result.json`。
- 每图只发一次推理请求，错误也保留；画框程序只绘制有效返回。格式校验不等于识别正确。

## 结果与复核

初始轮次8次请求全部返回可解析结果，总输入8576 token、输出2791 token，推理请求累计263.7秒。该轮次没有指尖点，不是用户修正后的最终交付。旧目视记录保存在 [visual_review.json](visual_review.json)，原始响应及图片全部保留。

## 归档与复现

- [plan.json](plan.json)：推理前固定的样本、选择理由、输入尺寸、哈希及协议。
- [samples.json](samples.json)：可直接复用的原始图片清单。
- [summary.json](summary.json)：逐帧检测、token、耗时和有效性。
- 每帧 `cases/<id>/request.readable.json` 是实际请求的可读归档；将其中 `original.jpg` 读为原始字节，再替换为 `data:image/jpeg;base64,<原字节Base64>`，即可恢复实际图像输入。
- 每帧 `response.body` / `response.http.json` 保存原始 HTTP 正文和状态；`response.json` 为解析后的模型响应，`result.json` 为格式校验和像素坐标。
- 每帧 `original.jpg` 为源图字节副本，`annotated.png` 为模型框，`comparison.png` 为左原图／右模型框。可读英文标签编号对应模型返回对象顺序；中文描述见 `result.json`。
- [脚本快照](script_used.py)、[准备检查](prepare_checks.json) 用于核对本次证据。

当前主脚本已按用户新要求改为手指、指尖、按钮和面板；复现当前任务请使用[新版报告中的命令](../20261103_qwen38_wrist_fingertips/README.md)。旧轮次的当时脚本保留在 `script_used.py`，精确提示和响应保留在每帧目录中。
