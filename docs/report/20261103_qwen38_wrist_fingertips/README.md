# Qwen3.8：夹爪手指、指尖、按钮和面板

日期：2026-11-03。按用户修正后的范围，只识别手指、手指物理尖端、按钮和面板；不设贴纸／红标检测类别。使用8张已保存的真实右腕视角原图重新独立调用Qwen3.8，框和指尖点均为本轮模型原答，未人工调整。

查看 [标注总览](annotated_contact_sheet.jpg)、[指尖放大对照](fingertip_details.jpg) 与 [原图总览](original_contact_sheet.jpg)。每张 [cases/](cases/) 下的 `comparison.png` 为左原图／右模型标注。

颜色：黄框=面板，绿框=按钮，蓝框=图像左手指，粉框=图像右手指；同色圆圈十字 `L tip` / `R tip` 为左右手指物理尖端预测。左右按图像方位，不代表左右机械臂。照片本身有时带贴纸，但本轮没有贴纸框。

## 样本与相机

真实相机话题：`/astribot_camera/right_wrist_rgbd/color_compress/compressed`。见 [相机映射代码](../../../scripts/s1b_camera_view.py) 和 [接口文档](../../astribot/planar_control/interfaces.md)。8张原图均640×360，按场景覆盖选择，包含完整按钮、亮灯遮挡、顶部裁切及无按钮负例；并非随机抽样，部分帧来自同一运行。

| 编号 | 场景 | 原图／本轮预测 | 原始来源 |
|---|---|---|---|
| 01 | 亮灯按钮被右指遮挡 | [对照图](cases/01_lit_occluded/comparison.png) | [源图](../../../docs/report/20261101_panel_shift_visual_action_mismatch/images/success_step10_right.jpg) |
| 02 | 小而完整的按钮 | [对照图](cases/02_small_white_arrow/comparison.png) | [源图](../../../docs/report/20261102_arrow_progress_validation/images/white_arrow_right.jpg) |
| 03 | 按钮在图像中部 | [对照图](cases/03_middle_red_marker/comparison.png) | [源图](../../../docs/report/20261102_arrow_progress_validation/images/round_10_right.jpg) |
| 04 | 按钮接近指尖高度 | [对照图](cases/04_low_red_marker/comparison.png) | [源图](../../../docs/report/20261102_arrow_progress_validation/images/round_19_right.jpg) |
| 05 | 较大的完整按钮 | [对照图](cases/05_large_unmarked/comparison.png) | [源图](../../../docs/report/20261102_live_alignment_progress/premature_press/round10_before_right.jpg) |
| 06 | 顶部仅露出部分按钮 | [对照图](cases/06_top_clipped/comparison.png) | [源图](../../../docs/report/20261102_red_marker_direct_descent/before_first_lower_right.jpg) |
| 07 | 面板和按钮均出画 | [对照图](cases/07_button_absent/comparison.png) | [源图](../../../docs/report/20261102_red_marker_direct_descent/after_125mm_lower_right.jpg) |
| 08 | 完整按钮，位置与02不同 | [对照图](cases/08_middle_white_arrow/comparison.png) | [源图](../../../docs/report/20261102_view_recovery/model_replay/round_08/right.jpg) |

## 方法

- 当前服务：`qwen38` → `Inferact/Qwen3.8-27B-NVFP4`，实时返回记录在 [models.json](models.json)；没有重新核验权重 revision。
- 一张原图、一次独立首答，`temperature=0`、`enable_thinking=false`、`max_tokens=1536`；无示范或动作历史，无纠错重试和结果挑选。
- 图片按原始JPEG字节传入，不缩放、不裁剪、不预先加框。样本文件名、选择理由和人工复核未发送给模型。
- `panel`、`button`、`image_left_finger`、`image_right_finger` 返回可见范围的 `bbox=[left,top,right,bottom]`。每根手指额外返回 `fingertip=[x,y]`；指尖无法可靠辨认时允许 `null`；面板和按钮的指尖字段必须为 `null`。
- 指尖定义是朝向场景、远离腕部／基部的物理末端表面的代表中心点，不是整根手指的中心、螺丝或基部。该二维视觉点未经接触点标定。
- 坐标0–1000，左上角为原点，x向右、y向下；使用图像边缘约定，`x_px=x*640/1000`、`y_px=y*360/1000`。模型原框和点不裁剪、不排序、不修正。
- `fingertip_inside_bbox` 只检查模型点是否落在自身预测框内，不会挪动点，也不代表视觉正确。
- 本次只访问模型，没有连接或控制机器人。每张图片只进行一次本轮推理。

## 结果与目视复核

本轮 **8次独立首答全部可解析，共29个框、16个指尖点**；只出现规定的四种物体类别。模型输入 9,816 token，输出 2,736 token，推理请求累计 258.5 秒。原图哈希、HTTP原始响应、坐标换算与脚本快照全部核对通过。**8/8格式合格不等于8/8识别准确。**

在这组样例中，7张可见按钮图都产生了按钮框；1张没有按钮的图没有误检按钮。16个指尖点均靠近相应手指前端，没有明显把整指中心、螺丝或贴纸中心当作指尖，但多数点集中于前端上缘，尚不能证明落在末端表面的中心。第05张仍漏检面板，部分按钮框偏移，出画面板的可见性也有误判。

建议结合 [指尖放大对照](fingertip_details.jpg) 查看点位：每行依次是左指原图／左指预测／右指原图／右指预测。固定区域3倍展示，没有改变任何预测，也未作为模型输入。可用 [像素坐标CSV](detections_pixels.csv) 对照原始坐标。

| 编号 | 框数／指尖点数 | 助手目视复核 |
|---|---|---|
| 01 | 4／2 | 四类检出；按钮被遮挡，本轮正确报 partial。两指尖点在前端附近。 |
| 02 | 4／2 | 四类检出；按钮框偏上、漏下缘；出画面板误报 full。两指尖点在前端附近。 |
| 03 | 4／2 | 四类检出；两指尖点贴近圆钝前端的上缘，未稳定定位到末端表面中心。 |
| 04 | 4／2 | 四类检出；面板出画却报 full；按钮框上下略紧；指尖点偏向上缘。 |
| 05 | 3／2 | 漏检清楚可见的面板；检出按钮和双指，按钮框偏上漏下缘；指尖点在前端上部。 |
| 06 | 4／2 | 面板、局部按钮和双指均检出，partial判断合理；按钮框略偏左，指尖点在前端上部。 |
| 07 | 2／2 | 正确只报双指，未虚构按钮或面板；两指尖点在前端附近。 |
| 08 | 4／2 | 四类检出；出画面板误报 full，按钮框略偏上；两指尖点贴近前端上缘。 |

以上是小规模、选取样例的定性审查，无人工标注真值，不计算准确率、mAP或指尖像素误差。即使16个点都位于各自预测框内，也不能作为指尖正确或物理接触准确的证明。完整目视记录见 [visual_review.json](visual_review.json)。


## 归档和复现

[plan.json](plan.json) 固定输入、哈希和协议，[summary.json](summary.json) 保存每帧结果、耗时和token，[samples.json](samples.json) 是可复用清单。[脚本快照](script_used.py)、[准备检查](prepare_checks.json)、[完成检查](postrun_checks.json) 记录实现与核验。

每帧保存 `original.jpg`、`request.readable.json`、原始HTTP正文 `response.body` 和状态 `response.http.json`、模型完整响应 `response.json`、含像素坐标的 `result.json`、标注 `annotated.png` 和对照 `comparison.png`。可读请求中 `original.jpg` 替换成该文件原始字节的 `data:image/jpeg;base64,...` 即可恢复输入。

在一个终端建立模型隧道：

```bash
ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:18538:127.0.0.1:8000 Thor
```

在另一个终端、仓库根目录运行，输出目录必须未存在：

```bash
/home/js/robot/VLA/PressButton/.conda/envs/pressb/bin/python \
  scripts/evaluate_qwen38_wrist_boxes.py \
  --manifest docs/report/20261103_qwen38_wrist_fingertips/samples.json \
  --output outputs/qwen38_wrist_fingertips_repeat \
  --model-base http://127.0.0.1:18538/v1
```

也可先加 `--prepare-only`，再用同一输出目录加 `--execute-prepared` 执行；准备后脚本、请求或图片被修改会拒绝运行。复现脚本生成模型结果及图片，报告中的目视评论属于本次助手事后审查。

指尖放大图可独立重建：`python docs/report/20261103_qwen38_wrist_fingertips/render_fingertip_details.py`，需要Pillow；不会发起模型请求。
