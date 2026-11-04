# 告知夹爪已合拢后的末端定位

2026-11-03。按用户要求，明确告诉Qwen3.8图片中的夹爪已经闭合、两根手指合拢，要求它标出合拢后整个夹爪的物理末端。沿用上一轮8张闭合夹爪真实右腕原图，每张独立请求；没有把上一轮预测或人工位置提示给模型。

查看 [标注总览](annotated_contact_sheet.jpg)、[末端放大对照](end_details.jpg) 和 [原图总览](original_contact_sheet.jpg)。**青框是模型给出的末端区域，粉色圆圈十字是模型给出的代表中心点**；不再分别标左右指尖，也不标按钮。

## 实际提示

> 这张图中有一个已经闭合、两根手指已经合拢的机械夹爪。请根据当前原图，标记合拢后夹爪的物理末端：远离腕部/基部、朝向前方场景的两指最前端区域。将合拢后的末端作为一个整体，给出覆盖该末端可见部分的紧致框，以及末端表面上的一个代表中心点。不要标整根夹爪的中心、关节、螺丝或电梯按钮。只标夹爪末端，不需要标其他物体。如果末端不可见或无法确定，可以返回not_visible/uncertain和null，不要猜测坐标。

完整系统提示、坐标约定和中文用户提示见 [prompt.txt](prompt.txt)。输入未提供“末端在哪个像素位置”等答案提示。

## 结果

本轮8次请求全部返回合法的一框一点；输入8,544 token、输出935 token，请求累计89.6秒。但**明确告知夹爪合拢后，末端定位仍不稳定**。

助手目视复核：02/04/08较接近合拢后的短指端；06点接近末端，但框混入按钮圆环；01/03/05/07把后方连接结构或紧固件区域当作末端。8/8格式合法、所有点都位于各自预测框内，均不能当作末端识别成功。

| 图 | 模型点像素(x,y) | 助手目视复核 |
|---|---|---|
| 01 | (326.40, 302.40) | 框包含大段连接结构，点落在较靠后的夹爪结构上，没有定位最前端。 |
| 02 | (343.04, 266.40) | 框与点接近合拢后的两块短指端及中缝，粗定位较合理，框略宽。 |
| 03 | (393.60, 306.00) | 大框几乎覆盖大部分夹爪，点偏到右侧后方连接结构，未定位末端。 |
| 04 | (333.44, 265.68) | 框与点接近两块短指端，粗定位较合理，框略偏左。 |
| 05 | (393.60, 324.00) | 框漏掉最前上方指端，点落在右侧较靠后银色紧固件附近，定位错误。 |
| 06 | (350.72, 268.20) | 点接近物理末端，但框还包含后方按钮发亮圆环，未分清末端与按钮。 |
| 07 | (390.40, 306.00) | 大框包含大段连接结构，点偏到右侧后部，未定位最前端。 |
| 08 | (333.44, 261.00) | 框与点位于前端短指块附近，粗定位较合理，点略偏左。 |

完整坐标见 [predictions_pixels.csv](predictions_pixels.csv)，复核见 [visual_review.json](visual_review.json)。这是这8张图的定性目视判断，没有端面中心标定或人工坐标真值，不计算准确率或像素定位误差。新提示与任务同时发生变化，没有同任务无“闭合”提示的受控对照，不能据此量化这句提示带来的提升。


## 逐图查看

| 图 | 原图／模型结果 | 原图来源 |
|---|---|---|
| 01 | [对照图](cases/01_small_upper/comparison.png) | [原图](cases/01_small_upper/original.jpg) |
| 02 | [对照图](cases/02_large_left/comparison.png) | [原图](cases/02_large_left/original.jpg) |
| 03 | [对照图](cases/03_middle_complete/comparison.png) | [原图](cases/03_middle_complete/original.jpg) |
| 04 | [对照图](cases/04_low_right/comparison.png) | [原图](cases/04_low_right/original.jpg) |
| 05 | [对照图](cases/05_occluded_unlit/comparison.png) | [原图](cases/05_occluded_unlit/original.jpg) |
| 06 | [对照图](cases/06_occluded_lit/comparison.png) | [原图](cases/06_occluded_lit/original.jpg) |
| 07 | [对照图](cases/07_top_clipped/comparison.png) | [原图](cases/07_top_clipped/original.jpg) |
| 08 | [对照图](cases/08_button_not_visible/comparison.png) | [原图](cases/08_button_not_visible/original.jpg) |

## 方法与记录

- 当前服务身份保存在 [models.json](models.json)，别名qwen38，模型为Inferact/Qwen3.8-27B-NVFP4；temperature=0，thinking关闭，max_tokens=768。
- 每帧只取一次独立首答，无历史图片、模型旧结果或机器人动作信息。原始640×360 JPEG按字节原样传入，没有裁剪、缩放或预先标记。
- `located`需返回一个`end_box`和一个`end_point`；看不见或不确定可返回null。框和点均使用0–1000归一化图像边缘坐标，x向右、y向下，按`x*640/1000`、`y*360/1000`换算。原答保留，不修正、不挪点，也不把模型框中心自动当作模型点。
- `point_inside_own_box`只检查点与模型自己的框是否一致，不代表视觉正确。二维末端点并非手眼标定或真实接触点。
- 原图来源及闭合依据沿用[上一轮闭合图片报告](../20261103_qwen38_closed_gripper_buttons/README.md)，完整路径、原图哈希和闭合证据也保留在 [plan.json](plan.json) 与 [samples.json](samples.json)。两张有实测闭合刻度约97.7，另六张为目视合拢、没有同步夹爪反馈。
- [summary.json](summary.json) 保存模型输出、token和耗时。每帧的`request.readable.json`、`response.body`、`response.http.json`、`response.json`、`result.json`保留实际请求、原始HTTP/响应和像素坐标。可读请求中的`original.jpg`按原字节转为`data:image/jpeg;base64,...`即可恢复图像输入。
- [准备检查](prepare_checks.json) 与 [完成检查](postrun_checks.json) 核对输入、协议、原答和坐标转换。脚本与辅助模块分别归档于`script_used.py`、`support_snapshot.py`。
- 放大图使用固定区域作展示，不改预测，也没有作为模型输入。没有连接或控制机器人。

## 复现

保持模型SSH隧道运行：

```bash
ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:18538:127.0.0.1:8000 Thor
```

在仓库根目录运行，输出目录必须未存在：

```bash
/home/js/robot/VLA/PressButton/.conda/envs/pressb/bin/python \
  scripts/evaluate_qwen38_closed_gripper_tip.py \
  --manifest docs/report/20261103_qwen38_closed_gripper_tip/samples.json \
  --output outputs/qwen38_closed_tip_repeat \
  --model-base http://127.0.0.1:18538/v1
```

也可先`--prepare-only`后`--execute-prepared`。在独立目录运行归档脚本时，应将`support_snapshot.py`复制为同目录的`evaluate_qwen38_wrist_boxes.py`；需要Pillow。目视评论不由推理脚本自动生成。

末端放大图可用同目录 `render_end_details.py` 独立重建；固定裁剪x=240..440、y=220..330后放大3倍，仅作展示，完整预测范围应以全图为准。
