# 第 1、4、8 轮保存画面的取景与定位检查

本次仅连接 `qwen38`，没有创建机器人客户端，没有发送任何运动命令。使用运行 `20261102_120741_0ca36af5` 的第 1、4、8 轮当前头部和腕部原图，检查新取景接口能否定位按钮、白箭头所指物理指端，并依据定位计算图像平面的上下关系。没有开启 thinking，没有加入案例、旧动作理由或旧复核结论。

**结论：三帧都正确指出头部视野不可用，因此这些观测不能放行手部对齐。腕图中能够找到按钮和标记指端的大致区域，但按钮框仍有明显偏差；三次均将接触端标为 `partial`，正式上下关系均为 `unknown`。没有再次输出错误的“按钮在下方”，但也不能将这三次算作上下关系判断通过。**

## 图片与定位

原图和定位叠层分别保存。叠层绿色框是模型给出的按钮框，橙色框是模型给出的箭头贴纸框，青色十字是模型给出的物理指端。图上标注不是标定真值；没有把叠层图重新提供给模型。

| 原运行轮次 | 头部判断 | 腕部定位检查 | 正式上下关系 | 原图／模型定位 |
|---|---|---|---|---|
| 1 | 按钮、指端不在视野内，`usable_head=false` | 按钮框向左上偏移并碰到图像顶边，未覆盖完整按钮；指端点靠近右指前端但偏左下；箭头框偏上 | `unknown` | [腕图原图](model_replay/round_01/right.jpg)、[定位叠层](model_replay/round_01/right_localization.png)、[头图](model_replay/round_01/head.jpg) |
| 4 | 按钮、指端不在视野内，`usable_head=false` | 按钮框偏左，未覆盖右边缘；指端点位于白箭头指示的右指前端附近；贴纸框不完整 | `unknown` | [腕图原图](model_replay/round_04/right.jpg)、[定位叠层](model_replay/round_04/right_localization.png)、[头图](model_replay/round_04/head.jpg) |
| 8 | 按钮、指端不在视野内，`usable_head=false` | 按钮框偏左，右半按钮超出框外；指端点在右指前端附近；箭头框偏上 | `unknown` | [腕图原图](model_replay/round_08/right.jpg)、[定位叠层](model_replay/round_08/right_localization.png)、[头图](model_replay/round_08/head.jpg) |

逐帧查看原图，按钮中心都明显位于箭头所指指端的上方偏左。这是对保存图片的助手复核，不是新增现场人员确认，也不是接触点标定。近似位置和逐帧说明见 [visual_review.json](model_replay/visual_review.json)。

模型给出的坐标本身也落在“按钮中心位于指端上方偏左”这一侧，但它同时报告指端只部分可见。程序按约定在 `contact_visibility=partial` 时保持关系未知，没有忽略这一不确定性后强行生成结论。未为通过检查修改输出或放宽可见性条件。

## 实际调用与开销

| 原运行轮次 | 输入图片／次 | 请求次数 | 格式修正次数 | 累计输入 token | 累计输出 token | 累计请求及校验耗时 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 2 | 1 | 5,832 | 459 | 44.61 s |
| 4 | 2 | 1 | 0 | 2,765 | 286 | 27.35 s |
| 8 | 2 | 1 | 0 | 2,764 | 291 | 27.87 s |
| 合计 | — | 4 | 1 | 11,361 | 1,036 | 99.82 s |

第 1 帧首次响应将视角字段错误地放到顶层，缺少合法的 `head`／`wrist` 对象，校验报 `invalid_inspect_view_fields`。已有修正流程随后返回合法结构。该错误与图像判断的准确性分开记录；其余两帧首次响应结构合法。全部调用使用 `temperature=0`、`enable_thinking=false`、`max_tokens=512`，没有因截断而修正。

## 适用范围与剩余问题

- 新接口能够暴露定位框、物理指端及可见性，不再只保存一句不可审查的“在上方／下方”。对这三帧，取景判断可以阻止在头部缺失关键目标时跳过取景阶段。
- 本次尚未证明定位达到可用于精细对齐的可靠程度。尤其是按钮框偏左、贴纸框不完整，以及对可见指端持续给出 `partial`，仍需如实保留。
- 没有检查低头后新画面能否满足取景条件，也没有执行修复后的真实闭环。本次保存的三帧都来自原运行没有低头的状态。
- 本次是三帧小检查，不是统计性评测、thinking 对照或任务成功验证。格式合法、没有输出“下方”及正确识别头部缺景，均不能替代准确定位和成功按压证据。

## 归档与复现

[完整汇总](model_replay/validation.json) 记录实际 token、耗时、观测 ID 和代码哈希；每个 `round_XX/localization/` 目录保留完整请求、响应、修正请求、校验及有效判读。`source.json` 记录原始观测路径和图像哈希，原运行日志未被修改。运行时脚本和模块另存为 `replay_script_used.py`、`view_acquisition_used.py`；模型调用后仅将叠层图的标题移到图像外，避免遮挡第 1 帧靠近顶边的按钮，见 [渲染记录](model_replay/artifact_render.json)。

```bash
.conda/envs/pressb/bin/python scripts/validate_agent_view_recovery.py \
  --output outputs/view_recovery_check_new
```

脚本只向模型端口请求 `/v1/models` 和 `/v1/chat/completions`。输出目录必须不存在，防止覆盖已有检查。
