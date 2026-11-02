# 25 条保存图片诊断请求

仅调用模型，无机器人客户端实例、无机器人动作。全部条件在模型调用前写入 `plan.json`；`script_used.py` 是准备和执行时的脚本快照。两幅原始腕图复制到 `source_images/`，原文件及副本的 SHA256 已核对。

- `plan.json`：25 条预声明条件、助手近似参考坐标及限制。第 20 条明确依赖前两条实际预测坐标，未编造坐标。
- `stimulus_manifest.json`、`figures/`：原位灰底、平移灰底、缩放、裁剪和标签刺激。除上一轮原始小标签 JPEG 外，本轮刺激统一 PNG，未新增 JPEG 损失。
- `cases/*/request.readable.json`：实际请求；图片替换成可读相对路径，`request_images.json` 保存 MIME、哈希和恢复信息。将路径指向字节按原 MIME 重新 Base64 编码，可还原实际请求。
- `cases/*/response.json`：原始首答，包括正文、工具调用、usage、finish_reason；没有修正或重复选答。
- `cases/*/result.json`：按预声明格式校验及连续像素误差，不将手工框当标定真值。
- `summary.json`：逐条件结果及 token、耗时总计。
- `prepare_checks.json`、`postrun_checks.json`：刺激、归档、依赖坐标及输出一致性检查。
- `diagnostic_overlays/`：回答后生成的 A/B 定位叠层，用于人工检查；没有发给模型，不属于测试刺激。黄框及十字为助手近似参考，青圆为单对象预测，粉方框为联合预测。

实际使用 `qwen38`、temperature=0、关闭 thinking、max_tokens=384。25 次请求全部完成，无修正、无跳过；23 条满足预声明输出格式。总输入 19,167 token，输出 1,656 token，逐请求耗时之和约 163.5 秒。模型报告的额外 reasoning token 均为 0，但两次仍在普通 content 中输出了分析文字。

两条格式异常必须保留：`large_photo_reverse` 输出了正文及关系正确的工具参数，因额外正文不符合协议而未接受；`A_translated_ROI_relation` 输出达到 384 token 后截断，没有工具调用，不能计为有效关系答案。两条均没有补请求。

运行同样实验时需使用新的空输出目录，避免覆盖这次记录：

```bash
.conda/envs/pressb/bin/python scripts/diagnose_agent_spatial_binding.py \
  --output /tmp/spatial_binding_repeat --prepare-only
.conda/envs/pressb/bin/python scripts/diagnose_agent_spatial_binding.py \
  --output /tmp/spatial_binding_repeat --execute-prepared
```

默认只建立到 Thor 的模型端口隧道（本地 18024）。每条是独立消息，不含历史动作、案例或机器人接口。新实验结果不能视为真机完成率；跨请求定位正确也不能证明另一个关系请求内部已正确定位。

## 看到初始结果后的补充检查

追加结果独立保存于 [`../supplement/`](../supplement/)，没有覆盖以上 25 条结果。先固定 8 条补充条件，再于任何补充请求发出之前加入 2 条等面积遮挡对照；`initial_8_plan.json`、`plan_extension.json` 和最终 `plan.json` 保留了这个时序。补充脚本为 `scripts/diagnose_agent_spatial_binding_supplement.py`。

10 次均为独立首答、不修正，全部满足格式。前 4 次直接恢复原请求，完整 payload 与初始请求一致。补充输入 7,337 token、输出 455 token，请求耗时之和约 45.6 秒；模型报告的额外 reasoning token 仍全为 0。

| 补充条件 | 实际结果 | 说明 |
|---|---|---|
| 交换 T/P 后直接问关系，再重复两次 | 均上／左 | 两次继续错误，正确应为下／右 |
| 原大标签关系重复一次 | 上／左 | 正确，与初次一致 |
| A 原图关系重复一次 | 上／右 | 继续错误，与初次一致 |
| 交换 T/P，同时输出两点及关系 | 下／右 | 正确，且关系与自己的两点一致 |
| 同一交换图，问青点相对品红点 | 下／右 | 正确；没有使用字母名称 |
| 交换标签的纯灰底图 | 下／右 | 正确 |
| 只给已有交换标签预测坐标 | 下／右 | 正确；精确使用前批实际预测值 |
| 遮掉原图左下方另一根手指 | 上／右 | 仍错误 |
| 遮等面积左上背景，保留两根手指 | 上／左 | 正确 |

两个遮挡都使用 27,500 像素的灰色矩形，按钮及带箭头指端完整保留，未遮挡部分逐像素不变。结果不支持“仅仅误用了另一根手指”这一简单解释；它进一步表明关系输出会受到图像上下文和提问组织影响，但不能据此定位到某个内部注意机制。联合输出成功与纯色参照成功也不是对所有实际画面的可靠性保证。

补充的原请求、响应、usage、finish_reason、连续定位误差和校验分别位于 `supplement/cases/`、`supplement/summary.json` 与 `supplement/postrun_checks.json`。
