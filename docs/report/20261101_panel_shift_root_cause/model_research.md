# 模型与服务配置核对

日期：2026-11-01。此次只读检查已运行的模型容器、既有响应与官方资料；没有发起模型推理、修改服务或接触机器人。机器侧证据保存在 [model_service_audit.json](model_service_audit.json)。

## 已确认的身份与配置

- `qwen38` 是服务别名。实际加载 `Inferact/Qwen3.8-27B-NVFP4`，权重与 tokenizer 均固定 revision `6128240ebaf4eaa7bad2b3d1c72c37d677c5f462`，是 27B 基础模型的第三方量化版本。
- 容器中实际版本为 vLLM 0.28.0、Transformers 5.15.1、PyTorch 2.13.0+cu130、XGrammar 0.2.3，与既有响应的 `system_fingerprint=vllm-0.28.0-8ea1b29c` 一致。
- `language_model_only=false`；架构配置为 `Qwen3_5ForConditionalGeneration`。Qwen 官方说明 3.8 建立在 3.5 架构上，因此这个类型名本身不是加载错模型的证据。[Qwen3.8-27B 官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)
- 量化配置是 ModelOpt NVFP4、部分 Linear 权重和激活均 4 bit；忽略列表明确包含 `model.visual*`。不能声称视觉塔也被量化成 4 bit，也不能据此断言多模态判断完全不受语言侧量化影响。
- 当前 preprocessor 设置 `shortest_edge=65536`、`longest_edge=16777216`，这里是像素面积阈值；patch 为 16、merge 为 2。启动参数没有 `mm_processor_kwargs` 或额外小分辨率上限。它不支持“服务把所有图片强制压到很低分辨率”的推断；最终有效图像尺寸仍应逐输入检查。
- 服务使用仓库自带 chat template，关闭 thinking 的默认值也确实通过启动参数设置。既有 on/off 回放中，开启 thinking 后出现了实际 reasoning token，排除了开关完全没有生效这一解释。
- 本机 `qwen3_coder` 与 `qwen3_xml` 都注册为同一个 `Qwen3EngineToolParser`，使用 `qwen_3_coder` 结构标签；模板也输出 `<tool_call>/<function=...>/<parameter=...>`。不能仅因为 parser 名字中有 coder 就判定格式不兼容。

Inferact 仓库明确标注来源为 Qwen3.8-27B；其简短模型卡没有提供这项机器人任务的验证。[量化仓库](https://huggingface.co/Inferact/Qwen3.8-27B-NVFP4)

## “模型能力应该够”应怎样判断

官方 27B 模型支持视觉、工具调用和多步 Agent 工作，但公开成绩不能直接换成这个机体的控制能力。例如官方列出的 ERQA 成绩为 65.5；ERQA 的题目是图像与文字组成的多选问答，不是向指定机器人输出毫米位移后闭环完成按键的成功率。识别面板、判断按钮与指尖的关系、把图像关系转换为本机运动、选择步长，是不同的子能力。[27B 官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)、[ERQA 官方任务定义](https://github.com/embodiedreasoning/ERQA)

因此，不能用 Qwen3.8-Max 的宣传或其它平台的 Agent benchmark 为本机 27B NVFP4 的可靠性作保证；也不能因当前失败就断言这个基础模型绝对做不到。项目已有“补入正确当前视觉事实后输出 -Y”的实验，比泛化 benchmark 更直接证明它至少具备相关的方向选择子能力。

## 尚未验证、但可以具体检查的因素

| 因素 | 当前证据与下一步 |
|---|---|
| 语言侧量化 | NVIDIA 明确说明低精度方案和标定方法会影响精度；本项目没有同帧同提示的 BF16/FP8 对照。只能列为候选因素，不能直接归因于 NVFP4。可以在不修改现有服务的前提下，以同一 27B 的 FP8/BF16 服务作对照。 |
| 采样配置 | 项目使用 temperature=0；官方 non-thinking 推荐 temperature=0.7、top_p=0.8、top_k=20、presence_penalty=1.5。当前尚未测该组合。它可能改变重复偏好，也可能增加波动，应只用保存观测、多次采样比较；不能直接更改真机默认值后假定修复。 |
| 工具输出限制 | `tool_choice=required` 要求调用工具，不等于要求执行运动；项目还有重新观察、完成和无法继续工具。它可能影响回答组织，但不是代码强制前推。要比较自由判读、结构化判读与动作调用，必须保持图片与问题语义尽量一致。 |
| 多图与历史 | 支持多图、接受请求和上下文没有超窗，只证明输入可处理，不保证当前证据压过旧理由。项目已经观察到去掉过程历史后动作明显改变，优先级高于尚无本场景证据的量化猜测。 |
| 字段缺失 | 必填字段曾被漏掉是已记录的格式问题，但不能把最新 vLLM 文档直接套到本机 0.28.0。当前 XGrammar 的 `strict=None` 路径仍保留 schema，显式 `False` 才放宽；缺字段的具体原因还需检查实际 grammar 与解析输出，不能贸然归因于未设置 `strict:true`。应用层校验仍必要。 |

量化机制依据：[NVIDIA Model Optimizer PTQ recipes](https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt_recipes/ptq.md)。采样依据：[Qwen3.8-27B 官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)。工具契约与版本差异依据：[vLLM Tool Calling](https://docs.vllm.ai/en/latest/features/tool_calling/)，实际本机行为优先采用本次容器源码核对。

结论：目前没有发现“加载成纯文本模型”“视觉塔被统一 4 bit 量化”“parser 别名选错”“thinking 开关失效”等服务层证据。最有本场景支持的调查方向仍是当前视觉事实不稳定，以及这些事实进入动作选择时被历史和任务先验覆盖；量化和官方采样设置应作为后续受控对照，而不是先定为根因。
