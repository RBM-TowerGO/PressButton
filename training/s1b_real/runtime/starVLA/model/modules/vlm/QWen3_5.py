"""Qwen3.5 multimodal features for continuous-action VLA-JEPA."""
from types import SimpleNamespace

import torch
from torch import nn
from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration


class Qwen3_5Interface(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        settings = config.framework.qwenvl
        model_id = settings.base_vlm
        backbone_config = AutoConfig.from_pretrained(model_id)
        if backbone_config.model_type != "qwen3_5":
            raise ValueError("Qwen3.5 requires a full multimodal qwen3_5 checkpoint")
        width = backbone_config.text_config.hidden_size
        if settings.get("vl_hidden_dim", width) != width:
            raise ValueError("vl_hidden_dim differs from Qwen3.5 text hidden_size")
        options = dict(attn_implementation=settings.get("attn_implementation", "sdpa"),
                       dtype=torch.bfloat16)
        device = settings.get("device_map", "cuda")
        if settings.get("init_from_config", False):
            self.model = Qwen3_5ForConditionalGeneration._from_config(backbone_config, **options).to(device)
        else:
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_id, config=backbone_config, device_map=device, **options)
        self.model.config.hidden_size = width
        self.model.config.use_cache = False
        self.model.config.text_config.use_cache = False
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.tokenizer.padding_side = "left"

    def forward(self, **kwargs):
        device = self.model.device.type
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            if self.config.framework.qwenvl.get("features_only", True):
                # Preserve gradients through the backbone without materializing
                # [batch, sequence, 248320] vocabulary logits or every layer.
                kwargs.pop("output_hidden_states", None)
                kwargs.pop("return_dict", None)
                kwargs["use_cache"] = False
                outputs = self.model.model(**kwargs, output_hidden_states=False, return_dict=True)
                return SimpleNamespace(hidden_states=(outputs.last_hidden_state,))
            return self.model(**kwargs)

    def build_qwenvl_inputs(self, images, instructions, solutions=None,
                           prompt_replace_dict=None, prompt_template=None, **kwargs):
        if solutions is not None:
            raise NotImplementedError("Qwen3.5 integration supports continuous actions, not FAST token labels")
        if len(images) != len(instructions) or not images:
            raise ValueError("Expected equal nonempty batches of images and instructions")
        if prompt_template is None:
            prompt_template = self.config.datasets.vla_data.get("CoT_prompt", "{instruction}")
        messages = []
        for views, instruction in zip(images, instructions):
            prompt = prompt_template.replace("{instruction}", instruction)
            for key, value in (prompt_replace_dict or {}).items():
                prompt = prompt.replace(key, value)
            content = [{"type": "image", "image": image} for image in views]
            content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, padding=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=True, return_tensors="pt")
        return inputs.to(self.model.device)
