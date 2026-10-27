"""Explicit Qwen3-VL joint8 -> Qwen3.5 joint10 warm start; never a resume."""
import re

import torch

from starVLA.training.train_s1b_real import ADAPTED_KEYS, COMPONENTS, select_pretrained_state


MODE = "qwen3_5_from_qwen3_vl_joint8"
VLM_PREFIX = "qwen_vl_interface."


def select_qwen3_5_state(source, target):
    """Retain the pretrained Qwen3.5; approve only known 2048->2560 interfaces.

    All non-VLM names must match. Reuse the existing joint8->joint10 checks
    after removing the explicitly resized VLM conditioning projections.
    """
    if not isinstance(source, dict) or not source:
        raise ValueError("Missing pretrained tensors")
    if all(key.startswith("module.") for key in source):
        source = {key[7:]: value for key, value in source.items()}
    if any(not isinstance(value, torch.Tensor) for value in source.values()):
        raise ValueError("Expected a plain tensor state_dict")
    for state in (source, target):
        if any(not any(key.startswith(prefix) for key in state) for prefix in COMPONENTS):
            raise ValueError("Expected complete VLM, encoder, predictor and action components")
    old = {key: value for key, value in source.items() if not key.startswith(VLM_PREFIX)}
    new = {key: value for key, value in target.items() if not key.startswith(VLM_PREFIX)}
    if set(old) != set(new):
        raise ValueError("Non-VLM tensor keys differ from complete architecture")
    resized = []
    for key in list(new):
        # DiT interleaves self attention and cross attention. Only weights with
        # the old VLM width qualify; self-attention projections remain strict.
        cross = re.fullmatch(r"action_model\.model\.transformer_blocks\.\d+\.attn1\.to_[kv]\.weight", key)
        predictor = key in {"vj_predictor.action_encoder.weight", "vj_predictor.state_encoder.weight",
                            "vj_predictor.extrinsics_encoder.weight"}
        if (cross or predictor) and old[key].shape != new[key].shape:
            offset = 1 if key == "vj_predictor.extrinsics_encoder.weight" else 0
            expected_old = (new[key].shape[0], 2048 - offset)
            expected_new = (new[key].shape[0], 2560 - offset)
            if tuple(old[key].shape) != expected_old or tuple(new[key].shape) != expected_new:
                raise ValueError(f"Unapproved conditioning shape: {key}")
            resized.append(dict(key=key, source_shape=list(old[key].shape), target_shape=list(new[key].shape),
                                reason="Qwen3.5 conditioning width 2048 -> 2560"))
            del old[key], new[key]
    if not resized:
        raise ValueError("Migration requires 2048 -> 2560 conditioning projections")
    loaded, report = select_pretrained_state(old, new)
    report.update(head_initialization=MODE,
                  missing_keys=sorted(key for key in target if key.startswith(VLM_PREFIX)),
                  skipped_source_vlm_keys=sorted(key for key in source if key.startswith(VLM_PREFIX)),
                  adapted_keys=report["adapted_keys"] + resized,
                  backbone_initialization="Qwen3.5 pretrained weights; new action tokens",
                  optimizer_restored=False)
    if {item["key"] for item in report["adapted_keys"] if item["key"] in ADAPTED_KEYS} != ADAPTED_KEYS:
        raise ValueError("Migration requires all five joint8 -> joint10 coordinate interfaces")
    return loaded, report
