from copy import deepcopy
import os

import pytest
import torch
from omegaconf import OmegaConf

from starVLA.training.qwen3_5_transfer import MODE, select_qwen3_5_state
from starVLA.training.train_s1b_real import training_contract


def states():
    source = {
        "qwen_vl_interface.old_backbone": torch.ones(2),
        "vj_encoder.weight": torch.ones(2, 2),
        "vj_predictor.predictor_embed.weight": torch.ones(2, 2),
        "action_model.state_encoder.layer1.weight": torch.ones(4, 8),
        "action_model.state_encoder.layer1.bias": torch.ones(4),
        "action_model.action_encoder.layer1.weight": torch.ones(4, 8),
        "action_model.action_decoder.layer2.weight": torch.ones(8, 4),
        "action_model.action_decoder.layer2.bias": torch.ones(8),
        "action_model.model.transformer_blocks.0.attn1.to_k.weight": torch.ones(4, 2048),
        "action_model.model.transformer_blocks.1.attn1.to_k.weight": torch.ones(4, 4),
        "vj_predictor.action_encoder.weight": torch.ones(4, 2048),
        "vj_predictor.extrinsics_encoder.weight": torch.ones(4, 2047),
    }
    target = {key: torch.zeros_like(value) for key, value in source.items()}
    del target["qwen_vl_interface.old_backbone"]
    target["qwen_vl_interface.new_backbone"] = torch.full((3,), 7.)
    for key in target:
        if key.endswith("layer1.weight"):
            target[key] = torch.zeros(4, 10)
        elif key.endswith("layer2.weight"):
            target[key] = torch.zeros(10, 4)
        elif key.endswith("layer2.bias"):
            target[key] = torch.zeros(10)
        elif target[key].ndim == 2 and target[key].shape[1] in (2047, 2048):
            target[key] = torch.zeros(4, target[key].shape[1] + 512)
    return source, target


def test_only_known_interfaces_reset_and_new_vlm_preserved():
    source, target = states()
    loaded, report = select_qwen3_5_state(source, target)
    assert report["head_initialization"] == MODE
    assert report["missing_keys"] == ["qwen_vl_interface.new_backbone"]
    assert not any(key.startswith("qwen_vl_interface.") for key in loaded)
    assert "action_model.model.transformer_blocks.1.attn1.to_k.weight" in loaded
    assert len(report["adapted_keys"]) == 8
    target.update(loaded)
    assert torch.equal(target["qwen_vl_interface.new_backbone"], torch.full((3,), 7.))
    assert torch.equal(target["vj_encoder.weight"], source["vj_encoder.weight"])
    assert not target["vj_predictor.action_encoder.weight"].any()


@pytest.mark.parametrize("fault", ["missing", "unexpected", "unapproved_shape", "wrong_width", "joint10_parent"])
def test_transfer_rejects_unapproved_changes(fault):
    source, target = states()
    if fault == "missing":
        del source["vj_encoder.weight"]
    elif fault == "unexpected":
        source["action_model.extra"] = torch.ones(2)
    elif fault == "unapproved_shape":
        target["vj_predictor.predictor_embed.weight"] = torch.zeros(3, 2)
    elif fault == "wrong_width":
        target["vj_predictor.action_encoder.weight"] = torch.zeros(4, 4096)
    else:
        source["action_model.action_decoder.layer2.bias"] = torch.ones(10)
    with pytest.raises(ValueError):
        select_qwen3_5_state(source, target)


def test_ddp_source_prefix_supported():
    source, target = states()
    loaded, _ = select_qwen3_5_state({"module." + key: value for key, value in source.items()}, target)
    assert "vj_encoder.weight" in loaded


def test_optimizer_and_initialization_bound_to_resume_contract():
    from pathlib import Path
    root = Path(os.environ.get("S1B_PROJECT", Path(__file__).resolve().parents[3]))
    config = OmegaConf.to_container(OmegaConf.load(root / "configs/vlajepa_s1b_real_joint10_qwen3_5_hpc2.yaml"))
    _, original = training_contract(config)
    for key, value in [("optimizer_sharding", False), ("pretrained_initialization", "sim_joint8")]:
        changed = deepcopy(config)
        changed["trainer"][key] = value
        assert training_contract(changed)[1] != original
