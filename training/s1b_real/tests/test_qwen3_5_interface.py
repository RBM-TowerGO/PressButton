from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip("transformers.models.qwen3_5")
from transformers import Qwen3_5Config
from starVLA.model.modules.vlm import QWen3_5


def tiny_config():
    return Qwen3_5Config(
        image_token_id=5, video_token_id=6, vision_start_token_id=7, vision_end_token_id=8,
        tie_word_embeddings=True,
        text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=16, layer_types=["linear_attention", "full_attention"],
                         linear_key_head_dim=8, linear_value_head_dim=8,
                         linear_num_key_heads=2, linear_num_value_heads=2,
                         pad_token_id=0, eos_token_id=2,
                         rope_parameters=dict(rope_type="default", rope_theta=10000.,
                                              partial_rotary_factor=1., mrope_section=[2, 2, 4])),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
                           patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
                           out_hidden_size=32, num_position_embeddings=16))


@pytest.mark.parametrize('rectangular_views', [False, True])
def test_multimodal_feature_path_backward_and_no_lm_logits(monkeypatch, rectangular_views):
    torch.set_num_threads(2)
    monkeypatch.setattr(QWen3_5.AutoConfig, "from_pretrained", lambda *_: tiny_config())
    processor = SimpleNamespace(tokenizer=SimpleNamespace(padding_side="right"))
    monkeypatch.setattr(QWen3_5.AutoProcessor, "from_pretrained", lambda *_: processor)
    cfg = OmegaConf.create(dict(framework=dict(qwenvl=dict(base_vlm="tiny", vl_hidden_dim=32,
        init_from_config=True, device_map="cpu", features_only=True, attn_implementation="sdpa"))))
    interface = QWen3_5.Qwen3_5Interface(cfg).float()
    interface.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert processor.tokenizer.padding_side == "left"
    assert not interface.model.config.text_config.use_cache
    # An image expands to one merged visual token for a 2x2 patch grid.
    inputs = dict(input_ids=torch.tensor([[0, 7, 5, 8, 9, 10], [7, 5, 8, 9, 10, 11]]),
                  attention_mask=torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]),
                  pixel_values=torch.randn(8, 3 * 2 * 16 * 16),
                  image_grid_thw=torch.tensor([[1, 2, 2], [1, 2, 2]]))
    if rectangular_views:
        # Two unequal rectangular views per sample, plus unequal text lengths.
        sequence = [7, 5, 5, 8, 7] + [5] * 6 + [8, 9, 10]
        inputs = dict(input_ids=torch.tensor([[0] + sequence, sequence + [11]]),
                      attention_mask=torch.tensor([[0] + [1] * len(sequence), [1] * (len(sequence) + 1)]),
                      pixel_values=torch.randn(64, 3 * 2 * 16 * 16),
                      image_grid_thw=torch.tensor([[1, 2, 4], [1, 4, 6]] * 2))
    def reject_logits(*_):
        raise AssertionError("features_only invoked the vocabulary head")
    monkeypatch.setattr(interface.model.lm_head, "forward", reject_logits)
    output = interface(**inputs, output_hidden_states=True, return_dict=True)
    assert output.hidden_states[-1].shape == (*inputs['input_ids'].shape, 32)
    assert torch.isfinite(output.hidden_states[-1]).all()
    output.hidden_states[-1][:, -1].square().mean().backward()
    language = interface.model.model.language_model
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in language.parameters())
    with pytest.raises(NotImplementedError):
        interface.build_qwenvl_inputs([], [], solutions=["action"])


def test_fresh_load_and_resume_construction_are_distinct(monkeypatch):
    import sys
    from starVLA.training.train_s1b_real import construct_training_model
    observed = []
    def build(config):
        observed.append(config.framework.qwenvl.init_from_config)
        return SimpleNamespace(config=config)
    monkeypatch.setitem(sys.modules, "starVLA.model.framework.VLA_JEPA", SimpleNamespace(VLA_JEPA=build))
    cfg = OmegaConf.create(dict(framework=dict(qwenvl=dict(model_type="qwen3_5", init_from_config=True)),
        trainer=dict(pretrained_initialization="qwen3_5_from_qwen3_vl_joint8")))
    construct_training_model(cfg)
    assert cfg.framework.qwenvl.init_from_config is True
    construct_training_model(cfg, restarting=True)
    assert observed == [False, True]
