"""Local forward-only wiring check; creates no optimizer and sends no robot commands."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / "runtime"))

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch

from starVLA.training.train_s1b_real import construct_training_model, load_pretrained


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/vlajepa_s1b_real_joint10_qwen3_5_hpc2.yaml")
    parser.add_argument("--base-vlm", type=Path, default=root / "models/astribot_backbones/Qwen3.5-4B")
    parser.add_argument("--base-encoder", type=Path, default=root / "models/astribot_backbones/vjepa2-vitl-fpc64-256")
    parser.add_argument("--parent-checkpoint", type=Path, default=root / "models/astribot_head_right_joint8_ft_20261018/checkpoints/step_008000/model.pt")
    parser.add_argument("--output", type=Path, default=root / "outputs/qwen3_5_wiring.json")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    config = OmegaConf.load(args.config)
    config.framework.qwenvl.base_vlm = str(args.base_vlm.resolve())
    config.framework.qwenvl.device_map = str(device)
    config.framework.vj2_model.base_encoder = str(args.base_encoder.resolve())
    started = time.perf_counter()
    model = construct_training_model(config)
    embedding = model.qwen_vl_interface.model.get_input_embeddings().weight
    before = embedding[:4].detach().cpu().clone()
    loading = load_pretrained(model, args.parent_checkpoint)
    assert torch.equal(before, embedding[:4].detach().cpu()), "Old checkpoint overwrote new Qwen weights"
    model.to(device).eval().requires_grad_(False)
    raw_images = [Image.new('RGB', size, color) for size, color in
                  [((1280, 720), 'gray'), ((640, 360), 'navy')]]
    image_mode = config.datasets.vla_data.get('image_mode', 'resize')
    resolution = int(config.datasets.vla_data.resolution_size)
    video_images = [image.resize((resolution, resolution), Image.Resampling.BILINEAR) for image in raw_images]
    images = [raw_images if image_mode == 'native' else video_images]
    inputs = model.qwen_vl_interface.build_qwenvl_inputs(
        images, ["Press the elevator button."],
        prompt_replace_dict={"{actions}": model.replace_prompt, "{e_actions}": model.embodied_replace_prompt})
    ids = inputs["input_ids"]
    action_count = int(torch.isin(ids, torch.tensor(model.action_token_ids, device=device)).sum())
    embodied_count = int((ids == model.embodied_action_token_id).sum())
    assert (action_count, embodied_count) == (24, 32)
    with torch.inference_mode():
        hidden = model.qwen_vl_interface(**inputs).hidden_states[-1]
        assert hidden.shape == (*ids.shape, 2560) and torch.isfinite(hidden).all()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        inference_started = time.perf_counter()
        actions = model.predict_action(
            batch_images=images, instructions=["Press the elevator button."],
            state=np.zeros((1, 1, 10), dtype=np.float32))["normalized_actions"]
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        inference_seconds = time.perf_counter() - inference_started
        video = np.stack([np.repeat(np.asarray(image)[None], 8, axis=0) for image in video_images])
        example = dict(image=images[0], video=video, lang="Press the elevator button.",
                       state=np.zeros((1, 10), dtype=np.float32), action=np.zeros((7, 10), dtype=np.float32),
                       action_is_pad=np.zeros(7, dtype=bool), video_is_pad=np.zeros(8, dtype=bool))
        forward_started = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            losses = model([example])
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        forward_seconds = time.perf_counter() - forward_started
        assert all(torch.isfinite(value).all() for value in losses.values())
    assert actions.shape == (1, 7, 10) and np.isfinite(actions).all()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    import transformers
    report = dict(success=True, kind="synthetic_forward_wiring_only", training_performed=False,
                  torch_version=torch.__version__, transformers_version=transformers.__version__,
                  device=str(device), hidden_shape=list(hidden.shape), action_shape=list(actions.shape),
                  action_token_count=action_count, embodied_token_count=embodied_count,
                  image_mode=image_mode, image_sizes=[list(image.size) for image in images[0]],
                  image_grid_thw=inputs['image_grid_thw'].tolist(),
                  vision_tokens_per_view=(inputs['image_grid_thw'].prod(-1) //
                      model.qwen_vl_interface.processor.image_processor.merge_size**2).tolist(),
                  inference_seconds=inference_seconds, training_forward_seconds=forward_seconds,
                  synthetic_losses={key: float(value) for key, value in losses.items()},
                  pretrained_backbone_preserved=True, loading=loading,
                  peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None,
                  seconds=time.perf_counter() - started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "loading"}))


if __name__ == "__main__":
    main()
