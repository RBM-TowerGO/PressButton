"""Real-model C01-C03 using an NPZ captured from synchronized native cameras."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.run / "runtime"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import torch
    from pressb.online_rl.astribot_policy import AstribotLatentPolicy, FrozenAstribotPolicy
    from pressb.online_rl.astribot_features import FEATURE
    started = time.perf_counter()
    policy = AstribotLatentPolicy(args.run / "checkpoints/step_008000", args.device)
    with np.load(args.input, allow_pickle=False) as data:
        kwargs = dict(head_image=Image.fromarray(data["head"]),
                      wrist_right_image=Image.fromarray(data["wrist_right"]),
                      state=data["state"].copy(), seed=int(data["seed"]))
    captured = []
    original = policy.model.predict_action
    def capture(*a, **kw):
        result = original(*a, **kw)
        captured.append(result["embodied_action_tokens"].copy())
        return result
    policy.model.predict_action = capture
    old = FrozenAstribotPolicy.predict(policy, **kwargs)
    new = policy.predict(**kwargs)
    repeated = policy.predict(**kwargs)
    assert len(captured) == 3, "Each request must make exactly one model forward"
    assert captured[1].shape == (1, 32, 2048)
    for key in ("action8", "action32", "normalized_actions"):
        np.testing.assert_array_equal(old[key], new[key])
        np.testing.assert_array_equal(new[key], repeated[key])
    assert new["vlm_latent"].dtype == np.float32 and np.isfinite(new["vlm_latent"]).all()
    np.testing.assert_array_equal(new["vlm_latent"], captured[1].astype(np.float32).mean(1)[0])
    manual = captured[1][0].astype(np.float64).sum(axis=0) / 32
    np.testing.assert_allclose(new["vlm_latent"], manual, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(new["vlm_latent"], repeated["vlm_latent"])
    assert not any(parameter.requires_grad for parameter in policy.model.parameters())
    np.savez(args.output / "evidence.npz", tokens=captured[1], **new)
    report = dict(passed=True, C01="passed", C02="passed", C03="passed", feature=FEATURE,
                  token_shape=list(captured[1].shape), token_dtype=str(captured[1].dtype),
                  latent_shape=list(new["vlm_latent"].shape), forwards=3,
                  model_sha256=policy.checked["manifest"]["model_sha256"],
                  input=str(args.input), input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  adapter_sha256=hashlib.sha256(Path(__file__).resolve().parents[1].joinpath(
                      "src/pressb/online_rl/astribot_policy.py").read_bytes()).hexdigest(),
                  frozen=True, wall_seconds=time.perf_counter()-started,
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(policy.device))
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
