"""Official OpenPI PyTorch policy behind the unchanged S1B HTTP contract."""
import time
from pathlib import Path
import numpy as np
from .artifacts import verify
from .runtime import upstream, config, check_cuda_build
from .transforms import numeric


class Policy:
    def __init__(self, run, step, lock, device='cuda:0'):
        started = time.perf_counter()
        self.identity = verify(run, step, lock)
        upstream()
        import torch
        from openpi.policies.policy_config import create_trained_policy
        self.device = torch.device(device)
        if self.device.type != 'cuda' or self.device.index is None:
            raise ValueError('Explicit CUDA device required')
        check_cuda_build(torch, self.device)
        torch.cuda.set_device(self.device)
        self.policy = create_trained_policy(config(run), Path(run), norm_stats={}, pytorch_device=str(self.device))
        torch.cuda.synchronize(self.device)
        self.load_seconds = time.perf_counter() - started

    def predict(self, head, right, state, seed=42):
        import torch
        # Explicit noise avoids relying on global Torch/JAX random state.
        noise = np.random.default_rng(seed).standard_normal((7,32)).astype(np.float32)
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            result = self.policy.infer(dict(head=head, right=right, state=state, prompt=self.identity['task']), noise=noise)
        torch.cuda.synchronize(self.device)
        actions = numeric(result['actions'], (7,10))
        return actions, dict(preprocess_seconds=0., inference_seconds=time.perf_counter()-started)
