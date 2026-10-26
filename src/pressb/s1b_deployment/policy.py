"""Full trained weights, saved processor, training-equivalent joint10 inference."""
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
from .contract import TASK, verify

class Policy:
    def __init__(self, run, step, lock, device='cuda:0'):
        started = time.perf_counter()
        self.identity = verify(run, step, lock)
        run = Path(run).resolve()
        sys.path.insert(0, str(run / 'runtime'))
        import torch
        from omegaconf import OmegaConf
        from transformers import AutoProcessor
        from starVLA.model.framework.VLA_JEPA import VLA_JEPA
        from starVLA.inference.piper_policy import install_saved_processor, load_trained_state, disable_unused_qwen_cache
        torch.set_num_threads(4)
        self.device = torch.device(device)
        if self.device.type != 'cuda' or self.device.index is None: raise ValueError('Explicit CUDA device required')
        torch.cuda.set_device(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        cfg = OmegaConf.load(run / 'config.local.yaml')
        cfg.framework.qwenvl.device_map = str(self.device)
        self.model = VLA_JEPA(config=cfg)
        install_saved_processor(self.model, AutoProcessor.from_pretrained(run / 'processor', local_files_only=True))
        load_trained_state(self.model, run / f'checkpoints/step_{step:06d}/model.pt')
        disable_unused_qwen_cache(self.model)
        self.model.to(self.device).eval().requires_grad_(False)
        torch.cuda.synchronize(self.device)
        self.load_peak_allocated_gib = torch.cuda.max_memory_allocated(self.device) / 1024**3
        self.load_seconds = time.perf_counter() - started
        self.stats = self.identity['normalization']

    def normalize_state(self, state):
        state = np.asarray(state)
        if state.shape != (10,) or state.dtype != np.float32 or not np.isfinite(state).all(): raise ValueError('Expected finite float32 state10')
        return ((state - np.asarray(self.stats['state']['mean'], np.float32)) / np.asarray(self.stats['state']['scale'], np.float32))[None, None]

    def predict_prepared(self, images, normalized_state, seed=42):
        import torch
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.random.fork_rng(devices=[self.device.index]), torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            result = self.model.predict_action(batch_images=[images], instructions=[TASK], state=normalized_state)['normalized_actions']
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        if result.shape != (1, 7, 10) or not np.isfinite(result).all(): raise ValueError('Invalid model output')
        raw = result[0] * np.asarray(self.stats['action']['scale'], np.float32) + np.asarray(self.stats['action']['mean'], np.float32)
        if not np.isfinite(raw).all(): raise ValueError('Nonfinite raw action')
        return raw.astype(np.float32), elapsed

    def predict(self, head, right, state, seed=42):
        started = time.perf_counter()
        images = []
        for value, shape in zip((head, right), ((720, 1280, 3), (360, 640, 3))):
            if value.shape != shape or value.dtype != np.uint8: raise ValueError('Expected raw RGB uint8 camera dimensions')
            images.append(Image.fromarray(value, mode='RGB').resize((224, 224), Image.Resampling.BILINEAR))
        normalized = self.normalize_state(state)
        preprocessing = time.perf_counter() - started
        raw, elapsed = self.predict_prepared(images, normalized, seed)
        return raw, dict(preprocess_seconds=preprocessing, inference_seconds=elapsed)

def sdk_gripper(action):
    value = np.asarray(action)
    if value.shape != (7, 10) or not np.isfinite(value).all(): raise ValueError('Invalid action')
    return np.where(value[:, 7] < .5, 0, 100)
