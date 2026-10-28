# Qwen3.5 DeltaNet kernels on the local RTX 5090

Enabled in `.venv-qwen3-5` on 2026-10-28. No checkpoint, image preprocessing,
action timing or robot execution configuration was changed.

Validated versions: Python 3.11, PyTorch 2.7.0+cu128, Transformers 5.3.0,
flash-linear-attention 0.4.2, fla-core 0.4.2, causal-conv1d 1.6.2.post1,
Triton 3.3.1. CUDA toolkit 12.8 was used to build causal-conv1d, including sm_120.

Installation used:

```bash
.venv-qwen3-5/bin/pip install --no-deps flash-linear-attention==0.4.2 fla-core==0.4.2
MAX_JOBS=4 CAUSAL_CONV1D_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST=12.0 \
  .venv-qwen3-5/bin/pip install --no-build-isolation --no-deps causal-conv1d==1.6.2.post1
.venv-qwen3-5/bin/pip install --no-deps triton==3.3.1
```

Triton 3.3.0 failed compiling FLA matmul with `computeCapability not supported`.
The 3.3.1 implementation handles sm_120. This deliberately overrides PyTorch
2.7.0's exact Triton 3.3.0 package requirement in this virtual environment;
it validates this eager inference path, not arbitrary torch.compile or training.
The shared base environment was not modified. Avoid dependency resolution that
silently downgrades Triton to 3.3.0.

The existing policy entry point automatically selects these optional kernels on
fresh process startup. Transformers reports `is_fast_path_available=True`, and
the loaded checkpoint has 24 Qwen3_5GatedDeltaNet layers using the FLA path.

## Offline verification

Checkpoint: native mixed Qwen3.5 run v2, step 4500. Three real validation samples,
native head/wrist images, saved normalized states, identical seed and weights.
Each mode uses two warmup calls and five timed calls per sample. Fallback mode
restores PyTorch convolution, chunk DeltaNet and gated RMS norm implementations.

| Mode | Median | P95 |
| --- | ---: | ---: |
| Original fallback | 253.40 ms | 257.15 ms |
| Fast kernels | 156.52 ms | 158.65 ms |
| Fast kernels repeated | 156.85 ms | 160.44 ms |

All outputs finite, each sample produces a 7x10 action chunk. Maximum absolute
joint difference: 0.000557 rad; gripper scalar difference: 0.000431. Repeated
fast-path outputs are identical on these samples. This is not a physical task
success test or an end-to-end robot latency measurement.

Local artifacts: `outputs/qwen35_fast_kernels/benchmark.py`, `results.json`,
`benchmark.log`, `environment_before.txt`, and `environment_after.txt`.
Inference timing includes processor work inside predict_prepared, but excludes
camera transport, JPEG decoding and robot control communication. The 233 ms
action horizon and stale-action rejection remain unchanged.
