"""Versioned, deployable observation contract for the frozen Astribot policy."""
import numpy as np

FEATURE = dict(supported=True, feature_type="embodied_action_tokens_mean",
               feature_dim=2048, contract_version=1, token_shape=[1, 32, 2048],
               pooling_axis=1, dtype="float32", layer_norm_epsilon=1e-5,
               layer_norm_affine=False, variance="population")
MODES = ("state_only", "state_latent")


def observation_contract(mode="state_only"):
    if mode not in MODES:
        raise ValueError(f"Unknown observation mode: {mode}")
    return dict(mode=mode, obs_dim=2112 if mode == "state_latent" else 64,
                feature=dict(FEATURE) if mode == "state_latent" else None)


def validate_capability(health, mode):
    contract = observation_contract(mode)
    if mode == "state_latent" and health.get("latent") != contract["feature"]:
        raise ValueError("Policy server latent capability/contract mismatch")


def pooled_tokens(tokens):
    tokens = np.asarray(tokens)
    if tokens.shape != (1, 32, 2048) or not np.issubdtype(tokens.dtype, np.floating):
        raise ValueError(f"Expected floating tokens [1,32,2048], got {tokens.shape}/{tokens.dtype}")
    tokens = tokens.astype(np.float32)
    if not np.isfinite(tokens).all():
        raise ValueError("Nonfinite embodied action tokens")
    return tokens.mean(axis=1, dtype=np.float32)[0]


def normalize_latent(value):
    z = np.asarray(value)
    if z.dtype != np.float32 or z.shape != (2048,) or not np.isfinite(z).all():
        raise ValueError("vlm_latent must be finite float32 (2048,)")
    centered = z - z.mean(dtype=np.float32)
    normalized = centered / np.sqrt(np.mean(centered * centered, dtype=np.float32) + np.float32(1e-5))
    if not np.isfinite(normalized).all():
        raise ValueError("Nonfinite normalized latent")
    return normalized


def validate_restore(saved, expected, mode):
    """Missing metadata is permitted only for a legacy proprioceptive run."""
    if saved is None and mode == "state_only":
        return
    if saved != expected:
        raise ValueError("Checkpoint observation/model/source/training contract mismatch")
