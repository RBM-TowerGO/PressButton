"""Raw pooled VLA features, matching the original PiPER residual observation.

This contract is deliberately distinct from main's normalized state_latent v1.
"""
import numpy as np

FEATURE = dict(supported=True, feature_type="embodied_action_tokens_mean",
               feature_dim=2048, contract_version=2, token_shape=[1, 32, 2048],
               source="vlm_last_hidden_embodied_action_tokens",
               token_cast="action_head_dtype_then_float32", pooling_axis=1,
               dtype="float32", normalization="none")
MODES = ("state_only", "state_latent_raw", "state_latent_raw_command")


def observation_contract(mode="state_only"):
    if mode not in MODES:
        raise ValueError(f"Unknown observation mode: {mode}")
    result = dict(mode=mode, obs_dim=64 if mode == "state_only" else 2112,
                  feature=None if mode == "state_only" else dict(FEATURE))
    if mode == "state_latent_raw_command":
        result.update(obs_dim=2120, previous_command=dict(
            version=1, source="last_completed_physics_tick_position_target",
            offset=2112, size=8, dtype="float32", units="rad", normalization="none",
            arm_indices=[5, 8, 11, 13, 15, 17, 19],
            gripper_indices=[24, 25, 26, 27, 30, 31],
            gripper_signs=[-1, -1, 1, 1, 1, -1], gripper_reduction="signed_mean",
            initialization="reset_targets_then_settling_tick"))
    return result


def needs_latent(mode):
    return observation_contract(mode)["feature"] is not None


def needs_previous_command(mode):
    return "previous_command" in observation_contract(mode)


def validate_capability(health, mode):
    contract = observation_contract(mode)
    if needs_latent(mode) and health.get("latent") != contract["feature"]:
        raise ValueError("Policy server latent capability/contract mismatch")


def pooled_tokens(tokens):
    """Mean of action-head-dtype tokens converted to float32 by the adapter."""
    tokens = np.asarray(tokens)
    if tokens.shape != (1, 32, 2048) or tokens.dtype != np.float32:
        raise ValueError(f"Expected float32 tokens [1,32,2048], got {tokens.shape}/{tokens.dtype}")
    if not np.isfinite(tokens).all():
        raise ValueError("Nonfinite embodied action tokens")
    return raw_latent(tokens.mean(axis=1, dtype=np.float32)[0])


def raw_latent(value):
    """Validate only: preserve the feature's offset and scale exactly."""
    z = np.asarray(value)
    if z.dtype != np.float32 or z.shape != (2048,) or not np.isfinite(z).all():
        raise ValueError("vlm_latent must be finite float32 (2048,)")
    return z


def validate_restore(saved, expected, mode):
    """Legacy missing metadata is accepted only for state-only observations."""
    observation_contract(mode)
    if saved is None and mode == "state_only":
        return
    if saved != expected:
        raise ValueError("Checkpoint observation/model/source/training contract mismatch")
