"""CPU contract checks for the Astribot joint8 residual-RL integration."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pressb.online_rl.astribot import (ARM, BASE_DIM, DEFAULT_GAMMA, DEFAULT_SCALE, FIXED, GRIP,
                                        HORIZON, OBS_DIM, SIGNS, ChunkJournal, chunk_flags,
                                        compose_joint8, expand_joint8, load_replay, observation,
                                        save_replay, success_reward, summarize_episode,
                                        transition_discount, validate_scale, within_limits)
from pressb.online_rl.learner import LearnerConfig, ReplayBuffer, SACLearner


@pytest.fixture(autouse=True, scope="module")
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def base_config(**kwargs):
    values = dict(method="action_residual", obs_dim=OBS_DIM, base_dim=BASE_DIM,
                  hidden_dim=8, batch_size=4)
    values.update(kwargs)
    return LearnerConfig(**values)


def test_contract_constants_match_the_frozen_controller():
    assert BASE_DIM == 8 and HORIZON == 7 and OBS_DIM == 64
    assert list(ARM) == [5, 8, 11, 13, 15, 17, 19]
    assert list(GRIP) == [24, 25, 26, 27, 30, 31]
    assert list(SIGNS) == [-1, -1, 1, 1, 1, -1]
    assert len(FIXED) == 32 - 13
    assert len(DEFAULT_SCALE) == BASE_DIM


def test_observation_is_proprioceptive_only_and_validated():
    q = np.linspace(-0.5, 0.5, 32).astype(np.float32)
    qd = np.linspace(0.2, -0.2, 32).astype(np.float32)
    vector = observation(q, qd)
    assert vector.shape == (OBS_DIM,) and vector.dtype == np.float32
    np.testing.assert_array_equal(vector[:32], q)
    np.testing.assert_array_equal(vector[32:], qd)
    with pytest.raises(ValueError, match="q32 must have shape"):
        observation(q[:31], qd)
    bad = q.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        observation(bad, qd)


def test_compose_clips_to_normalized_bounds_and_requires_eight_scales():
    base = np.zeros((HORIZON, BASE_DIM), np.float32)
    residual = np.ones((HORIZON, BASE_DIM), np.float32)
    target = compose_joint8(base, residual, DEFAULT_SCALE)
    np.testing.assert_allclose(target[0], np.asarray(DEFAULT_SCALE, np.float32), rtol=1e-6)
    # Flat residual is accepted and normalized bounds are enforced.
    np.testing.assert_allclose(compose_joint8(base, residual.reshape(-1), DEFAULT_SCALE), target)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        compose_joint8(base, residual * 1.01, DEFAULT_SCALE)
    with pytest.raises(ValueError, match="must contain 8"):
        validate_scale(DEFAULT_SCALE[:7])
    with pytest.raises(ValueError, match="positive"):
        validate_scale(tuple([0.02] * 7 + [0.0]))


def test_expand_matches_the_32_joint_controller_contract():
    fixed = np.arange(32, dtype=np.float32) * 0.01
    joint8 = np.zeros((HORIZON, BASE_DIM), np.float32)
    joint8[:, :7] = 0.3
    joint8[:, 7] = 0.02
    full = expand_joint8(joint8, fixed)
    assert full.shape == (HORIZON, 32)
    np.testing.assert_allclose(full[:, ARM], 0.3)
    np.testing.assert_allclose(full[:, GRIP], np.broadcast_to(np.float32(0.02) * SIGNS, (HORIZON, 6)))
    np.testing.assert_array_equal(full[:, FIXED], np.broadcast_to(fixed[FIXED], (HORIZON, len(FIXED))))
    limits = np.c_[np.full(32, -10.0), np.full(32, 10.0)]
    assert within_limits(full, limits)
    with pytest.raises(ValueError, match="tolerance"):
        within_limits(full, limits, tolerance=-1)


def test_residual_limit_violation_is_detectable_not_silently_projected():
    fixed = np.zeros(32, np.float32)
    joint8 = np.zeros((HORIZON, BASE_DIM), np.float32)
    joint8[:, 0] = 1.0
    full = expand_joint8(joint8, fixed)
    limits = np.c_[np.zeros(32), np.full(32, 0.5)]
    assert not within_limits(full, limits)
    assert within_limits(full, limits, 0.6)


def test_rewards_prefer_earlier_success_and_discounts_track_duration():
    late = success_reward(DEFAULT_GAMMA, HORIZON)
    early = success_reward(DEFAULT_GAMMA, 1)
    assert early == pytest.approx(1.0)
    assert 0 < late < early
    assert transition_discount(DEFAULT_GAMMA) == pytest.approx(DEFAULT_GAMMA ** HORIZON)
    assert transition_discount(1.0) == 1.0
    with pytest.raises(ValueError, match="gamma"):
        success_reward(1.5, 1)
    with pytest.raises(ValueError, match="control_steps"):
        transition_discount(DEFAULT_GAMMA, HORIZON + 1)


def test_chunk_flags_are_mutually_exclusive():
    success = chunk_flags(success=True, failure_reason=None, timed_out=False, success_value=0.7)
    assert success == dict(reward=0.7, terminated=True, truncated=False,
                           reason="physical_press_and_light")
    failure = chunk_flags(success=False, failure_reason="unexpected_contact", timed_out=False)
    assert failure["terminated"] and not failure["truncated"] and failure["reward"] == 0.0
    timeout = chunk_flags(success=False, failure_reason=None, timed_out=True)
    assert timeout["truncated"] and not timeout["terminated"]
    running = chunk_flags(success=False, failure_reason=None, timed_out=False)
    assert not running["terminated"] and not running["truncated"]
    with pytest.raises(ValueError, match="success cannot be combined"):
        chunk_flags(success=True, failure_reason="x", timed_out=False)


def test_episode_summary_is_json_safe():
    rows = [dict(travel=0.0, force=0.0, light=False),
            dict(travel=0.0012, force=0.4, light=True)]
    summary = summarize_episode(rows)
    json.dumps(summary)
    assert summary["physics_records"] == 2
    assert summary["max_travel_m"] == pytest.approx(0.0012)
    assert summary["light"] is True
    assert summarize_episode([])["physics_records"] == 0


def test_learner_and_replay_support_an_eight_dimensional_base_action():
    learner = SACLearner(base_config(), device="cpu")
    assert learner.action_dim == HORIZON * BASE_DIM
    replay = ReplayBuffer(8, OBS_DIM, learner.action_dim, seed=7, base_dim=BASE_DIM)
    rng = np.random.default_rng(3)
    for _ in range(6):
        replay.add(rng.normal(size=OBS_DIM), np.zeros(learner.action_dim), 0.0, 0.98,
                   rng.normal(size=OBS_DIM), False, False,
                   base_action=np.zeros((HORIZON, BASE_DIM)),
                   next_base_action=np.ones((HORIZON, BASE_DIM)))
    metrics = learner.update(replay.sample(4))
    assert metrics["updates"] == 1.0
    action = learner.act(rng.normal(size=(1, OBS_DIM)), np.zeros((1, HORIZON, BASE_DIM)))
    assert action.shape == (1, learner.action_dim)
    assert np.all(np.abs(action) <= 1 + 1e-6)


def test_xyz_mode_keeps_frozen_coordinates_for_an_eight_dimensional_base():
    learner = SACLearner(base_config(residual_mode="xyz",
                                     residual_scale=(0.01, 0.01, 0.01)), device="cpu")
    assert learner.action_dim == HORIZON * 3
    base = torch.zeros(2, HORIZON * BASE_DIM)
    action = torch.ones(2, HORIZON * 3)
    combined = learner._critic_action(action, base)
    assert combined.shape == (2, HORIZON * BASE_DIM)
    np.testing.assert_allclose(combined[:, [0, 1, 2]].detach().numpy(), 0.01, rtol=1e-6)
    np.testing.assert_allclose(combined[:, 7].detach().numpy(), 0.0)


def test_replay_roundtrip_keeps_base_width_and_ring_state(tmp_path):
    replay = ReplayBuffer(6, OBS_DIM, HORIZON * BASE_DIM, seed=11, base_dim=BASE_DIM)
    rng = np.random.default_rng(5)
    for index in range(4):
        replay.add(rng.normal(size=OBS_DIM), np.zeros(HORIZON * BASE_DIM), float(index), 0.9,
                   rng.normal(size=OBS_DIM), False, False,
                   base_action=np.full((HORIZON, BASE_DIM), index, np.float32),
                   next_base_action=np.full((HORIZON, BASE_DIM), index + 1, np.float32))
    path = tmp_path / "replay.npz"
    save_replay(path, replay)
    restored = load_replay(path, capacity=6, obs_dim=OBS_DIM,
                           action_dim=HORIZON * BASE_DIM, base_dim=BASE_DIM)
    assert len(restored) == 4 and restored.position == replay.position
    batch, original = restored.sample(4), replay.sample(4)
    for name in ("obs", "action", "reward", "base_action"):
        np.testing.assert_array_equal(batch[name], original[name])
    with pytest.raises(ValueError, match="dimensions"):
        load_replay(path, capacity=6, obs_dim=OBS_DIM,
                    action_dim=HORIZON * BASE_DIM, base_dim=9)


def test_config_defaults_follow_the_base_width_and_keep_piper_history():
    # The historical PiPER default is retained exactly (XYZ 0.03 m, rotation 0.1 rad).
    assert LearnerConfig(method="action_residual", obs_dim=4).residual_scale == (
        0.03, 0.03, 0.03, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
    assert base_config().residual_scale == (0.03,) * BASE_DIM
    assert base_config(base_dim=12).residual_scale == (0.03,) * 12
    with pytest.raises(ValueError, match="positive integer"):
        base_config(base_dim=0)
    with pytest.raises(ValueError, match="at least 3"):
        base_config(base_dim=2, residual_mode="xyz")


def test_protocol_helpers_cannot_select_a_non_train_split():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_astribot_online_rl as runner

    model_run = Path(__file__).resolve().parents[1] / "models/astribot_head_right_joint8_ft_20261018"
    contract, ids = runner.load_contract(model_run, "train")
    assert ids == contract["split"]["train_source_episode_ids"]
    val = contract["split"]["val_source_episode_ids"]
    assert not set(val) & set(ids)
    assert runner.load_contract(model_run, "val")[1] == val
    with pytest.raises(ValueError, match="split must be"):
        runner.load_contract(model_run, "heldout")


def test_chunk_journal_holds_running_transitions_until_the_successor_is_known():
    learner = SACLearner(base_config(), device="cpu")
    replay = ReplayBuffer(6, OBS_DIM, learner.action_dim, seed=1, base_dim=BASE_DIM)
    journal = ChunkJournal(replay, gamma=DEFAULT_GAMMA)
    rng = np.random.default_rng(9)
    obs = rng.normal(size=OBS_DIM).astype(np.float32)
    row = journal.start(obs, np.zeros(learner.action_dim, np.float32),
                        np.zeros((HORIZON, BASE_DIM), np.float32))
    journal.defer(row, 2)
    assert len(replay) == 0 and journal.pending is row
    with pytest.raises(ValueError, match="already held"):
        journal.defer(row, 2)
    successor, base = rng.normal(size=OBS_DIM), np.ones((HORIZON, BASE_DIM), np.float32)
    flags = journal.flush(successor, base)
    assert not flags["terminated"] and not flags["truncated"]
    assert len(replay) == 1 and journal.pending is None and journal.transitions == 1
    batch = replay.sample(1)
    np.testing.assert_allclose(batch["next_obs"], successor[None, :])
    np.testing.assert_allclose(batch["base_action"], np.zeros((1, HORIZON * BASE_DIM)))
    np.testing.assert_allclose(batch["next_base_action"], np.ones((1, HORIZON * BASE_DIM)))
    assert batch["discount"][0, 0] == pytest.approx(DEFAULT_GAMMA ** 2)
    assert batch["reward"][0, 0] == 0.0


def test_chunk_journal_terminal_flags_rewards_and_discounts():
    learner = SACLearner(base_config(), device="cpu")
    replay = ReplayBuffer(8, OBS_DIM, learner.action_dim, seed=2, base_dim=BASE_DIM)
    journal = ChunkJournal(replay, gamma=DEFAULT_GAMMA)
    rng = np.random.default_rng(4)
    obs = rng.normal(size=OBS_DIM).astype(np.float32)
    base8 = np.zeros((HORIZON, BASE_DIM), np.float32)

    def row():
        return journal.start(obs, np.zeros(learner.action_dim, np.float32), base8)

    success = journal.commit(row(), next_obs=obs, next_base=base8, steps=3, success=True)
    assert success["terminated"] and success["reward"] == pytest.approx(DEFAULT_GAMMA ** 2)
    failure = journal.commit(row(), next_obs=obs, next_base=base8, steps=4,
                             failure_reason="unexpected_contact")
    assert failure["terminated"] and failure["reward"] == 0.0
    timeout = journal.commit(row(), next_obs=obs, next_base=base8, steps=HORIZON, timed_out=True)
    assert timeout["truncated"] and not timeout["terminated"]
    # Ring arrays are in insertion order; sampling is with replacement.
    terminated = replay.arrays["terminated"][:3].ravel().astype(int)
    truncated = replay.arrays["truncated"][:3].ravel().astype(int)
    assert terminated.sum() == 2 and truncated.sum() == 1
    np.testing.assert_allclose(replay.arrays["discount"][:3].ravel(),
                               [DEFAULT_GAMMA ** 3, DEFAULT_GAMMA ** 4, DEFAULT_GAMMA ** HORIZON],
                               rtol=1e-6)
    np.testing.assert_allclose(replay.arrays["reward"][:3].ravel(),
                               [DEFAULT_GAMMA ** 2, 0.0, 0.0], rtol=1e-6)
    assert journal.transitions == 3


def test_evaluation_journal_never_touches_replay():
    learner = SACLearner(base_config(), device="cpu")
    replay = ReplayBuffer(4, OBS_DIM, learner.action_dim, seed=3, base_dim=BASE_DIM)
    journal = ChunkJournal(replay, train=False, gamma=DEFAULT_GAMMA)
    obs = np.zeros(OBS_DIM, np.float32)
    row = journal.start(obs, np.zeros(learner.action_dim, np.float32),
                        np.zeros((HORIZON, BASE_DIM), np.float32))
    journal.commit(row, next_obs=obs, next_base=np.zeros((HORIZON, BASE_DIM)),
                   steps=1, success=True)
    journal.defer(row, 1)
    journal.flush(obs, np.zeros((HORIZON, BASE_DIM)))
    assert len(replay) == 0 and journal.transitions == 0


def test_chunk_journal_validates_inputs():
    with pytest.raises(ValueError, match="training requires a replay"):
        ChunkJournal(None, train=True)
    journal = ChunkJournal(None, train=False)
    with pytest.raises(ValueError, match="no pending"):
        journal.flush(np.zeros(OBS_DIM), np.zeros((HORIZON, BASE_DIM)))
    with pytest.raises(ValueError, match="obs must have shape"):
        journal.start(np.zeros(3), np.zeros(HORIZON * BASE_DIM),
                      np.zeros((HORIZON, BASE_DIM)))
    with pytest.raises(ValueError, match="action must have shape"):
        journal.start(np.zeros(OBS_DIM), np.zeros(3), np.zeros((HORIZON, BASE_DIM)))


def run_cli(*arguments):
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts/run_astribot_online_rl.py"
    return subprocess.run([sys.executable, str(script), *arguments],
                          capture_output=True, text=True, timeout=120)


def test_cli_rejects_evaluation_without_a_trained_learner():
    result = run_cli("--model-run", "models/does-not-exist", "--checkpoint-step", "8000",
                     "--output", "outputs/does-not-exist", "--eval-episodes", "4")
    assert result.returncode != 0
    assert "requires --resume" in result.stderr


def test_cli_rejects_eval_only_without_eval_episodes():
    result = run_cli("--model-run", "models/does-not-exist", "--checkpoint-step", "8000",
                     "--output", "outputs/does-not-exist", "--resume", "--eval-only")
    assert result.returncode != 0
    assert "requires --eval-episodes" in result.stderr


def test_cli_requires_exactly_eight_scales():
    result = run_cli("--model-run", "models/does-not-exist", "--checkpoint-step", "8000",
                     "--output", "outputs/does-not-exist",
                     "--scale", *["0.02"] * 7)
    assert result.returncode == 2
    assert "expected 8 arguments" in result.stderr
