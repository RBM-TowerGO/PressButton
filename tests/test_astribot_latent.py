"""Latent math, semantic restore and real runner transition contracts (CPU)."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

pytest.importorskip("torch")
from pressb.online_rl.astribot_features import (FEATURE, normalize_latent, observation_contract,
                                               pooled_tokens, validate_capability, validate_restore)
from pressb.online_rl.astribot import ChunkJournal, observation, load_replay, save_replay
from pressb.online_rl.learner import LearnerConfig, ReplayBuffer, SACLearner
import run_astribot_online_rl as runner


def latent(index=0):
    return np.roll(np.arange(2048, dtype=np.float32), index)


def test_pool_and_normalization():
    tokens = np.arange(32 * 2048, dtype=np.float32).reshape(1, 32, 2048)
    np.testing.assert_array_equal(pooled_tokens(tokens), tokens[0].sum(0) / 32)
    np.testing.assert_array_equal(normalize_latent(np.ones(2048, np.float32)), np.zeros(2048))
    z = latent()
    expected = (z.astype(np.float64) - z.mean()) / np.sqrt(z.astype(np.float64).var() + 1e-5)
    np.testing.assert_allclose(normalize_latent(z), expected, atol=3e-7)
    with pytest.raises(ValueError):
        pooled_tokens(tokens.reshape(1, 16, 4096))


@pytest.mark.parametrize("bad", [None, np.zeros(2048, np.float64), np.zeros((1, 2048), np.float32),
                                  np.full(2048, np.nan, np.float32)])
def test_bad_features(bad):
    with pytest.raises(ValueError):
        observation(np.zeros(32), np.zeros(32), bad, mode="state_latent")


def test_modes_health_and_semantic_restore():
    validate_capability({}, "state_only")
    validate_capability({"latent": FEATURE}, "state_latent")
    for key in ("feature_type", "feature_dim", "contract_version"):
        bad = deepcopy(FEATURE)
        bad[key] = "wrong"
        with pytest.raises(ValueError):
            validate_capability({"latent": bad}, "state_latent")
    expected = dict(observation=observation_contract("state_latent"), model_sha256="abc")
    validate_restore(deepcopy(expected), expected, "state_latent")
    validate_restore(None, {}, "state_only")
    for bad in (None, {}, dict(expected, model_sha256="other")):
        with pytest.raises(ValueError):
            validate_restore(bad, expected, "state_latent")
    q, qd = np.arange(32), np.arange(32) + 32
    obs = observation(q, qd, latent(), mode="state_latent")
    np.testing.assert_array_equal(obs[:64], np.arange(64))
    np.testing.assert_array_equal(obs[64:], normalize_latent(latent()))
    assert observation(q, qd).shape == (64,)


def test_latent_network_replay_and_checkpoint(tmp_path):
    import torch
    torch.set_num_threads(1)
    cfg = LearnerConfig(method="action_residual", obs_dim=2112, base_dim=8,
                        hidden_dim=8, batch_size=2)
    learner = SACLearner(cfg)
    replay = ReplayBuffer(4, 2112, 56, base_dim=8)
    journal = ChunkJournal(replay, observation_mode="state_latent")
    a = observation(np.zeros(32), np.ones(32), latent(), mode="state_latent")
    b = observation(np.ones(32), np.zeros(32), latent(2), mode="state_latent")
    base = np.zeros((7, 8), np.float32)
    assert learner._actor_input(torch.zeros(1, 2112), torch.zeros(1, 56)).shape == (1, 2168)
    for _ in range(2):
        row = journal.start(a, np.zeros(56), base)
        journal.commit(row, next_obs=b, next_base=base, steps=7)
    metrics = learner.update(replay.sample(2))
    assert all(np.isfinite(list(metrics.values())))
    save_replay(tmp_path / "replay.npz", replay)
    restored = load_replay(tmp_path / "replay.npz", capacity=4, obs_dim=2112, action_dim=56, base_dim=8)
    np.testing.assert_array_equal(restored.arrays["obs"][:2], replay.arrays["obs"][:2])
    with pytest.raises(ValueError):
        load_replay(tmp_path / "replay.npz", capacity=4, obs_dim=64, action_dim=56, base_dim=8)
    learner.save(tmp_path / "learner.pt", extra={"contract": observation_contract("state_latent")})
    loaded = SACLearner.load(tmp_path / "learner.pt")
    np.testing.assert_array_equal(learner.act(a[None], base[None], deterministic=True),
                                  loaded.act(a[None], base[None], deterministic=True))


@pytest.mark.parametrize("outcome", ["timeout", "success", "failure", "rejected"])
@pytest.mark.parametrize("mode", ["state_only", "state_latent"])
def test_runner_successors_and_evaluation(monkeypatch, tmp_path, outcome, mode):
    run_fake(monkeypatch, tmp_path, outcome, mode)


@pytest.mark.parametrize("outcome,bad_request", [("timeout", 0), ("timeout", 1), ("timeout", 2),
                                                   ("success", 1), ("failure", 1)])
def test_runner_feature_failure_never_commits_incomplete(monkeypatch, tmp_path, outcome, bad_request):
    run_fake(monkeypatch, tmp_path, outcome, "state_latent", bad_request)


def run_fake(monkeypatch, tmp_path, outcome, mode, bad_request=None):
    import pressb.astribot_validation
    monkeypatch.setattr(pressb.astribot_validation, "initial_overlap_check", lambda *a, **kw: {"status": "clear"})
    monkeypatch.setattr(runner, "interpolate_target", lambda previous, target: [target])
    def validate(*a):
        if outcome == "rejected" and a[0][0, 0] > 0:
            raise ValueError("invalid policy command")
    monkeypatch.setattr(runner, "validate_prediction", validate)
    runtime = SimpleNamespace(q=np.zeros((1, 32), np.float32), qd=np.zeros((1, 32), np.float32),
                              world=SimpleNamespace(current_time=0., stage=None), last_images=[{}])
    runtime.prepare_wave = lambda *a: None
    runtime.render = lambda: [{}]
    ticks = []
    class Recorder:
        def __init__(self, runtime, *a):
            self.runtime = runtime
            self.command = np.zeros(32)
            self.env = SimpleNamespace(meta=None, prefix="")
            self.limits = np.tile([-100., 100.], (32, 1))
            self.records = []
            self.unexpected = []
            self.physics_step = 0
        def start(self): pass
        def close(self): pass
        def distance(self): return 0.1
        observe = runner.Recorder.observe
        def tick(self, target):
            ticks.append(target)
            self.physics_step += 1
            runtime.q += 1
            runtime.qd += 2
            runtime.world.current_time += 1 / 30
            self.records.append(dict(travel=0., force=0., light=False, distance=.1))
            if len(ticks) == 2 and outcome == "failure":
                raise runner.EpisodeStop("unexpected_contact")
            return len(ticks) == 2 and outcome == "success"
    monkeypatch.setattr(runner, "Recorder", Recorder)
    calls = []
    def predict(endpoint, images, state, seed):
        i = len(calls)
        calls.append((state.copy(), seed))
        result = dict(action8=np.full((7, 8), i * .01, np.float32), action32=np.zeros((7, 32)))
        if i != bad_request:
            result["vlm_latent"] = latent(i)
        return result
    monkeypatch.setattr(runner, "predict", predict)
    args = SimpleNamespace(observation_mode=mode, gamma=.995, shaping=1., endpoint="fake",
                           max_seconds=.3, exploration_probability=1., scale=[.01] * 8,
                           execute_horizon=7, status_every_requests=100)
    replay = ReplayBuffer(20, observation_contract(mode)["obs_dim"], 56, base_dim=8)
    learner = SimpleNamespace(act=lambda obs, base, **kw: np.zeros((1, 56), np.float32))
    counter = runner.EpisodeCounter(tmp_path)
    if bad_request is not None:
        with pytest.raises(ValueError, match="vlm_latent"):
            runner.run_episode(runtime, {}, {"fixed_action32": np.zeros(32)}, 1, args, learner, replay, counter)
        expected_rows = 1 if bad_request == 2 else 0
        assert len(replay) == expected_rows
        report = json.loads(next((tmp_path / "episodes").glob("*/report.json")).read_text())
        assert report["status"] == "infrastructure_error"
        expected_physical = "incomplete" if outcome == "timeout" and bad_request < 2 else {"timeout": "timeout", "success": "success", "failure": "failed"}[outcome]
        assert report["physical_outcome"]["status"] == expected_physical
        return
    report = runner.run_episode(runtime, {}, {"fixed_action32": np.zeros(32)}, 1, args, learner, replay, counter)
    assert len(replay) == (2 if outcome == "timeout" else 1)
    assert len(ticks) == {"timeout": 10, "rejected": 8, "success": 2, "failure": 2}[outcome]
    if mode == "state_latent":
        assert len(calls) == len(replay) + 1
        for i in range(len(replay)):
            np.testing.assert_array_equal(replay.arrays["obs"][i, :32], calls[i][0])
            np.testing.assert_array_equal(replay.arrays["next_obs"][i, :32], calls[i+1][0])
            np.testing.assert_array_equal(replay.arrays["obs"][i, 64:], normalize_latent(latent(i)))
            np.testing.assert_array_equal(replay.arrays["next_obs"][i, 64:], normalize_latent(latent(i+1)))
            np.testing.assert_allclose(replay.arrays["next_base_action"][i], (i+1) * .01)
        assert report["last_request"]["purpose"] == ("control" if outcome == "rejected" else "successor_only")
        assert [seed for _, seed in calls] == list(range(520003, 520003 + len(calls)))
    # A fresh episode has no pending row; evaluation must not touch replay.
    original = len(replay)
    runner.run_episode(runtime, {}, {"fixed_action32": np.zeros(32)}, 2, args, learner, replay, counter, train=False)
    assert len(replay) == original


def test_replay_semantic_version_rejected_even_at_same_width(tmp_path):
    contract = dict(observation=observation_contract("state_latent"), model_sha256="base")
    replay = ReplayBuffer(2, 2112, 56, base_dim=8)
    path = tmp_path / "replay.npz"
    save_replay(path, replay, contract=contract)
    kwargs = dict(capacity=2, obs_dim=2112, action_dim=56, base_dim=8)
    load_replay(path, **kwargs, expected_contract=contract)
    bad = deepcopy(contract)
    bad["observation"]["feature"]["contract_version"] = 2
    with pytest.raises(ValueError, match="contract mismatch"):
        load_replay(path, **kwargs, expected_contract=bad)
    save_replay(path, replay)
    with pytest.raises(ValueError, match="contract mismatch"):
        load_replay(path, **kwargs, expected_contract=contract)


def test_zero_residual_preserves_full_base_action():
    from pressb.online_rl.astribot import compose_joint8, expand_joint8
    base = np.linspace(-.3, .3, 56, dtype=np.float32).reshape(7, 8)
    fixed = np.arange(32, dtype=np.float32)
    np.testing.assert_array_equal(expand_joint8(compose_joint8(base, np.zeros(56)), fixed),
                                  expand_joint8(base, fixed))


def test_runner_critic_defaults_and_overrides():
    assert runner.critic_settings(SimpleNamespace(observation_mode='state_latent')) == ('fan_in', 3e-5)
    assert runner.critic_settings(SimpleNamespace(observation_mode='state_only')) == ('legacy', 3e-4)
    assert runner.critic_settings(SimpleNamespace(observation_mode='state_only',
                                  critic_init='fan_in', q_lr=1e-5)) == ('fan_in', 1e-5)
