import numpy as np
import pytest
import torch
from pressb.astribot_kinematics import AstribotKinematics, UnreachablePose
from pressb.online_rl.astribot_xyz import compose_xyz
from pressb.online_rl.learner import LearnerConfig, SACLearner, ReplayBuffer


def fixture():
    kin = AstribotKinematics()
    q = (kin.lower + kin.upper) / 2
    q[3] += .5
    return kin, np.tile(np.r_[q, .9], (7, 1)).astype(np.float32), np.eye(4)


def test_zero_exact_and_invalid():
    kin, base, tcp = fixture()
    got, d = compose_xyz(base, np.zeros(21), tcp, base[0])
    np.testing.assert_array_equal(got, base)
    assert d['attempts'] == 0
    with pytest.raises(ValueError):
        compose_xyz(base, np.full(21, np.nan), tcp, base[0])


@pytest.mark.parametrize('axis', range(3))
@pytest.mark.parametrize('sign', [-1, 1])
def test_xyz_pose(axis, sign):
    kin, base, tcp = fixture()
    action = np.zeros((7, 3)); action[:, axis] = sign * .1
    got, d = compose_xyz(base, action, tcp, base[0], kin=kin)
    assert d['factor'] > 0
    assert d['position_error_m'] <= .0005
    assert d['rotation_error_rad'] <= .01
    np.testing.assert_array_equal(got[:, 7], base[:, 7])
    assert np.max(np.abs(got[:, :7] - base[:, :7])) <= .1
    np.testing.assert_array_equal(got, compose_xyz(base, action, tcp, base[0], kin=kin)[0])


def test_fallback_and_programming_error():
    kin, base, tcp = fixture()
    def unreachable(*a): raise UnreachablePose('test')
    kin.solve = unreachable
    got, d = compose_xyz(base, np.ones(21), tcp, base[0], kin=kin)
    np.testing.assert_array_equal(got, base)
    assert d['factor'] == 0 and d['attempts'] == 4
    def bug(*a): raise ValueError('bug')
    kin.solve = bug
    with pytest.raises(ValueError, match='bug'):
        compose_xyz(base, np.ones(21), tcp, base[0], kin=kin)


def test_learning_contract(tmp_path):
    torch.set_num_threads(1)
    cfg = LearnerConfig(method='action_residual', base_dim=8, obs_dim=2120, residual_mode='astribot_xyz', residual_scale=(.01,)*3)
    learner = SACLearner(cfg)
    assert learner.action_dim == 21 and learner.target_entropy == -10.5
    assert learner.actor.net[0].in_features == 2176
    action = torch.zeros(2, 21, requires_grad=True)
    base = torch.randn(2, 56)
    values = learner.qs(torch.randn(2, 2120), learner._critic_action(action, base))
    values.sum().backward()
    assert action.grad.abs().sum() > 0
    assert learner.qs.net[0].weight.shape[-2] == 2197
    replay = ReplayBuffer(2, 2120, 21, base_dim=8)
    assert replay.arrays['action'].shape == (2, 21)
    learner.save(tmp_path/'learner.pt')
    restored = SACLearner.load(tmp_path/'learner.pt')
    assert restored.config.residual_mode == 'astribot_xyz'
    with pytest.raises(ValueError):
        LearnerConfig(method='action_residual', base_dim=8, obs_dim=2120, residual_mode='astribot_xyz', lambda_s=1.)


def test_constraints_backtrack_whole_chunk():
    kin, base, tcp = fixture()
    original = kin.solve
    calls = 0
    def jumping(target, seed, offset):
        nonlocal calls
        calls += 1
        q = original(target, seed, offset)
        if calls <= 7:
            q[0] = seed[0] + .11
        return q
    kin.solve = jumping
    action = np.zeros((7, 3)); action[:, 0] = .1
    got, d = compose_xyz(base, action, tcp, base[0], kin=kin)
    assert d['factor'] == .5 and d['attempts'] == 2
    assert np.max(np.abs(np.diff(got[:, :7] - base[:, :7], axis=0))) <= .03


def test_fixed_joint_expansion():
    from pressb.online_rl.astribot import expand_joint8, FIXED
    kin, base, tcp = fixture()
    got, _ = compose_xyz(base, np.zeros(21), tcp, base[0])
    fixed = np.zeros(32, np.float32)
    expanded = expand_joint8(got, fixed)
    np.testing.assert_array_equal(expanded[:, FIXED], np.zeros((7, len(FIXED))))


def test_checkpoint_pair_publication(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    import run_astribot_online_rl as runner
    cfg = LearnerConfig(method='action_residual', base_dim=8, obs_dim=64,
                        residual_mode='astribot_xyz', hidden_dim=8)
    learner = SACLearner(cfg)
    replay = ReplayBuffer(2, 64, 21, base_dim=8)
    runner.save_state(tmp_path, learner, replay, {'episodes': 0})
    first = (tmp_path/'current').resolve()
    assert (tmp_path/'learner.pt').resolve().parent == first
    assert (tmp_path/'replay.npz').resolve().parent == first
    def fail(*args, **kwargs): raise OSError('disk failure')
    monkeypatch.setattr(runner, 'save_replay', fail)
    with pytest.raises(OSError, match='disk failure'):
        runner.save_state(tmp_path, learner, replay, {'episodes': 1})
    assert (tmp_path/'current').resolve() == first
    assert SACLearner.load(tmp_path/'learner.pt').checkpoint_extra['status']['episodes'] == 0


def test_xyz_journal():
    from pressb.online_rl.astribot import ChunkJournal
    replay = ReplayBuffer(2, 64, 21, base_dim=8)
    journal = ChunkJournal(replay, action_dim=21)
    row = journal.start(np.zeros(64), np.zeros(21), np.zeros((7, 8)))
    journal.commit(row, next_obs=np.zeros(64), next_base=np.zeros((7, 8)), steps=7)
    assert len(replay) == 1
    evaluation = ChunkJournal(None, train=False, action_dim=21)
    evaluation.start(np.zeros(64), np.zeros(21), np.zeros((7, 8)))
