"""Contracts for read-only stochastic baseline diagnostics."""
import numpy as np
import torch
from run_astribot_fast_iteration import ReadOnlyPolicy, digest, rms
from pressb.online_rl.astribot import compose_joint8
from pressb.online_rl.learner import LearnerConfig, SACLearner


def learner():
    return SACLearner(LearnerConfig(method='action_residual', obs_dim=64,
        base_dim=8, hidden_dim=16, seed=4, residual_scale=[.01]*7+[.025]))


def test_zero_and_mean_use_existing_action_contract():
    model = learner()
    obs, base = np.ones((2, 64), np.float32), np.ones((2, 7, 8), np.float32)
    zero = ReadOnlyPolicy(model, 'zero', None, 7).act(obs, base)
    assert np.array_equal(compose_joint8(base[0], zero[0]), base[0])
    mean = ReadOnlyPolicy(model, 'B_mean', None, 7).act(obs, base)
    np.testing.assert_array_equal(mean, model.act(obs, base, deterministic=True))


def test_unconditional_indices_ignore_observation_and_preserve_blocks():
    model = learner()
    bank = np.random.default_rng(4).uniform(-1, 1, (9, 7, 8)).astype(np.float32)
    first = ReadOnlyPolicy(model, 'unconditional', bank, 5)
    second = ReadOnlyPolicy(model, 'unconditional', bank, 5)
    for _ in range(4):
        a = first.act(np.zeros((1, 64)), np.zeros((1, 7, 8)))
        b = second.act(np.ones((1, 64)), np.ones((1, 7, 8)))
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a[0].reshape(7, 8), bank[first.indices[-1]])
    assert first.indices == second.indices


def test_sampling_private_rng_and_parameters_unchanged():
    model = learner()
    original = digest(model)
    a = ReadOnlyPolicy(model, 'B_sample', None, 71)
    b = ReadOnlyPolicy(model, 'B_sample', None, 71)
    obs, base = np.zeros((1, 64)), np.zeros((1, 7, 8))
    state = torch.get_rng_state().clone()
    x = a.act(obs, base)
    assert torch.equal(state, torch.get_rng_state())
    torch.rand(200)  # Simulator/diagnostics consuming global RNG cannot change residuals.
    np.testing.assert_array_equal(x, b.act(obs, base))
    assert digest(model) == original and model.updates == 0


def test_rms_uses_per_joint_scale_once():
    result = rms(np.ones((3, 7, 8)), [.01]*7+[.025])
    assert np.isclose(result['arm']['radians_rms'], .01)
    assert np.isclose(result['gripper']['radians_rms'], .025)
    assert result['all']['normalized_rms'] == 1
    assert rms(np.empty((0, 8)), [.01]*8) is None


def test_derived_initialization_resets_updates_optimizers_and_preserves_networks():
    from astribot_fast_learner import derive
    model = learner()
    model.updates = 100000
    derived = derive(model, lambda_a=.1, alpha=.005)
    assert digest(model) == digest(derived)
    assert derived.updates == 0
    assert not derived.actor_optimizer.state and not derived.q_optimizer.state
    assert derived.config.lambda_a == .1 and model.config.lambda_a == 0
    for x,y in zip(model.actor.parameters(),derived.actor.parameters()):
        assert x is not y


def test_incremental_budget_never_counts_imported_full_ring():
    from astribot_fast_learner import IncrementalReplay
    from pressb.online_rl.learner import ReplayBuffer
    import pytest
    storage = ReplayBuffer(3,64,56,base_dim=8)
    def add(replay):
        replay.add(np.zeros(64),np.zeros(56),0.,.99,np.zeros(64),False,False,
            base_action=np.zeros((7,8)), next_base_action=np.zeros((7,8)))
    for _ in range(3):
        add(storage)
    wrapper = IncrementalReplay(storage,2)
    assert len(wrapper) == 0 and len(storage) == 3
    add(wrapper)
    add(wrapper)
    assert len(wrapper) == 2 and len(storage) == 3
    with pytest.raises(RuntimeError):
        add(wrapper)


def test_gradient_diagnostics_preserve_rng_parameters_and_optimizer_grads():
    from astribot_fast_learner import gradient_diagnostics
    model = learner()
    batch = dict(obs=np.zeros((3,64)),next_obs=np.zeros((3,64)),action=np.zeros((3,56)),
        base_action=np.zeros((3,56)),next_base_action=np.zeros((3,56)),
        reward=np.zeros((3,1)),discount=np.ones((3,1))*.99,
        terminated=np.zeros((3,1)),truncated=np.zeros((3,1)))
    rng = torch.get_rng_state().clone()
    before = digest(model)
    result = gradient_diagnostics(model,batch)
    assert torch.equal(rng,torch.get_rng_state())
    assert digest(model) == before and model.updates == 0
    assert all(p.grad is None for p in model.actor.parameters())
    assert all(p.grad is None for p in model.qs.parameters())
    assert result['gradient_sum_max_error'] < 1e-5
    assert result['gradient_pairs']['value__regularization']['angle_degrees'] is None


def test_shared_nonzero_broadcast_and_critic_action_scale():
    from astribot_fast_learner import SharedLearner
    cfg = LearnerConfig(method='action_residual', obs_dim=64,base_dim=8,hidden_dim=16,
        auto_alpha=False, init_alpha=.035,residual_scale=[.01]*7+[.025])
    model = SharedLearner(cfg)
    residual = torch.arange(8).reshape(1,8).float()/8
    base = torch.ones((1,56))
    actual = model._critic_action(residual,base).reshape(1,7,8)
    expected = base.reshape(1,7,8)+residual[:,None,:]*torch.tensor(cfg.residual_scale)
    torch.testing.assert_close(actual,expected)
    x = model._actor_input(torch.zeros((2,64)),torch.zeros((2,56)))
    sampled, logp, _ = model.actor.sample(x)
    assert sampled.shape == (2,8) and logp.shape == (2,1)
    action = model.act(np.zeros((2,64)),np.zeros((2,56)),deterministic=True)
    assert action.shape == (2,56)
    np.testing.assert_array_equal(action.reshape(2,7,8)[:,0],action.reshape(2,7,8)[:,6])


def test_shared_checkpoint_preserves_actions_and_structure(tmp_path):
    from astribot_fast_learner import SharedLearner, load_candidate
    cfg = LearnerConfig(method='action_residual',obs_dim=64,base_dim=8,hidden_dim=16)
    model = SharedLearner(cfg)
    obs,base = np.zeros((2,64)),np.zeros((2,56))
    before = model.act(obs,base,deterministic=True)
    model.save(tmp_path/'learner.pt',extra=dict(status=dict(structure='shared')))
    restored = load_candidate(tmp_path/'learner.pt')
    np.testing.assert_array_equal(before,restored.act(obs,base,deterministic=True))
    assert restored.actor.mean.out_features == 8 and restored.action_dim == 56


def test_shared_update_and_restored_optimizer_continue_identically(tmp_path):
    from astribot_fast_learner import SharedLearner, load_candidate
    model = SharedLearner(LearnerConfig(method='action_residual',obs_dim=64,base_dim=8,
        hidden_dim=16,batch_size=3,auto_alpha=False,init_alpha=.035))
    batch = dict(obs=np.zeros((3,64)),next_obs=np.zeros((3,64)),action=np.zeros((3,56)),
        base_action=np.zeros((3,56)),next_base_action=np.zeros((3,56)),
        reward=np.zeros((3,1)),discount=np.ones((3,1))*.99,
        terminated=np.zeros((3,1)),truncated=np.zeros((3,1)))
    model.update(batch)
    model.update(batch)
    model.save(tmp_path/'learner.pt',extra=dict(status=dict(structure='shared')))
    model.update(batch)
    expected = digest(model)
    restored = load_candidate(tmp_path/'learner.pt')
    restored.update(batch)
    assert digest(restored) == expected
    assert restored.updates == 3


def test_auxiliary_pool_filters_invalid_labels_and_keeps_validation_out():
    from astribot_fast_auxiliary import LabelPool
    pool=LabelPool();pool.excluded={8}
    z=np.zeros((3,2048),np.float32)
    pool.add(0,1,z,np.array([[10.,-10.,30.],[0,0,31],[np.nan,0,0]]))
    pool.add(1,2,z[:1],np.array([[20.,0.,0.]]))
    pool.add(1,8,z,np.zeros((3,3)))
    assert pool.count()==2 and 8 not in pool.groups[1]
    assert pool.groups[0][1][1].tolist()==[[1.,-1.]]
    x,y=pool.sample(300)
    assert x.shape==(300,2048) and y.shape==(300,2)
    saved=pool.state_dict()
    next_batch=pool.sample(20)
    restored=LabelPool();restored.load_state_dict(saved)
    a,b=restored.sample(20)
    torch.testing.assert_close(a,next_batch[0]);torch.testing.assert_close(b,next_batch[1])


def test_auxiliary_encoders_are_independent_and_labels_cannot_enter_forward():
    from astribot_fast_auxiliary import AuxiliaryLearner,LabelPool
    cfg=LearnerConfig(method='action_residual',obs_dim=2112,base_dim=8,hidden_dim=16,
        critic_init='fan_in',q_lr=3e-5,auto_alpha=False,batch_size=3)
    model=AuxiliaryLearner(cfg)
    actor_ids={id(p) for p in model.actor.parameters()}
    critic_ids={id(p) for p in model.qs.parameters()}
    target_ids={id(p) for p in model.q_targets.parameters()}
    assert not actor_ids&critic_ids and not critic_ids&target_ids
    assert not any(p.requires_grad for p in model.q_targets.parameters())
    optimizer_ids={id(p) for g in model.actor_optimizer.param_groups for p in g['params']}
    assert optimizer_ids==actor_ids
    z=np.zeros((1,2112),np.float32);base=np.zeros((1,56),np.float32)
    before=model.act(z,base,deterministic=True)
    model.pool=LabelPool()
    model.pool.add(0,1,z[:,64:],np.array([[20.,0.,0.]]))
    model.pool.add(1,2,z[:,64:],np.array([[-20.,0.,0.]]))
    np.testing.assert_array_equal(before,model.act(z,base,deterministic=True))


def test_auxiliary_update_and_checkpoint_restore(tmp_path):
    from astribot_fast_auxiliary import AuxiliaryLearner,LabelPool
    from astribot_fast_learner import load_candidate
    cfg=LearnerConfig(method='action_residual',obs_dim=2112,base_dim=8,hidden_dim=16,
        critic_init='fan_in',q_lr=3e-5,auto_alpha=False,batch_size=3)
    model=AuxiliaryLearner(cfg,shared=True)
    model.pool=LabelPool()
    for source in [0,1]:model.pool.add(source,source+1,np.zeros((1,2048)),np.ones((1,3)))
    batch=dict(obs=np.zeros((3,2112)),next_obs=np.zeros((3,2112)),action=np.zeros((3,56)),
        base_action=np.zeros((3,56)),next_base_action=np.zeros((3,56)),reward=np.zeros((3,1)),
        discount=np.ones((3,1))*.99,terminated=np.zeros((3,1)),truncated=np.zeros((3,1)))
    model.update(batch);metrics=model.update(batch)
    assert metrics['actor_updated']==1 and metrics['actor_aux_loss']>=0
    model.save(tmp_path/'learner.pt',extra=dict(status=dict(structure='auxiliary_shared')))
    model.update(batch);expected=digest(model)
    restored=load_candidate(tmp_path/'learner.pt');restored.update(batch)
    assert digest(restored)==expected
    assert restored.pool.count()==2 and restored.shared


def test_auxiliary_historical_scene_partition_has_no_leakage():
    from prepare_astribot_fast_auxiliary import select_scenes
    from run_astribot_fast_iteration import SCENES
    contract,candidates,controlled,excluded=select_scenes()
    assert len(controlled)==60 and len(set(controlled))==60
    assert not set(candidates)&set(excluded)
    assert not set(candidates)&set(SCENES)
    assert set(candidates)<=set(contract['split']['train_source_episode_ids'])
