"""Actual controller targets, not future actions, condition the residual."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pressb.online_rl.astribot import command_joint8, observation, expand_joint8, ChunkJournal, save_replay, load_replay
from pressb.online_rl.astribot_features import FEATURE, observation_contract, validate_capability, validate_restore
from pressb.online_rl.learner import LearnerConfig, ReplayBuffer, SACLearner
import run_astribot_online_rl as runner

MODE = 'state_latent_raw_command'


def test_layout_and_compression():
    command = np.arange(32, dtype=np.float32) / 10
    expected = np.r_[command[[5, 8, 11, 13, 15, 17, 19]],
                     np.mean(command[[24,25,26,27,30,31]].astype(np.float64) * [-1,-1,1,1,1,-1])].astype(np.float32)
    np.testing.assert_array_equal(command_joint8(command), expected)
    obs = observation(np.ones(32), np.ones(32)*2, np.arange(2048, dtype=np.float32),
                      mode=MODE, previous_command8=expected)
    assert obs.shape == (2120,) and obs.dtype == np.float32
    np.testing.assert_array_equal(obs[-8:], expected)
    np.testing.assert_array_equal(obs[64:2112], np.arange(2048))
    validate_capability({'latent': FEATURE}, MODE)
    with pytest.raises(ValueError): validate_capability({}, MODE)


@pytest.mark.parametrize('bad', [None, np.zeros(7), np.zeros((1,8)), np.full(8,np.nan),np.full(8,np.inf)])
def test_bad_command(bad):
    with pytest.raises(ValueError, match='previous_command8'):
        observation(np.zeros(32),np.zeros(32),np.zeros(2048,np.float32),mode=MODE,previous_command8=bad)


def test_network_and_restore(tmp_path):
    torch.set_num_threads(1)
    obs = observation(np.zeros(32),np.zeros(32),np.ones(2048,np.float32),mode=MODE,previous_command8=np.arange(8))
    base = np.zeros((7,8),np.float32)
    cfg=LearnerConfig(method='action_residual',obs_dim=2120,base_dim=8,hidden_dim=8,batch_size=2)
    learner=SACLearner(cfg)
    replay=ReplayBuffer(4,2120,56,base_dim=8)
    journal=ChunkJournal(replay,observation_mode=MODE)
    for _ in range(2):
        journal.commit(journal.start(obs,np.zeros(56),base),next_obs=obs,next_base=base,steps=7)
    seen=[]
    hook=learner.actor.net[0].register_forward_pre_hook(lambda _,args: seen.append(args[0].detach().clone()))
    action=learner.act(obs[None],base[None],deterministic=True)
    hook.remove()
    assert seen[0].shape == (1,2176)
    np.testing.assert_array_equal(seen[0][0,2112:2120],np.arange(8))
    seen=[]
    hook=learner.qs.register_forward_pre_hook(lambda _,args: seen.append(args[0].detach().clone()))
    metrics=learner.update(replay.sample(2));hook.remove()
    assert np.isfinite(list(metrics.values())).all()
    np.testing.assert_array_equal(seen[0][:,-8:],np.tile(np.arange(8),(2,1)))
    contract={'observation':observation_contract(MODE)}
    learner.save(tmp_path/'learner.pt',extra={'contract':contract})
    loaded=SACLearner.load(tmp_path/'learner.pt')
    np.testing.assert_array_equal(learner.act(obs[None],base[None],deterministic=True),loaded.act(obs[None],base[None],deterministic=True))
    validate_restore(loaded.checkpoint_extra['contract'],contract,MODE)
    save_replay(tmp_path/'replay.npz',replay,contract=contract)
    kw=dict(capacity=4,obs_dim=2120,action_dim=56,base_dim=8,expected_contract=contract)
    restored=load_replay(tmp_path/'replay.npz',**kw)
    np.testing.assert_array_equal(restored.arrays['obs'][:2],replay.arrays['obs'][:2])
    bad=deepcopy(contract);bad['observation']['previous_command']['source']='future_action'
    with pytest.raises(ValueError): load_replay(tmp_path/'replay.npz',**dict(kw,expected_contract=bad))
    with pytest.raises(ValueError): validate_restore({'observation':observation_contract('state_latent_raw')},contract,MODE)
    with pytest.raises(ValueError): load_replay(tmp_path/'replay.npz',**dict(kw,obs_dim=2112))


@pytest.mark.parametrize('outcome',['timeout','success','failure','rejected','command_changed'])
def test_actual_commands_and_reset(monkeypatch,tmp_path,outcome):
    import pressb.astribot_validation
    monkeypatch.setattr(pressb.astribot_validation,'initial_overlap_check',lambda *a,**kw: {'status':'clear'})
    # Real four-tick interpolation and real Recorder.observe; physics is deterministic fake.
    runtime=SimpleNamespace(q=np.zeros((1,32),np.float32),qd=np.zeros((1,32),np.float32),
                            targets=np.zeros((1,32),np.float32),world=SimpleNamespace(current_time=0.,stage=None),last_images=[{}])
    episodes=[]
    def prepare(*args):
        runtime.targets[0]=expand_joint8(np.full((7,8),.1*(len(episodes)+1),np.float32),np.zeros(32))[0]
        episodes.append([])
    runtime.prepare_wave=prepare;runtime.render=lambda:[{}]
    recorders=[]
    class Recorder:
        def __init__(self,runtime,*args):
            self.runtime=runtime;self.command=runtime.targets[0].copy()
            self.env=SimpleNamespace(meta=None,prefix='');self.limits=np.tile([-100,100],(32,1))
            self.physics_step=0;self.records=[];self.unexpected=[];recorders.append(self)
        def start(self): pass
        def close(self): pass
        def distance(self): return .1
        observe=runner.Recorder.observe
        def tick(self,target):
            self.command=np.asarray(target,np.float32).copy();episodes[-1].append(self.command.copy())
            self.physics_step+=1;runtime.world.current_time+=1/120
            self.records.append(dict(travel=0.,force=0.,light=False))
            if self.physics_step==3 and outcome=='failure': raise runner.EpisodeStop('unexpected_contact')
            return self.physics_step==3 and outcome=='success'
    monkeypatch.setattr(runner,'Recorder',Recorder)
    calls=[]
    def predict(*args):
        calls.append(command_joint8(recorders[-1].command))
        if outcome=='command_changed': recorders[-1].command+=1
        base=np.full((7,8),.4+len(calls)*.01,np.float32)
        return dict(action8=base,action32=expand_joint8(base,np.zeros(32)),vlm_latent=np.ones(2048,np.float32))
    monkeypatch.setattr(runner,'predict',predict)
    original_validate=runner.validate_prediction
    def validate(*args):
        if outcome=='rejected' and len(calls)==2: raise ValueError('reject')
        return original_validate(*args)
    monkeypatch.setattr(runner,'validate_prediction',validate)
    args=SimpleNamespace(observation_mode=MODE,gamma=.995,shaping=0.,endpoint='fake',max_seconds=.3,
                         exploration_probability=1.,scale=[.01]*8,execute_horizon=7,status_every_requests=100)
    learner=SimpleNamespace(action_dim=56,act=lambda *a,**kw:np.full((1,56),.5,np.float32))
    replay=ReplayBuffer(20,2120,56,base_dim=8);counter=runner.EpisodeCounter(tmp_path)
    if outcome=='command_changed':
        with pytest.raises(RuntimeError,match='command changed'):
            runner.run_episode(runtime,{}, {'fixed_action32':np.zeros(32)},1,args,learner,replay,counter)
        assert len(replay)==0
        return
    runner.run_episode(runtime,{}, {'fixed_action32':np.zeros(32)},1,args,learner,replay,counter)
    assert len(replay)==(2 if outcome=='timeout' else 1)
    np.testing.assert_array_equal(calls[0],command_joint8(episodes[0][0]))
    for i in range(len(replay)):
        np.testing.assert_array_equal(replay.arrays['obs'][i,-8:],calls[i])
        np.testing.assert_array_equal(replay.arrays['next_obs'][i,-8:],calls[i+1])
    np.testing.assert_array_equal(calls[-1],command_joint8(episodes[0][-1]))
    if outcome=='failure':
        # Failure on interpolation tick 2: never substitute the intended endpoint.
        expected=np.float32(.1)+(np.float32(.415)-np.float32(.1))*np.float32(.5)
        np.testing.assert_allclose(calls[-1],expected,atol=1e-7)
        assert len(episodes[0])==3
    count=len(calls);before=len(replay)
    runner.run_episode(runtime,{}, {'fixed_action32':np.zeros(32)},2,args,learner,replay,counter,train=False)
    assert len(replay)==before
    np.testing.assert_array_equal(calls[count],command_joint8(episodes[1][0]))
    np.testing.assert_allclose(calls[count],.2)
