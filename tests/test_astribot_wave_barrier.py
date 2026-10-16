"""A short episode must keep recording while its peer finishes the same wave."""
import time
from types import SimpleNamespace
import numpy as np
from pressb.astribot_collection import runtime as module


def test_waiting_environment_records_every_tick_before_shared_reset(tmp_path,monkeypatch):
    world=SimpleNamespace(ticks=0,resets=0)
    def step(render=False):world.ticks+=1
    def reset():world.resets+=1
    world.step=step;world.reset=reset
    records={0:[],1:[]}
    def episode(env):
        for _ in range(2 if env.index==0 else 5):
            yield ('step',np.zeros(32));records[env.index].append(world.ticks)
        while not (yield ('reset',np.zeros(32))):
            yield ('step',np.zeros(32));records[env.index].append(world.ticks)
        assert world.resets==1
        env.report['status']='passed'
    monkeypatch.setattr(module,'episode',episode)
    rt=object.__new__(module.Runtime);rt.world=world;rt.timings={};rt.shared=tmp_path/'shared'
    rt.targets=np.zeros((2,32));rt.defaults=rt.targets.copy();rt.tensor=lambda a:a;rt.refresh=lambda:None
    rt.view=SimpleNamespace(set_joints_default_state=lambda **k:None,set_joint_position_targets=lambda x:None)
    rt.envs=[SimpleNamespace(index=i,output=tmp_path/str(i),writers={},report={},started=time.perf_counter()) for i in range(2)]
    for e in rt.envs:e.output.mkdir()
    rt.prepare_wave=lambda configs,outputs:None
    rows=rt.run_wave([{},{}],[e.output for e in rt.envs])
    assert records[0]==records[1]==[1,2,3,4,5]
    assert world.resets==1 and all(r['status']=='passed' for r in rows)
