import copy
import os
import json
from pathlib import Path
import numpy as np
import pytest
from starVLA.training.gripper_sampling import (
    SAMPLING, DIAGNOSTICS, GripperDistributedSampler, sampling_groups,
    transition_anchors, gripper_record, gripper_metrics, diagnostics_report, gripper_candidate)
from starVLA.training.train_s1b_real import (
    training_contract, validate_continuation_contract, lr_functions, prune_checkpoints)


class Dataset:
    def __init__(self):
        self._episodes = {1: dict(dataset_from_index=0,length=40), 2: dict(dataset_from_index=40,length=80)}
        self._anchors = [(1,np.arange(34)), (2,np.arange(74))]
        self._states=np.zeros((120,10));self._actions=np.zeros((120,10))
        for start,stop in [(0,40),(40,120)]:
            self._actions[start+15:stop,7]=1
            self._states[start+18:stop,7]=100
        self._states[:,7]=(self._states[:,7]-80)/35
    def __len__(self):return sum(len(x) for _,x in self._anchors)
    def state_raw(self,x):
        x=x.copy();x[:,7]=x[:,7]*35+80;return x
    def action_raw(self,x):return x


def test_partition_and_future_targets():
    d=Dataset();g=sampling_groups(d)
    flat=[int(x) for group in g for ids in group.values() for x in ids]
    assert sorted(flat)==list(range(len(d)))
    assert 9 in g[0][1]  # first target still open, seventh closes
    assert 8 in g[1][1]
    assert 18 in g[2][1]
    assert set(g[0][1])==set(range(9,18))
    d._anchors[0]=(1,np.arange(35))
    with pytest.raises(ValueError,match='cross-episode'):sampling_groups(d)


def test_empty_group_and_only_approved_anchors():
    d=Dataset();d._anchors=[(1,np.array([9,10,18])),(2,np.array([9,10,18]))]
    with pytest.raises(ValueError,match='Missing sampling'):sampling_groups(d)
    d._anchors[0]=(1,np.array([0,9,18]));g=sampling_groups(d)
    assert sorted(x for group in g for ids in group.values() for x in ids)==list(range(6))


def test_ddp_and_resume_and_distribution():
    d=Dataset();samplers=[GripperDistributedSampler(d,8,r) for r in range(8)]
    for s in samplers:s.set_epoch(7)
    expected,labels=samplers[0].global_sequence()
    slices=[list(s) for s in samplers]
    assert np.array_equal(np.array(slices).T.ravel(),expected)
    for r,s in enumerate(samplers):
        s.set_start_index(5);assert list(s)==slices[r][5:]
        assert len(s)==len(slices[r])-5
    assert len({s.report()['sequence_sha256'] for s in samplers})==1
    counts=np.zeros(3);epcounts={1:0,2:0}
    for epoch in range(500):
        s=samplers[0];s.set_epoch(epoch);seq,labels=s.global_sequence()
        counts+=np.bincount(labels,minlength=3)
        for value in seq[labels==2]:epcounts[1 if value<34 else 2]+=1
    assert np.allclose(counts/counts.sum(),[.25,.25,.5],atol=.01)
    assert abs(epcounts[1]/sum(epcounts.values())-.5)<.02
    with pytest.raises(ValueError):samplers[0].set_start_index(-1)


def test_transition_coverage():
    extra,onsets=transition_anchors(Dataset())
    assert onsets=={(1,15),(2,15)}
    assert extra==[(sid,f) for sid in [1,2] for f in range(8,23)]


def row(pred,target,open_state=True,onset=False,source='elevator1',base=True,transition=True):
    return dict(gripper=gripper_record(pred,target,open_state,onset),error=np.zeros((7,10)).tolist(),
                episode=1,frame=15,source_batch=source,is_base=base,is_transition=transition)


def test_metrics_timing_and_misses():
    target=[0,0,1,1,1,1,1]
    rows=[row([1]*7,target),row(target,target),row([0]*7,target),
          row([0]*7,[0]*7,source='elevator2'),row([1]*7,[1]*7,onset=True)]
    m=gripper_metrics(rows)
    assert m['delay_frames']==[-2,0]
    assert m['missed_transitions']==1
    assert m['onset_first_close_recall']==1
    assert m['future_close_recall']==.75
    assert m['first_close_count']==1
    assert m['false_close_rate']==2/13
    report=diagnostics_report(rows)
    assert report['base']['by_source_batch']['elevator2']['false_close_rate']==0
    # A close only before the correct target is not a successful future close.
    assert gripper_metrics([row([1,0,0,0,0,0,0],target)])['future_close_recall']==0


def test_selection_guardrails():
    r=row([1]*7,[1]*7,onset=True)
    r['error']=np.ones((7,10)).tolist()
    # Include open targets to make the false-positive guard defined.
    r2=row([0]*7,[0]*7);r2['error']=np.ones((7,10)).tolist()
    base={'gripper':diagnostics_report([r,r2]),'overall':{'selection_score_normalized':.1}}
    assert gripper_candidate(base,base)==[1,-.1]
    bad=copy.deepcopy(base);bad['gripper']['base']['overall']['false_close_rate']=.02
    assert gripper_candidate(bad,base) is None
    bad=copy.deepcopy(base);bad['gripper']['base']['overall']['head_mae_rad'][0]=1.11
    assert gripper_candidate(bad,base) is None


def config():
    # Parent config is available in the source tree without requiring model files.
    from omegaconf import OmegaConf
    root=Path(os.environ.get("S1B_PROJECT",Path(__file__).resolve().parents[3]))
    return OmegaConf.to_container(OmegaConf.load(root/'configs/vlajepa_s1b_real_joint10_mixed_qwen3_5_8gpu_6000_hpc2.yaml'),resolve=True)


def test_contract_and_schedule():
    c=config();parent,fp=training_contract(c)
    changed=copy.deepcopy(c);changed['trainer'].update(gripper_sampling=SAMPLING,gripper_diagnostics=DIAGNOSTICS,
        max_train_steps=5500,num_warmup_steps=100,eval_interval=100)
    current,other=training_contract(changed)
    assert fp!=other
    validate_continuation_contract(parent,current)
    bad=copy.deepcopy(current);bad['trainer']['learning_rate']['action_model']=.01
    with pytest.raises(ValueError):validate_continuation_contract(parent,bad)
    bad=copy.deepcopy(changed);bad['trainer']['gripper_sampling']={'version':'unknown'}
    with pytest.raises(ValueError):training_contract(bad)
    schedule=dict(kind='checkpoint_lr_linear_warmup_cosine',start_step=4500,warmup_steps=100,end_step=5500,
                  start_factors=[.2,.3,.4],end_factor=.1)
    funcs=lr_functions(5500,100,schedule)
    assert [f(4500) for f in funcs]==pytest.approx([.2,.3,.4])
    assert [f(4600) for f in funcs]==pytest.approx([1,1,1])
    assert [f(5500) for f in funcs]==pytest.approx([.1,.1,.1])


def test_best_gripper_retention(tmp_path):
    for step in [4600,4700,4800]:(tmp_path/f'step_{step:06d}').mkdir()
    (tmp_path/'best_gripper').symlink_to('step_004600')
    (tmp_path/'last').symlink_to('step_004800')
    prune_checkpoints(tmp_path)
    assert (tmp_path/'step_004600').is_dir()
    assert not (tmp_path/'step_004700').exists()


def test_evaluation_preserves_base_and_adds_diagnostics(monkeypatch,tmp_path):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import torch
    from starVLA.training import train_s1b_real as trainer
    d=Dataset();d.root=tmp_path
    def get_frame(sid,frame):
        i=d._episodes[sid]['dataset_from_index']+frame
        return dict(image=[],lang='press',state=d._states[i:i+1],action=d._actions[i:i+7])
    d.get_frame=get_frame
    class Model:
        def eval(self):pass
        def predict_action(self,*args,**kwargs):return {'normalized_actions':np.zeros((1,7,10))}
        def __call__(self,example):return {'action_loss':torch.tensor(.1),'wm_loss':torch.tensor(.2)}
    monkeypatch.setattr(torch.random,'fork_rng',lambda **kwargs:nullcontext())
    monkeypatch.setattr(torch,'autocast',lambda *args,**kwargs:nullcontext())
    monkeypatch.setattr(trainer.dist,'all_gather_object',lambda out,rows:out.__setitem__(0,rows))
    anchors=[(1,0),(1,20),(2,0),(2,20)]
    base=trainer.evaluate(Model(),d,anchors,0,1,SimpleNamespace(index=0),42,0,tmp_path)
    new=trainer.evaluate(Model(),d,anchors,0,1,SimpleNamespace(index=0),42,1,tmp_path,DIAGNOSTICS)
    assert base['overall']==new['overall']
    assert new['gripper']['base']['overall']['anchors']==4
    assert new['gripper']['transitions']['overall']['anchors']==30
    assert new['gripper']['transitions']['overall']['onset_count']==2
