"""Acceptance gate for 4500 -> 4502 -> 4503 before launching the formal run."""
import argparse
import json
from pathlib import Path
import torch


def check(run):
    def read(name):return json.loads((run/name).read_text())
    completed=read('segment_completed.json')
    assert completed['success'] and completed['step']==4503
    loading=read('pretrained_loading.json')
    assert loading['optimizer_restored'] and loading['source_step']==4502
    assert loading['optimizer_state_evidence']['min_step']==4502
    assert loading['optimizer_state_evidence']['max_step']==4502
    changes=read('parameter_update_evidence.json')
    assert all(v['changed_elements']>0 for v in changes['probes'].values())
    lineage=read('continuation.json')
    assert lineage['parent_progress']['step']==4500
    assert lineage['learning_rate_schedule']['warmup_steps']==100
    assert lineage['learning_rate_schedule']['end_step']==5500
    assert lineage['sampling_change']['current']['version']=='gripper_episode_v1'
    baseline=read('gripper_baseline.json')
    assert baseline['step']==4500 and baseline['overall']['anchors']==480
    assert baseline['gripper']['transitions']['overall']['onset_count']==14
    metrics=read('validation_004503.json')
    assert metrics['overall']['anchors']==480
    state=torch.load(run/'checkpoints/last/training.pt',map_location='cpu',weights_only=False,mmap=True)
    assert state['step']==4503 and state['scheduler']['last_epoch']==4503
    assert state['gripper_state']['baseline']==baseline
    rates=state['scheduler']['_last_lr']
    assert all(rate>0 for rate in rates)
    reports=[json.loads(p.read_text()) for p in sorted(run.glob('sampling_epoch_*.json'))]
    assert len(reports)==2 and reports[0]['epoch']==reports[1]['epoch']
    assert reports[0]['sequence_sha256']==reports[1]['sequence_sha256']
    assert reports[1]['rank_start_index']-reports[0]['rank_start_index']==16
    assert [reports[0]['groups'][k]['anchors'] for k in ['initiate_close','hold_open','other']]==[952,4754,31827]
    samples=[json.loads(line) for line in (run/'train.jsonl').read_text().splitlines()]
    # Log frequency may omit smoke steps; parameter evidence and finite loss guards
    # in the trainer still apply. The submitter requests logging each step.
    assert [r['step'] for r in samples]==[4501,4502,4503]
    report=dict(status='passed',resumed_from=4502,final_step=4503,baseline_onsets=14,
                sampling_resume_verified=True,optimizer_moments_restored=True,parameter_updates=changes['probes'])
    (run/'smoke_acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path)
    check(p.parse_args().run)
