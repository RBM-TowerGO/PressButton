import json
from pathlib import Path
import numpy as np
import pytest
from pressb.astribot_collection.contract import (CAMERAS,JOINT_NAMES,create_manifest,extend_manifest,
    candidate_block,ordered_successes,training_arrays,commit_episode,validate_commit)

def test_frozen_rng_extension_and_order(tmp_path):
    m=create_manifest(tmp_path,{},seed=20261014)
    rng=np.random.Generator(np.random.PCG64(20261014))
    xy=rng.uniform([-.18,-.28],[-.02,.12],(512,2))+[-.12,.28];yaw=rng.uniform(0,10,512)
    np.testing.assert_array_equal([[r['x_m'],r['y_m']] for r in m['candidates']],xy)
    np.testing.assert_array_equal([r['yaw_deg'] for r in m['candidates']],yaw)
    ext=extend_manifest(tmp_path)
    assert ext['candidates'][:512]==m['candidates']
    assert ext['candidates'][512:]==candidate_block(rng,513)
    with pytest.raises(ValueError):extend_manifest(tmp_path)

def test_out_of_order_selection_waits_for_lower_ids():
    rows={2:dict(candidate_id=2,outcome='success'),3:dict(candidate_id=3,outcome='success')}
    assert ordered_successes(rows,1)==[]
    rows[1]=dict(candidate_id=1,outcome='physical_failure')
    assert ordered_successes(rows,1)==[rows[2]]

def fixture_trace():
    n=17;step=np.arange(n);trace=dict(q=np.tile(step[:,None],(1,32)),qd=np.zeros((n,32)),
      target=np.tile((100+step)[:,None],(1,32)).astype(float),epoch=np.zeros(n,int),simulation_time=(step+1)/120,
      phase=np.array(['press']*n),travel=np.zeros((n,1)),force=np.zeros((n,1)),light=np.zeros((n,1),bool))
    trace['travel'][6:]=.0015;trace['force'][6:]=.03;trace['light'][6:]=True
    trace['target'][6:9]=108
    rows=[dict(step=i,epoch=0,simulation_time=(i+1)/120,cameras={c:(i+1)/120 for c in CAMERAS}) for i in range(0,n,4)]
    return trace,rows

def test_actions_are_future_commands_and_terminal_repeats_previous():
    trace,rows=fixture_trace();a=training_arrays(trace,rows,list(JOINT_NAMES))
    np.testing.assert_array_equal(a['physical_steps'],[0,4,8])
    np.testing.assert_array_equal(a['action'][:,0],[104,108,108])
    np.testing.assert_array_equal(a['state'][:,0],[0,4,8])
    assert a['trigger_step']==6 and a['terminal_video_frame']==2

def test_schema_maps_names_and_rejects_gaps_and_stale_timestamps():
    trace,rows=fixture_trace()
    names=list(reversed(JOINT_NAMES));trace['q']+=np.arange(32)
    a=training_arrays(trace,rows,names)
    assert a['state'][0,0]==31
    with pytest.raises(ValueError):training_arrays(trace,rows[::2],names)
    rows[1]['cameras']['head']=0
    with pytest.raises(ValueError):training_arrays(trace,rows,names)

def test_atomic_commit_tamper_and_duplicate(tmp_path):
    src=tmp_path/'staging';src.mkdir();(src/'test.bin').write_bytes(b'original')
    dest=tmp_path/'committed'
    with pytest.raises(ValueError):commit_episode(src,dest,dict(status='failed'))
    commit_episode(src,dest,dict(status='passed'));validate_commit(dest)
    src.mkdir();(src/'other.bin').write_bytes(b'new')
    with pytest.raises(FileExistsError):commit_episode(src,dest,dict(status='passed'))
    (dest/'test.bin').write_bytes(b'corrupt')
    with pytest.raises(ValueError):validate_commit(dest)

def test_terminal_retract_is_rejected_even_when_lamp_is_on():
    trace,rows=fixture_trace();trace['phase'][8]='retract'
    with pytest.raises(ValueError,match='Retract'):training_arrays(trace,rows,list(JOINT_NAMES))
    trace['phase'][8]='press';trace['target'][8,0]+=.001
    with pytest.raises(ValueError,match='Contact command'):training_arrays(trace,rows,list(JOINT_NAMES))

def test_duplicate_commit_and_bad_video_cannot_be_classified_success(tmp_path):
    from pressb.astribot_collection.dispatch import classify
    raw=tmp_path/'episode';raw.mkdir()
    (raw/'report.json').write_text(json.dumps({'termination_reason':'initial_collision'}))
    assert classify({'status':'failed','output':str(raw)})=='physical_failure'
    (raw/'report.json').write_text(json.dumps({'status':'passed'}))
    assert classify({'status':'audit_failed','output':str(raw)})=='infrastructure_error'
    assert classify({'status':'audit_failed','output':str(raw),'physical_audit':{'status':'failed'}})=='physical_failure'
    with pytest.raises(FileNotFoundError):classify({'status':'accepted','output':str(raw)})

def test_production_gate_rejects_stale_source_and_missing_evidence(tmp_path,monkeypatch):
    from pressb.astribot_collection import dispatch
    monkeypatch.setattr(dispatch,'source_files',lambda root:{'source.py':'current_hash'})
    gate=tmp_path/'gate.json';gate.write_text(json.dumps({'status':'passed','source_sha256':{'source.py':'old_hash'}}))
    with pytest.raises(ValueError,match='stale'):dispatch.validate_gate(gate,tmp_path,[0,1],2,'cpu')

def test_replay_uses_exported_future_targets_and_preserves_return_suffix():
    from pressb.astribot_collection.replay import reconstruct_targets
    raw=np.zeros((13,32),np.float32);raw[9:]=-.5
    a=dict(physical_steps=np.array([0,4,8]),action=np.array([[.4]*32,[.8]*32,[.8]*32],np.float32))
    result=reconstruct_targets(raw,a)
    np.testing.assert_allclose(result[:9,0],np.arange(9)*.1,atol=1e-7)
    np.testing.assert_array_equal(result[9:],raw[9:])
    np.testing.assert_array_equal(raw[:9],0)

def test_replay_strata_are_unique_and_cover_full_yaw_range():
    from pressb.astribot_collection.replay import stratified_replays
    rng=np.random.Generator(np.random.PCG64(20261014));candidates=candidate_block(rng,count=400)
    selected=[dict(candidate_id=r['candidate_id'],directory=f'/fixture/{r["candidate_id"]}') for r in candidates]
    rows=stratified_replays(selected,candidates)
    assert len(rows)==len({r['candidate_id'] for r in rows})==20
    assert min(r['sample']['yaw_deg'] for r in rows)<2
    assert max(r['sample']['yaw_deg'] for r in rows)>8
    assert rows==stratified_replays(list(reversed(selected)),candidates)

@pytest.mark.parametrize('complete',[False,True])
def test_killed_process_recovery_never_accepts_partial_payload(tmp_path,complete):
    import os,subprocess,sys
    from pressb.astribot_collection.dispatch import recover_journal
    worker=tmp_path/'worker';worker.mkdir();job=dict(id='candidate_000001_attempt_01',candidate_id=1,attempt=1)
    row=dict(candidate_id=1,attempt=1,outcome='queued',worker=str(worker),job_id=job['id'],configuration=job)
    code='''import json,os,signal,sys
from pathlib import Path
from pressb.astribot_collection.contract import atomic_json,commit_episode
worker=Path(sys.argv[1]);job=json.loads(sys.argv[2]);complete=sys.argv[3]=='True'
atomic_json(worker/'started_jobs.json',[job['id']])
staging=worker/'episodes'/(job['id']+'.inprogress');staging.mkdir(parents=True)
(staging/'payload.bin').write_bytes(b'fixture payload, not robot data')
atomic_json(staging/'job.json',job)
if complete:commit_episode(staging,worker/'episodes'/job['id'],dict(status='passed',scope='atomicity_test_fixture'))
os.kill(os.getpid(),signal.SIGKILL)
'''
    result=subprocess.run([sys.executable,'-c',code,str(worker),json.dumps(job),str(complete)],env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'src')})
    assert result.returncode==-9
    recover_journal([row])
    assert row['outcome']==('success' if complete else 'infrastructure_error')
    if complete:
        (Path(row['directory'])/'payload.bin').write_bytes(b'corrupted after interruption')
        row['outcome']='running'
        recover_journal([row]);assert row['outcome']=='infrastructure_error'
        row['outcome']='success'
        recover_journal([row]);assert row['outcome']=='infrastructure_error'
