import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_vib820_random_base as runner
from summarize_vib820_random_base import startup_status, summarize_records, wilson


def test_asymmetric_sampling_and_default_compatibility():
    rows,_=runner.samples(50,20261014,(-.12,.28),(-.3,.05),(-.3,.3))
    expected=np.random.default_rng(20261014).uniform([-.3,-.3],[.05,.3],(50,2))
    np.testing.assert_array_equal([[r['dx_m'],r['dy_m']] for r in rows],expected)
    assert rows==runner.samples(50,20261014,(-.12,.28),(-.3,.05),(-.3,.3))[0]
    assert all(-.42<=r['x_m']<=-.07 and -.02<=r['y_m']<=.58 for r in rows)
    old=runner.samples(30,20261013)[0]
    np.testing.assert_array_equal([[r['dx_m'],r['dy_m']] for r in old],np.random.default_rng(20261013).uniform(-.2,.2,(30,2)))


def test_manifest_ready_config(tmp_path):
    directory=tmp_path/'batch'
    m=runner.create(directory,50,20261014,600,0,config=ROOT/'configs/astribot_vib820_ready.json',
        nominal=(-.12,.28),dx=(-.3,.05),dy=(-.3,.3),gpus=[0,1],workers=4,cpu_threads=8)
    assert len(m['config']['initial_arm_positions'])==14 and m['config']['return_to_initial_pose']
    assert m['nominal_xy_m']==[-.12,.28] and m['config']['robot_base_x']==-.22
    assert m['cpu_threads']==8
    assert 'configs/astribot_vib820_ready.json' in m['source_sha256']
    assert (directory/'manifest.json').exists() and not (directory/'episodes').exists()


def test_yaw_sampling_is_reproducible_and_preserves_xy():
    kwargs=dict(nominal=(-.12,.28),dx=(-.18,-.02),dy=(-.28,.12))
    rows,state=runner.samples(50,20261014,**kwargs,yaw_deg=(0,10))
    baseline,_=runner.samples(50,20261014,**kwargs)
    np.testing.assert_array_equal([[s['x_m'],s['y_m']] for s in rows],[[s['x_m'],s['y_m']] for s in baseline])
    rng=np.random.default_rng(20261014)
    rng.uniform([-.18,-.28],[-.02,.12],(50,2))
    np.testing.assert_array_equal([s['yaw_deg'] for s in rows],rng.uniform(0,10,50))
    assert all(0<s['yaw_rad']<np.radians(10) for s in rows)
    assert rows==runner.samples(50,20261014,**kwargs,yaw_deg=(0,10))[0]


def test_yaw_and_all_cameras_forwarded(tmp_path,monkeypatch):
    d,m=setup_fake(tmp_path,n=1)
    m.update(yaw_bounds_deg=[0,10],all_cameras=True)
    m['samples'][0]['yaw_rad']=np.radians(7)
    def fail_launch(command,**kwargs):
        assert '--all-cameras' in command
        assert '--video' not in command
        raise OSError('expected test stop')
    monkeypatch.setattr(runner.subprocess,'Popen',fail_launch)
    job=runner.begin(d,m,m['samples'][0],1)
    assert json.loads((job['home']/'config.json').read_text())['robot_base_yaw_rad']==np.radians(7)
    runner.finish(job,d,m)


@pytest.mark.parametrize('report,expected',[
    ({'termination_reason':'initial_collision'},'fail'),
    ({'termination_reason':'target_ik_failed'},'fail'),
    ({'termination_phase':'close_gripper','unexpected_contacts':[{}]},'fail'),
    ({'termination_phase':'move_to_work_pose','unexpected_contacts':[{}]},'fail'),
    ({'error':'crash'},'unknown'),
    ({'startup':{'status':'pass','work_pose_error_rad':.001,'unexpected_contacts_before_completion':0,'completed_step':900},
      'termination_phase':'return_to_ready','unexpected_contacts':[{'step':3000}]},'pass'),
    ({'startup':{'status':'fail','work_pose_error_rad':.03,'unexpected_contacts_before_completion':0,'completed_step':900}},'fail'),
])
def test_startup_evidence(report,expected):assert startup_status(report)==expected


def test_startup_denominator_unknown_and_wilson():
    rows=[dict(startup_status=s,primary_outcome='runtime_error',full_success=False) for s in ['pass','fail','unknown']]
    s=summarize_records(rows)
    assert s['metrics']['startup']['numerator']==1 and s['metrics']['startup']['denominator']==3
    assert s['N_startup_unknown']==1
    assert wilson(0,0) is None
    assert wilson(50,50)[0]==pytest.approx(.9286524,abs=1e-6)


def fake_summary(directory,prefix='summary'):
    m=json.loads((directory/'manifest.json').read_text())
    done=sum((directory/'episodes'/s['sample_id']/'attempts/0001/result.json').exists() for s in m['samples'])
    summary={'N_terminal':done,'N_planned':len(m['samples'])}
    runner.atomic(directory/(prefix+'.json'),summary)
    (directory/(prefix+'.csv')).write_text('sample_id\n')
    return summary


def setup_fake(tmp_path,n=5,timeout=5):
    directory=tmp_path/'batch';directory.mkdir()
    m=dict(batch_id='test',samples=runner.samples(n,1)[0],config={'initial_arm_positions':{'arm':1}},
        timeout_s=timeout,gpu=0,gpus=[0,1],workers=2,seed=1)
    runner.atomic(directory/'manifest.json',m)
    return directory,m


def factory(delay):
    def command(sample,attempt,gpu):
        # Each standalone process records enough physical-report structure to finish
        # without runtime_error; no simulator or real GPU needed for scheduler tests.
        duration=delay(sample)
        code='''import json,time,os
from pathlib import Path
p=Path(%r);p.mkdir()
(p/'pid').write_text(str(os.getpid()))
time.sleep(%r)
(p/'report.json').write_text(json.dumps({'status':'failed','termination_reason':'target_ik_failed','motion_started':False}))
'''%(str(attempt/'run'),duration)
        return [sys.executable,'-c',code]
    return command


def test_parallel_out_of_order_unique_gpu_and_resume(tmp_path):
    d,m=setup_fake(tmp_path)
    with runner.batch_lock(d):
        result=runner.schedule(d,m,monitor=False,command_factory=factory(lambda s:.7 if s['sample_id']=='sample_001' else .12),summarizer=fake_summary)
    assert result['N_terminal']==5
    starts=[];ends=[];hashes={}
    for s in m['samples']:
        a=d/'episodes'/s['sample_id']/'attempts/0001'
        start=json.loads((a/'started.json').read_text());end=json.loads((a/'result.json').read_text())
        starts.append((start['started_utc'],1,start['gpu']));ends.append((end['finished_utc'],-1,start['gpu']))
        assert end['gpu'] in [0,1] and end['attempt_id']=='0001'
        assert Path(end['command'][0]).exists()
        hashes[s['sample_id']]=runner.sha(a/'result.json')
    active=0;peak=0;gpu_active={0:0,1:0}
    for _,delta,gpu in sorted(starts+ends):
        active+=delta;peak=max(peak,active);gpu_active[gpu]+=delta
        assert active<=2 and gpu_active[gpu]<=1
    assert peak==2 and sorted(ends)[0][0]<json.loads((d/'episodes/sample_001/attempts/0001/result.json').read_text())['finished_utc']
    with runner.batch_lock(d):runner.schedule(d,m,monitor=False,command_factory=factory(lambda s:0),summarizer=fake_summary)
    assert hashes=={s['sample_id']:runner.sha(d/'episodes'/s['sample_id']/'attempts/0001/result.json') for s in m['samples']}


def test_batch_lock(tmp_path):
    with runner.batch_lock(tmp_path/'batch'):
        with pytest.raises(RuntimeError):
            with runner.batch_lock(tmp_path/'batch'):pass


def test_timeout_stops_dispatch_and_reaps(tmp_path):
    d,m=setup_fake(tmp_path,n=4,timeout=.25)
    with runner.batch_lock(d):r=runner.schedule(d,m,monitor=False,command_factory=factory(lambda s:5),summarizer=fake_summary)
    assert r['N_terminal']==2
    for s in m['samples'][:2]:
        a=d/'episodes'/s['sample_id']/'attempts/0001'
        result=json.loads((a/'result.json').read_text())
        pid=json.loads((a/'started.json').read_text())['pid']
        assert result['timed_out'] and result['primary_outcome']=='runtime_error'
        assert not Path(f'/proc/{pid}').exists()
    assert not (d/'episodes/sample_003').exists()


def test_interrupt_resume_and_explicit_retry(tmp_path):
    d,m=setup_fake(tmp_path,n=4);start=time.monotonic()
    with runner.batch_lock(d):
        r=runner.schedule(d,m,monitor=False,command_factory=factory(lambda s:5),summarizer=fake_summary,stop_event=lambda:time.monotonic()-start>.25)
    assert r['N_terminal']==2
    first=d/'episodes/sample_001/attempts/0001/result.json';original=first.read_bytes()
    with runner.batch_lock(d):
        r=runner.schedule(d,m,monitor=False,command_factory=factory(lambda s:.05),summarizer=fake_summary)
    assert r['N_terminal']==4 and first.read_bytes()==original
    with runner.batch_lock(d):
        r=runner.schedule(d,m,retry=['sample_001'],monitor=False,command_factory=factory(lambda s:.05),summarizer=fake_summary)
    assert r['N_terminal']==4 and first.read_bytes()==original
    assert (d/'episodes/sample_001/attempts/0002/result.json').exists()


def test_recovery_retains_collision(tmp_path):
    d,m=setup_fake(tmp_path,n=1)
    a=d/'episodes/sample_001/attempts/0001';(a/'run').mkdir(parents=True)
    runner.atomic(a/'run/report.json',dict(unexpected_contacts=[dict(step=12,force_n=.2)],termination_phase='close_gripper'))
    with runner.batch_lock(d):
        r=runner.schedule(d,m,monitor=False,command_factory=factory(lambda s:0),summarizer=fake_summary)
    record=json.loads((a/'result.json').read_text())
    assert r['N_terminal']==1 and record['primary_outcome']=='unexpected_contact'
    assert record['process_completion_unknown'] and not (a.parent/'0002').exists()


@pytest.mark.parametrize('workers',[4,8])
def test_four_workers_gpu_slots_and_checkpoint(tmp_path,workers):
    d,m=setup_fake(tmp_path,n=21)
    with runner.batch_lock(d):
        r=runner.schedule(d,m,workers=workers,monitor=False,command_factory=factory(lambda s:.3),summarizer=fake_summary)
    assert r['N_terminal']==21
    events=[]
    for s in m['samples']:
        a=d/'episodes'/s['sample_id']/'attempts/0001'
        begin=json.loads((a/'started.json').read_text());end=json.loads((a/'result.json').read_text())
        events.extend([(begin['started_utc'],1,begin['gpu']),(end['finished_utc'],-1,end['gpu'])])
    active={0:0,1:0};peak=0
    for _,delta,gpu in sorted(events):
        active[gpu]+=delta;peak=max(peak,sum(active.values()))
        assert active[gpu]<=workers//2 and sum(active.values())<=workers
    assert peak==workers
    checkpoint=json.loads((d/'checkpoints/20/checkpoint.json').read_text())
    assert checkpoint['N_terminal']==20
    assert json.loads((d/'interim_20.json').read_text())['N_terminal']==20


def test_manager_error_reaps_children(tmp_path,monkeypatch):
    d,m=setup_fake(tmp_path,n=3)
    original=runner.begin
    calls=[]
    def broken(*args,**kwargs):
        if calls:raise RuntimeError('injected dispatch error')
        job=original(*args,**kwargs);calls.append(job);return job
    monkeypatch.setattr(runner,'begin',broken)
    with runner.batch_lock(d),pytest.raises(RuntimeError,match='injected'):
        runner.schedule(d,m,monitor=False,command_factory=factory(lambda s:5),summarizer=fake_summary)
    assert calls[0]['process'].poll() is not None
    assert json.loads((calls[0]['attempt']/'result.json').read_text())['interrupted']


def test_ready_return_failure_keeps_startup_success(tmp_path):
    from summarize_vib820_random_base import classify
    runner.atomic(tmp_path/'report.json',dict(status='failed',episodes=[dict(pressed=True,success=True)],
        ready_return=dict(passed=False),startup=dict(status='pass',work_pose_error_rad=.001,
        unexpected_contacts_before_completion=0,completed_step=900)))
    result=classify(tmp_path,exit_code=1)
    assert result['primary_outcome']=='release_or_return_failed'
    assert result['startup_status']=='pass' and not result['full_success']


def test_cpu_thread_limit_forwarded_without_physics_changes(tmp_path,monkeypatch):
    d,m=setup_fake(tmp_path,n=1);m['cpu_threads']=8
    def fail_launch(command,**kwargs):
        assert command[command.index('--cpu-threads')+1]=='8'
        assert len(kwargs['env']['TMPDIR'].encode())<80
        assert '/.cache/sim_tmp/' in kwargs['env']['TMPDIR']
        raise OSError('constructed launch error')
    monkeypatch.setattr(runner.subprocess,'Popen',fail_launch)
    job=runner.begin(d,m,m['samples'][0],1)
    assert job['launch_error']=='OSError: constructed launch error'
    assert (job['attempt']/'tmp').is_symlink()
    assert (job['attempt']/'tmp').resolve().is_dir()
    assert json.loads((job['home']/'config.json').read_text())==dict(m['config'],robot_base_x=m['samples'][0]['x_m'],robot_base_y=m['samples'][0]['y_m'])
    runner.finish(job,d,m)
