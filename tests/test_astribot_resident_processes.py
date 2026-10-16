import json
import pytest
from pressb.astribot_collection.verification import resident_throughput, choose_resident_concurrency
from pressb.astribot_collection import dispatch
from pressb.astribot_collection.contract import sha256


def workers(count=2):
    return [dict(directory=f'/gpu{g}/slot{s}',gpu=g,envs_per_gpu=1,accepted=2,
                 measurement_start_unix_s=100+s*5,measurement_end_unix_s=200+g*20,
                 max_vram_fraction=.4,min_available_ram_gib=100)
            for g in (0,1) for s in range(count)]


def test_resident_throughput_counts_same_gpu_distinct_workers_over_real_wall_time():
    result=resident_throughput(workers(),2,[0,1])
    assert result['accepted']==8
    assert result['wall_s']==120
    assert result['accepted_per_minute']==4
    with pytest.raises(ValueError,match='Incomplete'):resident_throughput(workers()[:3],2,[0,1])
    duplicate=workers();duplicate[1]['directory']=duplicate[0]['directory']
    with pytest.raises(ValueError,match='Duplicate'):resident_throughput(duplicate,2,[0,1])
    sequential=workers();sequential[-1]['measurement_start_unix_s']=300
    with pytest.raises(ValueError,match='concurrently'):resident_throughput(sequential,2,[0,1])
    scenes=workers();scenes[0]['envs_per_gpu']=2
    with pytest.raises(ValueError,match='one environment'):resident_throughput(scenes,2,[0,1])


def test_resident_selection_respects_resource_rejection_and_five_percent_rule():
    groups=[dict(processes_per_gpu=c,accepted_per_minute=v,max_vram_fraction=.4,
                 min_available_ram_gib=100,status='passed') for c,v in ((1,1),(2,2),(4,2.08))]
    assert choose_resident_concurrency(groups)['processes_per_gpu']==2
    groups[-1].update(accepted_per_minute=4,max_vram_fraction=.81)
    assert choose_resident_concurrency(groups)['processes_per_gpu']==2
    groups[-1]=dict(processes_per_gpu=4,status='resource_rejected')
    assert choose_resident_concurrency(groups)['processes_per_gpu']==2
    with pytest.raises(ValueError,match='Incomplete'):choose_resident_concurrency(groups[:2])


def test_resident_gate_freezes_process_count_and_rejects_tampered_evidence(tmp_path,monkeypatch):
    monkeypatch.setattr(dispatch,'source_files',lambda root:{'capture.py':'frozen'})
    evidence=tmp_path/'evidence.json';evidence.write_text('{"status":"passed"}')
    gate=tmp_path/'gate.json'
    payload=dict(status='passed',source_sha256={'capture.py':'frozen'},
                 checks={name:dict(status='passed',path=str(evidence),sha256=sha256(evidence))
                         for name in dispatch.REQUIRED_GATES},
                 configuration=dict(gpus=[0,1],envs_per_gpu=1,processes_per_gpu=2,
                                    backend='cpu',render_subframes=1,render_layout='homogeneous'),
                 throughput_ratio=2.1,max_vram_fraction=.5,min_available_ram_gib=100)
    gate.write_text(json.dumps(payload))
    dispatch.validate_gate(gate,tmp_path,[0,1],1,'cpu','homogeneous',2)
    with pytest.raises(ValueError,match='differs'):
        dispatch.validate_gate(gate,tmp_path,[0,1],1,'cpu','homogeneous',4)
    with pytest.raises(ValueError,match='one environment'):
        dispatch.validate_gate(gate,tmp_path,[0,1],2,'cpu','homogeneous',2)
    evidence.write_text('{"status":"failed"}')
    with pytest.raises(ValueError,match='Invalid gate'):
        dispatch.validate_gate(gate,tmp_path,[0,1],1,'cpu','homogeneous',2)


def test_resident_backend_excludes_gpu_that_misses_initial_collision(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from run_astribot_resident_pipeline import select_backend
    for backend,seconds in (('cpu',20),('gpu',10)):
        directory=tmp_path/backend;directory.mkdir()
        rows=[]
        for sample in (24,1,2,3,5,10,20,35,40,50):
            episode=directory/f'reference{sample:03d}';episode.mkdir()
            report=dict(status='passed')
            if sample==50:
                report=dict(status='failed',motion_started=backend=='gpu',
                            termination_reason='initial_collision' if backend=='cpu' else None)
            (episode/'report.json').write_text(json.dumps(report))
            rows.append(dict(id=f'reference{sample:03d}',output=str(episode),
                             physical_audit=dict(status='passed' if sample!=50 else 'failed')))
        (directory/'worker_result.json').write_text(json.dumps(dict(status='finished',results=rows,
                                                                   timings=dict(physics_s=seconds))))
    selected=select_backend(tmp_path/'cpu',tmp_path/'gpu')
    assert selected['selected']=='cpu'
    assert selected['gpu_historical_acceptance'] is False
    assert '50' in selected['gpu_rejection']


def test_resident_source_reuse_requires_identical_capture_dependencies(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from run_astribot_resident_pipeline import check_capture_sources
    (tmp_path/'run_configuration.json').write_text(json.dumps(dict(source_sha256={
        'src/pressb/astribot_collection/runtime.py':'capture',
        'scripts/run_astribot_resident_pipeline.py':'previous-orchestrator'})))
    current={'src/pressb/astribot_collection/runtime.py':'capture',
             'scripts/run_astribot_resident_pipeline.py':'updated-orchestrator'}
    check_capture_sources(tmp_path,current)
    current['src/pressb/astribot_collection/runtime.py']='changed-capture'
    with pytest.raises(ValueError,match='Stale capture'):check_capture_sources(tmp_path,current)
