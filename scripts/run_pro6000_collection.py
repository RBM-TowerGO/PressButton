#!/usr/bin/env python3
"""Local migration validation and evidence-gated 400-episode collection."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/vib820_gpu_collection_400_pro6000'
PREFLIGHT=BASE/'preflight_resident_v2'
PY=ROOT/'.conda/envs/pressb/bin/python'
LEROBOT=ROOT/'.conda/envs/lerobot/bin/python'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from pressb.astribot_collection.contract import atomic_json, candidate_block, sha256, validate_commit
from pressb.astribot_collection.dispatch import source_files
import numpy as np

ENV=os.environ.copy()
for name in ('PYTHONPATH','PYTHONHOME'): ENV.pop(name,None)
ENV.update(PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='0,1',OMNI_KIT_ACCEPT_EULA='YES',
           OMNICLIENT_HUB_MODE='disabled',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
           PATH=str(PY.parent)+':'+str(LEROBOT.parent)+':'+ENV.get('PATH',''),
           TMPDIR=str(ROOT/'.cache/sim_tmp'),HF_HOME=str(ROOT/'.cache/hf'))

def stage(name,**extra):
    atomic_json(BASE/'execution_state.json',dict(status='running',stage=name,pid=os.getpid(),preflight_directory=str(PREFLIGHT),updated_unix_s=time.time(),**extra))
    print(name,extra,flush=True)

def command(args,name,env=None):
    with (BASE/(name+'.log')).open('ab') as log:
        subprocess.run(list(map(str,args)),cwd=ROOT,env=env or ENV,stdout=log,stderr=subprocess.STDOUT,check=True)

def main():
    BASE.mkdir(parents=True,exist_ok=True)
    lock=(BASE/'execution.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (ROOT/'.cache/sim_tmp').mkdir(parents=True,exist_ok=True)
    PREFLIGHT.mkdir(exist_ok=True)
    stage('waiting_for_environment')
    # Package imports and asset checks must pass before starting any simulator.
    command([PY,'-c','import torch,isaacsim,av,psutil,pytest; assert torch.cuda.is_available(); assert torch.cuda.device_count()==2'],'environment_imports')
    command([LEROBOT,'-c','import importlib.metadata as m; from lerobot.datasets.lerobot_dataset import LeRobotDataset; assert m.version("lerobot")=="0.6.1"'],'lerobot_imports')
    while True:
        try:
            command([PY,'-c','import torch; [torch.zeros(1,device=f"cuda:{i}") for i in (0,1)]; [torch.cuda.synchronize(i) for i in (0,1)]'],'dual_gpu_health')
            break
        except subprocess.CalledProcessError:
            stage('waiting_for_dual_gpu_recovery',reason='minimal CUDA allocation failed; see dual_gpu_health.log')
            time.sleep(60)

    command([PY,'scripts/fetch_astribot_assets.py','--check-only'],'asset_validation')
    frozen=ROOT/'outputs/vib820_gpu_collection_400_manifest_frozen/manifest.json'
    manifest=json.loads(frozen.read_text())
    assert sha256(frozen)=='44bd9612f8501d793e6ce346642215048160ae6deede9643159cf080593b9d27'
    rng=np.random.Generator(np.random.PCG64(20261014))
    assert manifest['candidates']==candidate_block(rng)
    assert manifest['rng_state']==rng.bit_generator.state
    assert manifest['config']==json.loads((ROOT/'docs/vib820_experiment_spec/config_v1_2.json').read_text())
    assert len(manifest['joint_names'])==32 and len(manifest['cameras'])==7
    raw=BASE/'raw';raw.mkdir(exist_ok=True)
    destination=raw/'manifest.json'
    if not destination.exists(): shutil.copy2(frozen,destination)
    if not (raw/'attempts.json').exists(): assert sha256(destination)==sha256(frozen)
    sources=source_files(ROOT)
    atomic_json(BASE/'migration_validation.json',dict(status='passed',manifest_sha256=sha256(frozen),
        frozen_candidates=512,source_sha256=sources,data_root=str(BASE),old_outputs_preserved=True))
    stage('unit_and_fault_checks')
    tests=['tests/test_astribot_render_controls.py','tests/test_astribot_collection_contract.py','tests/test_astribot_collection_verification.py',
        'tests/test_astribot_contact_routing.py','tests/test_astribot_gpu_pixels.py',
        'tests/test_astribot_resident_processes.py','tests/test_astribot_resource_admission.py',
        'tests/test_astribot_shared_assets.py','tests/test_astribot_wave_barrier.py',
        'tests/test_astribot_camera_video.py','tests/test_astribot_press_audit.py']
    command([PY,'-m','pytest','-q',*tests,'--junitxml',PREFLIGHT/'unit_fault_checks.xml'],'unit_fault_checks',dict(ENV,PRESSB_TEST_GPU='1'))
    command([PY,'-m','pytest','-q','tests/test_astribot_nvenc.py','--junitxml',PREFLIGHT/'nvenc_fault_checks.xml'],'nvenc_fault_checks',dict(ENV,PRESSB_TEST_NVENC='1'))
    stage('dual_gpu_simulation_and_official_export_pilot')
    jobs=BASE/'pilot_jobs.json'
    atomic_json(jobs,[dict(id='wave0_reference024',config=str(ROOT/'outputs/vib820_yaw_20261014/validated_n50_w8/episodes/sample_024/config.json'))])
    # Render on both physical cards using the real seven-camera collector.
    for gpu in (0,1):
        run=BASE/f'pilot_gpu{gpu}_v2'
        if (run/'worker_result.json').exists():
            result=json.loads((run/'worker_result.json').read_text())
            assert result['status']=='finished'
            assert json.loads((run/'run_configuration.json').read_text())['source_sha256']==sources
        else:
            if run.exists(): raise RuntimeError(f'Interrupted pilot retained at {run}; use a fresh run directory')
            command([PY,'scripts/collect_astribot_dataset.py','--output',run,'--jobs',jobs,
                     '--gpu',gpu,'--envs-per-gpu','1','--backend','cpu','--render-layout','homogeneous',
                     '--reset-probe','20','--audit-commit'],f'pilot_gpu{gpu}_v2')
            result=json.loads((run/'worker_result.json').read_text())
        assert len(result['results'])==1 and result['results'][0]['status']=='accepted',result
        validate_commit(result['results'][0]['output'])
    selection=BASE/'pilot_selection.json'
    atomic_json(selection,[dict(candidate_id=24,directory=result['results'][0]['output'])])
    pilot=BASE/'official_export_pilot_v2'
    command([LEROBOT,'scripts/export_astribot_lerobot.py','--selection',selection,'--output',pilot,
             '--gpu','1','--preflight'],'official_export_pilot_v2')
    assert json.loads((pilot/'meta/astribot_reader_audit.json').read_text())['status']=='passed'
    stage('resident_preflight_then_gated_production')
    command([PY,'scripts/run_astribot_resident_pipeline.py','--output',PREFLIGHT,'--raw',raw,
             '--official-export',pilot,'--render-layout','homogeneous','--execute-production'],'resident_pipeline')
    release=BASE/'release/release_report.json'
    report=json.loads(release.read_text())
    assert report['status']=='complete'
    atomic_json(BASE/'execution_state.json',dict(status='complete',release=str(release),completed_unix_s=time.time()))

if __name__=='__main__':
    try: main()
    except BlockingIOError:
        print('Another collection supervisor holds the execution lock.', flush=True)
        sys.exit(2)
    except BaseException as exc:
        BASE.mkdir(parents=True,exist_ok=True)
        atomic_json(BASE/'execution_state.json',dict(status='failed',error=f'{type(exc).__name__}: {exc}',
                    traceback=traceback.format_exc(),updated_unix_s=time.time()))
        raise
