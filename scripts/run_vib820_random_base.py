#!/usr/bin/env python3
"""Predeclare 30 independent ±20 cm stations and run isolated first attempts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import numpy as np
from summarize_vib820_random_base import classify, summarize
ROOT=Path(__file__).resolve().parents[1]


def atomic(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def samples(count,seed):
    rng=np.random.Generator(np.random.PCG64(seed))
    values=rng.uniform(-.2,.2,size=(count,2))
    return [dict(sample_id=f'sample_{i+1:03d}',dx_m=float(dx),dy_m=float(dy),x_m=float(-.12+dx),y_m=float(.28+dy),yaw_rad=0.) for i,(dx,dy) in enumerate(values)],rng.bit_generator.state


def create(directory,count,seed,timeout,gpu):
    directory.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((ROOT/'configs/astribot_vib820.json').read_text())
    rows,state=samples(count,seed)
    sources=directory/'sources';sources.mkdir()
    files=['scripts/run_vib820_random_base.py','scripts/summarize_vib820_random_base.py','scripts/run_astribot_press.py','scripts/audit_astribot_press.py','scripts/activate_local.sh','src/pressb/astribot_validation.py','src/pressb/astribot_kinematics.py','src/pressb/astribot_press.py','src/pressb/astribot_asset.py','src/pressb/vib820_scene.py','src/pressb/vib820_materials.py','src/pressb/scene.py','configs/astribot_vib820.json','assets/materials/vib820/manifest.json','tests/test_vib820_random_base.py','assets/generated/astribot/astribot_s1.usdc','vendor/astribot_descriptions/urdf/astribot_s1_urdf/astribot_whole_body_maniskill.urdf']
    files += [str(p.relative_to(ROOT)) for p in (ROOT/'docs/plan/vib820_random_base_validation').glob('*.md')]
    hashes={}
    for rel in files:
        p=sources/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,p);hashes[rel]=sha(p)
    (sources/'working_tree.patch').write_text(subprocess.check_output(['git','diff'],cwd=ROOT,text=True))
    status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)
    manifest=dict(batch_id=directory.name,created_utc=datetime.now(timezone.utc).isoformat(),N_planned=count,seed=seed,
        rng='numpy.Generator(PCG64)',numpy_version=np.__version__,rng_state_after_manifest=state,
        nominal_xy_m=[-.12,.28],delta_bounds_m=[-.2,.2],fixed_z='astribot_asset base_lift_m (asset-derived); preserved',yaw_rad=0.,
        config=cfg,timeout_s=timeout,gpu=gpu,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty_status=status,
        source_sha256=hashes,samples=rows)
    atomic(directory/'manifest.json',manifest)
    return manifest


def run_one(directory,manifest,sample,retry=False):
    home=directory/'episodes'/sample['sample_id'];home.mkdir(parents=True,exist_ok=True)
    cfg=dict(manifest['config'],robot_base_x=sample['x_m'],robot_base_y=sample['y_m'])
    atomic(home/'config.json',cfg)
    attempts=home/'attempts';attempts.mkdir(exist_ok=True)
    existing=sorted(p for p in attempts.iterdir() if p.is_dir())
    if existing and (existing[0]/'result.json').exists() and not retry:return
    # An interrupted first attempt is retained as runtime_error, never silently rerun.
    if existing and not (existing[0]/'result.json').exists():
        result=classify(existing[0]/'run')
        if result['primary_outcome'] == 'audit_failed':
            result['primary_outcome'] = 'runtime_error'
        result.update(process_completion_unknown=True,
            manager_error='batch manager interrupted before terminal result; no process exit status recovered',
            sample_id=sample['sample_id'],attempt_id=existing[0].name)
        atomic(existing[0]/'result.json',result)
        if not retry:return
    attempt=attempts/f'{len(existing)+1:04d}';attempt.mkdir()
    atomic(attempt/'started.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),sample_id=sample['sample_id'],attempt_id=attempt.name))
    command=[sys.executable,str(ROOT/'scripts/run_astribot_press.py'),'--headless','--gpu',str(manifest['gpu']),'--config',str(home/'config.json'),'--buttons','0','--repeats','1','--output',str(attempt/'run')]
    started=time.monotonic();timeout=False;exit_code=None;launch_error=None
    with (attempt/'stdout.log').open('w') as out,(attempt/'stderr.log').open('w') as err:
        try:
            process=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,start_new_session=True)
            try: exit_code=process.wait(timeout=manifest['timeout_s'])
            except subprocess.TimeoutExpired:
                timeout=True;os.killpg(process.pid,signal.SIGTERM)
                try:exit_code=process.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);exit_code=process.wait()
        except Exception as exc: launch_error=f'{type(exc).__name__}: {exc}'
    try:result=classify(attempt/'run',exit_code,timeout)
    except Exception as exc:
        result=dict(primary_outcome='runtime_error',ik_status='unknown',initial_overlap='unknown',dynamic_collision='unknown',motion_started=None,press_confirmed=None,full_success=False,error=f'classifier: {type(exc).__name__}: {exc}',failure_flags=['runtime_error'])
    result.update(batch_id=manifest['batch_id'],sample_id=sample['sample_id'],attempt_id=attempt.name,seed=manifest['seed'],
        command=command,wall_duration_s=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat(),
        config_sha256=sha(home/'config.json'),manifest_sha256=sha(directory/'manifest.json'),launch_error=launch_error)
    atomic(attempt/'result.json',result)
    print(sample['sample_id'],result['primary_outcome'],f"{result['wall_duration_s']:.1f}s",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=30);p.add_argument('--seed',type=int,default=20261013);p.add_argument('--gpu',type=int,default=1);p.add_argument('--timeout',type=float,default=600);p.add_argument('--resume',action='store_true');p.add_argument('--retry',nargs='+');p.add_argument('--manifest-only',action='store_true');a=p.parse_args()
    directory=a.output.resolve()
    if a.resume:
        manifest=json.loads((directory/'manifest.json').read_text())
        for rel,expected in manifest['source_sha256'].items():
            if rel.startswith(('scripts/','src/','configs/')) and sha(ROOT/rel)!=expected:raise RuntimeError(f'frozen source changed: {rel}')
    else:
        if not 20<=a.count<=50:p.error('count must be 20–50')
        manifest=create(directory,a.count,a.seed,a.timeout,a.gpu)
    if a.manifest_only:return
    for i,sample in enumerate(manifest['samples'],1):
        if a.retry and sample['sample_id'] not in a.retry:continue
        run_one(directory,manifest,sample,retry=bool(a.retry))
        atomic(directory/'state.json',dict(last_sample=sample['sample_id'],updated_utc=datetime.now(timezone.utc).isoformat()))
        if i==20 and not a.retry:summarize(directory,'interim_20')
    print(json.dumps(summarize(directory),indent=2),flush=True)

if __name__=='__main__':main()
