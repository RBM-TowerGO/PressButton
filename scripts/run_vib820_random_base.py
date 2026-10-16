#!/usr/bin/env python3
"""Predeclare reproducible stations and schedule isolated simulation attempts."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import tempfile
import numpy as np
from summarize_vib820_random_base import classify, summarize
ROOT=Path(__file__).resolve().parents[1]


def utc(): return datetime.now(timezone.utc).isoformat()


def atomic(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def batch_lock(directory):
    directory=Path(directory);directory.parent.mkdir(parents=True,exist_ok=True)
    with (directory.parent/(directory.name+'.lock')).open('a+') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError(f'batch already owned: {directory}')
        try: yield
        finally: fcntl.flock(lock,fcntl.LOCK_UN)


def samples(count,seed,nominal=(-.12,.28),dx=(-.2,.2),dy=(-.2,.2),yaw_deg=None):
    rng=np.random.Generator(np.random.PCG64(seed))
    values=rng.uniform([dx[0],dy[0]],[dx[1],dy[1]],size=(count,2))
    angles = rng.uniform(*yaw_deg, size=count) if yaw_deg is not None else np.zeros(count)
    return [dict(sample_id=f'sample_{i+1:03d}',dx_m=float(x),dy_m=float(y),
        x_m=float(nominal[0]+x),y_m=float(nominal[1]+y),
        yaw_rad=float(np.radians(angles[i])),yaw_deg=float(angles[i]))
        for i,(x,y) in enumerate(values)],rng.bit_generator.state


def create(directory,count,seed,timeout,gpu,*,config=None,nominal=(-.12,.28),
           dx=(-.2,.2),dy=(-.2,.2),gpus=None,workers=1,pilot=False,cpu_threads=32,
           yaw_deg=None,all_cameras=False,launch_interval=0.):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    config=Path(config or ROOT/'configs/astribot_vib820.json').resolve()
    cfg=json.loads(config.read_text())
    rows,state=samples(count,seed,nominal,dx,dy,yaw_deg)
    sources=directory/'sources';sources.mkdir()
    files=['scripts/run_vib820_random_base.py','scripts/summarize_vib820_random_base.py',
        'scripts/plot_vib820_random_base.py','scripts/audit_vib820_random_batch.py','scripts/run_astribot_press.py',
        'scripts/audit_astribot_press.py','scripts/activate_local.sh',
        'src/pressb/astribot_validation.py','src/pressb/astribot_kinematics.py',
        'src/pressb/base_pose.py','src/pressb/astribot_sensors.py','configs/astribot_sensors.json',
        'src/pressb/astribot_press.py','src/pressb/astribot_asset.py',
        'src/pressb/vib820_scene.py','src/pressb/vib820_materials.py','src/pressb/scene.py',
        'configs/astribot_vib820.json','assets/materials/vib820/manifest.json',
        'tests/test_vib820_random_base.py','tests/test_vib820_parallel.py',
        'assets/generated/astribot/astribot_s1.usdc',
        'vendor/astribot_descriptions/urdf/astribot_s1_urdf/astribot_whole_body_maniskill.urdf']
    files += [str(p.relative_to(ROOT)) for p in (ROOT/'docs/plan/vib820_ready_parallel_validation').glob('*.md')]
    if config.is_relative_to(ROOT): files.append(str(config.relative_to(ROOT)))
    hashes={}
    for rel in dict.fromkeys(files):
        if not (ROOT/rel).is_file(): continue
        p=sources/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,p);hashes[rel]=sha(p)
    shutil.copy2(config,sources/'selected_config.json')
    (sources/'working_tree.patch').write_text(subprocess.check_output(['git','diff'],cwd=ROOT,text=True))
    manifest=dict(schema_version=3,batch_id=directory.name,created_utc=utc(),N_planned=count,seed=seed,
        rng='numpy.Generator(PCG64)',numpy_version=np.__version__,rng_state_after_manifest=state,
        nominal_xy_m=list(nominal),delta_bounds_xy_m=[list(dx),list(dy)],
        fixed_z='astribot_asset base_lift_m (asset-derived); preserved',yaw_rad=0.,
        config=cfg,config_path=str(config),config_sha256=sha(config),timeout_s=timeout,
        gpu=gpu,gpus=list(gpus or [gpu]),workers=workers,pilot=pilot,cpu_threads=cpu_threads,
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        dirty_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        source_sha256=hashes,samples=rows)
    manifest['all_cameras'] = bool(all_cameras)
    manifest['launch_interval_s'] = float(launch_interval)
    if yaw_deg is not None:
        manifest.pop('yaw_rad')
        manifest['yaw_bounds_deg'] = list(yaw_deg)
        manifest['yaw_convention'] = 'counterclockwise about world +Z; root yaw relative to +X; fixed during each episode'
        manifest['sampling_order'] = 'PCG64: draw all Nx2 XY offsets, then N independent uniform yaw angles in degrees'
    if tuple(dx)==tuple(dy): manifest['delta_bounds_m']=list(dx)
    atomic(directory/'manifest.json',manifest)
    return manifest


def terminate(process):
    if process.poll() is not None:return
    try: os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return


def recover(directory,manifest):
    """Only called with exclusive batch lock; retain incomplete first attempts."""
    for sample in manifest['samples']:
        attempts=directory/'episodes'/sample['sample_id']/'attempts'
        for attempt in sorted(attempts.glob('*')):
            if not attempt.is_dir() or (attempt/'result.json').exists():continue
            started=json.loads((attempt/'started.json').read_text()) if (attempt/'started.json').exists() else {}
            pid=started.get('pid')
            if pid and Path(f'/proc/{pid}/cmdline').exists():
                command=Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace').split('\0')
                # A stale PID alone is not evidence of ownership.
                if str(attempt/'run') in command:
                    os.killpg(pid,signal.SIGTERM)
                    limit=time.monotonic()+15
                    while Path(f'/proc/{pid}/cmdline').exists() and time.monotonic()<limit:
                        if not Path(f'/proc/{pid}/cmdline').read_bytes():break
                        time.sleep(.1)
                    if Path(f'/proc/{pid}/cmdline').exists() and Path(f'/proc/{pid}/cmdline').read_bytes():
                        os.killpg(pid,signal.SIGKILL)
            result=safe_classify(attempt/'run',None,False)
            if result['primary_outcome'] not in ('unexpected_contact','initial_collision','target_ik_failed','path_ik_failed'):
                result['primary_outcome']='runtime_error';result['full_success']=False
            result.update(process_completion_unknown=True,manager_error='interrupted before terminal result',
                sample_id=sample['sample_id'],attempt_id=attempt.name,batch_id=manifest['batch_id'])
            atomic(attempt/'result.json',result)


def safe_classify(run,code,timeout):
    try:return classify(run,code,timeout)
    except Exception as exc:
        return dict(primary_outcome='runtime_error',ik_status='unknown',initial_overlap='unknown',
            dynamic_collision='unknown',startup_status='unknown',motion_started=None,
            press_confirmed=None,full_success=False,error=f'classifier: {type(exc).__name__}: {exc}',
            failure_flags=['runtime_error'])


def begin(directory,manifest,sample,gpu,command_factory=None):
    home=directory/'episodes'/sample['sample_id'];home.mkdir(parents=True,exist_ok=True)
    cfg=dict(manifest['config'],robot_base_x=sample['x_m'],robot_base_y=sample['y_m'])
    if 'yaw_bounds_deg' in manifest:
        cfg['robot_base_yaw_rad'] = sample['yaw_rad']
    if not (home/'config.json').exists():atomic(home/'config.json',cfg)
    elif json.loads((home/'config.json').read_text())!=cfg:raise RuntimeError('sample configuration changed')
    attempts=home/'attempts';attempts.mkdir(exist_ok=True)
    nums=[int(p.name) for p in attempts.iterdir() if p.is_dir() and p.name.isdigit()]
    attempt=attempts/f'{max(nums,default=0)+1:04d}';attempt.mkdir()
    # Keep native IPC paths short and isolated, with a retained per-attempt link.
    temp_root=ROOT/'.cache/sim_tmp';temp_root.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='s-',dir=temp_root))
    (attempt/'tmp').symlink_to(temp,target_is_directory=True)
    command=[sys.executable,str(ROOT/'scripts/run_astribot_press.py'),'--headless','--gpu',str(gpu),
        '--cpu-threads',str(manifest.get('cpu_threads',32)),'--config',str(home/'config.json'),'--buttons','0','--repeats','1','--output',str(attempt/'run')]
    if manifest.get('all_cameras'):
        command.append('--all-cameras')
    if command_factory:command=command_factory(sample,attempt,gpu)
    started=dict(started_utc=utc(),sample_id=sample['sample_id'],attempt_id=attempt.name,gpu=gpu,command=command,runtime_tmpdir=str(temp))
    atomic(attempt/'started.json',started)
    out=(attempt/'stdout.log').open('w');err=(attempt/'stderr.log').open('w')
    job=dict(sample=sample,attempt=attempt,home=home,command=command,gpu=gpu,
        start=time.monotonic(),out=out,err=err,timed_out=False,interrupted=False,process=None,launch_error=None)
    try:
        env=dict(os.environ,TMPDIR=str(temp),OMNICLIENT_HUB_MODE='disabled')
        process=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,start_new_session=True,env=env)
        job['process']=process;started['pid']=process.pid;atomic(attempt/'started.json',started)
    except Exception as exc:job['launch_error']=f'{type(exc).__name__}: {exc}'
    return job


def finish(job,directory,manifest):
    process=job['process'];code=process.returncode if process else None
    job['out'].close();job['err'].close()
    result=safe_classify(job['attempt']/'run',code,job['timed_out'])
    if manifest.get('all_cameras') and result['full_success'] and result.get('camera_audit',{}).get('status') != 'passed':
        result.update(primary_outcome='camera_failed', full_success=False)
        result.setdefault('failure_flags', []).append('camera_failed')
    if job['interrupted'] and result['primary_outcome'] not in ('unexpected_contact','initial_collision','target_ik_failed','path_ik_failed'):
        result.update(primary_outcome='runtime_error',full_success=False)
    result.update(batch_id=manifest['batch_id'],sample_id=job['sample']['sample_id'],attempt_id=job['attempt'].name,
        seed=manifest['seed'],gpu=job['gpu'],command=job['command'],exit_code=code,timed_out=job['timed_out'],
        interrupted=job['interrupted'],wall_duration_s=time.monotonic()-job['start'],finished_utc=utc(),
        config_sha256=sha(job['home']/'config.json'),manifest_sha256=sha(directory/'manifest.json'),
        launch_error=job['launch_error'])
    atomic(job['attempt']/'result.json',result)
    print(result['sample_id'],result['primary_outcome'],f"gpu={job['gpu']} {result['wall_duration_s']:.1f}s",flush=True)
    return result


def resource_sample():
    record=dict(time_utc=utc())
    try:
        output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True,timeout=5)
        record['gpus']=[dict(zip(('index','total_mib','used_mib','utilization_pct'),map(int,row.split(','))))
            for row in output.strip().splitlines()]
    except Exception as exc:record['gpu_error']=str(exc)
    mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    record['available_memory_gib']=mem['MemAvailable']/1024**2
    record['available_disk_gib']=shutil.disk_usage(ROOT).free/1024**3
    return record


def schedule(directory,manifest,*,workers=None,gpus=None,retry=None,monitor=True,command_factory=None,
             summarizer=summarize,stop_event=None):
    directory=Path(directory);workers=workers or manifest.get('workers',1);gpus=gpus or manifest.get('gpus',[manifest['gpu']])
    recover(directory,manifest)
    todo=[s for s in manifest['samples'] if (s['sample_id'] in retry if retry else
        not (directory/'episodes'/s['sample_id']/'attempts/0001/result.json').exists())]
    slots=[gpus[i%len(gpus)] for i in range(workers)]
    jobs={};stopping=False;stop_reason=None;old_handlers={};resources=[];last_monitor=0.;last_launch=-float('inf')
    started=time.monotonic();terminal=sum((directory/'episodes'/s['sample_id']/'attempts/0001/result.json').exists() for s in manifest['samples'])
    def on_signal(signum,frame):
        nonlocal stopping,stop_reason
        stopping=True;stop_reason=f'signal {signum}'
    for sig in (signal.SIGTERM,signal.SIGINT):
        old_handlers[sig]=signal.signal(sig,on_signal)
    session=dict(started_utc=utc(),workers=workers,gpus=gpus,initial_pending=len(todo),launch_interval_s=manifest.get('launch_interval_s',0.))
    try:
        while todo or jobs:
            if stop_event and stop_event():stopping=True;stop_reason=stop_reason or 'requested interruption'
            if monitor and time.monotonic()-last_monitor>=5:
                r=resource_sample();resources.append(r);last_monitor=time.monotonic()
                if r['available_disk_gib']<.5 or r['available_memory_gib']<32 or any(g['index'] in gpus and g['used_mib']>=.8*g['total_mib'] for g in r.get('gpus',[])):
                    stopping=True;stop_reason=stop_reason or 'resource headroom threshold'
            for slot,job in list(jobs.items()):
                process=job['process'];now=time.monotonic()
                if process and process.poll() is None:
                    if stopping or now-job['start']>=manifest['timeout_s']:
                        if 'terminate_at' not in job:
                            job['timed_out']=not stopping;job['interrupted']=stopping
                            terminate(process);job['terminate_at']=now
                        elif now-job['terminate_at']>=15:
                            try:os.killpg(process.pid,signal.SIGKILL)
                            except ProcessLookupError:pass
                    continue
                result=finish(job,directory,manifest);del jobs[slot]
                if not retry:terminal+=1
                atomic(directory/'state.json',dict(N_terminal=terminal,active=[j['sample']['sample_id'] for j in jobs.values()],
                    pending=[s['sample_id'] for s in todo],updated_utc=utc(),workers=workers,gpus=gpus))
                if terminal==20 and not retry and not (directory/'interim_20.json').exists():
                    summarizer(directory,'interim_20')
                    checkpoint=directory/'checkpoints/20';checkpoint.mkdir(parents=True,exist_ok=True)
                    for suffix in ('json','csv'):shutil.copy2(directory/f'interim_20.{suffix}',checkpoint/f'interim_20.{suffix}')
                    atomic(checkpoint/'checkpoint.json',dict(created_utc=utc(),N_terminal=20,
                        manifest_sha256=sha(directory/'manifest.json'),
                        sha256={p.name:sha(p) for p in checkpoint.glob('interim_20.*')}))
                if result['primary_outcome']=='runtime_error':stopping=True;stop_reason=stop_reason or 'infrastructure failure'
            if stopping and not jobs:break
            if not stopping:
                for slot,gpu in enumerate(slots):
                    if slot not in jobs and todo and time.monotonic()-last_launch>=manifest.get('launch_interval_s',0.):
                        jobs[slot]=begin(directory,manifest,todo.pop(0),gpu,command_factory)
                        last_launch=time.monotonic()
            time.sleep(.1)
    finally:
        # Unexpected manager exceptions also reap every owned child before releasing lock.
        for job in jobs.values():
            job['interrupted']=True
            if job['process']:terminate(job['process'])
        for job in jobs.values():
            if job['process']:
                try:job['process'].wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(job['process'].pid,signal.SIGKILL);job['process'].wait()
            finish(job,directory,manifest)
        for sig,handler in old_handlers.items():signal.signal(sig,handler)
        session.update(finished_utc=utc(),wall_duration_s=time.monotonic()-started,stop_reason=stop_reason,
            resource_samples=resources)
        history=json.loads((directory/'execution.json').read_text()) if (directory/'execution.json').exists() else []
        history.append(session);atomic(directory/'execution.json',history)
    result=summarizer(directory)
    atomic(directory/'state.json',dict(N_terminal=result['N_terminal'],pending=result['N_planned']-result['N_terminal'],
        active=[],updated_utc=utc(),stop_reason=stop_reason))
    return result


def run_one(directory,manifest,sample,retry=False):
    """Compatibility entry point for a single isolated sample."""
    directory=Path(directory)
    with batch_lock(directory):
        recover(directory,dict(manifest,samples=[sample]))
        first=directory/'episodes'/sample['sample_id']/'attempts/0001/result.json'
        if first.exists() and not retry:return
        job=begin(directory,manifest,sample,manifest['gpu'])
        try:
            if job['process']:
                try:job['process'].wait(timeout=manifest['timeout_s'])
                except subprocess.TimeoutExpired:
                    job['timed_out']=True;terminate(job['process'])
                    try:job['process'].wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(job['process'].pid,signal.SIGKILL);job['process'].wait()
        finally:
            if job['process'] and job['process'].poll() is None:
                job['interrupted']=True;terminate(job['process']);job['process'].wait()
            finish(job,directory,manifest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=30)
    p.add_argument('--seed',type=int,default=20261013);p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--gpus',type=int,nargs='+');p.add_argument('--workers',type=int)
    p.add_argument('--cpu-threads',type=int,default=32)
    p.add_argument('--launch-interval',type=float,default=0.,help='Minimum seconds between launches to spread initialization load')
    p.add_argument('--config',type=Path,default=ROOT/'configs/astribot_vib820.json')
    p.add_argument('--nominal',type=float,nargs=2,default=[-.12,.28])
    p.add_argument('--yaw-deg',type=float,nargs=2,help='Independent uniform base yaw range in degrees, positive about world +Z')
    p.add_argument('--all-cameras',action='store_true',help='Keep all seven camera streams and phase snapshots')
    p.add_argument('--dx',type=float,nargs=2,default=[-.2,.2]);p.add_argument('--dy',type=float,nargs=2,default=[-.2,.2])
    p.add_argument('--timeout',type=float,default=600);p.add_argument('--resume',action='store_true')
    p.add_argument('--retry',nargs='+');p.add_argument('--manifest-only',action='store_true');p.add_argument('--pilot',action='store_true')
    a=p.parse_args();directory=a.output.resolve()
    if a.workers is not None and not 1<=a.workers<=16:p.error('workers must be 1–16; choose according to available resources')
    if not 1<=a.cpu_threads<=(os.cpu_count() or 1):p.error('invalid CPU thread limit')
    if a.timeout<=0:p.error('positive timeout required')
    if not np.isfinite(a.launch_interval) or a.launch_interval<0:p.error('nonnegative finite launch interval required')
    if not np.isfinite(a.nominal+a.dx+a.dy).all() or a.dx[0]>a.dx[1] or a.dy[0]>a.dy[1]:p.error('invalid bounds')
    if a.yaw_deg is not None and (not np.isfinite(a.yaw_deg).all() or a.yaw_deg[0]>a.yaw_deg[1]):p.error('invalid yaw bounds')
    if a.retry and not a.resume:p.error('--retry requires --resume')
    with batch_lock(directory):
        if a.resume:
            manifest=json.loads((directory/'manifest.json').read_text())
            for rel,expected in manifest['source_sha256'].items():
                if rel.startswith(('scripts/','src/','configs/')) and sha(ROOT/rel)!=expected:raise RuntimeError(f'frozen source changed: {rel}')
            if a.retry and not set(a.retry)<=set(s['sample_id'] for s in manifest['samples']):p.error('unknown retry sample')
        else:
            if not (1<=a.count<=50 if a.pilot else 20<=a.count<=50):p.error('count must be 20–50 (pilot 1–50)')
            manifest=create(directory,a.count,a.seed,a.timeout,a.gpu,config=a.config,nominal=a.nominal,
                dx=a.dx,dy=a.dy,gpus=a.gpus,workers=a.workers or 1,pilot=a.pilot,cpu_threads=a.cpu_threads,
                yaw_deg=a.yaw_deg,all_cameras=a.all_cameras,launch_interval=a.launch_interval)
        if a.manifest_only:return
        result=schedule(directory,manifest,workers=a.workers,gpus=a.gpus,retry=a.retry)
        print(json.dumps(result,indent=2),flush=True)
        if result['N_terminal']<result['N_planned']:raise SystemExit(2)

if __name__=='__main__':main()
