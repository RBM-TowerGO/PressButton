#!/usr/bin/env python3
"""Persistent multi-environment Astribot collector; preflight before production."""
from __future__ import annotations
import argparse
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
import os
import shutil
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from pressb.astribot_collection.contract import atomic_json, create_manifest, exclusive_lock, commit_episode

def worker(args):
    runtime=None;app=None;started=time.perf_counter();results=[];started_jobs=[];waves=[]
    jobs=json.loads(args.jobs.read_text())
    cfg=json.loads(args.config.read_text())
    from pressb.astribot_collection.dispatch import source_files
    atomic_json(args.output/'run_configuration.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        source_sha256=source_files(ROOT),started_unix_s=time.time(),pid=os.getpid()))
    def event(kind,**data):
        with (args.output/'events.jsonl').open('a') as log:
            log.write(json.dumps(dict(kind=kind,unix_s=time.time(),monotonic_s=time.perf_counter(),**data))+'\n')
            log.flush();os.fsync(log.fileno())
    from pressb.astribot_collection.monitor import ResourceMonitor
    monitor=ResourceMonitor(args.output/'resources.jsonl',lambda: {
        f'{e.index}/{name}':writer.pending.qsize() for e in runtime.all_envs for name,writer in getattr(e,'writers',{}).copy().items()
    } if runtime else {})
    monitor.__enter__()
    try:
        from isaacsim import SimulationApp
        app=SimulationApp(dict(headless=True,create_new_stage=False,active_gpu=args.gpu,physics_gpu=args.gpu,
            multi_gpu=False,limit_cpu_threads=args.cpu_threads,renderer='RayTracedLighting',anti_aliasing=3,width=960,height=600,
            disable_viewport_updates=True,
            extra_args=['--/app/asyncRendering=false','--/rtx/ecoMode/enabled=false','--/validate/p2p/enabled=false',
                        '--/validate/iommu/enabled=false','--/app/extensions/fsWatcherEnabled=false']))
        app_startup=time.perf_counter()-started
        from pressb.astribot_collection.runtime import Runtime
        runtime=Runtime(app,cfg,args.output/'shared',args.envs_per_gpu,args.gpu,args.backend,args.render_subframes,args.render_layout,args.asset_cache)
        runtime.timings['simulation_app_startup_s']=app_startup
        event('runtime_ready')
        if args.reset_probe:
            probe=deepcopy(cfg)
            if jobs and 'config' in jobs[0]:probe=json.loads(Path(jobs[0]['config']).read_text())
            probe.setdefault('robot_base_yaw_rad',0.)
            runtime.probe_resets(probe,args.reset_probe,args.output/'reset_probe')
            event('reset_probe_done',rounds=args.reset_probe)
        for wave in range(0,len(jobs),args.envs_per_gpu):
            if (args.output/'STOP').exists():break
            if shutil.disk_usage(args.output).free<20*1024**3:
                atomic_json(args.output/'resource_stop.json',dict(reason='disk_free_below_20_GiB'));break
            selected=jobs[wave:wave+args.envs_per_gpu]
            configs=[];outputs=[]
            for job in selected:
                c=deepcopy(cfg)
                if 'config' in job:c=json.loads(Path(job['config']).read_text())
                else:c.update(robot_base_x=job['x_m'],robot_base_y=job['y_m'],robot_base_yaw_rad=job['yaw_rad'])
                c.setdefault('robot_base_yaw_rad',0.)
                configs.append(c);outputs.append(args.output/'episodes'/(str(job['id'])+'.inprogress' if args.audit_commit else str(job['id'])))
            print('WAVE_START',wave,[j['id'] for j in selected],flush=True)
            wave_started=time.perf_counter();stage_before=runtime.timings.copy();result_start=len(results)
            event('wave_start',wave=wave//args.envs_per_gpu,jobs=[j['id'] for j in selected])
            started_jobs.extend(j['id'] for j in selected)
            atomic_json(args.output/'started_jobs.json',started_jobs)
            reports=runtime.run_wave(configs,outputs,physics_only=args.physics_only_preflight,
                                     replay_sources=[j.get('replay_source') for j in selected])
            audit_started=time.perf_counter()
            def verify_one(item):
                job,report,output=item
                result=dict(id=job['id'],status=report['status'],elapsed_s=report['elapsed_s'],output=str(output))
                atomic_json(output/'job.json',job)
                if report['status']=='passed':
                    from audit_astribot_press import audit
                    result['physical_audit']=audit(output)
                    if args.audit_commit:
                        try:
                            from pressb.astribot_collection.audit import audit_raw
                            verified=audit_raw(output)
                            final=output.with_name(str(job['id']))
                            commit_episode(output,final,verified)
                            result.update(status='accepted',output=str(final),collection_audit=verified)
                        except Exception as exc:
                            result.update(status='audit_failed',error=f'{type(exc).__name__}: {exc}')
                result.update(job=job)
                return result
            # Each audit owns its episode directory and FFmpeg process. Bound
            # concurrency rather than leaving the GPU idle for N serial decodes.
            with ThreadPoolExecutor(max_workers=min(4,len(selected)),thread_name_prefix='astribot-audit') as audit_pool:
                results.extend(audit_pool.map(verify_one,zip(selected,reports,outputs)))
            runtime.timings['audit_s']=runtime.timings.get('audit_s',0.)+time.perf_counter()-audit_started
            waves.append(dict(wave=wave//args.envs_per_gpu,elapsed_s=time.perf_counter()-wave_started,
                results=[dict(id=r['id'],status=r['status']) for r in results[result_start:]],
                stages={k:v-stage_before.get(k,0.) for k,v in runtime.timings.items()}))
            event('wave_done',**waves[-1])
            atomic_json(args.output/'worker_result.json',dict(status='running',elapsed_s=time.perf_counter()-started,results=results,waves=waves))
            print('WAVE_DONE',wave,[r['status'] for r in reports],flush=True)
        atomic_json(args.output/'worker_result.json',dict(status='finished',elapsed_s=time.perf_counter()-started,
                    results=results,waves=waves,timings=runtime.timings,backend=runtime.actual_backend))
    except BaseException as exc:
        atomic_json(args.output/'worker_error.json',dict(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),results=results))
        raise
    finally:
        if runtime:runtime.close()
        monitor.__exit__()
        if app:app.app.post_quit();app.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'docs/vib820_experiment_spec/config_v1_2.json')
    p.add_argument('--target-successes',type=int,default=400);p.add_argument('--seed',type=int,default=20261014)
    p.add_argument('--candidate-limit',type=int,choices=[512,1024],default=1024)
    p.add_argument('--gpus',type=int,nargs='+',default=[0,1]);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--envs-per-gpu',type=int,choices=[1,2,4,8],default=2)
    p.add_argument('--processes-per-gpu',type=int,choices=[1,2,4],default=1)
    p.add_argument('--cpu-threads',type=int,default=8);p.add_argument('--backend',choices=['cpu','gpu'],default='cpu')
    p.add_argument('--jobs',type=Path,help='Frozen worker job JSON (preflight or dispatcher)')
    p.add_argument('--audit-commit',action='store_true',help='Full seven-video/lamp audit followed by atomic commit')
    p.add_argument('--render-subframes',type=int,choices=[1,2],default=1)
    p.add_argument('--render-layout',choices=['individual','homogeneous','isolated','padded'],default='individual',
                   help='Capture layout; all modes require a source-matched gate. Isolated is experimental; padded cannot enter production.')
    p.add_argument('--reset-probe',type=int,default=0,help='Preflight consecutive dirty-state resets before jobs')
    p.add_argument('--physics-only-preflight',action='store_true',help='Correctness comparison only; ineligible for production or speed baseline')
    p.add_argument('--freeze-only',action='store_true',help='Freeze candidate manifest without launching physics')
    p.add_argument('--preflight-report',type=Path,help='Passed, source-matched production gate report')
    p.add_argument('--asset-cache',type=Path,help='Shared content-addressed scene/geometry cache on the same filesystem')
    p.add_argument('--resume',action='store_true',help='Resume the existing verified production journal')
    args=p.parse_args();args.output=args.output.resolve()
    if args.processes_per_gpu > 1 and args.envs_per_gpu != 1:
        p.error('Multiple resident processes require exactly one environment per process')
    if args.physics_only_preflight and (not args.jobs or args.audit_commit):
        p.error('--physics-only-preflight requires --jobs and cannot commit dataset episodes')
    with exclusive_lock(args.output):
        if args.jobs:worker(args)
        elif args.freeze_only:
            create_manifest(args.output,json.loads(args.config.read_text()),args.seed,args.target_successes,args.candidate_limit)
        elif args.preflight_report:
            if (args.output/'attempts.json').exists() and not args.resume:raise SystemExit('Existing attempts require --resume')
            from pressb.astribot_collection.dispatch import run_production
            run_production(args,ROOT)
        else:raise SystemExit('Production is gated on completed preflight. Use --freeze-only or --jobs for preflight.')
if __name__=='__main__':main()
