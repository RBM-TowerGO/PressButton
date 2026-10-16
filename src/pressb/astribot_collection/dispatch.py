"""Fail-closed production dispatcher and attempt journal for resident GPU workers."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from .contract import (atomic_json,create_manifest,extend_manifest,ordered_successes,
                       sha256,validate_commit)

REQUIRED_GATES=('historical_cases','twenty_resets','environment_isolation','fault_injection',
                'backend_comparison','concurrency_sweep','official_export')

def resource_violation(sample,gpus):
    if sample['available_ram_bytes']<32*1024**3:return 'available_ram_below_32_GiB'
    if sample['disk_free_bytes']<20*1024**3:return 'disk_free_below_20_GiB'
    for gpu in sample.get('gpus',[]):
        if gpu['index'] in gpus and gpu['memory_used_mib']>.8*gpu['memory_total_mib']:
            return f'gpu_{int(gpu["index"])}_vram_above_80_percent'
    return None

def latest_resource(path):
    if not path.exists():return None
    with path.open('rb') as file:
        file.seek(max(0,path.stat().st_size-65536))
        lines=file.read().splitlines()
    for line in reversed(lines):
        try:return json.loads(line)
        except (ValueError,UnicodeDecodeError):continue
    return None

def source_files(root):
    root=Path(root)
    names=['scripts/collect_astribot_dataset.py','scripts/export_astribot_lerobot.py','scripts/audit_astribot_press.py',
           'scripts/finalize_astribot_collection.py','scripts/run_astribot_collection_pipeline.py',
           'scripts/probe_astribot_images.py','scripts/run_astribot_resident_pipeline.py',
           'scripts/seal_astribot_preflight.py','scripts/summarize_astribot_benchmarks.py',
           'src/pressb/astribot_kinematics.py','src/pressb/astribot_press.py','src/pressb/astribot_asset.py',
           'src/pressb/astribot_sensors.py','src/pressb/astribot_validation.py','src/pressb/base_pose.py',
           'src/pressb/vib820_scene.py','src/pressb/vib820_materials.py','src/pressb/scene.py','src/pressb/dataset_scene.py',
           'configs/astribot_sensors.json','docs/vib820_experiment_spec/config_v1_2.json',
           'assets/generated/astribot/astribot_s1.usdc',
           'vendor/astribot_descriptions/urdf/astribot_s1_urdf/astribot_whole_body_maniskill.urdf']
    names += [str(p.relative_to(root)) for p in sorted((root/'src/pressb/astribot_collection').glob('*.py'))]
    for directory in ('assets/materials/vib820','assets/generated/vib820'):
        names += [str(p.relative_to(root)) for p in sorted((root/directory).rglob('*')) if p.is_file()]
    return {n:sha256(root/n) for n in names}

def validate_gate(path,root,gpus,envs,backend,render_layout='individual',processes_per_gpu=1):
    gate=json.loads(Path(path).read_text())
    if gate.get('status')!='passed' or gate.get('source_sha256')!=source_files(root):raise ValueError('Preflight absent/stale/failed')
    for name in REQUIRED_GATES:
        evidence=gate['checks'][name]
        if evidence['status']!='passed' or sha256(evidence['path'])!=evidence['sha256']:raise ValueError(f'Invalid gate: {name}')
    expected=dict(gpus=gpus,envs_per_gpu=envs,backend=backend,render_subframes=1,render_layout=render_layout)
    if 'processes_per_gpu' in gate['configuration'] or processes_per_gpu != 1:
        expected['processes_per_gpu']=processes_per_gpu
    if processes_per_gpu > 1 and envs != 1:raise ValueError('Resident processes must each own one environment')
    if gate['configuration']!=expected:
        raise ValueError('Production configuration differs from verified configuration')
    if gate['throughput_ratio']<2 or gate['max_vram_fraction']>.8 or gate['min_available_ram_gib']<32:
        raise ValueError('Production performance/resource threshold not reached')
    return gate

def classify(row):
    if row['status']=='accepted':
        validate_commit(row['output']);return 'success'
    report_path=Path(row['output'])/'report.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else {}
    if row.get('physical_audit',{}).get('status')=='failed':return 'physical_failure'
    reason=report.get('termination_reason')
    if reason in ('initial_collision','target_ik_failed','path_ik_failed'):return 'physical_failure'
    if report.get('unexpected_contacts') or report.get('episodes') and any(not e['success'] for e in report['episodes']):
        return 'physical_failure'
    if report.get('motion_started') and any(s in report.get('error','') for s in
        ('drift','out-of-limit','TCP mismatch','stay closed','validated closed','initial raised-arm','Physical episode')):
        return 'physical_failure'
    return 'infrastructure_error'

def recover_journal(journal):
    """Recover completed commits, reject corrupt payloads, retry interrupted attempts."""
    for row in journal:
        if row.get('outcome')=='success':
            try:validate_commit(row['directory'])
            except (OSError,ValueError,KeyError) as exc:
                row.update(outcome='infrastructure_error',reason='committed_payload_failed_resume_validation',validation_error=str(exc))
        elif row.get('outcome') in ('running','queued'):
            committed=Path(row['worker'])/'episodes'/row['job_id']
            if (committed/'commit.json').exists():
                try:
                    validate_commit(committed)
                    if json.loads((committed/'job.json').read_text())!=row['configuration']:raise ValueError('Recovered commit identity mismatch')
                except (OSError,ValueError,KeyError) as exc:
                    row.update(outcome='infrastructure_error',reason='interrupted_commit_failed_validation',validation_error=str(exc))
                else:row.update(outcome='success',directory=str(committed),recovered=True)
            else:
                started=Path(row['worker'])/'started_jobs.json'
                if row['outcome']=='running' or started.exists() and row['job_id'] in json.loads(started.read_text()):
                    row.update(outcome='infrastructure_error',reason='interrupted_process')
    return journal

def run_production(args,root):
    """Jobs are frozen before launch. One app per GPU serves all assigned waves.

    Explicit infrastructure retries start fresh attempts; physical failures never
    retry. Interrupted journals and uncommitted files remain intact for diagnosis.
    """
    root=Path(root);out=args.output
    if args.render_layout=='padded':raise ValueError('Padded tiled camera projection has not passed acceptance')
    gate=validate_gate(args.preflight_report,root,args.gpus,args.envs_per_gpu,args.backend,args.render_layout,args.processes_per_gpu)
    manifest_path=out/'manifest.json'
    manifest=(json.loads(manifest_path.read_text()) if manifest_path.exists() else
              create_manifest(out,json.loads(args.config.read_text()),args.seed,args.target_successes,args.candidate_limit))
    if (manifest['seed'],manifest['target_successes'],manifest['maximum_candidates'])!=(args.seed,args.target_successes,args.candidate_limit):
        raise ValueError('Resume arguments differ from manifest')
    if manifest['config']!=json.loads(args.config.read_text()):raise ValueError('Resume configuration changed')
    frozen=out/'production_freeze.json'
    if frozen.exists():
        if json.loads(frozen.read_text())['source_sha256']!=gate['source_sha256']:raise ValueError('Production source changed')
    else:
        source_dir=out/'sources';source_dir.mkdir(exist_ok=True)
        for rel,digest in gate['source_sha256'].items():
            dst=source_dir/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/rel,dst)
        atomic_json(frozen,dict(source_sha256=gate['source_sha256'],gate=gate,started_unix_s=time.time()))
    journal_path=out/'attempts.json'
    journal=json.loads(journal_path.read_text()) if journal_path.exists() else []
    recover_journal(journal)
    def summarize():
        final={}
        for row in journal:
            cid=row['candidate_id']
            if row['outcome']=='success' or row['outcome']=='physical_failure':final[cid]=row
            elif row['outcome']=='infrastructure_error' and row['attempt']>=3:
                final[cid]={**row,'outcome':'infrastructure_exhausted'}
        selected=ordered_successes(final,args.target_successes)
        for row in journal:
            if row['outcome'] in ('physical_failure','infrastructure_error') and not row.get('reason') and row.get('directory'):
                report_path=Path(row['directory'])/'report.json'
                report=json.loads(report_path.read_text()) if report_path.exists() else {}
                row['reason']=report.get('termination_reason') or report.get('error') or row.get('result',{}).get('error') or row['outcome']
        atomic_json(out/'selection.json',[dict(candidate_id=r['candidate_id'],directory=r['directory']) for r in selected])
        atomic_json(out/'summary.json',dict(status='collected' if len(selected)==args.target_successes else 'incomplete',
            target=args.target_successes,accepted=len(selected),attempts=sum(r['outcome']!='queued' for r in journal),
            outcomes=dict(Counter(r['outcome'] for r in journal if r['outcome']!='queued')),
            failure_reasons=dict(Counter(r.get('reason',r['outcome']) for r in journal if r['outcome'] in ('physical_failure','infrastructure_error'))),
            queued_dispatches=sum(r['outcome']=='queued' for r in journal),candidate_outcomes=dict(Counter(r['outcome'] for r in final.values())),
            successful_only_distribution=[manifest['candidates'][r['candidate_id']-1] for r in selected]))
        return final,selected
    generation=0;startup_failures=0;resource_stopped=False
    while True:
        final,selected=summarize()
        if len(selected)==args.target_successes:break
        pending=[]
        for candidate in manifest['candidates']:
            cid=candidate['candidate_id']
            attempts=[r for r in journal if r['candidate_id']==cid and r['outcome']!='queued']
            if cid not in final and len(attempts)<3:
                attempt=len(attempts)+1;pending.append({**candidate,'attempt':attempt,'id':f'candidate_{cid:06d}_attempt_{attempt:02d}'})
        if not pending:
            if len(manifest['candidates'])>=args.candidate_limit:break
            manifest=extend_manifest(out);continue
        import psutil
        admission=dict(available_ram_bytes=psutil.virtual_memory().available,disk_free_bytes=shutil.disk_usage(out).free)
        reason=resource_violation(admission,args.gpus)
        if reason:
            atomic_json(out/'resource_stop.json',dict(reason=reason,unix_s=time.time(),sample=admission));break
        slots=[(gpu,slot) for gpu in args.gpus for slot in range(args.processes_per_gpu)]
        groups=[[] for _ in slots]
        for index,job in enumerate(pending):groups[(index//args.envs_per_gpu)%len(slots)].append(job)
        processes=[];seen=set()
        for (gpu,slot),jobs in zip(slots,groups):
            if not jobs:continue
            # Never reuse a prior process output directory, including interrupted runs.
            runs=out/'workers';runs.mkdir(exist_ok=True)
            run=Path(tempfile.mkdtemp(prefix=f'gpu{gpu}_slot{slot}_',dir=runs));jobs_path=run/'jobs.json';atomic_json(jobs_path,jobs)
            for job in jobs:
                journal.append(dict(candidate_id=job['candidate_id'],attempt=job['attempt'],outcome='queued',
                    worker=str(run),job_id=job['id'],gpu=gpu,worker_slot=slot,configuration=job))
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,args.gpus)),OMNICLIENT_HUB_MODE='disabled',
                OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMNI_KIT_ACCEPT_EULA='YES',PYTHONNOUSERSITE='1')
            env['TMPDIR']=tempfile.mkdtemp(prefix='a4_',dir=root/'.cache/sim_tmp')
            log=(run/'worker.log').open('wb')
            # All visible physical indices retain their meaning with the required [0,1] configuration.
            command=[sys.executable,str(root/'scripts/collect_astribot_dataset.py'),'--output',str(run),'--jobs',str(jobs_path),
                '--config',str(args.config),'--gpu',str(gpu),'--envs-per-gpu',str(args.envs_per_gpu),'--backend',args.backend,
                '--cpu-threads',str(args.cpu_threads),'--audit-commit','--render-subframes','1','--render-layout',args.render_layout,
                '--reset-probe','20','--asset-cache',str(out/'shared_assets')]
            process=subprocess.Popen(command,env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT)
            processes.append((process,run,log))
        atomic_json(journal_path,journal)
        try:
            while True:
                for process,run,log in processes:
                    sample=latest_resource(run/'resources.jsonl')
                    reason=resource_violation(sample,args.gpus) if sample else None
                    if reason and not resource_stopped:
                        resource_stopped=True
                        atomic_json(out/'resource_stop.json',dict(reason=reason,unix_s=time.time(),sample=sample))
                    started_path=run/'started_jobs.json'
                    if started_path.exists():
                        started_ids=set(json.loads(started_path.read_text()))
                        for row in journal:
                            if row['worker']==str(run) and row['outcome']=='queued' and row['job_id'] in started_ids:row['outcome']='running'
                    result_path=run/'worker_result.json'
                    if not result_path.exists():continue
                    for result in json.loads(result_path.read_text())['results']:
                        key=(str(run),result['id'])
                        if key in seen:continue
                        seen.add(key)
                        row=next(r for r in journal if r['worker']==str(run) and r['job_id']==result['id'])
                        row.update(outcome=classify(result),directory=result['output'],result=result)
                atomic_json(journal_path,journal)
                final,selected=summarize()
                if resource_stopped or len(selected)>=args.target_successes or shutil.disk_usage(out).free<20*1024**3:
                    for _,run,_ in processes:(run/'STOP').touch()
                if all(p.poll() is not None for p,_,_ in processes):break
                time.sleep(1)
        finally:
            for process,run,log in processes:
                if process.poll() is None:
                    (run/'STOP').touch();process.terminate();process.wait(timeout=30)
                log.close()
        recover_journal(journal)
        for row in journal:
            if row['outcome']=='running':row.update(outcome='infrastructure_error',reason='worker_exited_before_commit')
        atomic_json(journal_path,journal)
        generation+=1
        startup_failures=startup_failures+1 if all(not (run/'started_jobs.json').exists() for _,run,_ in processes) else 0
        if startup_failures>=3:
            raise RuntimeError('Three infrastructure launches failed before any episode; see worker logs')
        if resource_stopped or shutil.disk_usage(out).free<20*1024**3:break
    _,selected=summarize()
    if len(selected)!=args.target_successes:raise RuntimeError(f'Collection incomplete: {len(selected)}/{args.target_successes}; see journal')
    return selected
