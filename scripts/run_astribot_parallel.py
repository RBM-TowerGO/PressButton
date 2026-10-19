"""Eight concurrent isolated simulator environments, preserving finished trials."""
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
import urllib.request
from run_astribot_pair import ROOT,OUT,SIM,POLICY,environment,digest

PARENT=1096092
OLD_EVAL=1099910
OLD_POLICY=1096307

def read(path):return json.loads(path.read_text())
def save(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def status(phase,**values):
    row=dict(phase=phase,pid=os.getpid(),execution='8 concurrent isolated environments; 4 per GPU',updated_at=datetime.datetime.now().astimezone().isoformat(),**values)
    save(OUT/'status.json',row);print(json.dumps(row,ensure_ascii=False),flush=True)
def running(pid):
    p=Path(f'/proc/{pid}/stat')
    return p.exists() and p.read_text().split(') ')[1][0]!='Z'
def spawn(args,log,sim=True):
    with log.open('w') as stream:
        return subprocess.Popen(list(map(str,args)),cwd=ROOT,env=environment(sim),stdout=stream,stderr=subprocess.STDOUT)
def run(args,log):
    p=spawn(args,log);code=p.wait()
    if code:raise RuntimeError(f'Command failed {code}: {args}; see {log}')
def audit(path):
    run([SIM,'scripts/audit_astribot_closedloop.py',path],path.with_name(path.name+'_audit.log'))
    a=read(path/'audit.json');assert a['complete'] and a['passed'] and a['all_video_frames_decoded'];return a

def collect(root,step,parts,ids):
    root.mkdir(exist_ok=False)
    reports={};audits={};provenance=[];reference=None
    for part in parts:
        protocol=read(part/'protocol.json');a=read(part/'audit.json')
        assert a['passed'] and a['all_video_frames_decoded']
        provenance.append(dict(path=str(part),protocol_sha256=digest(part/'protocol.json'),candidate_ids=protocol['candidate_ids']))
        meta=read(part/'shared/metadata.json')
        geometry=dict(hand=meta['hand'],buttons=meta['scene']['buttons'],base_lift_m=meta['meta']['base_lift_m'])
        if reference is None:reference=geometry;(root/'shared').symlink_to(part/'shared',target_is_directory=True)
        else:assert geometry==reference,'Geometry changed across workers'
        for row in a['results']:
            cid=row['candidate_id'];assert cid not in reports,'Duplicate counted scene'
            path=part/f'candidate_{cid:06d}';reports[cid]=read(path/'report.json');audits[cid]=row
            (root/path.name).symlink_to(path,target_is_directory=True)
    assert set(reports)==set(ids)
    protocol=dict(protocol,candidate_ids=ids,checkpoint_step=step,component_protocols=provenance,
                  source_sha256=read(OUT/'parallel_transition.json')['evaluation_sources'],
                  execution='Serial completed prefix retained for 8000; remaining trials run in eight independent concurrent environments.',
                  note='User explicitly accepts parallel execution differences. Per-scene seeds, cameras, physics, action horizon and success criteria unchanged.')
    save(root/'protocol.json',protocol)
    rows=[reports[c] for c in ids]
    save(root/'summary.json',dict(complete=True,requested=40,completed=40,successes=sum(r['success'] for r in rows),success_rate=sum(r['success'] for r in rows)/40,results=rows))
    save(root/'audit.json',dict(passed=True,complete=True,requested=40,audited=40,successes=sum(r['success'] for r in rows),model_sha256=protocol['model_sha256'],all_video_frames_decoded=True,results=[audits[c] for c in ids],component_audits=[str(p/'audit.json') for p in parts]))

def main():
    lock=(OUT/'parallel.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=read(OUT/'experiment.json');ids=manifest['candidate_ids'];workers=[];new_server=None
    parallel=OUT/'parallel';parallel.mkdir(exist_ok=False)
    try:
        status('parallel_starting_policy')
        model=manifest['models']['17000'];weight=Path(model['run'])/'checkpoints/step_017000/model.pt';partial=weight.with_suffix('.pt.partial')
        if not weight.exists():
            assert partial.stat().st_size==model['size_bytes'] and digest(partial)==model['sha256'];partial.replace(weight)
        else:assert digest(weight)==model['sha256']
        new_server=spawn([POLICY,'scripts/astribot_paired_policy_server.py','--run',model['run'],'--checkpoint-step',17000,'--device','cuda:0','--port',8796],OUT/'policy_017000.log',False)
        deadline=time.monotonic()+600
        while True:
            if new_server.poll() is not None:raise RuntimeError('17000 policy exited')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8796/health',timeout=3) as response:health=json.load(response)
            except OSError:health=None
            if health:
                assert health['model_sha256']==model['sha256'] and health['loaded_tensors']==1617;break
            if time.monotonic()>deadline:raise TimeoutError('17000 policy startup')
            time.sleep(2)
        smoke=parallel/'smoke_017000'
        run([SIM,'scripts/eval_astribot_parallel.py','--model-run',model['run'],'--checkpoint-step',17000,'--candidate-ids',ids[0],'--gpu',1,'--endpoint','http://127.0.0.1:8796','--output',smoke,'--max-seconds',2,'--smoke'],parallel/'smoke_017000.log')
        audit(smoke);run([SIM,'scripts/check_astribot_pair_smoke.py',smoke],parallel/'smoke_017000_geometry.log')
        status('parallel_handoff',policy_pids=[OLD_POLICY,new_server.pid])
        old=OUT/'step_008000';initial=read(old/'summary.json')['completed'];deadline=time.monotonic()+180
        while running(OLD_EVAL) and read(old/'summary.json')['completed']==initial and time.monotonic()<deadline:time.sleep(.25)
        if running(OLD_EVAL):os.kill(OLD_EVAL,signal.SIGTERM)
        deadline=time.monotonic()+30
        while running(OLD_EVAL) and time.monotonic()<deadline:time.sleep(.5)
        if running(OLD_EVAL):raise RuntimeError('Original evaluator did not stop')
        if running(PARENT):os.kill(PARENT,signal.SIGTERM);os.kill(PARENT,signal.SIGCONT)
        legacy=OUT/'serial_step_008000';old.rename(legacy)
        completed=[r['candidate_id'] for r in read(legacy/'summary.json')['results']]
        original=read(legacy/'protocol.json');save(legacy/'protocol_original.json',original)
        # The preserved prefix is audited as an explicitly truncated component.
        save(legacy/'protocol.json',dict(original,candidate_ids=completed,truncated_for_parallel_handoff=True))
        remaining=[cid for cid in ids if cid not in completed]
        sources=dict(original['source_sha256']);sources['eval_astribot_parallel.py']=digest(ROOT/'scripts/eval_astribot_parallel.py')
        transition=dict(at=datetime.datetime.now().astimezone().isoformat(),user_authorized_parallel_differences=True,
            preserved_8000_ids=completed,remaining_8000_ids=remaining,all_17000_ids=ids,concurrent_environments=8,
            environments_per_gpu=4,policy_8000_gpu=1,policy_17000_gpu=0,evaluation_sources=sources,
            interrupted_unfinished_trial_excluded=True,original_supervisor_pid=PARENT,supervisor_pid=os.getpid())
        save(OUT/'parallel_transition.json',transition)
        for step,candidates,gpu,port in [(8000,remaining,0,8795),(17000,ids,1,8796)]:
            for index in range(4):
                shard=candidates[index::4]
                if not shard:continue
                gpu=index%2
                target=parallel/f'step_{step:06d}_worker_{index}'
                p=spawn([SIM,'scripts/eval_astribot_parallel.py','--model-run',manifest['models'][str(step)]['run'],'--checkpoint-step',step,
                    '--candidate-ids',','.join(map(str,shard)),'--gpu',gpu,'--endpoint',f'http://127.0.0.1:{port}','--output',target,'--max-seconds',40],target.with_suffix('.log'))
                workers.append(dict(step=step,gpu=gpu,pid=p.pid,path=str(target),candidate_ids=shard,process=p))
                time.sleep(2)
        save(parallel/'workers.json',[{k:v for k,v in w.items() if k!='process'} for w in workers])
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            pending={str(legacy):pool.submit(audit,legacy)};audited=set();last_status=0
            while True:
                active=[];progress=[]
                for w in workers:
                    p=w['process'];path=Path(w['path']);code=p.poll()
                    if code is not None and code!=0:raise RuntimeError(f'Worker {w["pid"]} failed {code}; see {path}.log')
                    if code is None:active.append(w['pid'])
                    elif w['path'] not in pending:pending[w['path']]=pool.submit(audit,path)
                    summary=read(path/'summary.json') if (path/'summary.json').exists() else {}
                    progress.append(dict(step=w['step'],gpu=w['gpu'],pid=w['pid'],completed=summary.get('completed',0),requested=len(w['candidate_ids']),running=code is None))
                for name,future in pending.items():
                    if future.done():future.result();audited.add(name)
                if time.monotonic()-last_status>15:
                    status('parallel_evaluating' if active else 'parallel_auditing',preserved_8000=len(completed),workers=progress,fully_audited_components=len(audited));last_status=time.monotonic()
                if not active and len(audited)==len(workers)+1:break
                time.sleep(3)
        status('parallel_comparing')
        for step in (8000,17000):
            parts=([legacy] if step==8000 else [])+[Path(w['path']) for w in workers if w['step']==step]
            collect(OUT/f'step_{step:06d}',step,parts,ids)
        run([SIM,'scripts/astribot_pair_analysis.py',OUT],OUT/'comparison.log')
        with (OUT/'report.md').open('a') as stream:
            stream.write('\n## 并行执行说明\n\n用户明确授权接受多环境实现差异。已完成的8,000步串行前缀保留，剩余试验采用8个独立仿真实例并发、每卡4个环境；原子场景的种子、观测分辨率、动作执行和成功标准未改。协议组件及切换记录均保留。同步仿真的墙钟耗时受并发影响，比较动作完成时间采用仿真秒数。\n')
        result=read(OUT/'comparison.json');result['parallel_transition']=transition;save(OUT/'comparison.json',result)
        status('completed',paired_initial_conditions_verified=result['paired_initial_conditions_verified'])
    except BaseException:
        status('failed',error=traceback.format_exc());raise
    finally:
        for w in workers:
            if w['process'].poll() is None:w['process'].terminate()
        if new_server and new_server.poll() is None:new_server.terminate()
        # Adopted server belongs exclusively to this evaluation.
        if not running(OLD_EVAL) and running(OLD_POLICY):os.kill(OLD_POLICY,signal.SIGTERM)

if __name__=='__main__':main()
