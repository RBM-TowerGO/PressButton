"""Bounded concurrent scheduler adopting completed/in-flight work without reruns."""
import concurrent.futures
import datetime
import fcntl
import os
from pathlib import Path
import signal
import time
import traceback
from run_astribot_parallel import ROOT,OUT,SIM,read,save,spawn,audit,collect,run

def running(pid):
    if not pid:return False
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ')[1][0]!='Z'
    except FileNotFoundError:return False

def status(phase,**values):
    row=dict(phase=phase,pid=os.getpid(),execution='4 concurrent isolated environments, 2 per GPU; work-conserving queue',updated_at=datetime.datetime.now().astimezone().isoformat(),**values)
    save(OUT/'status.json',row);print(row,flush=True)

def main():
    lock=(OUT/'parallel.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=read(OUT/'experiment.json');parallel=OUT/'parallel';workers=read(parallel/'workers.json')
    save(parallel/'workers_initial_attempt.json',workers)
    transition=read(OUT/'parallel_transition.json')
    transition.update(concurrent_environments=4,environments_per_gpu=2,attempted_concurrency=8,
        adjustment='Four later instances stalled during initialization with zero formal trials. They were stopped; queue admission now limits total active simulators to four. All finished and in-flight 8000 trials retained.',supervisor_pid=os.getpid())
    save(OUT/'parallel_transition.json',transition)
    for w in workers:
        if w['step']==17000:
            assert not running(w['pid'])
            path=Path(w['path']);assert not list(path.glob('candidate_*/report.json'))
            path.rename(path.with_name(path.name+'_initialization_attempt1'))
            path.with_suffix('.log').rename(path.with_name(path.name+'_initialization_attempt1.log'))
            w['initial_attempt_pid']=w['pid'];w['pid']=None
    save(parallel/'workers.json',workers)
    legacy=OUT/'serial_step_008000'
    pending={};done=set();children={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending[str(legacy)]=pool.submit(audit,legacy)
        try:
            last_status=0
            while True:
                for w in workers:
                    if w['pid'] and not running(w['pid']) and w['path'] not in pending:
                        path=Path(w['path']);summary=read(path/'summary.json') if (path/'summary.json').exists() else {}
                        if not summary.get('complete'):raise RuntimeError(f'Worker {w["pid"]} exited before completion: {path}')
                        if w['path'] in children:assert children[w['path']].wait()==0
                        pending[w['path']]=pool.submit(audit,path)
                for w in workers:
                    if w['pid'] is not None:continue
                    active=sum(running(x['pid']) and x['gpu']==w['gpu'] for x in workers)
                    if active>=2:continue
                    path=Path(w['path']);step=w['step']
                    args=[SIM,'scripts/eval_astribot_parallel.py','--model-run',manifest['models'][str(step)]['run'],
                        '--checkpoint-step',step,'--candidate-ids',','.join(map(str,w['candidate_ids'])),'--gpu',w['gpu'],
                        '--endpoint','http://127.0.0.1:8796','--output',path,'--max-seconds',40]
                    p=spawn(args,path.with_suffix('.log'));children[w['path']]=p;w['pid']=p.pid
                    save(parallel/'workers.json',workers)
                for name,future in pending.items():
                    if future.done():future.result();done.add(name)
                if time.monotonic()-last_status>15:
                    progress=[]
                    for w in workers:
                        path=Path(w['path']);s=read(path/'summary.json') if (path/'summary.json').exists() else {}
                        progress.append(dict(step=w['step'],gpu=w['gpu'],pid=w['pid'],completed=s.get('completed',0),requested=len(w['candidate_ids']),running=running(w['pid']),queued=w['pid'] is None))
                    status('parallel_evaluating',preserved_8000=len(transition['preserved_8000_ids']),workers=progress,fully_audited_components=len(done));last_status=time.monotonic()
                if len(done)==len(workers)+1:break
                time.sleep(3)
            status('parallel_comparing')
            for step in (8000,17000):
                parts=([legacy] if step==8000 else [])+[Path(w['path']) for w in workers if w['step']==step]
                collect(OUT/f'step_{step:06d}',step,parts,manifest['candidate_ids'])
            run([SIM,'scripts/astribot_pair_analysis.py',OUT],OUT/'comparison.log')
            result=read(OUT/'comparison.json');result['parallel_transition']=transition;save(OUT/'comparison.json',result)
            with (OUT/'report.md').open('a') as stream:
                stream.write('\n## 并行执行说明\n\n按用户授权优先吞吐，保留18个已完成串行场景。曾启动8个独立实例，但后4个在初始化阶段阻塞、未产生正式试验；改为每卡最多2个、合计4个环境并发，空出的槽位立即执行排队场景。未改逐场景随机种子、相机分辨率、物理步长、7步动作或成功标准。初始化失败尝试留档，不计入80次正式试验。\n')
            status('completed',paired_initial_conditions_verified=result['paired_initial_conditions_verified'])
            for pid in (1096307,1135021):
                if running(pid):os.kill(pid,signal.SIGTERM)
        except BaseException:
            status('failed',error=traceback.format_exc());raise

if __name__=='__main__':main()
