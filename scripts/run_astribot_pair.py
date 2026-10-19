"""Single detached owner of the frozen paired validation experiment."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import urllib.request

ROOT=Path('/home/js/robot/VLA/PressButton/PressB')
OUT=ROOT/'outputs/astribot_paired_val_20261019'
SIM=ROOT/'.conda/envs/pressb/bin/python'
POLICY=ROOT/'.venv-astribot-inference/bin/python'
PORT=8795

def save(name, value):
    path=OUT/name; temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)

def status(phase, **kwargs):
    row=dict(phase=phase,pid=os.getpid(),updated_at=datetime.datetime.now().astimezone().isoformat(),**kwargs)
    save('status.json',row); print(json.dumps(row,ensure_ascii=False),flush=True)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def environment(sim=False):
    env=os.environ.copy()
    for name in ('PYTHONPATH','PYTHONHOME'):env.pop(name,None)
    env.update(PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='0,1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1' if sim else '4',OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1' if sim else '4')
    if sim:
        env.update(OMNI_KIT_ACCEPT_EULA='YES',OMNICLIENT_HUB_MODE='disabled',TMPDIR=str(ROOT/'.cache/sim_tmp'),
                   HF_HOME=str(ROOT/'.cache/hf'),PATH=f'{SIM.parent}:{ROOT}/.conda/envs/lerobot/bin:'+env['PATH'])
    return env

def command(args,log):
    with (OUT/log).open('w') as stream:
        subprocess.run(list(map(str,args)),cwd=ROOT,env=environment(True),stdout=stream,stderr=subprocess.STDOUT,check=True)

def wait_model(step, model):
    path=Path(model['run'])/f'checkpoints/step_{step:06d}/model.pt'
    if not path.exists():
        partial=path.with_suffix('.pt.partial');status('waiting_checkpoint_transfer',step=step,expected_bytes=model['size_bytes'])
        deadline=time.monotonic()+7200
        previous=-1;changed=time.monotonic()
        while not path.exists():
            size=partial.stat().st_size if partial.exists() else 0
            if size==model['size_bytes']:
                status('verifying_checkpoint',step=step)
                if digest(partial)!=model['sha256']:raise ValueError('Transferred checkpoint SHA256 mismatch')
                partial.replace(path);break
            if size>model['size_bytes']:raise ValueError('Oversized checkpoint transfer')
            if size!=previous:previous=size;changed=time.monotonic()
            if time.monotonic()-changed>600:raise TimeoutError('Checkpoint transfer has stopped growing for 10 minutes')
            if time.monotonic()>deadline:raise TimeoutError('Checkpoint transfer wait exceeded 2 hours')
            time.sleep(15)
    if digest(path)!=model['sha256']:raise ValueError(f'Wrong model SHA256 at {path}')

def evaluate(step, model):
    wait_model(step,model)
    status('starting_policy',step=step)
    with (OUT/f'policy_{step:06d}.log').open('w') as stream:
        server=subprocess.Popen([str(POLICY),'scripts/astribot_paired_policy_server.py','--run',model['run'],
            '--checkpoint-step',str(step),'--device','cuda:1','--port',str(PORT)],cwd=ROOT,env=environment(),stdout=stream,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+600
            while True:
                if server.poll() is not None:raise RuntimeError(f'Policy server exited {server.returncode}')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health',timeout=3) as response:health=json.load(response)
                except (OSError,ValueError):health=None
                if health:
                    assert health['model_sha256']==model['sha256'] and health['checkpoint_step']==step and health['loaded_tensors']==1617
                    break
                if time.monotonic()>deadline:raise TimeoutError('Policy server readiness')
                time.sleep(2)
            common=[SIM,'scripts/eval_astribot_paired.py','--model-run',model['run'],'--checkpoint-step',step,
                    '--execute-horizon',7,'--endpoint',f'http://127.0.0.1:{PORT}']
            for smoke in (True,False):
                label=('smoke_' if smoke else '')+f'step_{step:06d}'
                status('smoke' if smoke else 'evaluating',step=step,policy_pid=server.pid,output=str(OUT/label))
                command(common+['--output',OUT/label,'--max-seconds',2 if smoke else 40]+(['--smoke'] if smoke else []),label+'.log')
                status('auditing_smoke' if smoke else 'auditing',step=step)
                command([SIM,'scripts/audit_astribot_closedloop.py',OUT/label],label+'_audit.log')
                if smoke:
                    audit=json.loads((OUT/label/'audit.json').read_text())
                    assert audit['passed'] and audit['complete'] and audit['all_video_frames_decoded']
                    # Exercise all added geometry diagnostics before the counted evaluation.
                    command([SIM,'scripts/check_astribot_pair_smoke.py',OUT/label],label+'_geometry.log')
        finally:
            server.terminate()
            try:server.wait(timeout=30)
            except subprocess.TimeoutExpired:server.kill();server.wait()

def main():
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'status.json').exists():raise RuntimeError('This experiment already has a status; no automatic retry')
    try:
        manifest=json.loads((OUT/'experiment.json').read_text())
        for step in (8000,17000):evaluate(step,manifest['models'][str(step)])
        status('comparing')
        command([SIM,'scripts/astribot_pair_analysis.py',OUT],'comparison.log')
        result=json.loads((OUT/'comparison.json').read_text())
        assert result['complete']
        status('completed',paired_initial_conditions_verified=result['paired_initial_conditions_verified'])
    except BaseException:
        status('failed',error=traceback.format_exc());raise

if __name__=='__main__':main()
