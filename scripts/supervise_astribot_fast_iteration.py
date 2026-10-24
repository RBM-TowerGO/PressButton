"""Gated fast-iteration supervisor; stop on infrastructure or learning errors.

Stage 0 is independently owned; adopt its completed artifact, never restart it.
Every candidate is an isolated process and retains its exact checkpoint/replay.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from run_astribot_online_rl import atomic_json,sha256
from run_astribot_fast_iteration import SOURCE,MODEL


def read(path):return json.loads(Path(path).read_text())


def contact(summary):
 return sum(r['unexpected_contacts']>0 for r in summary['episodes'])


def compare(a,b):
 aa={r['candidate_id']:r for r in a['episodes']}
 bb={r['candidate_id']:r for r in b['episodes']}
 if aa.keys()!=bb.keys():raise ValueError('Comparison scene sets differ')
 common=[(aa[k]['physics_steps']/120,bb[k]['physics_steps']/120) for k in aa if aa[k]['status']==bb[k]['status']=='success']
 return dict(net_successes=a['successes']-b['successes'],
  rescued=[k for k in aa if aa[k]['status']=='success' and bb[k]['status']!='success'],
  regressed=[k for k in aa if aa[k]['status']!='success' and bb[k]['status']=='success'],
  candidate_abnormal_contacts=contact(a),reference_abnormal_contacts=contact(b),common_success_simulation_seconds=common)


class Supervisor:
 def __init__(self,a):
  self.a=a;self.root=a.campaign;self.server=None;self.child=None
  self.env=os.environ.copy()
  for k in ['PYTHONPATH','PYTHONHOME']:self.env.pop(k,None)
  self.env.update(OMNI_KIT_ACCEPT_EULA='YES',PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
  self.state=dict(state='waiting_stage0',supervisor_pid=os.getpid(),events=[])
 def publish(self):
  self.state['updated_unix_s']=time.time();atomic_json(self.root/'supervisor.json',self.state)
 def worker(self,cmd,label):
  self.state.update(state=label,command=cmd);self.publish()
  atomic_json(self.root/f'{label}_command.json',cmd)
  with (self.root/f'{label}.log').open('w') as log:
   self.child=subprocess.Popen(cmd,cwd=ROOT,env=self.env,stdout=log,stderr=subprocess.STDOUT)
   self.state['worker_pid']=self.child.pid;self.publish()
   while self.child.poll() is None:
    if self.server.poll() is not None:raise RuntimeError('Policy server exited')
    time.sleep(5)
    self.publish()
   if self.child.returncode:raise RuntimeError(f'{label} exited {self.child.returncode}; see log')
  self.state['events'].append(dict(phase=label,completed_unix_s=time.time()));self.publish()
 def remaining(self):
  return 22*3600-(time.time()-read(self.root/'progress.json')['started_unix_s'])
 def start_server(self):
  with socket.socket() as probe:probe.bind(('127.0.0.1',self.a.port))
  with (self.root/'derived_policy_server.log').open('w') as log:
   self.server=subprocess.Popen([str(ROOT/'.venv-astribot-inference/bin/python'),str(ROOT/'scripts/astribot_policy_server.py'),'--run',str(MODEL),'--device','cuda:0','--port',str(self.a.port)],cwd=ROOT,env=self.env,stdout=log,stderr=subprocess.STDOUT)
  self.state['policy_pid']=self.server.pid;self.publish()
  deadline=time.monotonic()+300
  while True:
   if self.server.poll() is not None:raise RuntimeError('Derived policy server startup failed')
   try:
    with urllib.request.urlopen(self.endpoint+'/health',timeout=2) as response:
     if json.load(response).get('ready'):return
   except OSError:pass
   if time.monotonic()>deadline:raise TimeoutError('Policy startup timeout')
   time.sleep(2)
 @property
 def endpoint(self):return f'http://127.0.0.1:{self.a.port}'
 def evaluate(self,checkpoint,label,full=False):
  output=self.root/label
  cmd=[sys.executable,str(ROOT/'scripts/eval_astribot_fast_candidate.py'),'--checkpoint',str(checkpoint),'--output',str(output),'--endpoint',self.endpoint]
  cmd.extend(['--deadline-unix',str(read(self.root/'progress.json')['started_unix_s']+24*3600)])
  if full:cmd.append('--full-val')
  self.worker(cmd,label)
  result=read(output/'summary.json')
  if result.get('state')=='budget_not_covered':raise RuntimeError('Evaluation stopped at 24h budget; partial episodes preserved, not task failures')
  return result
 def qualifies(self,summary,checkpoint,label):
  diff=summary['successes']-self.baseline['successes']
  if contact(summary)>contact(self.baseline) or diff<1:return False
  if diff>=2:return True
  retest=self.evaluate(checkpoint,label+'_candidate_retest')
  base=self.evaluate(SOURCE/'learner.pt',label+'_B_retest')
  atomic_json(self.root/(label+'_retest_comparison.json'),compare(retest,base))
  return retest['successes']>base['successes'] and contact(retest)<=contact(base)
 def train(self,name,source,lam,alpha,budget=5000,structure='per_step',resume=False,aux_weight=.1):
  run=self.root/name
  cmd=[sys.executable,str(ROOT/'scripts/run_astribot_fast_candidate.py'),'--output',str(run),'--source',str(source),'--campaign',str(self.root),'--endpoint',self.endpoint,'--lambda-a',str(lam),'--alpha',str(alpha),'--budget',str(budget),'--structure',structure]
  if structure.startswith('auxiliary'):
   cmd.extend(['--aux-data',str(self.root/'auxiliary_data'),'--aux-weight',str(aux_weight)])
  if resume:cmd.append('--resume')
  self.worker(cmd,f'{name}_{budget}')
  for marker in ['learning_error.json','infrastructure_error.json']:
   if (run/marker).exists():raise RuntimeError(f'Candidate stopped: {run/marker}')
  status=read(run/'status.json')
  if status['state']=='budget_boundary':return None
  if status['new_transitions']!=budget or status['updates']!=budget:raise ValueError('Candidate budget mismatch')
  cp=run/'checkpoints'/f'new_{budget:06d}'
  result=read(cp/'eval_summary.json')
  return dict(name=name,source=str(source),lambda_a=lam,alpha=alpha,structure=structure,budget=budget,checkpoint=str(cp),summary=result)
 def extend(self,candidate):
  best=candidate;misses=0
  for budget in [10000,15000,20000]:
   if self.remaining()<3600:break
   new=self.train(candidate['name'],Path(candidate['source']),candidate['lambda_a'],candidate['alpha'],budget,candidate['structure'],True)
   if new is None:break
   if new['summary']['successes']>best['summary']['successes'] and contact(new['summary'])<=contact(self.baseline):
    best=new;misses=0
   else:misses+=1
   if misses>=2:break
  return best
 def run(self):
  try:
   self.publish()
   while not (self.root/'stage0/summary.json').exists():
    p=read(self.root/'progress.json')
    if p['state']=='infrastructure_interrupted':raise RuntimeError('Stage 0 interrupted; do not automatically restart')
    time.sleep(5);self.publish()
   # Wait for the independent stage-0 process to release its server and GPU.
   old_pid=read(self.root/'progress.json')['policy_pid']
   deadline=time.time()+120
   while Path(f'/proc/{old_pid}').exists():
    if time.time()>deadline:raise RuntimeError('Stage 0 did not clean up its policy server')
    time.sleep(2)
   self.start_server()
   self.baseline=read(self.root/'stage0/B_mean/summary.json')
   candidates=[]
   for name,lam in [('reg_01',.1),('reg_001',.01)]:
    if self.remaining()<3600:break
    result=self.train(name,SOURCE,lam,.005)
    if result is not None:candidates.append(result)
   qualified=[]
   for candidate in candidates:
    if self.qualifies(candidate['summary'],Path(candidate['checkpoint'])/'learner.pt',candidate['name']):qualified.append(candidate)
   if not qualified and len(candidates)==2 and self.remaining()>3600:
    import numpy as np
    from astribot_fast_learner import load_candidate
    from run_astribot_fast_iteration import ReadOnlyPolicy,rms
    reference=load_candidate(SOURCE/'learner.pt')
    policy=ReadOnlyPolicy(reference,'B_mean',None,20261024)
    with np.load(self.root/'diagnostic_observations.npz') as data:
     for i in range(0,len(data['obs']),256):policy.act(data['obs'][i:i+256],data['base_action'][i:i+256])
    reference_deg=rms(policy.actions,reference.config.residual_scale)['arm']['degrees_rms']
    del reference
    values=[read(Path(c['checkpoint'])/'fixed_observation_actions.json')['B_mean']['arm']['degrees_rms'] for c in candidates]
    trigger=all(v<=2*reference_deg for v in values)
    atomic_json(self.root/'low_entropy_trigger.json',dict(trigger=trigger,candidate_arm_rms_degrees=values,reference_arm_rms_degrees=reference_deg,reference_inputs='diagnostic_observations.npz'))
    if trigger:
     candidate=self.train('reg_001_low_entropy',SOURCE,.01,.001)
     if candidate is not None:
      candidates.append(candidate)
      if self.qualifies(candidate['summary'],Path(candidate['checkpoint'])/'learner.pt',candidate['name']):qualified.append(candidate)
   def rank(c):
    import statistics
    times=[x[0] for x in compare(c['summary'],self.baseline)['common_success_simulation_seconds']]
    return (-c['summary']['successes'],contact(c['summary']),statistics.median(times) if times else float('inf'),-c['lambda_a'])
   best=self.extend(sorted(qualified,key=rank)[0]) if qualified else None
   atomic_json(self.root/'stage1_decision.json',dict(candidates=candidates,qualified=[c['name'] for c in qualified],best=best,limitation='No equal-budget lambda_a=1 continuation control; extra training is confounded with regularization change.'))
   source=Path(best['checkpoint']) if best else SOURCE
   source_summary=best['summary'] if best else self.baseline
   lam=best['lambda_a'] if best else 1.
   alpha=best['alpha'] if best else .005
   if self.remaining()>3600:
    shared=self.train('shared',source,lam,alpha*7,structure='shared')
    if shared is not None:
     qualifies=shared['summary']['successes']>source_summary['successes'] and self.qualifies(shared['summary'],Path(shared['checkpoint'])/'learner.pt','shared')
     if qualifies:best=self.extend(shared)
     atomic_json(self.root/'stage2_decision.json',dict(candidate=shared,qualifies=qualifies,best=best,comparison_to_source=compare(shared['summary'],source_summary)))
   self.state.update(state='stage2_complete',best=best,next_stage='auxiliary_pending')
   self.publish()
  except BaseException:
   self.state.update(state='interrupted',error=traceback.format_exc());self.publish();raise
  finally:
   if self.child is not None and self.child.poll() is None:
    self.child.terminate()
    try:self.child.wait(timeout=30)
    except subprocess.TimeoutExpired:self.child.kill();self.child.wait()
   if self.server is not None and self.server.poll() is None:
    self.server.terminate()
    try:self.server.wait(timeout=30)
    except subprocess.TimeoutExpired:self.server.kill();self.server.wait()


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--campaign',type=Path,required=True)
 p.add_argument('--port',type=int,default=8796)
 a=p.parse_args();Supervisor(a).run()

if __name__=='__main__':main()
