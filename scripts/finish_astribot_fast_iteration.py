"""Continue the gated campaign with auxiliary learning and final val-40."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from supervise_astribot_fast_iteration import Supervisor,read,contact,compare
from run_astribot_online_rl import atomic_json
from run_astribot_fast_iteration import SOURCE


class Finish(Supervisor):
 def publish(self):
  self.state['updated_unix_s']=time.time();atomic_json(self.root/'finish_progress.json',self.state)
 def run(self):
  try:
   self.state['state']='waiting_stage2';self.publish()
   while True:
    progress=read(self.root/'supervisor.json')
    if progress['state']=='interrupted':raise RuntimeError('Earlier campaign interrupted; no automatic restart')
    if progress['state']=='stage2_complete':break
    time.sleep(5);self.publish()
   deadline=time.time()+120
   while Path(f"/proc/{progress['policy_pid']}").exists():
    if time.time()>deadline:raise RuntimeError('Earlier supervisor failed to release policy server')
    time.sleep(2)
   self.baseline=read(self.root/'stage0/B_mean/summary.json')
   best=progress['best']
   source=Path(best['checkpoint']) if best else SOURCE
   source_summary=best['summary'] if best else self.baseline
   self.start_server()
   if self.remaining()>5400:
    self.worker([sys.executable,str(ROOT/'scripts/prepare_astribot_fast_auxiliary.py'),'--output',str(self.root/'auxiliary_data'),'--endpoint',self.endpoint],'auxiliary_preparation')
    manifest=read(self.root/'auxiliary_data/manifest.json')
    if manifest['status']=='ready' and self.remaining()>3600:
     lam=best['lambda_a'] if best else 1.
     alpha=best['alpha'] if best else .005
     structure='auxiliary_shared' if best and best['structure']=='shared' else 'auxiliary_per_step'
     candidate=self.train('auxiliary',source,lam,alpha,structure=structure)
     if candidate is not None:
      qualifies=candidate['summary']['successes']>source_summary['successes'] and self.qualifies(candidate['summary'],Path(candidate['checkpoint'])/'learner.pt','auxiliary')
      control=None
      if qualifies:
       best=self.extend(candidate)
       if self.remaining()>3*3600:
        control=self.train('auxiliary_no_online',source,lam,alpha,structure=structure,aux_weight=0.)
        if control is not None:
         original_initial=read(self.root/'auxiliary/initial_actions.json')
         control_initial=read(self.root/'auxiliary_no_online/initial_actions.json')
         if original_initial!=control_initial:raise RuntimeError('Optional online-auxiliary control did not reproduce initialization probe outputs')
         atomic_json(self.root/'online_auxiliary_control.json',dict(candidate=candidate,control=control,same_initial_fixed_probe_outputs=True,comparison=compare(candidate['summary'],control['summary']),limitation='Both use the same auxiliary pretraining; this isolates online auxiliary weight only.'))
      atomic_json(self.root/'stage3_decision.json',dict(candidate=candidate,qualifies=qualifies,best=best,comparison_to_source=compare(candidate['summary'],source_summary),optional_zero_online_auxiliary='completed' if control else 'not_run_condition_or_budget'))
    else:
     atomic_json(self.root/'stage3_decision.json',dict(state=manifest['status'] if manifest['status']!='ready' else 'budget_not_covered',best=best))
   else:atomic_json(self.root/'stage3_decision.json',dict(state='budget_not_covered',best=best))
   # Final validation is required even when no trial earned extra training.
   # Keep continuation/source selection separate from choosing a val-40 candidate.
   import statistics
   observed=[]
   for cp in self.root.glob('*/checkpoints/new_*'):
    if not (cp/'eval_summary.json').exists():continue
    ev=read(cp/'eval_summary.json');protocol=read(cp/'protocol.json')
    observed.append(dict(name=cp.parent.parent.name,source=protocol['source'],lambda_a=protocol['lambda_a'],alpha=protocol['alpha'],structure=protocol['structure'],budget=int(cp.name.split('_')[1]),checkpoint=str(cp),summary=ev))
   def rank(c):
    common=compare(c['summary'],self.baseline)['common_success_simulation_seconds']
    return (-c['summary']['successes'],contact(c['summary']),statistics.median(x[0] for x in common) if common else float('inf'), {'reg_01':0,'reg_001':1,'reg_001_low_entropy':2,'shared':3,'auxiliary':4,'auxiliary_no_online':4}.get(c['name'],5))
   if observed:best=sorted(observed,key=rank)[0]
   decision=dict(best=best,passed=False,note='No new candidate available for complete validation' if best is None else 'Best observed candidate; continuation gates and final validation are reported separately.')
   if best is not None:
    full=self.evaluate(Path(best['checkpoint'])/'learner.pt','best_val40',True)
    decision.update(initial_val40=full,note='No independent generalization claim; val-40 participated in development')
    if full['successes']>=27 and contact(full)<=1:
     if self.remaining()>0:
      retest=self.evaluate(Path(best['checkpoint'])/'learner.pt','best_val40_retest',True)
      base=self.evaluate(SOURCE/'learner.pt','B_val40_retest',True)
      comparison=compare(retest,base)
      decision.update(candidate_retest=retest,contemporary_B_retest=base,comparison=comparison,passed=comparison['net_successes']>0 and contact(retest)<=contact(base))
     else:decision['note']='Preliminary improvement; retest budget not covered'
   atomic_json(self.root/'final_decision.json',decision)
   self.state.update(state='complete',best=best,passed=decision['passed']);self.publish()
  except BaseException:
   self.state.update(state='interrupted',error=traceback.format_exc());self.publish();raise
  finally:
   if self.child is not None and self.child.poll() is None:self.child.terminate()
   if self.server is not None and self.server.poll() is None:
    self.server.terminate()
    try:self.server.wait(timeout=30)
    except Exception:self.server.kill();self.server.wait()


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--campaign',type=Path,required=True);p.add_argument('--port',type=int,default=8796)
 Finish(p.parse_args()).run()
if __name__=='__main__':main()
