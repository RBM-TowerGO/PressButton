"""Read-only fast-iteration candidate evaluation, development or native val-40."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import traceback
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_astribot_online_rl import EpisodeCounter, atomic_json,check_health,load_contract,run_episode,sha256
from run_astribot_fast_iteration import MODEL, SCENES, ReadOnlyPolicy,digest,rms
from astribot_fast_learner import load_candidate


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--checkpoint',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--endpoint',default='http://127.0.0.1:8796')
 p.add_argument('--full-val',action='store_true')
 p.add_argument('--mode',choices=['zero','B_mean','B_sample'],default='B_mean')
 p.add_argument('--seed',type=int,default=20261024)
 p.add_argument('--deadline-unix',type=float,default=float('inf'))
 a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False)
 import torch
 torch.set_num_threads(1)
 learner=load_candidate(a.checkpoint)
 before,updates=digest(learner),learner.updates
 identity=sha256(a.checkpoint)
 check_health(a.endpoint,MODEL,8000,'state_latent')
 contract,candidates=load_contract(MODEL,'val')
 scenes=sorted(candidates) if a.full_val else SCENES
 cameras=['head','overview','wrist_left','wrist_right','panel','front','lobby'] if a.full_val else ['head','wrist_right']
 episode_args=argparse.Namespace(observation_mode='state_latent',gamma=.995,shaping=0.,endpoint=a.endpoint,max_seconds=40.,execute_horizon=7,exploration_probability=1.,scale=learner.config.residual_scale,status_every_requests=20)
 atomic_json(a.output/'protocol.json',dict(checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=identity,parameters_sha256=before,updates=updates,scenes=scenes,cameras=cameras,residual_seed=a.seed,mode=a.mode,code_sha256=sha256(Path(__file__))))
 runtime=app=None
 rows=[]
 try:
  from isaacsim import SimulationApp
  app=SimulationApp(dict(headless=True,create_new_stage=False,active_gpu=0,physics_gpu=0,multi_gpu=False,limit_cpu_threads=8,renderer='RayTracedLighting',anti_aliasing=3,width=960,height=600,disable_viewport_updates=True,extra_args=['--/app/asyncRendering=false','--/rtx/ecoMode/enabled=false','--/validate/p2p/enabled=false','--/validate/iommu/enabled=false','--/app/extensions/fsWatcherEnabled=false']))
  from astribot_closedloop_runtime import LiveRuntime
  cfg0=json.loads((ROOT/'docs/vib820_experiment_spec/config_v1_2.json').read_text())
  runtime=LiveRuntime(app,cfg0,a.output/'shared',1,0,'gpu',1,'individual',render_cameras=cameras)
  policy=ReadOnlyPolicy(learner,a.mode,None,a.seed)
  counter=EpisodeCounter(a.output)
  for candidate in scenes:
   if time.time()>=a.deadline_unix:
    atomic_json(a.output/'summary.json',dict(state='budget_not_covered',successes=sum(r['status']=='success' for r in rows),episodes=rows))
    return
   policy.actions=[]
   cfg=deepcopy(cfg0)
   c=contract['scene_conditions'][str(candidate)]
   cfg.update(robot_base_x=c['x_m'],robot_base_y=c['y_m'],robot_base_yaw_rad=c['yaw_rad'])
   row=run_episode(runtime,cfg,contract,candidate,episode_args,policy,None,counter,train=False)
   row.update(simulation_seconds=row['physics_steps']/120,executed_residual_rms=rms(np.asarray(policy.actions).reshape(-1,8)[:row['control_steps']],learner.config.residual_scale))
   out=counter.root/f'episode_{counter.next:06d}'
   np.savez_compressed(out/'residuals.npz',requested=np.asarray(policy.actions))
   atomic_json(out/'report.json',row)
   rows.append(row)
   atomic_json(a.output/'partial_summary.json',dict(episodes=rows))
  if digest(learner)!=before or learner.updates!=updates or sha256(a.checkpoint)!=identity:
   raise RuntimeError('Evaluation mutated checkpoint/parameters/counters')
  atomic_json(a.output/'summary.json',dict(successes=sum(r['status']=='success' for r in rows),abnormal_contact_episodes=sum(r['unexpected_contacts']>0 for r in rows),episodes=rows,parameters_unchanged=True,updates_unchanged=True,native_seven_cameras=a.full_val))
 except BaseException:
  atomic_json(a.output/'infrastructure_error.json',dict(error=traceback.format_exc()))
  raise
 finally:
  if runtime is not None:runtime.close()
  if app is not None:
   app.app.post_quit()
   app.close()


if __name__=='__main__':main()
