"""Leakage-controlled dynamic zero-residual collection and encoder pretraining."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_astribot_online_rl import atomic_json,check_health,load_contract,sha256
from run_astribot_fast_iteration import MODEL,SCENES
from collect_astribot_offset_probe import closedloop
from astribot_fast_auxiliary import pool_from_manifest,pretrain

HISTORY=ROOT/'outputs/astribot_online_rl/latent_offset_probe_20261023'


def select_scenes():
 contract,train=load_contract(MODEL,'train')
 history=set()
 for path in HISTORY.rglob('protocol.json'):
  value=json.loads(path.read_text());history.update(value.get('scene_ids',[]))
 for path in HISTORY.rglob('scene_*'):
  if path.is_dir():
   try:history.add(int(path.name.split('_')[1]))
   except ValueError:pass
 excluded=history|set(SCENES)|set(contract['split']['val_source_episode_ids'])|set(contract['split']['test_source_episode_ids'])
 available=sorted(set(train)-excluded)
 candidates=np.random.default_rng(20261024).permutation(available).tolist()
 controlled=[]
 for path in (HISTORY/'controlled').glob('scene_*/samples.npz'):
  with np.load(path,allow_pickle=False) as data:
   if np.all(data['split']==0):controlled.append(int(data['scene'][0]))
 if len(controlled)!=60:raise ValueError(f'Expected 60 historical controlled training scenes; got {len(controlled)}')
 controlled=np.random.default_rng(20261024).permutation(sorted(controlled)).tolist()
 return contract,candidates,controlled,sorted(excluded)


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--endpoint',default='http://127.0.0.1:8796')
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 import torch
 torch.set_num_threads(1)
 app=runtime=None
 try:
  contract,candidates,controlled,excluded=select_scenes()
  manifest=dict(seed=20261024,excluded_historical_and_heldout_ids=excluded,candidate_permutation=candidates,
   controlled=dict(directory=str(HISTORY/'controlled'),train=controlled[:48],validation=controlled[48:]),
   dynamic=dict(directory=str((a.output/'dynamic').resolve()),train=[],validation=[]),status='collecting',
   historical_controlled_protocol_sha256=sha256(HISTORY/'controlled/protocol.json'),
   code_sha256=sha256(Path(__file__)))
  atomic_json(a.output/'manifest.json',manifest)
  if len(candidates)<12:
   manifest.update(status='insufficient_unused_scenes',available=len(candidates));atomic_json(a.output/'manifest.json',manifest);return
  check_health(a.endpoint,MODEL,8000,'state_latent')
  from isaacsim import SimulationApp
  app=SimulationApp(dict(headless=True,create_new_stage=False,active_gpu=0,physics_gpu=0,multi_gpu=False,limit_cpu_threads=8,renderer='RayTracedLighting',anti_aliasing=3,width=960,height=600,disable_viewport_updates=True,extra_args=['--/app/asyncRendering=false','--/rtx/ecoMode/enabled=false','--/validate/p2p/enabled=false','--/validate/iommu/enabled=false','--/app/extensions/fsWatcherEnabled=false']))
  from astribot_closedloop_runtime import LiveRuntime
  cfg0=json.loads((ROOT/'docs/vib820_experiment_spec/config_v1_2.json').read_text())
  runtime=LiveRuntime(app,cfg0,a.output/'shared',1,0,'gpu',1,'individual',render_cameras=['head','wrist_right'])
  args=argparse.Namespace(seed=20261024,endpoint=a.endpoint,max_seconds=40.)
  summaries=[];valid=[]
  for scene in candidates[:24]:
   cfg=deepcopy(cfg0);condition=contract['scene_conditions'][str(scene)]
   cfg.update(robot_base_x=condition['x_m'],robot_base_y=condition['y_m'],robot_base_yaw_rad=condition['yaw_rad'])
   path=a.output/'dynamic'/f'scene_{scene:06d}'
   summary=closedloop(runtime,cfg,contract,scene,path,args)
   summaries.append(summary)
   if summary['near_observations']>0:valid.append(scene)
   atomic_json(a.output/'collection.json',dict(episodes=summaries,valid_scene_ids=valid))
   if len(summaries)>=12 and len(valid)>=8:break
  if len(valid)<8:
   manifest.update(status='insufficient_dynamic_coverage',valid_scene_ids=valid,episodes=len(summaries));atomic_json(a.output/'manifest.json',manifest);return
  nval=max(2,int(np.ceil(.2*len(valid))))
  manifest['dynamic'].update(train=valid[:-nval],validation=valid[-nval:],valid_scene_ids=valid,episodes=len(summaries))
  manifest.update(status='pretraining')
  atomic_json(a.output/'manifest.json',manifest)
  pool=pool_from_manifest(manifest,a.output);validation=pool_from_manifest(manifest,a.output,validation=True)
  enc,head,curve=pretrain(pool,validation)
  torch.save(dict(encoder=enc.state_dict(),head=head.state_dict()),a.output/'encoder.pt')
  atomic_json(a.output/'pretrain_curve.json',curve)
  manifest.update(status='ready',controlled_training_samples=sum(len(x[0]) for x in pool.groups[0].values()),dynamic_training_samples=sum(len(x[0]) for x in pool.groups[1].values()),encoder_sha256=sha256(a.output/'encoder.pt'))
  atomic_json(a.output/'manifest.json',manifest)
 except BaseException:
  atomic_json(a.output/'infrastructure_error.json',dict(error=traceback.format_exc()));raise
 finally:
  if runtime is not None:runtime.close()
  if app is not None:app.app.post_quit();app.close()


if __name__=='__main__':main()
