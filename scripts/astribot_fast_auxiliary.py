"""Scene-balanced labelled pools and independent actor/critic latent encoders."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pressb.online_rl.learner import LearnerConfig,SACLearner,_Actor,_Critics
from astribot_fast_learner import SharedLearner


def encoder():
 return nn.Sequential(nn.Linear(2048,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU())


class LabelPool:
 """Equal source probability; within source uniform scene then uniform frame."""
 def __init__(self,seed=20261024):
  self.groups=[{},{}];self.rng=np.random.default_rng(seed);self.excluded=set()
 def add(self,source,scene,latent,labels):
  if scene in self.excluded:return
  latent=np.asarray(latent,dtype=np.float32).reshape(-1,2048)
  labels=np.asarray(labels,dtype=np.float32).reshape(-1,3)
  if len(latent)!=len(labels):raise ValueError('Label/feature lengths differ')
  mask=np.isfinite(labels).all(1)&np.isfinite(latent).all(1)&(np.abs(labels)<=30).all(1)
  if not mask.any():return
  x,y=latent[mask],labels[mask,:2]/10.
  if scene in self.groups[source]:
   old=self.groups[source][scene];x=np.concatenate([old[0],x]);y=np.concatenate([old[1],y])
  self.groups[source][scene]=(x,y)
 def sample(self,size=256):
  if not all(self.groups):raise ValueError('Auxiliary batches require both controlled and dynamic sources')
  xs,ys=[],[]
  for source in self.rng.integers(2,size=size):
   scenes=sorted(self.groups[source]);scene=scenes[int(self.rng.integers(len(scenes)))]
   x,y=self.groups[source][scene];i=int(self.rng.integers(len(x)))
   xs.append(x[i]);ys.append(y[i])
  return torch.tensor(np.stack(xs)),torch.tensor(np.stack(ys))
 def count(self):return sum(len(v[0]) for g in self.groups for v in g.values())
 def state_dict(self):return dict(groups=self.groups,rng=self.rng.bit_generator.state,excluded=sorted(self.excluded))
 def load_state_dict(self,value):
  self.groups=value['groups'];self.rng.bit_generator.state=value['rng'];self.excluded=set(value['excluded'])


def pool_from_manifest(manifest,root,validation=False):
 pool=LabelPool()
 for source,name in enumerate(['controlled','dynamic']):
  section=manifest[name]
  for scene in section['validation' if validation else 'train']:
   path=Path(section['directory'])/f'scene_{scene:06d}'/'samples.npz'
   with np.load(path,allow_pickle=False) as data:pool.add(source,scene,data['latent'],data['label_mm'])
 pool.excluded=set(manifest['controlled']['validation']+manifest['dynamic']['validation'])
 # Validation additions are complete; exclude validation IDs from online training.
 return pool


def pretrain(pool,validation,max_epochs=20):
 torch.manual_seed(20261024)
 enc,head=encoder(),nn.Linear(64,2)
 opt=torch.optim.Adam([*enc.parameters(),*head.parameters()],lr=1e-3)
 best=None;best_loss=float('inf');misses=0;curve=[]
 for epoch in range(max_epochs):
  losses=[]
  for _ in range(math.ceil(pool.count()/256)):
   x,y=pool.sample();loss=F.huber_loss(head(enc(x)),y,delta=1.)
   opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.detach()))
  with torch.no_grad():
   source_losses=[]
   for group in validation.groups:
    scene_losses=[]
    for x,y in group.values():
     scene_losses.append(float(F.huber_loss(head(enc(torch.from_numpy(x))),torch.from_numpy(y),delta=1.)))
    source_losses.append(float(np.mean(scene_losses)))
   val=float(np.mean(source_losses))
  curve.append(dict(epoch=epoch+1,training_loss=float(np.mean(losses)),controlled_validation_loss=source_losses[0],dynamic_validation_loss=source_losses[1],validation_loss=val))
  if val<best_loss:
   best_loss=val;best=(deepcopy(enc.state_dict()),deepcopy(head.state_dict()));misses=0
  else:misses+=1
  if misses>=3:break
 enc.load_state_dict(best[0]);head.load_state_dict(best[1])
 return enc,head,curve


class Fusion(nn.Module):
 def __init__(self,enc,net):super().__init__();self.encoder=deepcopy(enc);self.net=net
 def forward(self,x):return self.net(torch.cat([x[:,:64],self.encoder(x[:,64:2112]),x[:,2112:]],-1))


class EncodedCritic(nn.Module):
 def __init__(self,cfg,enc,head):
  super().__init__();self.encoder=deepcopy(enc);self.aux_head=deepcopy(head)
  cfg=deepcopy(cfg);cfg.obs_dim=128
  self.core=_Critics(cfg,56);self.bounded=self.core.bounded;self.count=self.core.count
 def logits(self,obs,action):return self.core.logits(torch.cat([obs[:,:64],self.encoder(obs[:,64:]),],-1),action)
 def forward(self,obs,action):
  logits=self.logits(obs,action)
  return .5*(logits.tanh()+1) if self.bounded else logits


class AuxiliaryLearner(SharedLearner):
 """Independent encoders; true geometry exists only in auxiliary losses."""
 def __init__(self,config,shared=False,pretrained=None,device='cpu',aux_weight=.1):
  SACLearner.__init__(self,config,device)
  self.shared=shared;self.aux_weight=aux_weight;self.pool=None
  enc,head=pretrained if pretrained is not None else (encoder(),nn.Linear(64,2))
  actor=_Actor(128+56,8 if shared else 56,config)
  actor.net=Fusion(enc,actor.net)
  actor.aux_head=deepcopy(head)
  self.actor=actor.to(self.device)
  self.qs=EncodedCritic(config,enc,head).to(self.device)
  self.q_targets=deepcopy(self.qs).requires_grad_(False)
  self.reset_optimizers()
 def reset_optimizers(self):
  self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=self.config.policy_lr)
  self.q_optimizer=torch.optim.Adam(self.qs.parameters(),lr=self.config.q_lr)
  self.alpha_optimizer=torch.optim.Adam([self.log_alpha],lr=self.config.q_lr)
 def update(self,batch):
  if self.config.auto_alpha:raise ValueError('Auxiliary campaign requires fixed alpha')
  if self.pool is None:raise ValueError('Labelled training pool required')
  b=self._prepare_batch(batch);target=self._bellman_target(b)
  logits=self.qs.logits(b['obs'],self._critic_action(b['action'],b['base_action']))
  current=.5*(logits.tanh()+1) if self.qs.bounded else logits
  rl=(current-target.unsqueeze(0)).square().mean((1,2)).sum()
  x,y=self.pool.sample(self.config.batch_size);x,y=x.to(self.device),y.to(self.device)
  auxiliary=F.huber_loss(self.qs.aux_head(self.qs.encoder(x)),y,delta=1.)
  loss=rl+self.aux_weight*auxiliary
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite critic loss')
  self.q_optimizer.zero_grad(set_to_none=True);loss.backward()
  norm=nn.utils.clip_grad_norm_(self.qs.parameters(),self.config.max_grad_norm,error_if_nonfinite=True)
  self.q_optimizer.step();self.updates+=1
  target_updated=self.updates%self.config.target_freq==0
  if target_updated:
   with torch.no_grad():
    for t,s in zip(self.q_targets.parameters(),self.qs.parameters(),strict=True):t.lerp_(s,self.config.tau)
  metrics=dict(updates=float(self.updates),critic_loss=float(rl.detach())/self.config.num_qs,
   critic_aux_loss=float(auxiliary.detach()),critic_grad_norm=float(norm),q_predicted=float(current.detach().mean()),q_target=float(target.mean()),actor_updated=0.,target_updated=float(target_updated),alpha=self.alpha)
  with torch.no_grad():
   for index in range(self.config.num_qs):
    metrics[f'q{index}_logit_min']=float(logits[index].min());metrics[f'q{index}_logit_max']=float(logits[index].max())
    metrics[f'q{index}_exact_boundary_fraction']=float(((current[index]==0)|(current[index]==1)).float().mean())
   for p,v in zip([0,5,50,95,100],torch.quantile(target.flatten(),target.new_tensor([0.,.05,.5,.95,1.]))):metrics[f'q_target_p{p}']=float(v)
   metrics['q_target_outside_unit_fraction']=float(((target<0)|(target>1)).float().mean())
  if self.updates%self.config.policy_freq==0:
   self.qs.requires_grad_(False)
   try:
    actor_loss,more=self._actor_loss(b)
    x,y=self.pool.sample(self.config.batch_size);x,y=x.to(self.device),y.to(self.device)
    aux=F.huber_loss(self.actor.aux_head(self.actor.net.encoder(x)),y,delta=1.)
    total=actor_loss+self.aux_weight*aux
    if not torch.isfinite(total):raise FloatingPointError('Nonfinite actor loss')
    self.actor_optimizer.zero_grad(set_to_none=True);total.backward()
    norm=nn.utils.clip_grad_norm_(self.actor.parameters(),self.config.max_grad_norm,error_if_nonfinite=True)
    self.actor_optimizer.step()
    metrics.update(more,actor_loss=float(total.detach()),actor_aux_loss=float(aux.detach()),actor_grad_norm=float(norm),actor_updated=1.)
   finally:self.qs.requires_grad_(True)
  if not all(math.isfinite(v) for v in metrics.values()):raise FloatingPointError('Nonfinite auxiliary metrics')
  return metrics


def initialize_auxiliary(teacher,replay,enc,head,pool,shared,steps=2000,aux_weight=.1):
 result=AuxiliaryLearner(teacher.config,shared=shared,pretrained=(enc,head),aux_weight=aux_weight)
 # During actor imitation, the pretrained encoder is fixed.
 result.actor.net.encoder.requires_grad_(False)
 rng=np.random.default_rng(20261024);curve=[]
 for step in range(steps):
  ids=rng.integers(len(replay),size=256)
  obs=torch.tensor(replay.arrays['obs'][ids]);base=torch.tensor(replay.arrays['base_action'][ids])
  x=result._actor_input(obs,base)
  with torch.no_grad():target=teacher.actor.deterministic(x)
  loss=F.mse_loss(result.actor.deterministic(x),target)
  result.actor_optimizer.zero_grad(set_to_none=True);loss.backward();result.actor_optimizer.step()
  if step%100==0 or step==steps-1:curve.append(dict(step=step+1,mse=float(loss.detach())))
 result.actor.net.encoder.requires_grad_(True)
 # Bellman-only critic warmup; no auxiliary or actor updates in this phase.
 warmup=[]
 for step in range(steps):
  ids=rng.integers(len(replay),size=256)
  b=result._prepare_batch({k:v[ids] for k,v in replay.arrays.items()})
  target=result._bellman_target(b)
  current=result.qs(b['obs'],result._critic_action(b['action'],b['base_action']))
  loss=(current-target.unsqueeze(0)).square().mean((1,2)).sum()
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite Bellman warmup')
  result.q_optimizer.zero_grad(set_to_none=True);loss.backward()
  nn.utils.clip_grad_norm_(result.qs.parameters(),result.config.max_grad_norm,error_if_nonfinite=True);result.q_optimizer.step()
  if (step+1)%result.config.target_freq==0:
   with torch.no_grad():
    for t,s in zip(result.q_targets.parameters(),result.qs.parameters(),strict=True):t.lerp_(s,result.config.tau)
  if step%100==0 or step==steps-1:warmup.append(dict(step=step+1,critic_loss=float(loss.detach())))
 result.reset_optimizers();result.pool=pool
 return result,dict(distillation=curve,critic_warmup=warmup,distillation_updates=steps,critic_warmup_updates=steps)


def save_auxiliary(self,path,extra=None):
 extra={} if extra is None else dict(extra)
 extra['auxiliary']=dict(shared=self.shared,weight=self.aux_weight,pool=self.pool.state_dict() if self.pool is not None else None)
 SACLearner.save(self,path,extra)

AuxiliaryLearner.save=save_auxiliary
