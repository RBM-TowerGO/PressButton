"""Derived learner accounting and non-mutating actor gradient diagnostics."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
import math
import numpy as np
import torch
from pressb.online_rl.learner import LearnerConfig, SACLearner


class IncrementalReplay:
    """Ring storage and newly acquired experience have separate counters.

    The episode runner uses len/capacity to enforce the new interaction budget;
    sampling and persistence still use the full underlying source replay.
    """
    def __init__(self, replay, budget, acquired=0):
        self.storage, self.capacity, self.acquired = replay, budget, acquired
        self.obs_dim = replay.obs_dim

    def __len__(self):
        return self.acquired

    def add(self, *args, **kwargs):
        if self.acquired >= self.capacity:
            raise RuntimeError('New transition budget exhausted')
        self.storage.add(*args, **kwargs)
        self.acquired += 1


def derive(source, *, lambda_a, alpha, seed=20261024):
    config = asdict(source.config)
    config.update(lambda_a=lambda_a, init_alpha=alpha, seed=seed)
    result = SACLearner(LearnerConfig(**config), device=source.device)
    for name in ['actor', 'qs', 'q_targets']:
        getattr(result, name).load_state_dict(getattr(source, name).state_dict(), strict=True)
    return result


def gradient_diagnostics(learner, batch, seed=20261024):
    """One common reparameterized draw; autograd.grad never writes .grad.

    Diagnostics restore global RNG state and preserve optimizer/network state.
    Zero-norm cosines/angles are missing rather than fictitious zero angles.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        b = learner._prepare_batch(batch)
        actor_input = learner._actor_input(b['obs'], b['base_action'])
        action, log_prob, _ = learner.actor.sample(actor_input)
        pre_tanh = learner.actor.mean(learner.actor.net(actor_input))
        combined = learner._critic_action(action, b['base_action'])
        q = learner.qs(b['obs'], combined).mean(0)
        raw = dict(value=-q.mean(), entropy=log_prob.mean(), regularization=pre_tanh.square().mean())
        weighted = dict(value=raw['value'], entropy=learner.alpha*raw['entropy'], regularization=learner.config.lambda_a*raw['regularization'])
        params = list(learner.actor.mean.parameters())
        vectors = {}
        for name, loss in weighted.items():
            grads = torch.autograd.grad(loss, params, retain_graph=True)
            vectors[name] = torch.cat([g.flatten() for g in grads])
        total = torch.cat([g.flatten() for g in torch.autograd.grad(sum(weighted.values()), params, retain_graph=True)])
        residual_grad, = torch.autograd.grad(q.sum(), action, retain_graph=True)
        combined_grad, = torch.autograd.grad(q.sum(), combined, retain_graph=True)
        result = dict(updates=learner.updates,
            raw_loss={k: float(v.detach()) for k,v in raw.items()},
            weighted_loss={k: float(v.detach()) for k,v in weighted.items()},
            mean_head_gradient_norm={k: float(v.norm()) for k,v in vectors.items()},
            gradient_sum_max_error=float((sum(vectors.values())-total).abs().max()),
            critic_normalized_residual_gradient_rms=float(residual_grad.square().mean().sqrt()),
            critic_combined_action_gradient_rms=float(combined_grad.square().mean().sqrt()),
            q_min=float(q.detach().min()), q_max=float(q.detach().max()))
        pairs = {}
        for a, c in [('value','entropy'), ('value','regularization'), ('entropy','regularization')]:
            denom = float(vectors[a].norm()*vectors[c].norm())
            cosine = max(-1., min(1., float(torch.dot(vectors[a], vectors[c]))/denom)) if denom else None
            pairs[a+'__'+c] = dict(cosine=cosine, angle_degrees=math.degrees(math.acos(cosine)) if cosine is not None else None)
        result['gradient_pairs'] = pairs
        deterministic=learner.actor.deterministic(actor_input).detach()
        for name,values in [('deterministic',deterministic),('sampled',action.detach())]:
            chunks=values.reshape(-1,1 if values.shape[-1]==8 else 7,8)
            scale=torch.tensor(learner.config.residual_scale,device=values.device)
            stats={}
            for group,indices in [('arm',slice(0,7)),('gripper',slice(7,8))]:
                normalized=chunks[...,indices]
                physical=(chunks*scale)[...,indices]
                stats[group]=dict(normalized_rms=float(normalized.square().mean().sqrt()),radians_rms=float(physical.square().mean().sqrt()),degrees_rms=float(torch.rad2deg(physical.square().mean().sqrt())),saturation_fraction=float((normalized.abs()>=.99).float().mean()))
            result[name+'_residual_statistics']=stats
        return result


class SharedLearner(SACLearner):
    """Eight actor random variables; full 56-wide replay/critic actions."""
    def __init__(self, config, device='cpu'):
        from pressb.online_rl.learner import _Actor
        super().__init__(config, device)
        self.actor = _Actor(config.obs_dim + 7*config.base_dim, config.base_dim, config).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.policy_lr)
        # No automatic temperature tuning in this campaign.
        self.target_entropy = -config.base_dim/2

    def _critic_action(self, action, base):
        if action.shape[-1] == self.config.base_dim:
            action = action.repeat(1,7)
        return super()._critic_action(action, base)

    def act(self, *args, **kwargs):
        result = super().act(*args, **kwargs)
        if result.shape[-1] == self.config.base_dim:
            result = np.tile(result,(1,7))
        return result


def load_candidate(path):
    """Restore the exact derived structure, optimizers, counters and RNG."""
    payload = torch.load(path, map_location='cpu', weights_only=False)
    auxiliary = payload.get('extra',{}).get('auxiliary')
    if auxiliary is None and payload.get('extra',{}).get('status',{}).get('structure') != 'shared':
        return SACLearner.load(path)
    import random
    from pressb.online_rl.learner import CHECKPOINT_VERSION, SOURCE_COMMIT
    if payload['version'] != CHECKPOINT_VERSION or payload['source_commit'] != SOURCE_COMMIT:
        raise ValueError('Unsupported shared checkpoint')
    if auxiliary is None:
        result = SharedLearner(LearnerConfig(**payload['config']))
    else:
        from astribot_fast_auxiliary import AuxiliaryLearner, LabelPool
        result = AuxiliaryLearner(LearnerConfig(**payload['config']),shared=auxiliary['shared'],aux_weight=auxiliary['weight'])
        if auxiliary['pool'] is not None:
            result.pool=LabelPool()
            result.pool.load_state_dict(auxiliary['pool'])
    for name in ['actor','qs','q_targets']:
        getattr(result,name).load_state_dict(payload[name], strict=True)
    for name in ['actor_optimizer','q_optimizer','alpha_optimizer']:
        getattr(result,name).load_state_dict(payload[name])
    with torch.no_grad():
        result.log_alpha.copy_(payload['log_alpha'])
    result.updates = payload['updates']
    result.checkpoint_extra = payload['extra']
    rng = payload['rng']
    torch.set_rng_state(rng['torch'])
    np.random.set_state(rng['numpy'])
    random.setstate(rng['python'])
    result.rng.bit_generator.state = rng['learner']
    return result


def distill_shared(teacher, replay, steps=2000, seed=20261024):
    """Copy compatible hidden layers; fit mean(tanh(mu)), never tanh(mean(mu))."""
    config = asdict(teacher.config)
    config.update(init_alpha=teacher.alpha*7, auto_alpha=False, seed=seed)
    result = SharedLearner(LearnerConfig(**config))
    result.actor.net.load_state_dict(teacher.actor.net.state_dict())
    result.qs.load_state_dict(teacher.qs.state_dict())
    result.q_targets.load_state_dict(teacher.q_targets.state_dict())
    rng = np.random.default_rng(seed)
    curve = []
    for step in range(steps):
        indices = rng.integers(len(replay), size=256)
        obs = result._tensor(replay.arrays['obs'][indices], (256,config['obs_dim']), 'obs')
        base = result._base_tensor(replay.arrays['base_action'][indices],256,'base')
        x = result._actor_input(obs,base)
        with torch.no_grad():
            target = teacher.actor.deterministic(x).reshape(-1,7,8).mean(1)
        prediction = result.actor.deterministic(x)
        loss = (target-prediction).square().mean()
        result.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        result.actor_optimizer.step()
        if step % 100 == 0 or step == steps-1:
            curve.append(dict(step=step+1,mse=float(loss.detach())))
    result.actor_optimizer = torch.optim.Adam(result.actor.parameters(),lr=result.config.policy_lr)
    return result, curve
