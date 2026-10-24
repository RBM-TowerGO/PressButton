"""Isolated derived mean-regularization training with incremental budgets.

Import B weights and the full training replay, reset optimizers once, and
resume only this derived job's own state. Never edits source artifacts.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import shutil
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from run_astribot_online_rl import (EpisodeCounter, atomic_json, check_health, load_contract,
    run_episode, sha256, learning_stall, LearningStop)
from run_astribot_fast_iteration import SOURCE, MODEL, SCENES, ReadOnlyPolicy, digest, rms
from astribot_fast_learner import IncrementalReplay, derive, gradient_diagnostics, load_candidate, distill_shared
from pressb.online_rl.learner import SACLearner
from pressb.online_rl.astribot import load_replay, save_replay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--endpoint', default='http://127.0.0.1:8796')
    p.add_argument('--lambda-a', type=float, required=True)
    p.add_argument('--alpha', type=float, default=.005)
    p.add_argument('--budget', type=int, default=5000)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--structure', choices=['per_step','shared','auxiliary_per_step','auxiliary_shared'], default='per_step')
    p.add_argument('--aux-data',type=Path)
    p.add_argument('--aux-weight',type=float,default=.1)
    a = p.parse_args()
    auxiliary=a.structure.startswith('auxiliary')
    shared=a.structure in ('shared','auxiliary_shared')
    clear_online=a.structure != 'per_step'
    import torch
    torch.set_num_threads(1)
    if not 0 < a.budget <= 20000 or a.budget % 5000:
        raise ValueError('Budget must be 5k, 10k, 15k or 20k cumulative new transitions')
    if not (a.campaign / 'stage0/summary.json').exists():
        raise ValueError('Stage-0 diagnosis must complete before stage 1')
    campaign_start = json.loads((a.campaign / 'progress.json').read_text())['started_unix_s']
    # A resumed candidate may extend only its new-experience limit.
    source_hashes = {name: sha256(a.source / name) for name in ['learner.pt', 'replay.npz', 'status.json', 'protocol.json']}
    code_paths = ['scripts/run_astribot_fast_candidate.py', 'scripts/astribot_fast_learner.py',
        'scripts/run_astribot_fast_iteration.py', 'scripts/run_astribot_online_rl.py',
        'src/pressb/online_rl/learner.py', 'src/pressb/online_rl/astribot.py']
    identity = dict(version=1, source=str(a.source.resolve()), source_sha256=source_hashes,
        code_sha256={s: sha256(ROOT / s) for s in code_paths}, lambda_a=a.lambda_a,
        alpha=a.alpha, seed=20261024, structure=a.structure, action_dof=8 if shared else 56, critic_action_width=56,
        campaign=str(a.campaign.resolve()), initialization='shared_distillation_empty_online_replay' if a.structure == 'shared' else 'copy_actor_critic_target_full_replay_reset_all_optimizers',
        config_difference_whitelist=['lambda_a', 'init_alpha', 'seed'])
    if auxiliary:
        if a.aux_data is None:raise ValueError('Auxiliary structure requires prepared labels')
        manifest=json.loads((a.aux_data/'manifest.json').read_text())
        if manifest['status']!='ready':raise ValueError('Auxiliary coverage/pretraining gate not ready')
        identity['auxiliary']=dict(manifest_sha256=sha256(a.aux_data/'manifest.json'),encoder_sha256=sha256(a.aux_data/'encoder.pt'),weight=a.aux_weight)
        identity['code_sha256']['scripts/astribot_fast_auxiliary.py']=sha256(ROOT/'scripts/astribot_fast_auxiliary.py')
    a.output.mkdir(parents=True, exist_ok=a.resume)
    status_path = a.output / 'status.json'
    if a.resume:
        if json.loads((a.output / 'protocol.json').read_text()) != identity:
            raise ValueError('Derived resume identity differs')
        learner = load_candidate(a.output / 'learner.pt')
        status = learner.checkpoint_extra['status']
        replay = load_replay(a.output / 'replay.npz', capacity=status['replay_capacity'], obs_dim=learner.config.obs_dim, action_dim=56, base_dim=8)
        if status.get('replay_position')!=replay.position or status.get('replay_rng_state')!=replay.rng.bit_generator.state:
            raise ValueError('Derived replay/checkpoint save boundary differs; use a complete checkpoint')
        if json.loads(status_path.read_text())!=status:
            raise ValueError('Derived status/checkpoint save boundary differs; preserve evidence before recovery')
    else:
        teacher = load_candidate(a.source / 'learner.pt')
        source_status = teacher.checkpoint_extra['status']
        source_contract = source_status.get('restore_contract', source_status.get('source_contract'))
        replay = load_replay(a.source / 'replay.npz', capacity=100000,
            obs_dim=teacher.config.obs_dim, action_dim=56, base_dim=8,
            expected_contract=source_contract if a.source.resolve() == SOURCE.resolve() else None)
        imported_rows = len(replay)
        offline_updates = 0
        if auxiliary:
            from astribot_fast_auxiliary import encoder, pool_from_manifest, initialize_auxiliary
            if a.lambda_a != teacher.config.lambda_a or not np.isclose(a.alpha,teacher.alpha):
                raise ValueError('Auxiliary structure must inherit lambda and alpha')
            pool=pool_from_manifest(manifest,a.aux_data)
            enc,head=encoder(),torch.nn.Linear(64,2)
            pretrained=torch.load(a.aux_data/'encoder.pt',weights_only=False)
            enc.load_state_dict(pretrained['encoder']);head.load_state_dict(pretrained['head'])
            learner,curve=initialize_auxiliary(teacher,replay,enc,head,pool,shared,aux_weight=a.aux_weight)
            atomic_json(a.output/'offline_initialization.json',curve)
            offline_updates=4000
        elif a.structure == 'shared':
            if a.lambda_a != teacher.config.lambda_a or not np.isclose(a.alpha, teacher.alpha*7):
                raise ValueError('Shared structure must inherit lambda and use 7x source alpha')
            learner, curve = distill_shared(teacher,replay)
            atomic_json(a.output / 'distillation.json',dict(curve=curve, updates=2000,
                variance_initialization='existing fresh actor output-head initialization'))
            offline_updates = 2000
        else:
            learner = derive(teacher, lambda_a=a.lambda_a, alpha=a.alpha)
            changed = {k for k,v in asdict(learner.config).items() if v != asdict(teacher.config)[k]}
            if not changed.issubset(identity['config_difference_whitelist']):
                raise ValueError(f'Unexpected source configuration changes: {changed}')
            if digest(teacher) != digest(learner) or learner.actor_optimizer.state or learner.q_optimizer.state:
                raise ValueError('Derived initialization changed outputs or retained optimizer state')
        if clear_online:
            from pressb.online_rl.learner import ReplayBuffer
            replay=ReplayBuffer(100000,learner.config.obs_dim,56,20261024,base_dim=8)
        replay.rng = np.random.default_rng(20261024)
        status = dict(new_transitions=0, episodes=0, successes=0, replay_capacity=replay.capacity,
            imported_rows=imported_rows, online_pool_cleared=clear_online,
            offline_updates=offline_updates, actor_updates=0, structure=a.structure,
            initialization_parameters_equal=a.structure == 'per_step', optimizer_states_empty=True,
            model_sha256=source_status['model_sha256'], source_contract=source_contract)
        del teacher
        atomic_json(a.output / 'protocol.json', identity)
        initial_stats={}
        with np.load(a.campaign/'diagnostic_observations.npz') as inputs:
            for mode in ['B_mean','B_sample']:
                policy=ReadOnlyPolicy(learner,mode,None,20261024)
                for i in range(0,len(inputs['obs']),256):policy.act(inputs['obs'][i:i+256],inputs['base_action'][i:i+256])
                initial_stats[mode]=rms(policy.actions,learner.config.residual_scale)
        atomic_json(a.output/'initial_actions.json',initial_stats)
    incremental = IncrementalReplay(replay, a.budget, status['new_transitions'])
    episode_args = argparse.Namespace(observation_mode='state_latent', gamma=.995, shaping=1., endpoint=a.endpoint, max_seconds=40., execute_horizon=7, exploration_probability=1., scale=learner.config.residual_scale, status_every_requests=20)
    check_health(a.endpoint, MODEL, 8000, 'state_latent')
    contract, candidates = load_contract(MODEL, 'train')
    if auxiliary:
        excluded=set(manifest['excluded_historical_and_heldout_ids']) | learner.pool.excluded
        # Controlled 48 training scenes are allowed for RL; other historical scenes remain excluded.
        excluded-=set(manifest['controlled']['train'])
        candidates=[c for c in candidates if c not in excluded]
        if not candidates:raise ValueError('No isolated auxiliary RL training scenes')
    order = np.random.default_rng(20261024).permutation(candidates)
    counter = EpisodeCounter(a.output)
    runtime = app = None
    import run_astribot_online_rl as physical_runner
    original_recorder = physical_runner.Recorder
    if auxiliary or shared:
        from collect_astribot_offset_probe import ProbeRecorder
        original_recorder=ProbeRecorder
    class TrajectoryRecorder(original_recorder):
        def tick(self,target):
            try:
                return super().tick(target)
            finally:
                if self.records and (auxiliary or shared):
                    self.records[-1]['target_contact']=bool(self.contacts_this_tick)
                    self.records[-1]['ever_target_contact']=self.ever_contact
        def observe(self,result=None,mode='state_only'):
            obs=super().observe(result,mode)
            if auxiliary and result is not None:
                geometry=self.geometry()
                scene=int(current_scene[0])
                if not evaluating[0]:
                    learner.pool.add(1,scene,obs[64:][None],np.asarray(geometry['label_mm'])[None])
                with torch.no_grad():
                    z=torch.from_numpy(obs[None,64:])
                    actor_prediction=learner.actor.aux_head(learner.actor.net.encoder(z))[0].numpy()*10
                    critic_prediction=learner.qs.aux_head(learner.qs.encoder(z))[0].numpy()*10
                with (self.env.output/'auxiliary_labels.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(scene=scene,simulation_time=float(runtime.world.current_time),actor_prediction_mm=actor_prediction.tolist(),critic_prediction_mm=critic_prediction.tolist(),**geometry),allow_nan=False)+'\n')
            return obs
        def close(self):
            super().close()
            if self.records:
                np.savez_compressed(self.env.output / 'physics_trace.npz',
                    **{k:np.asarray([r.get(k,False) for r in self.records]) for k in self.records[0]})
            atomic_json(self.env.output / 'unexpected_contacts.json', self.unexpected)
    physical_runner.Recorder = TrajectoryRecorder
    class TracePolicy:
        def __init__(self):
            self.actions = []
        def act(self,*args,**kwargs):
            value = learner.act(*args,**kwargs)
            self.actions.extend(value.reshape(-1,7,8).copy())
            return value
    traced = TracePolicy()
    current_scene=[None]
    evaluating=[False]
    streaks = {}
    started = time.time()
    def save(state):
        status.update(new_transitions=len(incremental), updates=learner.updates,
            replay_rows=len(replay), replay_position=replay.position, replay_rng_state=replay.rng.bit_generator.state, state=state, wall_seconds_this_process=time.time()-started)
        learner.save(a.output / 'learner.pt', extra=dict(status=status))
        save_replay(a.output / 'replay.npz', replay)
        atomic_json(status_path, status)
    try:
        save('initialized')
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=True, create_new_stage=False, active_gpu=0, physics_gpu=0, multi_gpu=False, limit_cpu_threads=8, renderer='RayTracedLighting', anti_aliasing=3, width=960, height=600, disable_viewport_updates=True, extra_args=['--/app/asyncRendering=false', '--/rtx/ecoMode/enabled=false', '--/validate/p2p/enabled=false', '--/validate/iommu/enabled=false', '--/app/extensions/fsWatcherEnabled=false']))
        from astribot_closedloop_runtime import LiveRuntime
        base_cfg = json.loads((ROOT / 'docs/vib820_experiment_spec/config_v1_2.json').read_text())
        runtime = LiveRuntime(app, base_cfg, a.output / 'shared', 1, 0, 'gpu', 1, 'individual', render_cameras=['head','wrist_right'])
        last_save = learner.updates
        while len(incremental) < a.budget:
            if time.time()-campaign_start > 22*3600:
                save('budget_boundary')
                return
            candidate = int(order[status['episodes'] % len(order)])
            condition = contract['scene_conditions'][str(candidate)]
            cfg = deepcopy(base_cfg)
            cfg.update(robot_base_x=condition['x_m'], robot_base_y=condition['y_m'], robot_base_yaw_rad=condition['yaw_rad'])
            current_scene[0]=candidate
            traced.actions = []
            row = run_episode(runtime, cfg, contract, candidate, episode_args, traced, incremental, counter, train=True)
            row['executed_residual_rms'] = rms(np.asarray(traced.actions).reshape(-1,8)[:row['control_steps']],learner.config.residual_scale)
            out = counter.root / f'episode_{counter.next:06d}'
            np.savez_compressed(out/'residuals.npz',requested=np.asarray(traced.actions))
            atomic_json(out/'report.json',row)
            status['episodes'] += 1
            status['successes'] += row['status'] == 'success'
            while (not clear_online or len(incremental) >= 1000) and learner.updates < len(incremental):
                batch = replay.sample(256)
                if (learner.updates + 1) % 500 == 0:
                    diagnostic = gradient_diagnostics(learner, batch)
                    with (a.output / 'gradient_diagnostics.jsonl').open('a') as stream:
                        stream.write(json.dumps(diagnostic, allow_nan=False)+'\n')
                metrics = learner.update(batch)
                status['actor_updates'] += int(metrics['actor_updated'])
                with (a.output / 'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(metrics, allow_nan=False)+'\n')
                if learning_stall(metrics, streaks):
                    raise LearningStop('Three consecutive critic learning failures')
            status.update(new_transitions=len(incremental), updates=learner.updates)
            atomic_json(status_path, dict(status, state='training'))
            print(json.dumps(dict(transitions=len(incremental), updates=learner.updates, episode=row)), flush=True)
            if len(incremental)>=1000 and not (a.output/'first_1k_throughput.json').exists():
                elapsed=time.time()-started
                atomic_json(a.output/'first_1k_throughput.json',dict(new_transitions=len(incremental),wall_seconds=elapsed,new_transitions_per_wall_second=len(incremental)/elapsed,estimated_remaining_training_seconds=max(0,a.budget-len(incremental))*elapsed/len(incremental),note='Measured collection+update+save/startup rate; development evaluation time must be added separately'))
            if learner.updates-last_save >= 1000:
                save('training')
                last_save = learner.updates
        if learner.updates != a.budget:
            raise ValueError('Incremental update budget mismatch')
        save('evaluating')
        checkpoint = a.output / 'checkpoints' / f'new_{a.budget:06d}'
        checkpoint.mkdir(parents=True, exist_ok=False)
        learner.save(checkpoint / 'learner.pt', extra=dict(status=status))
        atomic_json(checkpoint / 'status.json', status)
        atomic_json(checkpoint / 'protocol.json', identity)
        shutil.copyfile(a.output / 'replay.npz', checkpoint / 'replay.npz')
        with np.load(a.campaign / 'diagnostic_observations.npz') as data:
            obs, base = data['obs'], data['base_action']
        before = digest(learner)
        action_stats = {}
        for mode in ['B_mean','B_sample']:
            policy = ReadOnlyPolicy(learner, mode, None, 20261024)
            for i in range(0,len(obs),256):
                policy.act(obs[i:i+256],base[i:i+256])
            action_stats[mode] = rms(policy.actions, learner.config.residual_scale)
        atomic_json(checkpoint / 'fixed_observation_actions.json', action_stats)
        evaluating[0]=True
        val_contract, _ = load_contract(MODEL, 'val')
        rows = []
        for candidate in SCENES:
            current_scene[0]=candidate
            cfg = deepcopy(base_cfg)
            condition = val_contract['scene_conditions'][str(candidate)]
            cfg.update(robot_base_x=condition['x_m'],robot_base_y=condition['y_m'],robot_base_yaw_rad=condition['yaw_rad'])
            traced.actions = []
            eval_counter = EpisodeCounter(checkpoint)
            row = run_episode(runtime,cfg,val_contract,candidate,episode_args,traced,None,eval_counter,train=False)
            row['executed_residual_rms'] = rms(np.asarray(traced.actions).reshape(-1,8)[:row['control_steps']],learner.config.residual_scale)
            out = eval_counter.root / f'episode_{eval_counter.next:06d}'
            np.savez_compressed(out/'residuals.npz',requested=np.asarray(traced.actions))
            atomic_json(out/'report.json',row)
            row['simulation_seconds'] = row['physics_steps']/120
            rows.append(row)
            atomic_json(checkpoint / 'partial_eval.json', rows)
        if digest(learner) != before or learner.updates != a.budget:
            raise RuntimeError('Evaluation changed candidate')
        atomic_json(checkpoint / 'eval_summary.json',dict(successes=sum(r['status']=='success' for r in rows), abnormal_contact_episodes=sum(r['unexpected_contacts']>0 for r in rows), episodes=rows, parameters_unchanged=True, updates_unchanged=True))
        for name, checksum in source_hashes.items():
            if sha256(a.source/name) != checksum:
                raise RuntimeError('Source changed during derived training')
        save('complete')
    except BaseException as exc:
        atomic_json(a.output / ('learning_error.json' if isinstance(exc,LearningStop) else 'infrastructure_error.json'),dict(error=traceback.format_exc()))
        save('interrupted')
        raise
    finally:
        if runtime is not None:
            runtime.close()
        if app is not None:
            app.app.post_quit()
            app.close()


if __name__ == '__main__':
    main()
