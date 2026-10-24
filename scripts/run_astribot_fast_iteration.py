"""Read-only stage-0 diagnostics for the latent fast-iteration plan.

Uses the existing physical episode runner without modifying the original
checkpoint, replay, evaluator, reward, or termination contract.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from run_astribot_online_rl import (EpisodeCounter, atomic_json, check_health,
    load_contract, run_episode, sha256)
from pressb.online_rl.learner import SACLearner

SOURCE = ROOT / 'outputs/astribot_online_rl/latent_fixed_campaign_20261022/run/formal/state_latent_seed20261024'
MODEL = ROOT / 'models/astribot_head_right_joint8_ft_20261018'
SCENES = sorted([1, 62, 91, 97, 159, 180, 210, 217, 3, 9, 51, 52])
SEEDS = [20261024, 20261025]


def digest(learner):
    h = hashlib.sha256()
    for net in (learner.actor, learner.qs, learner.q_targets):
        for name, value in net.state_dict().items():
            h.update(name.encode())
            h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def rms(actions, scale):
    x = np.asarray(actions, dtype=np.float64).reshape(-1, 8)
    if not x.size:
        return None
    result = {}
    for name, group in [('all', slice(None)), ('arm', slice(0, 7)), ('gripper', slice(7, 8))]:
        normalized = x[:, group]
        physical = (x * np.asarray(scale))[..., group]
        result[name] = dict(normalized_rms=float(np.sqrt(np.mean(normalized**2))),
            radians_rms=float(np.sqrt(np.mean(physical**2))),
            degrees_rms=float(np.rad2deg(np.sqrt(np.mean(physical**2)))))
    return result


class ReadOnlyPolicy:
    def __init__(self, learner, mode, bank, seed):
        import torch
        self.learner, self.mode, self.bank = learner, mode, bank
        self.rng = np.random.default_rng(seed)
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.actions, self.indices = [], []

    def act(self, obs, base_actions, **kwargs):
        import torch
        if self.mode == 'zero':
            result = np.zeros((len(obs), 56), np.float32)
        elif self.mode == 'unconditional':
            indices = self.rng.integers(len(self.bank), size=len(obs))
            self.indices.extend(indices.tolist())
            result = self.bank[indices].reshape(-1, 56).copy()
        elif self.mode == 'B_mean':
            result = self.learner.act(obs, base_actions, deterministic=True)
        elif self.mode == 'B_sample':
            # A private generator isolates residual sampling from simulation RNG.
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(self.generator.get_state())
                result = self.learner.act(obs, base_actions, deterministic=False)
                self.generator.set_state(torch.get_rng_state())
        else:
            raise ValueError(self.mode)
        self.actions.extend(result.reshape(-1, 7, 8).copy())
        return result


def make_bank(source, learner, output):
    import torch
    rng = np.random.default_rng(SEEDS[0])
    with np.load(source / 'replay.npz', allow_pickle=False) as data:
        meta = json.loads(str(data['meta']))
        if meta.get('astribot_contract') != learner.checkpoint_extra['status']['restore_contract']:
            raise ValueError('Source replay/checkpoint contracts differ')
        count = meta['size']
        if count != 100000:
            raise ValueError(f'Expected final B replay of 100k, got {count}')
        indices = rng.integers(count, size=10000)
        diagnostic_indices = np.random.default_rng(SEEDS[0]).choice(count, 2048, replace=False)
        obs, base = data['obs'][indices], data['base_action'][indices]
        np.savez_compressed(output / 'diagnostic_observations.npz',
            indices=diagnostic_indices, obs=data['obs'][diagnostic_indices],
            base_action=data['base_action'][diagnostic_indices])
    blocks = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEEDS[0])
        for start in range(0, len(indices), 256):
            blocks.append(learner.act(obs[start:start+256], base[start:start+256]))
    bank = np.concatenate(blocks).reshape(-1, 7, 8)
    np.savez_compressed(output / 'unconditional_bank.npz', residual=bank, replay_indices=indices)
    return bank


def summarize(rows):
    return dict(completed=len(rows), successes=sum(r['status'] == 'success' for r in rows),
        abnormal_contact_episodes=sum(r['unexpected_contacts'] > 0 for r in rows),
        scene_count=len(set(r['candidate_id'] for r in rows)),
        note='Small development set, deliberately enriched for historical failures; seeds are repeated scenes.',
        episodes=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--port', type=int, default=8796)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    server = app = runtime = None
    progress = dict(state='preflight', started_unix_s=started, completed=[])
    def publish():
        progress['elapsed_seconds'] = time.time() - started
        atomic_json(args.output / 'progress.json', progress)
    publish()
    try:
        import torch
        torch.set_num_threads(1)
        hashes = {name: sha256(args.source / name) for name in ['learner.pt', 'replay.npz', 'status.json', 'protocol.json']}
        learner = SACLearner.load(args.source / 'learner.pt')
        original_digest, original_updates = digest(learner), learner.updates
        saved = learner.checkpoint_extra['status']
        if original_updates != 100000 or saved['model_sha256'] != json.loads((MODEL / 'checkpoints/step_008000/manifest.json').read_text())['model_sha256']:
            raise ValueError('Source is not the final B model')
        old_contract = saved['restore_contract']
        for name, checksum in old_contract['source_sha256'].items():
            if sha256(ROOT / name) != checksum:
                raise ValueError(f'Original execution source changed: {name}')
        contract, candidates = load_contract(MODEL, 'val')
        if not set(SCENES).issubset(candidates):
            raise ValueError('Development scenes absent from val split')
        bank = make_bank(args.source, learner, args.output)
        identity = dict(source=str(args.source), source_sha256=hashes,
            source_updates=original_updates, parameter_sha256=original_digest,
            scenes=SCENES, residual_seeds=SEEDS, train_seed=SEEDS[0],
            environment_seed_rule='Frozen scene geometry; existing simulator reset; VLA request seed=420000+candidate*100003+request_index',
            bank_sha256=sha256(args.output / 'unconditional_bank.npz'),
            bank_rms=rms(bank, learner.config.residual_scale),
            code_sha256=sha256(Path(__file__)),
            input_cameras=['head', 'wrist_right'], source_contract=old_contract)
        atomic_json(args.output / 'protocol.json', identity)
        diff = subprocess.run(['git', 'diff', '--binary'], cwd=ROOT, capture_output=True, check=True)
        (args.output / 'working_tree.patch').write_bytes(diff.stdout)
        env = os.environ.copy()
        for key in ('PYTHONPATH', 'PYTHONHOME'):
            env.pop(key, None)
        env.update(PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with (args.output / 'policy_server.log').open('w') as log:
            server = subprocess.Popen([str(ROOT / '.venv-astribot-inference/bin/python'), str(ROOT / 'scripts/astribot_policy_server.py'), '--run', str(MODEL), '--device', 'cuda:0', '--port', str(args.port)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        progress.update(state='starting_policy', policy_pid=server.pid)
        publish()
        endpoint = f'http://127.0.0.1:{args.port}'
        deadline = time.monotonic() + 300
        while True:
            if server.poll() is not None:
                raise RuntimeError('Policy server startup failed; see policy_server.log')
            try:
                check_health(endpoint, MODEL, 8000, 'state_latent')
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError('Policy startup timed out')
                time.sleep(2)
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=True, create_new_stage=False, active_gpu=0, physics_gpu=0, multi_gpu=False, limit_cpu_threads=8, renderer='RayTracedLighting', anti_aliasing=3, width=960, height=600, disable_viewport_updates=True, extra_args=['--/app/asyncRendering=false', '--/rtx/ecoMode/enabled=false', '--/validate/p2p/enabled=false', '--/validate/iommu/enabled=false', '--/app/extensions/fsWatcherEnabled=false']))
        from astribot_closedloop_runtime import LiveRuntime
        base_cfg = json.loads((ROOT / 'docs/vib820_experiment_spec/config_v1_2.json').read_text())
        runtime = LiveRuntime(app, base_cfg, args.output / 'shared', 1, 0, 'gpu', 1, 'individual', render_cameras=['head', 'wrist_right'])
        episode_args = argparse.Namespace(observation_mode='state_latent', gamma=.995, shaping=1., endpoint=endpoint, max_seconds=40., execute_horizon=7, exploration_probability=1., scale=learner.config.residual_scale, status_every_requests=20)
        all_rows = []
        for mode in ['zero', 'B_mean', 'B_sample', 'unconditional']:
            seeds = SEEDS if mode in ('B_sample', 'unconditional') else SEEDS[:1]
            mode_rows = []
            for seed in seeds:
                policy = ReadOnlyPolicy(learner, mode, bank, seed)
                for candidate in SCENES:
                    if time.time() - started > 22 * 3600:
                        raise TimeoutError('22h execution boundary; reserve 2h for reporting/retest')
                    group = args.output / 'stage0' / mode / str(seed)
                    counter = EpisodeCounter(group)
                    progress.update(state='stage0', mode=mode, seed=seed, candidate=candidate)
                    publish()
                    condition = contract['scene_conditions'][str(candidate)]
                    cfg = deepcopy(base_cfg)
                    cfg.update(robot_base_x=condition['x_m'], robot_base_y=condition['y_m'], robot_base_yaw_rad=condition['yaw_rad'])
                    policy.actions, policy.indices = [], []
                    row = run_episode(runtime, cfg, contract, candidate, episode_args, policy, None, counter, train=False)
                    actions = np.asarray(policy.actions, np.float32).reshape(-1, 7, 8)
                    # The last chunk may have executed only a prefix.
                    executed = actions.reshape(-1, 8)[:row['control_steps']]
                    row.update(mode=mode, residual_seed=seed, simulation_seconds=row['physics_steps']/120,
                        requested_residual_rms=rms(actions, episode_args.scale),
                        executed_residual_rms=rms(executed, episode_args.scale))
                    out = counter.root / f'episode_{counter.next:06d}'
                    np.savez_compressed(out / 'residuals.npz', requested=actions, executed=executed, bank_indices=np.asarray(policy.indices, np.int64))
                    atomic_json(out / 'report.json', row)
                    mode_rows.append(row)
                    all_rows.append(row)
                    progress['completed'].append(dict(mode=mode, seed=seed, candidate=candidate, status=row['status']))
                    publish()
                    atomic_json(args.output / 'stage0' / 'partial_summary.json', summarize(all_rows))
                    if digest(learner) != original_digest or learner.updates != original_updates:
                        raise RuntimeError('Read-only evaluation mutated the learner')
            atomic_json(args.output / 'stage0' / mode / 'summary.json', summarize(mode_rows))
        for name, checksum in hashes.items():
            if sha256(args.source / name) != checksum:
                raise RuntimeError(f'Source artifact mutated: {name}')
        atomic_json(args.output / 'stage0' / 'summary.json', dict(**summarize(all_rows), source_unchanged=True, parameters_unchanged=True))
        progress.update(state='stage0_complete', next_stage='mean_regularization_pending_implementation')
        publish()
    except BaseException:
        progress.update(state='infrastructure_interrupted', error=traceback.format_exc())
        publish()
        raise
    finally:
        # SimulationApp.close may terminate the interpreter, so reap the
        # independently spawned inference server before closing Isaac Sim.
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        if runtime is not None:
            runtime.close()
        if app is not None:
            app.app.post_quit()
            app.close()


if __name__ == '__main__':
    main()
