"""Launch one isolated XYZ run, then paired val-40; no automatic retries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--existing-server-pid', type=int, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    acceptance = json.loads((out / 'smoke_acceptance.json').read_text())
    if not all(name in acceptance for name in ('restore', 'deterministic', 'zero')):
        raise RuntimeError('Smoke acceptance is incomplete')
    snapshot = out / 'source'
    snapshot.mkdir()
    for name in ('src', 'scripts'):
        shutil.copytree(ROOT / name, snapshot / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for entry in ROOT.iterdir():
        if entry.name not in ('src', 'scripts', '.git'):
            (snapshot / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024**2), b''): h.update(block)
        return h.hexdigest()
    sources = {str(p.relative_to(snapshot)): digest(p) for name in ('src', 'scripts')
               for p in (snapshot/name).rglob('*.py')}
    (out/'source_identity.json').write_text(json.dumps(sources, indent=2))
    model = ROOT/'models/astribot_head_right_joint8_ft_20261018'
    common = [str(ROOT/'.conda/envs/pressb/bin/python'), str(snapshot/'scripts/run_astribot_online_rl.py'),
        '--model-run', str(model), '--checkpoint-step', '8000', '--output', str(out/'run'),
        '--observation-mode', 'state_latent_raw_command', '--residual-mode', 'astribot_xyz',
        '--xyz-scale', '.01', '.01', '.01', '--max-transitions', '100000', '--warmup-transitions', '2000',
        '--batch-size', '256', '--training-frequency', '1', '--max-seconds', '40', '--execute-horizon', '7',
        '--gamma', '.995', '--exploration-probability', '1', '--shaping', '0', '--action-penalty', '0',
        '--policy-lr', '.0001', '--init-alpha', '.01', '--checkpoint-every', '1000', '--seed', '20261024',
        '--gpu', '0', '--physics-backend', 'gpu', '--endpoint', 'http://127.0.0.1:8799']
    evaluation = common + ['--resume', '--eval-only', '--eval-episodes', '40', '--eval-split', 'val',
        '--eval-max-seconds', '40', '--render-cameras', 'head', 'overview', 'wrist_left', 'wrist_right', 'panel', 'front', 'lobby']
    commands = dict(train=common+['--render-cameras', 'head', 'wrist_right'],
        baseline=evaluation+['--zero-residual', '--eval-output', str(out/'eval_baseline')],
        xyz=evaluation+['--eval-output', str(out/'eval_xyz')])
    (out/'commands.json').write_text(json.dumps(commands, indent=2))
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME'): env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', OMNI_KIT_ACCEPT_EULA='YES',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')
    job = dict(supervisor_pid=os.getpid(), server_pid=args.existing_server_pid,
               started_unix_s=time.time(), phase='preflight')
    child = None
    stopping = False
    def save():
        temp=out/'job.tmp';temp.write_text(json.dumps(job, indent=2));temp.replace(out/'job.json')
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        job['stop_requested'] = True
        save()
        if child is not None and child.poll() is None: child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGINT, stop);signal.signal(signal.SIGTERM, stop)
    def phase(name):
        nonlocal child
        job.update(phase=name);save()
        with (out/(name+'.log')).open('w') as log:
            child=subprocess.Popen(commands[name], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            job['child_pid']=child.pid;save()
            code=child.wait()
        if code: raise RuntimeError(f'{name} exited {code}')
    try:
        phase('train')
        status=json.loads((out/'run/status.json').read_text())
        if stopping:
            job['phase']='stopped_at_episode_boundary'
            return
        if status['transitions'] != 100000 or status['updates'] != 100000:
            raise RuntimeError('Training budget incomplete')
        protected=['learner.pt','replay.npz','status.json','protocol.json']
        before={n:digest(out/'run'/n) for n in protected}
        for name in ('baseline','xyz'):
            if stopping: return
            phase(name)
            if before != {n:digest(out/'run'/n) for n in protected}:
                raise RuntimeError('Evaluation mutated training artifacts')
        summaries={n:json.loads((out/('eval_'+n)/'summary.json').read_text()) for n in ('baseline','xyz')}
        if any(s['completed'] != 40 for s in summaries.values()): raise RuntimeError('Incomplete val-40')
        baseline={r['candidate_id']:r for r in summaries['baseline']['episodes']}
        pairs=[dict(candidate_id=r['candidate_id'],baseline=baseline[r['candidate_id']]['status']=='success',
                    xyz=r['status']=='success') for r in summaries['xyz']['episodes']]
        report=dict(per_scene=pairs,baseline_successes=summaries['baseline']['successes'],
                    xyz_successes=summaries['xyz']['successes'],
                    improved=sum(not r['baseline'] and r['xyz'] for r in pairs),
                    regressed=sum(r['baseline'] and not r['xyz'] for r in pairs),
                    note='Single seed; action space and critic representation both changed.')
        report['diagnostics'] = {}
        for name, summary in summaries.items():
            rows = summary['episodes']
            execution = [r.get('xyz_execution', {}) for r in rows]
            chunks = sum(r.get('chunks', 0) for r in execution)
            report['diagnostics'][name] = dict(
                unexpected_contacts=sum(r['unexpected_contacts'] for r in rows),
                timeouts=sum(r['status'] == 'timeout' for r in rows),
                reduced_fraction=sum(r.get('reduced', 0) for r in execution) / max(1, chunks),
                fallback_fraction=sum(r.get('fallback', 0) for r in execution) / max(1, chunks),
                ik_seconds=sum(r.get('seconds', 0) for r in execution),
                control_seconds=sum(r['timings']['control_s'] for r in rows),
                max_position_error_m=max((r.get('max_position_error_m', 0) for r in execution), default=0),
                mean_episode_saturation_fraction=sum(r['residual_saturation_fraction'] for r in rows)/len(rows))
        (out/'paired_comparison.json').write_text(json.dumps(report,indent=2))
        job['phase']='complete'
    except BaseException:
        job.update(phase='failed',error=traceback.format_exc())
        raise
    finally:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            child.wait()
        try:
            os.kill(args.existing_server_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        job['finished_unix_s']=time.time();save()


if __name__ == '__main__':
    main()
