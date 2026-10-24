"""One reviewed A recovery; preserve completed B and finish A plus val-40."""
import argparse
from datetime import datetime
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
from zoneinfo import ZoneInfo

from resume_astribot_b_then_a_job import RecoveryJob, acknowledged_episodes
from run_astribot_b_then_a_job import SequentialJob
from run_astribot_latent_campaign import ROOT, metrics_summary, paired_results, read_json, sha, write_json
sys.path.insert(0, str(ROOT / 'src'))
from pressb.online_rl.learner import SACLearner
from pressb.online_rl.astribot import load_replay, BASE_DIM


class ARecoveryJob(RecoveryJob):
    def __init__(self, output, port, root, prior, acknowledged):
        SequentialJob.__init__(self, output, port, root, prior)
        self.acknowledged = acknowledged
        name = 'scripts/resume_astribot_a_job.py'
        self.source[name] = sha(ROOT / name)
        (output / 'source' / name).write_bytes((ROOT / name).read_bytes())
        self.job.update(completed=prior['completed'], adopted_worker_pid=None,
                        recovery='User-authorized reviewed A infrastructure recovery',
                        saved_A_transitions=84851, saved_A_updates=84794,
                        acknowledged_A_interruptions=acknowledged)
        write_json(output / 'protocol.json', self.job)
        self.publish()

    def gate(self, run, budget):
        status = read_json(run / 'status.json')
        if (status['transitions'], status['updates'], status['critic_init'], status['q_lr']) != (budget, budget, 'fan_in', 3e-5):
            raise RuntimeError('Recovered A budget/config mismatch')
        episodes = acknowledged_episodes(run, self.acknowledged)
        metrics = metrics_summary(run / 'metrics.jsonl')
        if episodes['actual_transitions'] != budget or metrics['updates'] != budget:
            raise RuntimeError('Recovered A data/update accounting mismatch')
        write_json(run / 'training_gate.json', dict(passed=True, metrics=metrics, episodes=episodes,
                   acknowledged_interruptions=self.acknowledged))

    def run(self):
        self.start_server()
        a = self.campaign_root / 'formal/state_only_seed20261024'
        self.worker(self.command('state_only', 20261024, 100000, a)
                    + ['--resume', '--render-cameras', 'head', 'wrist_right'],
                    'formal_20261024_state_only_train', a)
        self.gate(a, 100000)
        self.evaluate_one('state_only')
        paired_results(self.campaign_root)
        self.job.update(state='complete', result='B and A completed 100k and val-40',
                        paired_results=str(self.campaign_root / 'paired_results.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8798)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root, output = args.campaign_root.resolve(), args.output.resolve()
    prior = read_json(root / 'progress.json')
    if prior['state'] != 'stopped_with_error':
        raise RuntimeError('Expected stopped campaign')
    for key in ('supervisor_pid', 'worker_pid', 'policy_pid'):
        if Path(f'/proc/{prior[key]}').exists():
            raise RuntimeError(f'Prior process still exists: {key}')
    if any(sha(ROOT / name) != h for name, h in prior['source_sha256'].items()):
        raise RuntimeError('Training source changed')
    a = root / 'formal/state_only_seed20261024'
    b = root / 'formal/state_latent_seed20261024'
    if not all(read_json(b / name, {}).get('passed') for name in ('training_gate.json', 'evaluation_gate.json')):
        raise RuntimeError('B gates not complete')
    if (a / 'learning_error.json').exists() or (a / 'eval').exists():
        raise RuntimeError('Unexpected learning error or prior evaluation')
    if 'Connection reset by peer' not in read_json(a / 'infrastructure_error.json')['error']:
        raise RuntimeError('Unreviewed error')
    status = read_json(a / 'status.json')
    if (status['transitions'], status['updates']) != (84851, 84794):
        raise RuntimeError('Saved counts changed')
    contract = status['restore_contract']
    learner = SACLearner.load(a / 'learner.pt', device='cpu')
    if learner.updates != status['updates'] or learner.checkpoint_extra['status']['restore_contract'] != contract:
        raise RuntimeError('Checkpoint/status mismatch')
    replay = load_replay(a / 'replay.npz', capacity=100000, obs_dim=learner.config.obs_dim,
                         action_dim=learner.action_dim, base_dim=BASE_DIM, expected_contract=contract)
    if len(replay) != status['transitions']:
        raise RuntimeError('Replay/status mismatch')
    acknowledged = {}
    for path in (a / 'episodes').glob('*/report.json'):
        row = read_json(path)
        if row['status'] == 'infrastructure_error':
            acknowledged[str(path.relative_to(a))] = dict(sha256=sha(path), transitions=row['transitions'],
                status='infrastructure_interruption, not physical failure')
    if len(acknowledged) != 1 or sum(r['transitions'] for r in acknowledged.values()) != 57:
        raise RuntimeError('Unexpected interruption accounting')
    if acknowledged_episodes(a, acknowledged)['actual_transitions'] != status['transitions']:
        raise RuntimeError('Episode/replay mismatch')
    print('Verified A learner, optimizer, replay, source, reports, and completed B gates', flush=True)
    if args.verify_only:
        return
    output.mkdir(parents=True, exist_ok=False)
    archive = a / 'recovery' / datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    archive.mkdir(parents=True)
    names = ('learner.pt', 'replay.npz', 'status.json', 'protocol.json', 'infrastructure_error.json')
    for name in names:
        shutil.copy2(a / name, archive / name)
    write_json(archive / 'manifest.json', {name: sha(archive / name) for name in names})
    write_json(output / 'prior_failed_progress.json', prior)
    write_json(output / 'recovery_review.json', dict(instruction='继续跑', archive=str(archive),
        checkpoint_verified=True, preserved_transitions=84851, preserved_updates=84794,
        acknowledged_reports=acknowledged, cause='Connection reset; underlying cause unconfirmed',
        retry_policy='One reviewed recovery; stop on any new error'))
    (a / 'infrastructure_error.json').unlink()
    job = ARecoveryJob(output, args.port, root, prior, acknowledged)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        job.run()
    except BaseException:
        job.job.update(state='stopped_with_error', error=traceback.format_exc())
        write_json(output / 'campaign_error.json', job.job)
        raise
    finally:
        if job.job['state'] == 'complete':
            job.job['finished_unix_s'] = time.time()
        job.close()

if __name__ == '__main__':
    main()
