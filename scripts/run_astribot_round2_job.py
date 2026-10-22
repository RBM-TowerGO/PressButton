"""Run stabilized SAC, evaluate val-40, and publish local progress every 30 s."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TZ = timezone(timedelta(hours=8))


class ExistingProcess:
    """Observe a detached worker without restarting its learner or simulation."""
    def __init__(self, pid):
        self.pid, self.returncode = pid, None

    def poll(self):
        try:
            state = Path(f'/proc/{self.pid}/stat').read_text().split(') ',1)[1].split()[0]
            if state != 'Z':
                return None
        except OSError:
            pass
        # Its original wait status is unavailable. Completion is independently
        # checked through the budget, evaluation summary and error marker.
        self.returncode = 0
        return 0

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while self.poll() is None:
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(str(self.pid), timeout)
            time.sleep(min(1, max(0, deadline-time.monotonic())))
        return self.returncode

    def terminate(self):
        os.kill(self.pid, signal.SIGTERM)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def archive_attempt(output):
    """Preserve interruption evidence and recoverable state before resuming."""
    archive = output / 'recovery' / datetime.now(TZ).strftime('%Y%m%dT%H%M%S%f')
    archive.mkdir(parents=True, exist_ok=False)
    files = ('job.json', 'progress.json', 'training.log', 'evaluating.log',
             'run/infrastructure_error.json', 'run/protocol.json', 'run/status.json',
             'run/learner.pt', 'run/replay.npz')
    hashes = {}
    for relative in files:
        source = output / relative
        if source.exists():
            target = archive / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            with target.open('rb') as stream:
                hashes[relative] = hashlib.file_digest(stream, 'sha256').hexdigest()
    atomic_json(archive / 'manifest.json', hashes)
    # The archived copy remains immutable; only the active failure marker clears.
    (output / 'run/infrastructure_error.json').unlink(missing_ok=True)
    return archive


def last_metrics(path):
    try:
        with Path(path).open('rb') as stream:
            stream.seek(max(0, stream.seek(0, 2) - 65536))
            lines = stream.read().decode().splitlines()
        # The last update can omit actor metrics; include the last actor update.
        result = {}
        for line in reversed(lines):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            for key, value in row.items():
                result.setdefault(key, value)
            if 'normalized_action_norm' in result:
                break
        return result
    except OSError:
        return {}


def snapshot(output, doc, job):
    run = output / 'run'
    status = read_json(run / 'status.json', {})
    metrics = last_metrics(run / 'metrics.jsonl')
    now = datetime.now(TZ).isoformat(timespec='seconds')
    try:
        gpu = subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,power.draw',
            '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        gpu = 'unavailable'
    elapsed = time.time() - job['started_unix_s']
    transitions = status.get('transitions', 0)
    # Include initialization/update overhead so the ETA is conservative.
    added = max(0, transitions-job.get('initial_transitions', 0))
    rate = added / elapsed if elapsed > 0 else 0.0
    eta = (50000 - transitions) / rate / 3600 if rate > 0 and job['phase'] == 'training' else None
    current = dict(updated_at=now, job=job, status=status, metrics=metrics,
                   elapsed_hours=elapsed / 3600, transitions_per_second=rate,
                   remaining_training_hours=eta, gpu_csv=gpu)
    atomic_json(output / 'progress.json', current)
    with (output / 'resource_samples.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(time=now, phase=job['phase'], transitions=transitions,
                                     updates=status.get('updates', 0), gpu_csv=gpu)) + '\n')
    recent = status.get('last_episodes', [])
    episode_rate = sum(r.get('transitions', 0) for r in recent) / max(
        1e-9, sum(r.get('wall_seconds', 0) for r in recent)) if recent else 0.0
    lines = [f'更新时间：{now}（北京时间）', '',
             f'- 阶段：`{job["phase"]}`；监督进程 PID：{os.getpid()}；工作进程 PID：{job.get("child_pid", "—")}。',
             f'- 训练 transitions：{transitions:,} / 50,000；更新：{status.get("updates", 0):,}；回合：{status.get("episodes", 0)}。',
             f'- 近 20 回合训练成功率：{status.get("rolling_success_rate")}（含探索，仅用于监控）。',
             f'- 从启动起平均吞吐：{rate:.2f} transition/s；近回合采样吞吐：{episode_rate:.2f} transition/s。']
    if eta is not None:
        lines.append(f'- 按当前平均速度估算剩余训练：{eta:.2f} 小时，不含 val-40 评估。')
    if metrics:
        lines.append(f'- 最新指标：alpha={metrics.get("alpha")}；动作范数={metrics.get("normalized_action_norm")}；幅度惩罚={metrics.get("action_penalty_loss")}；critic loss={metrics.get("critic_loss")}。')
    if job.get('error'):
        lines.append(f'- 异常：{job["error"]}')
    evaluation = read_json(run / 'eval/summary.json')
    if evaluation:
        lines.append(f'- val 评估：{evaluation.get("successes")} / {evaluation.get("completed")}；目标 40 场景。')
    lines += ['', 'GPU 采样（index, utilization %, memory MiB, power W）：', '', '```text', gpu, '```', '',
              f'完整状态：`{output.relative_to(ROOT)}/progress.json`；日志、checkpoint、replay 和逐回合证据位于同目录。']
    text = doc.read_text()
    start, end = '<!-- ROUND2_LIVE_START -->', '<!-- ROUND2_LIVE_END -->'
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError('Expected one live-progress block in the document')
    before, rest = text.split(start)
    _, after = rest.split(end)
    temporary = doc.with_suffix('.md.tmp')
    temporary.write_text(before + start + '\n' + '\n'.join(lines) + '\n' + end + after)
    temporary.replace(doc)
    atomic_json(output / 'job.json', job)


def paired_comparison(output):
    reference = read_json(ROOT / 'outputs/astribot_online_rl/reference_round1/paired_comparison.json')
    result = read_json(output / 'eval/summary.json')
    if not reference or not result:
        return
    rows = {}
    for row in result['episodes']:
        candidate = str(row['candidate_id'])
        baseline = bool(reference['per_scene'][candidate]['baseline'])
        success = row['status'] == 'success' and row['reason'] == 'physical_press_and_light'
        rows[candidate] = dict(baseline=baseline, round2=success)
    assert len(rows) == result['completed'] == 40
    improved = sum(not r['baseline'] and r['round2'] for r in rows.values())
    regressed = sum(r['baseline'] and not r['round2'] for r in rows.values())
    n = improved + regressed
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(improved, regressed) + 1)) / 2**n)
    atomic_json(output / 'eval/paired_comparison.json', dict(
        scenes=40, baseline_successes=sum(r['baseline'] for r in rows.values()),
        round2_successes=sum(r['round2'] for r in rows.values()), improved=improved,
        regressed=regressed, exact_mcnemar_p=p, per_scene=rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--managed-policy-pid', type=int)
    parser.add_argument('--physics-backend', choices=('cpu', 'gpu'), default='gpu')
    parser.add_argument('--seed', type=int, default=20261021)
    parser.add_argument('--doc', type=Path,
                        default=ROOT / 'docs/astribot_rl/astribot_round2_20261020.md')
    parser.add_argument('--resume-job', action='store_true')
    parser.add_argument('--adopt-running-job', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    output = args.output.absolute()
    if args.resume_job and args.adopt_running_job:
        raise ValueError('Choose resume or adopt, not both')
    output.mkdir(parents=True, exist_ok=args.resume_job or args.adopt_running_job)
    run = output / 'run'
    previous = read_json(output/'job.json', {})
    if args.resume_job:
        for key in ('supervisor_pid', 'child_pid'):
            pid = previous.get(key)
            if pid and Path(f'/proc/{pid}').exists():
                raise RuntimeError(f'Previous {key} is still alive: {pid}')
        for name in ('learner.pt', 'replay.npz', 'status.json'):
            if not (run/name).is_file():
                raise RuntimeError(f'Resume requires {name}')
        archive = archive_attempt(output)
    if args.adopt_running_job:
        pid = previous['child_pid']
        cmdline = Path(f'/proc/{pid}/cmdline').read_bytes()
        if b'run_astribot_online_rl.py' not in cmdline or str(run).encode() not in cmdline:
            raise ValueError('Worker identity does not match the experiment')
        archive = output/'monitor_history'/datetime.now(TZ).strftime('%Y%m%dT%H%M%S%f')
        archive.mkdir(parents=True)
        for name in ('job.json','progress.json'):
            if (output/name).exists():shutil.copy2(output/name,archive/name)
    doc = args.doc.absolute()
    command = [sys.executable, str(ROOT / 'scripts/run_astribot_online_rl.py'),
               '--model-run', 'models/astribot_head_right_joint8_ft_20261018',
               '--checkpoint-step', '8000', '--output', str(run),
               '--max-transitions', '50000', '--warmup-transitions', '5000',
               '--batch-size', '256', '--max-seconds', '40', '--training-frequency', '1',
               '--checkpoint-every', '2000', '--shaping', '1.0', '--action-penalty', '0.1',
               '--policy-lr', '3e-5', '--init-alpha', '0.02', '--no-auto-alpha',
               '--seed', str(args.seed), '--gpu', '0', '--physics-backend', args.physics_backend,
               '--render-cameras', 'head', 'wrist_right']
    if args.resume_job or args.adopt_running_job:
        command.append('--resume')
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', OMNI_KIT_ACCEPT_EULA='YES',
               CUDA_VISIBLE_DEVICES='0,1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1')
    job = dict(phase='initializing', started_unix_s=time.time(), supervisor_pid=os.getpid(),
               training_command=command, policy_pid=args.managed_policy_pid,
               initial_transitions=read_json(run/'status.json',{}).get('transitions',0),
               resume_from=str(archive) if args.resume_job or args.adopt_running_job else None,
               adopted_worker_pid=previous.get('child_pid') if args.adopt_running_job else None,
               previous_training_seconds=(previous.get('finished_unix_s', time.time())
                   - previous.get('started_unix_s', time.time())) if previous else 0,
               source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (Path(__file__), ROOT/'scripts/run_astribot_online_rl.py',
                                        ROOT/'src/pressb/online_rl/learner.py',
                                        ROOT/'src/pressb/astribot_collection/runtime.py',
                                        ROOT/'scripts/astribot_closedloop_runtime.py')})
    child = None
    try:
        for phase, extra in [('training', []), ('evaluating', [
                '--resume', '--eval-only', '--eval-episodes', '40', '--eval-split', 'val',
                '--physics-backend', 'gpu', '--render-cameras', 'head', 'overview',
                'wrist_left', 'wrist_right', 'panel', 'front', 'lobby'])]:
            job['phase'] = phase
            for restart in range(4):
                adopting = args.adopt_running_job and phase == 'training' and restart == 0
                if adopting:
                    child = ExistingProcess(previous['child_pid'])
                else:
                    with (output / f'{phase}.log').open('w') as log:
                        child = subprocess.Popen(command + extra, env=env, stdout=log, stderr=subprocess.STDOUT)
                job['child_pid'] = child.pid
                while child.poll() is None:
                    snapshot(output, doc, job)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
                job[f'{phase}_exitcode'] = None if adopting else child.returncode
                if adopting:job['training_exitcode_unknown'] = True
                error = read_json(run/'infrastructure_error.json')
                if child.returncode or error is not None:
                    can_resume = all((run/name).is_file() for name in ('learner.pt','replay.npz','status.json'))
                    if (phase == 'training' and restart < 3 and can_resume
                            and 'Rendered camera/body pose differs' in (error or {}).get('error','')):
                        recovery = archive_attempt(output)
                        job.setdefault('automatic_restarts', []).append(dict(
                            at=time.time(), reason='strict_camera_pose_capture_failed', archive=str(recovery)))
                        if '--resume' not in command:
                            command.append('--resume')
                        snapshot(output, doc, job)
                        continue
                    raise RuntimeError(f'{phase} failed; inspect {phase}.log and infrastructure_error.json')
                if phase == 'training' and read_json(run/'status.json', {}).get('transitions', 0) != 50000:
                    raise RuntimeError('Training exited before reaching 50,000 transitions')
                if phase == 'evaluating' and read_json(run/'eval/summary.json', {}).get('completed') != 40:
                    raise RuntimeError('Evaluation did not complete all 40 validation scenes')
                break
        paired_comparison(run)
        job['phase'] = 'complete'
    except BaseException as error:
        job.update(phase='failed', error=f'{type(error).__name__}: {error}')
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=60)
        raise
    finally:
        job['finished_unix_s'] = time.time()
        if args.managed_policy_pid:
            try:
                cmdline = Path(f'/proc/{args.managed_policy_pid}/cmdline').read_bytes()
                if b'scripts/astribot_policy_server.py' in cmdline:
                    os.kill(args.managed_policy_pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
        snapshot(output, doc, job)


if __name__ == '__main__':
    main()
