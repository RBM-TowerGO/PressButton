import json
import time
import hashlib

import run_astribot_round2_job as job


def test_progress_reads_child_run_and_preserves_document(tmp_path, monkeypatch):
    monkeypatch.setattr(job, 'ROOT', tmp_path)
    monkeypatch.setattr(job.subprocess, 'check_output', lambda *a, **k: '0, 75, 8000, 220\n1, 80, 11000, 280')
    output = tmp_path / 'experiment'
    run = output / 'run'
    run.mkdir(parents=True)
    (run / 'status.json').write_text(json.dumps(dict(transitions=100, updates=20, episodes=2)))
    (run / 'metrics.jsonl').write_text(
        '{"alpha":0.02,"normalized_action_norm":0.3,"updates":18}\n'
        '{"alpha":0.02,"updates":19}\n{"unfinished":')
    doc = tmp_path / 'progress.md'
    doc.write_text('intro\n<!-- ROUND2_LIVE_START -->\nold\n<!-- ROUND2_LIVE_END -->\nend')
    state = dict(phase='training', started_unix_s=time.time()-10, initial_transitions=50)
    job.snapshot(output, doc, state)
    progress = json.loads((output / 'progress.json').read_text())
    assert progress['status']['transitions'] == 100
    assert progress['metrics']['updates'] == 19
    assert progress['metrics']['normalized_action_norm'] == 0.3
    assert 4 < progress['transitions_per_second'] <= 5
    assert '100 / 50,000' in doc.read_text()
    assert doc.read_text().startswith('intro\n') and doc.read_text().endswith('\nend')


def test_paired_report_counts_regressions_and_uses_physical_success(tmp_path, monkeypatch):
    monkeypatch.setattr(job, 'ROOT', tmp_path)
    reference = tmp_path / 'outputs/astribot_online_rl/reference_round1'
    reference.mkdir(parents=True)
    baseline = {str(i): dict(baseline=i < 20) for i in range(40)}
    (reference / 'paired_comparison.json').write_text(json.dumps(dict(per_scene=baseline)))
    run = tmp_path / 'run'
    (run / 'eval').mkdir(parents=True)
    episodes = []
    for i in range(40):
        success = (5 <= i < 20) or i == 20
        episodes.append(dict(candidate_id=i, status='success' if success else 'timeout',
                             reason='physical_press_and_light' if success else 'timeout'))
    (run / 'eval/summary.json').write_text(json.dumps(dict(completed=40, episodes=episodes)))
    job.paired_comparison(run)
    report = json.loads((run / 'eval/paired_comparison.json').read_text())
    assert report['baseline_successes'] == 20 and report['round2_successes'] == 16
    assert report['improved'] == 1 and report['regressed'] == 5
    assert report['exact_mcnemar_p'] == 0.21875


def test_recovery_archives_checkpoint_and_failure_before_clearing_marker(tmp_path):
    output = tmp_path/'experiment'
    run = output/'run'
    run.mkdir(parents=True)
    (run/'learner.pt').write_bytes(b'learner checkpoint')
    (run/'replay.npz').write_bytes(b'replay checkpoint')
    (run/'infrastructure_error.json').write_text('{"error":"camera pose"}')
    archive = job.archive_attempt(output)
    manifest = json.loads((archive/'manifest.json').read_text())
    assert manifest['run/learner.pt'] == hashlib.sha256(b'learner checkpoint').hexdigest()
    assert (archive/'run/infrastructure_error.json').exists()
    assert not (run/'infrastructure_error.json').exists()
    assert (run/'learner.pt').read_bytes() == b'learner checkpoint'


def test_infrastructure_interruption_is_not_a_physical_timeout(tmp_path):
    from run_astribot_online_rl import completed_episode_reports
    for index, status in enumerate(('success', 'timeout', 'infrastructure_error', 'failed')):
        path=tmp_path/'episodes'/f'episode_{index:06d}'
        path.mkdir(parents=True)
        (path/'report.json').write_text(json.dumps(dict(status=status)))
    assert [row['status'] for row in completed_episode_reports(tmp_path)] == ['success','timeout','failed']
