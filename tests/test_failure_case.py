"""Failure context preserves action boundaries, provenance and live-image separation."""
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from pressb.agent_loop.common import SESSION_SCHEMA, atomic_json, sha256
from pressb.agent_loop.client import build_request
from pressb.agent_loop.demonstration import Demonstration
from pressb.agent_loop.failure_case import FailureCase
import s1b_agent
import s1b_agent_run


@pytest.fixture
def case(tmp_path):
    root = tmp_path/'failed'
    run = root/'agent'
    atomic_json(root/'case_review.json', dict(schema='pressb-case-review-v1', outcome='failure',
        analysis_version='test.v1', source_session_id='failed-session', completed_commands=11,
        selected_rounds=[5,7,11], findings=[dict(analysis='Target lost; height uncertain.')]))
    atomic_json(run/'session.json', dict(schema=SESSION_SCHEMA, session_id='failed-session'))
    atomic_json(run/'result.json', dict(status='operator_stopped', completed_motion_commands=11))
    for i in range(12):
        oid = f'obs{i}'
        folder = run/'observations'/oid
        image = io.BytesIO()
        Image.new('RGB', (64,36), (i*20, 0, 0)).save(image, 'JPEG')
        data = image.getvalue()
        atomic_json(folder/'manifest.json', dict(observation_id=oid, session_id='failed-session',
            state=dict(pose_m_xyzw=[i,0,1,0,0,0,1], command_pose_m_xyzw=[i+.1,0,1,0,0,0,1]),
            depth=dict(median_m=None if i == 0 else .2), selection=dict(center_px=[32,20]),
            artifacts={v+'.jpg':dict(sha256=sha256(data)) for v in ('head','right')}))
        for view in ('head','right'):
            (folder/(view+'.jpg')).write_bytes(data)
        rd = run/'rounds'/f'{i:04d}'
        context = dict(context_type='LIVE_OBSERVATION', observation_id=oid)
        atomic_json(rd/'request.json', dict(messages=[dict(role='user', content=[
            dict(type='text', text=json.dumps(context))])]))
        if i == 11:
            atomic_json(rd/'error.json', dict(stage='model_request', error_type='RemoteDisconnected'))
            continue
        command = dict(type='move', xyz_mm=[10,0,0], observation_id=oid, session_id='failed-session')
        atomic_json(rd/'proposal.json', dict(name='propose_move', arguments=dict(xyz_mm=[10,0,0],reason=f'old reason {i}')))
        atomic_json(rd/'execution.json', dict(decision_observation_id=oid, execution_observation_id=oid))
        atomic_json(rd/'result.json', dict(status='completed', request=command, task_id=str(i),
            motion=dict(completion_basis='command_interpolation', state_trace=['large trace'],
                        final_state=dict(pose_m_xyzw=[i,0,1,0,0,0,1])),
            capture=dict(status='complete', observation_id=f'obs{i+1}'), error=None))
    return root


def snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def test_complete_failure_timeline_with_interrupted_round_and_source_unchanged(case, tmp_path):
    before = snapshot(case)
    failure = FailureCase(case)
    rows = failure.summary['rounds']
    assert len(rows) == 12 and sum(row['executed'] for row in rows) == 11
    assert rows[-1]['status'] == 'request_interrupted' and rows[-1]['proposal'] is None
    assert rows[-1]['after'] is None
    assert rows[4]['before']['observation_id'] == 'obs4'
    assert rows[4]['after']['observation_id'] == 'obs5'
    assert rows[4]['proposal']['arguments']['reason'] == 'old reason 4'
    assert rows[4]['before']['state']['pose_m_xyzw'][0] == 4
    assert rows[4]['before']['state']['command_pose_m_xyzw'][0] == 4.1
    assert rows[0]['before']['depth']['median_m'] is None
    assert 'state_trace' not in rows[4]['result']['motion']
    assert [name for name, _ in failure.images] == ['round_0005.jpg','round_0007.jpg','round_0011.jpg']
    failure.save(tmp_path/'new_run')
    assert snapshot(case) == before
    saved = json.loads((tmp_path/'new_run/failure_case/reference.json').read_text())
    assert saved['review']['analysis_version'] == 'test.v1'
    assert 'case_review.json' in saved['source_sha256']


def test_fifteen_images_and_complete_text_keep_live_separate(case, tmp_path):
    demo_root = tmp_path/'success'
    atomic_json(demo_root/'session.json', dict(schema='pressb-keyboard-record-v1',
        settings=dict(frame='chassis',units='mm'), final=dict(status='missing')))
    atomic_json(demo_root/'case_review.json', dict(schema='pressb-case-review-v1',outcome='success',outcome_source='operator_feedback'))
    data = (case/'agent/observations/obs0/head.jpg').read_bytes()
    for n in range(10):
        op = demo_root/'operations'/f'{n+1:06d}'
        atomic_json(op/'operation.json', dict(operation_id=f'{n+1:06d}',kind='move',command=dict(xyz_mm=[1,0,0]),
            before=dict(status='complete'),after=dict(status='complete')))
        for phase in ('before','after'):
            atomic_json(op/phase/'metadata.json', dict(depth={}))
            for view in ('head','right'):
                (op/phase/(view+'.jpg')).write_bytes(data)
    demo, failure = Demonstration(demo_root), FailureCase(case)
    obs = dict(observation_id='NOW',state={},selection={},depth={})
    history = [dict(round=n) for n in range(8)]
    args = ('press',obs,b'current-wrist',dict(schema=SESSION_SCHEMA),history,'prompt',b'current-head',demo,failure)
    body = build_request(*args)
    content = body['messages'][1]['content']
    images = [c for c in content if c['type']=='image_url']
    assert len(images) == 15
    assert json.loads(content[0]['text'])['case_review']['outcome'] == 'success'
    assert json.loads(content[0]['text'])['session']['final']['status'] == 'missing'
    fail_context = next(json.loads(c['text']) for c in content if c['type']=='text'
                        and c['text'].startswith('{') and 'HISTORICAL_FAILURE_NOT_LIVE' in c['text'])
    assert len(fail_context['rounds']) == 12 and fail_context['completed_commands'] == 11
    live = json.loads(content[-5]['text'])
    assert live['observation_id'] == 'NOW' and live['recent_history'] == history[-5:]
    assert content[-4]['text'].startswith('LIVE HEAD') and content[-2]['text'].startswith('LIVE RIGHT WRIST')
    assert images[10] is failure.content[2]
    assert build_request(*args)['messages'][1]['content'][23] is content[23]


@pytest.mark.parametrize('relative', ['case_review.json', 'agent/observations/obs4/right.jpg',
    'agent/observations/obs11/head.jpg', 'agent/rounds/0004/result.json'])
def test_missing_required_failure_evidence_is_an_error(case, relative):
    (case/relative).unlink()
    with pytest.raises(ValueError, match='failure_case_missing_artifact'):
        FailureCase(case)


def test_mismatched_before_observation_is_rejected(case):
    atomic_json(case/'agent/rounds/0004/execution.json', dict(decision_observation_id='obs3', execution_observation_id='obs4'))
    with pytest.raises(ValueError, match='failure_action_observation_mismatch'):
        FailureCase(case)


def test_corrupt_image_is_rejected(case):
    (case/'agent/observations/obs4/right.jpg').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='failure_image_hash_mismatch'):
        FailureCase(case)


def test_launcher_defaults_and_explicit_optout():
    assert s1b_agent_run.parse_args([]).demo.endswith('20261101_182424_352b5047')
    assert s1b_agent_run.parse_args([]).failure_case.endswith('20261101_184824_dcb23038')
    assert s1b_agent_run.parse_args(['--no-failure-case']).failure_case is None
    assert s1b_agent_run.parse_args(['--failure-case','custom']).failure_case == 'custom'
    with pytest.raises(SystemExit):
        s1b_agent_run.parse_args(['--failure-case','custom','--no-failure-case'])


def test_missing_failure_stops_launcher_before_service_start(case, monkeypatch):
    monkeypatch.setattr(s1b_agent_run.sys, 'stdin', SimpleNamespace(isatty=lambda:True))
    monkeypatch.setattr(s1b_agent_run, 'check_port_available', lambda port:None)
    def forbidden(*a, **k):
        raise AssertionError('No process may start with missing failure evidence')
    monkeypatch.setattr(s1b_agent_run.subprocess, 'Popen', forbidden)
    args = s1b_agent_run.parse_args(['--demo',str(case/'agent'),'--failure-case',str(case/'missing')])
    with pytest.raises(ValueError, match='failure_case_missing_artifact'):
        s1b_agent_run.launch(args)


def test_replay_loads_failure_and_logs_context_without_robot(case, tmp_path, monkeypatch):
    captured = []
    def forbidden(*a, **k):
        raise AssertionError('Replay must never contact a robot')
    monkeypatch.setattr(s1b_agent, 'RobotClient', forbidden)
    def propose(base, body, *a, **k):
        captured.append(body)
        return dict(name='request_observation',arguments=dict(reason='test only'))
    monkeypatch.setattr(s1b_agent, 'propose', propose)
    prompt = tmp_path/'prompt.txt';prompt.write_text('test prompt')
    out = tmp_path/'replay'
    args = SimpleNamespace(replay=str(case/'agent/observations/obs11'), session=None,
        output=str(out),prompt=str(prompt),failure_case=str(case),model_base='http://unused/v1',
        thor_port=18000,thor_host='unused',goal='press',max_rounds=20)
    s1b_agent.run(args)
    assert len([v for v in captured[0]['messages'][1]['content'] if v['type']=='image_url']) == 5
    task = json.loads((out/'task.json').read_text())
    assert task['expected_images_per_request'] == 5 and task['failure_analysis_version'] == 'test.v1'
    assert (out/'failure_case/round_0011.jpg').is_file()
    assert json.loads((out/'result.json').read_text())['status'] == 'replay_only'
