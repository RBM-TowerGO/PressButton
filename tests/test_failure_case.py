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
    history = []
    args = ('press',obs,b'current-wrist',dict(schema=SESSION_SCHEMA),history,'prompt',b'current-head',demo,failure)
    body = build_request(*args)
    content = [item for message in body['messages'][1:] for item in message['content']][1:]
    images = [c for c in content if c['type']=='image_url']
    assert len(images) == 15
    assert json.loads(content[0]['text'])['case_review']['outcome'] == 'success'
    assert json.loads(content[0]['text'])['session']['final']['status'] == 'missing'
    fail_context = next(json.loads(c['text']) for c in content if c['type']=='text'
                        and c['text'].startswith('{') and 'HISTORICAL_FAILURE_NOT_LIVE' in c['text'])
    assert len(fail_context['rounds']) == 12 and fail_context['completed_commands'] == 11
    live = json.loads(content[-5]['text'])
    assert live['observation_id'] == 'NOW' and 'recent_history' not in live
    assert content[-4]['text'].startswith('LIVE HEAD') and content[-2]['text'].startswith('LIVE RIGHT WRIST')
    assert images[10] is failure.content[2]
    assert build_request(*args)['messages'][1]['content'][24] is content[23]


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
    assert s1b_agent_run.parse_args([]).demo is None
    assert s1b_agent_run.parse_args([]).failure_case is None
    assert s1b_agent_run.parse_args(['--decision-mode','separated']).failure_case.endswith('20261101_184824_dcb23038')
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
    from pressb.agent_loop import view_acquisition as view
    from test_view_acquisition import reply as view_reply
    captured = []
    def forbidden(*a, **k):
        raise AssertionError('Replay must never contact a robot')
    monkeypatch.setattr(s1b_agent, 'RobotClient', forbidden)
    def propose(base, body, *a, **k):
        captured.append(body)
        return dict(name='request_observation',arguments=dict(reason='test only'))
    monkeypatch.setattr(s1b_agent, 'propose', propose)
    monkeypatch.setattr('pressb.agent_loop.conversation.tokenize', lambda *a:1000)
    monkeypatch.setattr(view, 'inspect', lambda base, body, oid, *a, **k: view.validate(view_reply(oid=oid), oid))
    prompt = tmp_path/'prompt.txt';prompt.write_text('test prompt')
    out = tmp_path/'replay'
    args = SimpleNamespace(replay=str(case/'agent/observations/obs11'), session=None,
        output=str(out),prompt=str(prompt),failure_case=str(case),model_base='http://unused/v1',
        thor_port=18000,thor_host='unused',goal='press',max_rounds=20,decision_mode='direct')
    s1b_agent.run(args)
    assert len([v for v in [x for m in captured[0]['messages'][1:] for x in m['content']] if v['type']=='image_url']) == 5
    task = json.loads((out/'task.json').read_text())
    assert task['initial_image_count'] == 5 and task['failure_analysis_version'] == 'test.v1'
    assert (out/'failure_case/round_0011.jpg').is_file()
    assert json.loads((out/'result.json').read_text())['status'] == 'replay_only'


def test_new_conversation_uses_explicit_current_observation(case):
    from pressb.agent_loop.model_context import FORMAT
    rd = case/'agent/rounds/0004'
    body = json.loads((rd/'request.json').read_text())
    body['messages'].insert(0, dict(role='user', content=[dict(type='text',text=json.dumps(
        dict(context_type='LIVE_OBSERVATION', observation_id='obs0')))]))
    atomic_json(rd/'request.json',body)
    atomic_json(rd/'decision_context.json',dict(format=FORMAT,observation_id='obs4',compression_generation=1))
    assert FailureCase(case).summary['rounds'][4]['before']['observation_id'] == 'obs4'
    atomic_json(rd/'decision_context.json',dict(format=FORMAT,observation_id='obs0'))
    with pytest.raises(ValueError,match='failure_decision_observation_mismatch'):
        FailureCase(case)


def test_separated_perception_interrupt_has_no_action_request(case):
    rd=case/'agent/rounds/0011'
    (rd/'request.json').unlink()
    atomic_json(rd/'decision_context.json',dict(format='pressb-conversation-v2',observation_id='obs11',
        previous_observation_id='obs10',decision_mode='separated',stage='interrupted'))
    before=snapshot(case)
    failure=FailureCase(case)
    assert failure.summary['completed_commands']==11 and len(failure.summary['rounds'])==12
    row=failure.summary['rounds'][-1]
    assert row['status']=='perception_interrupted' and row['proposal'] is None and not row['executed']
    assert row['before']['observation_id']=='obs11' and snapshot(case)==before


def test_separated_assessment_must_match_decision(case):
    rd=case/'agent/rounds/0004'
    atomic_json(rd/'decision_context.json',dict(format='pressb-conversation-v2',observation_id='obs4',previous_observation_id='obs3'))
    atomic_json(rd/'perception/assessment.json',dict(format='pressb-perception-v1',observation_id='obs4',previous_observation_id='obs3'))
    assert FailureCase(case).summary['rounds'][4]['perception']['observation_id']=='obs4'
    atomic_json(rd/'perception/assessment.json',dict(format='pressb-perception-v1',observation_id='obs0',previous_observation_id='obs3'))
    with pytest.raises(ValueError,match='failure_perception_observation_mismatch'):FailureCase(case)


def test_grounded_perception_interrupt_has_no_action_request(case):
    rd = case/'agent/rounds/0011'
    (rd/'request.json').unlink()
    atomic_json(rd/'decision_context.json', dict(format='pressb-conversation-v3', observation_id='obs11',
                previous_observation_id='obs10', decision_mode='grounded', stage='interrupted',
                memory_mode='bounded_current_state'))
    before = snapshot(case)
    failure = FailureCase(case)
    row = failure.summary['rounds'][-1]
    assert row['status'] == 'perception_interrupted' and not row['executed'] and row['proposal'] is None
    assert row['before']['observation_id'] == 'obs11' and row['after'] is None
    assert failure.summary['completed_commands'] == 11 and snapshot(case) == before
    atomic_json(rd/'proposal.json', dict(name='request_observation', arguments=dict(reason='invalid archive')))
    with pytest.raises(ValueError, match='failure_execution_without_action_request'):
        FailureCase(case)


@pytest.mark.parametrize('inspection_format', ['pressb-grounding-v1', 'pressb-grounding-v2'])
def test_grounded_pair_evidence_stays_model_judgment(case, inspection_format):
    rd = case/'agent/rounds/0004'
    atomic_json(rd/'decision_context.json', dict(format='pressb-conversation-v3', observation_id='obs4',
                previous_observation_id='obs3', decision_mode='grounded', stage='action_request'))
    assessment = dict(format=inspection_format, observation_id='obs4', source='model_not_human',
                      head_visible='yes', wrist_visible='no', contact_finger='head_image_right',
                      head_horizontal='right', head_vertical='level', wrist_relation='not_visible',
                      lamp='off', lamp_view='head')
    if inspection_format == 'pressb-grounding-v2':
        assessment['geometry'] = dict(head_button_box=[.5,.3,.6,.4], head_finger_tip=[.3,.35],
            delta_image_norm=[.25,0], error_button_units=[2.5,0],
            magnitude=dict(horizontal='large', vertical='small'))
    atomic_json(rd/'perception/assessment.json', assessment)
    progress = dict(status='unchanged_or_unresolved', current_observation_id='obs4', before_observation_id='obs3',
                    source='categorical_model_comparison_not_metric')
    atomic_json(rd/'alignment_progress.json', progress)
    activation = dict(observation_id='obs4', previous_observation_id='obs3', light_on='unknown', source='model_not_human')
    atomic_json(rd/'activation/assessment.json', activation)
    withdrawal = dict(observation_id='obs4', previous_observation_id='obs3', withdrawal='unknown', source='model_not_human')
    atomic_json(rd/'withdrawal/assessment.json', withdrawal)
    basis = dict(observation_id='obs4', expected_effect='reduce_horizontal_offset', validated=True)
    atomic_json(rd/'decision_basis.json', basis)
    memory = dict(selected_contact='head_image_right', evidence=dict(light_on=None,withdrawal=None))
    atomic_json(rd/'control_memory.json', memory)
    before = snapshot(case)
    row = FailureCase(case).summary['rounds'][4]
    assert row['perception'] == assessment and row['alignment_progress'] == progress
    assert row['activation'] == activation
    assert row['withdrawal'] == withdrawal and row['decision_basis'] == basis and row['control_memory'] == memory
    assert row['executed'] and row['command']['xyz_mm'] == [10,0,0]
    assert snapshot(case) == before


@pytest.mark.parametrize('file,record,expected', [
    ('perception/assessment.json', dict(format='pressb-grounding-v1', observation_id='wrong'), 'perception'),
    ('perception/assessment.json', dict(format='pressb-grounding-v2', observation_id='wrong'), 'perception'),
    ('perception/assessment.json', dict(format='pressb-grounding-v99', observation_id='obs4'), 'perception'),
    ('alignment_progress.json', dict(current_observation_id='obs4', before_observation_id='wrong'), 'alignment_progress'),
    ('withdrawal/assessment.json', dict(observation_id='wrong', previous_observation_id='obs3'), 'withdrawal'),
    ('activation/assessment.json', dict(observation_id='obs4', previous_observation_id='wrong'), 'activation'),
    ('decision_basis.json', dict(observation_id='wrong'), 'decision_basis'),
    ('alignment_progress.json', dict(current_observation_id='obs4', observation_id='wrong', before_observation_id='obs3'), 'alignment_progress'),
])
def test_grounded_evidence_cannot_reference_another_observation(case, file, record, expected):
    rd = case/'agent/rounds/0004'
    atomic_json(rd/'decision_context.json', dict(format='pressb-conversation-v3', observation_id='obs4',
                previous_observation_id='obs3', decision_mode='grounded', stage='action_request'))
    atomic_json(rd/file, record)
    with pytest.raises(ValueError, match='failure_'+expected+'_observation_mismatch'):
        FailureCase(case)


def test_grounded_withdrawal_failure_is_not_an_execution(case):
    rd = case/'agent/rounds/0011'
    (rd/'request.json').unlink()
    atomic_json(rd/'decision_context.json', dict(format='pressb-conversation-v3', observation_id='obs11',
                previous_observation_id='obs10', decision_mode='grounded', stage='interrupted'))
    atomic_json(rd/'withdrawal/request.json', dict(messages=[]))
    atomic_json(rd/'withdrawal/error.json', dict(error='operator_stopped', stage='withdrawal_verification'))
    row = FailureCase(case).summary['rounds'][-1]
    assert row['status'] == 'withdrawal_verification_interrupted' and not row['executed']
    assert row['proposal'] is None and row['after'] is None


def test_grounded_activation_failure_is_not_an_execution(case):
    rd = case/'agent/rounds/0011'
    (rd/'request.json').unlink()
    (rd/'error.json').unlink()
    atomic_json(rd/'decision_context.json', dict(format='pressb-conversation-v3', observation_id='obs11',
                previous_observation_id='obs10', decision_mode='grounded', stage='interrupted'))
    atomic_json(rd/'activation/request.json', dict(messages=[]))
    atomic_json(rd/'activation/error.json', dict(error='operator_stopped', stage='activation_verification'))
    row = FailureCase(case).summary['rounds'][-1]
    assert row['status'] == 'activation_verification_interrupted' and not row['executed']
    assert row['proposal'] is None and row['after'] is None
    assert row['error']['stage'] == 'activation_verification'


def control_plan(oid='obs11', previous='obs10'):
    return dict(format='pressb-control-plan-v1', observation_id=oid,
        source='program_advice_from_model_judgments_not_human_confirmation', stage='align',
        focus='vertical', step_hint_mm=20, geometry_progress=dict(current_observation_id=oid,
            previous_observation_id=previous, comparable=False, reason='geometry_missing_or_invalid'),
        head_target_info=dict(last_absolute_target_deg=[0,30], current_measured_head_deg=[0,25]))


def test_grounded_control_plan_keeps_provenance_and_is_not_executed(case):
    rd = case/'agent/rounds/0011'
    (rd/'request.json').unlink()
    atomic_json(rd/'decision_context.json', dict(format='pressb-conversation-v3', observation_id='obs11',
                previous_observation_id='obs10', decision_mode='grounded', stage='interrupted'))
    plan = control_plan()
    atomic_json(rd/'control_plan.json', plan)
    before = snapshot(case)
    failure = FailureCase(case)
    row = failure.summary['rounds'][-1]
    assert row['control_plan'] == plan
    assert row['proposal'] is None and row['after'] is None and not row['executed']
    assert 'command' not in row and failure.summary['completed_commands'] == 11
    assert snapshot(case) == before


@pytest.mark.parametrize(('field', 'value', 'error'), [
    ('observation_id', 'other', 'control_plan_observation_mismatch'),
    ('observation_id', None, 'control_plan_observation_mismatch'),
    ('source', 'human_confirmed', 'control_plan_model_provenance_required'),
    ('format', 'unknown', 'control_plan_model_provenance_required'),
    ('geometry_progress', dict(current_observation_id='obs11', previous_observation_id='other'),
     'control_plan_geometry_progress_observation_mismatch'),
    ('geometry_progress', dict(current_observation_id='other', previous_observation_id='obs10'),
     'control_plan_geometry_progress_observation_mismatch'),
])
def test_grounded_control_plan_rejects_other_observation_or_false_provenance(case, field, value, error):
    rd = case/'agent/rounds/0011'
    atomic_json(rd/'decision_context.json', dict(format='pressb-conversation-v3', observation_id='obs11',
                previous_observation_id='obs10', decision_mode='grounded', stage='interrupted'))
    atomic_json(rd/'control_plan.json', dict(control_plan(), **{field: value}))
    with pytest.raises(ValueError, match='failure_'+error):
        FailureCase(case)
