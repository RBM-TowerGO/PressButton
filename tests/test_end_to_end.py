import io
import json
from pathlib import Path

from PIL import Image
import pytest

from pressb.agent_loop import client, end_to_end
from pressb.agent_loop import conversation as conversation_module
from pressb.agent_loop import calibrated_tip
from pressb.agent_loop.common import SESSION_SCHEMA
from pressb.agent_loop.conversation import image_count
from pressb.agent_loop.end_to_end import EndToEndConversation

import s1b_agent
import s1b_agent_run

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT/'docs/astribot/planar_control/fingertip_calibration/right_wrist_fingertip_point.json'


@pytest.fixture
def jpeg():
    stream = io.BytesIO()
    Image.new('RGB', (640, 360)).save(stream, 'JPEG')
    return stream.getvalue()


@pytest.fixture(autouse=True)
def no_calibration():
    calibrated_tip.deactivate()
    yield
    calibrated_tip.deactivate()


def obs(n):
    return dict(observation_id=f'live-{n}', state=dict(pose_m_xyzw=[.4, -.23, 1.2, 0, 0, .707, .707]),
                selection={}, depth=dict(median_m=.2043, valid_pixels=25))


def response(name, arguments, cid='call-0'):
    return dict(choices=[dict(finish_reason='tool_calls', message=dict(role='assistant', content=None,
        tool_calls=[dict(id=cid, type='function', function=dict(name=name, arguments=json.dumps(arguments)))]))],
        usage=dict(prompt_tokens=1000, completion_tokens=30))


def make(jpeg, prompt='prompt'):
    return EndToEndConversation('press', obs(0), jpeg, jpeg, dict(schema=SESSION_SCHEMA), prompt)


def test_large_forward_move_is_one_request_and_needs_no_guard(jpeg, monkeypatch, tmp_path):
    calls = []
    def fake_request(base, route, body, **kwargs):
        calls.append(route)
        return response('propose_move', dict(xyz_mm=[100, 0, 0], reason='button on aim pixel; press'))
    monkeypatch.setattr(client, 'request', fake_request)
    monkeypatch.setattr(conversation_module, 'request', fake_request)
    c = make(jpeg)
    body = c.prepare_decision('fake', tmp_path/'0')
    proposal = client.propose('fake', body, obs(0), dict(schema=SESSION_SCHEMA), tmp_path/'0', conversation=c)
    assert proposal == dict(name='propose_move', arguments=dict(xyz_mm=[100, 0, 0], reason='button on aim pixel; press'))
    assert calls == ['/chat/completions']
    assert c.forward_guard_decision is None
    assert not (tmp_path/'0/forward_guard').exists()
    meta = json.loads((tmp_path/'0/decision_context.json').read_text())
    assert meta['decision_mode'] == 'end_to_end' and meta['forward_guard_policy'] == 'none'


def test_robot_client_executes_forward_without_permission(monkeypatch, tmp_path):
    sent = []
    monkeypatch.setattr(client, 'request', lambda base, route, body=None, **k: (
        dict(schema=SESSION_SCHEMA, session_id='s') if route == '/v2/session' else sent.append(body) or dict(task_id='t')))
    robot = client.RobotClient('http://robot', tmp_path, forward_guard_policy='none')
    monkeypatch.setattr(robot, 'wait', lambda receipt: dict(capture=dict(observation_id='live-1')))
    monkeypatch.setattr(robot, 'download', lambda oid: (dict(observation_id=oid), tmp_path))
    robot.execute(dict(name='propose_move', arguments=dict(xyz_mm=[100, 0, 0], reason='r')), obs(0), None)
    assert sent[0]['type'] == 'move' and sent[0]['xyz_mm'] == [100, 0, 0]


def test_tools_do_not_claim_a_wrist_check_and_history_has_no_extra_stages(jpeg):
    c = make(jpeg)
    names = [t['function']['name'] for t in c.body()['tools']]
    assert names == [t['function']['name'] for t in client.TOOLS]
    move = next(t for t in c.body()['tools'] if t['function']['name'] == 'propose_move')
    assert 'check' in move['function']['description'] and 'wrist' not in move['function']['description'].lower()
    # The shared tool list keeps its original text for other modes.
    assert 'independent CURRENT wrist check' in client.TOOLS[0]['function']['description']
    c.accept(response('propose_move', dict(xyz_mm=[0, 5, 0], reason='r'), 'call-1'))
    c.append_result(dict(name='propose_move', arguments=dict(xyz_mm=[0, 5, 0], reason='r')),
                    dict(status='completed', request=dict(type='move', xyz_mm=[0, 5, 0], command_id='c1'),
                         motion=dict(action_completed=True), capture=dict(observation_id='live-1')),
                    obs(1), jpeg, jpeg)
    messages = c.body()['messages']
    assert image_count(messages) == 4
    assert [m['role'] for m in messages] == ['system', 'user', 'user', 'assistant', 'tool', 'user']


def test_calibrated_pixel_is_given_as_reference(jpeg):
    calibrated_tip.activate(CALIBRATION, 'open')
    c = make(jpeg)
    fixed = [json.loads(p['text']) for p in c.fixed[1]['content'] if p.get('type') == 'text']
    aim = next(v for v in fixed if v.get('context_type') == 'CALIBRATED_MERGED_END_WRIST_PIXEL')
    assert aim['pixel'] == [343, 248]
    assert fixed[0]['input_format'] == end_to_end.FORMAT


def test_cli_selects_end_to_end_prompt_and_policy():
    args = s1b_agent.parse_args(['--goal', 'g', '--output', 'o', '--decision-mode', 'end_to_end'])
    assert Path(args.prompt) == s1b_agent.END_TO_END_PROMPT and Path(args.prompt).exists()
    assert s1b_agent.conversation_class('end_to_end') is EndToEndConversation
    assert s1b_agent.forward_guard_policy('end_to_end') == 'none'
    assert s1b_agent.forward_guard_policy('localized') == 'wrist_tip_overlap'
    launch = s1b_agent_run.parse_args(['--decision-mode', 'end_to_end'])
    assert launch.decision_mode == 'end_to_end' and Path(launch.prompt) == s1b_agent_run.END_TO_END_PROMPT
    assert launch.failure_case is None
