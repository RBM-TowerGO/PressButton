import copy
import io
import json
import threading
from pathlib import Path

from PIL import Image
import pytest

from pressb.agent_loop import conversation as module
from pressb.agent_loop.conversation import Conversation, image_count
from pressb.agent_loop.client import propose
from pressb.agent_loop.common import SESSION_SCHEMA
from pressb.agent_loop.model_context import state, depth


@pytest.fixture
def jpeg():
    stream = io.BytesIO()
    Image.new('RGB', (640, 360)).save(stream, 'JPEG')
    return stream.getvalue()


def obs(n):
    return dict(observation_id=f'live-{n}', state=dict(pose_m_xyzw=[.123456, 0, 1, 0, 0, 0, 1]),
                selection={}, depth=dict(median_m=None, valid_pixels=0))


def response(name='propose_move', arguments=None, cid='call-0'):
    return dict(choices=[dict(finish_reason='tool_calls', message=dict(role='assistant', content=None,
        tool_calls=[dict(id=cid, type='function', function=dict(name=name,
            arguments=json.dumps(arguments or dict(xyz_mm=[1.123456,0,0], reason='test'))))]))],
        usage=dict(prompt_tokens=1000, completion_tokens=30))


def advance(c, jpeg, n, name='propose_move'):
    args = dict(reason='target visible; adjust')
    if name == 'propose_move':
        args['xyz_mm'] = [1.123456, 0, 0]
    c.accept(response(name, args, f'call-{n}'))
    p = dict(name=name, arguments=args)
    result = None if name == 'request_observation' else dict(status='completed', request=dict(
        type='move', xyz_mm=args['xyz_mm'], command_id=f'cmd-{n}'), motion=dict(action_completed=True),
        capture=dict(observation_id=f'live-{n}'))
    c.append_result(p, result, obs(n), jpeg, jpeg)
    return p


def make(jpeg, budget=65536):
    return Conversation('press', obs(0), jpeg, jpeg, dict(schema=SESSION_SCHEMA), 'prompt', budget=budget)


def summary_response(**changes):
    data = dict(progress='alignment uncertain', uncertainties='No human confirmation',
                initial_light=dict(observation_id='live-0', assessment='initial state unclear'),
                light_on=dict(observation_id='live-1', assessment='model claimed light; unverified'),
                withdrawal=dict(observation_id=None, assessment='unknown'))
    data.update(changes)
    return response('summarize_context', data)


def test_append_prefix_pairing_and_observe(jpeg):
    c = make(jpeg)
    first = copy.deepcopy(c.body())
    advance(c, jpeg, 1)
    second = c.body()
    assert second['messages'][:len(first['messages'])] == first['messages']
    assert [m['role'] for m in second['messages']] == ['system','user','user','assistant','tool','user']
    assert second['messages'][3]['tool_calls'][0]['id'] == second['messages'][4]['tool_call_id'] == 'call-1'
    assert json.loads(second['messages'][3]['tool_calls'][0]['function']['arguments'])['xyz_mm'][0] == 1.123456
    advance(c, jpeg, 2, 'request_observation')
    assert c.turns[-1]['ledger']['result']['status'] == 'observed'
    assert image_count(c.messages()) == 6
    assert c.body()['max_tokens'] == 512
    assert not any('recent_history' in str(m) for m in c.messages())


def test_model_views_are_lossy_not_sources():
    original = dict(pose_m_xyzw=[.123456789,0,1,0,0,0,1], command_pose_m_xyzw=[.2,0,1,0,0,0,1],
        joints=dict(astribot_gripper_right=dict(position=[.123456], torque=[99]), astribot_arm_right=dict(torque=[99])))
    saved = copy.deepcopy(original)
    compact = state(original)
    assert compact['measured_pose']['xyz_mm'][0] == 123.5
    assert compact['command_pose']['xyz_mm'][0] == 200
    assert compact['measured_gripper_sdk_native'] == [.1235]
    assert compact['gripper_command'] is None and 'torque' not in str(compact)
    assert original == saved
    assert depth(dict(median_m=None))['median_mm'] is None


def test_budget_tokenize_only_start_and_near_threshold(jpeg, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(module, 'tokenize', lambda base, body: calls.append(body) or 1000)
    c = make(jpeg)
    c.prepare('fake', tmp_path/'0')
    advance(c, jpeg, 1)
    c.prepare('fake', tmp_path/'1')
    assert len(calls) == 1
    meta = json.loads((tmp_path/'1/decision_context.json').read_text())
    assert meta['observation_id'] == 'live-1' and meta['image_count'] == 4


def test_compaction_retains_ledger_recent_five_and_pinned_images(jpeg, monkeypatch, tmp_path):
    c = make(jpeg, budget=20000)
    for n in range(1, 9):
        advance(c, jpeg, n)
    old = copy.deepcopy(c.turns)
    counts = iter([21000, 19000])
    monkeypatch.setattr(module, 'tokenize', lambda *a: next(counts))
    monkeypatch.setattr(module, 'request', lambda *a, **k: summary_response())
    c.estimated_tokens = 21000
    c.prepare('fake', tmp_path)
    assert c.generation == 1 and c.start == 3 and c.turns == old
    data = json.loads((tmp_path/'compaction/result.json').read_text())
    assert data['pinned_observation_ids'] == ['live-0', 'live-1']
    assert data['earlier_action_ledger'] == [t['ledger'] for t in old[:3]]
    assert len([m for m in c.messages() if m['role'] == 'assistant']) == 5
    assert image_count(c.messages()) == 16  # 6 recent boundaries + 2 pinned boundaries
    assert 'MODEL_MEMORY_NOT_HUMAN_VERIFICATION' in c.memory[0]['content'][0]['text']
    assert 'EARLIER_OBSERVATION_THIS_RUN_NOT_CURRENT' in c.memory[1]['content'][0]['text']


def test_summary_rejects_foreign_ids_until_corrected_without_retry_limit(jpeg, monkeypatch, tmp_path):
    c = make(jpeg)
    for n in range(1, 8):
        advance(c, jpeg, n)
    bad = summary_response(light_on=dict(observation_id='DEMO-step-10', assessment='lit'))
    calls = []
    replies = iter([bad, bad, bad, summary_response()])
    monkeypatch.setattr(module, 'request', lambda *a, **k: calls.append(a[2]) or next(replies))
    c.compact('fake', tmp_path)
    assert len(calls) == 4 and c.generation == 1
    assert (tmp_path/'repair-request.json').is_file()
    assert (tmp_path/'repair-request-3.json').is_file()
    assert not (tmp_path/'error.json').exists()
    assert 'evidence_must_reference_this_run' in (tmp_path/'validation-2.json').read_text()
    assert all(len(body['messages']) == len(calls[0]['messages'])+1 for body in calls[1:])
    assert json.loads((tmp_path/'metrics.json').read_text())['validation_failures'] == 3


def test_compaction_cancel_discards_late_summary(jpeg, monkeypatch, tmp_path):
    c = make(jpeg)
    for n in range(1, 8):
        advance(c, jpeg, n)
    cancel = threading.Event()
    def late(*a, **k):
        cancel.set()
        return summary_response()
    monkeypatch.setattr(module, 'request', late)
    with pytest.raises(InterruptedError):
        c.compact('fake', tmp_path, cancel)
    assert c.generation == 0 and not c.memory


def test_budget_never_silently_drops_fixed_cases(jpeg, monkeypatch, tmp_path):
    c = make(jpeg, budget=100)
    monkeypatch.setattr(module, 'tokenize', lambda *a: 101)
    before = copy.deepcopy(c.body())
    with pytest.raises(ValueError, match='context_budget_too_small'):
        c.prepare('fake', tmp_path)
    assert c.body() == before


def test_real_propose_repair_preserves_raw_call_and_cancellation(jpeg, monkeypatch, tmp_path, approved_forward_view):
    from pressb.agent_loop import client
    c = make(jpeg)
    invalid = response(); invalid['choices'][0]['message']['tool_calls'][0].pop('id')
    good = response(cid='server-generated-id')
    replies = iter([invalid, good])
    monkeypatch.setattr(client, 'request', lambda *a, **k: next(replies))
    p = propose('fake', c.body(), obs(0), dict(schema=SESSION_SCHEMA), tmp_path/'ok', conversation=c)
    assert c.pending == good['choices'][0]['message']
    assert p['arguments']['xyz_mm'][0] == 1.123456
    assert (tmp_path/'ok/repair-request.json').exists()
    c = make(jpeg);cancel = threading.Event()
    def late(*a, **k):
        cancel.set()
        return good
    monkeypatch.setattr(client, 'request', late)
    with pytest.raises(InterruptedError):
        propose('fake', c.body(), obs(0), dict(schema=SESSION_SCHEMA), tmp_path/'cancel', cancel=cancel, conversation=c)
    assert c.pending is None


def test_default_cases_compact_and_complete(jpeg):
    from pressb.agent_loop.demonstration import Demonstration
    from pressb.agent_loop.failure_case import FailureCase
    root = Path(__file__).resolve().parents[1]
    success = root/'outputs/keyboard_sessions/20261101_182424_352b5047'
    failed = root/'outputs/agent_web/20261101_184824_dcb23038'
    if not success.exists() or not failed.exists():
        pytest.skip('local recorded cases not deployed')
    d, f = Demonstration(success), FailureCase(failed)
    old = sum(len(json.dumps(x.summary, ensure_ascii=False)) for x in (d, f))
    new = sum(len(x.content[0]['text']) for x in (d, f))
    assert new/old <= .3
    assert len(d.images) == 11 and len(f.images) == 3
    assert len(d.model_summary['operations']) == 11
    assert d.model_summary['operations'][-1]['status'] == 'interrupted'
    assert d.model_summary['reviewed_activation_depth_references'][0]['median_mm'] == 131.3
    assert f.model_summary['completed_commands'] == 14 and len(f.model_summary['rounds']) == 15
    assert f.model_summary['rounds'][-1]['status'] == 'request_interrupted'
    assert 'source_sha256' not in f.content[0]['text']
    assert 'source_sha256' in f.summary
    c=Conversation('press',obs(0),jpeg,jpeg,dict(schema=SESSION_SCHEMA),'prompt',d,f,decision_mode='separated')
    assert image_count(c.body()['messages'])==16
    assert len([part for part in c.fixed[1]['content'] if part['type']=='image_url'])==14
    assert c.format=='pressb-conversation-v2'


def test_compressed_context_still_too_large_stops(jpeg, monkeypatch, tmp_path):
    c = make(jpeg, budget=20000)
    for n in range(1, 8):
        advance(c, jpeg, n)
    c.estimated_tokens = 21000
    monkeypatch.setattr(module, 'tokenize', lambda *a:21000)
    monkeypatch.setattr(module, 'request', lambda *a, **k:summary_response())
    with pytest.raises(ValueError, match='context_budget_exceeded_after_compaction'):
        c.prepare('fake', tmp_path)
    assert (tmp_path/'request.json').is_file()
    assert json.loads((tmp_path/'context_metrics.json').read_text())['compression_status'] == 'failed'


def test_summary_repair_then_success_and_order(jpeg, monkeypatch, tmp_path):
    c = make(jpeg)
    for n in range(1, 8):
        advance(c, jpeg, n)
    bad = summary_response(withdrawal=dict(observation_id='live-0',assessment='retreated'))
    replies = iter([bad,summary_response()])
    monkeypatch.setattr(module, 'request', lambda *a, **k:next(replies))
    c.compact('fake',tmp_path)
    assert c.generation == 1 and (tmp_path/'repair-request.json').is_file()


@pytest.mark.parametrize('ending', ['cancel', 'network_error'])
def test_compaction_repeated_invalid_remains_stoppable_and_does_not_retry_network(jpeg, monkeypatch, tmp_path, ending):
    c = make(jpeg)
    for n in range(1, 8):
        advance(c, jpeg, n)
    original_turns = copy.deepcopy(c.turns)
    cancel = threading.Event(); calls = []
    invalid = summary_response(light_on=dict(observation_id='foreign', assessment='invalid reference'))
    def request(*args, **kwargs):
        calls.append(args[2])
        assert json.loads((tmp_path/'metrics.json').read_text())['attempts'] == len(calls)
        if len(calls) == 4:
            if ending == 'network_error':
                raise TimeoutError('network error must end compaction')
            cancel.set()
        return invalid
    monkeypatch.setattr(module, 'request', request)
    with pytest.raises(InterruptedError if ending == 'cancel' else TimeoutError):
        c.compact('fake', tmp_path, cancel)
    assert len(calls) == 4 and c.generation == 0 and not c.memory
    assert c.turns == original_turns and not (tmp_path/'result.json').exists()
    assert (tmp_path/'repair-request-3.json').exists()
    metrics = json.loads((tmp_path/'metrics.json').read_text())
    assert metrics['validation_failures'] == 3
    assert metrics['status'] == ('interrupted' if ending == 'cancel' else 'failed')
