"""Perception isolation, temporal grounding, cancellation and policy integration."""
import copy
import json
import threading
from types import SimpleNamespace

import pytest

from pressb.agent_loop import perception as p
from pressb.agent_loop.common import SESSION_SCHEMA
from pressb.agent_loop.conversation import Conversation, live_message, image_count
from pressb.agent_loop.client import propose
from test_conversation import jpeg, obs, response, advance, summary_response


def unknown(oid='live-0', previous=None):
    result = dict(observation_id=oid, previous_observation_id=previous,
                  alignment_change='unknown', light_on='unknown', withdrawal='unknown',
                  evidence_observation_ids=[], note='Insufficient visible evidence.')
    for view in ('head', 'wrist'):
        result.update({view+'_visibility': 'unknown', view+'_bbox': None,
                       view+'_relation': 'unknown', view+'_contact': 'unknown', view+'_light': 'unknown'})
    return result


def inspect_response(body):
    frames = [json.loads(x['text']) for x in body['messages'][1]['content']
              if x.get('type') == 'text' and x['text'].startswith('{')]
    current = next(x['observation_id'] for x in frames if x.get('frame') == 'CURRENT')
    previous = next((x['observation_id'] for x in frames if x.get('frame') == 'BEFORE'), None)
    return response('inspect_scene', unknown(current, previous))


def separated(jpeg, budget=65536):
    return Conversation('press', obs(0), jpeg, jpeg, dict(schema=SESSION_SCHEMA), 'prompt',
                        budget=budget, decision_mode='separated')


def test_inputs_two_then_four_images_and_no_narrative(jpeg):
    first = live_message(obs(0), jpeg, jpeg)
    second = live_message(obs(1), jpeg, jpeg)
    body = p.build_request('press', first)
    assert image_count(body['messages']) == 2
    feedback = dict(status='completed', command=dict(type='move', xyz_mm=[0,-10,0], reason='SECRET_REASON'),
                    proposal='SECRET_PROPOSAL', previous_assessment='SECRET_ASSESSMENT')
    body = p.build_request('press', second, first, feedback)
    assert image_count(body['messages']) == 4
    assert 'SECRET' not in json.dumps(body)
    assert [t['function']['name'] for t in body['tools']] == ['inspect_scene']
    assert body['max_tokens'] == 512 and body['chat_template_kwargs']['enable_thinking'] is False
    body = p.build_request('press', second, first, dict(status='observed'))
    assert json.loads(body['messages'][1]['content'][-1]['text'])['between_observations']['no_motion_command']
    replay=dict(status='not_executed_replay',command=dict(type='move',xyz_mm=[999,0,0]),
                replay=dict(recorded_actual_command=dict(type='move',xyz_mm=[5,0,0]),note='SECRET_REASON'))
    replay_body=p.build_request('press',second,first,replay)
    actual=json.loads(replay_body['messages'][1]['content'][-1]['text'])['between_observations']
    assert actual['actual_command']['xyz_mm']==[5,0,0] and 'SECRET_REASON' not in json.dumps(replay_body)


@pytest.mark.parametrize('change', [
    dict(observation_id='foreign'), dict(previous_observation_id='foreign'),
    dict(head_bbox=[0,0,1001,50]), dict(head_bbox=[50,0,10,50]),
    dict(head_bbox=[True,0,10,50]), dict(head_visibility='not_visible',head_bbox=[1,1,5,5]),
    dict(evidence_observation_ids=['foreign']), dict(head_visibility='maybe'),
    dict(light_on='yes'), dict(withdrawal='yes'), dict(alignment_change='improved'),
    dict(next_axis='-Y'),
])
def test_invalid_perception_rejected(change):
    data = unknown();data.update(change)
    with pytest.raises((ValueError, TypeError)):
        p.validate(response('inspect_scene',data),'live-0',None)


def test_change_requires_actual_pair_and_unknown_is_legal():
    assert p.validate(response('inspect_scene',unknown()),'live-0',None)['source'].startswith('model_')
    data = unknown('live-1','live-0');data.update(light_on='yes',evidence_observation_ids=['live-0','live-1'])
    assert p.validate(response('inspect_scene',data),'live-1','live-0')['light_on'] == 'yes'
    data['evidence_observation_ids'] = ['live-1']
    with pytest.raises(ValueError, match='both_evidence'):
        p.validate(response('inspect_scene',data),'live-1','live-0')


def test_repeated_repairs_until_valid_or_cancel_and_request_failure(tmp_path,jpeg,monkeypatch):
    body = p.build_request('press', live_message(obs(0),jpeg,jpeg))
    bad = response('inspect_scene',dict(unknown(),head_bbox=[-1,0,10,10]))
    replies = iter([bad, bad, bad, response('inspect_scene',unknown())])
    monkeypatch.setattr(p,'request',lambda *a,**k:next(replies))
    p.inspect('fake',body,'live-0',None,tmp_path/'ok')
    assert json.loads((tmp_path/'ok/metrics.json').read_text())['attempts'] == 4
    assert (tmp_path/'ok/repair-request-3.json').exists()
    assert len(json.loads((tmp_path/'ok/repair-request-3.json').read_text())['messages']) == len(body['messages'])+1
    assert (tmp_path/'ok/repair-request.json').exists()
    calls = []
    cancel = threading.Event()
    def invalid_until_cancel(*a, **k):
        calls.append(a)
        if len(calls) == 4:
            cancel.set()
        return bad
    monkeypatch.setattr(p,'request',invalid_until_cancel)
    with pytest.raises(InterruptedError):
        p.inspect('fake',body,'live-0',None,tmp_path/'bad',cancel)
    assert len(calls) == 4 and not (tmp_path/'bad/assessment.json').exists()
    def failed(*a,**k):raise TimeoutError('timeout')
    monkeypatch.setattr(p,'request',failed)
    with pytest.raises(TimeoutError):p.inspect('fake',body,'live-0',None,tmp_path/'network')
    assert not (tmp_path/'network/repair-request.json').exists()


def test_repair_identifies_missing_observation_field(tmp_path,jpeg,monkeypatch):
    body=p.build_request('press',live_message(obs(0),jpeg,jpeg))
    bad=unknown();bad.pop('observation_id')
    calls=[]
    def request(base,route,payload,**kwargs):
        calls.append(payload)
        return response('inspect_scene',bad if len(calls)==1 else unknown())
    monkeypatch.setattr(p,'request',request)
    p.inspect('fake',body,'live-0',None,tmp_path)
    repair=calls[1]['messages'][-1]['content']
    assert "missing=['observation_id']" in repair and 'observation_id="live-0"' in repair


def test_independent_assessment_precedes_policy_and_preserves_prefix(jpeg,tmp_path,monkeypatch,approved_forward_view):
    c = separated(jpeg)
    with pytest.raises(ValueError,match='current_perception'):
        c.prepare('fake',tmp_path/'early')
    events=[]
    monkeypatch.setattr(p,'request',lambda base,route,body,**k:events.append('perception') or inspect_response(body))
    monkeypatch.setattr('pressb.agent_loop.conversation.tokenize',lambda *a:1000)
    first = c.prepare_decision('fake',tmp_path/'0')
    saved = copy.deepcopy(first)
    assert json.loads(first['messages'][-1]['content'][0]['text'])['context_type'].startswith('INDEPENDENT_PERCEPTION')
    from pressb.agent_loop import client
    monkeypatch.setattr(client,'request',lambda *a,**k:events.append('action') or response())
    proposal = propose('fake',first,obs(0),dict(schema=SESSION_SCHEMA),tmp_path/'0',conversation=c)
    assert events == ['perception','action']
    c.append_result(proposal,None,obs(1),jpeg,jpeg)
    estimate=c.estimated_tokens
    second=c.prepare_decision('fake',tmp_path/'1')
    assert c.estimated_tokens > estimate
    assert second['messages'][:len(saved['messages'])] == saved['messages']
    assert image_count(second['messages']) == 4
    assert json.loads((tmp_path/'1/perception/metrics.json').read_text())['image_count'] == 4
    assert json.loads((tmp_path/'1/decision_context.json').read_text())['format']=='pressb-conversation-v2'
    with pytest.raises(ValueError,match='already_committed'):c.attach_assessment(c.assessments['live-1'])
    with pytest.raises(ValueError,match='decision_observation'):c.check_observation('live-0')


def test_late_perception_never_commits_or_runs_policy(jpeg,tmp_path,monkeypatch):
    cancel=threading.Event();c=separated(jpeg)
    def late(base,route,body,**k):
        cancel.set();return inspect_response(body)
    monkeypatch.setattr(p,'request',late)
    with pytest.raises(InterruptedError):c.prepare_decision('fake',tmp_path,cancel)
    assert not c.assessments and c.pending is None
    assert not (tmp_path/'perception/assessment.json').exists()
    assert not (tmp_path/'request.json').exists()
    assert json.loads((tmp_path/'decision_context.json').read_text())['stage']=='interrupted'


def test_old_assessment_does_not_override_approved_wrist_guard_or_change_proposal(jpeg,tmp_path,monkeypatch,approved_forward_view):
    c=separated(jpeg)
    assessment=unknown();assessment.update(wrist_visibility='not_visible',head_visibility='not_visible')
    c.attach_assessment(p.validate(response('inspect_scene',assessment),'live-0',None))
    from pressb.agent_loop import client
    action=response('propose_move',dict(xyz_mm=[2.12345,-3.45678,0],reason='exploratory motion; alignment unknown'))
    monkeypatch.setattr(client,'request',lambda *a,**k:action)
    proposal=propose('fake',c.body(),obs(0),dict(schema=SESSION_SCHEMA),tmp_path,conversation=c)
    assert proposal['arguments']['xyz_mm']==[2.12345,-3.45678,0]


def test_budget_failure_does_not_rerun_inspection(jpeg,tmp_path,monkeypatch):
    c=separated(jpeg,budget=10);calls=[]
    monkeypatch.setattr(p,'request',lambda base,route,body,**k:calls.append(body) or inspect_response(body))
    monkeypatch.setattr('pressb.agent_loop.conversation.tokenize',lambda *a:100)
    with pytest.raises(ValueError,match='context_budget_too_small'):c.prepare_decision('fake',tmp_path)
    assert len(calls)==1 and 'live-0' in c.assessments
    with pytest.raises(ValueError,match='context_budget_too_small'):c.prepare_decision('fake',tmp_path/'retry')
    assert len(calls)==1 and c.pending is None


def test_compaction_keeps_recent_assessments_and_earlier_ledger(jpeg,tmp_path,monkeypatch):
    c=separated(jpeg,budget=20000)
    for n in range(8):
        c.attach_assessment(p.validate(response('inspect_scene',unknown(f'live-{n}',None if n==0 else f'live-{n-1}')),
                                      f'live-{n}',None if n==0 else f'live-{n-1}'))
        advance(c,jpeg,n+1)
    c.attach_assessment(p.validate(response('inspect_scene',unknown('live-8','live-7')),'live-8','live-7'))
    monkeypatch.setattr('pressb.agent_loop.conversation.request',lambda *a,**k:summary_response())
    old=copy.deepcopy(c.turns)
    c.compact('fake',tmp_path)
    assert c.start==3 and c.turns==old
    assert sum(m['role']=='assistant' for m in c.messages())==5
    serialized=json.dumps(c.messages())
    assert 'perception_summary' in serialized and 'EARLIER_PERCEPTION_MODEL_ASSESSMENT_NOT_CURRENT' in serialized
    assert 'INDEPENDENT_PERCEPTION_MODEL_ASSESSMENT' in serialized


@pytest.mark.parametrize('cancel_inspection',[False,True])
def test_real_runner_fake_robot_two_stages_or_cancel(tmp_path,monkeypatch,cancel_inspection,approved_forward_view):
    import s1b_agent
    from test_agent_loop import FakeBackend,FakeCamera
    from pressb.agent_loop.service import Service,make_server
    service=Service(FakeBackend(),FakeCamera(),tmp_path/'robot')
    server=make_server(service,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    controls=[];original=s1b_agent.StopControl
    def factory():
        control=original();controls.append(control);return control
    monkeypatch.setattr(s1b_agent,'StopControl',factory)
    calls=[]
    def inspect_call(base,route,body,**k):
        calls.append('perception')
        if cancel_inspection:controls[0].cancel.set()
        return inspect_response(body)
    monkeypatch.setattr(p,'request',inspect_call)
    monkeypatch.setattr('pressb.agent_loop.conversation.tokenize',lambda *a:1000)
    from pressb.agent_loop import client
    monkeypatch.setattr(client,'request',lambda *a,**k:calls.append('action') or response())
    # RobotClient uses client.request too; keep its HTTP requests real for the fake local server.
    from pressb.agent_loop.common import request as http_request
    monkeypatch.setattr(client,'request',lambda base,route,*a,**k:
        (calls.append('action') or response()) if route=='/chat/completions' else http_request(base,route,*a,**k))
    prompt=tmp_path/'prompt';prompt.write_text('test')
    args=SimpleNamespace(output=str(tmp_path/'client'),prompt=str(prompt),model_base='http://unused/v1',
        replay=None,thor_port=18000,thor_host='unused',robot_base=f'http://127.0.0.1:{server.server_port}',max_rounds=1,goal='test',decision_mode='separated')
    try:
        if cancel_inspection:
            with pytest.raises(InterruptedError):s1b_agent.run(args)
            assert calls==['perception'] and not service.backend.calls
        else:
            s1b_agent.run(args)
            assert calls==['perception','action'] and len(service.backend.calls)==1
        assert service.backend.stopped
    finally:
        server.shutdown();server.server_close();service.close();thread.join()
