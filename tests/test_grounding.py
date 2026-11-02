"""Small visual questions, contradiction rejection and cancellation, without a robot."""
import ast
import copy
import json
import threading
from pathlib import Path

import pytest

from pressb.agent_loop import grounding as g
from pressb.agent_loop.conversation import live_message, image_count
from test_conversation import jpeg, obs, response


def unknown():
    return dict(head_visible='unknown', wrist_visible='unknown', contact_finger='unknown',
                head_horizontal='unknown', head_vertical='unknown', wrist_relation='unknown',
                lamp='unknown', lamp_view='unknown')


def shifted():
    return dict(head_visible='yes', wrist_visible='no', contact_finger='head_image_right',
                head_horizontal='right', head_vertical='level', wrist_relation='not_visible',
                lamp='off', lamp_view='head')


def assessment(values=None):
    return response('inspect_alignment', unknown() if values is None else values)


def withdrawal(value='unknown'):
    return response('confirm_withdrawal', dict(withdrawal=value))


def test_current_input_is_two_original_images_without_state_or_narrative(jpeg):
    current = live_message(obs(0), jpeg, jpeg)
    data = json.loads(current['content'][0]['text'])
    data.update(previous_reason='SECRET_REASON', previous_assessment='SECRET_ALIGNMENT',
                demonstration='SECRET_DEMO', proposed_command='SECRET_PROPOSAL')
    data['state']['measured_pose'] = 'SECRET_STATE'
    current['content'][0]['text'] = json.dumps(data)
    current['content'].append(dict(type='text', text='SECRET_EXTRA'))
    original = copy.deepcopy(current)
    body = g.build_request('press down button', current, 'head_image_right')
    assert current == original
    assert image_count(body['messages']) == 2
    assert 'SECRET' not in json.dumps(body)
    images = [part for part in body['messages'][1]['content'] if part['type']=='image_url']
    assert images == [part for part in current['content'] if part['type']=='image_url']
    assert body['chat_template_kwargs']['enable_thinking'] is False
    assert body['temperature'] == 0 and body['max_tokens'] == 512
    assert [tool['function']['name'] for tool in body['tools']] == ['inspect_alignment']
    fields = body['tools'][0]['function']['parameters']['properties']
    assert set(fields) == set(unknown()) and len(fields) == 8
    assert not any('bbox' in key or 'observation_id' in key for key in fields)
    assert 'selected_contact' in body['messages'][1]['content'][0]['text']


def test_default_contact_reference_is_head_image_right(jpeg):
    body=g.build_request('press',live_message(obs(0),jpeg,jpeg))
    preferences=json.loads(body['messages'][1]['content'][0]['text'])
    assert g.DEFAULT_CONTACT=='head_image_right'
    assert preferences['selected_contact']=='head_image_right'


def test_requested_contact_accepts_unknown_but_rejects_silent_finger_switch():
    assert g.validate(assessment(shifted()),'live-0',selected_contact='head_image_right')['contact_finger']=='head_image_right'
    assert g.validate(assessment(unknown()),'live-0',selected_contact='head_image_right')['contact_finger']=='unknown'
    left=dict(shifted(),contact_finger='head_image_left')
    with pytest.raises(ValueError,match='contact_reference_mismatch'):
        g.validate(assessment(left),'live-0',selected_contact='head_image_right')
    # The validator remains reusable for an explicitly selected left reference and old unbound callers.
    assert g.validate(assessment(left),'live-0',selected_contact='head_image_left')['contact_finger']=='head_image_left'
    assert g.validate(assessment(left),'live-0')['contact_finger']=='head_image_left'


def test_inspect_repairs_wrong_finger_using_request_reference(jpeg,tmp_path,monkeypatch):
    body=g.build_request('press',live_message(obs(0),jpeg,jpeg))
    replies=iter([assessment(dict(shifted(),contact_finger='head_image_left')),assessment(shifted())])
    calls=[]
    monkeypatch.setattr(g,'request',lambda *a,**k:calls.append(a[2]) or next(replies))
    result=g.inspect('fake',body,'live-0',tmp_path)
    assert result['contact_finger']=='head_image_right' and len(calls)==2
    assert 'contact_reference_mismatch' in calls[1]['messages'][-1]['content']
    assert json.loads((tmp_path/'metrics.json').read_text())['attempts']==2


def test_unknown_and_shifted_answers_are_legal_and_ids_are_program_bound():
    for data in (unknown(), shifted()):
        result = g.validate(assessment(data), 'current-only-id')
        assert result == dict(format=g.FORMAT, source='model_not_human', observation_id='current-only-id', **data)
    with pytest.raises(ValueError, match='extra'):
        g.validate(assessment(dict(unknown(), observation_id='invented')), 'current-only-id')


@pytest.mark.parametrize('changes', [
    dict(head_visible='partial'), dict(head_visible=True), dict(contact_finger='robot_right'),
    dict(head_visible='no', head_horizontal='left'),
    dict(contact_finger='unknown', head_vertical='above'),
    dict(wrist_visible='no', wrist_relation='center'),
    dict(wrist_visible='unknown', wrist_relation='right'),
    dict(wrist_visible='yes', wrist_relation='not_visible'),
    dict(lamp='on', lamp_view='wrist'),
    dict(lamp='on', lamp_view='unknown'),
    dict(xyz_mm=[0,-10,0]),
])
def test_structural_and_cross_field_contradictions_are_rejected(changes):
    data = shifted(); data.update(changes)
    with pytest.raises(ValueError):
        g.validate(assessment(data), 'live-0')


@pytest.mark.parametrize('mutate', [
    lambda r: r['choices'][0].update(finish_reason='length'),
    lambda r: r['choices'][0]['message'].update(content='move right'),
    lambda r: r['choices'][0]['message']['tool_calls'].append(copy.deepcopy(r['choices'][0]['message']['tool_calls'][0])),
    lambda r: r['choices'][0]['message']['tool_calls'][0]['function'].update(name='propose_move'),
])
def test_wrong_tool_multiple_calls_prose_and_truncation_rejected(mutate):
    result = assessment(); mutate(result)
    with pytest.raises(ValueError):
        g.validate(result, 'live-0')


def test_input_validation(jpeg):
    current = live_message(obs(0), jpeg, jpeg)
    with pytest.raises(ValueError, match='selected_contact'):
        g.build_request('press', current, 'left')
    current['content'].pop()
    with pytest.raises(ValueError, match='dual_view'):
        g.build_request('press', current)


def test_repeated_repairs_until_valid_or_cancel_without_action_fallback(jpeg,tmp_path,monkeypatch):
    body = g.build_request('press', live_message(obs(0),jpeg,jpeg))
    invalid = assessment(dict(shifted(), wrist_relation='right'))
    replies = iter([invalid, invalid, invalid, assessment(shifted())]); calls=[]
    def request(base, route, payload, **kwargs):
        live_metrics=json.loads((tmp_path/'ok/metrics.json').read_text())
        assert live_metrics['attempts'] == len(calls)+1
        assert live_metrics['validation_failures'] == len(calls)
        calls.append((payload, kwargs)); return next(replies)
    monkeypatch.setattr(g, 'request', request)
    result = g.inspect('fake',body,'live-0',tmp_path/'ok')
    assert result['wrist_visible']=='no' and len(calls)==4
    assert all(kwargs['timeout']==120 for _,kwargs in calls)
    repair = json.loads((tmp_path/'ok/repair-request.json').read_text())
    assert 'invisible_wrist_target' in repair['messages'][-1]['content']
    assert json.loads((tmp_path/'ok/metrics.json').read_text())['attempts']==4
    assert all(len(payload['messages']) == len(body['messages'])+1 for payload, _ in calls[1:])
    assert all((tmp_path/'ok'/f'repair-request-{n}.json').exists() for n in range(1,4))
    assert all((tmp_path/'ok'/f'validation-{n}.json').exists() for n in range(4))
    calls.clear()
    cancel = threading.Event()
    def invalid_until_cancel(*a, **k):
        calls.append(a)
        if len(calls) == 4:
            cancel.set()
        return invalid
    monkeypatch.setattr(g,'request',invalid_until_cancel)
    with pytest.raises(InterruptedError):
        g.inspect('fake',body,'live-0',tmp_path/'bad',cancel)
    assert len(calls)==4 and not (tmp_path/'bad/assessment.json').exists()
    assert (tmp_path/'bad/error.json').exists()


def test_request_failure_is_not_retried(jpeg,tmp_path,monkeypatch):
    def failed(*a,**k): raise TimeoutError('model timeout')
    monkeypatch.setattr(g,'request',failed)
    with pytest.raises(TimeoutError):
        g.inspect('fake',g.build_request('press',live_message(obs(0),jpeg,jpeg)),'live-0',tmp_path)
    assert not (tmp_path/'repair-request.json').exists()
    assert json.loads((tmp_path/'metrics.json').read_text())['attempts']==1


def test_caller_cannot_rebind_assessment_to_another_observation(jpeg,tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(g,'request',lambda *a,**k:calls.append(a) or assessment())
    with pytest.raises(ValueError,match='observation_mismatch'):
        g.inspect('fake',g.build_request('press',live_message(obs(0),jpeg,jpeg)),'other',tmp_path)
    assert not calls


def test_non_object_response_gets_one_format_repair(jpeg,tmp_path,monkeypatch):
    replies=iter([None, assessment()])
    monkeypatch.setattr(g,'request',lambda *a,**k:next(replies))
    result=g.inspect('fake',g.build_request('press',live_message(obs(0),jpeg,jpeg)),'live-0',tmp_path)
    assert result['observation_id']=='live-0'
    assert json.loads((tmp_path/'metrics.json').read_text())['attempts']==2


@pytest.mark.parametrize('already_stopped',[False,True])
def test_stop_prevents_late_assessment_and_repairs(jpeg,tmp_path,monkeypatch,already_stopped):
    cancel=threading.Event(); calls=[]
    if already_stopped: cancel.set()
    def late(*a,**k):
        calls.append(a);cancel.set();return assessment(dict(shifted(),wrist_relation='right'))
    monkeypatch.setattr(g,'request',late)
    with pytest.raises(InterruptedError):
        g.inspect('fake',g.build_request('press',live_message(obs(0),jpeg,jpeg)),'live-0',tmp_path,cancel)
    assert len(calls)==(0 if already_stopped else 1)
    assert not (tmp_path/'assessment.json').exists() and not (tmp_path/'repair-request.json').exists()


def test_withdrawal_uses_actual_pair_and_only_actual_command(jpeg):
    before,current=live_message(obs(0),jpeg,jpeg),live_message(obs(1),jpeg,jpeg)
    command=dict(type='move',xyz_mm=[-10.12345,0,0],reason='SECRET_REASON',next_action='SECRET_NEXT')
    body=g.build_withdrawal_request(before,current,command)
    assert image_count(body['messages'])==4 and 'SECRET' not in json.dumps(body)
    content=body['messages'][1]['content']
    frames=[json.loads(part['text']) for part in content if part['type']=='text' and part['text'].startswith('{')]
    assert [(frame['frame'],frame['observation_id']) for frame in frames[:2]]==[('BEFORE','live-0'),('CURRENT','live-1')]
    assert frames[-1]['actual_command']==dict(type='move',xyz_mm=[-10.12345,0,0])
    assert not frames[-1]['no_motion_command']
    assert [t['function']['name'] for t in body['tools']]==['confirm_withdrawal']
    for value in ('yes','no','unknown'):
        result=g.validate_withdrawal(withdrawal(value),'live-0','live-1')
        assert result['withdrawal']==value and result['source']=='model_not_human'
        assert result['evidence_observation_ids']==['live-0','live-1']
    with pytest.raises(ValueError,match='distinct'):
        g.build_withdrawal_request(before,before,command)


def test_withdrawal_repair_and_cancellation(jpeg,tmp_path,monkeypatch):
    before,current=live_message(obs(0),jpeg,jpeg),live_message(obs(1),jpeg,jpeg)
    responses=iter([response('confirm_withdrawal',dict(withdrawal='yes',observation_id='other')),withdrawal('unknown')])
    monkeypatch.setattr(g,'request',lambda *a,**k:next(responses))
    result=g.verify_withdrawal('fake',before,current,dict(type='move',xyz_mm=[-10,0,0]),tmp_path/'ok')
    assert result['withdrawal']=='unknown' and result['observation_id']=='live-1'
    assert json.loads((tmp_path/'ok/metrics.json').read_text())['attempts']==2
    cancel=threading.Event()
    def late(*a,**k):cancel.set();return withdrawal('yes')
    monkeypatch.setattr(g,'request',late)
    with pytest.raises(InterruptedError):
        g.verify_withdrawal('fake',before,current,None,tmp_path/'stop',cancel)
    assert not (tmp_path/'stop/assessment.json').exists()


def test_module_does_not_import_robot_or_execution_modules():
    tree=ast.parse(Path(g.__file__).read_text())
    imports=[node.module or '' for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
    imports += [alias.name for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names]
    assert not any(word in name for name in imports for word in ('client','backend','executor','keyboard','sdk','conversation'))


def test_activation_request_is_actual_four_image_pair_without_prior_judgments(jpeg):
    before,current=live_message(obs(0),jpeg,jpeg),live_message(obs(1),jpeg,jpeg)
    data=json.loads(before['content'][0]['text'])
    data['previous_assessment']='SECRET_LAMP_ON';before['content'][0]['text']=json.dumps(data)
    body=g.build_activation_request(before,current,dict(type='move',xyz_mm=[5,0,0],reason='SECRET_PRESS_SUCCESS'))
    assert image_count(body['messages'])==4 and 'SECRET' not in json.dumps(body)
    assert body['chat_template_kwargs']['enable_thinking'] is False
    assert body['tools'][0]['function']['name']=='confirm_activation'
    assert body['tools'][0]['function']['parameters']['required']==['light_on']
    assert 'single lit image' in body['messages'][0]['content']
    content=body['messages'][1]['content']
    frames=[json.loads(part['text']) for part in content if part['type']=='text' and part['text'].startswith('{')]
    assert [(frame['frame'],frame['observation_id']) for frame in frames[:2]]==[('BEFORE','live-0'),('CURRENT','live-1')]
    assert frames[-1]['actual_command']==dict(type='move',xyz_mm=[5,0,0])
    with pytest.raises(ValueError,match='distinct'):
        g.build_activation_request(current,current,None)
    with pytest.raises(ValueError,match='observation_message'):
        g.build_activation_request(None,current,None)


@pytest.mark.parametrize('value',['yes','no','unknown'])
def test_activation_ids_are_bound_by_program_and_uncertainty_is_valid(value):
    reply=response('confirm_activation',dict(light_on=value))
    result=g.validate_activation(reply,'live-before','live-current')
    assert result==dict(format=g.ACTIVATION_FORMAT,source='model_not_human',previous_observation_id='live-before',
        observation_id='live-current',evidence_observation_ids=['live-before','live-current'],light_on=value)
    with pytest.raises(ValueError,match='extra'):
        g.validate_activation(response('confirm_activation',dict(light_on=value,observation_id='foreign')),'live-before','live-current')
    with pytest.raises(ValueError,match='distinct'):
        g.validate_activation(reply,'live-current','live-current')


def test_activation_format_repair_and_metrics(jpeg,tmp_path,monkeypatch):
    before,current=live_message(obs(0),jpeg,jpeg),live_message(obs(1),jpeg,jpeg)
    replies=iter([response('confirm_activation',dict(light_on='on')),response('confirm_activation',dict(light_on='unknown'))])
    monkeypatch.setattr(g,'request',lambda *a,**k:next(replies))
    result=g.verify_activation('fake',before,current,dict(type='move',xyz_mm=[5,0,0]),tmp_path)
    assert result['light_on']=='unknown'
    metrics=json.loads((tmp_path/'metrics.json').read_text())
    assert metrics['attempts']==2 and metrics['image_count']==4 and metrics['tool']=='confirm_activation'
    assert metrics['previous_observation_id']=='live-0' and metrics['observation_id']=='live-1'
    assert (tmp_path/'assessment.json').exists() and (tmp_path/'repair-request.json').exists()


def test_activation_late_response_cannot_create_evidence(jpeg,tmp_path,monkeypatch):
    before,current=live_message(obs(0),jpeg,jpeg),live_message(obs(1),jpeg,jpeg)
    cancel=threading.Event()
    def late(*a,**k):cancel.set();return response('confirm_activation',dict(light_on='yes'))
    monkeypatch.setattr(g,'request',late)
    with pytest.raises(InterruptedError):
        g.verify_activation('fake',before,current,dict(type='move',xyz_mm=[5,0,0]),tmp_path,cancel)
    assert not (tmp_path/'assessment.json').exists()
    assert json.loads((tmp_path/'error.json').read_text())['stage']=='confirm_activation'
