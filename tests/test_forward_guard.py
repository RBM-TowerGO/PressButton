"""Current wrist-only authorization is independent, bound and fail-closed."""
import ast
import copy
import json
import threading
from pathlib import Path

import pytest

from pressb.agent_loop import grounding, forward_guard as guard
from pressb.agent_loop.conversation import live_message, image_count
from test_conversation import jpeg, obs, response


def values(value='yes'):
    return {name: value for name in guard.FIELDS}


def answer(fields=None):
    return response('inspect_forward_alignment', fields if fields is not None else values())


def assessed(fields=None, oid='now'):
    return guard.validate(answer(fields), oid)


def test_request_contains_only_current_original_wrist_and_goal(jpeg):
    current=live_message(obs(0), jpeg, jpeg)
    current['content'][0]['text']=json.dumps(dict(observation_id='now', depth='SECRET_DEPTH',
        state='SECRET_STATE', previous_reason='SECRET_REASON', prior_assessment='SECRET_ASSESSMENT'))
    images=[p for p in current['content'] if p['type']=='image_url']
    images[0]['image_url']['url']='HEAD_IMAGE_MUST_NOT_APPEAR'
    images[1]['image_url']['url']='CURRENT_WRIST_ORIGINAL'
    current['content'].append(dict(type='text',text='SECRET_DEMO'))
    original=copy.deepcopy(current)
    body=guard.build_request('press the down-call button',current)
    assert current==original
    assert image_count(body['messages'])==1
    serialized=json.dumps(body)
    assert 'CURRENT_WRIST_ORIGINAL' in serialized
    assert 'SECRET' not in serialized and 'HEAD_IMAGE_MUST_NOT_APPEAR' not in serialized
    assert 'press the down-call button' in serialized
    assert body['temperature']==0 and body['max_tokens']==512
    assert body['chat_template_kwargs']==dict(enable_thinking=False)
    assert set(body['tools'][0]['function']['parameters']['properties'])==set(guard.FIELDS)
    assert 'HORIZONTAL ONLY' in guard.PROMPT
    assert 'above or below' in guard.PROMPT and 'single-finger' in guard.PROMPT


def test_all_positive_current_wrist_fields_allow_forward_without_height_gate():
    evaluation=guard.evaluate(assessed(), 'now')
    assert evaluation['allowed'] is True and evaluation['reasons']==[]
    assert evaluation['policy']=='wrist_horizontal'
    assert evaluation['basis']==values()
    assert evaluation['observation_id']==evaluation['assessment_observation_id']=='now'
    assert 'height' not in guard.FIELDS
    assert 'height alignment' in evaluation['limitation']


@pytest.mark.parametrize('field', guard.FIELDS)
@pytest.mark.parametrize('value', ['no','unknown',None,True,[],1])
def test_nonpositive_missing_or_malformed_field_never_allows(field,value):
    data=assessed();data[field]=value
    assert guard.evaluate(data,'now')['allowed'] is False
    del data[field]
    assert guard.evaluate(data,'now')['allowed'] is False


@pytest.mark.parametrize('data', [None,{},dict(wrist_relation='center',head_vertical='level')])
def test_legacy_or_absent_assessment_never_authorizes(data):
    assert guard.evaluate(data,'now')['allowed'] is False


@pytest.mark.parametrize('changes', [dict(format='pressb-grounding-v1'),dict(source='human'),
                                    dict(observation_id='old')])
def test_wrong_format_source_or_observation_is_rejected(changes):
    data=assessed();data.update(changes)
    assert guard.evaluate(data,'now')['allowed'] is False


def test_no_latch_to_next_observation_or_mutation_of_assessment():
    data=assessed();original=copy.deepcopy(data)
    assert guard.evaluate(data,'now')['allowed']
    assert not guard.evaluate(data,'next')['allowed']
    assert data==original
    with pytest.raises(ValueError,match='invalid_forward_guard_policy'):
        guard.evaluate(data,'now',policy='dual_view')


@pytest.mark.parametrize('changes', [dict(button_visible='no'),dict(two_fingertips_visible='unknown')])
def test_between_yes_requires_both_visibility_fields(changes):
    data=values();data.update(changes)
    with pytest.raises(ValueError,match='between_requires_visible'):
        guard.validate(answer(data),'now')


def test_unknown_and_negative_results_are_legal_but_blocked():
    for value in ('no','unknown'):
        result=assessed(values(value))
        assert result['format']==guard.FORMAT and result['source']=='model_not_human'
        assert not guard.evaluate(result,'now')['allowed']


@pytest.mark.parametrize('mutate',[
    lambda r:r['choices'][0].update(finish_reason='length'),
    lambda r:r['choices'][0]['message'].update(content='go forward'),
    lambda r:r['choices'][0]['message']['tool_calls'][0]['function'].update(name='propose_move'),
    lambda r:r['choices'][0]['message']['tool_calls'].append(copy.deepcopy(r['choices'][0]['message']['tool_calls'][0])),
    lambda r:r['choices'][0]['message']['tool_calls'][0]['function'].update(arguments=json.dumps(dict(values(),observation_id='invented'))),
])
def test_invalid_tool_or_extra_fields_rejected(mutate):
    raw=answer();mutate(raw)
    with pytest.raises((ValueError,KeyError,TypeError)):
        guard.validate(raw,'now')


def test_inspection_repairs_once_and_archives_one_image(jpeg,tmp_path,monkeypatch):
    current=live_message(obs(0),jpeg,jpeg)
    body=guard.build_request('press',current)
    bad=answer(dict(values(),button_visible='unknown'))
    replies=iter([bad,answer()]);calls=[]
    monkeypatch.setattr(grounding,'request',lambda *a,**k:calls.append((a,k)) or next(replies))
    result=guard.inspect('fake',body,'live-0',tmp_path)
    assert guard.evaluate(result,'live-0')['allowed']
    assert len(calls)==2 and all(k['timeout']==120 for _,k in calls)
    assert (tmp_path/'repair-request.json').exists()
    assert json.loads((tmp_path/'metrics.json').read_text())['image_count']==1
    assert json.loads((tmp_path/'assessment.json').read_text())==result


def test_repeated_invalid_results_remain_cancellable_without_assessment(jpeg,tmp_path,monkeypatch):
    body=guard.build_request('press',live_message(obs(0),jpeg,jpeg));calls=[]
    bad=answer(dict(values(),two_fingertips_visible='no'))
    cancel=threading.Event()
    def invalid_until_cancel(*a,**k):
        calls.append(a)
        if len(calls)==4:cancel.set()
        return bad
    monkeypatch.setattr(grounding,'request',invalid_until_cancel)
    with pytest.raises(InterruptedError):
        guard.inspect('fake',body,'live-0',tmp_path,cancel)
    assert len(calls)==4 and not (tmp_path/'assessment.json').exists()
    assert (tmp_path/'repair-request-3.json').exists()
    assert (tmp_path/'error.json').exists()


@pytest.mark.parametrize('when',['before','after'])
def test_stop_and_late_response_cannot_authorize(jpeg,tmp_path,monkeypatch,when):
    body=guard.build_request('press',live_message(obs(0),jpeg,jpeg));cancel=threading.Event();calls=[]
    if when=='before':cancel.set()
    def request(*a,**k):
        calls.append(a);cancel.set();return answer()
    monkeypatch.setattr(grounding,'request',request)
    with pytest.raises(InterruptedError):guard.inspect('fake',body,'live-0',tmp_path,cancel)
    assert len(calls)==(0 if when=='before' else 1)
    assert not (tmp_path/'assessment.json').exists()


def test_request_error_is_not_retried_or_replaced(jpeg,tmp_path,monkeypatch):
    body=guard.build_request('press',live_message(obs(0),jpeg,jpeg));calls=[]
    def request(*a,**k):calls.append(a);raise TimeoutError('model timeout')
    monkeypatch.setattr(grounding,'request',request)
    with pytest.raises(TimeoutError):guard.inspect('fake',body,'live-0',tmp_path)
    assert len(calls)==1 and not (tmp_path/'assessment.json').exists()


def test_mismatched_request_is_rejected_before_any_network(jpeg,tmp_path,monkeypatch):
    body=guard.build_request('press',live_message(obs(0),jpeg,jpeg))
    monkeypatch.setattr(grounding,'request',lambda *a,**k:pytest.fail('must not request'))
    with pytest.raises(ValueError,match='request_observation_mismatch'):
        guard.inspect('fake',body,'other-id',tmp_path)
    body['messages'][1]['content'].append(copy.deepcopy(body['messages'][1]['content'][-1]))
    with pytest.raises(ValueError,match='request_observation_mismatch'):
        guard.inspect('fake',body,'live-0',tmp_path)


def test_module_has_no_robot_or_execution_dependency():
    source=Path(guard.__file__).read_text();tree=ast.parse(source)
    imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert not any(x in ('client','s1b_keyboard_control','s1b_agent') for x in imports)
    assert 'RobotClient' not in source and 'CommandExecutor' not in source
