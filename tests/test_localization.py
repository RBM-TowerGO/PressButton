"""Image localization and direction derivation are independent of robot execution."""
import ast
import copy
import json
import threading
from pathlib import Path

import pytest

from pressb.agent_loop import grounding
from pressb.agent_loop import localization as loc
from pressb.agent_loop.conversation import image_count, live_message
from test_conversation import jpeg, obs, response


def unknown():
    return dict(head_visible='unknown', wrist_visible='unknown', contact_finger='unknown',
                head_button_box=None, head_finger_tip=None, wrist_relation='unknown',
                lamp='unknown', lamp_view='unknown')


def located(**changes):
    data = dict(head_visible='yes', wrist_visible='no', contact_finger='head_image_right',
                head_button_box=[0.4, 0.4, 0.6, 0.6], head_finger_tip=[0.4, 0.8],
                wrist_relation='not_visible', lamp='off', lamp_view='head')
    data.update(changes)
    return data


def answer(data=None):
    return response('locate_alignment', unknown() if data is None else data)


def test_input_is_two_original_images_without_cases_state_actions_or_old_claims(jpeg):
    current = live_message(obs(0), jpeg, jpeg)
    data = json.loads(current['content'][0]['text'])
    data.update(state='SECRET_STATE', case='SECRET_CASE', reason='SECRET_REASON',
                command='SECRET_COMMAND', assessment='SECRET_OLD_CLAIM')
    current['content'][0]['text'] = json.dumps(data)
    current['content'].append(dict(type='text', text='SECRET_EXTRA'))
    before = copy.deepcopy(current)
    body = loc.build_request('press down button', current)
    assert current == before and 'SECRET' not in json.dumps(body)
    assert image_count(body['messages']) == 2
    pictures = [p for p in body['messages'][1]['content'] if p['type'] == 'image_url']
    assert pictures == [p for p in current['content'] if p['type'] == 'image_url']
    assert body['temperature'] == 0 and body['max_tokens'] == 512
    assert body['chat_template_kwargs'] == dict(enable_thinking=False)
    assert body['tools'][0]['function']['name'] == 'locate_alignment'
    fields = body['tools'][0]['function']['parameters']['properties']
    assert set(fields) == set(unknown()) and len(fields) == 8
    assert 'head_horizontal' not in fields and 'head_vertical' not in fields
    config = json.loads(body['messages'][1]['content'][0]['text'])
    assert config['selected_contact'] == 'head_image_right' == loc.DEFAULT_CONTACT
    assert 'x increases toward image right and y increases DOWN' in loc.PROMPT
    assert 'base or shaft being cropped' in loc.PROMPT
    assert 'visible tip' in loc.PROMPT


@pytest.mark.parametrize('tip,horizontal,vertical,error', [
    ([0.4, 0.8], 'right', 'above', [0.5, -1.5]),
    ([0.7, 0.3], 'left', 'below', [-1, 1]),
    ([0.5, 0.5], 'center', 'level', [0, 0]),
    ([0.45, 0.55], 'center', 'level', [0.25, -0.25]),
])
def test_directions_are_program_derived_from_button_minus_tip(tip,horizontal,vertical,error):
    result = loc.validate(answer(located(head_finger_tip=tip)), 'only-this-observation')
    assert result['format'] == 'pressb-grounding-v2'
    assert result['source'] == 'model_not_human'
    assert result['observation_id'] == 'only-this-observation'
    assert (result['head_horizontal'], result['head_vertical']) == (horizontal, vertical)
    assert result['geometry']['delta_image_norm'] == pytest.approx([0.5-tip[0], 0.5-tip[1]])
    assert result['geometry']['error_button_units'] == pytest.approx(error)
    assert 'not millimeters' in result['geometry']['semantics']


@pytest.mark.parametrize('error,magnitude,direction', [
    (0, 'small', 'center'), (0.25, 'small', 'center'), (0.2501, 'small', 'right'),
    (0.5, 'small', 'right'), (0.5001, 'medium', 'right'),
    (2, 'medium', 'right'), (2.0001, 'large', 'right'),
    (-0.5, 'small', 'left'), (-2.1, 'large', 'left'),
])
def test_error_magnitude_uses_button_size_and_explicit_thresholds(error,magnitude,direction):
    result = loc.validate(answer(located(head_button_box=[.45,.45,.55,.55],
                                        head_finger_tip=[.5-error*.1,.5])), 'obs')
    assert result['geometry']['magnitude'] == dict(horizontal=magnitude, vertical='small')
    assert result['head_horizontal'] == direction
    assert result['geometry']['classification'] == dict(center_fraction=.25,small_max=.5,medium_max=2)


def test_image_scale_does_not_change_button_relative_error():
    a = loc.validate(answer(located(head_button_box=[.4,.4,.6,.6],head_finger_tip=[.4,.8])), 'a')
    b = loc.validate(answer(located(head_button_box=[.2,.2,.3,.3],head_finger_tip=[.2,.4])), 'b')
    assert a['geometry']['error_button_units'] == pytest.approx(b['geometry']['error_button_units'])
    assert a['geometry']['magnitude'] == b['geometry']['magnitude']


def test_recorded_above_finger_geometry_cannot_be_reversed_to_below():
    # Approximate pixel locations from the failed round 6 HEAD snapshot (1280x720).
    data = located(head_button_box=[1135/1280,480/720,1185/1280,530/720],
                   head_finger_tip=[1075/1280,690/720])
    result = loc.validate(answer(data), 'recorded-round-6')
    assert result['head_vertical'] == 'above' and result['head_horizontal'] == 'right'
    assert result['geometry']['error_button_units'] == pytest.approx([1.7,-3.7])
    assert result['geometry']['magnitude'] == dict(horizontal='medium',vertical='large')


@pytest.mark.parametrize('data', [
    unknown(), located(head_button_box=None), located(head_finger_tip=None),
    located(contact_finger='unknown',head_finger_tip=None),
    located(head_visible='no',head_button_box=None,lamp='unknown',lamp_view='unknown'),
])
def test_missing_coordinates_preserve_unknown_without_inventing_offsets(data):
    result = loc.validate(answer(data), 'obs')
    assert result['head_horizontal'] == result['head_vertical'] == 'unknown'
    assert result['geometry']['delta_image_norm'] is None
    assert result['geometry']['error_button_units'] is None
    assert result['geometry']['magnitude'] == dict(horizontal='unknown',vertical='unknown')
    assert result['geometry']['head_button_box'] == data['head_button_box']
    assert result['geometry']['head_finger_tip'] == data['head_finger_tip']


@pytest.mark.parametrize('changes', [
    dict(head_button_box=[0,0,1]), dict(head_button_box=[0,0,1,1,1]),
    dict(head_button_box=[0,0,0,1]), dict(head_button_box=[1,0,0,1]),
    dict(head_button_box=[0,1,1,0]), dict(head_button_box=[0,.5,1,.5]),
    dict(head_button_box=[False,0,1,1]), dict(head_button_box=[0,0,float('nan'),1]),
    dict(head_button_box=[0,0,float('inf'),1]), dict(head_button_box=[0,0,1.01,1]),
    dict(head_button_box=[0,0,10**400,1]), dict(head_button_box=[0,0,5e-324,1]),
    dict(head_button_box=[-.01,0,1,1]), dict(head_button_box='[0,0,1,1]'),
    dict(head_finger_tip=[0]), dict(head_finger_tip=[0,0,1]), dict(head_finger_tip=[True,1]),
    dict(head_finger_tip=[0,float('-inf')]), dict(head_finger_tip=['0',1]),
    dict(head_finger_tip=[0,-.1]), dict(head_finger_tip=[1.1,1]),
    dict(head_visible='no'), dict(head_visible='unknown'),
    dict(contact_finger='unknown'), dict(contact_finger='robot_right'),
    dict(wrist_visible='no',wrist_relation='center'),
    dict(wrist_visible='unknown',wrist_relation='right'),
    dict(wrist_visible='yes',wrist_relation='not_visible'),
    dict(lamp='on',lamp_view='wrist'), dict(lamp='on',lamp_view='unknown'),
    dict(head_vertical='above'), dict(observation_id='invented'), dict(xyz_mm=[0,0,1]),
])
def test_malformed_locations_and_cross_field_contradictions_rejected(changes):
    with pytest.raises(ValueError):
        loc.validate(answer(located(**changes)), 'obs')


@pytest.mark.parametrize('bad', [None, [], {}, dict(choices=None), dict(choices=[None]),
    dict(choices=[dict(finish_reason='tool_calls',message=None)])])
def test_malformed_response_shapes_are_repairable_value_errors(bad):
    with pytest.raises(ValueError):
        loc.validate(bad, 'obs')


@pytest.mark.parametrize('mutate', [
    lambda r:r['choices'][0].update(finish_reason='length'),
    lambda r:r['choices'][0]['message'].update(content='move right'),
    lambda r:r['choices'][0]['message'].update(content=[]),
    lambda r:r['choices'][0]['message']['tool_calls'].append(copy.deepcopy(r['choices'][0]['message']['tool_calls'][0])),
    lambda r:r['choices'][0]['message']['tool_calls'][0].update(type='other'),
    lambda r:r['choices'][0]['message']['tool_calls'][0]['function'].update(name='inspect_alignment'),
    lambda r:r['choices'][0]['message']['tool_calls'][0]['function'].update(arguments={}),
    lambda r:r['choices'][0]['message']['tool_calls'][0]['function'].update(arguments='[]'),
])
def test_wrong_tool_multiple_calls_prose_and_non_json_arguments_rejected(mutate):
    result = answer(); mutate(result)
    with pytest.raises(ValueError):
        loc.validate(result, 'obs')


def test_duplicate_json_fields_are_not_silently_overwritten():
    result = answer()
    result['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = (
        '{"head_visible":"yes",'+json.dumps(unknown())[1:])
    with pytest.raises(ValueError,match='duplicate'):
        loc.validate(result, 'obs')


def test_default_contact_cannot_silently_switch_but_unknown_is_allowed():
    assert loc.validate(answer(),'obs',loc.DEFAULT_CONTACT)['contact_finger'] == 'unknown'
    left = answer(located(contact_finger='head_image_left'))
    with pytest.raises(ValueError,match='contact_reference_mismatch'):
        loc.validate(left, 'obs', loc.DEFAULT_CONTACT)
    assert loc.validate(left,'obs','head_image_left')['contact_finger'] == 'head_image_left'
    with pytest.raises(ValueError,match='selected_contact'):
        loc.validate(left, 'obs', 'right')
    with pytest.raises(ValueError,match='observation_id'):
        loc.validate(left, '')


def test_request_id_image_count_and_preference_are_bound_before_call(jpeg,tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(grounding,'request',lambda *a,**k:calls.append(a) or answer())
    body = loc.build_request('press', live_message(obs(0),jpeg,jpeg))
    with pytest.raises(ValueError,match='observation_mismatch'):
        loc.inspect('fake',body,'other',tmp_path)
    wrong = copy.deepcopy(body)
    wrong['messages'][1]['content'].pop()
    with pytest.raises(ValueError,match='dual_view'):
        loc.inspect('fake',wrong,'live-0',tmp_path)
    wrong = copy.deepcopy(body)
    wrong['messages'][1]['content'][0]['text']='{}'
    with pytest.raises(ValueError,match='preference'):
        loc.inspect('fake',wrong,'live-0',tmp_path)
    assert not calls
    with pytest.raises(ValueError,match='selected_contact'):
        loc.build_request('press',live_message(obs(0),jpeg,jpeg),'right')


def test_corrections_continue_until_valid_or_cancelled_with_archived_metrics(jpeg,tmp_path,monkeypatch):
    body = loc.build_request('press',live_message(obs(0),jpeg,jpeg))
    wrong = answer(located(contact_finger='head_image_left'))
    replies=iter([wrong,answer(located())]);calls=[]
    def request(base,route,payload,**kwargs):
        calls.append((payload,kwargs));return next(replies)
    monkeypatch.setattr(grounding,'request',request)
    result=loc.inspect('fake',body,'live-0',tmp_path/'ok')
    assert result['head_vertical']=='above' and len(calls)==2
    assert all(kwargs['timeout']==120 for _,kwargs in calls)
    assert 'contact_reference_mismatch' in calls[1][0]['messages'][-1]['content']
    metrics=json.loads((tmp_path/'ok/metrics.json').read_text())
    assert metrics['attempts']==2 and metrics['image_count']==2
    assert metrics['prompt_tokens']==2000 and metrics['completion_tokens']==60
    assert metrics['tool']=='locate_alignment'
    calls.clear()
    cancel=threading.Event()
    def invalid(*a,**k):
        calls.append(a)
        if len(calls)==4:cancel.set()
        return wrong
    monkeypatch.setattr(grounding,'request',invalid)
    with pytest.raises(InterruptedError,match='operator_stopped'):
        loc.inspect('fake',body,'live-0',tmp_path/'bad',cancel=cancel)
    assert len(calls)==4 and (tmp_path/'bad/error.json').exists()
    assert not (tmp_path/'bad/assessment.json').exists()


def test_request_failure_has_no_retry_or_action_fallback(jpeg,tmp_path,monkeypatch):
    def failed(*a,**k):raise TimeoutError('timeout')
    monkeypatch.setattr(grounding,'request',failed)
    with pytest.raises(TimeoutError):
        loc.inspect('fake',loc.build_request('press',live_message(obs(0),jpeg,jpeg)),'live-0',tmp_path)
    assert not (tmp_path/'repair-request.json').exists()
    assert json.loads((tmp_path/'metrics.json').read_text())['attempts']==1


@pytest.mark.parametrize('stopped_before',[False,True])
def test_stop_discards_late_localization_and_no_repair_occurs(jpeg,tmp_path,monkeypatch,stopped_before):
    cancel=threading.Event();calls=[]
    if stopped_before:cancel.set()
    def late(*a,**k):calls.append(a);cancel.set();return answer(located())
    monkeypatch.setattr(grounding,'request',late)
    with pytest.raises(InterruptedError):
        loc.inspect('fake',loc.build_request('press',live_message(obs(0),jpeg,jpeg)),
                    'live-0',tmp_path,cancel)
    assert len(calls)==(0 if stopped_before else 1)
    assert not (tmp_path/'assessment.json').exists()
    assert not (tmp_path/'repair-request.json').exists()


def test_module_cannot_import_robot_or_execution_modules():
    tree=ast.parse(Path(loc.__file__).read_text())
    imports=[node.module or '' for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
    imports += [alias.name for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names]
    assert not any(word in name for name in imports for word in
                   ('client','backend','executor','keyboard','sdk','conversation'))
