"""Default localization/control integration, using read-only fake model replies."""
import copy
import json
import threading

import pytest

from pressb.agent_loop import client, grounding, short_loop as loop
from pressb.agent_loop.common import SESSION_SCHEMA
from pressb.agent_loop.conversation import image_count
from test_conversation import jpeg, obs, response


SESSION = dict(schema=SESSION_SCHEMA, speed_mm_s=20, head_speed_deg_s=20)


def observation(n, head=(0, 25)):
    value = obs(n)
    value['state'].update(head_deg=list(head), gripper_command='o')
    return value


def make(jpeg, budget=None):
    return loop.ShortConversation('Press the down-call button and withdraw',
        observation(0), jpeg, jpeg, SESSION, 'Robot coordinate and tool instructions', budget=budget)


def coordinates(unknown=False):
    if unknown:
        return dict(head_visible='unknown', wrist_visible='unknown', contact_finger='unknown',
            head_button_box=None, head_finger_tip=None, wrist_relation='unknown',
            lamp='unknown', lamp_view='unknown')
    return dict(head_visible='yes', wrist_visible='yes', contact_finger='head_image_right',
        head_button_box=[.45,.2,.55,.3], head_finger_tip=[.5,.8],
        wrist_relation='center', lamp='off', lamp_view='head')


def model_localizer(monkeypatch, rows, guard_between='yes', stop_on=None, cancel=None):
    """Exercise the actual localization and forward-guard request/validation paths."""
    rows = iter(rows)
    calls = []
    def request(base, route, body, **kwargs):
        assert route == '/chat/completions'
        name = body['tools'][0]['function']['name']
        calls.append(copy.deepcopy(body))
        if name == 'locate_alignment':
            reply = response(name, next(rows))
        elif name == 'inspect_forward_alignment':
            reply = response(name, dict(button_visible='yes', two_fingertips_visible='yes',
                                        button_between_fingers=guard_between))
        else:
            raise AssertionError('Unexpected read-only tool: '+name)
        if name == stop_on:
            cancel.set()
        return reply
    monkeypatch.setattr(grounding, 'request', request)
    monkeypatch.setattr(loop, 'tokenize', lambda *a: 1000)
    return calls


def action(c, name='propose_move', values=None, effect='reduce_vertical_error', cid='action'):
    arguments = copy.deepcopy(values if values is not None else dict(
        xyz_mm=[0,0,20], reason='HEAD selected finger needs vertical correction.'))
    arguments.update(visual_basis={key:c.assessments[c.current_id][key] for key in loop.BASIS_FIELDS},
                     expected_effect=effect)
    return response(name, arguments, cid)


def accept(c, directory, name, values, effect, cid='action'):
    reply = action(c, name, values, effect, cid)
    proposal = c.parse_response(reply, dict(observation_id=c.current_id), SESSION, directory)
    c.accept(reply)
    return proposal


def completed(n, command):
    return dict(status='completed', request=dict(command, command_id=f'cmd-{n}'),
        motion=dict(action_completed=True, completion_basis='command_interpolation_finished',elapsed_s=.3),
        capture=dict(observation_id=f'live-{n}'))


def test_default_loop_localizes_two_images_and_exports_above_large_twenty_advice(jpeg,tmp_path,monkeypatch):
    c=make(jpeg)
    calls=model_localizer(monkeypatch,[coordinates()])
    body=c.prepare_decision('fake',tmp_path)
    assert len(calls)==1
    request=calls[0]
    assert request['tools'][0]['function']['name']=='locate_alignment'
    assert image_count(request['messages'])==2
    fields=request['tools'][0]['function']['parameters']['properties']
    assert 'head_button_box' in fields and 'head_finger_tip' in fields
    assert 'head_vertical' not in fields
    judgment=c.assessments['live-0']
    assert judgment['format']=='pressb-grounding-v2'
    assert judgment['head_vertical']=='above' and judgment['head_horizontal']=='center'
    assert judgment['geometry']['error_button_units']==pytest.approx([0,-5.5])
    plan=c.memory['control_plan']
    assert (plan['stage'],plan['focus'],plan['step_hint_mm'])==('align','vertical',20)
    assert plan['alignment_size']['vertical']=='large'
    memory=json.loads(body['messages'][2]['content'][0]['text'])
    assert memory['control_plan']==plan
    current=json.loads(body['messages'][-1]['content'][0]['text'])
    assert current['context_type']=='CURRENT_ALIGNMENT_MODEL_JUDGMENT'
    assert current['geometry']==judgment['geometry']
    assert current['observation_id']=='live-0'
    assert image_count(body['messages'])==2
    archive=json.loads((tmp_path/'control_plan.json').read_text())
    assert archive==plan
    context=json.loads((tmp_path/'decision_context.json').read_text())
    assert context['perception_format']=='pressb-grounding-v2'
    assert context['control_policy_version']=='pressb-control-plan-v1'
    assert not c.turns and c.pending is None


def test_prepare_same_observation_is_idempotent_and_does_not_request_new_perception(jpeg,tmp_path,monkeypatch):
    c=make(jpeg);calls=model_localizer(monkeypatch,[coordinates()])
    first=c.prepare_decision('fake',tmp_path/'first')
    memory=copy.deepcopy(c.memory);judgments=copy.deepcopy(c.assessments)
    second=c.prepare_decision('fake',tmp_path/'second')
    assert len(calls)==1 and first==second
    assert c.memory==memory and c.assessments==judgments and c.processed=={'live-0'}
    assert not c.turns and c.pending is None


def test_program_policy_and_localization_agree_at_floating_point_size_boundary(jpeg,tmp_path,monkeypatch):
    c=make(jpeg)
    data=coordinates()
    data.update(head_button_box=[.45,.45,.55,.55], head_finger_tip=[.44999999999999996,.5])
    model_localizer(monkeypatch,[data])
    c.prepare_decision('fake',tmp_path)
    geometry=c.assessments['live-0']['geometry']
    assert geometry['error_button_units'][0]==pytest.approx(.5)
    assert geometry['magnitude']['horizontal']=='small'
    plan=c.memory['control_plan']
    assert plan['alignment_size']==geometry['magnitude']
    assert (plan['stage'],plan['focus'],plan['step_hint_mm'])==('align','horizontal',5)


@pytest.mark.parametrize('extra_observation',[False,True])
def test_repeating_sent_absolute_head_target_rejected_but_new_target_unchanged(jpeg,tmp_path,monkeypatch,extra_observation):
    c=make(jpeg)
    model_localizer(monkeypatch,[coordinates(True)]*3)
    c.prepare_decision('fake',tmp_path/'0')
    proposal=accept(c,tmp_path/'0','propose_head',
        dict(head_deg=[0,30],reason='Improve the HEAD panel view.'),'improve_head_view')
    c.append_result(proposal,completed(1,dict(type='head',head_deg=[0,30])),observation(1,(0,30)),jpeg,jpeg)
    c.prepare_decision('fake',tmp_path/'1')
    current=1
    if extra_observation:
        proposal=accept(c,tmp_path/'1','request_observation',dict(reason='Inspect current framing.'),'observe','observe')
        c.append_result(proposal,None,observation(2,(0,30)),jpeg,jpeg)
        c.prepare_decision('fake',tmp_path/'2');current=2
    plan=c.memory['control_plan']
    assert plan['stage']=='get_view'
    assert plan['head_target_info']['last_absolute_target_deg']==[0,30]
    assert plan['head_target_info']['measured_pose_stable_since_last_target'] is True
    again=action(c,'propose_head',dict(head_deg=[0,30],reason='Improve the HEAD panel view.'),'improve_head_view')
    with pytest.raises(ValueError,match='same_absolute_head_target'):
        c.parse_response(again,observation(current,(0,30)),SESSION,tmp_path/'repeat')
    assert c.pending is None
    different=action(c,'propose_head',dict(head_deg=[2.123456,35.765432],reason='Improve the HEAD panel view.'),'improve_head_view')
    proposal=c.parse_response(different,observation(current,(0,30)),SESSION,tmp_path/'different')
    assert proposal['arguments']['head_deg']==[2.123456,35.765432]


@pytest.mark.parametrize('effect',['reduce_horizontal_error','reduce_vertical_error','reduce_both_errors'])
@pytest.mark.parametrize('x',[1.123456,-1.123456])
def test_correction_intent_cannot_hide_any_X_motion(jpeg,tmp_path,monkeypatch,effect,x):
    c=make(jpeg);model_localizer(monkeypatch,[coordinates()]);c.prepare_decision('fake',tmp_path/'prepare')
    raw=action(c,values=dict(xyz_mm=[x,-10,20],reason='HEAD selected finger needs correction.'),effect=effect)
    with pytest.raises(ValueError,match='alignment_correction_requires_X_zero'):
        c.parse_response(raw,observation(0),SESSION,tmp_path/'action')
    assert c.pending is None and not c.turns


@pytest.mark.parametrize('effect,xyz,error',[
    ('reduce_horizontal_error',[0,0,10],'horizontal_correction_requires_nonzero_Y'),
    ('reduce_vertical_error',[0,10,0],'vertical_correction_requires_nonzero_Z'),
    ('reduce_both_errors',[0,0,10],'horizontal_correction_requires_nonzero_Y'),
    ('reduce_both_errors',[0,10,0],'vertical_correction_requires_nonzero_Z'),
])
def test_correction_intent_requires_the_named_actuator_axis(jpeg,tmp_path,monkeypatch,effect,xyz,error):
    c=make(jpeg);model_localizer(monkeypatch,[coordinates()]);c.prepare_decision('fake',tmp_path/'prepare')
    raw=action(c,values=dict(xyz_mm=xyz,reason='HEAD selected finger needs correction.'),effect=effect)
    with pytest.raises(ValueError,match=error):
        c.parse_response(raw,observation(0),SESSION,tmp_path/'action')
    assert c.pending is None


@pytest.mark.parametrize('effect,xyz',[
    ('reduce_horizontal_error',[0,-2.123456,0]),
    ('reduce_vertical_error',[0,0,3.141592]),
    ('reduce_both_errors',[0,-2.123456,3.141592]),
])
def test_advisory_twenty_mm_does_not_change_or_limit_chosen_correction(jpeg,tmp_path,monkeypatch,effect,xyz):
    c=make(jpeg);model_localizer(monkeypatch,[coordinates()]);c.prepare_decision('fake',tmp_path/'prepare')
    assert c.memory['control_plan']['step_hint_mm']==20
    raw=action(c,values=dict(xyz_mm=xyz,reason='HEAD selected finger needs correction.'),effect=effect)
    proposal=c.parse_response(raw,observation(0),SESSION,tmp_path/'action')
    assert proposal['arguments']['xyz_mm']==xyz
    assert set(proposal['arguments'])=={'xyz_mm','reason'}


@pytest.mark.parametrize('head_unknown',[False,True])
def test_explicit_approach_is_authorized_by_existing_wrist_guard_without_height_depth_gate(jpeg,tmp_path,monkeypatch,head_unknown):
    c=make(jpeg);calls=model_localizer(monkeypatch,[coordinates(head_unknown)])
    body=c.prepare_decision('fake',tmp_path/'prepare')
    assert c.memory['control_plan']['stage']==('get_view' if head_unknown else 'align')
    xyz=[3.141592,-2.123456,0]
    raw=action(c,values=dict(xyz_mm=xyz,reason='Use the current horizontal check for a single-finger approach.'),effect='approach')
    monkeypatch.setattr(client,'request',lambda *a,**k:raw)
    proposal=client.propose('fake',body,observation(0),SESSION,tmp_path/'action',conversation=c)
    assert [r['tools'][0]['function']['name'] for r in calls]==['locate_alignment','inspect_forward_alignment']
    assert image_count(calls[1]['messages'])==1
    assert proposal['arguments']['xyz_mm']==xyz
    assert c.forward_guard_decision['allowed'] is True
    assert c.forward_guard_decision['policy']=='wrist_horizontal'
    assert c.forward_guard_decision['authorized_command']==dict(type='move',xyz_mm=xyz)
    client.require_forward_permission(proposal,observation(0),c.forward_guard_decision)
    current=json.loads(c.observations['live-0']['content'][0]['text'])
    assert current['depth']['median_mm'] is None and current['state']['gripper_command']=='o'
    assert c.memory['evidence']['light_on'] is None
    assert c.pending['tool_calls'][0]['function']['name']=='propose_move'
    assert not c.turns  # No robot was created or executed during this model-only decision.


@pytest.mark.parametrize('guard_between',['no','unknown'])
def test_explicit_approach_still_cannot_bypass_wrist_guard(jpeg,tmp_path,monkeypatch,guard_between):
    c=make(jpeg);calls=model_localizer(monkeypatch,[coordinates()],guard_between=guard_between)
    body=c.prepare_decision('fake',tmp_path/'prepare')
    forward=action(c,values=dict(xyz_mm=[5,0,0],reason='Try the current approach.'),effect='approach')
    observe=action(c,'request_observation',dict(reason='Inspect the current wrist image.'),'observe','observe')
    replies=iter([forward,observe]);requests=[]
    monkeypatch.setattr(client,'request',lambda *a,**k:requests.append(a[2]) or next(replies))
    proposal=client.propose('fake',body,observation(0),SESSION,tmp_path/'action',conversation=c)
    assert len(requests)==2 and len(calls)==2
    assert 'forward_blocked' in requests[1]['messages'][-1]['content']
    assert proposal['name']=='request_observation'
    denied=json.loads((tmp_path/'action/forward_guard/decision.json').read_text())
    assert denied['allowed'] is False and denied['basis']['button_between_fingers']==guard_between
    assert c.pending['tool_calls'][0]['function']['name']=='request_observation'
    assert c.forward_guard_decision is None and not c.turns


@pytest.mark.parametrize('phase',['localization','tokenize','action','forward_guard'])
def test_stop_during_every_model_boundary_never_commits_action(jpeg,tmp_path,monkeypatch,phase):
    c=make(jpeg,65536 if phase=='tokenize' else None);cancel=threading.Event()
    stop_on={'localization':'locate_alignment','forward_guard':'inspect_forward_alignment'}.get(phase)
    calls=model_localizer(monkeypatch,[coordinates()],stop_on=stop_on,cancel=cancel)
    if phase=='tokenize':
        def stopped_tokenize(*a):cancel.set();return 1000
        monkeypatch.setattr(loop,'tokenize',stopped_tokenize)
    if phase in ('localization','tokenize'):
        with pytest.raises(InterruptedError):
            c.prepare_decision('fake',tmp_path/'prepare',cancel)
        if phase=='localization':
            assert not c.assessments and not c.processed
            assert not (tmp_path/'prepare/perception/assessment.json').exists()
    else:
        body=c.prepare_decision('fake',tmp_path/'prepare',cancel)
        def late_action(*a,**k):
            if phase=='action':cancel.set()
            return action(c,values=dict(xyz_mm=[5,0,0],reason='Try a current approach.'),effect='approach')
        monkeypatch.setattr(client,'request',late_action)
        with pytest.raises(InterruptedError):
            client.propose('fake',body,observation(0),SESSION,tmp_path/'action',cancel=cancel,conversation=c)
        assert not (tmp_path/'action/proposal.json').exists()
    assert c.pending is None and not c.turns
    if phase=='forward_guard':
        assert len(calls)==2 and c.forward_guard_decision is None


def test_stop_before_prepare_makes_no_model_request(jpeg,tmp_path,monkeypatch):
    c=make(jpeg);cancel=threading.Event();cancel.set()
    calls=model_localizer(monkeypatch,[])
    with pytest.raises(InterruptedError):
        c.prepare_decision('fake',tmp_path,cancel)
    assert not calls and not c.assessments and c.pending is None


def test_late_localization_cannot_bind_to_changed_current_observation(jpeg,tmp_path,monkeypatch):
    c=make(jpeg)
    monkeypatch.setattr(loop,'tokenize',lambda *a:1000)
    def changed(*a,**k):
        c.current_id='another-observation'
        return response('locate_alignment',coordinates())
    monkeypatch.setattr(grounding,'request',changed)
    with pytest.raises(ValueError,match='decision_observation_mismatch'):
        c.prepare_decision('fake',tmp_path)
    assert not c.assessments and not c.processed and c.pending is None
