"""Current-state control, evidence memory, and original action semantics."""
import copy
import json
import threading
from types import SimpleNamespace

import pytest

from pressb.agent_loop import client, grounding, localization, short_loop as s
from pressb.agent_loop.common import SESSION_SCHEMA
from pressb.agent_loop.conversation import image_count
from pressb.agent_loop.demonstration import image_content
from test_conversation import jpeg, obs, response
from test_grounding import unknown, shifted, assessment


SESSION = dict(schema=SESSION_SCHEMA, speed_mm_s=20, head_speed_deg_s=20)


def legacy_inspections(monkeypatch):
    # These original protocol/evidence tests intentionally retain their v1 scene
    # fixtures. The default v2 locator and its integration have separate tests.
    monkeypatch.setattr(localization, 'build_request', grounding.build_request)
    monkeypatch.setattr(localization, 'inspect', grounding.inspect)


@pytest.fixture(autouse=True)
def legacy_perception_protocol(monkeypatch):
    legacy_inspections(monkeypatch)


def state(n, head=(0, 25)):
    result = obs(n)
    result['state']['head_deg'] = list(head)
    result['state']['gripper_command'] = 'o'
    return result


def make(jpeg, demo=None, budget=65536):
    return s.ShortConversation('press down button and retract', state(0), jpeg, jpeg,
                               SESSION, 'robot commands and coordinates', demo=demo, budget=budget)


def mock_inspections(monkeypatch, rows, events=None, activation='unknown'):
    legacy_inspections(monkeypatch)
    rows = iter(rows)
    def request(base, route, payload, **kwargs):
        if payload['tools'][0]['function']['name']=='confirm_activation':
            if events is not None: events.append('activation')
            return response('confirm_activation',dict(light_on=activation))
        if events is not None: events.append('perception')
        return assessment(next(rows))
    monkeypatch.setattr(grounding, 'request', request)
    monkeypatch.setattr(s, 'tokenize', lambda *a: 1000)


def action(c, name='propose_move', args=None, effect='approach', cid='call-action'):
    arguments = copy.deepcopy(args or dict(xyz_mm=[1.123456,-2.456789,0], reason='HEAD selected finger needs adjustment.'))
    arguments.update(visual_basis={k:c.assessments[c.current_id][k] for k in s.BASIS_FIELDS}, expected_effect=effect)
    return response(name, arguments, cid)


def commit(c, directory, name='propose_move', args=None, effect='approach', cid='call-action'):
    raw = action(c, name, args, effect, cid)
    proposal = c.parse_response(raw, dict(observation_id=c.current_id), SESSION, directory)
    c.accept(raw)
    return proposal, raw


def result(n, xyz, **extra):
    return dict(status='completed', request=dict(type='move', xyz_mm=xyz, command_id=f'cmd-{n}'),
                motion=dict(action_completed=True, completion_basis='command_interpolation_finished', elapsed_s=.4),
                capture=dict(observation_id=f'live-{n}'), **extra)


def test_inspection_precedes_action_and_current_basis_is_consumed(jpeg,tmp_path,monkeypatch,approved_forward_view):
    c=make(jpeg); events=[]
    mock_inspections(monkeypatch,[shifted()],events)
    body=c.prepare_decision('fake',tmp_path/'round')
    assert events==['perception']
    assert image_count(json.loads((tmp_path/'round/perception/request.json').read_text())['messages'])==2
    current=json.loads(body['messages'][-1]['content'][0]['text'])
    assert current['context_type']=='CURRENT_ALIGNMENT_MODEL_JUDGMENT'
    assert current['observation_id']=='live-0' and current['head_horizontal']=='right'
    raw=action(c)
    monkeypatch.setattr(client,'request',lambda *a,**k:events.append('action') or raw)
    p=client.propose('fake',body,state(0),SESSION,tmp_path/'round',conversation=c)
    assert events==['perception','action']
    assert set(p['arguments'])=={'xyz_mm','reason'}
    assert p['arguments']['xyz_mm']==[1.123456,-2.456789,0]
    assert c.pending==raw['choices'][0]['message']
    assert json.loads((tmp_path/'round/decision_context.json').read_text())['format']=='pressb-conversation-v3'


def test_right_contact_and_adaptive_head_reference_reach_model_input(jpeg):
    c=make(jpeg)
    assert c.memory['preferred_contact']=='head_image_right'
    assert c.memory['selected_contact']=='head_image_right'
    preferences=json.loads(c.fixed[1]['content'][0]['text'])['operator_preferences']
    assert preferences['contact_finger']=='head_image_right'
    head=preferences['head_view']
    assert head['reference_pitch_deg']==30.0
    assert head['absolute_example_only'] is True and head['adapt_to_current_view'] is True
    assert c.pending is None and not c.turns


def test_unknown_contact_keeps_right_preference_across_observations(jpeg,tmp_path,monkeypatch):
    c=make(jpeg);requests=[];replies=iter([assessment(unknown()),assessment(shifted())])
    monkeypatch.setattr(grounding,'request',lambda *a,**k:requests.append(a[2]) or next(replies))
    monkeypatch.setattr(s,'tokenize',lambda *a:1000)
    c.prepare_decision('fake',tmp_path/'0')
    assert c.assessments['live-0']['contact_finger']=='unknown'
    assert c.memory['preferred_contact']==c.memory['selected_contact']=='head_image_right'
    p,_=commit(c,tmp_path/'0','request_observation',dict(reason='Look for the preferred finger.'),'observe')
    c.append_result(p,None,state(1),jpeg,jpeg)
    c.prepare_decision('fake',tmp_path/'1')
    assert [json.loads(r['messages'][1]['content'][0]['text'])['selected_contact'] for r in requests]==['head_image_right']*2
    assert c.memory['preferred_contact']==c.memory['selected_contact']=='head_image_right'


def test_wrong_finger_is_repaired_before_action_without_changing_preference(jpeg,tmp_path,monkeypatch):
    c=make(jpeg)
    mock_inspections(monkeypatch,[dict(shifted(),contact_finger='head_image_left'),shifted()])
    body=c.prepare_decision('fake',tmp_path)
    assert json.loads((tmp_path/'perception/metrics.json').read_text())['attempts']==2
    assert 'contact_reference_mismatch' in (tmp_path/'perception/repair-request.json').read_text()
    assert c.assessments[c.current_id]['contact_finger']=='head_image_right'
    assert c.memory['preferred_contact']==c.memory['selected_contact']=='head_image_right'
    assert c.pending is None
    assert json.loads(body['messages'][-1]['content'][0]['text'])['contact_finger']=='head_image_right'


def test_head_reference_does_not_force_or_clip_thirty_degree_motion(jpeg,tmp_path,monkeypatch):
    c=make(jpeg);mock_inspections(monkeypatch,[unknown()]);body=c.prepare_decision('fake',tmp_path/'prepare')
    assert c.pending is None and not c.turns  # Preparing the view never injects a startup head command.
    raw=action(c,'propose_head',dict(head_deg=[12.345,22.5],reason='Lower the head to reveal the panel.'),'improve_head_view')
    monkeypatch.setattr(client,'request',lambda *a,**k:raw)
    proposal=client.propose('fake',body,state(0),SESSION,tmp_path/'action',conversation=c)
    assert proposal['arguments']['head_deg']==[12.345,22.5]
    assert set(proposal['arguments'])=={'head_deg','reason'}
    assert c.pending['tool_calls'][0]['function']['name']=='propose_head'


def test_basis_mismatch_gets_exactly_one_repair_before_accept(jpeg,tmp_path,monkeypatch,approved_forward_view):
    c=make(jpeg);mock_inspections(monkeypatch,[shifted()])
    body=c.prepare_decision('fake',tmp_path/'prepare')
    bad=action(c); args=json.loads(bad['choices'][0]['message']['tool_calls'][0]['function']['arguments'])
    args['visual_basis']['wrist_visible']='yes'
    bad['choices'][0]['message']['tool_calls'][0]['function']['arguments']=json.dumps(args)
    good=action(c,cid='repaired-original-call-id')
    calls=[]; replies=iter([bad,good])
    monkeypatch.setattr(client,'request',lambda *a,**k:calls.append(a[2]) or next(replies))
    p=client.propose('fake',body,state(0),SESSION,tmp_path/'action',conversation=c)
    assert len(calls)==2 and c.pending['tool_calls'][0]['id']=='repaired-original-call-id'
    assert p['arguments']['xyz_mm']==[1.123456,-2.456789,0]
    repair=json.loads((tmp_path/'action/repair-request.json').read_text())
    assert 'visual_basis_must_match_CURRENT' in repair['messages'][-1]['content']
    assert json.loads((tmp_path/'action/action_metrics.json').read_text())['attempts']==2


def test_repeated_invalid_basis_remains_cancellable_without_accepted_action(jpeg,tmp_path,monkeypatch):
    c=make(jpeg);mock_inspections(monkeypatch,[shifted()])
    body=c.prepare_decision('fake',tmp_path/'prepare')
    bad=response('propose_move',dict(xyz_mm=[5,0,0],reason='forward'))
    calls=[]
    cancel=threading.Event()
    def invalid_until_cancel(*a,**k):
        calls.append(a)
        if len(calls)==4:cancel.set()
        return bad
    monkeypatch.setattr(client,'request',invalid_until_cancel)
    with pytest.raises(InterruptedError):
        client.propose('fake',body,state(0),SESSION,tmp_path/'action',conversation=c,cancel=cancel)
    assert len(calls)==4 and c.pending is None and not (tmp_path/'action/proposal.json').exists()


@pytest.mark.parametrize('reason',[
    'Wrist view shows the button between the fingers; press forward.',
    '腕图可见按钮在两指之间，因此前推。',
    '腕部相机按钮位于两指之间。',
])
def test_explicit_invisible_target_prose_is_rejected(jpeg,tmp_path,monkeypatch,reason):
    c=make(jpeg);mock_inspections(monkeypatch,[shifted()]);c.prepare_decision('fake',tmp_path)
    with pytest.raises(ValueError,match='reason_conflicts'):
        c.parse_response(action(c,args=dict(xyz_mm=[5,0,0],reason=reason)),state(0),SESSION,tmp_path)
    assert c.pending is None


@pytest.mark.parametrize('reason',[
    'Wrist view does not show the button; inspect head alignment.',
    'Wrist button is not visible; HEAD selected finger remains left of target.',
    '腕图看不到按钮；使用头图判断接近。',
    '腕图未见按钮，头图可见按钮在右侧，向右移以纠偏。',
    'Wrist cannot see the button, head shows the button right of the selected finger; move right.',
])
def test_explicit_unknown_or_negative_visibility_is_allowed(jpeg,tmp_path,monkeypatch,reason):
    c=make(jpeg);mock_inspections(monkeypatch,[shifted()]);c.prepare_decision('fake',tmp_path)
    p=c.parse_response(action(c,args=dict(xyz_mm=[2.123456,0,0],reason=reason)),state(0),SESSION,tmp_path)
    assert p['arguments']['xyz_mm']==[2.123456,0,0]


def test_canonical_move_preserves_unknown_depth_and_open_finger(jpeg,tmp_path,monkeypatch):
    c=make(jpeg);mock_inspections(monkeypatch,[unknown()]);c.prepare_decision('fake',tmp_path)
    raw=action(c,args=dict(xyz_mm=[3.141592,0,0],reason='Single open finger exploratory approach; alignment unknown.'))
    p=c.parse_response(raw,state(0),SESSION,tmp_path)
    assert p['arguments']['xyz_mm']==[3.141592,0,0]
    live=json.loads(c.observations['live-0']['content'][0]['text'])
    assert live['state']['gripper_command']=='o' and live['depth']['median_mm'] is None


def test_head_cannot_claim_wrist_actuation_effect(jpeg,tmp_path,monkeypatch):
    c=make(jpeg);mock_inspections(monkeypatch,[shifted()]);c.prepare_decision('fake',tmp_path)
    raw=action(c,'propose_head',dict(head_deg=[0,20],reason='Change view.'),'improve_wrist_view')
    with pytest.raises(ValueError,match='expected_effect_incompatible'):
        c.parse_response(raw,state(0),SESSION,tmp_path)


@pytest.mark.parametrize('phase',['perception','action'])
def test_late_stop_never_commits_a_response(jpeg,tmp_path,monkeypatch,phase):
    c=make(jpeg);cancel=threading.Event()
    if phase=='perception':
        monkeypatch.setattr(s,'tokenize',lambda *a:1000)
        def late(*a,**k):cancel.set();return assessment(shifted())
        monkeypatch.setattr(grounding,'request',late)
        with pytest.raises(InterruptedError):c.prepare_decision('fake',tmp_path,cancel)
        assert not c.assessments and not c.processed
    else:
        mock_inspections(monkeypatch,[shifted()]);body=c.prepare_decision('fake',tmp_path/'prepare')
        def late(*a,**k):cancel.set();return action(c)
        monkeypatch.setattr(client,'request',late)
        with pytest.raises(InterruptedError):
            client.propose('fake',body,state(0),SESSION,tmp_path/'action',cancel=cancel,conversation=c)
    assert c.pending is None and not c.turns


def test_context_images_stay_bounded_while_full_history_remains(jpeg,tmp_path,monkeypatch):
    demo=SimpleNamespace(content=[dict(type='text',text='HISTORICAL_DEMO_NOT_CURRENT')]+[image_content(jpeg) for _ in range(3)])
    c=make(jpeg,demo);mock_inspections(monkeypatch,[shifted() for _ in range(8)])
    prior=[]
    for n in range(8):
        body=c.prepare_decision('fake',tmp_path/str(n))
        assert image_count(body['messages'])==5
        assert all(m['role'] not in ('assistant','tool') for m in body['messages'])
        if n==7:break
        proposal,raw=commit(c,tmp_path/str(n),cid=f'call-{n}',args=dict(xyz_mm=[0,-2.12345,0],reason=f'old reason {n}'))
        c.append_result(proposal,result(n+1,proposal['arguments']['xyz_mm']),state(n+1),jpeg,jpeg)
        prior.append(copy.deepcopy(c.turns[-1]))
    assert len(c.turns)==7 and c.turns==prior and len(c.observations)==8
    assert len(c.assessments)==8
    assert all(turn['assistant']['tool_calls'][0]['id']==turn['tool']['tool_call_id'] for turn in c.turns)
    assert 'old reason' not in json.dumps(c.messages())
    assert c.turns[0]['proposal']['arguments']['xyz_mm']==[0,-2.12345,0]
    with pytest.raises(ValueError,match='decision_observation_mismatch'):c.check_observation('live-0')


def test_replay_effect_uses_recorded_command_and_preserves_proposal(jpeg,tmp_path,monkeypatch):
    c=make(jpeg);mock_inspections(monkeypatch,[shifted(),shifted()]);c.prepare_decision('fake',tmp_path/'0')
    p,_=commit(c,tmp_path/'0',args=dict(xyz_mm=[999,0,0],reason='Unexecuted replay proposal.'))
    recorded=dict(type='move',xyz_mm=[0,-20,0])
    r=result(1,[999,0,0],replay=dict(recorded_actual_command=recorded,reason='not executed'))
    r['status']='not_executed_replay'
    c.append_result(p,r,state(1),jpeg,jpeg);c.prepare_decision('fake',tmp_path/'1')
    assert c.memory['latest_result']['command']==recorded
    assert c.memory['latest_result']['command_source']=='saved_recording_not_replay_proposal'
    assert c.memory['alignment_progress']['previous_command']==recorded
    assert c.turns[-1]['proposal']['arguments']['xyz_mm']==[999,0,0]
    assert c.turns[-1]['result']['replay']['recorded_actual_command']==recorded
    assert c.memory['alignment_progress']['status']=='unchanged_or_unresolved'
    assert 'do not prove no physical improvement' in c.memory['alignment_progress']['limitation']


def test_observe_only_does_not_invent_actual_motion(jpeg,tmp_path,monkeypatch):
    c=make(jpeg);mock_inspections(monkeypatch,[shifted(),shifted()]);c.prepare_decision('fake',tmp_path/'0')
    p,_=commit(c,tmp_path/'0','request_observation',dict(reason='Refresh current scene.'),'observe')
    c.append_result(p,None,state(1),jpeg,jpeg);c.prepare_decision('fake',tmp_path/'1')
    assert c.memory['latest_result']['status']=='observed'
    assert c.memory['latest_result']['command'] is None
    assert c.memory['alignment_progress']['status']=='unknown'


@pytest.mark.parametrize('first_lamp',['on','unknown'])
def test_initial_lit_or_unknown_cannot_be_activation(jpeg,tmp_path,monkeypatch,first_lamp):
    c=make(jpeg);data=dict(shifted(),lamp=first_lamp)
    mock_inspections(monkeypatch,[data,dict(shifted(),lamp='on')])
    c.prepare_decision('fake',tmp_path/'0')
    p,_=commit(c,tmp_path/'0','request_observation',dict(reason='Refresh lamp observation.'),'observe')
    c.append_result(p,None,state(1),jpeg,jpeg);c.prepare_decision('fake',tmp_path/'1')
    assert c.memory['evidence']['light_on'] is None
    with pytest.raises(ValueError,match='light_on_and_later_verified_withdrawal'):
        c.parse_response(action(c,'report_done',dict(evidence='live-0 live-1'),'finish'),state(1),SESSION,tmp_path/'done')


@pytest.mark.parametrize('withdrawal',['yes','no','unknown'])
def test_completion_requires_off_to_on_then_visual_retreat_and_both_ids(jpeg,tmp_path,monkeypatch,withdrawal):
    c=make(jpeg)
    mock_inspections(monkeypatch,[shifted(),dict(shifted(),lamp='on'),dict(shifted(),lamp='on')])
    c.prepare_decision('fake',tmp_path/'0')
    p,_=commit(c,tmp_path/'0',args=dict(xyz_mm=[5,0,0],reason='Approach with current visual basis.'),effect='approach')
    c.append_result(p,result(1,[5,0,0]),state(1),jpeg,jpeg);c.prepare_decision('fake',tmp_path/'1')
    assert c.memory['evidence']['light_on']['previous_observation_id']=='live-0'
    assert c.memory['evidence']['light_on']['observation_id']=='live-1'
    p,_=commit(c,tmp_path/'1',args=dict(xyz_mm=[-10,0,0],reason='Withdraw after light-on.'),effect='retract')
    c.append_result(p,result(2,[-10,0,0]),state(2),jpeg,jpeg)
    checked=[]
    def verify(base,before,current,command,directory,cancel=None):
        checked.append((before,current,command))
        return dict(withdrawal=withdrawal,evidence_observation_ids=['live-1','live-2'],source='model_not_human')
    monkeypatch.setattr(grounding,'verify_withdrawal',verify)
    c.prepare_decision('fake',tmp_path/'2')
    assert len(checked)==1 and checked[0][2]==dict(type='move',xyz_mm=[-10,0,0])
    done=action(c,'report_done',dict(evidence='Light-on live-1; visible withdrawal live-2.'),'finish')
    if withdrawal=='yes':
        assert c.parse_response(done,state(2),SESSION,tmp_path/'done')['name']=='report_done'
        missing=action(c,'report_done',dict(evidence='Light-on live-1; withdrawn.'),'finish')
        with pytest.raises(ValueError,match='must_cite'):
            c.parse_response(missing,state(2),SESSION,tmp_path/'missing')
    else:
        assert c.memory['evidence']['withdrawal'] is None
        with pytest.raises(ValueError,match='light_on_and_later_verified_withdrawal'):
            c.parse_response(done,state(2),SESSION,tmp_path/'done')


def test_light_first_seen_after_retreat_is_not_post_activation_withdrawal(jpeg,tmp_path,monkeypatch):
    c=make(jpeg);mock_inspections(monkeypatch,[shifted(),dict(shifted(),lamp='on')]);c.prepare_decision('fake',tmp_path/'0')
    p,_=commit(c,tmp_path/'0',args=dict(xyz_mm=[-10,0,0],reason='Change separation while lamp not confirmed.'),effect='retract')
    c.append_result(p,result(1,[-10,0,0]),state(1),jpeg,jpeg)
    calls=[]
    monkeypatch.setattr(grounding,'verify_withdrawal',lambda *a,**k:calls.append(a))
    c.prepare_decision('fake',tmp_path/'1')
    assert c.memory['evidence']['light_on'] is not None
    assert c.memory['evidence']['withdrawal'] is None and not calls


def test_cancellation_during_withdrawal_never_reaches_action_request(jpeg,tmp_path,monkeypatch):
    c=make(jpeg);mock_inspections(monkeypatch,[shifted(),dict(shifted(),lamp='on'),dict(shifted(),lamp='on')])
    c.prepare_decision('fake',tmp_path/'0')
    p,_=commit(c,tmp_path/'0');c.append_result(p,result(1,[1,0,0]),state(1),jpeg,jpeg)
    c.prepare_decision('fake',tmp_path/'1')
    p,_=commit(c,tmp_path/'1',args=dict(xyz_mm=[-10,0,0],reason='Retract.'),effect='retract')
    c.append_result(p,result(2,[-10,0,0]),state(2),jpeg,jpeg)
    cancel=threading.Event()
    def late(*a,**k):cancel.set();return dict(withdrawal='yes')
    monkeypatch.setattr(grounding,'verify_withdrawal',late)
    with pytest.raises(InterruptedError):c.prepare_decision('fake',tmp_path/'2',cancel)
    assert c.memory['evidence']['withdrawal'] is None and 'live-2' not in c.processed
    assert json.loads((tmp_path/'2/decision_context.json').read_text())['stage']=='interrupted'


def test_same_observation_prepare_does_not_repeat_perception(jpeg,tmp_path,monkeypatch):
    c=make(jpeg);events=[];mock_inspections(monkeypatch,[shifted()],events)
    first=c.prepare_decision('fake',tmp_path/'0')
    again=c.prepare_decision('fake',tmp_path/'again')
    assert events==['perception'] and first==again


@pytest.mark.parametrize('verified',['yes','no','unknown'])
def test_ambiguous_single_frame_lamp_uses_pair_verification_on_demand(jpeg,tmp_path,monkeypatch,verified):
    c=make(jpeg);events=[]
    mock_inspections(monkeypatch,[dict(shifted(),lamp='unknown'),dict(shifted(),lamp='on')],events,activation=verified)
    c.prepare_decision('fake',tmp_path/'0')
    assert c.memory['evidence']['light_on'] is None and events==['perception']
    p,_=commit(c,tmp_path/'0',args=dict(xyz_mm=[999,0,0],reason='Unexecuted replay proposal.'),effect='press')
    actual=dict(type='move',xyz_mm=[5,0,0])
    r=result(1,[999,0,0],replay=dict(recorded_actual_command=actual));r['status']='not_executed_replay'
    c.append_result(p,r,state(1),jpeg,jpeg)
    c.prepare_decision('fake',tmp_path/'1')
    assert events==['perception','perception','activation']
    check=c.memory['evidence']['activation_check']
    assert check['light_on']==verified and check['previous_observation_id']=='live-0' and check['observation_id']=='live-1'
    request_body=json.loads((tmp_path/'1/activation/request.json').read_text())
    assert image_count(request_body['messages'])==4
    assert json.loads(request_body['messages'][1]['content'][-1]['text'])['actual_command']==actual
    assert check['source']=='model_not_human'
    if verified=='yes':
        assert c.memory['evidence']['light_on']['evidence_observation_ids']==['live-0','live-1']
        assert c.memory['evidence']['light_on']['source'].startswith('model_visual_off_to_on')
    else:
        assert c.memory['evidence']['light_on'] is None
    assert c.memory['evidence']['withdrawal'] is None
    with pytest.raises(ValueError,match='light_on_and_later_verified_withdrawal'):
        c.parse_response(action(c,'report_done',dict(evidence='light live-1'),'finish'),state(1),SESSION,tmp_path/'done')


def test_already_known_off_to_on_does_not_add_activation_request(jpeg,tmp_path,monkeypatch):
    c=make(jpeg);events=[]
    mock_inspections(monkeypatch,[shifted(),dict(shifted(),lamp='on')],events)
    c.prepare_decision('fake',tmp_path/'0')
    p,_=commit(c,tmp_path/'0');c.append_result(p,result(1,[1,0,0]),state(1),jpeg,jpeg)
    c.prepare_decision('fake',tmp_path/'1')
    assert events==['perception','perception']
    assert c.memory['evidence']['light_on']['observation_id']=='live-1'
    assert not (tmp_path/'1/activation').exists()


def test_pair_activation_first_confirmed_after_retreat_does_not_prove_post_activation_retreat(jpeg,tmp_path,monkeypatch):
    c=make(jpeg);events=[]
    mock_inspections(monkeypatch,[dict(shifted(),lamp='unknown'),dict(shifted(),lamp='on')],events,activation='yes')
    c.prepare_decision('fake',tmp_path/'0')
    p,_=commit(c,tmp_path/'0',args=dict(xyz_mm=[-10,0,0],reason='Withdraw while activation uncertain.'),effect='retract')
    c.append_result(p,result(1,[-10,0,0]),state(1),jpeg,jpeg)
    calls=[]
    monkeypatch.setattr(grounding,'verify_withdrawal',lambda *a,**k:calls.append(a))
    c.prepare_decision('fake',tmp_path/'1')
    assert events==['perception','perception','activation']
    assert c.memory['evidence']['light_on'] is not None
    assert c.memory['evidence']['withdrawal'] is None and not calls


def test_stop_during_pair_activation_rejects_late_light_event(jpeg,tmp_path,monkeypatch):
    c=make(jpeg);mock_inspections(monkeypatch,[dict(shifted(),lamp='unknown')]);c.prepare_decision('fake',tmp_path/'0')
    p,_=commit(c,tmp_path/'0');c.append_result(p,result(1,[1,0,0]),state(1),jpeg,jpeg)
    cancel=threading.Event();events=[]
    def request(base,route,body,**kwargs):
        name=body['tools'][0]['function']['name'];events.append(name)
        if name=='inspect_alignment':return assessment(dict(shifted(),lamp='on'))
        cancel.set();return response('confirm_activation',dict(light_on='yes'))
    monkeypatch.setattr(grounding,'request',request)
    with pytest.raises(InterruptedError):c.prepare_decision('fake',tmp_path/'1',cancel)
    assert events==['inspect_alignment','confirm_activation']
    assert c.memory['evidence']['light_on'] is None
    assert 'activation_check' not in c.memory['evidence'] and 'live-1' not in c.processed
    assert c.pending is None and not (tmp_path/'1/request.json').exists()
    assert not (tmp_path/'1/activation/assessment.json').exists()
    assert json.loads((tmp_path/'1/decision_context.json').read_text())['stage']=='interrupted'


def test_budget_failure_is_explicit_and_keeps_cases(jpeg,tmp_path,monkeypatch):
    c=make(jpeg,budget=500);mock_inspections(monkeypatch,[shifted()])
    with pytest.raises(ValueError,match='context_budget_exceeded'):
        c.prepare_decision('fake',tmp_path)
    assert c.pending is None
    assert json.loads((tmp_path/'decision_context.json').read_text())['stage']=='failed'


def test_non_object_action_arguments_get_one_format_repair(jpeg,tmp_path,monkeypatch,approved_forward_view):
    c=make(jpeg);mock_inspections(monkeypatch,[shifted()]);body=c.prepare_decision('fake',tmp_path/'prepare')
    malformed=action(c)
    malformed['choices'][0]['message']['tool_calls'][0]['function']['arguments']='[]'
    replies=iter([malformed,action(c)]);calls=[]
    monkeypatch.setattr(client,'request',lambda *a,**k:calls.append(a) or next(replies))
    p=client.propose('fake',body,state(0),SESSION,tmp_path/'action',conversation=c)
    assert p['name']=='propose_move' and len(calls)==2


def test_retract_claim_must_have_negative_x_without_changing_values(jpeg,tmp_path,monkeypatch):
    c=make(jpeg);mock_inspections(monkeypatch,[shifted()]);c.prepare_decision('fake',tmp_path)
    with pytest.raises(ValueError,match='negative_X'):
        c.parse_response(action(c,args=dict(xyz_mm=[1,0,0],reason='Retreat.'),effect='retract'),state(0),SESSION,tmp_path)
    p=c.parse_response(action(c,args=dict(xyz_mm=[-1.234567,2.345678,0],reason='Retreat and adjust.'),effect='retract'),state(0),SESSION,tmp_path)
    assert p['arguments']['xyz_mm']==[-1.234567,2.345678,0]


def test_later_hand_motion_invalidates_older_withdrawal_for_completion(jpeg,tmp_path,monkeypatch):
    c=make(jpeg)
    mock_inspections(monkeypatch,[shifted()]+[dict(shifted(),lamp='on') for _ in range(3)])
    c.prepare_decision('fake',tmp_path/'0')
    p,_=commit(c,tmp_path/'0');c.append_result(p,result(1,[1,0,0]),state(1),jpeg,jpeg)
    c.prepare_decision('fake',tmp_path/'1')
    p,_=commit(c,tmp_path/'1',args=dict(xyz_mm=[-10,0,0],reason='Withdraw.'),effect='retract')
    c.append_result(p,result(2,[-10,0,0]),state(2),jpeg,jpeg)
    monkeypatch.setattr(grounding,'verify_withdrawal',lambda *a,**k:dict(withdrawal='yes',evidence_observation_ids=['live-1','live-2']))
    c.prepare_decision('fake',tmp_path/'2')
    assert c.memory['evidence']['withdrawal']['observation_id']=='live-2'
    p,_=commit(c,tmp_path/'2',args=dict(xyz_mm=[5,0,0],reason='Approach again.'),effect='approach')
    c.append_result(p,result(3,[5,0,0]),state(3),jpeg,jpeg)
    assert c.memory['evidence']['withdrawal'] is None
    assert c.memory['evidence']['withdrawal_invalidated_by']['previous_withdrawal_observation_id']=='live-2'
    c.prepare_decision('fake',tmp_path/'3')
    with pytest.raises(ValueError,match='light_on_and_later_verified_withdrawal'):
        c.parse_response(action(c,'report_done',dict(evidence='light-on live-1; retreat live-2'),'finish'),state(3),SESSION,tmp_path/'done')


@pytest.mark.parametrize('condition',['normal','stop_perception','stop_action','capture_failed'])
def test_real_runner_fake_robot_preserves_stop_and_no_replay(tmp_path,monkeypatch,condition,approved_forward_view):
    import s1b_agent
    from test_agent_loop import FakeBackend,FakeCamera
    from pressb.agent_loop.service import Service,make_server
    from pressb.agent_loop.common import request as http_request
    backend,camera=FakeBackend(),FakeCamera()
    service=Service(backend,camera,tmp_path/'robot')
    server=make_server(service,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    controls=[];original_control=s1b_agent.StopControl;events=[]
    def control_factory():
        control=original_control();controls.append(control);return control
    monkeypatch.setattr(s1b_agent,'StopControl',control_factory)
    def inspect_call(*a,**k):
        events.append('perception')
        if condition=='stop_perception':controls[0].cancel.set()
        return assessment(shifted())
    def model_or_http(base,route,*args,**kwargs):
        if route!='/chat/completions':return http_request(base,route,*args,**kwargs)
        events.append('action')
        if condition=='stop_action':controls[0].cancel.set()
        if condition=='capture_failed':camera.fail=True
        current=json.loads(args[0]['messages'][-1]['content'][0]['text'])
        return response('propose_move',dict(xyz_mm=[1.123456,-2.456789,0],reason='Use selected HEAD finger for exploratory approach.',
            visual_basis={k:current[k] for k in s.BASIS_FIELDS},expected_effect='approach'))
    monkeypatch.setattr(grounding,'request',inspect_call)
    monkeypatch.setattr(client,'request',model_or_http)
    monkeypatch.setattr(s,'tokenize',lambda *a:1000)
    prompt=tmp_path/'prompt.txt';prompt.write_text('robot command contract')
    args=SimpleNamespace(output=str(tmp_path/'client'),prompt=str(prompt),model_base='http://model.invalid/v1',
        replay=None,thor_port=18000,thor_host='unused',robot_base=f'http://127.0.0.1:{server.server_port}',
        max_rounds=3 if condition=='capture_failed' else 1,goal='press',decision_mode='grounded')
    try:
        if condition.startswith('stop_'):
            with pytest.raises(InterruptedError):s1b_agent.run(args)
            assert not backend.calls
        elif condition=='capture_failed':
            with pytest.raises(ValueError,match='task_failed'):s1b_agent.run(args)
            assert len(backend.calls)==1
        else:
            s1b_agent.run(args)
            assert len(backend.calls)==1
            assert backend.calls[0][0]=='move'
            assert backend.calls[0][1]['xyz_mm']==[1.123456,-2.456789,0]
            assert 'visual_basis' not in backend.calls[0][1]
        assert events==(['perception'] if condition=='stop_perception' else ['perception','action'])
        assert backend.stopped
        if condition=='capture_failed':
            assert len(list((tmp_path/'client/commands').glob('*.json')))==1
    finally:
        server.shutdown();server.server_close();service.close();thread.join()
