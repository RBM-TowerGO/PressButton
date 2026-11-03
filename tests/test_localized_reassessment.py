import copy
import json
import threading

import pytest

from pressb.agent_loop import grounding, client, conversation, model_localization as loc, localized_reassessment as review
from pressb.agent_loop.conversation import live_message, image_count
from test_conversation import jpeg, obs, response, summary_response
from test_model_localization import locations, reply, view, SIZES
from test_localized_conversation import make, action, advance, observation


def scene(n, x=.7, partial=False):
    return loc.assess(reply(locations(head=view(contact_tip=[x,.55]),
        wrist=view(contact_tip=[x,.55], button_visibility='partial' if partial else 'full'))), 'live-'+str(n), SIZES)


def turns(xyz=(5,0,0), count=3):
    return [dict(ledger=dict(before_observation_id='live-'+str(n), after_observation_id='live-'+str(n+1),
        result=dict(command_id='c'+str(n), command=dict(type='move', xyz_mm=list(xyz)),
                    status='completed', sends_completed=True))) for n in range(count)]


def answer(effect='unknown'):
    return response('review_motion_hypothesis', dict(observed_effect=effect,
        distance_only_hypothesis='unsupported', light_change='unknown', prediction_check='not_supplied',
        evidence='Button remains above and left of marked end in wrist view.',
        alternative_explanation='Incomplete lateral and vertical alignment.',
        next_check='Check if both projections converge and whether target illuminates.'))


def test_trigger_counts_only_actual_unique_completed_commands_not_observation_refresh():
    history=turns(); locs={f'live-{n}':scene(n) for n in range(4)}
    assert not review.monitor(history[:2],locs,'live-2')['review_required']
    history.insert(1,copy.deepcopy(history[0]))
    history.insert(2,dict(ledger=dict(result=dict(status='observed'))))
    signal=review.monitor(history,locs,'live-3')
    assert signal['review_required'] and signal['windows'][0]['count']==3
    assert signal['windows'][0]['total_mm']==15
    assert signal['baseline_observation_id']=='live-0'
    for status in ('failed','interrupted'):
        broken=copy.deepcopy(history);broken[-1]['ledger']['result']['status']=status
        assert not review.monitor(broken,locs,'live-3')['review_required']


def test_replay_uses_recorded_motion_and_known_projection_improvement_is_not_failure():
    history=turns(xyz=(0,5,0));locs={f'live-{n}':scene(n,x=.7-n*.08) for n in range(4)}
    # Radial error also includes vertical component; this still improves >10%.
    assert not review.monitor(history,locs,'live-3')['review_required']
    for t in history:
        t['ledger']['result']['replay']=dict(recorded_actual_command=dict(type='move',xyz_mm=[5,0,0]),proposal_executed=False)
    signal=review.monitor(history,locs,'live-3')
    assert signal['review_required'] and signal['windows'][0]['axis']=='X'
    assert all(c['command']['xyz_mm']==[5,0,0] for c in signal['windows'][0]['commands'])


def test_partial_box_or_reference_change_cannot_claim_projection_improvement():
    locs={f'live-{n}':scene(n,x=.7-n*.08,partial=True) for n in range(4)}
    for item in locs.values(): item['head']['contact_visibility']='unknown'
    signal=review.monitor(turns((0,5,0)),locs,'live-3')
    assert signal['review_required']
    assert signal['windows'][0]['views']['wrist']['projection_effect']=='unknown'
    locs={f'live-{n}':scene(n,x=.7-n*.08) for n in range(4)}
    locs['live-3']['contact_reference']='other_marker'
    assert review.monitor(turns((0,5,0)),locs,'live-3')['review_required']


def test_review_original_four_images_binding_no_old_reason_and_prediction_pair(jpeg):
    history=turns();locs={f'live-{n}':scene(n) for n in range(4)}
    signal=review.monitor(history,locs,'live-3')
    before=live_message(obs(0),jpeg,jpeg);current=live_message(obs(3),jpeg,jpeg)
    body=review.build_request('press',before,current,signal,locs)
    assert image_count(body['messages'])==4
    assert 'raw_locations' not in json.dumps(body) and 'raw_output' not in json.dumps(body)
    assert all('xyz_mm' not in t['function']['parameters']['properties'] for t in body['tools'])
    with pytest.raises(ValueError,match='observation_mismatch'):
        review.build_request('press',current,before,signal,locs)
    history[-1]['ledger']['reassessment']=dict(expected_visual_change='test next image')
    signal=review.monitor(history,locs,'live-3')
    assert signal['baseline_observation_id']=='live-2'
    body=review.build_request('press',live_message(obs(2),jpeg,jpeg),current,signal,locs)
    payload=json.loads(body['messages'][1]['content'][0]['text'])
    assert len(payload['actual_commands'])==1
    assert payload['review_trigger']['previous_prediction']['expected_visual_change']=='test next image'


def test_zero_move_after_review_never_breaks_boundary_or_counts_as_progress(jpeg):
    history=turns();history.append(dict(ledger=dict(before_observation_id='live-3',after_observation_id='live-4',
        reassessment=dict(expected_visual_change='should remain still'),
        result=dict(status='completed',sends_completed=True,command_id='zero',command=dict(type='move',xyz_mm=[0,0,0])))))
    locs={f'live-{n}':scene(n) for n in range(5)}
    signal=review.monitor(history,locs,'live-4')
    assert signal['windows'][0]['count']==3 and signal['baseline_observation_id']=='live-0'
    body=review.build_request('press',live_message(obs(0),jpeg,jpeg),live_message(obs(4),jpeg,jpeg),signal,locs)
    assert len(json.loads(body['messages'][1]['content'][0]['text'])['actual_commands'])==4


def test_prediction_check_cannot_invent_or_drop_a_previous_prediction():
    r=answer()
    with pytest.raises(ValueError,match='prediction_check_required'):
        review.validate(r,'live-0','live-3',has_prediction=True)
    data=json.loads(r['choices'][0]['message']['tool_calls'][0]['function']['arguments'])
    data['prediction_check']='not_met'
    r['choices'][0]['message']['tool_calls'][0]['function']['arguments']=json.dumps(data)
    assert review.validate(r,'live-0','live-3',True)['prediction_check']=='not_met'
    with pytest.raises(ValueError,match='no_previous_prediction'):
        review.validate(r,'live-0','live-3',False)


def test_invalid_review_repairs_without_robot_and_stop_rejects_late_assessment(jpeg, monkeypatch, tmp_path):
    locs={f'live-{n}':scene(n) for n in range(4)}
    body=review.build_request('press',live_message(obs(0),jpeg,jpeg),live_message(obs(3),jpeg,jpeg),
                              review.monitor(turns(),locs,'live-3'),locs)
    answers=[response('propose_move',dict(xyz_mm=[5,0,0],reason='wrong tool')),answer()]
    monkeypatch.setattr(grounding,'request',lambda *a,**k:answers.pop(0))
    result=review.inspect('fake',body,'live-3','live-0',tmp_path/'repair')
    assert result['evidence_observation_ids']==['live-0','live-3']
    assert json.loads((tmp_path/'repair/metrics.json').read_text())['validation_failures']==1
    cancel=threading.Event()
    def late(*a,**k):cancel.set();return answer()
    monkeypatch.setattr(grounding,'request',late)
    with pytest.raises(InterruptedError):review.inspect('fake',body,'live-3','live-0',tmp_path/'stop',cancel)
    assert not (tmp_path/'stop/assessment.json').exists()


def repeated_conversation(jpeg,monkeypatch,tmp_path):
    c,_=make(jpeg,monkeypatch)
    for n in range(3):
        c.prepare_decision('fake',tmp_path/str(n));advance(c,jpeg,n+1,tmp_path/str(n),xyz=[5,0,0])
    return c


def test_review_precedes_action_and_reassessment_is_stripped_before_execution(jpeg,monkeypatch,tmp_path):
    c=repeated_conversation(jpeg,monkeypatch,tmp_path)
    body=c.prepare_decision('fake',tmp_path/'3')
    assert c.hypothesis_reviews['live-3']['observation_id']=='live-3'
    assert 'MOTION_HYPOTHESIS_REVIEW' in body['messages'][-2]['content'][0]['text']
    hint=json.loads(body['messages'][-1]['content'][0]['text'])
    assert hint['context_type']=='CURRENT_WRIST_ALIGNMENT_DIRECTION_RULE'
    assert hint['observation_id']=='live-3'
    assert 'reassessment' in body['tools'][0]['function']['parameters']['required']
    missing=action(c,xyz=[-2.123,4.567,0])
    raw=json.loads(missing['choices'][0]['message']['tool_calls'][0]['function']['arguments'])
    del raw['reassessment'];missing['choices'][0]['message']['tool_calls'][0]['function']['arguments']=json.dumps(raw)
    with pytest.raises(ValueError,match='fresh_reassessment_required'):
        c.parse_response(missing,observation(3),c.session,tmp_path/'3')
    valid=action(c,xyz=[-2.123,4.567,0]);replies=[missing,valid]
    monkeypatch.setattr(client,'request',lambda *a,**k:replies.pop(0))
    p=client.propose('fake',body,observation(3),c.session,tmp_path/'3',conversation=c)
    assert p['arguments']['xyz_mm']==[-2.123,4.567,0] and 'reassessment' not in p['arguments']
    archived=json.loads((tmp_path/'3/hypothesis_review/decision.json').read_text())
    assert archived['expected_visual_change'] and archived['status']=='validated_proposal_not_execution'
    c.check_execution(p,'live-3')
    assert json.loads(c.pending['tool_calls'][0]['function']['arguments'])['reassessment']['review_id']=='review-live-3'
    with pytest.raises(ValueError):c.check_execution(p,'live-2')


def test_stopping_during_review_cannot_accept_or_execute(jpeg,monkeypatch,tmp_path):
    c=repeated_conversation(jpeg,monkeypatch,tmp_path);cancel=threading.Event()
    def late(base,body,oid,previous,directory,cancel=None):
        cancel.set();return review.validate(answer(),previous,oid)
    monkeypatch.setattr(review,'inspect',late)
    with pytest.raises(InterruptedError):c.prepare_decision('fake',tmp_path/'3',cancel)
    assert c.pending is None and not c.hypothesis_reviews
    with pytest.raises(ValueError,match='review_not_completed'):c.check_observation('live-3')


def test_compaction_preserves_review_history_and_does_not_reset_streak(jpeg,monkeypatch,tmp_path):
    c,_=make(jpeg,monkeypatch)
    initial=None
    for n in range(8):
        body=c.prepare_decision('fake',tmp_path/str(n))
        if initial is None:initial=copy.deepcopy(body['messages'])
        assert body['messages'][:len(initial)]==initial
        advance(c,jpeg,n+1,tmp_path/str(n),xyz=[5,0,0])
    c.prepare_decision('fake',tmp_path/'8')
    before=copy.deepcopy(c.turns)
    monkeypatch.setattr(conversation,'request',lambda *a,**k:summary_response())
    c.compact('fake',tmp_path/'compact')
    assert c.turns==before and c.start==3
    assert review.monitor(c.turns,c.localizations,c.current_id)['windows'][0]['count']==8
    assert 'MOTION_HYPOTHESIS_REVIEW' in json.dumps(c.messages())
    assert 'raw_locations' not in json.dumps(c.messages())
