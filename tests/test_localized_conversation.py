"""Current localized mode leaves visual decisions to the model, not heuristics."""
import copy
import json
import threading

import pytest

from pressb.agent_loop import client, conversation, grounding, localized_conversation as module
from pressb.agent_loop import model_localization as loc, localized_reassessment as review, wrist_overlap_guard as overlap
from pressb.agent_loop.common import SESSION_SCHEMA
from test_conversation import jpeg, obs, response, summary_response
from test_model_localization import locations, view, reply, SIZES


def observation(n):
    result = obs(n)
    result['state']['head_deg'] = [0, n*10]
    return result


def make(jpeg, monkeypatch, data=None, budget=65536):
    calls = []
    def inspect(base, body, oid, directory, cancel=None, revision=1):
        calls.append((oid, str(directory), revision))
        payload = data(oid) if callable(data) else data
        return loc.assess(reply(payload), oid, SIZES, revision)
    monkeypatch.setattr(loc, 'inspect', inspect)
    monkeypatch.setattr(review, 'inspect', lambda base, body, oid, before, directory, cancel=None:
        review.validate(response('review_motion_hypothesis', dict(observed_effect='unknown',
            distance_only_hypothesis='unknown', light_change='unknown', prediction_check='not_supplied', evidence='Objects uncertain.',
            alternative_explanation='Alignment may be incomplete.', next_check='Reacquire both objects.')), before, oid))
    monkeypatch.setattr(conversation, 'tokenize', lambda *a: 1000)
    monkeypatch.setattr(grounding, 'request', lambda *a, **k: pytest.fail('no automatic visual review or forward verifier'))
    c = module.LocalizedConversation('press', observation(0), jpeg, jpeg,
                                     dict(schema=SESSION_SCHEMA), 'prompt', budget=budget)
    return c, calls


def action(c, name='propose_move', xyz=None, **extra):
    args = dict(reason='Decide from the images and actual feedback')
    if name == 'propose_move': args['xyz_mm'] = [0, 0, 5] if xyz is None else xyz
    if name == 'propose_head': args['head_deg'] = [0, 30]
    if name == 'propose_gripper': args['key'] = 'o'
    if name == 'report_done': args = dict(evidence='I observed the button illuminate and then the finger withdraw.')
    args.update(extra)
    if c.current_id in c.hypothesis_reviews and name not in ('report_done', 'report_unable'):
        args['reassessment'] = dict(observation_id=c.current_id, review_id='review-'+c.current_id,
            response_to_review='accept', evidence='Current wrist button is partial; contact uncertain.',
            expected_visual_change='Target and marked finger become visible together.',
            if_no_change='Reconsider the view and contact identity.')
    return response(name, args, 'call-'+c.current_id)


def overlapping(oid):
    return overlap.validate_raw(dict(coordinate_units='normalized_01', button_visibility='full',
        button_box=[.3,.3,.6,.6], contact_visibility='full', contact_tip=[.4,.4],
        note='Synthetic coincident tip and face for control-flow tests only.'), oid, [640,360])


def advance(c, jpeg, n, directory, xyz=None, name='propose_move'):
    r = action(c, name, xyz)
    proposal = c.parse_response(r, observation(n-1), c.session, directory)
    c.accept(r)
    if client.is_forward(proposal):
        # Simulate a current independent check, not a bypass in the live path.
        permission=overlap.evaluate(overlapping(c.current_id),c.current_id)
        permission['authorized_command']=dict(type='move',xyz_mm=proposal['arguments']['xyz_mm'])
        c.forward_guard_decision=permission
    c.check_execution(proposal, c.current_id)
    result = None if name == 'request_observation' else dict(status='completed', sends_completed=True,
        request=dict(type='move', xyz_mm=proposal['arguments']['xyz_mm'], command_id='cmd-'+str(n)),
        motion=dict(action_completed=True), capture=dict(observation_id='live-'+str(n)))
    c.append_result(proposal, result, observation(n), jpeg, jpeg)
    return proposal


@pytest.mark.parametrize('data', [locations(head={}, wrist={}),
    locations(head=view(button_visibility='unknown'), wrist=view(contact_visibility='partial')),
    dict(head='not a view', wrist={})])
def test_all_tools_available_despite_partial_unknown_or_unreadable_localization(jpeg, monkeypatch, tmp_path, data):
    c, calls = make(jpeg, monkeypatch, data)
    body = c.prepare_decision('fake', tmp_path)
    assert {t['function']['name'] for t in body['tools']} == {t['function']['name'] for t in client.TOOLS}
    assert all('visual_basis' not in t['function']['parameters']['properties'] for t in body['tools'])
    for name in ('propose_move', 'propose_head', 'propose_gripper', 'request_observation', 'report_done'):
        p = c.parse_response(action(c, name), observation(0), c.session, tmp_path)
        c.review_proposal('fake', p, tmp_path)
    assert len(calls) == 1
    assert not (tmp_path/'localization_review').exists()
    context = json.loads((tmp_path/'decision_context.json').read_text())
    assert context['view_phase']=='model_decides' and context['execution_policy']=='yz_alignment_before_forward'
    assert context['forward_guard_policy']=='wrist_tip_overlap' and context['format']=='pressb-conversation-v5'


def test_prefix_pairing_and_original_numbers_preserved_without_basis(jpeg, monkeypatch, tmp_path):
    c, calls = make(jpeg, monkeypatch)
    first = c.prepare_decision('fake', tmp_path/'0')
    c.prepare_decision('fake', tmp_path/'0'); assert len(calls)==1
    xyz=[0, -3.123456, 2.987654]
    r=action(c,xyz=xyz);r['choices'][0]['message']['content']='Brief judgment accompanying the command.'
    monkeypatch.setattr(client,'request',lambda *a,**k:r)
    p=client.propose('fake',first,observation(0),c.session,tmp_path/'0',conversation=c)
    assert p['arguments']['xyz_mm']==xyz
    c.check_execution(p,c.current_id)
    c.append_result(p,None,observation(1),jpeg,jpeg)
    second=c.prepare_decision('fake',tmp_path/'1')
    assert second['messages'][:len(first['messages'])]==first['messages']
    assert c.turns[0]['assistant']==r['choices'][0]['message']
    assert c.turns[0]['tool']['tool_call_id']=='call-live-0'
    assert conversation.image_count(second['messages'])==4
    assert c.turns[0]['ledger']['localization_summary']['observation_id']=='live-0'


def test_direction_hint_binds_current_wrist_and_preserves_history(jpeg, monkeypatch, tmp_path):
    def data(oid):
        return locations(head=view(button_box=[.8,.8,.9,.9], contact_tip=[.2,.2]),
                         wrist=view(button_box=[.1,.1,.2,.2], contact_tip=[.7,.7])
                         if oid == 'live-0' else view(button_box=[.8,.8,.9,.9], contact_tip=[.2,.2]))
    c,_=make(jpeg,monkeypatch,data)
    first=c.prepare_decision('fake',tmp_path/'0')
    hint=json.loads((tmp_path/'0'/'wrist_direction_rule.json').read_text())
    assert hint['observation_id']=='live-0' and hint['signs']=={'Y':1,'Z':1}
    assert hint['camera']=='wrist'
    c.prepare_decision('fake',tmp_path/'0')
    assert c.body()['messages']==first['messages']
    advance(c,jpeg,1,tmp_path/'0',name='request_observation')
    second=c.prepare_decision('fake',tmp_path/'1')
    assert second['messages'][:len(first['messages'])]==first['messages']
    assert c.direction_hints['live-1']['signs']=={'Y':-1,'Z':-1}
    assert c.turns[0]['ledger']['wrist_direction_rule']==hint
    contexts=[]
    for message in second['messages']:
        if message['role']=='user' and isinstance(message['content'],list):
            try:
                data=json.loads(message['content'][0].get('text',''))
            except (ValueError,IndexError):
                continue
            if data.get('context_type')=='CURRENT_WRIST_ALIGNMENT_DIRECTION_RULE':
                contexts.append(data)
    assert [item['observation_id'] for item in contexts]==['live-0','live-1']
    assert contexts[-1]['signs']=={'Y':-1,'Z':-1}


def test_encoded_xyz_and_old_basis_do_not_request_model_repair(jpeg, monkeypatch, tmp_path):
    c,_=make(jpeg,monkeypatch,locations(head={},wrist={}))
    body=c.prepare_decision('fake',tmp_path)
    r=action(c,xyz='[0, 20, 0]',visual_basis='{"view":"wrist","observation_id":"old"}',extra='narrative')
    requests=[]
    monkeypatch.setattr(client,'request',lambda *a,**k:requests.append(a[2]) or r)
    p=client.propose('fake',body,observation(0),c.session,tmp_path,conversation=c)
    assert len(requests)==1 and p['arguments']['xyz_mm']==[0,20,0]
    assert 'visual_basis' not in p['arguments'] and 'extra' not in p['arguments']
    c.check_execution(p,'live-0');c.append_result(p,None,observation(1),jpeg,jpeg)
    assert c.turns[0]['assistant']==r['choices'][0]['message']
    assert not (tmp_path/'repair-request.json').exists()


def test_positive_X_reaches_existing_HTTP_path_only_with_current_overlap_no_cap(jpeg, monkeypatch, tmp_path):
    from test_forward_guard_execution import robot
    c,_=make(jpeg,monkeypatch,locations(head={},wrist={}))
    body=c.prepare_decision('fake',tmp_path/'decision')
    xyz=[7.123456,-2.7182818,1.4142135];r=action(c,xyz=xyz)
    monkeypatch.setattr(client,'request',lambda *a,**k:r)
    monkeypatch.setattr(overlap,'inspect',lambda base,body,oid,directory,cancel=None:overlapping(oid))
    p=client.propose('fake',body,observation(0),c.session,tmp_path/'decision',conversation=c)
    assert c.forward_guard_decision['allowed'] is True
    assert (tmp_path/'decision/forward_guard/decision.json').exists()
    c.check_execution(p,'live-0')
    executor,submitted=robot(tmp_path/'robot',monkeypatch)
    executor.forward_guard_policy='wrist_tip_overlap'
    executor.execute(p,observation(0),forward_check=c.forward_guard_decision)
    assert len(submitted)==1 and submitted[0]['xyz_mm']==xyz
    assert set(submitted[0])=={'session_id','command_id','observation_id','type','xyz_mm'}
    executor.cancel.set()
    with pytest.raises(InterruptedError):executor.execute(p,observation(0))
    assert len(submitted)==1


def test_repetition_with_unknown_geometry_triggers_review_not_motion_replacement(jpeg, monkeypatch, tmp_path):
    def data(oid):
        n=int(oid.split('-')[1]);w=.04 if n%2 else .2
        return locations(head=view(button_box=[.1,.1,.1+w,.4]),
                         wrist=view(button_visibility='partial',contact_visibility='unknown',
                                    button_box=[.1,.1,.1+w,.4]))
    c,calls=make(jpeg,monkeypatch,data)
    for n in range(6):
        c.prepare_decision('fake',tmp_path/str(n))
        advance(c,jpeg,n+1,tmp_path/str(n),xyz=[0,20,0])
    assert len(calls)==6
    progress=c.localized_progress['live-5']
    assert progress['head_delta_deg']==[0,10]
    assert progress['views']['wrist']['current_minus_previous']['box_size'] is not None
    assert not any(k in json.dumps(progress) for k in ['needs_recheck','comparable','repeat_limit','threshold'])
    assert not list(tmp_path.rglob('localization_review'))
    assert set(c.hypothesis_reviews) == {'live-3', 'live-5'}
    assert all(t['ledger']['result']['command']['xyz_mm'] == [0,20,0] for t in c.turns)


def test_stopped_or_stale_response_still_cannot_execute(jpeg, monkeypatch, tmp_path):
    c,_=make(jpeg,monkeypatch);cancel=threading.Event()
    def late(*a,**k):cancel.set();return loc.assess(reply(),'live-0',SIZES)
    monkeypatch.setattr(loc,'inspect',late)
    with pytest.raises(InterruptedError):c.prepare_decision('fake',tmp_path/'loc',cancel)
    assert not c.localizations and c.pending is None
    c,_=make(jpeg,monkeypatch);cancel.clear();body=c.prepare_decision('fake',tmp_path/'action')
    def late_action(*a,**k):cancel.set();return action(c)
    monkeypatch.setattr(client,'request',late_action)
    with pytest.raises(InterruptedError):
        client.propose('fake',body,observation(0),c.session,tmp_path/'action',cancel=cancel,conversation=c)
    assert c.pending is None
    with pytest.raises(ValueError,match='observation_mismatch'):c.check_observation('old')


def test_done_is_model_judgment_without_independent_verifier(jpeg, monkeypatch, tmp_path):
    c,_=make(jpeg,monkeypatch,locations(head={},wrist={}))
    body=c.prepare_decision('fake',tmp_path)
    r=action(c,'report_done');requests=[]
    monkeypatch.setattr(client,'request',lambda *a,**k:requests.append(a[2]) or r)
    p=client.propose('fake',body,observation(0),c.session,tmp_path,conversation=c)
    c.check_execution(p,'live-0')
    assert len(requests)==1 and not (tmp_path/'activation').exists() and not (tmp_path/'withdrawal').exists()


def test_compaction_and_recorded_actual_feedback_are_preserved(jpeg, monkeypatch, tmp_path):
    c,_=make(jpeg,monkeypatch)
    for n in range(8):
        c.prepare_decision('fake',tmp_path/str(n));advance(c,jpeg,n+1,tmp_path/str(n),name='request_observation')
        c.turns[-1]['ledger']['result'].update(command=dict(type='move',xyz_mm=[9,0,0]),
            replay=dict(recorded_actual_command=dict(type='move',xyz_mm=[0,0,10]),proposal_executed=False))
    c.prepare_decision('fake',tmp_path/'8')
    assert c.localized_progress['live-8']['actual_commands']==[dict(type='move',xyz_mm=[0,0,10])]
    before=copy.deepcopy(c.turns)
    monkeypatch.setattr(conversation,'request',lambda *a,**k:summary_response())
    c.compact('fake',tmp_path/'compact')
    assert c.turns==before and c.start==3
    assert len([m for m in c.messages() if m['role']=='assistant'])==5
    assert 'EARLIER_LOCALIZATION_THIS_RUN_NOT_CURRENT' in json.dumps(c.messages())


@pytest.mark.parametrize('xyz', [[1,2], [1,float('nan'),3], 'not JSON'])
def test_only_unexecutable_command_shapes_still_need_repair(jpeg, monkeypatch, tmp_path, xyz):
    c,_=make(jpeg,monkeypatch);c.prepare_decision('fake',tmp_path)
    with pytest.raises((ValueError,TypeError)):
        c.parse_response(action(c,xyz=xyz),observation(0),c.session,tmp_path)


def prepare_compaction(c, jpeg, tmp_path):
    for n in range(8):
        c.prepare_decision('fake',tmp_path/str(n))
        advance(c,jpeg,n+1,tmp_path/str(n),name='request_observation')
    c.prepare_decision('fake',tmp_path/'8')


@pytest.mark.parametrize('kind', ['extra_and_empty', 'prose', 'broken_json', 'bad_structure'])
def test_compaction_accepts_one_advisory_response_without_format_retries(jpeg, monkeypatch, tmp_path, kind):
    c,_=make(jpeg,monkeypatch)
    prepare_compaction(c,jpeg,tmp_path)
    answer=summary_response(progress='',uncertainties='',extra=dict(model_note='keep raw value'))
    message=answer['choices'][0]['message']
    message['content']='Additional model judgment, not operator confirmation.'
    if kind=='prose':
        message['tool_calls']=[]
        answer['choices'][0]['finish_reason']='stop'
    elif kind=='broken_json':
        message['tool_calls'][0]['function']['arguments']='{"progress":'
    elif kind=='bad_structure':
        answer['choices']=[dict(finish_reason='length',message=['unexpected','shape'])]
    calls=[]
    monkeypatch.setattr(conversation,'request',lambda *a,**k:calls.append(a[2]) or answer)
    c.compact('fake',tmp_path/'compact')
    assert len(calls)==1 and c.start==3 and c.generation==1
    assert not list((tmp_path/'compact').glob('repair*'))
    summary=json.loads((tmp_path/'compact/result.json').read_text())['summary']
    assert summary['source']=='model_not_human'
    assert summary['raw_output']
    if kind=='extra_and_empty':
        assert summary['progress']==summary['uncertainties']==''
        assert summary['raw_summary']['extra']==dict(model_note='keep raw value')
    elif kind=='bad_structure':
        assert summary['progress']=='unknown' and summary['parse_notes']
    else:
        assert summary['progress']==message['content']
    assert len([m for m in c.messages() if m['role']=='assistant'])==5
    metrics=json.loads((tmp_path/'compact/metrics.json').read_text())
    assert metrics['attempts']==1 and metrics['validation_failures']==0


def test_compaction_preserves_wrong_order_as_model_claim_without_semantic_repair(jpeg, monkeypatch, tmp_path):
    c,_=make(jpeg,monkeypatch)
    prepare_compaction(c,jpeg,tmp_path)
    answer=summary_response(
        initial_light=dict(observation_id='live-2',assessment='Model calls this the initial lamp.'),
        light_on=dict(observation_id='live-2',assessment='Model says it is lit here.'),
        withdrawal=dict(observation_id='live-1',assessment='Model claims withdrawal earlier.'))
    calls=[]
    monkeypatch.setattr(conversation,'request',lambda *a,**k:calls.append(a[2]) or answer)
    c.compact('fake',tmp_path/'compact')
    result=json.loads((tmp_path/'compact/result.json').read_text())
    assert len(calls)==1 and result['pinned_observation_ids']==['live-2','live-1']
    assert result['summary']['light_on']['observation_id']=='live-2'
    assert result['summary']['withdrawal']['observation_id']=='live-1'
    assert result['summary']['source']=='model_not_human'


def test_compaction_never_pins_missing_or_bad_model_references_or_invents_replacements(jpeg, monkeypatch, tmp_path):
    c,_=make(jpeg,monkeypatch)
    prepare_compaction(c,jpeg,tmp_path)
    answer=summary_response(
        initial_light=dict(observation_id='HISTORICAL-DEMO-STEP-10',assessment='Model cites a demonstration.'),
        light_on=dict(observation_id=['live-2'],assessment=None),
        withdrawal='a narrative instead of an evidence object')
    raw=json.loads(answer['choices'][0]['message']['tool_calls'][0]['function']['arguments'])
    calls=[]
    monkeypatch.setattr(conversation,'request',lambda *a,**k:calls.append(a[2]) or answer)
    c.compact('fake',tmp_path/'compact')
    result=json.loads((tmp_path/'compact/result.json').read_text())
    assert len(calls)==1 and result['pinned_observation_ids']==[]
    summary=result['summary']
    assert summary['raw_summary']==raw
    assert all(summary[field]['observation_id'] is None for field in conversation.EVIDENCE_FIELDS)
    assert summary['initial_light']['model_observation_id']=='HISTORICAL-DEMO-STEP-10'
    assert summary['light_on']['model_observation_id']==['live-2']
    assert summary['light_on']['assessment']=='unknown'
    assert summary['parse_notes']
    assert not any(json.loads(m['content'][0]['text']).get('context_type')=='EARLIER_OBSERVATION_THIS_RUN_NOT_CURRENT'
                   for m in c.memory if isinstance(m.get('content'),list))


def test_legacy_summary_validation_keeps_its_existing_behavior(jpeg):
    c=conversation.Conversation('press',observation(0),jpeg,jpeg,dict(schema=SESSION_SCHEMA),'prompt')
    with pytest.raises(ValueError,match='invalid_summary_fields'):
        c.validate_summary(summary_response(extra='legacy still strict'))
