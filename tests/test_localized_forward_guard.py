import copy
import json
import threading

import pytest

from pressb.agent_loop import grounding, localized_forward_guard as guard
from pressb.agent_loop.client import require_forward_permission
from pressb.agent_loop.conversation import image_count, live_message
from test_conversation import jpeg, obs, response


def raw(**changes):
    value = dict(button_visibility='full', button_box=[.4,.05,.6,.15],
                 finger_a_visibility='full', finger_a_tip=[.2,.8],
                 finger_b_visibility='full', finger_b_tip=[.8,.9], note='Button face and distinct front ends.')
    value.update(changes); return value


def reply(**changes):
    return response(guard.TOOL_NAME, raw(**changes))


def test_independent_wrist_only_locations_not_containment_guess(jpeg):
    current = live_message(obs(0), jpeg, jpeg)
    meta = json.loads(current['content'][0]['text']); meta.update(reason='SECRET_REASON', action='SECRET_ACTION')
    current['content'][0]['text'] = json.dumps(meta)
    body = guard.build_request('press', current)
    assert image_count(body['messages']) == 1
    assert 'SECRET' not in json.dumps(body)
    assert 'button_between_fingers' not in body['tools'][0]['function']['parameters']['properties']


@pytest.mark.parametrize('center,allowed', [(.2,True),(.5,True),(.8,True),(.19,False),(.81,False)])
def test_horizontal_interval_includes_boundaries_and_ignores_height(center, allowed):
    assessment = guard.validate(reply(button_box=[center-.01,.01,center+.01,.03]), 'live-0')
    decision = guard.evaluate(assessment, 'live-0')
    assert decision['allowed'] is allowed
    assert decision['geometry']['button_center'][1] == pytest.approx(.02)
    assert assessment['format'] == 'pressb-forward-localization-v1'
    assert decision['policy'] == 'wrist_horizontal'


def test_finger_order_does_not_change_containment_and_open_single_side_is_allowed():
    assessment = guard.validate(reply(finger_a_tip=[.8,.9],finger_b_tip=[.2,.8],button_box=[.79,.01,.81,.03]),'live-0')
    decision = guard.evaluate(assessment,'live-0')
    proposal = dict(name='propose_move',arguments=dict(xyz_mm=[5,2,-3],reason='single-side press'))
    decision['authorized_command'] = dict(type='move',xyz_mm=[5,2,-3])
    require_forward_permission(proposal,obs(0),decision)
    assert proposal['arguments']['xyz_mm'] == [5,2,-3]


@pytest.mark.parametrize('changes',[
    dict(button_visibility='partial'),
    dict(finger_a_visibility='unknown',finger_a_tip=None),
    dict(finger_b_visibility='not_visible',finger_b_tip=None),
    dict(finger_b_tip=[.2,.8]),
])
def test_missing_or_partial_coordinates_do_not_authorize_forward(changes):
    assessment = guard.validate(reply(**changes),'live-0')
    assert not guard.evaluate(assessment,'live-0')['allowed']


def test_stale_or_forged_boolean_cannot_override_coordinates():
    assessment=guard.validate(reply(button_box=[.88,.1,.92,.2]),'live-0')
    assert not guard.evaluate(assessment,'live-1')['allowed']
    assessment['button_between_fingers']='yes'
    assert not guard.evaluate(assessment,'live-0')['allowed']
    assessment['format']='pressb-forward-alignment-v1'
    assert not guard.evaluate(assessment,'live-0')['allowed']


@pytest.mark.parametrize('changes',[
    dict(button_box=[.2,.1,.1,.2]),dict(button_box=[0,0,float('nan'),.2]),
    dict(finger_a_tip=[True,.5]),dict(finger_b_tip=[1.1,.3]),
    dict(finger_a_visibility='unknown'),dict(extra='field'),
])
def test_bad_localization_rejected(changes):
    with pytest.raises(ValueError): guard.validate(reply(**changes),'live-0')


def test_archive_repair_unknown_and_late_stop(jpeg,tmp_path,monkeypatch):
    body=guard.build_request('press',live_message(obs(0),jpeg,jpeg))
    invalid=reply();invalid['choices'][0]['finish_reason']='length'
    replies=iter([invalid]*3+[reply(finger_b_visibility='unknown',finger_b_tip=None)])
    monkeypatch.setattr(grounding,'request',lambda *a,**k: next(replies))
    result=guard.inspect('fake',body,'live-0',tmp_path/'repaired')
    assert not guard.evaluate(result,'live-0')['allowed']
    assert result['image_size']==[640,360]
    assert json.loads((tmp_path/'repaired/metrics.json').read_text())['validation_failures']==3
    cancel=threading.Event()
    def late(*a,**k):cancel.set();return reply()
    monkeypatch.setattr(grounding,'request',late)
    with pytest.raises(InterruptedError):guard.inspect('fake',body,'live-0',tmp_path/'late',cancel)
    assert not (tmp_path/'late/assessment.json').exists()


def test_mismatched_observation_is_rejected_before_network(jpeg,tmp_path,monkeypatch):
    body=guard.build_request('press',live_message(obs(0),jpeg,jpeg))
    monkeypatch.setattr(grounding,'request',lambda *a,**k:pytest.fail('no model request'))
    with pytest.raises(ValueError,match='observation_mismatch'):
        guard.inspect('fake',body,'live-9',tmp_path)
