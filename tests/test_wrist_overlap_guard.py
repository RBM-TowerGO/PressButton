import copy
import io
import json
import threading

from PIL import Image
import pytest

from pressb.agent_loop import grounding, wrist_overlap_guard as guard
from pressb.agent_loop.conversation import image_count, live_message
from test_conversation import jpeg, obs, response


def raw(**changes):
    value = dict(coordinate_units='normalized_1000', button_visibility='full',
        button_box=[400, 300, 600, 500], contact_visibility='full',
        contact_tip=[500, 400], note='Button face and arrow-marked physical finger end are identifiable.')
    value.update(changes)
    return value


def reply(**changes):
    return response(guard.TOOL_NAME, raw(**changes))


def permission(**changes):
    return guard.evaluate(guard.validate(reply(**changes), 'live-0'), 'live-0')


def proposal(xyz):
    return dict(name='propose_move', arguments=dict(xyz_mm=xyz, reason='Image-based adjustment.'))


def test_current_wrist_only_original_image_and_no_prior_claims(jpeg):
    stream = io.BytesIO()
    Image.new('RGB', (640, 360), 'red').save(stream, 'JPEG')
    current = live_message(obs(0), jpeg, stream.getvalue())
    meta = json.loads(current['content'][0]['text'])
    meta.update(reason='SECRET_REASON', action='SECRET_ACTION', assessment='SECRET_ASSESSMENT', demo='SECRET_DEMO')
    current['content'][0]['text'] = json.dumps(meta)
    body = guard.build_request('Press the down-call button.', current)
    assert image_count(body['messages']) == 1
    assert 'SECRET' not in json.dumps(body)
    pictures = [part for part in current['content'] if part['type'] == 'image_url']
    assert body['messages'][1]['content'][-1] == pictures[1]
    assert body['messages'][1]['content'][-1] != pictures[0]
    properties = body['tools'][0]['function']['parameters']['properties']
    assert 'tip_inside_button' not in properties and 'xyz_mm' not in properties
    assert body['max_tokens'] == 512 and body['temperature'] == 0
    assert body['chat_template_kwargs'] == {'enable_thinking': False}


@pytest.mark.parametrize('tip,allowed', [([500,400],True), ([400,300],True), ([600,500],True),
    ([399,400],False), ([601,400],False), ([500,299],False), ([500,501],False)])
def test_requires_both_axes_inclusive_boundaries(tip, allowed):
    decision = permission(contact_tip=tip)
    assert decision['allowed'] is allowed
    assert decision['basis']['tip_inside_button'] == ('yes' if allowed else 'no')
    assert decision['geometry']['containment'] is allowed
    assert decision['policy'] == 'wrist_tip_overlap'
    assert decision['localization_format'] == 'pressb-wrist-tip-overlap-v1'


def test_explicit_units_equivalent_and_raw_coordinates_preserved():
    a = guard.validate(reply(), 'live-0', [640,360])
    b = guard.validate(reply(coordinate_units='normalized_01', button_box=[.4,.3,.6,.5],
                             contact_tip=[.5,.4]), 'live-0', [640,360])
    assert a['button_box'] == b['button_box'] == [.4,.3,.6,.5]
    assert a['contact_tip'] == b['contact_tip'] == [.5,.4]
    assert a['geometry'] == b['geometry']
    assert a['coordinate_conversion']['divisor'] == 1000
    assert a['raw_locations']['contact_tip'] == [500,400]
    assert b['raw_locations']['contact_tip'] == [.5,.4]
    assert guard.evaluate(a,'live-0')['allowed'] and guard.evaluate(b,'live-0')['allowed']


@pytest.mark.parametrize('changes', [dict(coordinate_units=None), dict(coordinate_units='pixels'),
    dict(coordinate_units=['normalized_01']), dict(coordinate_units='normalized_01'),
    dict(button_box=[400,300,1001,500]), dict(contact_tip=[-1,400]),
    dict(contact_tip=[500,1001]), dict(button_box=[600,300,400,500]),
    dict(button_box=[400,300,400,500]), dict(button_box=[400,300,600,300]),
    dict(button_box=[400,300,float('nan'),500]), dict(contact_tip=[float('inf'),400]),
    dict(button_box=None), dict(contact_tip=None)])
def test_unusable_units_and_geometry_fail_closed_without_clipping(changes):
    value = guard.validate(reply(**changes),'live-0')
    assert not guard.evaluate(value,'live-0')['allowed']
    assert value['geometry']['containment'] is None
    assert value['tip_inside_button'] == 'unknown'


@pytest.mark.parametrize('changes', [dict(button_visibility='partial'), dict(contact_visibility='partial'),
    dict(button_visibility='not_visible',button_box=None),
    dict(contact_visibility='unknown',contact_tip=None),
    dict(button_visibility='unknown'), dict(contact_visibility='not_visible')])
def test_unknown_or_partial_is_valid_denied_assessment(changes):
    decision = permission(**changes)
    assert not decision['allowed']
    assert decision['basis']['tip_inside_button'] == 'unknown'


@pytest.mark.parametrize('changes', [dict(button_box='[400,300,600,500]'), dict(button_box=[400,300]),
    dict(contact_tip=[True,400]), dict(contact_tip=['500',400]), dict(note=''),
    dict(button_visibility='yes'), dict(extra='unrequested')])
def test_wrong_output_schema_requires_repair(changes):
    with pytest.raises(ValueError):
        guard.validate(reply(**changes),'live-0')


def test_missing_units_wrong_tool_prose_and_duplicate_fields_rejected():
    missing = raw(); del missing['coordinate_units']
    wrong = response('propose_move',raw())
    prose = reply(); prose['choices'][0]['message']['content'] = 'Already aligned.'
    duplicate = reply()
    duplicate['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = (
        json.dumps(raw())[:-1]+',"coordinate_units":"normalized_01"}')
    for value in (response(guard.TOOL_NAME,missing),wrong,prose,duplicate):
        with pytest.raises(ValueError):
            guard.validate(value,'live-0')


def test_stale_or_forged_assessment_is_not_permission():
    assessment = guard.validate(reply(contact_tip=[700,400]),'live-0')
    assessment['tip_inside_button'] = 'yes'
    assert not guard.evaluate(assessment,'live-0')['allowed']
    valid = guard.validate(reply(),'live-0')
    assert not guard.evaluate(valid,'live-1')['allowed']
    for field, value in [('contact_tip',[.9,.4]),('button_box',[.2,.2,.7,.7]),
                          ('source','human'),('format','pressb-forward-alignment-v1')]:
        bad = copy.deepcopy(valid); bad[field] = value
        assert not guard.evaluate(bad,'live-0')['allowed']


@pytest.mark.parametrize('xyz', [[1,0,0],[100,2,-3],[8,-20,5]])
def test_exact_authorized_command_allows_any_original_positive_x_magnitude(xyz):
    decision = permission()
    decision['authorized_command'] = dict(type='move',xyz_mm=copy.deepcopy(xyz))
    move = proposal(xyz)
    original = copy.deepcopy(move)
    guard.validate_permission(move,obs(0),decision)
    assert move == original


def test_stale_changed_or_forged_permission_rejected_at_send_boundary():
    decision = permission()
    decision['authorized_command'] = dict(type='move',xyz_mm=[50,0,0])
    with pytest.raises(ValueError,match='current_wrist_tip_overlap'):
        guard.validate_permission(proposal([50,0,0]),obs(1),decision)
    with pytest.raises(ValueError,match='current_wrist_tip_overlap'):
        guard.validate_permission(proposal([51,0,0]),obs(0),decision)
    for field, value in [('allowed',False),('policy','wrist_horizontal'),('basis',{}),
                          ('geometry',{}),('coordinate_evidence',{})]:
        bad = copy.deepcopy(decision); bad[field] = value
        with pytest.raises(ValueError,match='current_wrist_tip_overlap'):
            guard.validate_permission(proposal([50,0,0]),obs(0),bad)
    bad = copy.deepcopy(decision)
    bad['coordinate_evidence']['raw_locations']['contact_tip'] = [700,400]
    with pytest.raises(ValueError,match='current_wrist_tip_overlap'):
        guard.validate_permission(proposal([50,0,0]),obs(0),bad)


@pytest.mark.parametrize('xyz', [[-100,0,0],[0,20,-20]])
def test_retraction_and_planar_motion_do_not_require_visual_permission(xyz):
    guard.validate_permission(proposal(xyz),obs(0),None)
    guard.validate_permission(dict(name='propose_head',arguments=dict(head_deg=[0,30])),obs(0),None)


def test_unlimited_schema_repair_unknown_valid_and_raw_archive(jpeg,tmp_path,monkeypatch):
    body = guard.build_request('press',live_message(obs(0),jpeg,jpeg))
    invalid = reply(); invalid['choices'][0]['finish_reason'] = 'length'
    replies = iter([invalid]*4+[reply(contact_visibility='unknown',contact_tip=None)])
    monkeypatch.setattr(grounding,'request',lambda *a,**k:next(replies))
    assessment = guard.inspect('fake',body,'live-0',tmp_path)
    assert not guard.evaluate(assessment,'live-0')['allowed']
    assert assessment['image_size'] == [640,360]
    assert list(assessment['images']) == ['wrist']
    assert assessment['raw_locations']['button_box'] == [400,300,600,500]
    metrics = json.loads((tmp_path/'metrics.json').read_text())
    assert metrics['image_count'] == 1 and metrics['attempts'] == 5
    assert metrics['validation_failures'] == 4
    assert (tmp_path/'repair-request-4.json').exists()
    assert (tmp_path/'assessment.json').exists()


def test_unknown_unit_is_archived_without_guess_or_repeat(jpeg,tmp_path,monkeypatch):
    calls = []
    monkeypatch.setattr(grounding,'request',lambda *a,**k:calls.append(1) or reply(coordinate_units='pixels'))
    body = guard.build_request('press',live_message(obs(0),jpeg,jpeg))
    value = guard.inspect('fake',body,'live-0',tmp_path)
    assert len(calls) == 1
    assert value['coordinate_conversion']['status'] == 'invalid_units'
    assert not guard.evaluate(value,'live-0')['allowed']


def test_late_response_after_stop_and_stop_during_repairs_never_authorize(jpeg,tmp_path,monkeypatch):
    body = guard.build_request('press',live_message(obs(0),jpeg,jpeg))
    for variant in ('valid','invalid'):
        cancel = threading.Event()
        def late(*a,**k):
            cancel.set()
            value = reply()
            if variant == 'invalid':
                value['choices'][0]['finish_reason'] = 'length'
            return value
        monkeypatch.setattr(grounding,'request',late)
        directory = tmp_path/variant
        with pytest.raises(InterruptedError):
            guard.inspect('fake',body,'live-0',directory,cancel)
        assert not (directory/'assessment.json').exists()
        assert json.loads((directory/'metrics.json').read_text())['status'] == 'interrupted'


def test_stopped_and_mismatched_observation_do_not_issue_request(jpeg,tmp_path,monkeypatch):
    body = guard.build_request('press',live_message(obs(0),jpeg,jpeg))
    monkeypatch.setattr(grounding,'request',lambda *a,**k:pytest.fail('no model request'))
    with pytest.raises(ValueError,match='observation_mismatch'):
        guard.inspect('fake',body,'live-9',tmp_path/'wrong')
    cancel = threading.Event(); cancel.set()
    with pytest.raises(InterruptedError):
        guard.inspect('fake',body,'live-0',tmp_path/'stopped',cancel)
