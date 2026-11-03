import copy
import hashlib
import json
import threading

import pytest

from pressb.agent_loop import grounding, spatial_localization as loc
from pressb.agent_loop.conversation import image_count, live_message
from test_conversation import jpeg, obs, response


SIZES = dict(head=[640, 360], wrist=[1280, 720])


def unknown():
    return dict(button_visibility='unknown', contact_visibility='unknown', button_box=None,
                arrow_box=None, contact_tip=None, note='Cannot identify the target or indicated end.')


def located(**changes):
    value = dict(button_visibility='full', contact_visibility='full', button_box=[.3, .1, .4, .2],
                 arrow_box=[.6, .5, .7, .6], contact_tip=[.65, .55], note='Button face and marked physical end.')
    value.update(changes)
    return value


def raw(oid='live-0', head=None, wrist=None):
    return dict(observation_id=oid, head=unknown() if head is None else head,
                wrist=located() if wrist is None else wrist)


def reply(**kwargs):
    return response(loc.TOOL_NAME, raw(**kwargs))


def test_request_is_current_dual_view_only(jpeg):
    message = live_message(obs(0), jpeg, jpeg)
    meta = json.loads(message['content'][0]['text'])
    meta.update(demo='SECRET_DEMO', prior='SECRET_OLD', reason='SECRET_REASON', state='SECRET_STATE')
    message['content'][0]['text'] = json.dumps(meta)
    original = copy.deepcopy(message)
    body = loc.build_request('press down-call', message)
    assert image_count(body['messages']) == 2 and message == original
    assert 'SECRET' not in json.dumps(body)
    assert body['temperature'] == 0 and body['max_tokens'] == 512
    assert body['chat_template_kwargs'] == dict(enable_thinking=False)
    assert loc.PROPERTIES['head']['properties'] == loc.VIEW_PROPERTIES
    assert not any(key in loc.VIEW_PROPERTIES for key in ('horizontal', 'vertical', 'progress', 'xyz_mm'))


def test_views_independent_and_wrist_only_is_usable():
    result = loc.validate(reply(), 'live-0', SIZES, revision=3)
    assert result['revision'] == 3 and result['usable_views'] == ['wrist']
    assert result['head']['derived_relation']['dx'] is None
    wrist = result['wrist']['derived_relation']
    assert wrist['horizontal'] == 'left' and wrist['vertical'] == 'above'
    assert wrist['pixel_delta'] == pytest.approx([-384, -288])
    assert wrist['error_button_units'] == pytest.approx([-3, -4])
    assert not any(key in result for key in ('xyz_mm', 'command', 'step_hint_mm'))
    both = loc.validate(reply(head=located(button_box=[.7, .7, .8, .8], contact_tip=[.1, .1])), 'live-0', SIZES)
    assert both['head']['derived_relation']['horizontal'] == 'right'
    assert both['wrist']['derived_relation']['horizontal'] == 'left'
    assert both['head']['image_size'] == [640, 360]


def test_partial_visible_box_never_becomes_full_button_center():
    result = loc.validate(reply(wrist=located(button_visibility='partial', button_box=[0, .1, .2, .2])), 'live-0', SIZES)
    assert result['usable_views'] == []
    assert result['wrist']['button_box'] == [0, .1, .2, .2]
    assert result['wrist']['derived_relation']['error_button_units'] is None


def test_one_percent_boundary_is_inclusive_and_is_projection_only():
    result = loc.validate(reply(wrist=located(button_box=[.49, .49, .53, .53], contact_tip=[.5, .5])), 'live-0', SIZES)
    relation = result['wrist']['derived_relation']
    assert relation['horizontal'] == 'center' and relation['vertical'] == 'level'
    assert relation['tolerance'] == .01
    assert relation['pixel_delta'] == pytest.approx([12.8, 7.2])
    assert relation['source'] == 'program_from_model_coordinates'


@pytest.mark.parametrize('changes', [
    dict(button_box=[.1, .1, .1, .2]), dict(button_box=[.4, .1, .3, .2]),
    dict(button_box=[0, 0, 1.1, .2]), dict(button_box=[0, 0, float('nan'), .2]),
    dict(contact_tip=[True, .3]), dict(contact_tip=[.3, float('inf')]),
    dict(contact_visibility='unknown'), dict(button_box=None), dict(extra='field'),
])
def test_bad_coordinates_and_inconsistent_visibility_rejected(changes):
    with pytest.raises(ValueError):
        loc.validate(reply(wrist=located(**changes)), 'live-0', SIZES)


def test_bad_binding_schema_and_duplicate_fields_rejected():
    with pytest.raises(ValueError, match='observation_mismatch'):
        loc.validate(reply(oid='old'), 'live-0', SIZES)
    candidate = reply()
    candidate['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = '{"observation_id":"live-0","observation_id":"live-0"}'
    with pytest.raises(ValueError, match='duplicate'):
        loc.validate(candidate, 'live-0', SIZES)
    for mutate in (lambda r:r['choices'][0].update(finish_reason='length'),
                   lambda r:r['choices'][0]['message'].update(content='above'),
                   lambda r:r['choices'][0]['message']['tool_calls'][0]['function'].update(name='propose_move')):
        candidate = reply(); mutate(candidate)
        with pytest.raises(ValueError):
            loc.validate(candidate, 'live-0', SIZES)
    with pytest.raises(ValueError, match='revision'):
        loc.validate(reply(), 'live-0', SIZES, revision=True)


def test_unlimited_format_repairs_archive_exact_images_and_unknown_is_valid(jpeg, tmp_path, monkeypatch):
    body = loc.build_request('press', live_message(obs(0), jpeg, jpeg))
    responses = iter([reply(oid='wrong')]*4+[reply(wrist=unknown())])
    calls = []
    def request(*args, **kwargs):
        calls.append(copy.deepcopy(args[2])); assert kwargs['timeout'] == 120
        return next(responses)
    monkeypatch.setattr(grounding, 'request', request)
    result = loc.inspect('fake', body, 'live-0', tmp_path, revision=2)
    assert result['revision'] == 2 and result['usable_views'] == []
    assert len(calls) == 5 and all(image_count(call['messages']) == 2 for call in calls)
    assert all(len(call['messages']) == 3 for call in calls[1:])
    assert result['images']['wrist']['sha256'] == hashlib.sha256(jpeg).hexdigest()
    assert result['wrist']['image_size'] == [640, 360]
    assert json.loads((tmp_path/'assessment.json').read_text()) == result
    assert json.loads((tmp_path/'metrics.json').read_text())['validation_failures'] == 4


def test_wrong_request_rejected_and_stop_blocks_late_result(jpeg, tmp_path, monkeypatch):
    body = loc.build_request('press', live_message(obs(0), jpeg, jpeg))
    monkeypatch.setattr(grounding, 'request', lambda *a, **k: pytest.fail('must not request'))
    with pytest.raises(ValueError, match='request_observation_mismatch'):
        loc.inspect('fake', body, 'live-1', tmp_path/'wrong')
    cancel = threading.Event(); cancel.set()
    with pytest.raises(InterruptedError):
        loc.inspect('fake', body, 'live-0', tmp_path/'before', cancel)
    cancel.clear()
    def late(*args, **kwargs):
        cancel.set(); return reply()
    monkeypatch.setattr(grounding, 'request', late)
    with pytest.raises(InterruptedError):
        loc.inspect('fake', body, 'live-0', tmp_path/'late', cancel)
    assert not (tmp_path/'late'/'assessment.json').exists()


def test_transport_failure_does_not_repair(jpeg, tmp_path, monkeypatch):
    body = loc.build_request('press', live_message(obs(0), jpeg, jpeg))
    calls = []
    def fail(*args, **kwargs):
        calls.append(args); raise OSError('service unavailable')
    monkeypatch.setattr(grounding, 'request', fail)
    with pytest.raises(OSError):
        loc.inspect('fake', body, 'live-0', tmp_path)
    assert len(calls) == 1 and not (tmp_path/'assessment.json').exists()
