"""Current-view localization and framing checks do not generate robot actions."""
import ast
import copy
import json
import threading
from pathlib import Path

import pytest

from pressb.agent_loop import grounding, view_acquisition as v
from pressb.agent_loop.conversation import image_count, live_message
from test_conversation import jpeg, obs, response


def unknown():
    return dict(panel_visible='unknown', button_visibility='unknown', contact_visibility='unknown',
                button_box=None, arrow_box=None, contact_tip=None, note='The objects are not identifiable.')


def located(**changes):
    result = dict(panel_visible='yes', button_visibility='full', contact_visibility='full',
                  button_box=[.3, .1, .4, .2], arrow_box=[.6, .5, .7, .6],
                  contact_tip=[.65, .55], note='The button face and marked physical finger end are visible.')
    result.update(changes)
    return result


def reply(oid='live-0', head=None, wrist=None):
    return response('inspect_view', dict(observation_id=oid,
                    head=unknown() if head is None else head, wrist=located() if wrist is None else wrist))


def test_request_has_only_current_original_dual_view_and_contact_reference(jpeg):
    message = live_message(obs(0), jpeg, jpeg)
    meta = json.loads(message['content'][0]['text'])
    meta.update(demonstration='SECRET_DEMO', reason='SECRET_REASON', prior_review='SECRET_REVIEW',
                command='SECRET_COMMAND', state='SECRET_STATE')
    message['content'][0]['text'] = json.dumps(meta)
    original = copy.deepcopy(message)
    body = v.build_request('press down-call', message)
    assert message == original and image_count(body['messages']) == 2
    assert 'SECRET' not in json.dumps(body)
    assert body['chat_template_kwargs'] == dict(enable_thinking=False)
    assert body['max_tokens'] == 512 and body['temperature'] == 0
    assert [tool['function']['name'] for tool in body['tools']] == ['inspect_view']
    assert 'WHITE/CREAM ARROW' in body['messages'][0]['content']
    with pytest.raises(ValueError, match='contact_reference'):
        v.build_request('press', message, 'imagined_marker')


def test_location_relation_uses_image_y_down_without_robot_mapping():
    result = v.validate(reply(), 'live-0')
    assert result['format'] == v.FORMAT and result['source'] == 'model_not_human'
    assert not result['usable_head']
    relation = result['wrist']['derived_relation']
    assert relation['vertical'] == 'above' and relation['horizontal'] == 'left'
    assert relation['dx'] == pytest.approx(-.3) and relation['dy'] == pytest.approx(-.4)
    assert relation['units'] == 'normalized_image'
    assert not any(key in result for key in ('xyz_mm', 'command', 'next_action'))


def test_unknown_partial_and_boundary_views_do_not_satisfy_framing():
    unknown_result = v.validate(reply(head=unknown(), wrist=unknown()), 'live-0')
    assert not unknown_result['usable_head']
    assert unknown_result['head']['derived_relation']['vertical'] == 'unknown'
    partial = located(button_visibility='partial', button_box=[0, .1, .2, .3])
    result = v.validate(reply(head=partial), 'live-0')
    assert not result['usable_head'] and result['head']['derived_relation']['vertical'] == 'unknown'
    for head in (located(panel_visible='no'), located(button_box=[0, .1, .2, .3]),
                 located(contact_tip=[1, .5])):
        assert not v.validate(reply(head=head), 'live-0')['usable_head']
    assert v.validate(reply(head=located()), 'live-0')['usable_head']


def test_level_and_center_are_only_image_tolerance_not_contact_claims():
    data = located(button_box=[.6, .5, .7, .6], contact_tip=[.65, .55])
    result = v.validate(reply(head=data), 'live-0')['head']['derived_relation']
    assert result['horizontal'] == 'center' and result['vertical'] == 'level'
    assert result['tolerance'] == .01
    below = v.validate(reply(wrist=located(button_box=[.7, .7, .8, .8])), 'live-0')
    assert below['wrist']['derived_relation']['vertical'] == 'below'
    assert below['wrist']['derived_relation']['horizontal'] == 'right'


@pytest.mark.parametrize('change', [
    dict(button_box=[0, .1, 1.1, .2]), dict(button_box=[.3, .1, .2, .2]),
    dict(button_box=[.2, .1, .2, .2]), dict(button_box=[.2, .2, .3]),
    dict(button_box=[.2, float('nan'), .3, .4]), dict(arrow_box=[.1, .1, float('inf'), .3]),
    dict(contact_tip=[True, .5]), dict(contact_tip=[-.1, .5]), dict(contact_tip=[.5, .6, .7]),
    dict(contact_tip=None), dict(button_visibility='not_visible'), dict(contact_visibility='unknown'),
    dict(button_visibility='visible'), dict(panel_visible=True), dict(note=''), dict(extra='field'),
])
def test_invalid_locations_and_visibility_consistency_rejected(change):
    with pytest.raises(ValueError):
        v.validate(reply(head=located(**change)), 'live-0')


def test_wrong_observation_tool_finish_reason_and_prose_rejected():
    with pytest.raises(ValueError, match='observation_mismatch'):
        v.validate(reply(oid='historical'), 'live-0')
    for mutate in (
        lambda r: r['choices'][0].update(finish_reason='length'),
        lambda r: r['choices'][0]['message'].update(content='move upward'),
        lambda r: r['choices'][0]['message']['tool_calls'][0]['function'].update(name='propose_move'),
        lambda r: r['choices'][0]['message']['tool_calls'].append(copy.deepcopy(r['choices'][0]['message']['tool_calls'][0])),
    ):
        candidate = reply(); mutate(candidate)
        with pytest.raises(ValueError):
            v.validate(candidate, 'live-0')


def test_inspection_repairs_until_valid_and_archives_each_attempt(jpeg, tmp_path, monkeypatch):
    body = v.build_request('press', live_message(obs(0), jpeg, jpeg))
    responses = iter([reply(oid='wrong')]*4+[reply()])
    calls = []
    monkeypatch.setattr(grounding, 'request', lambda *args, **kwargs: calls.append(copy.deepcopy(args[2])) or next(responses))
    result = v.inspect('fake', body, 'live-0', tmp_path)
    assert result['observation_id'] == 'live-0' and len(calls) == 5
    assert all(image_count(call['messages']) == 2 for call in calls)
    assert all(len(call['messages']) == 3 for call in calls[1:])
    assert len(list(tmp_path.glob('response-*.json'))) == 5
    assert len(list(tmp_path.glob('repair-request-*.json'))) == 4
    assert json.loads((tmp_path/'metrics.json').read_text())['validation_failures'] == 4
    assert json.loads((tmp_path/'assessment.json').read_text()) == result


def test_request_binding_rejected_before_network(jpeg, tmp_path, monkeypatch):
    body = v.build_request('press', live_message(obs(0), jpeg, jpeg))
    monkeypatch.setattr(grounding, 'request', lambda *a, **k: pytest.fail('network must not be called'))
    with pytest.raises(ValueError, match='request_observation_mismatch'):
        v.inspect('fake', body, 'live-1', tmp_path)
    body['messages'][1]['content'][0]['text'] = '{}'
    with pytest.raises(ValueError, match='contact_reference_required'):
        v.inspect('fake', body, 'live-0', tmp_path)


def test_cancel_before_request_and_late_response_not_committed(jpeg, tmp_path, monkeypatch):
    body = v.build_request('press', live_message(obs(0), jpeg, jpeg))
    cancel = threading.Event(); cancel.set()
    monkeypatch.setattr(grounding, 'request', lambda *a, **k: pytest.fail('network must not be called'))
    with pytest.raises(InterruptedError):
        v.inspect('fake', body, 'live-0', tmp_path/'before', cancel)
    cancel.clear()
    def request(*args, **kwargs):
        cancel.set()
        return reply()
    monkeypatch.setattr(grounding, 'request', request)
    with pytest.raises(InterruptedError):
        v.inspect('fake', body, 'live-0', tmp_path/'late', cancel)
    assert not (tmp_path/'late'/'assessment.json').exists()
    assert json.loads((tmp_path/'late'/'metrics.json').read_text())['status'] == 'interrupted'


def test_module_has_no_robot_or_motion_dependencies():
    source = Path(v.__file__).read_text()
    tree = ast.parse(source)
    imports = [node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(any(word in module for word in ('client', 'executor', 'robot', 'sdk')) for module in imports)
