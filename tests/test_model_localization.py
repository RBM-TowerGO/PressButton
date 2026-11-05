import copy
import hashlib
import json
import threading

import pytest

from pressb.agent_loop import model_localization as loc
from pressb.agent_loop.conversation import image_count, live_message
from test_conversation import jpeg, obs, response


SIZES = dict(head=[640, 360], wrist=[1280, 720])


def view(**changes):
    result = dict(button_visibility='full', contact_visibility='full', button_box=[.3, .1, .4, .2],
                  arrow_box=[.6, .5, .7, .6], contact_tip=[.65, .55], note='Face and marked physical end.')
    result.update(changes)
    return result


def locations(**changes):
    result = dict(coordinate_units='normalized_01', head=view(), wrist=view())
    result.update(changes)
    return result


def reply(data=None):
    return response(loc.TOOL_NAME, locations() if data is None else data)


def body(jpeg):
    return loc.build_request('press down-call', live_message(obs(0), jpeg, jpeg))


def test_request_contains_current_original_images_without_previous_claims(jpeg):
    current = live_message(obs(0), jpeg, jpeg)
    current['content'][0]['text'] = json.dumps(dict(observation_id='live-0',
        demo='SECRET_DEMO', prior='SECRET_OLD', reason='SECRET_REASON', state='SECRET_STATE'))
    original = copy.deepcopy(current)
    request = loc.build_request('press down-call', current)
    assert image_count(request['messages']) == 2 and current == original
    assert 'SECRET' not in json.dumps(request)
    assert request['temperature'] == 0 and request['max_tokens'] == 512
    assert request['chat_template_kwargs'] == dict(enable_thinking=False)
    assert 'observation_id' not in request['tools'][0]['function']['parameters']['properties']
    assert not set(loc.VIEW_PROPERTIES) & {'horizontal', 'vertical', 'progress', 'xyz_mm'}


def test_partial_unknown_and_mismatched_labels_are_advisory_only():
    result = loc.assess(reply(locations(
        head=view(button_visibility='partial', contact_visibility='unknown',
                  button_box=[.7, .7, .8, .8], contact_tip=[.1, .1]),
        wrist=view(button_visibility='not_visible', extra='retain me'))), 'live-0', SIZES, revision=3)
    assert result['format'] == 'pressb-localization-v3' and result['revision'] == 3
    assert not {'usable_views', 'recovery_views'} & set(result)
    assert result['head']['button_visibility'] == 'partial'
    assert result['head']['derived_relation']['horizontal'] == 'right'
    assert result['wrist']['derived_relation']['horizontal'] == 'left'
    assert result['wrist']['derived_relation']['pixel_delta'] == pytest.approx([-384, -288])
    assert result['wrist']['derived_relation']['error_button_units'] == pytest.approx([-3, -4])
    assert result['raw_locations']['wrist']['extra'] == 'retain me'
    assert 'returned rectangle' in result['head']['derived_relation']['center_semantics']


def test_arithmetic_has_no_near_zero_tolerance_and_rejects_wrong_range_without_clipping():
    relation = loc.derive(view(button_box=[.49, .49, .53, .53], contact_tip=[.5, .5]), [640, 360])
    assert relation['horizontal'] == 'right' and relation['vertical'] == 'below'
    assert relation['dx'] == pytest.approx(.01) and 'tolerance' not in relation
    relation = loc.derive(view(button_box=[1, 1, 3, 3], contact_tip=[-1, -2]), [640, 360])
    assert relation['returned_box_center'] is None
    assert relation['pixel_delta'] is None
    relation = loc.derive(view(button_box=[0, 0, 1, 1], contact_tip=[.5, .5]), [640, 360])
    assert relation['horizontal'] == 'center' and relation['vertical'] == 'level'
    assert relation['dx'] == relation['dy'] == 0


@pytest.mark.parametrize('box', [[.1, .1, .1, .2], [.4, .1, .3, .2]])
def test_zero_or_reversed_rectangle_is_unknown_and_preserves_raw(box):
    result = loc.assess(reply(locations(wrist=view(button_box=box))), 'live-0', SIZES)
    assert result['wrist']['button_box'] is None
    assert result['raw_locations']['wrist']['button_box'] == box
    relation = result['wrist']['derived_relation']
    assert relation['dx'] is None and relation['pixel_delta'] is None
    assert relation['error_button_units'] is None


@pytest.mark.parametrize('bad', [[float('nan'), .3], [float('inf'), .3], [True, .3], ['not-a-number', .3], [.3]])
def test_unreadable_coordinates_preserve_raw_and_do_not_raise(bad):
    result = loc.assess(reply(locations(wrist=view(contact_tip=bad))), 'live-0', SIZES)
    assert result['wrist']['contact_tip'] is None
    assert result['wrist']['derived_relation']['dx'] is None
    assert result['raw_output'] and result['raw_locations']['wrist']['contact_tip']
    assert any('wrist.contact_tip' in item for item in result['parse_notes'])
    json.dumps(result, allow_nan=False)


def test_numeric_strings_and_extra_fields_do_not_need_format_repair():
    data = locations(wrist=view(button_box='["0.3", "0.1", "0.4", "0.2"]',
                               contact_tip=['0.65', '0.55']))
    data.update(observation_id='a-model-invented-id', extra='an extra field')
    result = loc.assess(reply(data), 'live-0', SIZES)
    assert result['observation_id'] == 'live-0'
    assert result['wrist']['derived_relation']['dx'] == pytest.approx(-.3)
    assert result['raw_locations']['observation_id'] == 'a-model-invented-id'
    assert result['parse_notes'] == []


@pytest.mark.parametrize('kind', ['bad_json', 'plain_prose', 'missing_views', 'wrong_tool', 'length'])
def test_single_request_retains_model_answer_without_repair(kind, jpeg, tmp_path, monkeypatch):
    answer = reply()
    message = answer['choices'][0]['message']
    if kind == 'bad_json':
        message['tool_calls'][0]['function']['arguments'] = '{"head":'
    elif kind == 'plain_prose':
        message.update(tool_calls=[], content='The button is possibly visible but I cannot locate its front end.')
    elif kind == 'missing_views':
        message['tool_calls'][0]['function']['arguments'] = '{"anything":"kept"}'
    elif kind == 'wrong_tool':
        message['tool_calls'][0]['function']['name'] = 'unexpected_name'
        message['content'] = 'An extra explanatory sentence.'
    else:
        answer['choices'][0]['finish_reason'] = 'length'
    calls = []
    def request(*args, **kwargs):
        calls.append(args)
        assert kwargs['timeout'] == 120
        return answer
    monkeypatch.setattr(loc, 'request', request)
    result = loc.inspect('fake', body(jpeg), 'live-0', tmp_path, revision=2)
    # A readable wrist box gets one zoomed second pass; this reply is unusable
    # there, so the first-pass answer is kept unchanged.
    readable = kind in ('wrong_tool', 'length')
    assert len(calls) == (2 if readable else 1) and result['revision'] == 2
    assert result['wrist_refinement']['status'] == ('no_box_kept_first_pass' if readable else 'skipped_no_first_box')
    assert result['raw_output'] == [message]
    assert not list(tmp_path.glob('repair*'))
    assert json.loads((tmp_path/'response-0.json').read_text())['response'] == answer
    assert json.loads((tmp_path/'assessment.json').read_text()) == result
    assert result['images']['wrist']['sha256'] == hashlib.sha256(jpeg).hexdigest()
    metrics = json.loads((tmp_path/'metrics.json').read_text())
    assert metrics['attempts'] == 1 and metrics['validation_failures'] == 0
    assert metrics['status'] == 'completed'
    if kind in ('bad_json', 'plain_prose', 'missing_views'):
        assert result['wrist']['contact_tip'] is None and result['parse_notes']
    else:
        assert result['wrist']['contact_tip'] == [.65, .55]


def test_archive_normalization_does_not_restore_old_visibility_gate():
    result = loc.assess(reply(locations(wrist=view(button_visibility='partial', contact_visibility='unknown'))),
                        'live-0', SIZES)
    restored = loc.normalize(result['raw_locations'], result['observation_id'], SIZES,
        result['revision'], result['contact_reference'], result['raw_output'], result['parse_notes'])
    assert restored == result
    assert restored['wrist']['derived_relation']['dx'] is not None


def test_explicit_thousand_units_are_canonicalized_before_pixel_arithmetic():
    data = locations(coordinate_units='normalized_1000', wrist=view(
        button_visibility='partial', button_box=[180, 0, 390, 130],
        arrow_box=[670, 730, 790, 870], contact_tip=[690, 760]))
    result = loc.assess(reply(data), 'current', dict(head=[1280, 720], wrist=[640, 360]))
    wrist = result['wrist']
    assert wrist['button_box'] == [.18, 0, .39, .13]
    assert wrist['contact_tip'] == [.69, .76]
    assert wrist['derived_relation']['pixel_delta'] == pytest.approx([-259.2, -250.2])
    assert wrist['derived_relation']['error_button_units'] == pytest.approx([-405/210, -695/130])
    assert wrist['derived_relation']['horizontal'] == 'left'
    assert wrist['derived_relation']['vertical'] == 'above'
    assert result['coordinate_conversion'] == dict(source_units='normalized_1000',
        target_units='normalized_01', divisor=1000, source='model_declared', status='converted')
    assert result['raw_locations'] == data
    model = loc.model_view(result)
    assert not {'raw_locations', 'raw_output', 'images'} & set(model)
    assert model['wrist'] == wrist
    model['wrist']['contact_tip'][0] = .1
    assert result['wrist']['contact_tip'] == [.69, .76]


@pytest.mark.parametrize('units', [None, 'pixels', ['normalized_01']])
def test_missing_or_unsupported_units_are_not_guessed_from_coordinate_magnitude(units):
    data = locations()
    if units is None:
        data.pop('coordinate_units')
    else:
        data['coordinate_units'] = units
    result = loc.assess(reply(data), 'current', SIZES)
    assert result['wrist']['contact_tip'] is None
    assert result['wrist']['derived_relation']['horizontal'] == 'unknown'
    assert result['coordinate_conversion']['status'] == 'invalid_units'
    assert result['raw_locations'] == data


def test_legacy_replay_override_is_explicit_and_cannot_override_declared_unit():
    data = locations()
    data.pop('coordinate_units')
    result = loc.normalize(data, 'archived', SIZES, source_units='normalized_01')
    assert result['wrist']['contact_tip'] == [.65, .55]
    assert result['coordinate_conversion']['source'] == 'explicit_legacy_override'
    assert loc.normalize(data, 'archived', SIZES)['wrist']['contact_tip'] is None
    conflicting = loc.normalize(locations(), 'archived', SIZES, source_units='normalized_1000')
    assert conflicting['wrist']['contact_tip'] is None
    assert conflicting['coordinate_conversion']['status'] == 'invalid_units'
    with pytest.raises(ValueError, match='unsupported_coordinate_source_units'):
        loc.normalize(data, 'archived', SIZES, source_units='pixels')


def test_mixed_declared_units_and_out_of_range_values_are_unknown_not_clipped():
    data = locations(head=view(coordinate_units='normalized_1000'),
                     wrist=view(button_box=[180, 0, 390, 130], contact_tip=[.65, .55]))
    result = loc.assess(reply(data), 'current', SIZES)
    assert result['head']['contact_tip'] is None
    assert result['wrist']['button_box'] is None
    assert result['wrist']['contact_tip'] == [.65, .55]
    assert result['wrist']['derived_relation']['pixel_delta'] is None
    assert any('conflicts' in note for note in result['parse_notes'])


def test_live_request_declares_coordinate_contract_in_schema_and_prompt(jpeg):
    request = body(jpeg)
    schema = request['tools'][0]['function']['parameters']
    assert 'coordinate_units' in schema['required']
    assert set(schema['properties']['coordinate_units']['enum']) == {'normalized_01', 'normalized_1000'}
    item = schema['properties']['wrist']['properties']['contact_tip']['items']
    assert item['minimum'] == 0 and item['maximum'] == 1000
    assert 'NOT pixel coordinates' in request['messages'][0]['content']


def test_raw_comparison_keeps_changes_across_scale_pose_identity_and_gripper():
    before = loc.assess(reply(), 'live-0', SIZES)
    after = loc.assess(reply(locations(
        head=view(button_box=[.2, .1, .8, .7], button_visibility='partial'),
        wrist=view(contact_visibility='unknown', contact_tip=[.8, .8]))), 'live-1', SIZES,
        contact_reference='red_marker')
    commands = [dict(type='move', xyz_mm=[5, 20, -10]), dict(type='head', head_deg=[0, 30]),
                dict(type='gripper', key='o')]
    previous_state = dict(head_deg=[0, 0], gripper_command='c')
    current_state = dict(head_deg=[0, 30], gripper_command='o',
                         joints=dict(astribot_gripper_right=dict(position=[.4])))
    result = loc.compare(before, after, previous_state, current_state, commands)
    assert result['head_delta_deg'] == [0, 30] and result['actual_commands'] == commands
    assert result['current_state']['measured_gripper_sdk_native'] == [.4]
    assert result['views']['head']['current_minus_previous']['box_size'] == pytest.approx([.5, .5])
    assert result['views']['head']['current_minus_previous']['offset_normalized'] == pytest.approx([.15, .25])
    assert result['views']['wrist']['current_minus_previous']['offset_normalized'] == pytest.approx([-.15, -.25])
    encoded = json.dumps(result)
    assert all(key not in encoded for key in ('comparable', 'needs_recheck', 'trend', 'threshold', '20_percent', 'usable_views'))
    assert result['previous_contact_reference'] == 'white_arrow' and result['current_contact_reference'] == 'red_marker'


def test_first_comparison_has_unknown_before_but_keeps_current_coordinates():
    current = loc.assess(reply(), 'live-0', SIZES)
    result = loc.compare(None, current, None, {}, [])
    assert result['previous_observation_id'] is None
    assert result['views']['head']['current_minus_previous']['offset_normalized'] is None
    assert result['views']['head']['current']['offset_normalized'] == pytest.approx([-.3, -.4])


def test_request_binding_and_stop_still_block_late_result(jpeg, tmp_path, monkeypatch):
    request = body(jpeg)
    monkeypatch.setattr(loc, 'request', lambda *a, **k: pytest.fail('must not request'))
    with pytest.raises(ValueError, match='request_observation_mismatch'):
        loc.inspect('fake', request, 'live-1', tmp_path/'wrong')
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(InterruptedError):
        loc.inspect('fake', request, 'live-0', tmp_path/'before', cancel)
    cancel.clear()
    def late(*args, **kwargs):
        cancel.set()
        return reply()
    monkeypatch.setattr(loc, 'request', late)
    with pytest.raises(InterruptedError):
        loc.inspect('fake', request, 'live-0', tmp_path/'late', cancel)
    assert not (tmp_path/'late'/'assessment.json').exists()
    assert (tmp_path/'late'/'response-0.json').exists()
    assert json.loads((tmp_path/'late'/'metrics.json').read_text())['status'] == 'interrupted'


def test_transport_failure_ends_without_repair(jpeg, tmp_path, monkeypatch):
    calls = []
    def fail(*args, **kwargs):
        calls.append(args)
        raise OSError('service unavailable')
    monkeypatch.setattr(loc, 'request', fail)
    with pytest.raises(OSError):
        loc.inspect('fake', body(jpeg), 'live-0', tmp_path)
    assert len(calls) == 1 and not (tmp_path/'assessment.json').exists()
    assert json.loads((tmp_path/'metrics.json').read_text())['status'] == 'failed'


def refine_reply(visibility='full', box=(250, 250, 750, 750)):
    return response(loc.REFINE_TOOL_NAME, dict(button_visibility=visibility,
                                              button_box=None if box is None else list(box)))


def test_crop_window_is_square_scaled_and_kept_inside_image():
    # Box is 64x36 px; side = 3*64 = 192 px around its center.
    assert loc.crop_window([.5, .5, .6, .6], [640, 360]) == [256, 102, 448, 294]
    # Near an edge the window shifts inward instead of shrinking.
    assert loc.crop_window([.0, .9, .1, 1.], [640, 360]) == [0, 168, 192, 360]
    # A window larger than the image is clamped to the shorter image side.
    assert loc.crop_window([.1, .1, .9, .9], [640, 360]) == [140, 0, 500, 360]


def test_wrist_refinement_replaces_box_everywhere_and_keeps_first_pass(jpeg, tmp_path, monkeypatch):
    first = locations(wrist=view(button_box=[.5, .5, .6, .6]))
    replies = iter([reply(first), refine_reply()])
    bodies = []
    monkeypatch.setattr(loc, 'request', lambda *a, **k: bodies.append(a[2]) or next(replies))
    result = loc.inspect('fake', body(jpeg), 'live-0', tmp_path)
    # Crop [256,102,448,294] px on 640x360; crop box 250..750 maps to its middle half.
    expected = [(256+.25*192)/640, (102+.25*192)/360, (256+.75*192)/640, (102+.75*192)/360]
    assert result['wrist']['button_box'] == pytest.approx(expected)
    assert result['raw_locations']['wrist']['button_box'] == pytest.approx(expected)
    assert result['wrist']['derived_relation']['returned_box_center'] == pytest.approx(
        [(expected[0]+expected[2])/2, (expected[1]+expected[3])/2])
    assert result['head']['button_box'] == first['head']['button_box']
    info = result['wrist_refinement']
    assert info['status'] == 'refined' and info['first_box'] == [.5, .5, .6, .6]
    assert info['crop_window_px'] == [256, 102, 448, 294]
    refine = bodies[1]
    assert refine['tools'][0]['function']['name'] == loc.REFINE_TOOL_NAME
    assert image_count(refine['messages']) == 1 and refine['temperature'] == 0
    assert 'wrist_refinement' not in loc.model_view(result)
    assert json.loads((tmp_path/'assessment.json').read_text()) == result
    assert json.loads((tmp_path/'metrics.json').read_text())['wrist_refinement'] == 'refined'


@pytest.mark.parametrize('second', ['not_visible', 'no_box', 'error'])
def test_wrist_refinement_failure_keeps_first_pass_box(second, jpeg, tmp_path, monkeypatch):
    first = locations(wrist=view(button_box=[.5, .5, .6, .6]))
    def request(*args, **kwargs):
        if args[2]['tools'][0]['function']['name'] == loc.TOOL_NAME:
            return reply(first)
        if second == 'error':
            raise OSError('service unavailable')
        return refine_reply('not_visible') if second == 'not_visible' else refine_reply(box=None)
    monkeypatch.setattr(loc, 'request', request)
    result = loc.inspect('fake', body(jpeg), 'live-0', tmp_path)
    assert result['wrist']['button_box'] == [.5, .5, .6, .6]
    assert result['raw_locations']['wrist']['button_box'] == [.5, .5, .6, .6]
    assert result['wrist_refinement']['status'] == (
        'failed_kept_first_pass' if second == 'error' else 'no_box_kept_first_pass')


def test_stop_during_wrist_refinement_still_blocks_result(jpeg, tmp_path, monkeypatch):
    cancel = threading.Event()
    first = locations(wrist=view(button_box=[.5, .5, .6, .6]))
    def request(*args, **kwargs):
        if args[2]['tools'][0]['function']['name'] == loc.TOOL_NAME:
            return reply(first)
        cancel.set()
        return refine_reply()
    monkeypatch.setattr(loc, 'request', request)
    with pytest.raises(InterruptedError):
        loc.inspect('fake', body(jpeg), 'live-0', tmp_path, cancel)
    assert not (tmp_path/'assessment.json').exists()
